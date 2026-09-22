"""ROS-free interface for the RealHand L30 dexterous hand.

The L30 exposes its native 17-element motor-position vector.  It supports the
vendor metal CANFD analyser through ``libcanbus`` and, when available, native
SocketCAN through the same protocol implementation.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from queue import Full
from typing import Literal

from realhand.exceptions import CANError, StateError, ValidationError
from realhand.queue import IterableQueue

from . import realhand_l30_v6_2_canfd as protocol_v2
from . import realhand_l30_v6_canfd as protocol_v1
from .acceleration import AccelerationManager
from .current import CurrentManager
from .fault import FaultManager
from .force_sensor import ForceSensorManager
from .position import PositionManager, PositionPercentManager
from .speed import SpeedManager
from .temperature import TemperatureManager
from .torque import TorqueLimitManager, TorqueManager
from .version import InfoManager, L30DeviceInfo, VersionManager
from .events import (
    AccelerationData,
    AccelerationEvent,
    CurrentData,
    CurrentEvent,
    FaultData,
    FaultEvent,
    ForceSensorData,
    ForceSensorEvent,
    L30Snapshot,
    PositionData,
    PositionEvent,
    SensorEvent,
    SensorSource,
    SpeedData,
    SpeedEvent,
    TemperatureData,
    TemperatureEvent,
    TorqueData,
    TorqueEvent,
)

HandSide = Literal["left", "right"]
TransportType = Literal["libcanbus", "socketcan"]


class L30:
    """Direct Python interface for an L30 hand.

    ``libcanbus`` is the default for the blue/black metal USB CANFD analyser.
    Install the matching vendor library under ``/usr/local/lib`` first.  No
    ROS 2 node, topic, or launch file is used by this class.
    """

    JOINT_COUNT = 17
    _DEFAULT_POLL_INTERVALS: dict[SensorSource, float] = {
        SensorSource.POSITION: 1 / 20,
        SensorSource.SPEED: 1 / 10,
        SensorSource.ACCELERATION: 1.0,
        SensorSource.TORQUE: 1.0,
        SensorSource.TEMPERATURE: 1.0,
        SensorSource.CURRENT: 1.0,
        SensorSource.FAULT: 1.0,
        SensorSource.FORCE_SENSOR: 1 / 10,
    }

    def __init__(
        self,
        side: HandSide,
        canfd_id: int = 0,
        interface_type: TransportType = "libcanbus",
        interface_name: str | None = None,
        bitrate: int = 1_000_000,
        dbitrate: int = 5_000_000,
        auto_setup: bool = False,
    ) -> None:
        if side not in ("left", "right"):
            raise ValidationError(f"side must be 'left' or 'right', got {side!r}")
        if interface_type not in ("libcanbus", "socketcan"):
            raise ValidationError("interface_type must be 'libcanbus' or 'socketcan'")
        if canfd_id < 0:
            raise ValidationError("canfd_id must be non-negative")

        self.side = side
        self.canfd_id = canfd_id
        self.interface_type = interface_type
        self._closed = False
        self._motion_enabled = False
        self._lock = threading.RLock()
        self._poll_stop = threading.Event()
        self._poll_stop.set()
        self._poll_thread: threading.Thread | None = None
        self._stream: IterableQueue[SensorEvent] | None = None
        self._snapshot_lock = threading.Lock()
        self._position: PositionData | None = None
        self._speed: SpeedData | None = None
        self._acceleration: AccelerationData | None = None
        self._torque: TorqueData | None = None
        self._temperature: TemperatureData | None = None
        self._current: CurrentData | None = None
        self._fault: FaultData | None = None
        self._force_sensor: ForceSensorData | None = None
        channel: int | str = interface_name if interface_name is not None else ("can0" if interface_type == "socketcan" else 0)
        self._controller = self._connect(
            canfd_id=canfd_id,
            channel=channel,
            # The copied vendor controllers call this transport selector
            # ``comm_type``.  Keep ``interface_type`` as the public Python
            # API name, but translate it at this boundary.
            comm_type=interface_type,
            bitrate=bitrate,
            dbitrate=dbitrate,
            auto_setup=auto_setup,
        )
        self._using_v62_protocol = hasattr(self._controller.protocol, "get_device_info")
        self._torque_bounds = (60, 800) if self._using_v62_protocol else (-2047, 2047)
        self._speed_bounds = (1, 150) if self._using_v62_protocol else (0, 150)

        self.position = PositionManager(self)
        self.position_percent = PositionPercentManager(self)
        self.speed = SpeedManager(self)
        self.acceleration = AccelerationManager(self)
        self.torque = TorqueManager(self)
        self.torque_limit = TorqueLimitManager(self)
        self.temperature = TemperatureManager(self)
        self.current = CurrentManager(self)
        self.fault = FaultManager(self)
        self.force_sensor = ForceSensorManager(self)
        self.version = VersionManager(self)
        # ``info`` remains the original L30 public name.
        self.info: InfoManager = self.version
        self.start_polling()

    def _connect(self, **kwargs):
        failures: list[str] = []
        for protocol in (protocol_v1, protocol_v2):
            controller_kwargs = dict(kwargs)
            # V6 uses legacy device ID 0x06.  V6.2 instead uses a 29-bit
            # CANFD address and defaults to NodeID 1; do not overwrite that
            # V6.2 default with the legacy ID.
            if protocol is protocol_v1:
                controller_kwargs["device_id"] = 0x06
            controller = protocol.L30DexterousHandController(
                enable_on_connect=False, **controller_kwargs
            )
            accepted = False
            try:
                result = controller.connect()
                connected, detected_side = result if isinstance(result, tuple) else (bool(result), None)
                if connected and detected_side == self.side:
                    accepted = True
                    return controller
                failures.append(f"{protocol.__name__}: detected side {detected_side!r}")
            except Exception as error:
                failures.append(f"{protocol.__name__}: {error}")
            finally:
                if not accepted:
                    controller.disconnect()
        detail = "; ".join(failures) or "no protocol could connect"
        raise CANError(f"Unable to connect to L30 on CANFD device {self.canfd_id}: {detail}")

    def joint_names(self) -> list[str]:
        """Return native joint names in the same order as position vectors."""
        return list(self._controller.get_joint_name()[0])

    @property
    def protocol_version(self) -> str:
        """Protocol selected during connection: ``V6`` or ``V6.2``."""
        return "V6.2" if self._using_v62_protocol else "V6"

    def joint_ranges(self) -> dict[str, tuple[int, int]]:
        """Return the safe native range for each named L30 joint."""
        return self._controller.get_joint_range()

    def _position_ranges(self) -> list[tuple[int, int]]:
        """Return the connected controller's 17 limits in command order."""
        ranges = self._controller.get_joint_range()
        if all(index in ranges for index in range(1, self.JOINT_COUNT + 1)):
            return [ranges[index] for index in range(1, self.JOINT_COUNT + 1)]
        ordered = list(ranges.values())
        if len(ordered) != self.JOINT_COUNT:
            raise StateError("L30 controller returned an invalid joint-limit table")
        return ordered

    def get_snapshot(self) -> L30Snapshot:
        """Return the latest cached readings without sending a CAN request."""
        with self._snapshot_lock:
            return L30Snapshot(
                position=self._position,
                speed=self._speed,
                acceleration=self._acceleration,
                torque=self._torque,
                temperature=self._temperature,
                current=self._current,
                fault=self._fault,
                force_sensor=self._force_sensor,
                timestamp=time.time(),
            )

    def _sensor_snapshot(self, source: SensorSource):
        """Return one cached source reading for manager-level snapshots."""
        with self._snapshot_lock:
            return deepcopy(getattr(self, f"_{source.value}"))

    def stream(self, maxsize: int = 100) -> IterableQueue[SensorEvent]:
        """Return a queue that receives readings produced by polling.

        Calling this again closes the previous stream. Start or reconfigure
        polling with :meth:`start_polling` to choose the source intervals.
        """
        self._ensure_open()
        if maxsize <= 0:
            raise ValidationError("maxsize must be positive")
        self.stop_stream()
        self._stream = IterableQueue(maxsize=maxsize)
        return self._stream

    def stop_stream(self) -> None:
        """Close the current event stream, if any."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def start_polling(self, intervals: dict[SensorSource, float] | None = None) -> None:
        """Start one serialized polling worker for cached readings and events.

        The L30 protocol uses synchronous request/response exchanges; a single
        worker deliberately polls all sources in sequence rather than issuing
        concurrent CAN transactions.
        """
        self._ensure_open()
        selected = dict(self._DEFAULT_POLL_INTERVALS if intervals is None else intervals)
        if not self.acceleration.supported:
            selected.pop(SensorSource.ACCELERATION, None)
        if not selected:
            raise ValidationError("intervals must contain at least one sensor source")
        for source, interval in selected.items():
            if not isinstance(source, SensorSource):
                raise ValidationError("polling keys must be L30 SensorSource values")
            if interval <= 0:
                raise ValidationError(f"interval for {source.value} must be positive")
        self.stop_polling()
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(selected,),
            daemon=True,
            name="L30-polling",
        )
        self._poll_thread.start()

    def stop_polling(self) -> None:
        """Stop the serialized polling worker. Safe to call repeatedly."""
        self._poll_stop.set()
        thread = self._poll_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self, intervals: dict[SensorSource, float]) -> None:
        due = {source: 0.0 for source in intervals}
        while not self._poll_stop.is_set() and not self._closed:
            now = time.monotonic()
            for source, interval in intervals.items():
                if now < due[source]:
                    continue
                try:
                    self._read_sensor(source)
                except (CANError, StateError):
                    # A later poll can recover from an intermittent timeout.
                    pass
                due[source] = time.monotonic() + interval
            self._poll_stop.wait(0.002)

    def _read_sensor(self, source: SensorSource):
        self._ensure_open()
        with self._lock:
            timestamp = time.time()
            if source is SensorSource.POSITION:
                values = self._controller.get_positions()
                if values is None:
                    raise CANError("L30 did not return joint positions")
                data = PositionData(tuple(values), timestamp)
            elif source is SensorSource.SPEED:
                values = self._controller.get_velocities()
                if values is None:
                    raise CANError("L30 did not return joint velocities")
                data = SpeedData(tuple(values), timestamp)
            elif source is SensorSource.ACCELERATION:
                if not self.acceleration.supported:
                    raise StateError("L30 acceleration reads are unavailable for this protocol")
                values = self._controller.get_accelerations()
                if values is None:
                    raise CANError("L30 did not return joint accelerations")
                data = AccelerationData(tuple(values), timestamp)
            elif source is SensorSource.TORQUE:
                values = self._controller.get_torques()
                if values is None:
                    raise CANError("L30 did not return joint torques")
                data = TorqueData(tuple(values), timestamp)
            elif source is SensorSource.TEMPERATURE:
                values = self._controller.protocol.get_joint_temperatures()
                if values is None:
                    raise CANError("L30 did not return joint temperatures")
                data = TemperatureData(tuple(values), timestamp)
            elif source is SensorSource.CURRENT:
                values = self._controller.protocol.get_joint_currents()
                if values is None:
                    raise CANError("L30 did not return joint currents")
                data = CurrentData(tuple(values), timestamp)
            elif source is SensorSource.FAULT:
                values = self._controller.protocol.get_joint_error_codes()
                if values is None:
                    raise CANError("L30 did not return joint error codes")
                data = FaultData(tuple(values), timestamp)
            else:
                values = self._controller.get_matrix_touch()
                if values is None:
                    raise CANError("L30 did not return fingertip pressure data")
                data = ForceSensorData(deepcopy(values), timestamp)
        self._cache_and_emit(source, data)
        return data

    def _cache_and_emit(self, source: SensorSource, data) -> None:
        event_type = {
            SensorSource.POSITION: PositionEvent,
            SensorSource.SPEED: SpeedEvent,
            SensorSource.ACCELERATION: AccelerationEvent,
            SensorSource.TORQUE: TorqueEvent,
            SensorSource.TEMPERATURE: TemperatureEvent,
            SensorSource.CURRENT: CurrentEvent,
            SensorSource.FAULT: FaultEvent,
            SensorSource.FORCE_SENSOR: ForceSensorEvent,
        }[source]
        with self._snapshot_lock:
            setattr(self, f"_{source.value}", data)
        if self._stream is not None:
            try:
                self._stream.put_nowait(event_type(data=data))
            except Full:
                pass
            except StateError:
                pass

    def _read_device_info(self) -> L30DeviceInfo:
        """Read and normalize information returned by V6 and V6.2 firmware."""
        self._ensure_open()
        with self._lock:
            raw = getattr(self._controller, "get_device_info", lambda: None)()
            protocol = self._controller.protocol
            if raw is not None:
                return L30DeviceInfo(
                    serial_number=raw.get("serial_no"),
                    product_code=getattr(protocol, "get_product_code", lambda: None)(),
                    node_id=raw.get("node_id"),
                    hand_type=raw.get("hand_type"),
                    software_version=raw.get("sw_version"),
                    hardware_version=raw.get("hw_version"),
                    mechanical_version=None,
                    structure_version=raw.get("struct_version"),
                    sensor_type=raw.get("sensor_type"),
                    origin=raw.get("origin"),
                )
            version = getattr(protocol, "get_device_version", lambda: None)()
            if version is None:
                raise CANError("L30 did not return device information")
            return L30DeviceInfo(
                serial_number=None,
                product_code=None,
                node_id=None,
                hand_type=self.side,
                software_version=version.get("software"),
                hardware_version=version.get("hardware"),
                mechanical_version=version.get("mechanical"),
                structure_version=None,
                sensor_type=None,
                origin=None,
            )

    def emergency_stop(self) -> None:
        self._ensure_open()
        with self._lock:
            stopped = self._controller.stop()
        if not stopped:
            raise CANError("L30 emergency-stop command failed")

    def enable_all(self) -> None:
        self._ensure_open()
        with self._lock:
            enabled = self._controller.enable_all()
        if not enabled:
            raise CANError("L30 enable-all command failed")
        self._motion_enabled = True

    def disable_all(self) -> None:
        """Explicitly disable every joint and block further motion commands."""
        self._ensure_open()
        with self._lock:
            if hasattr(self._controller, "disable_all"):
                disabled = self._controller.disable_all()
            else:
                disabled = self._controller.set_enable([0] * self.JOINT_COUNT)
        if not disabled:
            raise CANError("L30 disable-all command failed")
        self._motion_enabled = False

    def close(self) -> None:
        if self._closed:
            return
        self.stop_polling()
        self.stop_stream()
        with self._lock:
            self._controller.disconnect()
            self._closed = True
            self._motion_enabled = False

    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateError("L30 is closed")

    def __enter__(self) -> "L30":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

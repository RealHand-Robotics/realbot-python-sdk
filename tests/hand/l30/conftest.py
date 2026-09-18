"""Hardware-free fixtures for L30 tests.

The normal L30 test suite never opens libcanbus, enables motors, or sends a
CAN frame.  Optional live tests are kept in separate files and are skipped
unless explicitly requested.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from realhand.hand.l30 import L30
from realhand.hand.l30.l30 import (
    AccelerationManager,
    CurrentManager,
    FaultManager,
    ForceSensorManager,
    InfoManager,
    PositionManager,
    SpeedManager,
    TemperatureManager,
    TorqueLimitManager,
    TorqueManager,
)


class FakeProtocol:
    """In-memory V6 protocol response provider."""

    def get_joint_temperatures(self):
        return list(range(30, 47))

    def get_joint_currents(self):
        return list(range(17))

    def get_joint_error_codes(self):
        return [0] * 17

    def get_device_version(self):
        return {"hardware": "2.0.0", "software": "6.1.5", "mechanical": "2.2.0"}


class FakeController:
    """Records commands instead of talking to a physical L30."""

    def __init__(self, acceleration_supported: bool = True, torque_limit_supported: bool = True):
        self.protocol = FakeProtocol()
        self.calls: list[tuple[str, list[int] | None]] = []
        self.acceleration_supported = acceleration_supported
        self.torque_limit_supported = torque_limit_supported

    def get_joint_name(self):
        return ([f"joint_{index}" for index in range(1, 18)], [f"J{index}" for index in range(1, 18)])

    def get_joint_range(self):
        return {index: (-200, 200) if index in (5, 10, 13, 14) else (0, 1600) for index in range(1, 18)}

    def get_positions(self): return list(range(17))
    def get_velocities(self): return [75] * 17
    def get_torques(self): return [100] * 17
    def get_matrix_touch(self): return {finger: [0] * 72 for finger in ("thumb", "index", "middle", "ring", "pinky")}
    def set_positions(self, values): self.calls.append(("positions", values)); return True
    def set_velocities(self, values): self.calls.append(("speed", values)); return True
    def set_torques(self, values): self.calls.append(("current", values)); return True
    def enable_all(self): self.calls.append(("enable", None)); return True
    def disable_all(self): self.calls.append(("disable", None)); return True
    def stop(self): self.calls.append(("stop", None)); return True
    def disconnect(self): self.calls.append(("disconnect", None))

    def set_accelerations(self, values):
        if not self.acceleration_supported:
            raise AttributeError
        self.calls.append(("acceleration", values)); return True

    def get_accelerations(self): return [20] * 17

    def set_torque_limits(self, values):
        if not self.torque_limit_supported:
            raise AttributeError
        self.calls.append(("torque_limit", values)); return True


def make_l30(*, acceleration_supported: bool = True, torque_limit_supported: bool = True):
    """Build an L30 instance without opening hardware or starting polling."""
    hand = object.__new__(L30)
    hand.side = "left"
    hand.canfd_id = 0
    hand.interface_type = "libcanbus"
    hand._closed = False
    hand._motion_enabled = False
    hand._lock = threading.RLock()
    hand._poll_stop = threading.Event(); hand._poll_stop.set()
    hand._poll_thread = None
    hand._stream = None
    hand._snapshot_lock = threading.Lock()
    hand._position = hand._speed = hand._acceleration = hand._torque = None
    hand._temperature = hand._current = hand._fault = hand._force_sensor = None
    hand._controller = FakeController(acceleration_supported, torque_limit_supported)
    hand._using_v62_protocol = False
    hand._speed_bounds = (0, 150)
    hand._torque_bounds = (-2047, 2047)
    hand.position = PositionManager(hand)
    hand.speed = SpeedManager(hand)
    hand.acceleration = AccelerationManager(hand)
    hand.torque = TorqueManager(hand)
    hand.torque_limit = TorqueLimitManager(hand)
    hand.temperature = TemperatureManager(hand)
    hand.current = CurrentManager(hand)
    hand.fault = FaultManager(hand)
    hand.force_sensor = ForceSensorManager(hand)
    hand.info = InfoManager(hand)
    return hand


@pytest.fixture
def l30():
    hand = make_l30()
    yield hand
    hand.close()

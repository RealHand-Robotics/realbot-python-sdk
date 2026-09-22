"""Commanded-current and torque-limit control for the L30 hand."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from realhand.exceptions import CANError, StateError, ValidationError

from ._validation import JointValues, validate_values
from .events import SensorSource
from .joints import L30Torque

if TYPE_CHECKING:
    from .l30 import L30


class TorqueManager:
    """Control commanded motor current and read native torque feedback."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def set_commanded(self, currents: Sequence[int]) -> None:
        self._hand._ensure_open()
        values = validate_values("commanded current", currents, *self._hand._torque_bounds)
        with self._hand._lock:
            accepted = self._hand._controller.set_torques(values)
        if not accepted:
            raise CANError("L30 rejected the commanded-current command")

    def set_limits(self, values: Sequence[int]) -> None:
        """Compatibility alias for :meth:`set_commanded`."""
        self.set_commanded(values)

    def get(self) -> JointValues:
        return list(self._hand._read_sensor(SensorSource.TORQUE).torques)

    def set_torques(self, currents: Sequence[int]) -> None:
        """Compatibility alias for :meth:`set_commanded`.

        For L30 this sends commanded motor current, not physical torque.
        """
        self.set_commanded(currents)

    def get_blocking(self, timeout_ms: float = 1000):
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.TORQUE)

    def get_snapshot(self):
        return self._hand._sensor_snapshot(SensorSource.TORQUE)

    def get_named(self) -> L30Torque:
        return L30Torque.from_list(self.get())


class TorqueLimitManager:
    """Set the separate V6 torque-limit threshold (0..1000, 0.1% units)."""

    MINIMUM = 0
    MAXIMUM = 1000

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    @property
    def supported(self) -> bool:
        return hasattr(self._hand._controller, "set_torque_limits")

    def set(self, limits: Sequence[int]) -> None:
        self._hand._ensure_open()
        if not self.supported:
            raise StateError("The connected L30 protocol does not support torque-limit commands")
        values = validate_values("torque limits", limits, self.MINIMUM, self.MAXIMUM)
        with self._hand._lock:
            accepted = self._hand._controller.set_torque_limits(values)
        if not accepted:
            raise CANError("L30 rejected the torque-limit command")

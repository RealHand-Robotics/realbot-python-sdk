"""Native speed control for the L30 hand."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from realhand.exceptions import CANError, ValidationError

from ._validation import JointValues, validate_values
from .events import SensorSource
from .joints import L30Speed

if TYPE_CHECKING:
    from .l30 import L30


class SpeedManager:
    """Control and read the 17 native L30 velocity values."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def set(self, speeds: Sequence[int]) -> None:
        self._hand._ensure_open()
        values = validate_values("speeds", speeds, *self._hand._speed_bounds)
        with self._hand._lock:
            accepted = self._hand._controller.set_velocities(values)
        if not accepted:
            raise CANError("L30 rejected the speed command")

    def get(self) -> JointValues:
        return list(self._hand._read_sensor(SensorSource.SPEED).speeds)

    def set_speeds(self, speeds: Sequence[int]) -> None:
        """Compatibility alias for :meth:`set`."""
        self.set(speeds)

    def get_blocking(self, timeout_ms: float = 1000):
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.SPEED)

    def get_snapshot(self):
        return self._hand._sensor_snapshot(SensorSource.SPEED)

    def get_named(self) -> L30Speed:
        return L30Speed.from_list(self.get())

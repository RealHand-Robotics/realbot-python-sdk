"""Measured-current readback for the L30 hand."""

from typing import TYPE_CHECKING

from realhand.exceptions import ValidationError

from ._validation import JointValues
from .events import SensorSource

if TYPE_CHECKING:
    from .l30 import L30


class CurrentManager:
    """Read the 17 L30 motor-current values."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def get(self) -> JointValues:
        return list(self._hand._read_sensor(SensorSource.CURRENT).currents)

    def get_blocking(self, timeout_ms: float = 1000):
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.CURRENT)

    def get_snapshot(self):
        return self._hand._sensor_snapshot(SensorSource.CURRENT)

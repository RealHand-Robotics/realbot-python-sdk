"""Fault readback for the L30 hand."""

from typing import TYPE_CHECKING

from realhand.exceptions import ValidationError

from .events import FaultData, SensorSource

if TYPE_CHECKING:
    from .l30 import L30


class FaultManager:
    """Read the native error code for each of the 17 L30 joints."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def get(self) -> FaultData:
        return self._hand._read_sensor(SensorSource.FAULT)

    def get_blocking(self, timeout_ms: float = 1000) -> FaultData:
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.FAULT)

    def get_snapshot(self) -> FaultData | None:
        return self._hand._sensor_snapshot(SensorSource.FAULT)

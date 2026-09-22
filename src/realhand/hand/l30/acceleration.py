"""Acceleration control for L30 firmware that supports it."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from realhand.exceptions import CANError, StateError, ValidationError

from ._validation import JointValues, validate_values
from .events import SensorSource
from .joints import L30Acceleration

if TYPE_CHECKING:
    from .l30 import L30


class AccelerationManager:
    """Control and read V6 acceleration values (0..254)."""

    MINIMUM = 0
    MAXIMUM = 254

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    @property
    def supported(self) -> bool:
        return hasattr(self._hand._controller, "set_accelerations")

    def set(self, accelerations: Sequence[int]) -> None:
        self._hand._ensure_open()
        if not self.supported:
            raise StateError("The connected L30 protocol does not support acceleration commands")
        values = validate_values("accelerations", accelerations, self.MINIMUM, self.MAXIMUM)
        with self._hand._lock:
            accepted = self._hand._controller.set_accelerations(values)
        if not accepted:
            raise CANError("L30 rejected the acceleration command")

    def get(self) -> JointValues:
        if not self.supported:
            raise StateError("The connected L30 protocol does not support acceleration reads")
        return list(self._hand._read_sensor(SensorSource.ACCELERATION).accelerations)

    def set_accelerations(self, accelerations: Sequence[int]) -> None:
        """Compatibility alias for :meth:`set`."""
        self.set(accelerations)

    def get_blocking(self, timeout_ms: float = 1000):
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.ACCELERATION)

    def get_snapshot(self):
        return self._hand._sensor_snapshot(SensorSource.ACCELERATION)

    def get_named(self) -> L30Acceleration:
        return L30Acceleration.from_list(self.get())

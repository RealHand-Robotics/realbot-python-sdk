"""Fingertip pressure-matrix readback for the L30 hand."""

from copy import deepcopy
from typing import TYPE_CHECKING

from realhand.exceptions import ValidationError

from .events import SensorSource

if TYPE_CHECKING:
    from .l30 import L30


class ForceSensorManager:
    """Read the five fingertip 12×6 pressure matrices."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def get(self) -> dict[str, list[int]]:
        return deepcopy(self._hand._read_sensor(SensorSource.FORCE_SENSOR).matrices)

    def get_blocking(self, timeout_ms: float = 1000):
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")
        return self._hand._read_sensor(SensorSource.FORCE_SENSOR)

    def get_snapshot(self):
        return self._hand._sensor_snapshot(SensorSource.FORCE_SENSOR)

    def get_finger(self, name: str) -> list[list[int]]:
        """Read one fingertip's 12×6 pressure matrix.

        Accepted names are ``thumb``, ``index``, ``middle``, ``ring``, and
        ``little`` (``pinky`` is accepted as an alias for ``little``).
        """
        requested_key = name.lower().strip()
        key = "little" if requested_key == "pinky" else requested_key
        matrices = self.get()
        for matrix_key in (f"{key}_matrix", key, f"{requested_key}_matrix", requested_key):
            if matrix_key in matrices:
                return deepcopy(matrices[matrix_key])
        raise ValidationError(f"unknown L30 finger {name!r}")

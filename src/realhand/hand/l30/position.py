"""Native 17-element position control for the L30 hand."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from realhand.exceptions import CANError, StateError, ValidationError

from ._validation import JointValues, validate_values
from .events import SensorSource
from .joints import L30Position

if TYPE_CHECKING:
    from .l30 import L30


class PositionManager:
    """Control and read L30 native 17-element joint-position values."""

    # V6 feedback can be a few signed encoder counts outside a zero command
    # minimum. Only this narrow boundary tolerance is converted to the exact
    # vendor command limit; meaningful out-of-range commands are rejected.
    FEEDBACK_LIMIT_TOLERANCE = 16

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def set(self, positions: Sequence[int]) -> None:
        self._hand._ensure_open()
        values = validate_values("positions", positions, -32768, 32767)
        if not self._hand._motion_enabled:
            raise StateError("L30 motion is disabled; call enable_all() only after the work area is safe")
        for index, (value, bounds) in enumerate(zip(values, self._hand._position_ranges())):
            low, high = bounds
            if low <= value <= high:
                continue
            if low - self.FEEDBACK_LIMIT_TOLERANCE <= value < low:
                values[index] = low
                continue
            if high < value <= high + self.FEEDBACK_LIMIT_TOLERANCE:
                values[index] = high
                continue
            raise ValidationError(
                f"positions[{index}]={value} is outside its native safe range [{low}, {high}]"
            )
        with self._hand._lock:
            accepted = self._hand._controller.set_positions(values)
        if not accepted:
            raise CANError("L30 rejected the position command")

    def set_positions(self, positions: Sequence[int]) -> None:
        """Compatibility alias for :meth:`set`."""
        self.set(positions)

    def get(self) -> JointValues:
        return list(self._hand._read_sensor(SensorSource.POSITION).positions)

    def get_blocking(self, timeout_ms: float = 1000):
        """Synchronously read native positions.

        The controller owns the request timeout; ``timeout_ms`` is accepted for
        API compatibility and must be positive.
        """
        self._validate_timeout(timeout_ms)
        return self._hand._read_sensor(SensorSource.POSITION)

    def get_snapshot(self):
        """Return the latest cached position reading, if any."""
        return self._hand._sensor_snapshot(SensorSource.POSITION)

    def get_named(self) -> L30Position:
        """Synchronously read positions as a named native-value container."""
        return L30Position.from_list(self.get())

    @staticmethod
    def _validate_timeout(timeout_ms: float) -> None:
        if timeout_ms <= 0:
            raise ValidationError("timeout_ms must be positive")


class PositionPercentManager:
    """Optional 0.00..100.00% facade over native L30 position commands."""

    DECIMAL_PLACES = 2

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def set(self, percentages: Sequence[float]) -> None:
        if len(percentages) != self._hand.JOINT_COUNT:
            raise ValidationError(
                f"position percentages must contain exactly {self._hand.JOINT_COUNT} values"
            )
        native: list[int] = []
        for index, (percentage, (low, high)) in enumerate(
            zip(percentages, self._hand._position_ranges())
        ):
            if isinstance(percentage, bool) or not isinstance(percentage, (int, float)):
                raise ValidationError(f"position percentages[{index}] must be numeric")
            if not 0.0 <= float(percentage) <= 100.0:
                raise ValidationError(f"position percentages[{index}]={percentage} is outside [0, 100]")
            native.append(round(low + float(percentage) * (high - low) / 100.0))
        self._hand.position.set(native)

    def get(self) -> list[float]:
        percentages: list[float] = []
        for value, (low, high) in zip(self._hand.position.get(), self._hand._position_ranges()):
            bounded = min(high, max(low, value))
            percentages.append(round(100.0 * (bounded - low) / (high - low), self.DECIMAL_PLACES))
        return percentages

    def get_snapshot(self) -> list[float] | None:
        snapshot = self._hand.position.get_snapshot()
        if snapshot is None:
            return None
        percentages: list[float] = []
        for value, (low, high) in zip(snapshot.positions, self._hand._position_ranges()):
            bounded = min(high, max(low, value))
            percentages.append(round(100.0 * (bounded - low) / (high - low), self.DECIMAL_PLACES))
        return percentages

"""Shared validation for L30 native 17-element values."""

from collections.abc import Sequence

from realhand.exceptions import ValidationError

JOINT_COUNT = 17
JointValues = list[int]


def validate_values(name: str, values: Sequence[int], minimum: int, maximum: int) -> JointValues:
    """Validate and copy one native L30 17-element integer vector."""
    if len(values) != JOINT_COUNT:
        raise ValidationError(f"{name} must contain exactly {JOINT_COUNT} values, got {len(values)}")
    result: JointValues = []
    for index, value in enumerate(values):
        if not isinstance(value, int):
            raise ValidationError(f"{name}[{index}] must be an int, got {type(value).__name__}")
        if not minimum <= value <= maximum:
            raise ValidationError(f"{name}[{index}]={value} is outside [{minimum}, {maximum}]")
        result.append(value)
    return result

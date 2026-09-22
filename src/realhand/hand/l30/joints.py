"""Named native 17-joint value containers for the L30 hand."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from realhand.exceptions import ValidationError

JOINT_COUNT = 17
JOINT_NAMES = (
    "thumb_base_flexion", "thumb_tip_flexion", "thumb_side", "thumb_rotation",
    "ring_side", "ring_tip", "ring_base", "middle_base", "middle_tip",
    "little_base", "little_tip", "little_side", "middle_side", "index_side",
    "index_base", "index_tip", "wrist_pitch",
)


@dataclass(frozen=True)
class L30JointValues(Sequence[int]):
    """Immutable native L30 values with named joint properties.

    Subclasses identify the command or reading type while retaining the exact
    native 17-element integer order required by the L30 firmware.
    """

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != JOINT_COUNT:
            raise ValidationError(f"L30 values must contain exactly {JOINT_COUNT} values")
        if any(not isinstance(value, int) for value in self.values):
            raise ValidationError("L30 values must all be integers")

    @classmethod
    def from_list(cls, values: Sequence[int]) -> "L30JointValues":
        return cls(tuple(values))

    def to_list(self) -> list[int]:
        return list(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]

    def __len__(self) -> int:
        return JOINT_COUNT

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)


def _joint_property(index: int):
    return property(lambda self: self.values[index])


for _index, _name in enumerate(JOINT_NAMES):
    setattr(L30JointValues, _name, _joint_property(_index))


class L30Position(L30JointValues):
    """Native L30 target or feedback positions."""


class L30Speed(L30JointValues):
    """Native L30 speed values."""


class L30Torque(L30JointValues):
    """Native L30 commanded-current or torque-feedback values."""


class L30Acceleration(L30JointValues):
    """Native L30 acceleration values."""

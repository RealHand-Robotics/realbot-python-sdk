"""Device-information access for the L30 hand."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .l30 import L30


@dataclass(frozen=True)
class L30DeviceInfo:
    """Normalized metadata from either supported L30 firmware protocol."""

    serial_number: int | None
    product_code: str | None
    node_id: int | None
    hand_type: str | None
    software_version: str | None
    hardware_version: str | None
    mechanical_version: str | None
    structure_version: str | None
    sensor_type: int | None
    origin: int | None


class InfoManager:
    """Read serial number, versions, and other L30 device metadata."""

    def __init__(self, hand: "L30") -> None:
        self._hand = hand

    def get(self) -> L30DeviceInfo:
        return self._hand._read_device_info()


class VersionManager(InfoManager):
    """Compatibility device-information manager matching other hand packages."""

    def get_device_info(self) -> L30DeviceInfo:
        return self.get()

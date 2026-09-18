"""Snapshots and stream event types for the ROS-free L30 API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class PositionData:
    positions: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class SpeedData:
    speeds: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class AccelerationData:
    accelerations: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class TorqueData:
    torques: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class TemperatureData:
    temperatures: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class CurrentData:
    currents: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class FaultData:
    """The 17 native L30 joint error-code values."""

    error_codes: tuple[int, ...]
    timestamp: float


@dataclass(frozen=True)
class ForceSensorData:
    """Five fingertip pressure matrices in the device's native layout."""

    matrices: dict[str, Any]
    timestamp: float


@dataclass(frozen=True)
class PositionEvent:
    data: PositionData


@dataclass(frozen=True)
class SpeedEvent:
    data: SpeedData


@dataclass(frozen=True)
class AccelerationEvent:
    data: AccelerationData


@dataclass(frozen=True)
class TorqueEvent:
    data: TorqueData


@dataclass(frozen=True)
class TemperatureEvent:
    data: TemperatureData


@dataclass(frozen=True)
class CurrentEvent:
    data: CurrentData


@dataclass(frozen=True)
class FaultEvent:
    data: FaultData


@dataclass(frozen=True)
class ForceSensorEvent:
    data: ForceSensorData


SensorEvent = (
    PositionEvent
    | SpeedEvent
    | AccelerationEvent
    | TorqueEvent
    | TemperatureEvent
    | CurrentEvent
    | FaultEvent
    | ForceSensorEvent
)


class SensorSource(str, Enum):
    POSITION = "position"
    SPEED = "speed"
    ACCELERATION = "acceleration"
    TORQUE = "torque"
    TEMPERATURE = "temperature"
    CURRENT = "current"
    FAULT = "fault"
    FORCE_SENSOR = "force_sensor"


@dataclass(frozen=True)
class L30Snapshot:
    position: PositionData | None
    speed: SpeedData | None
    acceleration: AccelerationData | None
    torque: TorqueData | None
    temperature: TemperatureData | None
    current: CurrentData | None
    fault: FaultData | None
    force_sensor: ForceSensorData | None
    timestamp: float

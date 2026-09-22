"""Tests for L30 conveniences shared with the other hand packages."""

from realhand.hand.l30 import L30Position, L30Speed, L30Torque
from realhand.hand.l30.events import PositionData


def test_native_named_value_containers_preserve_order():
    values = list(range(17))
    position = L30Position.from_list(values)
    assert position.to_list() == values
    assert position.thumb_base_flexion == 0
    assert position.middle_tip == 8
    assert position.wrist_pitch == 16
    assert L30Speed.from_list(values).to_list() == values
    assert L30Torque.from_list(values).to_list() == values


def test_standard_setter_aliases_keep_native_l30_commands(l30):
    l30.enable_all()
    l30.position.set_positions([0] * 17)
    l30.speed.set_speeds([75] * 17)
    l30.torque.set_torques([100] * 17)
    l30.acceleration.set_accelerations([20] * 17)
    assert l30._controller.calls[-4:] == [
        ("positions", [0] * 17),
        ("speed", [75] * 17),
        ("current", [100] * 17),
        ("acceleration", [20] * 17),
    ]


def test_position_percent_converts_using_each_native_range(l30):
    l30.enable_all()
    l30.position_percent.set([50.0] * 17)
    values = l30._controller.calls[-1][1]
    assert values[0] == 800
    assert values[4] == 0
    assert values[8] == 800


def test_manager_snapshots_and_blocking_reads(l30):
    cached = PositionData(tuple([10] * 17), 1.0)
    l30._position = cached
    snapshot = l30.position.get_snapshot()
    assert snapshot == cached
    assert snapshot is not cached
    assert l30.position.get_blocking().positions == tuple(range(17))
    assert l30.speed.get_blocking().speeds == tuple([75] * 17)
    assert l30.temperature.get_blocking().temperatures == tuple(range(30, 47))


def test_version_alias_and_per_finger_touch_access(l30):
    assert l30.version.get_device_info().software_version == "6.1.5"
    assert l30.info.get().software_version == "6.1.5"
    assert l30.force_sensor.get_finger("pinky") == [0] * 72

"""Unit tests for L30 read-only managers and V6 metadata normalization."""

import pytest

pytestmark = [pytest.mark.l30, pytest.mark.sensor]


def test_temperature_current_fault_and_tactile_reads(l30):
    assert l30.temperature.get() == list(range(30, 47))
    assert l30.current.get() == list(range(17))
    assert l30.fault.get().error_codes == (0,) * 17
    matrices = l30.force_sensor.get()
    assert set(matrices) == {"thumb", "index", "middle", "ring", "pinky"}
    assert all(len(matrix) == 72 for matrix in matrices.values())


def test_force_sensor_returns_copy(l30):
    matrices = l30.force_sensor.get()
    matrices["thumb"][0] = 999
    assert l30.force_sensor.get()["thumb"][0] == 0


def test_v6_info_has_versions_but_no_serial_number(l30):
    info = l30.info.get()
    assert info.serial_number is None
    assert info.software_version == "6.1.5"
    assert info.hardware_version == "2.0.0"
    assert info.mechanical_version == "2.2.0"
    assert info.hand_type == "left"

"""Unit tests for native-position control and its safety gate."""

import pytest

from realhand.exceptions import StateError, ValidationError

pytestmark = [pytest.mark.l30, pytest.mark.control, pytest.mark.safety]


def test_position_requires_explicit_enable(l30):
    with pytest.raises(StateError, match="motion is disabled"):
        l30.position.set([0] * 17)
    assert l30._controller.calls == []


def test_enable_then_set_position_uses_native_17_vector(l30):
    l30.enable_all()
    positions = [0] * 17
    positions[4] = -200
    l30.position.set(positions)
    assert l30._controller.calls[-1] == ("positions", positions)


def test_position_rejects_out_of_range_native_joint(l30):
    l30.enable_all()
    positions = [0] * 17
    positions[0] = 1617
    with pytest.raises(ValidationError, match="native safe range"):
        l30.position.set(positions)


def test_position_clamps_small_signed_feedback_offset_at_minimum(l30):
    """V6 can report a few signed counts below a zero command minimum."""
    l30.enable_all()
    positions = [0] * 17
    positions[8] = -6
    l30.position.set(positions)
    assert l30._controller.calls[-1] == ("positions", [0] * 17)


def test_position_get_reads_cached_protocol_value(l30):
    assert l30.position.get() == list(range(17))

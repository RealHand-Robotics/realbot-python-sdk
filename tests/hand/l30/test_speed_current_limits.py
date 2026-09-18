"""Unit tests for L30 speed, commanded-current, and torque-limit controls."""

import pytest

from realhand.exceptions import ValidationError

pytestmark = [pytest.mark.l30, pytest.mark.control]


def test_speed_set_and_get(l30):
    l30.speed.set([128] * 17)
    assert l30._controller.calls[-1] == ("speed", [128] * 17)
    assert l30.speed.get() == [75] * 17


def test_speed_rejects_invalid_value(l30):
    with pytest.raises(ValidationError):
        l30.speed.set([151] * 17)


def test_commanded_current_set_get_and_compatibility_alias(l30):
    l30.torque.set_commanded([1740] * 17)
    assert l30._controller.calls[-1] == ("current", [1740] * 17)
    l30.torque.set_limits([100] * 17)
    assert l30._controller.calls[-1] == ("current", [100] * 17)
    assert l30.torque.get() == [100] * 17


def test_torque_limit_is_a_distinct_command(l30):
    l30.torque_limit.set([850] * 17)
    assert l30._controller.calls[-1] == ("torque_limit", [850] * 17)


def test_torque_limit_rejects_protocol_values_above_100_percent(l30):
    with pytest.raises(ValidationError):
        l30.torque_limit.set([1001] * 17)

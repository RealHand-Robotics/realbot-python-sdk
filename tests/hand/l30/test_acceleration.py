"""Unit tests for the V6 L30 acceleration command and readback."""

import pytest

from realhand.exceptions import ValidationError

pytestmark = [pytest.mark.l30, pytest.mark.control]


def test_acceleration_set_and_get(l30):
    l30.acceleration.set([20] * 17)
    assert l30._controller.calls[-1] == ("acceleration", [20] * 17)
    assert l30.acceleration.get() == [20] * 17


@pytest.mark.parametrize("value", [-1, 255])
def test_acceleration_rejects_out_of_range_values(l30, value):
    with pytest.raises(ValidationError):
        l30.acceleration.set([value] * 17)

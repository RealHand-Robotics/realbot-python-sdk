"""Optional live L30 read-only smoke test.

Run explicitly with:
  REALHAND_L30_LIVE_TEST=1 CANFD_ID=0 L30_SIDE=left \
  PYTHONPATH=src pytest -m l30 tests/hand/l30/test_live_readonly.py
"""

import os

import pytest

from realhand import L30

pytestmark = [pytest.mark.l30, pytest.mark.sensor, pytest.mark.interactive]


@pytest.mark.skipif(os.environ.get("REALHAND_L30_LIVE_TEST") != "1", reason="set REALHAND_L30_LIVE_TEST=1 to use hardware")
def test_live_readonly_status_and_tactile_data():
    """Connect without enabling, then verify all supported reads return 17 values."""
    side = os.environ.get("L30_SIDE", "left")
    canfd_id = int(os.environ.get("CANFD_ID", "0"))
    with L30(side=side, canfd_id=canfd_id, interface_type="libcanbus") as hand:
        assert hand.protocol_version in {"V6", "V6.2"}
        assert len(hand.position.get()) == 17
        assert len(hand.speed.get()) == 17
        assert len(hand.torque.get()) == 17
        assert len(hand.temperature.get()) == 17
        assert len(hand.current.get()) == 17
        assert len(hand.fault.get().error_codes) == 17
        assert set(hand.force_sensor.get()) == {"thumb", "index", "middle", "ring", "pinky"}

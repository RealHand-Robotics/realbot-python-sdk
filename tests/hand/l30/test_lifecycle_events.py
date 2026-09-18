"""Unit tests for lifecycle, snapshots, streams, and event typing."""

import pytest

from realhand.exceptions import StateError, ValidationError
from realhand.hand.l30 import PositionEvent, SensorSource

pytestmark = [pytest.mark.l30, pytest.mark.lifecycle, pytest.mark.streaming]


def test_snapshot_starts_empty_and_populates_after_read(l30):
    assert l30.get_snapshot().position is None
    l30.position.get()
    snapshot = l30.get_snapshot()
    assert snapshot.position is not None
    assert snapshot.position.positions == tuple(range(17))


def test_stream_emits_matching_event(l30):
    stream = l30.stream()
    l30._read_sensor(SensorSource.POSITION)
    event = stream.get(timeout=0.1)
    assert isinstance(event, PositionEvent)
    assert event.data.positions == tuple(range(17))


def test_stream_and_polling_validation(l30):
    with pytest.raises(ValidationError):
        l30.stream(maxsize=0)
    with pytest.raises(ValidationError):
        l30.start_polling({SensorSource.POSITION: 0})


def test_disable_close_and_post_close_guard(l30):
    l30.enable_all()
    l30.disable_all()
    assert not l30._motion_enabled
    l30.close()
    l30.close()
    assert l30.is_closed()
    with pytest.raises(StateError):
        l30.speed.set([10] * 17)

"""Package exports and safe protocol-selection integration tests."""

from types import SimpleNamespace

import pytest

import realhand
from realhand.hand import L30 as HandL30
from realhand.hand.l30 import L30, SensorSource
from realhand.hand.l30 import l30 as l30_module

pytestmark = [pytest.mark.l30, pytest.mark.basic]


def test_l30_is_exported_from_public_package_layers():
    assert realhand.L30 is L30
    assert HandL30 is L30
    assert SensorSource.ACCELERATION.value == "acceleration"


def test_connect_uses_legacy_v6_id_and_never_enables_during_probe(monkeypatch):
    calls = []

    class Controller:
        def __init__(self, **kwargs): calls.append(kwargs); self.protocol = SimpleNamespace()
        def connect(self): return True, "left"
        def disconnect(self): pass

    monkeypatch.setattr(l30_module.protocol_v1, "L30DexterousHandController", Controller)
    hand = object.__new__(L30)
    hand.side = "left"
    controller = hand._connect(canfd_id=0, channel=0, comm_type="libcanbus", bitrate=1_000_000, dbitrate=5_000_000, auto_setup=False)
    assert isinstance(controller, Controller)
    assert calls == [{
        "enable_on_connect": False, "canfd_id": 0, "channel": 0,
        "comm_type": "libcanbus", "bitrate": 1_000_000, "dbitrate": 5_000_000,
        "auto_setup": False, "device_id": 0x06,
    }]


def test_connect_tries_v62_without_overwriting_its_node_id(monkeypatch):
    calls = []

    class V6:
        def __init__(self, **kwargs): pass
        def connect(self): return True, "right"
        def disconnect(self): pass

    class V62:
        def __init__(self, **kwargs): calls.append(kwargs)
        def connect(self): return True, "left"
        def disconnect(self): pass

    monkeypatch.setattr(l30_module.protocol_v1, "L30DexterousHandController", V6)
    monkeypatch.setattr(l30_module.protocol_v2, "L30DexterousHandController", V62)
    hand = object.__new__(L30); hand.side = "left"
    hand._connect(canfd_id=0, channel=0, comm_type="libcanbus", bitrate=1, dbitrate=2, auto_setup=False)
    assert calls[0]["enable_on_connect"] is False
    assert "device_id" not in calls[0]

"""Pure unit tests for V6 CANFD framing and payload helpers."""

import pytest

from realhand.hand.l30.realhand_l30_v6_canfd import (
    Command,
    L30CANFDProtocol,
    MessageDirection,
    ReadWrite,
    get_dlc_from_length,
    get_length_from_dlc,
)

pytestmark = [pytest.mark.l30, pytest.mark.basic]


class RecordingTransport:
    def __init__(self): self.sent = []
    def send(self, frame_id, payload): self.sent.append((frame_id, payload)); return True


def make_protocol():
    protocol = object.__new__(L30CANFDProtocol)
    protocol.device_id = 0x06
    protocol.frame_counter = 0
    protocol.is_connected = True
    protocol.transport = RecordingTransport()
    return protocol


def test_dlc_length_round_trip_and_vendor_extensions():
    assert get_dlc_from_length(8) == 8
    assert get_dlc_from_length(9) == 9
    assert get_dlc_from_length(64) == 15
    assert get_length_from_dlc(15) == 64
    assert get_length_from_dlc(0x10) == 16
    assert get_length_from_dlc(0x40) == 64


def test_v6_can_id_build_and_parse_are_inverse():
    protocol = make_protocol()
    frame_id = protocol._build_canfd_id(
        priority=2, direction=MessageDirection.RESPONSE, rw=ReadWrite.READ,
        device_id=6, command=Command.JOINT_SPEED, subcommand=3,
    )
    assert protocol._parse_canfd_id(frame_id) == {
        "priority": 2, "direction": 1, "rw": 0,
        "device_id": 6, "command": Command.JOINT_SPEED, "subcommand": 3,
    }


def test_v6_send_message_builds_expected_transaction_payload():
    protocol = make_protocol()
    assert protocol.send_message(Command.JOINT_ACCELERATION, bytes([20]) * 17)
    frame_id, payload = protocol.transport.sent[-1]
    fields = protocol._parse_canfd_id(frame_id)
    assert fields["command"] == Command.JOINT_ACCELERATION
    assert fields["rw"] == ReadWrite.WRITE
    assert payload == bytes([17, 0x10]) + bytes([20]) * 17


def test_v6_pack_unpack_preserves_signed_joint_values():
    protocol = make_protocol()
    values = [-200, 0, 1600]
    packed = protocol._pack_joint_data(values, -200, 1600)
    assert protocol._unpack_joint_data(packed, 0, len(values)) == values

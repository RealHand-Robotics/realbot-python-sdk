#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'L30 smart hand CANFD communication control program\n17 degrees of freedom dexterous hand control system based on the new version of CANFD protocol\n\nProtocol specifications:\n- Physical interface: CANFD\n- Arbitration baud rate: 1Mbps (80%)\n- Data baud rate: 5Mbps (75%)\n- Frame format: extended frame, data frame protocol\n\nAuthor: hejianxin\nVersion: 2.1\nDate: 2026-04-14'

import sys
import os
import time
import logging
import struct
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from ctypes import *

import numpy as np

from .realhand_l30_canfd_comm import (
    LibCanBusTransport,
    SocketCANTransport,
    create_transport,
)

logger = logging.getLogger(__name__)


#CANFD library constants and structure definitions
STATUS_OK = 0


class CanFD_Config(Structure):
    _fields_ = [
        ("NomBaud", c_uint),
        ("DatBaud", c_uint),
        ("NomPres", c_ushort),
        ("NomTseg1", c_char),
        ("NomTseg2", c_char),
        ("NomSJW", c_char),
        ("DatPres", c_char),
        ("DatTseg1", c_char),
        ("DatTseg2", c_char),
        ("DatSJW", c_char),
        ("Config", c_char),
        ("Model", c_char),
        ("Cantype", c_char)
    ]


class CanFD_Msg(Structure):
    _fields_ = [
        ("ID", c_uint),
        ("TimeStamp", c_uint),
        ("FrameType", c_ubyte),
        ("DLC", c_ubyte),
        ("ExternFlag", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("BusSatus", c_ubyte),
        ("ErrSatus", c_ubyte),
        ("TECounter", c_ubyte),
        ("RECounter", c_ubyte),
        ("Data", c_ubyte * 64)
    ]


class Dev_Info(Structure):
    _fields_ = [
        ("HW_Type", c_char * 32),
        ("HW_Ser", c_char * 32),
        ("HW_Ver", c_char * 32),
        ("FW_Ver", c_char * 32),
        ("MF_Date", c_char * 32)
    ]


#DLC to actual data length mapping table
DLC_TO_LENGTH = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}

#Non-standard DLC mapping
NON_STANDARD_DLC_MAP = {0x10: 16, 0x40: 64}


def get_dlc_from_length(length: int) -> int:
    'Get the DLC value based on the data length'
    if length <= 8:
        return length
    if length <= 12:
        return 9
    if length <= 16:
        return 10
    if length <= 20:
        return 11
    if length <= 24:
        return 12
    if length <= 32:
        return 13
    if length <= 48:
        return 14
    return 15


def get_length_from_dlc(dlc: int) -> int:
    'Get the actual data length based on the DLC value\n\n    The device may return non-standard DLC values (such as 0x40=64 or 0x10=16), which requires special handling'
    if dlc in DLC_TO_LENGTH:
        return DLC_TO_LENGTH[dlc]
    if dlc in NON_STANDARD_DLC_MAP:
        return NON_STANDARD_DLC_MAP[dlc]
    return min(dlc, 64)


# =============================================================================
#Protocol constant definition
# =============================================================================

class Priority(IntEnum):
    'CANFDID priority'
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    LOWEST = 3


class MessageDirection(IntEnum):
    'Message direction'
    REQUEST = 0  #request/TX
    RESPONSE = 1  #response/RX


class ReadWrite(IntEnum):
    'Read and write flag'
    READ = 0
    WRITE = 1


class HandType(IntEnum):
    'Right and left handed type'
    LEFT = 0
    RIGHT = 1


class StatusCode(IntEnum):
    'Return status code'
    OK = 0x00
    HAND_NOT_SET = 0x01
    MOTOR_DISABLED = 0x02
    NOT_CALIBRATED = 0x03
    READ_ERROR = 0xF0


class Command(IntEnum):
    'Command code'
    JOINT_POSITION = 0x01       #Joint position reading and writing
    JOINT_TORQUE = 0x02          #Joint torque writing
    JOINT_TORQUE_LIMIT = 0x03   #Joint torque limit
    JOINT_SPEED = 0x05          #Joint speed reading and writing
    JOINT_ACCELERATION = 0x07   #Joint acceleration write
    JOINT_ENABLE = 0x08         #Joint enable reading and writing
    JOINT_TEMPERATURE = 0x33    #Joint temperature reading
    JOINT_ERROR_CODE = 0x35      #Joint error code reading
    JOINT_CURRENT = 0x36         #Joint current reading
    CALIBRATE_ZERO = 0x37        #calibration zero point
    EMERGENCY_STOP = 0x37        #Emergency stop
    THUMB_PRESSURE = 0xB1        #Thumb pressure data
    INDEX_PRESSURE = 0xB2        #Index finger pressure data
    MIDDLE_PRESSURE = 0xB3      #Middle finger pressure data
    RING_PRESSURE = 0xB4        #Ring finger pressure data
    PINKY_PRESSURE = 0xB5       #Little finger pressure data
    DEVICE_CODE = 0xC0          #Device encoding
    DEVICE_VERSION = 0xC1       #Device Version
    DEVICE_CANFDID = 0xC3       #CANFDID and left and right hands
    RESTORE_DEVICE_ID = 0xC4   #Restore Device ID


# =============================================================================
#Joint information definition
# =============================================================================

@dataclass
class JointInfo:
    'Joint information'
    id: int
    name: str
    finger: str
    min_pos: int = -32768
    max_pos: int = 32767
    current_pos: int = 0
    target_pos: int = 0
    current_vel: int = 0
    target_vel: int = 0
    target_torque: int = 0
    torque_limit: int = 1000
    acceleration: int = 0
    enabled: bool = False
    temperature: int = 0
    error_code: int = 0
    current: int = 0

    def update_from_reading(self, positions: Optional[List[int]] = None,
                            velocities: Optional[List[int]] = None,
                            currents: Optional[List[int]] = None,
                            temperature: Optional[int] = None,
                            error_code: Optional[int] = None,
                            enabled: Optional[bool] = None) -> None:
        'Update joint information from read results'
        if positions is not None:
            self.current_pos = positions
            self.target_pos = positions
        if velocities is not None:
            self.current_vel = velocities
            self.target_vel = velocities
        if currents is not None:
            self.current = currents
        if temperature is not None:
            self.temperature = temperature
        if error_code is not None:
            self.error_code = error_code
        if enabled is not None:
            self.enabled = enabled

    @property
    def range(self) -> Tuple[int, int]:
        'Get joint range'
        return (self.min_pos, self.max_pos)


#Joint Definitions - Arranged in Protocol Order
#Sequence: Thumb base bend, thumb tip bend, thumb side swing, thumb rotation,
#The ring finger swings sideways, the tip of the ring finger is bent, the base of the ring finger is bent,
#The base of the middle finger is bent, the tip of the middle finger is bent, the root of the little finger is bent, the tip of the little finger is bent, the little finger swings sideways,
#Middle finger side swing, index finger side swing, index finger base bend, index finger tip bend, wrist bend
JOINT_DEFINITIONS = [
    #Thumb (4 DOF)
    JointInfo(1, 'Bent finger root', 'Thumb', 0, 1000),
    JointInfo(2, 'Fingertips curved', 'Thumb', 0, 1500),
    JointInfo(3, 'Side swing', 'Thumb', 0, 1000),
    JointInfo(4, 'rotation', 'Thumb', 0, 900),
    #Ring Finger (3 DOF)
    JointInfo(5, 'Side swing', 'Ring finger', -200, 200),
    JointInfo(6, 'Fingertips curved', 'Ring finger', 0, 1500),
    JointInfo(7, 'Bent finger root', 'Ring finger', 0, 1600),
    #Middle finger (3 DOF)
    JointInfo(8, 'Bent finger root', 'middle finger', 0, 1600),
    JointInfo(9, 'Fingertips curved', 'middle finger', 0, 1500),
    JointInfo(10, 'Side swing', 'middle finger', -200, 200),  #Note: The side swing of the middle finger in the picture is index 12
    #Pinky (3 DOF)
    JointInfo(11, 'Bent finger root', 'pinky', 0, 1600),
    JointInfo(12, 'Fingertips curved', 'pinky', 0, 1500),
    JointInfo(13, 'Side swing', 'pinky', -200, 200),
    #Index finger (3 DOF)
    JointInfo(14, 'Side swing', 'index finger', -200, 200),
    JointInfo(15, 'Bent finger root', 'index finger', 0, 1600),
    JointInfo(16, 'Fingertips curved', 'index finger', 0, 1500),
    #Wrist (1 DOF)
    JointInfo(17, 'Pitch', 'wrist', -1000, 1000),
]

#Joint limit dictionary
JOINT_LIMITS = {
    1: (0, 1000),      #0: The base of the thumb is bent
    2: (0, 1500),      #1: Thumb tip is bent
    3: (0, 1000),      #2: Thumb side swing
    4: (0, 900),       #3: Thumb rotation
    5: (-200, 200),    #4: Ring finger swing
    6: (0, 1500),      #5: The tip of the ring finger is bent
    7: (0, 1600),      #6: The base of the ring finger is bent
    8: (0, 1600),      #7: The base of the middle finger is bent
    9: (0, 1500),      #8: The tip of the middle finger is bent
    10: (0, 1600),     #9: The base of the little finger is bent
    11: (0, 1500),     #10: The tip of the little finger is bent
    12: (-200, 200),   #11: little finger side swing
    13: (-200, 200),   #12: Middle finger side swing
    14: (-200, 200),   #13: Index finger swing
    15: (0, 1600),     #14: The base of the index finger is bent
    16: (0, 1500),     #15: The tip of the index finger is bent
    17: (-1000, 1000), #16: Wrist
}
#Joint name mapping between Chinese and English
JOINT_NAME_EN = ["thumb_cmc_pitch", "thumb_ip_pitch", "thumb_cmc_yaw", "thumb_cmc_roll",
                 "ring_mcp_roll", "ring_pip_pitch", "ring_mcp_pitch",
                 "middle_mcp_pitch", "middle_pip_pitch",
                 "pinky_mcp_pitch", "pinky_pip_pitch", "pinky_mcp_roll",
                 "middle_mcp_roll", "index_mcp_roll",
                 "index_mcp_pitch", "index_pip_pitch",
                 "wrist_pitch"]

JOINT_NAME_CN = ['The base of the thumb is bent', 'Thumb tip bent', 'Thumb side swing', 'Thumb rotation',
                 'Ring finger side swing', 'The tip of the ring finger is bent', 'The base of the ring finger is bent',
                 'The base of the middle finger is bent', 'The tip of the middle finger is bent', 'The base of the little finger is bent',
                 'Bent tip of little finger', 'Little finger side swing', 'Middle finger side swing', 'Index finger side swing',
                 'The base of the index finger is bent', 'The tip of the index finger is bent', 'wrist']

#joint range mapping (string key version)
JOINT_RANGE = {
    "thumb_cmc_pitch": (0, 1000),    #The base of the thumb is bent
    "thumb_mcp_pitch": (0, 1500),     #Thumb tip bent
    "thumb_cmc_yaw": (0, 1000),      #Thumb side swing
    "thumb_cmc_roll": (0, 900),      #Thumb rotation
    "ring_mcp_roll": (-200, 200),    #Ring finger side swing
    "ring_pip_pitch": (0, 1500),     #The tip of the ring finger is bent
    "ring_mcp_pitch": (0, 1600),     #The base of the ring finger is bent
    "middle_mcp_pitch": (0, 1600),   #The base of the middle finger is bent
    "middle_pip_pitch": (0, 1500),   #The tip of the middle finger is bent
    "pinky_mcp_pitch": (0, 1600),    #The base of the little finger is bent
    "pinky_pip_pitch": (0, 1500),    #Bent tip of little finger
    "pinky_mcp_roll": (-200, 200),   #Little finger side swing
    "middle_mcp_roll": (-200, 200),  #Middle finger side swing
    "index_mcp_roll": (-200, 200),   #Index finger side swing
    "index_mcp_pitch": (0, 1600),    #The base of the index finger is bent
    "index_pip_pitch": (0, 1500),    #The tip of the index finger is bent
    "wrist": (-1000, 1000)           #wrist
}


# =============================================================================
#CANFD communication class
# =============================================================================

class L30CANFDProtocol:
    'L30 smart hand CANFD communication protocol implementation'

    ARBITRATION_BAUD = 1000000
    DATA_BAUD = 5000000
    DEFAULT_DEVICE_ID = 0x06

    def __init__(self, device_id: int = 0x06, canfd_device=0, channel=0,
                 comm_type: str = "libcanbus", bitrate: int = 1000000,
                 dbitrate: int = 5000000, auto_setup: bool = True):
        'Args:\n            device_id: Smart hand device ID\n            canfd_device: CANFD device (box) index, only used by the libcanbus backend\n            channel: channel - libcanbus is int (default 0), socketcan is the interface name str (such as "can0")\n            comm_type: communication backend - "libcanbus" (default, vendor private library) or\n                       "socketcan" (kernel can0 + python-can, transparent plastic USB-CANFD device)\n            bitrate/dbitrate/auto_setup: only used by socketcan - arbitration/data segment baud rate and\n                       Whether to automatically pull up the interface'
        self.device_id = device_id
        self.canfd_device = canfd_device
        self.channel = channel
        self.comm_type = comm_type
        #Pluggable transport backend: libcanbus (default) or socketcan; actual sending and receiving is delegated to it
        self.transport = create_transport(
            comm_type=comm_type, canfd_device=canfd_device, channel=channel,
            bitrate=bitrate, dbitrate=dbitrate, auto_setup=auto_setup)
        self.is_connected = False
        self.frame_counter = 0
        self._pressure_matrices = {
            'thumb': np.full((12, 6), -1),
            'index': np.full((12, 6), -1),
            'middle': np.full((12, 6), -1),
            'ring': np.full((12, 6), -1),
            'little': np.full((12, 6), -1),
        }

    # =========================================================================
    #Underlying communication method (delegate pluggable transport backend)
    # =========================================================================

    def initialize(self) -> bool:
        'Initialize CANFD communication (entrusted transmission backend: libcanbus or socketcan)'
        logger.info('Initializing CANFD communication...')
        ok = self.transport.initialize()
        self.is_connected = ok
        if ok:
            logger.info('CANFD communication initialization completed')
        return ok

    def close(self) -> None:
        'Close CANFD connection (delegated transport backend)'
        self.transport.close()
        self.is_connected = False
        logger.info('CANFD connection closed')

    def _increment_frame_counter(self) -> int:
        'Increment frame counter'
        self.frame_counter = (self.frame_counter + 1) & 0xF
        return self.frame_counter

    def _build_canfd_id(
        self,
        priority: int = 0,
        direction: int = MessageDirection.REQUEST,
        rw: int = ReadWrite.WRITE,
        device_id: int = None,
        command: int = 0,
        subcommand: int = 0
    ) -> int:
        'Build CANFD extended frame ID\n        \n        BIT[28:27] - Priority\n        BIT[26:26] - Message direction (0: request, 1: response)\n        BIT[25:25] - R/W (0: read, 1: write)\n        BIT[24:20] - Device ID\n        BIT[19:12] - command\n        BIT[11:08] - Subcommand\n        BIT[07:00] - Reserved'
        if device_id is None:
            device_id = self.device_id

        frame_id = 0
        frame_id |= (priority & 0x3) << 27
        frame_id |= (direction & 0x1) << 26
        frame_id |= (rw & 0x1) << 25
        frame_id |= (device_id & 0x1F) << 20
        frame_id |= (command & 0xFF) << 12
        frame_id |= (subcommand & 0xF) << 8
        return frame_id

    def _parse_canfd_id(self, frame_id: int) -> Dict:
        'Parse CANFD extension frame ID'
        return {
            'priority': (frame_id >> 27) & 0x3,
            'direction': (frame_id >> 26) & 0x1,
            'rw': (frame_id >> 25) & 0x1,
            'device_id': (frame_id >> 20) & 0x1F,
            'command': (frame_id >> 12) & 0xFF,
            'subcommand': (frame_id >> 8) & 0xF,
        }

    def _build_transaction_control(self, total_frames: int = 0, seq_num: int = 0) -> int:
        'Build transaction control bytes\n\n        BIT[07:04] - Frame count\n        BIT[03:02] - Total number of frames (0: single frame, >0: multiple frames)\n        BIT[01:00] - multi-frame sequence number'
        return ((self.frame_counter & 0xF) << 4) | ((total_frames & 0x3) << 2) | (seq_num & 0x3)

    def _parse_transaction_control(self, control: int) -> Dict:
        'Parse transaction control byte'
        return {
            'frame_counter': (control >> 4) & 0xF,
            'total_frames': (control >> 2) & 0x3,
            'seq_num': control & 0x3,
        }

    def _pack_joint_data(self, values: List[int], min_val: int, max_val: int) -> bytes:
        'Common method: Pack joint values into big-endian bytes'
        data = bytearray()
        for val in values:
            clamped = max(min_val, min(max_val, int(val)))
            data.extend(clamped.to_bytes(2, byteorder='big', signed=True))
        return bytes(data)

    def _unpack_joint_data(self, data: bytes, offset: int, count: int) -> List[int]:
        'General method: Unpack joint data'
        values = []
        for i in range(count):
            val = int.from_bytes(data[offset + i*2:offset + i*2 + 2], byteorder='big', signed=True)
            values.append(val)
        return values

    def _check_response_status(self, data: bytes, min_len: int) -> Tuple[bool, int]:
        'Check response status'
        if len(data) >= min_len:
            return data[2] == StatusCode.OK, data[2]
        return False, -1

    def send_message(
        self,
        command: int,
        data: bytes,
        is_write: bool = True,
        subcommand: int = 0
    ) -> bool:
        'Send CANFD messages (delegate transport backend)\n\n        Args:\n            command: command code\n            data: data payload\n            is_write: whether it is a write operation\n            subcommand: subcommand'
        if not self.is_connected:
            logger.error('Error: CANFD not connected')
            return False

        try:
            frame_id = self._build_canfd_id(
                priority=Priority.HIGH,
                direction=MessageDirection.REQUEST,
                rw=ReadWrite.WRITE if is_write else ReadWrite.READ,
                device_id=self.device_id,
                command=command,
                subcommand=subcommand
            )

            self._increment_frame_counter()
            data_len = min(len(data), 62)

            #Valid bytes of data segment: BYTE0 length / BYTE1 transaction control / BYTE2~ data
            payload = bytearray(2 + data_len)
            payload[0] = data_len
            payload[1] = self._build_transaction_control()
            payload[2:2 + data_len] = bytes(data[:data_len])

            return self.transport.send(frame_id, bytes(payload))

        except Exception as e:
            logger.error(f"Exception sending message: {e}")
            return False

    def receive_messages(
        self,
        timeout_ms: int = 3,
        filter_device_id: bool = True,
        expected_command: int = None
    ) -> List[Tuple[int, bytes, Dict]]:
        'Receive CANFD message (entrust the transmission backend to retrieve the original frame and parse it)\n\n        Returns:\n            List of (frame_id, data, parsed_info)'
        if not self.is_connected:
            return []

        try:
            messages = []
            for frame_id, data in self.transport.receive(timeout_ms):
                parsed = self._parse_canfd_id(frame_id)
                parsed['transaction'] = self._parse_transaction_control(data[1]) if len(data) > 1 else {}

                if filter_device_id and parsed['device_id'] != self.device_id:
                    continue

                if expected_command is not None and parsed['command'] != expected_command:
                    continue

                messages.append((frame_id, data, parsed))

            return messages

        except Exception as e:
            logger.error(f"Exception receiving message: {e}")
            return []

    def _wait_for_response(
        self,
        expected_command: int,
        timeout_ms: int = 200,
        expected_rw: int = ReadWrite.READ
    ) -> Optional[Tuple[bytes, int]]:
        'Wait and get response\n\n        Returns:\n            (data, status_code) or None'
        start_time = time.time()
        while time.time() - start_time < timeout_ms / 1000:
            messages = self.receive_messages(
                timeout_ms=50,
                expected_command=expected_command
            )

            for frame_id, data, parsed in messages:
                if parsed['direction'] == MessageDirection.RESPONSE:
                    if parsed['rw'] == expected_rw:
                        status = data[2] if len(data) > 2 else 0
                        return data, status

            time.sleep(0.005)

        return None

    # =========================================================================
    #Device Operation
    # =========================================================================

    def enable_all_joints(self) -> bool:
        'Enable all 17 joints\n\n        Important: This function must be called to enable the joint before performing other operations!'
        return self.set_joint_enable([1] * 17)

    def query_device_type(self) -> Optional[str]:
        'Query device type (left/right hand)'
        if not self.send_message(Command.DEVICE_CANFDID, b'', is_write=False):
            return None

        response = self._wait_for_response(Command.DEVICE_CANFDID)

        if response:
            data, status = response
            if status == 0 and len(data) >= 5:
                device_id = data[3]
                hand_type = "right" if data[4] == HandType.RIGHT else "left"
                logger.info(f"Detected device ID: {device_id}, type: {hand_type}")
                return hand_type

        return None

    def get_device_version(self) -> Optional[Dict]:
        'Read device version'
        if not self.send_message(Command.DEVICE_VERSION, b'', is_write=False):
            return None

        response = self._wait_for_response(Command.DEVICE_VERSION)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 12:
                def parse_version(d, start):
                    return f"{d[start]}.{d[start + 1]}.{d[start + 2]}"

                return {
                    'hardware': parse_version(data, 3),
                    'software': parse_version(data, 6),
                    'mechanical': parse_version(data, 9)
                }

        return None

    def set_device_id(self, device_id: int, hand_type: HandType) -> bool:
        'Set device ID and left and right hands'
        return self.send_message(Command.DEVICE_CANFDID, bytes([device_id, hand_type]), is_write=True)

    def restore_device_id(self) -> bool:
        'Restore device ID to 0x01'
        return self.send_message(Command.RESTORE_DEVICE_ID, b'', is_write=True)

    # =========================================================================
    #Joint control
    # =========================================================================

    def set_joint_positions(self, positions: List[int]) -> bool:
        'Set 17 joint positions'
        if len(positions) != 17:
            logger.error(f"Position data length error: expected 17, actual {len(positions)}")
            return False

        data = bytearray()
        for i, pos in enumerate(positions):
            min_val, max_val = JOINT_LIMITS.get(i + 1, (-32768, 32767))
            clamped_pos = max(min_val, min(max_val, int(pos)))
            data.extend(clamped_pos.to_bytes(2, byteorder='big', signed=True))

        return self.send_message(Command.JOINT_POSITION, bytes(data), is_write=True)

    def get_joint_positions(self) -> Optional[List[int]]:
        'Read 17 joint positions'
        if not self.send_message(Command.JOINT_POSITION, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_POSITION)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 36:
                return self._unpack_joint_data(data, 3, 17)

        return None

    def set_joint_torques(self, torques: List[int]) -> bool:
        'Set 17 joint torques (range -2047~2047, unit 6.5mA)'
        if len(torques) != 17:
            logger.error(f"Torque data length error: expected 17, actual {len(torques)}")
            return False

        return self.send_message(Command.JOINT_TORQUE,
                                self._pack_joint_data(torques, -2047, 2047),
                                is_write=True)

    def get_joint_torques(self) -> Optional[List[int]]:
        'Read 17 joint torques'
        if not self.send_message(Command.JOINT_TORQUE, b'\x00\x10', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_TORQUE, expected_rw=ReadWrite.READ)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 38:
                return self._unpack_joint_data(data, 3, 17)

        return None

    def set_joint_torque_limits(self, limits: List[int]) -> bool:
        'Set 17 joint torque limits (range 0~1000, unit 0.1%)'
        if len(limits) != 17:
            logger.error(f"Torque limit data length error: expected 17, actual {len(limits)}")
            return False

        return self.send_message(Command.JOINT_TORQUE_LIMIT,
                                self._pack_joint_data(limits, 0, 1000),
                                is_write=True)

    def set_joint_velocities(self, velocities: List[int]) -> bool:
        'Set 17 joint speeds (range -32767~32767, unit 0.732RPM)'
        if len(velocities) != 17:
            logger.error(f"Velocity data length error: expected 17, actual {len(velocities)}")
            return False

        return self.send_message(Command.JOINT_SPEED,
                                self._pack_joint_data(velocities, 0, 150),
                                is_write=True)

    def get_joint_velocities(self) -> Optional[List[int]]:
        'Read 17 joint speeds'
        if not self.send_message(Command.JOINT_SPEED, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_SPEED)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 36:
                return self._unpack_joint_data(data, 3, 17)

        return None

    def set_joint_accelerations(self, accelerations: List[int]) -> bool:
        'Set 17 joint accelerations (range 0~254, unit 8.7 degrees/second²)'
        if len(accelerations) != 17:
            logger.error(f"Acceleration data length error: expected 17, actual {len(accelerations)}")
            return False

        data = bytes(max(0, min(254, int(acc))) for acc in accelerations)
        return self.send_message(Command.JOINT_ACCELERATION, data, is_write=True)

    def get_joint_accelerations(self) -> Optional[List[int]]:
        'Read the acceleration of 17 joints'
        if not self.send_message(Command.JOINT_ACCELERATION, b'\x00\x10', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_ACCELERATION, expected_rw=ReadWrite.READ)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 20:
                return list(data[3:20])

        return None

    def set_joint_enable(self, enables: List[int]) -> bool:
        'Set 17 joint enable states'
        if len(enables) != 17:
            logger.error(f"Enable data length error: expected 17, actual {len(enables)}")
            return False

        data = bytes(1 if e else 0 for e in enables)
        return self.send_message(Command.JOINT_ENABLE, data, is_write=True)

    def get_joint_enable(self) -> Optional[List[int]]:
        'Read 17 joint enable status'
        if not self.send_message(Command.JOINT_ENABLE, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_ENABLE)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 20:
                return list(data[3:20])

        return None

    def get_joint_temperatures(self) -> Optional[List[int]]:
        'Read 17 joint temperatures (in °C)'
        if not self.send_message(Command.JOINT_TEMPERATURE, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_TEMPERATURE)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 20:
                return list(data[3:20])

        return None

    def get_joint_error_codes(self) -> Optional[List[int]]:
        'Read 17 joint error codes'
        if not self.send_message(Command.JOINT_ERROR_CODE, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_ERROR_CODE)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 20:
                return list(data[3:20])

        return None

    def get_joint_currents(self) -> Optional[List[int]]:
        'Read the current current of 17 joints (unit 6.5mA)'
        if not self.send_message(Command.JOINT_CURRENT, b'\x00', is_write=False):
            return None

        response = self._wait_for_response(Command.JOINT_CURRENT)
        if response:
            data, status = response
            if status == StatusCode.OK and len(data) >= 36:
                return self._unpack_joint_data(data, 3, 17)

        return None

    def calibrate_zero_point(self) -> bool:
        'Calibrate zero point - after execution, put your hand into the mold and wait for about 1 second to enter the disabled mode'
        return self.send_message(Command.CALIBRATE_ZERO, b'\x00', is_write=True)

    def emergency_stop(self) -> bool:
        'Emergency stop'
        return self.send_message(Command.EMERGENCY_STOP, b'\x00', is_write=True)

    # =========================================================================
    #Tactile sensor
    # =========================================================================

    def get_finger_pressure(self, command: Command) -> Optional[List[int]]:
        'Read finger pressure data\n\n        Args:\n            command: 0xB1-0xB5 corresponds to thumb to little finger\n\n        Returns:\n            72-byte pressure data list, or None'
        if not self.send_message(command, b'\x00', is_write=False):
            return None

        frame1_data = None
        frame2_data = None
        start_time = time.time()

        while time.time() - start_time < 0.002:
            messages = self.receive_messages(timeout_ms=2, filter_device_id=True, expected_command=command)

            for frame_id, data, parsed in messages:
                if len(data) < 4:
                    continue

                data_len = data[0]

                if data_len == 0x3E and frame1_data is None:
                    frame1_data = list(data[3:])
                elif data_len == 0x0C and frame1_data is not None and frame2_data is None:
                    frame2_data = list(data[3:-2])
                    return frame1_data + frame2_data

            # time.sleep(0.001)

        if frame1_data is not None and frame2_data is not None:
            return frame1_data + frame2_data
        return None

    def _process_pressure_matrix(self, data: Optional[List[int]], key: str) -> np.ndarray:
        'Processing pressure matrix data'
        if data is not None:
            arr = np.array(data).reshape(12, 6)
            self._pressure_matrices[key] = arr[::-1]
        return self._pressure_matrices[key]

    def get_thumb_pressure(self) -> np.ndarray:
        'Read thumb pressure data'
        return self._process_pressure_matrix(self.get_finger_pressure(Command.THUMB_PRESSURE), 'thumb')

    def get_index_pressure(self) -> np.ndarray:
        'Read index finger pressure data'
        return self._process_pressure_matrix(self.get_finger_pressure(Command.INDEX_PRESSURE), 'index')

    def get_middle_pressure(self) -> np.ndarray:
        'Read middle finger pressure data'
        return self._process_pressure_matrix(self.get_finger_pressure(Command.MIDDLE_PRESSURE), 'middle')

    def get_ring_pressure(self) -> np.ndarray:
        'Read ring finger pressure data'
        return self._process_pressure_matrix(self.get_finger_pressure(Command.RING_PRESSURE), 'ring')

    def get_little_pressure(self) -> np.ndarray:
        'Read little finger pressure data'
        return self._process_pressure_matrix(self.get_finger_pressure(Command.PINKY_PRESSURE), 'little')

    def get_all_pressures(self) -> Dict[str, List[int]]:
        'Read all finger pressure data'
        return {
            'thumb_matrix': self.get_thumb_pressure().tolist(),
            'index_matrix': self.get_index_pressure().tolist(),
            'middle_matrix': self.get_middle_pressure().tolist(),
            'ring_matrix': self.get_ring_pressure().tolist(),
            'little_matrix': self.get_little_pressure().tolist()
        }


# =============================================================================
#Dexterous Hand Controller
# =============================================================================

class L30DexterousHandController:
    'L30 Dexterous Hand Advanced Controller'

    JOINT_COUNT = 17

    def __init__(self, device_id: int = 0x06, canfd_id=0, comm_type: str = "libcanbus",
                 channel=None, bitrate: int = 1000000, dbitrate: int = 5000000,
                 auto_setup: bool = True, enable_on_connect: bool = False):
        'Args:\n            device_id: Smart hand device ID\n            canfd_id: CANFD device (box) index\n            comm_type: communication backend - "libcanbus" (default, vendor private library) or\n                       "socketcan" (kernel can0 + python-can, transparent plastic USB-CANFD device)\n            channel: channel - libcanbus is int (default 0), socketcan is the interface name (default "can0")\n            bitrate/dbitrate/auto_setup: only used by socketcan'
        #channel is not explicitly specified, the default value is given by the backend: socketcan -> "can0", libcanbus -> 0
        if channel is None:
            channel = "can0" if comm_type == "socketcan" else 0
        self.protocol = L30CANFDProtocol(device_id, canfd_id, channel=channel,
                                         comm_type=comm_type, bitrate=bitrate,
                                         dbitrate=dbitrate, auto_setup=auto_setup)
        self.device_id = device_id
        self.hand_type: Optional[str] = None
        self.enable_on_connect = enable_on_connect
        self.joints = {joint.id: joint for joint in JOINT_DEFINITIONS}

    def connect(self) -> Tuple[bool, Optional[str]]:
        'Connect with smart hands'
        logger.info('Start connecting smart hands...')

        if not self.protocol.initialize():
            return False, None

        if self.enable_on_connect:
            logger.info('Enable all joints...')
            print('Enable all joints...')
            if not self.protocol.enable_all_joints():
                logger.warning('Joint enable failed, continue trying...')
                print('Joint enable failed, continue trying...')
            time.sleep(0.1)

        #Device type query occasionally times out/loses packets, retry several times to avoid misjudgment as hand_type mismatch
        hand_type = None
        for _ in range(5):
            hand_type = self.protocol.query_device_type()
            if hand_type:
                break
            time.sleep(0.05)
        if hand_type:
            self.hand_type = hand_type
            logger.info(f"Connection successful: device ID={self.device_id}, type={hand_type}")
            return True, hand_type

        return True, 'Unknown'

    def disconnect(self) -> None:
        'Disconnect'
        self.protocol.close()

    @property
    def is_connected(self) -> bool:
        'Check if connected'
        return self.protocol.is_connected

    def set_positions(self, positions: List[int]) -> bool:
        'Set joint position'
        if len(positions) != self.JOINT_COUNT:
            raise ValueError(f"requires {self.JOINT_COUNT} joint values")
        clamped = [self._clamp_joint(i + 1, pos) for i, pos in enumerate(positions)]
        return self.protocol.set_joint_positions(clamped)

    def _clamp_joint(self, joint_id: int, value: int) -> int:
        'Limit joint values to range'
        min_val, max_val = JOINT_LIMITS.get(joint_id, (-32768, 32767))
        return max(min_val, min(max_val, int(value)))

    def get_positions(self) -> Optional[List[int]]:
        'Get joint position'
        return self.protocol.get_joint_positions()

    def set_velocities(self, velocities: List[int]) -> bool:
        'Set joint speed'
        return self.protocol.set_joint_velocities(velocities)

    def get_velocities(self) -> Optional[List[int]]:
        'Get joint speed'
        return self.protocol.get_joint_velocities()

    def set_accelerations(self, accelerations: List[int]) -> bool:
        'Set joint acceleration (0~254, unit 8.7 degrees/second²).'
        return self.protocol.set_joint_accelerations(accelerations)

    def get_accelerations(self) -> Optional[List[int]]:
        'Read joint acceleration.'
        return self.protocol.get_joint_accelerations()

    def set_torques(self, torques: List[int]) -> bool:
        'Set joint torque'
        return self.protocol.set_joint_torques(torques)

    def set_torque_limits(self, limits: List[int]) -> bool:
        'Set joint torque limit (0~1000, unit 0.1%).'
        return self.protocol.set_joint_torque_limits(limits)

    def get_torques(self) -> Optional[List[int]]:
        'Get joint torque'
        return self.protocol.get_joint_torques()

    def set_enable(self, enables: List[int]) -> bool:
        'Set joint enable'
        return self.protocol.set_joint_enable(enables)

    def calibrate(self) -> bool:
        'calibration zero point'
        return self.protocol.calibrate_zero_point()

    def stop(self) -> bool:
        'Emergency stop'
        return self.protocol.emergency_stop()

    def enable_all(self) -> bool:
        'Enable all joints'
        return self.protocol.enable_all_joints()

    def get_matrix_touch(self) -> Dict[str, List[int]]:
        'Get touch matrix'
        return self.protocol.get_all_pressures()

    def get_joint_name(self) -> Tuple[List[str], List[str]]:
        'Get joint name'
        return JOINT_NAME_EN, JOINT_NAME_CN

    def get_joint_range(self) -> Dict[str, Tuple[int, int]]:
        'Get joint range'
        return JOINT_RANGE

    def get_all_state(self,is_touch=False) -> Optional[Dict]:
        'Get complete status'
        all_state =  {
            'positions': self.protocol.get_joint_positions(),
            'velocities': self.protocol.get_joint_velocities(),
            'currents': self.protocol.get_joint_currents(),
            'temperatures': self.protocol.get_joint_temperatures(),
            'error_codes': self.protocol.get_joint_error_codes()
        }
        if is_touch == True:
            all_state['matrix_touch'] = self.get_matrix_touch()
        return all_state

    def normalize_positions(self, raw_positions: List[int]) -> List[float]:
        'Normalizes raw position values to the 0-1 range'
        normalized = []
        for i, pos in enumerate(raw_positions):
            motor_id = i + 1
            if motor_id in JOINT_LIMITS:
                min_val, max_val = JOINT_LIMITS[motor_id]
                if max_val != min_val:
                    norm = (pos - min_val) / (max_val - min_val)
                    normalized.append(max(0, min(1, norm)))
                else:
                    normalized.append(0.5)
            else:
                normalized.append(0.5)
        return normalized

    def denormalize_positions(self, normalized: List[float]) -> List[int]:
        'Convert 0-1 normalized values back to original positions'
        positions = []
        for i, norm in enumerate(normalized):
            motor_id = i + 1
            if motor_id in JOINT_LIMITS:
                min_val, max_val = JOINT_LIMITS[motor_id]
                pos = min_val + norm * (max_val - min_val)
                positions.append(int(round(pos)))
            else:
                positions.append(0)
        return positions

    def update_joint_states(self) -> None:
        'Update all joint status'
        positions = self.get_positions()
        if positions:
            for i, pos in enumerate(positions):
                joint_id = i + 1
                if joint_id in self.joints:
                    self.joints[joint_id].current_pos = pos
                    self.joints[joint_id].target_pos = pos

        currents = self.protocol.get_joint_currents()
        if currents:
            for i, current in enumerate(currents):
                joint_id = i + 1
                if joint_id in self.joints:
                    self.joints[joint_id].current = current

        temperatures = self.protocol.get_joint_temperatures()
        if temperatures:
            for i, temp in enumerate(temperatures):
                joint_id = i + 1
                if joint_id in self.joints:
                    self.joints[joint_id].temperature = temp

        error_codes = self.protocol.get_joint_error_codes()
        if error_codes:
            for i, code in enumerate(error_codes):
                joint_id = i + 1
                if joint_id in self.joints:
                    self.joints[joint_id].error_code = code

    def get_joint_state(self, joint_id: int) -> Optional[Dict]:
        'Get single joint status'
        if joint_id in self.joints:
            return {
                'id': self.joints[joint_id].id,
                'name': self.joints[joint_id].name,
                'finger': self.joints[joint_id].finger,
                'current_pos': self.joints[joint_id].current_pos,
                'target_pos': self.joints[joint_id].target_pos,
                'current': self.joints[joint_id].current,
                'temperature': self.joints[joint_id].temperature,
                'error_code': self.joints[joint_id].error_code,
                'enabled': self.joints[joint_id].enabled,
                'range': self.joints[joint_id].range
            }
        return None

    def __enter__(self) -> 'L30DexterousHandController':
        'Context Manager Entry'
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        'Context Manager Exit'
        self.disconnect()


# =============================================================================
#Convenience function
# =============================================================================

def create_default_controller(device_id: int = 0x01) -> L30DexterousHandController:
    'Create default controller'
    return L30DexterousHandController(device_id)


def setup_logging(level: int = logging.INFO) -> None:
    'Configuration Log'
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

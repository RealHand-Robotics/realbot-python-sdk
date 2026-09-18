#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'L30 dexterous hand CANFD extended frame communication control program (new version of protocol v1.0.6)\n\nImplemented according to the document "L30 Smart Hand CANFD Extended Frame Communication Protocol.md".\n\nKey points of agreement:\n- Physical interface: CANFD, extended frame, arbitration segment 1Mbps (80%), data segment 5Mbps (75%)\n- Multi-byte data uniformly uses Big-Endian byte order\n- CANFDID (29 bits) structure:\n    BIT28:26 Priority (3) | BIT25 Access type (1, 0=read/1=write) | BIT24:21 Parent command (4)\n    | BIT20:13 Subcommand(8) | BIT12:8 DstID(5) | BIT7:3 SrcID(5) | BIT2:0 Reserved(3)\n    CAN_ID = (Pri<<26)|(Access<<25)|(Parent<<21)|(Sub<<13)|(Dst<<8)|(Src<<3)\n- Frame data segment:\n    BYTE0 data length (requests start from BYTE2, responses start from BYTE3, excluding status code)\n    BYTE1 transaction control = (N<<4)|seq, fixed 0x00 for single frame, actual number of frames for multiple frames K=N+1\n    Request BYTE2~: write data\n    Response BYTE2: status code; BYTE3~: read data\n\nThe underlying CANFD communication method (initialization/sending and receiving) of this file refers to the old version of realhand_l30_v6_canfd.py.\n\nVersion: 1.0\nDate: 2026-07-22'

import time
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum
from ctypes import (
    Structure, CDLL, cdll, cast, byref, RTLD_GLOBAL,
    c_uint, c_ushort, c_char, c_ubyte, c_uint16, POINTER,
)

import numpy as np

from .realhand_l30_canfd_comm import (
    LibCanBusTransport,
    SocketCANTransport,
    create_transport,
)

logger = logging.getLogger(__name__)


# =============================================================================
#CANFD underlying library constants and structures (consistent with the old version)
# =============================================================================

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
        ("Cantype", c_char),
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
        ("Data", c_ubyte * 64),
    ]


#DLC encoding -> number of line bytes (CAN FD standard table, see protocol §2.2)
DLC_TO_LENGTH = {
    0x00: 0, 0x01: 1, 0x02: 2, 0x03: 3, 0x04: 4, 0x05: 5, 0x06: 6, 0x07: 7,
    0x08: 8, 0x09: 12, 0x0A: 16, 0x0B: 20, 0x0C: 24, 0x0D: 32, 0x0E: 48, 0x0F: 64,
}
#A few devices may backfill non-standard DLC values
NON_STANDARD_DLC_MAP = {0x10: 16, 0x40: 64}


def get_dlc_from_length(length: int) -> int:
    "Based on the actual line byte count, round up to the nearest legal DLC encoding.\n\n    CAN FD's DLC can only take standard values (0~8,12,16,20,24,32,48,64). When the number of valid bytes\n    When it is between two standard values, the larger one is used, and the insufficient bytes are fixed with 0x00 by the device.\n\n    Args:\n        length: The total number of line bytes to be transmitted (including BYTE0 data length + BYTE1 transaction control + data)\n    Returns:\n        Corresponding DLC encoding value (0x00~0x0F)"
    if length <= 8:
        return length          #0~8 bytes: DLC encoding corresponds to the number of bytes
    if length <= 12:
        return 0x09            #9~12 bytes -> 12 bytes frame
    if length <= 16:
        return 0x0A            #13~16 bytes -> 16 bytes frame
    if length <= 20:
        return 0x0B            #17~20 bytes -> 20 bytes frame
    if length <= 24:
        return 0x0C            #21~24 bytes -> 24 bytes frame
    if length <= 32:
        return 0x0D            #25~32 bytes -> 32 bytes frame
    if length <= 48:
        return 0x0E            #33~48 bytes -> 48 bytes frame
    return 0x0F               #49~64 bytes -> 64 bytes frame


def get_length_from_dlc(dlc: int) -> int:
    'Based on the received DLC code, the number of line bytes is checked back.\n\n    Check the standard table first; if the device backfills non-standard values (such as 0x10/0x40), use the compatibility table;\n    Then truncate the value to 64.\n\n    Args:\n        dlc: DLC field value in received frame\n    Returns:\n        The number of line bytes in the frame'
    if dlc in DLC_TO_LENGTH:
        return DLC_TO_LENGTH[dlc]
    if dlc in NON_STANDARD_DLC_MAP:
        return NON_STANDARD_DLC_MAP[dlc]
    return min(dlc, 64)


# =============================================================================
#Protocol constant definition (new version)
# =============================================================================

class Priority(IntEnum):
    'CANFDID Arbitration priority, the smaller the value, the higher the priority'
    HIGHEST = 0
    LOWEST = 7


class Access(IntEnum):
    'Access type BIT25'
    READ = 0
    WRITE = 1


class ParentCmd(IntEnum):
    'Parent command BIT24:21'
    MULTI_JOINT = 0x1     #Multi-joint control
    TOUCH = 0x2           #Tactile sensor
    CONFIG = 0x3          #Configuration information
    PERIODIC = 0x4        #Periodic reporting
    QUERY = 0x5           #Single query
    SINGLE_JOINT = 0x6    #Single joint debugging control
    BOOTLOADER = 0xF      #Firmware upgrade


class MultiJointSub(IntEnum):
    'Parent command 0x1 Subcommand'
    POSITION = 0x01
    TORQUE = 0x02
    SPEED = 0x03
    STOP = 0x05           #Global emergency stop (not supported yet)
    PAUSE = 0x06          #Global pause (not supported yet)
    ENABLE = 0x07         #Global enable
    DISABLE = 0x08        #Global Disablement


class TouchSub(IntEnum):
    'Parent command 0x2 Child command (single finger touch)'
    THUMB = 0x01
    INDEX = 0x02
    MIDDLE = 0x03
    RING = 0x04
    PINKY = 0x05


class ConfigSub(IntEnum):
    'Parent command 0x3 Subcommand'
    UNLOCK = 0x01
    DEVICE_INFO = 0x02
    PRODUCT_CODE = 0x03
    NODE_ID = 0x04
    CALIBRATE_ZERO = 0x05
    HAND_TYPE = 0x06


class QuerySub(IntEnum):
    'Parent command 0x5 Subcommand (single query)'
    POSITION = 0x01
    CURRENT = 0x02
    SPEED = 0x03
    TEMPERATURE = 0x04
    ERROR_CODE = 0x05


class PeriodicSub(IntEnum):
    'Parent command 0x4 Subcommand (periodic reporting)'
    POSITION = 0x01
    CURRENT = 0x02
    SPEED = 0x03
    TEMPERATURE = 0x04
    ERROR_CODE = 0x05


class SingleJointFunc(IntEnum):
    'Parent command 0x6 function code, subcommand = (function code<<5)|Joint number'
    POSITION = 0x1
    TORQUE = 0x2
    SPEED = 0x3


class HandType(IntEnum):
    'Right and left handed type'
    LEFT = 0x00
    RIGHT = 0x01


class StatusCode(IntEnum):
    'Universal status code (parent command 0x1~0x6, see protocol §8.2)'
    OK = 0x00
    ERR_DLC = 0x10
    ERR_PARAM = 0x11
    ERR_CMD = 0x12
    ERR_SUBCMD = 0x13
    ERR_FORMAT = 0x14
    ERR_PERMISSION = 0x20      #Insufficient permissions (not unlocked/wrong password)
    ERR_NOT_CALIBRATED = 0x21
    ERR_MOTOR_DISABLED = 0x22
    ERR_STATE = 0x23
    ERR_NOT_SET = 0x24
    ERR_READ = 0x30
    ERR_WRITE = 0x31
    ERR_MULTIFRAME_TIMEOUT = 0x32
    ERR_MULTIFRAME_INCOMPLETE = 0x33
    ERR_PERIOD_CFG = 0x34
    ERR_PERIOD_RANGE = 0x35
    ERR_MASK = 0x36
    ERR_COMM_TIMEOUT = 0x40
    ERR_COMM_LOST = 0x41
    ERR_BUSY = 0xF0


class Command(IntEnum):
    '[Compatible with old modules] Command code enumeration of the old version of realhand_l30_v6_canfd.py.\n\n    The new version of the protocol actually uses ParentCmd + each subcommand (MultiJointSub/QuerySub, etc.) to organize commands.\n    This enumeration is reserved only to maintain the same external import interface as the old module, so realhand_l30_api.py\n    The `from ...core.xxx import Command` can be imported successfully between both modules.'
    JOINT_POSITION = 0x01
    JOINT_TORQUE = 0x02
    JOINT_TORQUE_LIMIT = 0x03
    JOINT_SPEED = 0x05
    JOINT_ACCELERATION = 0x07
    JOINT_ENABLE = 0x08
    JOINT_TEMPERATURE = 0x33
    JOINT_ERROR_CODE = 0x35
    JOINT_CURRENT = 0x36
    CALIBRATE_ZERO = 0x37
    THUMB_PRESSURE = 0xB1
    INDEX_PRESSURE = 0xB2
    MIDDLE_PRESSURE = 0xB3
    RING_PRESSURE = 0xB4
    PINKY_PRESSURE = 0xB5
    DEVICE_CODE = 0xC0
    DEVICE_VERSION = 0xC1
    DEVICE_CANFDID = 0xC3
    RESTORE_DEVICE_ID = 0xC4


#Configure unlocking default password (see Agreement §12.1)
UNLOCK_PASSWORD = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC])

#Number of joints
JOINT_COUNT = 17


# =============================================================================
#Joint definition (left-hand default range, see protocol §9.1; value is int16 big-endian semantic position quantity)
# =============================================================================

@dataclass
class JointInfo:
    id: int
    name: str
    finger: str
    min_pos: int
    max_pos: int

    @property
    def range(self) -> Tuple[int, int]:
        'Returns a (minimum position, maximum position) tuple for this joint.'
        return (self.min_pos, self.max_pos)


JOINT_DEFINITIONS = [
    JointInfo(1,  'The base of the thumb is bent', 'Thumb',   0,  900),
    JointInfo(2,  'Thumb tip bent', 'Thumb',   0, 1200),
    JointInfo(3,  'Thumb side swing',    'Thumb',   0,  900),
    JointInfo(4,  'Thumb rotation',    'Thumb',   0,  800),
    JointInfo(5,  'Ring finger side swing',   'Ring finger', -200,  200),
    JointInfo(6,  'The tip of the ring finger is bent', 'Ring finger',  0, 1200),
    JointInfo(7,  'The base of the ring finger is bent', 'Ring finger',  0, 1200),
    JointInfo(8,  'The base of the middle finger is bent', 'middle finger',   0, 1200),
    JointInfo(9,  'The tip of the middle finger is bent', 'middle finger',   0, 1200),
    JointInfo(10, 'The base of the little finger is bent',   'pinky',   0, 1500),
    JointInfo(11, 'Bent tip of little finger', 'pinky',   0, 1200),
    JointInfo(12, 'Little finger side swing',    'pinky', -200,  200),
    JointInfo(13, 'Middle finger side swing',    'middle finger', -200,  200),
    JointInfo(14, 'Index finger side swing',    'index finger', -200,  200),
    JointInfo(15, 'The base of the index finger is bent', 'index finger',   0, 1200),
    JointInfo(16, 'The tip of the index finger is bent', 'index finger',   0, 1200),
    JointInfo(17, 'wrist',       'wrist', -900,  900),
]

#Joint limit dictionary {Joint number (1~17): (min, max)}
JOINT_LIMITS = {j.id: (j.min_pos, j.max_pos) for j in JOINT_DEFINITIONS}

#Joint name (English/Chinese, in the order of J1~J17)
JOINT_NAME_EN = [
    "thumb_cmc_pitch", "thumb_ip_pitch", "thumb_cmc_yaw", "thumb_cmc_roll",
    "ring_mcp_roll", "ring_pip_pitch", "ring_mcp_pitch",
    "middle_mcp_pitch", "middle_pip_pitch",
    "pinky_mcp_pitch", "pinky_pip_pitch", "pinky_mcp_roll",
    "middle_mcp_roll", "index_mcp_roll",
    "index_mcp_pitch", "index_pip_pitch",
    "wrist_pitch",
]
JOINT_NAME_CN = [j.name for j in JOINT_DEFINITIONS]

#Tactile sensor matrix specifications (Warwick 12x6)
TOUCH_ROWS, TOUCH_COLS = 12, 6
TOUCH_BYTES = TOUCH_ROWS * TOUCH_COLS  # 72


# =============================================================================
#CANFD communication protocol implementation
# =============================================================================

class L30CANFDProtocol:
    'L30 Smart Hand CANFD extended frame communication protocol (new version v1.0.6)'

    ARBITRATION_BAUD = 1000000   #Arbitration segment 1Mbps
    DATA_BAUD = 5000000          #Data segment 5Mbps
    DEFAULT_NODE_ID = 1          #Device default NodeID(DstID)
    MASTER_ID = 0                #Master SrcID

    def __init__(self, node_id: int = DEFAULT_NODE_ID, src_id: int = MASTER_ID,
                 canfd_device: int = 0, channel=0, comm_type: str = "libcanbus",
                 bitrate: int = 1000000, dbitrate: int = 5000000,
                 auto_setup: bool = True):
        'Construct the protocol object (it will not be connected immediately, and initialize needs to be called separately).\n\n        Args:\n            node_id: device node ID (DstID), range 1~31; used for addressing and response filtering when sending and receiving\n            src_id: Master site source ID (SrcID), master site default 0\n            canfd_device: CANFD device (box) index, only used by the libcanbus backend\n            channel: channel number - libcanbus is int (default 0), socketcan is the interface name str (such as "can0")\n            comm_type: communication backend - "libcanbus" (default, vendor private library) or\n                       "socketcan" (kernel can0 + python-can, transparent plastic USB-CANFD device)\n            bitrate/dbitrate/auto_setup: only used by socketcan - arbitration/data segment baud rate and\n                       Whether to automatically pull up the interface'
        self.node_id = node_id
        self.src_id = src_id
        self.canfd_device = canfd_device
        self.channel = channel
        self.comm_type = comm_type
        #Pluggable transport backend: libcanbus (default) or socketcan; actual sending and receiving is delegated to it
        self.transport = create_transport(
            comm_type=comm_type, canfd_device=canfd_device, channel=channel,
            bitrate=bitrate, dbitrate=dbitrate, auto_setup=auto_setup)
        self.is_connected = False     #Connection status flag (maintained by initialize/close)
        #5 refers to the tactile matrix cache (12x6), the initial value -1 indicates that valid data has not been read yet
        self._touch_matrices = {
            'thumb': np.full((TOUCH_ROWS, TOUCH_COLS), -1),
            'index': np.full((TOUCH_ROWS, TOUCH_COLS), -1),
            'middle': np.full((TOUCH_ROWS, TOUCH_COLS), -1),
            'ring': np.full((TOUCH_ROWS, TOUCH_COLS), -1),
            'little': np.full((TOUCH_ROWS, TOUCH_COLS), -1),
        }

    # =========================================================================
    #Underlying communication (initialization / shutdown / transceiver, delegated pluggable transmission backend)
    # =========================================================================

    def initialize(self) -> bool:
        'Initialize CANFD communication (delegated transport backend).\n\n        libcanbus: Load library -> Scan -> Open channel -> CANFD configuration -> Filter;\n        socketcan: automatically pull up the can0 interface -> open the python-can bus.\n\n        Returns:\n            Returns True if initialization is successful, otherwise False.'
        logger.info('Initializing CANFD communication...')
        ok = self.transport.initialize()
        self.is_connected = ok
        if ok:
            logger.info('CANFD communication initialization completed')
        return ok

    def close(self) -> None:
        'Closes the CANFD connection and resets the connection flag (delegated transport backend).'
        self.transport.close()
        self.is_connected = False
        logger.info('CANFD connection closed')

    # =========================================================================
    #CANFDID / Transaction Control Codec
    # =========================================================================

    def _build_canfd_id(self, access: int, parent_cmd: int, sub_cmd: int,
                        dst_id: Optional[int] = None, src_id: Optional[int] = None,
                        priority: int = Priority.HIGHEST) -> int:
        'Constructs the 29-bit extension frame CANFDID.\n\n        Bit field: CAN_ID = (Pri<<26)|(Access<<25)|(Parent<<21)|(Sub<<13)|(Dst<<8)|(Src<<3)\n\n        Args:\n            access: access type (0=read/1=write), accounting for BIT25\n            parent_cmd: parent command (0x1~0xF), accounting for BIT24:21\n            sub_cmd: subcommand (0x00~0xFF), accounting for BIT20:13\n            dst_id: target node ID, the default is the local node_id, accounting for BIT12:8\n            src_id: source node ID, the default is the local src_id, accounting for BIT7:3\n            priority: arbitration priority (0 is the highest), accounting for BIT28:26\n        Returns:\n            Assembled 29-bit CANFDID integer.'
        if dst_id is None:
            dst_id = self.node_id
        if src_id is None:
            src_id = self.src_id
        #is masked field by field and shifted to the corresponding position, and finally combined by bitwise OR
        frame_id = (priority & 0x7) << 26
        frame_id |= (access & 0x1) << 25
        frame_id |= (parent_cmd & 0xF) << 21
        frame_id |= (sub_cmd & 0xFF) << 13
        frame_id |= (dst_id & 0x1F) << 8
        frame_id |= (src_id & 0x1F) << 3
        return frame_id

    def _parse_canfd_id(self, frame_id: int) -> Dict:
        'Parse the 29-bit CANFDID and separate out each field.\n\n        Args:\n            frame_id: 29-bit ID of received frame\n        Returns:\n            Dictionary containing priority/access/parent_cmd/sub_cmd/dst_id/src_id.'
        return {
            'priority': (frame_id >> 26) & 0x7,
            'access': (frame_id >> 25) & 0x1,
            'parent_cmd': (frame_id >> 21) & 0xF,
            'sub_cmd': (frame_id >> 13) & 0xFF,
            'dst_id': (frame_id >> 8) & 0x1F,
            'src_id': (frame_id >> 3) & 0x1F,
        }

    @staticmethod
    def _build_transaction(total_field: int = 0, seq: int = 0) -> int:
        'Build transaction control byte BYTE[1] = (N<<4)|seq.\n\n        Args:\n            total_field: high 4-digit total frame number field N (fill in 0 for single frame, actual number of frames for multiple frames K=N+1)\n            seq: low 4-bit frame sequence number (single frame 0, multi-frame 0~N)\n        Returns:\n            Merged transaction control byte; 0x00 for single frame.'
        return ((total_field & 0xF) << 4) | (seq & 0xF)

    @staticmethod
    def _parse_transaction(control: int) -> Dict:
        'Parse transaction control bytes.\n\n        Args:\n            control: BYTE[1] transaction control value\n        Returns:\n            A dictionary containing total_field(N), seq(serial number), and total_frames(actual number of frames K=N+1).'
        n = (control >> 4) & 0xF
        return {'total_field': n, 'seq': control & 0xF, 'total_frames': n + 1}

    # =========================================================================
    #Frame sending and receiving
    # =========================================================================

    def send_frame(self, access: int, parent_cmd: int, sub_cmd: int,
                   data: bytes = b'', total_field: int = 0, seq: int = 0,
                   priority: int = Priority.HIGHEST) -> bool:
        'Send a single frame CANFD message (delegated transport backend).\n\n        Frame data segment layout: BYTE0=effective data length, BYTE1=transaction control, BYTE2~=data;\n        The transport backend is responsible for scaling the DLC and padding zeros to the legal frame length.\n\n        Args:\n            access: Access.READ / Access.WRITE\n            parent_cmd: parent command\n            sub_cmd: subcommand\n            data: write data (starting from BYTE2); read request is empty\n            total_field: Transaction control total frame number field N (fill in 0 for a single frame)\n            seq: frame sequence number (fill in 0 for a single frame)\n            priority: arbitration priority (highest by default)\n        Returns:\n            Returns True if sent successfully, otherwise False.'
        if not self.is_connected:
            logger.error('Error: CANFD not connected')
            return False
        try:
            #Assemble 29-bit ID (target/source uses native defaults)
            frame_id = self._build_canfd_id(access, parent_cmd, sub_cmd, priority=priority)
            data_len = min(len(data), 62)  #Data segment maximum 62 bytes (BYTE2~63)

            #Valid bytes of data segment: BYTE0 length / BYTE1 transaction control / BYTE2~ data
            payload = bytearray(2 + data_len)
            payload[0] = data_len
            payload[1] = self._build_transaction(total_field, seq)
            payload[2:2 + data_len] = bytes(data[:data_len])

            return self.transport.send(frame_id, bytes(payload))
        except Exception as e:
            logger.error(f"Exception sending message: {e}")
            return False

    def receive_messages(self, timeout_ms: int = 3, filter_node: bool = True,
                         expected_parent: Optional[int] = None,
                         expected_sub: Optional[int] = None
                         ) -> List[Tuple[int, bytes, Dict]]:
        'Receive CANFD messages (parse after entrusting the transmission backend to retrieve the original frame), and filter by source/parent command/child command.\n\n        Args:\n            timeout_ms: underlying receive blocking timeout (milliseconds)\n            filter_node: When True, only the frames with SrcID==local node_id are retained (that is, the target device responds)\n            expected_parent: If specified, only the frames of this parent command will be retained.\n            expected_sub: If specified, only the frames of this subcommand will be retained.\n        Returns:\n            (frame_id, data, parsed_info) list; parsed_info contains ID bit field and transaction control parsing.'
        if not self.is_connected:
            return []
        try:
            messages = []
            for frame_id, data in self.transport.receive(timeout_ms):
                parsed = self._parse_canfd_id(frame_id)
                parsed['transaction'] = (
                    self._parse_transaction(data[1]) if len(data) > 1 else {}
                )

                #The SrcID of the response frame should be the device NodeID, thereby discarding packets from non-target devices
                if filter_node and parsed['src_id'] != self.node_id:
                    continue
                if expected_parent is not None and parsed['parent_cmd'] != expected_parent:
                    continue
                if expected_sub is not None and parsed['sub_cmd'] != expected_sub:
                    continue

                messages.append((frame_id, data, parsed))
            return messages
        except Exception as e:
            logger.error(f"Exception receiving message: {e}")
            return []

    def _wait_for_response(self, parent_cmd: int, sub_cmd: Optional[int] = None,
                           timeout_ms: int = 200) -> Optional[Tuple[bytes, int, Dict]]:
        'Poll within the timeout period and return the matching first frame response.\n\n        Args:\n            parent_cmd: the parent command to which the response is expected\n            sub_cmd: subcommand to expect a response (optional)\n            timeout_ms: total wait timeout (milliseconds)\n        Returns:\n            (data, status_code, parsed) triple; status_code gets the response BYTE2; returns None when timeout.'
        start = time.time()
        while (time.time() - start) < timeout_ms / 1000.0:
            for _, data, parsed in self.receive_messages(
                timeout_ms=20, expected_parent=parent_cmd, expected_sub=sub_cmd
            ):
                status = data[2] if len(data) > 2 else -1  #response BYTE2 is the status code
                return data, status, parsed
            time.sleep(0.002)
        return None

    def _read_response_joints(self, parent_cmd: int, sub_cmd: int, count: int,
                              signed: bool, byte_per_joint: int = 2,
                              timeout_ms: int = 200) -> Optional[List[int]]:
        'Send a read request and parse the N joint data in the response (from BYTE3 onwards, skip the status code).\n\n        Args:\n            parent_cmd/sub_cmd: target read command\n            count: number of joints (L30 is 17)\n            signed: whether the data is signed\n            byte_per_joint: number of bytes per joint (position/current/speed=2, temperature/error code=1)\n            timeout_ms: response waiting timeout\n        Returns:\n            List of joint values; returns None on failure or status code other than 0.'
        if not self.send_frame(Access.READ, parent_cmd, sub_cmd):
            return None
        resp = self._wait_for_response(parent_cmd, sub_cmd, timeout_ms)
        if not resp:
            return None
        data, status, _ = resp
        if status != StatusCode.OK:
            logger.warning(f"Failed to read parent=0x{parent_cmd:X} sub=0x{sub_cmd:X} status code=0x{status:02X}")
            return None
        return self._unpack_joints(data, 3, count, byte_per_joint, signed)  #BYTE3 onwards for reading data

    # =========================================================================
    #Data Packing/Unpacking
    # =========================================================================

    @staticmethod
    def _pack_joints(values: List[int], signed: bool, byte_per_joint: int = 2,
                     limits: Optional[Dict[int, Tuple[int, int]]] = None,
                     hard_min: Optional[int] = None, hard_max: Optional[int] = None) -> bytes:
        'Pack the 17-way joint values into big-endian bytes in the order of J1~J17, and trim the range as needed.\n\n        Args:\n            values: joint value list (press J1~J17)\n            signed: whether there is symbol encoding\n            byte_per_joint: Number of bytes per joint\n            limits: joint-by-joint limit dictionary {joint number: (min,max)} (such as position control)\n            hard_min/hard_max: unified upper and lower limits (such as torque 60~800, speed 1~250)\n        Returns:\n            Packed big-endian byte string.'
        out = bytearray()
        for i, val in enumerate(values):
            v = int(val)
            if limits is not None:                       #Joint-by-joint limit
                lo, hi = limits.get(i + 1, (-32768, 32767))
                v = max(lo, min(hi, v))
            if hard_min is not None:                     #Uniform lower limit crop
                v = max(hard_min, v)
            if hard_max is not None:                     #Uniform upper limit crop
                v = min(hard_max, v)
            out.extend(v.to_bytes(byte_per_joint, byteorder='big', signed=signed))
        return bytes(out)

    @staticmethod
    def _unpack_joints(data: bytes, offset: int, count: int,
                       byte_per_joint: int = 2, signed: bool = True) -> List[int]:
        'Unpack count big-endian joint values starting from data[offset].\n\n        When the actual data is insufficient, the quantity is automatically shrunk to avoid crossing the boundary.\n\n        Args:\n            data: raw response bytes\n            offset: data starting offset (single query response is 3)\n            count: expected number of joints\n            byte_per_joint: Number of bytes per joint\n            signed: whether there is symbol resolution\n        Returns:\n            The parsed list of joint values.'
        values = []
        need = offset + count * byte_per_joint
        if len(data) < need:  #Shrink according to the actual number that can be parsed when there is insufficient data
            count = max(0, (len(data) - offset) // byte_per_joint)
        for i in range(count):
            s = offset + i * byte_per_joint
            values.append(int.from_bytes(data[s:s + byte_per_joint], byteorder='big', signed=signed))
        return values

    # =========================================================================
    #Parent command 0x1: Multi-joint control
    # =========================================================================

    def set_joint_positions(self, positions: List[int]) -> bool:
        'Multi-joint position control (subcommand 0x01, no response, int16 big end).\n\n        17 channels of target positions are delivered, and out-of-range values \u200b\u200bare cut joint by joint according to JOINT_LIMITS (the device will also be limited according to the hand shape).\n\n        Args:\n            positions: The length must be 17, in the order of J1~J17\n        Returns:\n            True is returned if the transmission is successful; False is returned if the length does not match or the transmission fails.'
        if len(positions) != JOINT_COUNT:
            logger.error(f"Wrong position data length: expected {JOINT_COUNT}, actual {len(positions)}")
            return False
        data = self._pack_joints(positions, signed=True, limits=JOINT_LIMITS)
        return self.send_frame(Access.WRITE, ParentCmd.MULTI_JOINT, MultiJointSub.POSITION, data)

    def set_joint_torques(self, torques: List[int]) -> bool:
        'Multi-joint torque control (subcommand 0x02, no response).\n\n        The unit is 6.5ma, the effective range is 60~800, and it will be uniformly cropped if it exceeds the range.\n\n        Args:\n            torques: The length must be 17, in the order of J1~J17\n        Returns:\n            True is returned if the transmission is successful; False is returned if the length does not match or the transmission fails.'
        if len(torques) != JOINT_COUNT:
            logger.error(f"Torque data length error: expected {JOINT_COUNT}, actual {len(torques)}")
            return False
        data = self._pack_joints(torques, signed=False, hard_min=60, hard_max=800)
        return self.send_frame(Access.WRITE, ParentCmd.MULTI_JOINT, MultiJointSub.TORQUE, data)

    def set_joint_velocities(self, velocities: List[int]) -> bool:
        'Multi-joint speed control (subcommand 0x03, no response).\n\n        The unit is 0.732rpm, the effective range is 1~250, and it will be uniformly cut if it exceeds the range.\n\n        Args:\n            velocities: The length must be 17, in the order of J1~J17\n        Returns:\n            True is returned if the transmission is successful; False is returned if the length does not match or the transmission fails.'
        if len(velocities) != JOINT_COUNT:
            logger.error(f"Velocity data length error: expected {JOINT_COUNT}, actual {len(velocities)}")
            return False
        data = self._pack_joints(velocities, signed=False, hard_min=1, hard_max=250)
        return self.send_frame(Access.WRITE, ParentCmd.MULTI_JOINT, MultiJointSub.SPEED, data)

    def _write_with_ack(self, parent_cmd: int, sub_cmd: int, data: bytes = b'',
                        timeout_ms: int = 200) -> Tuple[bool, int]:
        'Send a write command and wait for the response status code (applicable to write operations with responses).\n\n        Args:\n            parent_cmd/sub_cmd: target command\n            data: write data (can be empty)\n            timeout_ms: response waiting timeout\n        Returns:\n            (success, status code); returns (False, -1) when no response is received.'
        if not self.send_frame(Access.WRITE, parent_cmd, sub_cmd, data):
            return False, -1
        resp = self._wait_for_response(parent_cmd, sub_cmd, timeout_ms)
        if not resp:
            return False, -1
        _, status, _ = resp
        return status == StatusCode.OK, status

    def enable_all_joints(self) -> bool:
        'Globally enable all joints (subcommand 0x07, with response). Return True on success.'
        ok, _ = self._write_with_ack(ParentCmd.MULTI_JOINT, MultiJointSub.ENABLE)
        return ok

    def disable_all_joints(self) -> bool:
        'Globally disables all joints (subcommand 0x08, with response). Return True on success.'
        ok, _ = self._write_with_ack(ParentCmd.MULTI_JOINT, MultiJointSub.DISABLE)
        return ok

    def emergency_stop(self) -> bool:
        'Global emergency stop (subcommand 0x05, firmware does not support it yet, there is response). Return True on success.'
        ok, _ = self._write_with_ack(ParentCmd.MULTI_JOINT, MultiJointSub.STOP)
        return ok

    def pause(self) -> bool:
        'Globally pause and maintain the current position (subcommand 0x06, not supported by firmware yet, with response). Return True on success.'
        ok, _ = self._write_with_ack(ParentCmd.MULTI_JOINT, MultiJointSub.PAUSE)
        return ok

    # =========================================================================
    #Parent command 0x2: Tactile sensor (multi-frame read, 12x6=72 bytes)
    # =========================================================================

    def get_finger_touch(self, sub_cmd: int, timeout_ms: int = 20) -> Optional[List[int]]:
        'Read the single-finger tactile matrix (multi-frame read, return 72-byte list).\n\n        The protocol returns 12x6=72 bytes in multi-frame reading mode: 61 bytes from the first frame (seq=0) BYTE3,\n        Frame 2 (seq=1) continues with 11 bytes; BYTE2 of each frame is the status code, and BYTE0 is the effective length of this frame.\n        This method splices the data segments starting from BYTE3 of each frame in seq order.\n\n        Args:\n            sub_cmd: tactile subcommand (TouchSub.THUMB~PINKY)\n            timeout_ms: multi-frame packetization timeout (the protocol recommends 10ms level)\n        Returns:\n            72-element integer list; returns None on timeout/status exception/incomplete data.'
        if not self.send_frame(Access.READ, ParentCmd.TOUCH, sub_cmd):
            return None

        frames: Dict[int, bytes] = {}          #seq -> data segment, used for sequential splicing
        expected_frames: Optional[int] = None  #Total number of frames parsed by transaction control K
        start = time.time()
        while (time.time() - start) < timeout_ms / 1000.0:
            for _, data, parsed in self.receive_messages(
                timeout_ms=5, expected_parent=ParentCmd.TOUCH, expected_sub=sub_cmd
            ):
                if len(data) < 3:
                    continue
                if data[2] != StatusCode.OK:               #BYTE2 status code
                    logger.warning(f"Tactile reading status abnormality: 0x{data[2]:02X}")
                    return None
                tc = parsed.get('transaction', {})
                seq = tc.get('seq', 0)
                expected_frames = tc.get('total_frames', expected_frames)
                seg_len = data[0]                          #BYTE0: Effective data length of this frame (excluding status code)
                frames[seq] = data[3:3 + seg_len]          #BYTE3 starting from matrix data
                if expected_frames is not None and len(frames) >= expected_frames:
                    break
            if expected_frames is not None and len(frames) >= expected_frames:
                break
            time.sleep(0.001)

        if not frames:
            return None
        #Splice seq in ascending order to get the complete matrix
        payload = bytearray()
        for seq in sorted(frames.keys()):
            payload.extend(frames[seq])
        if len(payload) < TOUCH_BYTES:
            logger.warning(f"Incomplete tactile data: {len(payload)}/{TOUCH_BYTES}")
            return None
        return list(payload[:TOUCH_BYTES])

    def _process_touch_matrix(self, data: Optional[List[int]], key: str) -> np.ndarray:
        'Shape 72 bytes of haptic data into a 12x6 matrix and cache it.\n\n        Flip up and down (arr[::-1]) to match the display direction; return the last cached value when the data is invalid.\n\n        Args:\n            data: 72-element list returned by get_finger_touch (or None)\n            key: finger key name (thumb/index/middle/ring/little)\n        Returns:\n            12x6 numpy matrix.'
        if data is not None and len(data) == TOUCH_BYTES:
            arr = np.array(data).reshape(TOUCH_ROWS, TOUCH_COLS)
            self._touch_matrices[key] = arr[::-1]
        return self._touch_matrices[key]

    def get_thumb_touch(self) -> np.ndarray:
        'Read the thumb tactile matrix (subcommand 0x01), returning a 12x6 numpy matrix.'
        return self._process_touch_matrix(self.get_finger_touch(TouchSub.THUMB), 'thumb')

    def get_index_touch(self) -> np.ndarray:
        'Read the index finger tactile matrix (subcommand 0x02), returning a 12x6 numpy matrix.'
        return self._process_touch_matrix(self.get_finger_touch(TouchSub.INDEX), 'index')

    def get_middle_touch(self) -> np.ndarray:
        'Read the middle finger tactile matrix (subcommand 0x03), returning a 12x6 numpy matrix.'
        return self._process_touch_matrix(self.get_finger_touch(TouchSub.MIDDLE), 'middle')

    def get_ring_touch(self) -> np.ndarray:
        'Read the ring finger tactile matrix (subcommand 0x04), returning a 12x6 numpy matrix.'
        return self._process_touch_matrix(self.get_finger_touch(TouchSub.RING), 'ring')

    def get_little_touch(self) -> np.ndarray:
        'Read the little finger tactile matrix (subcommand 0x05), returning a 12x6 numpy matrix.'
        return self._process_touch_matrix(self.get_finger_touch(TouchSub.PINKY), 'little')

    def get_all_touch(self) -> Dict[str, list]:
        'reads all 5 finger tactile matrices in sequence and returns the {finger_matrix: 12x6 list} dictionary.'
        return {
            'thumb_matrix': self.get_thumb_touch().tolist(),
            'index_matrix': self.get_index_touch().tolist(),
            'middle_matrix': self.get_middle_touch().tolist(),
            'ring_matrix': self.get_ring_touch().tolist(),
            'little_matrix': self.get_little_touch().tolist(),
        }

    # =========================================================================
    #Parent command 0x3: Configuration information
    # =========================================================================

    def unlock(self, password: bytes = UNLOCK_PASSWORD) -> bool:
        'Configuration unlock (subcommand 0x01).\n\n        It must be unlocked before writing DeviceInfo/NodeID/hand shape/calibrating zero point and other operations.\n        If the consecutive incorrect passwords reach the upper limit (10 times), the device will be locked and the device needs to be restarted.\n\n        Args:\n            password: 6-byte unlock password, the default password is the protocol agreed password\n        Returns:\n            True is returned if the unlock is successful; False is returned if the password is incorrect/insufficient permissions (0x20).'
        ok, status = self._write_with_ack(ParentCmd.CONFIG, ConfigSub.UNLOCK, bytes(password))
        if not ok:
            logger.error(f"Configuration unlock failed, status code=0x{status:02X}")
        return ok

    def get_device_info(self, timeout_ms: int = 200) -> Optional[Dict]:
        'Read DeviceInfo (subcommand 0x02).\n\n        The response is 18 bytes structure starting from BYTE3: product_id / serial_no(4B) / sw(3B) /\n        hw(3B)/struct(3B)/node_id/hand_type/sensor_type/origin.\n\n        Args:\n            timeout_ms: response waiting timeout\n        Returns:\n            The parsed device information dictionary; returns None on failure.'
        if not self.send_frame(Access.READ, ParentCmd.CONFIG, ConfigSub.DEVICE_INFO):
            return None
        resp = self._wait_for_response(ParentCmd.CONFIG, ConfigSub.DEVICE_INFO, timeout_ms)
        if not resp:
            return None
        data, status, _ = resp
        if status != StatusCode.OK or len(data) < 21:
            return None
        info = data[3:21]  #BYTE3~20 Total 18 bytes device information
        return {
            'product_id': info[0],                                    # 0x13=L30
            'serial_no': int.from_bytes(info[1:5], 'big'),            #Global serial number
            'sw_version': f"{info[5]}.{info[6]}.{info[7]}",           #Software version
            'hw_version': f"{info[8]}.{info[9]}.{info[10]}",          #Hardware version
            'struct_version': f"{info[11]}.{info[12]}.{info[13]}",    #Structure/Protocol Version
            'node_id': info[14],                                      #Current NodeID
            'hand_type': 'right' if info[15] == HandType.RIGHT else 'left',
            'sensor_type': info[16],                                  #Sensor Type
            'origin': info[17],                                       #Origin
        }

    def set_device_info(self, serial_no: int, origin: int) -> bool:
        'Write DeviceInfo (subcommand 0x02, need to be unlocked first).\n\n        Only serial_no and origin are writable, the remaining fields are maintained by the device.\n\n        Args:\n            serial_no: global serial number (uint32)\n            origin: origin code (1=self-assembled in Beijing, 2=big factory, 3=Gu’an)\n        Returns:\n            Returns True if the writing is successful; returns False if it is not unlocked (0x20) and other failures.'
        data = serial_no.to_bytes(4, 'big') + bytes([origin & 0xFF])  #serial_no big endian + origin
        ok, status = self._write_with_ack(ParentCmd.CONFIG, ConfigSub.DEVICE_INFO, data)
        if not ok:
            logger.error(f"Failed to write DeviceInfo, status code=0x{status:02X}")
        return ok

    def get_product_code(self, timeout_ms: int = 200) -> Optional[str]:
        "Read the product encoding string (subcommand 0x03, ASCII, no need to unlock).\n\n        Response BYTE0 is the string length, BYTE2 is the status code, and BYTE3 and above are ASCII encoding.\n        (eg 'LHT30-06-169-L-B-1-A').\n\n        Args:\n            timeout_ms: response waiting timeout\n        Returns:\n            Product encoding string; returns None on failure."
        if not self.send_frame(Access.READ, ParentCmd.CONFIG, ConfigSub.PRODUCT_CODE):
            return None
        resp = self._wait_for_response(ParentCmd.CONFIG, ConfigSub.PRODUCT_CODE, timeout_ms)
        if not resp:
            return None
        data, status, _ = resp
        if status != StatusCode.OK:
            return None
        length = data[0]  #BYTE0: Effective length of string
        try:
            return data[3:3 + length].decode('ascii', errors='ignore')
        except Exception:
            return None

    def get_node_id(self, timeout_ms: int = 200) -> Optional[int]:
        'Read the current NodeID (subcommand 0x04, no need to unlock).\n\n        Response BYTE3 is the current NodeID.\n\n        Args:\n            timeout_ms: response waiting timeout\n        Returns:\n            Current NodeID(1~31); Returns None on failure.'
        if not self.send_frame(Access.READ, ParentCmd.CONFIG, ConfigSub.NODE_ID):
            return None
        resp = self._wait_for_response(ParentCmd.CONFIG, ConfigSub.NODE_ID, timeout_ms)
        if not resp:
            return None
        data, status, _ = resp
        if status != StatusCode.OK or len(data) < 4:
            return None
        return data[3]

    def set_node_id(self, new_node_id: int) -> bool:
        'Write NodeID (subcommand 0x04, needs to be unlocked first).\n\n        The modification takes effect immediately, and the device uses the new NodeID to respond to subsequent frames; the local node_id is updated synchronously\n        in order to continue addressing and filtering responses correctly.\n\n        Args:\n            new_node_id: new node ID, range 1~31\n        Returns:\n            True is returned if writing is successful; False is returned if failure such as out-of-bounds/unlocked failure occurs.'
        if not (1 <= new_node_id <= 31):
            logger.error('NodeID is out of bounds, should be 1~31')
            return False
        ok, status = self._write_with_ack(ParentCmd.CONFIG, ConfigSub.NODE_ID,
                                          bytes([new_node_id]))
        if ok:
            self.node_id = new_node_id  #The device has responded to subsequent frames with the new NodeID
        else:
            logger.error(f"Failed to write NodeID, status code=0x{status:02X}")
        return ok

    def calibrate_zero(self) -> bool:
        'Calibrate the zero points of all joints (subcommand 0x05).\n\n        Preconditions: It must be unlocked and globally disabled (disable_all_joints), otherwise the device will reject it.\n        After success, the zero reference is written to EEPROM.\n\n        Returns:\n            True is returned if calibration is successful; False is returned if not unlocked (0x20)/status not allowed (0x23)/write failure (0x31).'
        ok, status = self._write_with_ack(ParentCmd.CONFIG, ConfigSub.CALIBRATE_ZERO)
        if not ok:
            logger.error(f"Zero point calibration failed, status code=0x{status:02X}")
        return ok

    def get_hand_type(self, timeout_ms: int = 200) -> Optional[str]:
        "Read hand shape (subcommand 0x06, no need to unlock).\n\n        Args:\n            timeout_ms: response waiting timeout\n        Returns:\n            'left'/'right'; Not configured (0x24) or returns None on failure."
        if not self.send_frame(Access.READ, ParentCmd.CONFIG, ConfigSub.HAND_TYPE):
            return None
        resp = self._wait_for_response(ParentCmd.CONFIG, ConfigSub.HAND_TYPE, timeout_ms)
        if not resp:
            return None
        data, status, _ = resp
        if status == StatusCode.ERR_NOT_SET:
            return None  #Hand shape not configured
        if status != StatusCode.OK or len(data) < 4:
            return None
        return 'right' if data[3] == HandType.RIGHT else 'left'  #BYTE3 is hand type

    def set_hand_type(self, hand_type: int) -> bool:
        'Writing hand type (subcommand 0x06, needs to be unlocked first).\n\n        After writing, the forward and reverse rotation range and limit of the joint are refreshed.\n\n        Args:\n            hand_type: 0=left hand, not 0=right hand\n        Returns:\n            True is returned if writing is successful; False is returned if failure such as not unlocking or illegal parameters occurs.'
        ht = HandType.RIGHT if hand_type else HandType.LEFT
        ok, status = self._write_with_ack(ParentCmd.CONFIG, ConfigSub.HAND_TYPE, bytes([int(ht)]))
        if not ok:
            logger.error(f"Writer type failed, status code=0x{status:02X}")
        return ok

    # =========================================================================
    #Parent command 0x4: Periodic reporting
    # =========================================================================

    def config_periodic_report(self, sub_cmd: int, enable: bool,
                               period_ms: int = 20, joint_mask: int = 0x00000000) -> bool:
        'Configuration cycle reporting (subcommand 0x01~0x05, write).\n\n        Load: Enable(1B) + Period(uint32 big endian) + joint bit mask (uint32 big endian).\n        Mask Bit0->J1 ... Bit16->J17, joint_mask=0 means reporting all 17 joints.\n        The recommended period is 20~1000ms. If the device exceeds the range, it will return 0x35.\n\n        Args:\n            sub_cmd: Report item subcommand (PeriodicSub.*)\n            enable: True=enable reporting, False=disable\n            period_ms: reporting period (milliseconds)\n            joint_mask: joint selection bit mask\n        Returns:\n            Returns True if the configuration is successful; False if it fails.'
        data = bytes([0x01 if enable else 0x00])   # Enable
        data += period_ms.to_bytes(4, 'big')        #Period(big endian)
        data += joint_mask.to_bytes(4, 'big')       #Joint bit mask (big endian)
        ok, status = self._write_with_ack(ParentCmd.PERIODIC, sub_cmd, data)
        if not ok:
            logger.error(f"Periodic reporting configuration failure sub=0x{sub_cmd:X}, status code=0x{status:02X}")
        return ok

    def stop_periodic_report(self, sub_cmd: int) -> bool:
        'Turn off periodic reporting of the specified subcommand (Enable=0).\n\n        Args:\n            sub_cmd: Report item subcommand (PeriodicSub.*)\n        Returns:\n            Returns True if the configuration is successful.'
        return self.config_periodic_report(sub_cmd, enable=False, period_ms=0, joint_mask=0)

    def read_periodic_report(self, sub_cmd: int, joint_count: int,
                             byte_per_joint: int = 2, signed: bool = True,
                             timeout_ms: int = 50) -> Optional[List[int]]:
        'Read a frame and the device actively reports data.\n\n        The active reporting frame is Access=0, single frame, no status code, and the data starts from BYTE2 according to J1->J17\n        Arrange in compact order (only selected joints).\n\n        Args:\n            sub_cmd: Report item subcommand (PeriodicSub.*)\n            joint_count: The number of selected joints (17 when the mask is 0)\n            byte_per_joint: number of bytes per joint (position/current/speed=2, temperature/error code=1)\n            signed: whether there is a sign\n            timeout_ms: Timeout waiting for reported frame\n        Returns:\n            List of joint values; returns None on timeout.'
        start = time.time()
        while (time.time() - start) < timeout_ms / 1000.0:
            for _, data, parsed in self.receive_messages(
                timeout_ms=10, expected_parent=ParentCmd.PERIODIC, expected_sub=sub_cmd
            ):
                if parsed['access'] != Access.READ or len(data) < 2:
                    continue
                #Actively reports no status code, data starting from BYTE2
                return self._unpack_joints(data, 2, joint_count, byte_per_joint, signed)
            time.sleep(0.002)
        return None

    # =========================================================================
    #Parent command 0x5: Single query
    # =========================================================================

    def get_joint_positions(self) -> Optional[List[int]]:
        'Single query of 17 current joint positions (subcommand 0x01, int16 big endian). Returns None on failure.'
        return self._read_response_joints(ParentCmd.QUERY, QuerySub.POSITION,
                                          JOINT_COUNT, signed=True, byte_per_joint=2)

    def get_joint_currents(self) -> Optional[List[int]]:
        'Single query of 17 current currents (subcommand 0x02, unit 6.5ma). Returns None on failure.'
        return self._read_response_joints(ParentCmd.QUERY, QuerySub.CURRENT,
                                          JOINT_COUNT, signed=True, byte_per_joint=2)

    def get_joint_torques(self) -> Optional[List[int]]:
        '[Compatible with old interface] Read 17-way joint torque/current.\n\n        The new version of the protocol has no independent "torque" read in a single query, which is consistent with the old version 0x02 read semantics.\n        Return current query results (unit 6.5ma) to keep consistent with realhand_l30_v6_canfd\n        The same external method name is convenient for universal calling of realhand_l30_api.py.\n        Returns None on failure.'
        return self.get_joint_currents()

    def get_joint_velocities(self) -> Optional[List[int]]:
        'Single query of the current speed of channel 17 (subcommand 0x03, unit 0.732rpm). Returns None on failure.'
        return self._read_response_joints(ParentCmd.QUERY, QuerySub.SPEED,
                                          JOINT_COUNT, signed=True, byte_per_joint=2)

    def get_joint_temperatures(self) -> Optional[List[int]]:
        'Single query of 17 channels of current temperature (subcommand 0x04, unit 1°C, 1 byte per joint). Returns None on failure.'
        return self._read_response_joints(ParentCmd.QUERY, QuerySub.TEMPERATURE,
                                          JOINT_COUNT, signed=False, byte_per_joint=1)

    def get_joint_error_codes(self) -> Optional[List[int]]:
        'Single query of 17-channel servo error codes (subcommand 0x05, 1 byte per joint). Returns None on failure.'
        return self._read_response_joints(ParentCmd.QUERY, QuerySub.ERROR_CODE,
                                          JOINT_COUNT, signed=False, byte_per_joint=1)

    # =========================================================================
    #Parent command 0x6: Single joint debugging control (write only, subcommand = (function code<<5)|joint number)
    # =========================================================================

    @staticmethod
    def _single_joint_subcmd(func: int, joint_idx: int) -> int:
        'Calculate single joint subcommand = (function code<<5)|joint number.\n\n        Args:\n            func: function code (1=position, 2=torque, 3=speed)\n            joint_idx: joint number 1~17 (corresponding to J1~J17)\n        Returns:\n            Subcommand value (position 0x21~0x31 / torque 0x41~0x51 / speed 0x61~0x71).'
        return ((func & 0x7) << 5) | (joint_idx & 0x1F)

    def set_single_joint_position(self, joint_idx: int, value: int) -> Tuple[bool, int]:
        'Single joint position control (function code 0x1, subcommand 0x21~0x31, response).\n\n        Args:\n            joint_idx: joint number 1~17\n            value: target position (int16, cropped by JOINT_LIMITS)\n        Returns:\n            (Success, status code); The joint number is out of bounds and returns (False, -1).'
        if not (1 <= joint_idx <= JOINT_COUNT):
            logger.error('Joint number is out of bounds, should be 1~17')
            return False, -1
        lo, hi = JOINT_LIMITS.get(joint_idx, (-32768, 32767))
        v = max(lo, min(hi, int(value)))
        sub = self._single_joint_subcmd(SingleJointFunc.POSITION, joint_idx)
        return self._write_with_ack(ParentCmd.SINGLE_JOINT, sub,
                                    v.to_bytes(2, 'big', signed=True))

    def set_single_joint_torque(self, joint_idx: int, value: int) -> Tuple[bool, int]:
        'Single joint torque control (function code 0x2, subcommand 0x41~0x51, response).\n\n        Unit 6.5ma, range 60~800, over-range clipping.\n\n        Args:\n            joint_idx: joint number 1~17\n            value: target torque\n        Returns:\n            (Success, status code); The joint number is out of bounds and returns (False, -1).'
        if not (1 <= joint_idx <= JOINT_COUNT):
            logger.error('Joint number is out of bounds, should be 1~17')
            return False, -1
        v = max(60, min(800, int(value)))
        sub = self._single_joint_subcmd(SingleJointFunc.TORQUE, joint_idx)
        return self._write_with_ack(ParentCmd.SINGLE_JOINT, sub, v.to_bytes(2, 'big'))

    def set_single_joint_speed(self, joint_idx: int, value: int) -> Tuple[bool, int]:
        'Single joint speed control (function code 0x3, subcommand 0x61~0x71, response).\n\n        The unit is 0.732rpm, the range is 1~250, and it will be clipped if it exceeds the range.\n\n        Args:\n            joint_idx: joint number 1~17\n            value: target speed\n        Returns:\n            (Success, status code); The joint number is out of bounds and returns (False, -1).'
        if not (1 <= joint_idx <= JOINT_COUNT):
            logger.error('Joint number is out of bounds, should be 1~17')
            return False, -1
        v = max(1, min(250, int(value)))
        sub = self._single_joint_subcmd(SingleJointFunc.SPEED, joint_idx)
        return self._write_with_ack(ParentCmd.SINGLE_JOINT, sub, v.to_bytes(2, 'big'))


# =============================================================================
#CRC algorithm (for Bootloader upgrade, see protocol §16.6)
# =============================================================================

def crc16_modbus(data: bytes) -> int:
    'Calculate single packet CRC16-Modbus (see protocol §16.6.1, for upgrade subcommand 0x02).\n\n    Parameters are firmware data bytes; initial value 0xFFFF, polynomial 0xA001 (reverse), no final XOR.\n\n    Args:\n        data: byte sequence to be verified\n    Returns:\n        16-bit CRC value.'
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def crc32_firmware(data: bytes) -> int:
    'Calculate the entire packet CRC32-IEEE (see protocol §16.6.2, for upgrade subcommand 0x03).\n\n    The initial value is 0xFFFFFFFF, the polynomial is 0xEDB88320 (reverse), and the result is XORed with 0xFFFFFFFF.\n\n    Args:\n        data: All valid bytes of the firmware (spliced in order of sending packets)\n    Returns:\n        32-bit CRC value.'
    c = 0xFFFFFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if (c & 1) else (c >> 1)
    return (c ^ 0xFFFFFFFF) & 0xFFFFFFFF


# =============================================================================
#Advanced Controller
# =============================================================================

class L30DexterousHandController:
    'L30 dexterous hand advanced controller (new version of CANFD protocol).\n\n    Encapsulates connection management, batch control, status reading and normalization on L30CANFDProtocol,\n    Provides an interface closer to the application layer, and supports automatic connection/disconnection with context.'

    JOINT_COUNT = JOINT_COUNT

    def __init__(self, device_id: int = 1, canfd_id: int = 0, src_id: int = 0,
                 node_id: Optional[int] = None, canfd_device: Optional[int] = None,
                 comm_type: str = "libcanbus", channel=None, bitrate: int = 1000000,
                 dbitrate: int = 5000000, auto_setup: bool = True,
                 enable_on_connect: bool = False):
        'Construct high-level controllers.\n\n        To be consistent with the old version of realhand_l30_v6_canfd.L30DexterousHandController\n        The external interface is compatible with two sets of parameter naming:\n          - Old naming: device_id (device/node ID) / canfd_id (CANFD box number)\n          - New naming: node_id / canfd_device\n\n        Args:\n            device_id: device node ID (DstID), default 1; the old interface uses this name\n            canfd_id: CANFD device (box) index, default 0; the old interface continues to use this name\n            src_id: main site source ID (SrcID), default 0\n            node_id: The newly named device node ID, overrides device_id if given\n            canfd_device: Newly named CANFD device index, overrides canfd_id if given\n            comm_type: communication backend - "libcanbus" (default, vendor private library) or\n                       "socketcan" (kernel can0 + python-can, transparent plastic USB-CANFD device)\n            channel: channel - libcanbus is int (default 0), socketcan is the interface name (default "can0")\n            bitrate/dbitrate/auto_setup: only used by socketcan'
        #The new naming takes precedence, otherwise it falls back to the old naming, so that both calling methods can work
        nid = node_id if node_id is not None else device_id
        dev = canfd_device if canfd_device is not None else canfd_id
        #channel is not explicitly specified, the default value is given by the backend: socketcan -> "can0", libcanbus -> 0
        if channel is None:
            channel = "can0" if comm_type == "socketcan" else 0
        self.protocol = L30CANFDProtocol(nid, src_id, dev, channel=channel,
                                         comm_type=comm_type, bitrate=bitrate,
                                         dbitrate=dbitrate, auto_setup=auto_setup)
        self.node_id = nid
        self.device_id = nid                            #Compatible with old attribute names
        self.hand_type: Optional[str] = None            #populate left/right
        self.enable_on_connect = enable_on_connect
        self.joints = {j.id: j for j in JOINT_DEFINITIONS}  #Joint static information table

    #---- Connection Management --
    def connect(self) -> Tuple[bool, Optional[str]]:
        'Connect and enable dexterous hands while reading hand patterns.\n\n        Returns:\n            (Whether the connection was successful, hand string or None).'
        logger.info('Start connecting smart hands...')
        if not self.protocol.initialize():
            return False, None

        if self.enable_on_connect:
            print('Enable all joints...')
            if not self.protocol.enable_all_joints():
                logger.warning('Joint enable failed, continue trying...')
            time.sleep(0.1)

        #Device type query occasionally times out/loses packets, retry several times to avoid misjudgment as hand_type mismatch
        self.hand_type = None
        for _ in range(5):
            self.hand_type = self.protocol.get_hand_type()
            if self.hand_type:
                break
            time.sleep(0.05)
        logger.info(f"Connection successful: NodeID={self.node_id}, Type={self.hand_type}")
        return True, self.hand_type

    def disconnect(self) -> None:
        'Disconnect (turn off underlying CANFD).'
        self.protocol.close()

    @property
    def is_connected(self) -> bool:
        'is connected.'
        return self.protocol.is_connected

    #---- Joint control --
    def set_positions(self, positions: List[int]) -> bool:
        'Set the 17-way joint position; if the length does not match, a ValueError will be thrown.'
        if len(positions) != self.JOINT_COUNT:
            raise ValueError(f"requires {self.JOINT_COUNT} joint values")
        return self.protocol.set_joint_positions(positions)

    def set_velocities(self, velocities: List[int]) -> bool:
        'Set 17 joint speeds (unit 0.732rpm).'
        return self.protocol.set_joint_velocities(velocities)

    def set_torques(self, torques: List[int]) -> bool:
        'Set 17 joint torques (unit 6.5ma).'
        return self.protocol.set_joint_torques(torques)

    def get_positions(self) -> Optional[List[int]]:
        'Reads the current joint position of channel 17.'
        return self.protocol.get_joint_positions()

    def get_velocities(self) -> Optional[List[int]]:
        'Reads the current joint speed of channel 17.'
        return self.protocol.get_joint_velocities()

    def get_currents(self) -> Optional[List[int]]:
        'Read 17 current joint currents.'
        return self.protocol.get_joint_currents()

    def get_temperatures(self) -> Optional[List[int]]:
        'Reads the current joint temperature of channel 17.'
        return self.protocol.get_joint_temperatures()

    def get_error_codes(self) -> Optional[List[int]]:
        'Reads the 17th servo error code.'
        return self.protocol.get_joint_error_codes()

    def enable_all(self) -> bool:
        'globally enables all joints.'
        return self.protocol.enable_all_joints()

    def disable_all(self) -> bool:
        'Globally disables all joints.'
        return self.protocol.disable_all_joints()

    def stop(self) -> bool:
        'Global emergency stop (not supported by firmware yet).'
        return self.protocol.emergency_stop()

    def calibrate(self) -> bool:
        'Calibrate zero point (first unlock and disable internally to meet the preconditions for calibration).'
        self.protocol.unlock()
        self.protocol.disable_all_joints()
        return self.protocol.calibrate_zero()

    #---- Tactile ----
    def get_matrix_touch(self) -> Dict[str, list]:
        'Reads all 5-finger tactile matrices.'
        return self.protocol.get_all_touch()

    #---- Information ----
    def get_joint_name(self) -> Tuple[List[str], List[str]]:
        'Get joint names (English list, Chinese list).'
        return JOINT_NAME_EN, JOINT_NAME_CN

    def get_joint_range(self) -> Dict[int, Tuple[int, int]]:
        'Get the joint limit dictionary {joint number: (min,max)}.'
        return JOINT_LIMITS

    def get_device_info(self) -> Optional[Dict]:
        'Read device information (DeviceInfo).'
        return self.protocol.get_device_info()

    def get_all_state(self, is_touch: bool = False) -> Dict:
        'Get the complete status (position/speed/current/temperature/error code).\n\n        Args:\n            is_touch: When True, a tactile matrix is attached (which takes longer)\n        Returns:\n            Status dictionary.'
        state = {
            'positions': self.protocol.get_joint_positions(),
            'velocities': self.protocol.get_joint_velocities(),
            'currents': self.protocol.get_joint_currents(),
            'temperatures': self.protocol.get_joint_temperatures(),
            'error_codes': self.protocol.get_joint_error_codes(),
        }
        if is_touch:
            state['matrix_touch'] = self.get_matrix_touch()
        return state

    #---- Normalization --
    def normalize_positions(self, raw_positions: List[int]) -> List[float]:
        'Normalize the original position value to 0~1 according to the limit of each joint.\n\n        Args:\n            raw_positions: original position list (by J1~J17)\n        Returns:\n            Normalized floating point list; when the limit is invalid, it takes 0.5.'
        result = []
        for i, pos in enumerate(raw_positions):
            lo, hi = JOINT_LIMITS.get(i + 1, (0, 1))
            result.append(max(0.0, min(1.0, (pos - lo) / (hi - lo))) if hi != lo else 0.5)
        return result

    def denormalize_positions(self, normalized: List[float]) -> List[int]:
        'Restore the normalized value of 0~1 to the original position according to the limit of each joint.\n\n        Args:\n            normalized: normalized list (by J1~J17)\n        Returns:\n            List of original position integers.'
        result = []
        for i, norm in enumerate(normalized):
            lo, hi = JOINT_LIMITS.get(i + 1, (0, 0))
            result.append(int(round(lo + norm * (hi - lo))))
        return result

    def __enter__(self) -> 'L30DexterousHandController':
        'Context Manager Entry: Automatic connection.'
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        'Context Manager Exit: Automatically disconnected.'
        self.disconnect()


# =============================================================================
#Convenience function
# =============================================================================

def create_default_controller(node_id: int = 1) -> L30DexterousHandController:
    'Create a default advanced controller.\n\n    Args:\n        node_id: device node ID, default 1\n    Returns:\n        L30DexterousHandController instance (not yet connected).'
    return L30DexterousHandController(node_id)


def setup_logging(level: int = logging.INFO) -> None:
    'Configure the root log format and level.\n\n    Args:\n        level: log level, default logging.INFO'
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

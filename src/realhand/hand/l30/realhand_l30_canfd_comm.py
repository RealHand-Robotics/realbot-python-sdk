#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'L30 Dexterity CANFD transport layer (pluggable backend)\n\nL30 uses 29-bit extended frames. This module decouples "underlying transceiver" from the protocol logic and provides two backends for\nrealhand_l30_v6_canfd.py and realhand_l30_v6_2_canfd.py are shared by two control classes:\n\n  - LibCanBusTransport: Manufacturer\'s private library libcanbus.so (default, original communication method).\n  - SocketCANTransport: Kernel native SocketCAN + python-can (for "transparent plastic USB to\n                         CANFD device"). Does not depend on libcanbus.so, sends and receives through the can0 interface,\n                         The frame format is consistent with the manufacturer\'s equipment (29-bit extended frame).\n\nThe two backend interfaces are consistent (polling transceiver, matching the original synchronous transceiver model of the L30 protocol):\n    initialize() -> bool initialize and connect\n    send(frame_id: int, payload: bytes) -> bool Send an extended frame CANFD\n    receive(timeout_ms: int) -> List[(frame_id, bytes)] Poll to retrieve received frames\n    close() -> None close\n    is_connected: bool connection status\n\nConvention: payload is "data segment valid bytes" (BYTE0 data length + BYTE1 transaction control + data), each backend\nResponsible for padding zeros to the legal CANFD frame length. The upper layer protocol only cares about framing/parsing and does not care about the specific transmission method.'

import os
import time
import logging
import subprocess
from typing import List, Optional, Tuple
from ctypes import (
    Structure, CDLL, cdll, cast, byref, RTLD_GLOBAL,
    c_uint, c_ushort, c_char, c_ubyte, c_uint16, POINTER,
)

logger = logging.getLogger(__name__)

STATUS_OK = 0


# =============================================================================
#DLC Encoding <-> Number of Line Bytes (CAN FD Standard Table)
# =============================================================================

DLC_TO_LENGTH = {
    0x00: 0, 0x01: 1, 0x02: 2, 0x03: 3, 0x04: 4, 0x05: 5, 0x06: 6, 0x07: 7,
    0x08: 8, 0x09: 12, 0x0A: 16, 0x0B: 20, 0x0C: 24, 0x0D: 32, 0x0E: 48, 0x0F: 64,
}
NON_STANDARD_DLC_MAP = {0x10: 16, 0x40: 64}  #A few devices may backfill non-standard DLC values


def get_dlc_from_length(length: int) -> int:
    'According to the number of line bytes, take the closest legal DLC code (if any is missing, the device/this layer will fill in 0x00).'
    if length <= 8:
        return length
    if length <= 12:
        return 0x09
    if length <= 16:
        return 0x0A
    if length <= 20:
        return 0x0B
    if length <= 24:
        return 0x0C
    if length <= 32:
        return 0x0D
    if length <= 48:
        return 0x0E
    return 0x0F


def get_length_from_dlc(dlc: int) -> int:
    'Gets the number of line bytes based on DLC encoding.'
    if dlc in DLC_TO_LENGTH:
        return DLC_TO_LENGTH[dlc]
    if dlc in NON_STANDARD_DLC_MAP:
        return NON_STANDARD_DLC_MAP[dlc]
    return min(dlc, 64)


# =============================================================================
#libcanbus structure
# =============================================================================

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


# =============================================================================
#Transport layer base class
# =============================================================================

class L30Transport:
    'CANFD transport layer interface (29-bit extended frame, polling transmission and reception). Subclasses implement concrete backends.'

    def initialize(self) -> bool:
        'initializes and connects to the underlying device. Return True on success.'
        raise NotImplementedError

    def send(self, frame_id: int, payload: bytes) -> bool:
        'sends an extended frame CANFD. The payload is the valid bytes of the data segment, padded with zeros from this layer to the legal frame length.'
        raise NotImplementedError

    def receive(self, timeout_ms: int = 3) -> List[Tuple[int, bytes]]:
        'Polls to retrieve received frames and returns [(frame_id, data_bytes), ...]; If there is no data, an empty list is returned.'
        raise NotImplementedError

    def close(self) -> None:
        'Turns off the underlying device.'
        raise NotImplementedError


# =============================================================================
#Backend 1: Manufacturer’s private library libcanbus.so (default)
# =============================================================================

class LibCanBusTransport(L30Transport):
    "CANFD transport layer (29-bit extended frame, polling transceiver) based on the manufacturer's private library libcanbus.so.\n\n    The arbitration segment is 1Mbps and the data segment is 5Mbps; consistent with the original implementation of L30."

    ARBITRATION_BAUD = 1000000
    DATA_BAUD = 5000000

    def __init__(self, canfd_device: int = 0, channel: int = 0):
        'Args:\n            canfd_device: CANFD device (box) index\n            channel: channel number'
        self.canfd_device = canfd_device
        self.channel = channel
        self.canDLL = None
        self.is_connected = False

    def initialize(self) -> bool:
        'Load library -> Scan -> Open channel -> CANFD configuration (1M/5M) -> Set filter (all pass).'
        try:
            CDLL("/usr/local/lib/libusb-1.0.so", RTLD_GLOBAL)
            time.sleep(0.1)
            self.canDLL = cdll.LoadLibrary("/usr/local/lib/libcanbus.so")

            ret = self.canDLL.CAN_ScanDevice()
            if ret <= 0:
                logger.error(f"CANFD device not found, error code: {ret}")
                return False
            print(f"{ret} devices found", flush=True)

            ret = self.canDLL.CAN_OpenDevice(self.canfd_device, self.channel)
            if ret != STATUS_OK:
                logger.error(f"Failed to open the device, error code: {ret}")
                return False
            print(f"Device channel {self.channel} opened successfully", flush=True)

            can_config = CanFD_Config(
                self.ARBITRATION_BAUD, self.DATA_BAUD,
                0x0, 0x0, 0x0, 0x0,
                0x0, 0x0, 0x0, 0x0,
                0x0, 0x0, 0x1,
            )
            ret = self.canDLL.CANFD_Init(self.canfd_device, self.channel, byref(can_config))
            if ret != STATUS_OK:
                logger.error(f"CANFD initialization failed, error code: {ret}")
                self.canDLL.CAN_CloseDevice(self.canfd_device, self.channel)
                return False

            ret = self.canDLL.CAN_SetFilter(self.canfd_device, self.channel, 0, 0, 0, 0, 1)
            if ret != STATUS_OK:
                logger.error(f"Failed to set filter, error code: {ret}")
                self.canDLL.CAN_CloseDevice(self.canfd_device, self.channel)
                return False

            self.is_connected = True
            return True
        except OSError as e:
            logger.error(f"Failed to load CAN library: {e}")
            return False
        except Exception as e:
            logger.error(f"CANFD initialization exception: {e}")
            return False

    def send(self, frame_id: int, payload: bytes) -> bool:
        'Copy the payload into the 64-byte buffer, get the DLC by length, and send it in the extended frame CANFD.'
        if not self.is_connected:
            return False
        try:
            dlc = get_dlc_from_length(len(payload))
            buf = (c_ubyte * 64)()
            for i, b in enumerate(payload[:64]):
                buf[i] = b
            #FrameType=4(CANFD), ExternFlag=1(29-bit extended frame)
            msg = CanFD_Msg(
                ID=frame_id, TimeStamp=0, FrameType=4, DLC=dlc,
                ExternFlag=1, RemoteFlag=0, BusSatus=0, ErrSatus=0,
                TECounter=0, RECounter=0, Data=buf,
            )
            time.sleep(0.001)
            ret = self.canDLL.CANFD_Transmit(self.canfd_device, self.channel, byref(msg), 1, 100)
            return ret == 1
        except Exception as e:
            logger.error(f"Exception sending message (libcanbus): {e}")
            return False

    def receive(self, timeout_ms: int = 3) -> List[Tuple[int, bytes]]:
        'retrieves several frames at a time from the driver receive buffer, and DLC checks the length and returns (id, data).'
        if not self.is_connected:
            return []
        try:
            class MsgArray(Structure):
                _fields_ = [('SIZE', c_uint16), ('ARRAY', CanFD_Msg * 100)]

                @property
                def ptr(self):
                    return cast(byref(self.ARRAY), POINTER(CanFD_Msg))

            receive_buffer = MsgArray()
            receive_buffer.SIZE = 100
            ret = self.canDLL.CANFD_Receive(self.canfd_device, self.channel,
                                            receive_buffer.ptr, 100, timeout_ms)
            if ret <= 0:
                return []
            out = []
            for i in range(ret):
                msg = receive_buffer.ARRAY[i]
                data_len = get_length_from_dlc(msg.DLC)
                out.append((msg.ID, bytes(msg.Data[:data_len])))
            return out
        except Exception as e:
            logger.error(f"Exception receiving message (libcanbus): {e}")
            return []

    def close(self) -> None:
        'Turns off the CANFD device.'
        if self.canDLL and self.is_connected:
            try:
                self.canDLL.CAN_CloseDevice(self.canfd_device, self.channel)
            except Exception as e:
                logger.error(f"Failed to close CANFD connection: {e}")
        self.is_connected = False


# =============================================================================
#Backend 2: Kernel native SocketCAN + python-can (transparent plastic-sealed USB to CANFD device)
# =============================================================================

class SocketCANTransport(L30Transport):
    'CANFD transport layer (29-bit extended frame, polling transmission and reception) based on the kernel\'s native SocketCAN (python-can).\n\n    For "transparent plastic USB to CANFD device": use standard SocketCAN under Linux, no manufacturer\'s private library is required\n    (libcanbus.so), sends and receives through the kernel can0 interface + python-can. The frame format is consistent with the manufacturer\'s equipment\n    (L30 is a 29-bit extended frame), so the upper layer protocol does not require any changes.'

    def __init__(self, channel: str = "can0", bitrate: int = 1000000,
                 dbitrate: int = 5000000, auto_setup: bool = True,
                 bitrate_switch: bool = False):
        'Args:\n            channel: SocketCAN interface name, such as "can0"\n            bitrate: arbitration section baud rate (default 1Mbps, consistent with L30 libcanbus)\n            dbitrate: data segment baud rate (default 5Mbps)\n            auto_setup: whether to automatically execute ip link configuration and pull up the interface (sudo required)\n            bitrate_switch: Whether to enable BRS (data segment switching speed) when sending frames. Default False - actual measurement\n                BRS is on + the high-speed data segment is easy to drive the bus into BUS-OFF. When it is closed, the data segment maintains arbitration.\n                Baud rate, more stable sending and receiving.'
        self.channel = channel
        self.bitrate = bitrate
        self.dbitrate = dbitrate
        self.auto_setup = auto_setup
        self.bitrate_switch = bitrate_switch
        self.bus = None
        self.is_connected = False

    def _setup_interface(self):
        'Automatically configure the can0 interface (requires sudo permissions).\n\n        Equivalent to manual execution:\n            sudo ip link set can0 down\n            sudo ip link set can0 type can bitrate <b> dbitrate <d> fd on\n            sudo ip link set can0 up\n            sudo ip link set can0 txqueuelen 1000\n        Fault tolerance one by one, no interruption in case of failure (the interface may have been configured by the user in advance).'
        cmds = [
            (["sudo", "ip", "link", "set", self.channel, "down"], True),
            (["sudo", "ip", "link", "set", self.channel, "type", "can",
              "bitrate", str(self.bitrate), "dbitrate", str(self.dbitrate),
              "fd", "on"], True),
            #restart-ms is not supported by some controllers, is delivered separately and allows failure
            (["sudo", "ip", "link", "set", self.channel, "type", "can",
              "restart-ms", "100"], False),
            (["sudo", "ip", "link", "set", self.channel, "up"], True),
            (["sudo", "ip", "link", "set", self.channel, "txqueuelen", "1000"], True),
        ]
        for c, required in cmds:
            try:
                ret = subprocess.run(c, check=False, timeout=5,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if ret.returncode != 0:
                    err = ret.stderr.decode(errors='ignore').strip()
                    if required:
                        print(f"[SocketCAN] Interface configuration command failed: {' '.join(c)} Reason: {err}")
                    else:
                        print(f"[SocketCAN] Tip: Optional options not supported by this device have been skipped: {' '.join(c)} ({err})")
                time.sleep(0.1)  #Give the kernel time to complete the state switch (especially after down)
            except Exception as e:
                print(f"[SocketCAN] Interface configuration command exception: {' '.join(c)} -> {e}")

        #Read back the actual effective timing and status to facilitate locating baud rate mismatch / BUS-OFF
        try:
            ret = subprocess.run(["ip", "-details", "link", "show", self.channel],
                                 check=False, timeout=5,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = ret.stdout.decode(errors='ignore')
            timing = next((l.strip() for l in out.splitlines() if "bitrate" in l), '(not read)')
            state = next((l.strip() for l in out.splitlines() if "state" in l), "")
            print(f"[SocketCAN] expects bitrate={self.bitrate} dbitrate={self.dbitrate}")
            print(f"Actual effective: {timing}")
            print(f"status line: {state}")
            if "BUS-OFF" in state or "ERROR-PASSIVE" in state:
                print('[SocketCAN] ⚠️ The bus is in an error state, usually the baud rate is inconsistent with the device or a wiring/termination resistor issue.')
        except Exception:
            pass

    def initialize(self) -> bool:
        'Automatically pull up the interface -> Verify the existence of the interface -> Open the python-can CANFD bus.'
        try:
            import can  # noqa: F401
        except ImportError:
            print('❌ The python-can library is missing, please execute: pip install python-can')
            return False

        try:
            import can
            print(f"Start initializing SocketCAN device (channel: {self.channel})...")

            if self.auto_setup:
                print(f"Automatically configuring interface {self.channel}"
                      f"(bitrate={self.bitrate}, dbitrate={self.dbitrate}, fd on)...")
                self._setup_interface()
                time.sleep(0.2)

            #Confirm that the interface exists before opening the bus to avoid python-can throwing obscure exceptions
            if not os.path.exists(f"/sys/class/net/{self.channel}"):
                print(f"❌ Network interface {self.channel} not found.")
                print('This transparent plastic device needs to be set to Linux mode to enumerate to the native CAN interface;')
                print("If lsusb displays 'STM32 Virtual ComPort', it means it is still in serial port mode.")
                print('Please: 1) Turn the switch below type-c to Linux mode 2) Re-plug and unplug the USB')
                print(f"3) Use `ip -br link show type can` to confirm that {self.channel} appears")
                self.is_connected = False
                self.bus = None
                return False

            self.bus = can.interface.Bus(channel=self.channel,
                                         interface='socketcan', fd=True)
            self.is_connected = True
            print(f"✅ SocketCAN channel {self.channel} opened successfully")
            return True
        except Exception as e:
            print(f"❌ SocketCAN initialization failed: {e}")
            print('Please check:')
            print('1. Whether the device is connected and whether the switch under type-c is set to Linux mode')
            print(f"2. Whether the interface {self.channel} exists (ip -br link show type can)")
            print('3. Do you have sudo permissions to automatically configure the interface')
            self.is_connected = False
            self.bus = None
            return False

    def send(self, frame_id: int, payload: bytes) -> bool:
        'sends the payload in 29-bit extended frame CANFD (zeros padded to the legal frame length).'
        if not self.is_connected or self.bus is None:
            return False
        try:
            import can
            data = bytes(payload)
            data_len = min(len(data), 64)
            data = data[:data_len]
            #Add zeros to the legal CANFD frame length (0-8,12,16,20,24,32,48,64)
            dlc = get_dlc_from_length(data_len)
            padded_len = DLC_TO_LENGTH[dlc]
            body = data + b'\x00' * (padded_len - data_len)

            msg = can.Message(
                arbitration_id=frame_id,
                is_extended_id=True,          #L30 is a 29-bit extended frame
                is_fd=True,
                bitrate_switch=self.bitrate_switch,
                data=body,
            )
            self.bus.send(msg, timeout=0.2)
            return True
        except Exception as e:
            print(f"Exception in sending message (SocketCAN): {e}")
            return False

    def receive(self, timeout_ms: int = 3) -> List[Tuple[int, bytes]]:
        'blocks to get the first frame (at most timeout_ms), then empties the buffer non-blockingly, and returns the (id, data) list.'
        if not self.is_connected or self.bus is None:
            return []
        out: List[Tuple[int, bytes]] = []
        try:
            first = self.bus.recv(timeout=timeout_ms / 1000.0)
            if first is None:
                return out
            out.append((first.arbitration_id, bytes(first.data)))
            #Drain the remaining frames that have currently arrived (non-blocking)
            while True:
                m = self.bus.recv(timeout=0.0)
                if m is None:
                    break
                out.append((m.arbitration_id, bytes(m.data)))
        except Exception:
            pass
        return out

    def close(self) -> None:
        'Close the SocketCAN bus.'
        if self.bus is not None:
            try:
                self.bus.shutdown()
            except Exception as e:
                print(f"Close SocketCAN connection failed: {e}")
        self.bus = None
        self.is_connected = False


def create_transport(comm_type: str = "libcanbus", canfd_device: int = 0,
                     channel=0, bitrate: int = 1000000, dbitrate: int = 5000000,
                     auto_setup: bool = True) -> L30Transport:
    'Create the corresponding CANFD transmission backend according to comm_type.\n\n    Args:\n        comm_type: "libcanbus" (default, vendor library) or "socketcan" (kernel can0 + python-can)\n        canfd_device: libcanbus only - device (box) index\n        channel: under libcanbus is the channel number (int, default 0); under socketcan is the interface name (str, default "can0")\n        bitrate/dbitrate/auto_setup: only socketcan - arbitration/data segment baud rate and whether to automatically pull up the interface\n    Returns:\n        L30Transport subclass instance.'
    if comm_type == "socketcan":
        ch = channel if isinstance(channel, str) else "can0"
        return SocketCANTransport(channel=ch, bitrate=bitrate,
                                  dbitrate=dbitrate, auto_setup=auto_setup)
    ch = channel if isinstance(channel, int) else 0
    return LibCanBusTransport(canfd_device=canfd_device, channel=ch)

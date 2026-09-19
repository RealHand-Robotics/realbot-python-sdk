#!/usr/bin/env python3
"""Autodetecting PyQt5 GUI controller for the current RealHand SDK.

This is a self-contained replacement for the older
``example/gui_control/gui_control.py`` style GUI.  It uses the new SDK classes
directly: L6, O6, L20, L20lite, L25, and L30.  SocketCAN hands are
autodetected on can0 through can3; if none responds, the GUI also makes a
read-only L30 probe through the vendor ``libcanbus`` transport.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
# When this file is run directly from src/realhand, sys.path[0] points at the
# package directory. That makes Python resolve stdlib "queue" as realhand/queue.
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != CURRENT_DIR
]
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import can

from realhand import L6, L20, L20lite, L25, L30, O6  # noqa: E402
from realhand.comm import CANMessageDispatcher  # noqa: E402
from realhand.gui_presets import HAND_CONFIGS  # noqa: E402


LOOP_TIME_MS = 1000
PUBLISH_INTERVAL_MS = 100
LIVE_COMMAND_INTERVAL_MS = 100
STREAM_JOIN_TIMEOUT_S = 1.0
MAX_JOINT_POSITION = 100
DEFAULT_MODEL = "L20"
DEFAULT_SIDE = "right"
SENSOR_READ_MODES = ("stream", "snapshot", "get_blocking")
SENSOR_READ_MODE_ALIASES = {
    "blocking": "get_blocking",
}
DEFAULT_SENSOR_READ_MODE = "stream"
AUTO_GRAB_START_TORQUE = 10
AUTO_GRAB_MAX_DURATION_S = 10.0
SNAPSHOT_READ_INTERVAL_S = 0.05
BLOCKING_DEFAULT_TIMEOUT_MS = 250
BLOCKING_TIMEOUTS_BY_NAME = {
    "fault": 500,
    "force_sensor": 1000,
}
DEFAULT_CAN_BITRATE = 1_000_000
CAN_PROBE_INTERFACES = tuple(f"can{index}" for index in range(4))
PROBE_SIDES = (
    ("right", 0x27),
    ("left", 0x28),
)
SERIAL_NUMBER_CMD = 0xC0
SERIAL_TIMEOUT_S = 1.0


MODEL_CLASSES = {
    "L6": L6,
    "O6": O6,
    "L20lite": L20lite,
    "L20": L20,
    "L25": L25,
    "L30": L30,
}


EVENT_MODULES = {
    "L6": "realhand.hand.l6.events",
    "O6": "realhand.hand.o6.events",
    "L20lite": "realhand.hand.l20lite.events",
    "L20": "realhand.hand.l20.events",
    "L25": "realhand.hand.l25.events",
    "L30": "realhand.hand.l30.events",
}


POLL_INTERVALS_BY_NAME = {
    "angle": 1 / 30,
    "position": 1 / 30,
    "force_sensor": 1 / 15,
    "torque": 0.20,
    "speed": 0.50,
    "acceleration": 0.50,
    "temperature": 1.00,
    "current": 0.50,
    "fault": 1.00,
}

SENSOR_DATA_FIELDS = (
    ("angles", "angle"),
    ("torques", "torque"),
    ("temperatures", "temperature"),
    ("currents", "current"),
    ("speeds", "speed"),
    ("accelerations", "acceleration"),
    ("faults", "fault"),
)


def normalize_sensor_read_mode(value: Any) -> str:
    mode = str(value).lower()
    return SENSOR_READ_MODE_ALIASES.get(mode, mode)


DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "gui_control_config.json")
DEFAULT_FAULT_REPORT_DIR = os.path.join(CURRENT_DIR, "hand_fault_reports")
DEFAULT_MOTION_TIMING_DIR = os.path.join(CURRENT_DIR, "motion_timing_logs")
MOTION_START_THRESHOLD = 1.0
MOTION_TARGET_DELTA_THRESHOLD = 1.0
MOTION_COMPARABLE_MIN_TARGET_DELTA = 5.0
MOTION_PROGRESS_THRESHOLDS = (
    ("5pct", 0.05),
    ("10pct", 0.10),
    ("50pct", 0.50),
    ("90pct", 0.90),
)

MOTION_TIMING_COLUMNS = (
    "session_id",
    "command_index",
    "preset_name",
    "status",
    "send_started_at_iso",
    "send_started_at_unix",
    "send_duration_ms",
    "first_angle_sample_latency_ms",
    "first_motion_latency_ms",
    "send_complete_to_first_motion_ms",
    "comparable_dead_time_ms",
    "time_to_5pct_target_delta_ms",
    "time_to_10pct_target_delta_ms",
    "time_to_50pct_target_delta_ms",
    "time_to_90pct_target_delta_ms",
    "baseline_sample_age_ms",
    "sensor_sample_timestamp_iso",
    "sensor_sample_timestamp_unix",
    "moved_joint_indexes",
    "moved_joint_names",
    "comparable_joint_indexes",
    "comparable_joint_names",
    "threshold_crossed_joint_indexes",
    "threshold_crossed_joint_names",
    "target_values",
    "baseline_values",
    "first_angle_sample_values",
    "first_motion_values",
    "first_10pct_values",
    "model",
    "side",
    "serial_number",
    "interface_name",
    "interface_type",
    "sensor_read_mode",
)

SERIAL_MODEL_CODES = {
    "L6": "L6",
    "T6": "L6",
    "O6": "O6",
    "L20": "L20",
    "T20": "L20",
    "L20LITE": "L20lite",
    "L25": "L25",
}

SERIAL_SIDE_CODES = {
    "L": "left",
    "R": "right",
}

FAULT_REPORT_COLUMNS = (
    "test_id",
    "row_type",
    "read_at_iso",
    "read_at_unix",
    "report_result",
    "check_name",
    "check_result",
    "check_message",
    "selected_model",
    "selected_side",
    "serial_number",
    "serial_model",
    "serial_side",
    "serial_prefix",
    "serial_section_2",
    "serial_section_3",
    "serial_side_code",
    "serial_remaining",
    "firmware_version",
    "mechanical_version",
    "pcb_version",
    "interface_name",
    "interface_type",
    "fault_timestamp_iso",
    "fault_timestamp_unix",
    "joint_index",
    "joint_name",
    "fault_code_value",
    "fault_has_fault",
    "fault_names",
)


@dataclass(frozen=True)
class InterfaceSetupResult:
    should_probe: bool
    message: str


@dataclass(frozen=True)
class DetectedHand:
    interface: str
    interface_type: str
    model: str
    side: str
    serial_number: str
    arbitration_side: str

    def label(self) -> str:
        return (
            f"{self.interface}: {self.side} {self.model} "
            f"(serial {self.serial_number}, arbitration {self.arbitration_side})"
        )


@dataclass
class MotionTimingRecord:
    session_id: str
    command_index: int
    preset_name: str
    target_values: list[float]
    baseline_values: list[Any] | None
    baseline_perf_ns: int | None
    baseline_wall_time: float | None
    send_start_ns: int
    send_end_ns: int
    send_start_wall_time: float
    next_command_ns: int | None = None
    status: str = "pending"
    first_sample_ns: int | None = None
    first_sample_wall_time: float | None = None
    first_sample_values: list[Any] | None = None
    first_motion_ns: int | None = None
    first_motion_wall_time: float | None = None
    first_motion_values: list[Any] | None = None
    moved_joint_indexes: list[int] = field(default_factory=list)
    progress_ns_by_threshold: dict[str, int] = field(default_factory=dict)
    progress_wall_time_by_threshold: dict[str, float] = field(default_factory=dict)
    progress_values_by_threshold: dict[str, list[Any]] = field(default_factory=dict)
    progress_joint_indexes_by_threshold: dict[str, list[int]] = field(default_factory=dict)


def timestamp_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def parse_serial_identity(serial_number: str) -> dict[str, str]:
    sections = serial_number.strip().split("-") if serial_number else []
    serial_prefix = sections[0].strip() if len(sections) >= 1 else ""
    section_2 = sections[1].strip() if len(sections) >= 2 else ""
    section_3 = sections[2].strip() if len(sections) >= 3 else ""
    side_code = sections[3].strip().upper() if len(sections) >= 4 else ""
    remaining = "-".join(section.strip() for section in sections[4:])

    model_code = serial_prefix.upper()
    if model_code.startswith("LH"):
        model_code = model_code[2:]

    return {
        "serial_model": SERIAL_MODEL_CODES.get(model_code, model_code),
        "serial_side": SERIAL_SIDE_CODES.get(side_code, ""),
        "serial_prefix": serial_prefix,
        "serial_section_2": section_2,
        "serial_section_3": section_3,
        "serial_side_code": side_code,
        "serial_remaining": remaining,
    }


def _run_ip_link(args: list[str], *, timeout_s: float = 2.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["ip", "link", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return None


def configure_socketcan_interface(interface: str, bitrate: int, *, enabled: bool) -> InterfaceSetupResult:
    if not enabled:
        return InterfaceSetupResult(True, f"{interface}: CAN setup skipped")

    exists = _run_ip_link(["show", interface])
    if exists is None:
        return InterfaceSetupResult(True, f"{interface}: ip command not found; probing without link setup")
    if exists.returncode != 0:
        return InterfaceSetupResult(False, f"{interface}: not present")

    commands = (
        ["set", interface, "down"],
        ["set", interface, "type", "can", "bitrate", str(bitrate)],
        ["set", interface, "up"],
    )
    for command in commands:
        result = _run_ip_link(command)
        if result is None:
            return InterfaceSetupResult(True, f"{interface}: ip command not found; probing without link setup")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return InterfaceSetupResult(
                True,
                f"{interface}: setup failed ({detail or 'unknown ip link error'}); probing anyway",
            )
    return InterfaceSetupResult(True, f"{interface}: up at {bitrate} bps")


def query_can_value(
    *,
    dispatcher: CANMessageDispatcher,
    arbitration_id: int,
    request_data: list[int],
    handler: Any,
    timeout_s: float,
) -> Any:
    condition = threading.Condition()
    result: dict[str, Any] = {"ready": False, "value": None}

    def callback(msg: can.Message) -> None:
        value = handler(msg)
        if value is None:
            return
        with condition:
            result["ready"] = True
            result["value"] = value
            condition.notify()

    dispatcher.subscribe(callback)
    try:
        dispatcher.send(
            can.Message(
                arbitration_id=arbitration_id,
                data=request_data,
                is_extended_id=False,
            )
        )
        deadline = time.monotonic() + timeout_s
        with condition:
            while not result["ready"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                condition.wait(remaining)
        if not result["ready"]:
            raise TimeoutError("request timed out")
        return result["value"]
    finally:
        dispatcher.unsubscribe(callback)


def decode_standard_serial(frames: dict[int, bytes]) -> str:
    data = bytearray()
    for frame_id in range(4):
        data.extend(frames[frame_id])
    return data.rstrip(b"\x00").decode("ascii", errors="ignore")


def decode_indexed_serial(frames: dict[int, bytes]) -> str:
    data = bytearray(24)
    for byte_index, frame_data in frames.items():
        for offset, value in enumerate(frame_data):
            if byte_index + offset < len(data):
                data[byte_index + offset] = value
    return data.rstrip(b"\x00").decode("ascii", errors="ignore")


def read_serial_number(
    dispatcher: CANMessageDispatcher,
    arbitration_id: int,
    *,
    timeout_s: float = SERIAL_TIMEOUT_S,
) -> str:
    standard_frames: dict[int, bytes] = {}
    indexed_frames: dict[int, bytes] = {}

    def handler(msg: can.Message) -> str | None:
        if msg.arbitration_id != arbitration_id:
            return None
        if len(msg.data) < 2 or msg.data[0] != SERIAL_NUMBER_CMD:
            return None

        frame_key = int(msg.data[1])
        frame_data = bytes(msg.data[2:8])
        if frame_key in range(4):
            standard_frames[frame_key] = frame_data
            if all(index in standard_frames for index in range(4)):
                return decode_standard_serial(standard_frames)

        if frame_key in (0, 6, 12, 18):
            indexed_frames[frame_key] = frame_data
            if all(index in indexed_frames for index in (0, 6, 12, 18)):
                return decode_indexed_serial(indexed_frames)

        return None

    return query_can_value(
        dispatcher=dispatcher,
        arbitration_id=arbitration_id,
        request_data=[SERIAL_NUMBER_CMD],
        handler=handler,
        timeout_s=timeout_s,
    )


def detect_hands_on_interface(
    *,
    interface: str,
    interface_type: str,
) -> tuple[list[DetectedHand], list[str]]:
    detected: list[DetectedHand] = []
    messages: list[str] = []
    dispatcher = None

    try:
        dispatcher = CANMessageDispatcher(interface_name=interface, interface_type=interface_type)
    except Exception as exc:
        return detected, [f"{interface}: open failed: {exc}"]

    try:
        for arbitration_side, arbitration_id in PROBE_SIDES:
            try:
                serial_number = read_serial_number(dispatcher, arbitration_id)
            except Exception as exc:
                messages.append(f"{interface}/{arbitration_side}: no serial response ({exc})")
                continue

            identity = parse_serial_identity(serial_number)
            model = identity["serial_model"]
            side = identity["serial_side"] or arbitration_side
            if model not in MODEL_CLASSES:
                messages.append(
                    f"{interface}/{arbitration_side}: unsupported model "
                    f"{model or 'unknown'} from serial {serial_number}"
                )
                continue
            if side not in ("left", "right"):
                messages.append(
                    f"{interface}/{arbitration_side}: unknown side in serial "
                    f"{serial_number}; using arbitration side {arbitration_side}"
                )
                side = arbitration_side

            detected.append(
                DetectedHand(
                    interface=interface,
                    interface_type=interface_type,
                    model=model,
                    side=side,
                    serial_number=serial_number,
                    arbitration_side=arbitration_side,
                )
            )
    finally:
        if dispatcher is not None:
            try:
                dispatcher.stop()
            except Exception:
                pass

    return detected, messages


def autodetect_hands(
    *,
    interface_type: str,
    bitrate: int,
    setup_can: bool,
    interfaces: tuple[str, ...] = CAN_PROBE_INTERFACES,
) -> tuple[list[DetectedHand], list[str]]:
    detected: list[DetectedHand] = []
    messages: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for interface in interfaces:
        if interface_type == "socketcan":
            setup = configure_socketcan_interface(interface, bitrate, enabled=setup_can)
            messages.append(setup.message)
            if not setup.should_probe:
                continue

        interface_hands, interface_messages = detect_hands_on_interface(
            interface=interface,
            interface_type=interface_type,
        )
        messages.extend(interface_messages)
        for hand in interface_hands:
            key = (hand.interface, hand.side, hand.serial_number)
            if key in seen:
                continue
            seen.add(key)
            detected.append(hand)

    return detected, messages


def autodetect_l30_hand() -> tuple[list[DetectedHand], list[str]]:
    """Safely probe CANFD analyser 0 for one L30 hand.

    L30 uses the vendor libcanbus transport, so it cannot answer the regular
    SocketCAN serial-number probe above.  Constructing ``L30`` performs only
    its protocol/side handshake: the L30 wrapper explicitly disables the
    vendor controller's former enable-on-connect behavior.
    """
    messages: list[str] = []
    for side in ("right", "left"):
        hand = None
        try:
            hand = L30(side=side, canfd_id=0, interface_type="libcanbus")
            info = hand.info.get()
            serial_number = str(info.serial_number or "unknown")
            messages.append(f"libcanbus/0: detected {side} L30 (serial {serial_number})")
            return [
                DetectedHand(
                    interface="0",
                    interface_type="libcanbus",
                    model="L30",
                    side=side,
                    serial_number=serial_number,
                    arbitration_side=side,
                )
            ], messages
        except Exception as exc:
            messages.append(f"libcanbus/0: no {side} L30 response ({exc})")
        finally:
            if hand is not None:
                try:
                    hand.close()
                except Exception:
                    pass
    return [], messages


def choose_detected_hand(hands: list[DetectedHand]) -> tuple[DetectedHand | None, bool]:
    if not hands:
        return None, False
    if len(hands) == 1:
        return hands[0], True

    labels = [hand.label() for hand in hands]
    selected_label, accepted = QInputDialog.getItem(
        None,
        "Select RealHand",
        "Multiple RealHand devices detected. Select one to connect:",
        labels,
        0,
        False,
    )
    if not accepted:
        return None, True
    return hands[labels.index(selected_label)], True


def selection_from_detected_hand(hand: DetectedHand) -> dict[str, str]:
    return {
        "model": hand.model,
        "side": hand.side,
        "interface": hand.interface,
        "interface_type": hand.interface_type,
    }


def safe_filename_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    token = token.strip("_")
    return token or "unknown_hand"


def fault_code_value(code: Any) -> str:
    value = getattr(code, "value", code)
    try:
        return str(int(value))
    except Exception:
        return str(value)


def fault_code_has_fault(code: Any) -> bool:
    if hasattr(code, "has_fault"):
        return bool(code.has_fault())
    try:
        return int(getattr(code, "value", code)) != 0
    except Exception:
        return bool(code)


def fault_code_names(code: Any) -> list[str]:
    if hasattr(code, "get_fault_names"):
        return [str(name) for name in code.get_fault_names()]
    return [str(code)]


def append_fault_report_rows(path: str, rows: list[dict[str, str]]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FAULT_REPORT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FAULT_REPORT_COLUMNS})


class DotMatrixWidget(QWidget):
    """Small touch heatmap widget compatible with the old GUI layout."""

    RAW_VALUE_COLOR_MAX = 255.0
    HEATMAP_STOPS = (
        (0.0, (0, 0, 255)),
        (0.25, (0, 255, 255)),
        (0.50, (0, 255, 0)),
        (0.75, (255, 255, 0)),
        (1.0, (255, 0, 0)),
    )

    def __init__(self, parent: QWidget | None = None, rows: int = 12, cols: int = 6, cell_px: int = 12):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.cell_px = cell_px
        self.data = np.zeros((rows, cols), dtype=float)
        self.setMinimumSize(cols * cell_px + 4, rows * cell_px + 4)
        self.setMaximumSize(cols * cell_px + 4, rows * cell_px + 4)

    def set_data(self, data: Any) -> None:
        self.data = self._normalize_matrix(data)
        self.update()

    def _heatmap_color(self, value: float) -> QColor:
        for index in range(1, len(self.HEATMAP_STOPS)):
            low_stop, low_rgb = self.HEATMAP_STOPS[index - 1]
            high_stop, high_rgb = self.HEATMAP_STOPS[index]
            if value <= high_stop:
                span = high_stop - low_stop
                t = 0.0 if span <= 0 else (value - low_stop) / span
                red = int(low_rgb[0] + (high_rgb[0] - low_rgb[0]) * t)
                green = int(low_rgb[1] + (high_rgb[1] - low_rgb[1]) * t)
                blue = int(low_rgb[2] + (high_rgb[2] - low_rgb[2]) * t)
                return QColor(red, green, blue)
        return QColor(*self.HEATMAP_STOPS[-1][1])

    def _normalize_matrix(self, data: Any) -> np.ndarray:
        if data is None:
            return np.zeros((self.rows, self.cols), dtype=float)
        try:
            arr = np.asarray(data, dtype=float)
        except Exception:
            return np.zeros((self.rows, self.cols), dtype=float)
        if arr.ndim == 1:
            flat = np.zeros(self.rows * self.cols, dtype=float)
            usable = min(flat.size, arr.size)
            if usable:
                flat[:usable] = arr.flatten()[:usable]
            return flat.reshape((self.rows, self.cols))
        out = np.zeros((self.rows, self.cols), dtype=float)
        rows = min(self.rows, arr.shape[0])
        cols = min(self.cols, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
        return out

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        for row in range(self.rows):
            for col in range(self.cols):
                raw_value = float(self.data[row, col])
                value = raw_value / self.RAW_VALUE_COLOR_MAX
                value = max(0.0, min(1.0, value))
                if raw_value <= 0:
                    color = QColor("#C8C8C8")
                else:
                    color = self._heatmap_color(value)
                x = 2 + col * self.cell_px
                y = 2 + row * self.cell_px
                painter.fillRect(x, y, self.cell_px - 1, self.cell_px - 1, color)


class MatrixDisplayWidget(QWidget):
    """Finger touch heatmap panel."""

    def __init__(self, parent: QWidget | None = None, rows: int = 12, cols: int = 6, cell_px: int = 12):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.cell_px = cell_px
        self.finger_matrices: dict[str, DotMatrixWidget] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)

        row_layout = QHBoxLayout()
        for display_name, key in (
            ("Thumb", "thumb_matrix"),
            ("Index", "index_matrix"),
            ("Middle", "middle_matrix"),
            ("Ring", "ring_matrix"),
            ("Pinky", "pinky_matrix"),
        ):
            row_layout.addWidget(self._create_finger_frame(display_name, key))
        row_layout.addStretch()

        main_layout.addLayout(row_layout)
        main_layout.addStretch()

    def _create_finger_frame(self, display_name: str, key: str) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)
        label = QLabel(display_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold;")
        matrix = DotMatrixWidget(rows=self.rows, cols=self.cols, cell_px=self.cell_px)
        layout.addWidget(label)
        layout.addWidget(matrix, 0, Qt.AlignCenter)
        self.finger_matrices[key] = matrix
        return frame

    def update_matrix_data(self, key: str, data: Any) -> None:
        if key == "little_matrix":
            key = "pinky_matrix"
        matrix = self.finger_matrices.get(key)
        if matrix is not None:
            matrix.set_data(data)


class ConnectionDialog(QDialog):
    """Startup selector for model, side, and CAN interface."""

    def __init__(self, defaults: dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Connect RealHand")
        self.setModal(True)
        layout = QVBoxLayout(self)

        form = QGridLayout()
        form.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CLASSES))
        self.model_combo.setCurrentText(defaults["model"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        form.addWidget(self.model_combo, 0, 1)

        form.addWidget(QLabel("Side:"), 1, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItems(["left", "right"])
        self.side_combo.setCurrentText(defaults["side"])
        form.addWidget(self.side_combo, 1, 1)

        form.addWidget(QLabel("CAN interface:"), 2, 0)
        self.interface_edit = QLineEdit(defaults["interface"])
        form.addWidget(self.interface_edit, 2, 1)

        form.addWidget(QLabel("Backend:"), 3, 0)
        self.interface_type_combo = QComboBox()
        self.interface_type_combo.addItems(["socketcan", "pcan", "virtual", "libcanbus"])
        self.interface_type_combo.setCurrentText(defaults["interface_type"])
        form.addWidget(self.interface_type_combo, 3, 1)

        form.addWidget(QLabel("Sensor read mode:"), 4, 0)
        self.sensor_read_mode_combo = QComboBox()
        self.sensor_read_mode_combo.addItems(list(SENSOR_READ_MODES))
        self.sensor_read_mode_combo.setCurrentText(
            normalize_sensor_read_mode(
                defaults.get("sensor_read_mode", DEFAULT_SENSOR_READ_MODE)
            )
        )
        form.addWidget(self.sensor_read_mode_combo, 4, 1)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_model_changed(self, model: str) -> None:
        if model == "L30":
            self.interface_type_combo.setCurrentText("libcanbus")
            if self.interface_edit.text().strip().lower() == "can0":
                self.interface_edit.setText("0")

    def selection(self) -> dict[str, str]:
        return {
            "model": self.model_combo.currentText(),
            "side": self.side_combo.currentText(),
            "interface": self.interface_edit.text().strip() or "can0",
            "interface_type": self.interface_type_combo.currentText(),
            "sensor_read_mode": self.sensor_read_mode_combo.currentText(),
        }


class HandSdkManager(QObject):
    """Qt-safe wrapper around the new RealHand SDK hand classes."""

    status_updated = pyqtSignal(str, str)
    matrix_data_updated = pyqtSignal(dict)
    sensor_values_updated = pyqtSignal(dict)
    angle_sample_observed = pyqtSignal(object, object, object)

    def __init__(
        self,
        model: str,
        side: str,
        interface_name: str,
        interface_type: str,
        poll_intervals: dict[str, float] | None = None,
        sensor_read_mode: str = DEFAULT_SENSOR_READ_MODE,
    ):
        super().__init__()
        sensor_read_mode = normalize_sensor_read_mode(sensor_read_mode)
        if sensor_read_mode not in SENSOR_READ_MODES:
            raise ValueError(
                f"sensor_read_mode must be one of {', '.join(SENSOR_READ_MODES)}"
            )
        self.model = model
        self.side = side
        self.interface_name = interface_name
        self.interface_type = interface_type
        self.poll_intervals = poll_intervals or POLL_INTERVALS_BY_NAME
        self.sensor_read_mode = sensor_read_mode
        self.hand = None
        self.device_info: Any | None = None
        self.firmware_version = "Unavailable"
        self.mechanical_version = "Unavailable"
        self.serial_number = "Unavailable"
        self.joint_limits: list[tuple[int, int]] | None = None
        self.initial_positions: list[int] | None = None
        self.speed_limits: tuple[int, int] = (0, 100)
        self.torque_limits: tuple[int, int] = (0, 100)
        # User-facing percentage.  The L30 protocol stores this in tenths of a percent.
        self.torque_limit_bounds: tuple[int, int] = (0, 100)
        self.torque_limit_supported = False
        self.acceleration_bounds: tuple[int, int] = (0, 100)
        self.acceleration_supported = False
        self.motion_enabled = model != "L30"
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._last_reader_error: str | None = None
        self._last_positions: tuple[int, ...] | None = None
        self._angle_sample_lock = threading.Lock()
        self._latest_angle_sample: tuple[list[Any], int, float] | None = None
        self._connect()

    @property
    def joint_count(self) -> int:
        return len(HAND_CONFIGS[self.model].joint_names)

    def _connect(self) -> None:
        hand_class = MODEL_CLASSES[self.model]
        self.status_updated.emit(
            "info",
            f"Connecting {self.side} {self.model} on {self.interface_name} ({self.interface_type})",
        )
        if self.model == "L30":
            if self.interface_type != "libcanbus":
                raise ValueError("L30 requires the libcanbus backend for the supplied CANFD analyser")
            channel = self.interface_name.strip().lower()
            canfd_id = 0 if channel in ("", "can0", "0") else int(channel)
            self.hand = hand_class(side=self.side, canfd_id=canfd_id, interface_type="libcanbus")
            self.joint_limits = list(self.hand._position_ranges())
            self.speed_limits = tuple(self.hand._speed_bounds)
            self.torque_limits = tuple(self.hand._torque_bounds)
            self.torque_limit_supported = self.hand.torque_limit.supported
            self.acceleration_supported = self.hand.acceleration.supported
            if self.acceleration_supported:
                self.acceleration_bounds = (
                    self.hand.acceleration.MINIMUM,
                    self.hand.acceleration.MAXIMUM,
                )
            self.initial_positions = self.hand.position.get()
        else:
            self.hand = hand_class(
                side=self.side,
                interface_name=self.interface_name,
                interface_type=self.interface_type,
            )
        self._read_device_info()
        self._start_sensor_reader()
        protocol_text = (
            f" ({self.hand.protocol_version})"
            if self.model == "L30" else ""
        )
        self.status_updated.emit("info", f"Hand SDK connected: {self.side} {self.model}{protocol_text}")

    def _read_device_info(self) -> None:
        if self.hand is None:
            return
        try:
            if hasattr(self.hand, "stop_polling"):
                self.hand.stop_polling()
            if self.model == "L30":
                self.device_info = self.hand.info.get()
                self.firmware_version = str(self.device_info.software_version or "Unavailable")
                self.mechanical_version = str(self.device_info.mechanical_version or "Unavailable")
                self.serial_number = str(self.device_info.serial_number or "Unavailable")
            elif hasattr(self.hand, "version"):
                self.device_info = self.hand.version.get_device_info()
                self.firmware_version = str(self.device_info.firmware_version)
                self.mechanical_version = str(self.device_info.mechanical_version)
                self.serial_number = self.device_info.serial_number
            else:
                return
        except Exception as exc:
            self.device_info = None
            self.firmware_version = "Unavailable"
            self.mechanical_version = "Unavailable"
            self.serial_number = "Unavailable"
            message = f"Device info unavailable: {exc}"
            print(message)
            self.status_updated.emit("warning", message)
            return

        message = (
            f"Firmware version: {self.firmware_version}, "
            f"mechanical version: {self.mechanical_version}, "
            f"serial number: {self.serial_number}"
        )
        print(message)
        self.status_updated.emit("info", message)

    def _supported_sensor_sources(self) -> list[Any]:
        events_module = importlib.import_module(EVENT_MODULES[self.model])
        sensor_source = getattr(events_module, "SensorSource")
        return list(sensor_source)

    def _start_supported_polling(self) -> None:
        if self.hand is None:
            return
        intervals = {}
        for source in self._supported_sensor_sources():
            interval = self.poll_intervals.get(source.value)
            if interval is not None:
                intervals[source] = interval
        if intervals:
            self.hand.start_polling(intervals)

    def _start_sensor_reader(self) -> None:
        self._stop_sensor_readers()
        self._last_reader_error = None
        if self.hand is None:
            return
        if self.sensor_read_mode == "stream":
            self._start_supported_polling()
            self._start_stream_thread()
        elif self.sensor_read_mode == "snapshot":
            self._start_supported_polling()
            self._start_snapshot_thread()
        elif self.sensor_read_mode == "get_blocking":
            self._start_blocking_thread()
        self.status_updated.emit("info", f"Sensor read mode: {self.sensor_read_mode}")

    def set_sensor_read_mode(self, mode: str) -> None:
        mode = normalize_sensor_read_mode(mode)
        if mode not in SENSOR_READ_MODES:
            self.status_updated.emit("error", f"Unsupported sensor read mode: {mode}")
            return
        if mode == self.sensor_read_mode:
            return
        self.sensor_read_mode = mode
        try:
            self._start_sensor_reader()
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to switch sensor read mode: {exc}")

    def _stop_sensor_readers(self) -> None:
        self._stream_stop.set()
        if self.hand is not None:
            try:
                self.hand.stop_stream()
            except Exception:
                pass
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=STREAM_JOIN_TIMEOUT_S)
        self._stream_thread = None

        self._reader_stop.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=STREAM_JOIN_TIMEOUT_S)
        self._reader_thread = None

        if self.hand is not None:
            try:
                self.hand.stop_polling()
            except Exception:
                pass

    def _start_stream_thread(self) -> None:
        if self.hand is None:
            return
        queue = self.hand.stream(maxsize=300)
        self._stream_stop.clear()

        def run() -> None:
            try:
                for event in queue:
                    if self._stream_stop.is_set():
                        break
                    self._handle_event(event)
            except Exception as exc:
                if not self._stream_stop.is_set():
                    self.status_updated.emit("error", f"Stream stopped: {exc}")

        self._stream_thread = threading.Thread(
            target=run,
            name=f"{self.model}-GUI-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def _start_snapshot_thread(self) -> None:
        if self.hand is None:
            return
        self._reader_stop.clear()

        def run() -> None:
            while not self._reader_stop.is_set():
                try:
                    if self.hand is None:
                        break
                    self._handle_snapshot(self.hand.get_snapshot())
                except Exception as exc:
                    self._report_reader_error("Snapshot sensor read failed", exc)
                self._reader_stop.wait(self._snapshot_read_interval())

        self._reader_thread = threading.Thread(
            target=run,
            name=f"{self.model}-GUI-snapshot",
            daemon=True,
        )
        self._reader_thread.start()

    def _start_blocking_thread(self) -> None:
        if self.hand is None:
            return
        self._reader_stop.clear()
        source_names = [
            source.value
            for source in self._supported_sensor_sources()
            if source.value in self.poll_intervals
            and source.value in {"angle", "torque", "temperature", "current", "force_sensor"}
        ]
        if "force_sensor" not in source_names:
            force_sensor = getattr(self.hand, "force_sensor", None)
            if force_sensor is not None and hasattr(force_sensor, "get_blocking"):
                source_names.append("force_sensor")
        if not source_names:
            self.status_updated.emit("warning", "No get_blocking sensors available")
            return
        next_due = {name: 0.0 for name in source_names}

        def run() -> None:
            while not self._reader_stop.is_set():
                now = time.monotonic()
                for source_name in source_names:
                    if self._reader_stop.is_set():
                        break
                    if now < next_due[source_name]:
                        continue
                    next_due[source_name] = now + self.poll_intervals[source_name]
                    self._read_blocking_source(source_name)
                wait_s = self._next_blocking_wait(next_due)
                self._reader_stop.wait(wait_s)

        self._reader_thread = threading.Thread(
            target=run,
            name=f"{self.model}-GUI-blocking",
            daemon=True,
        )
        self._reader_thread.start()

    def _handle_event(self, event: Any) -> None:
        name = type(event).__name__
        data = getattr(event, "data", None)
        if name == "ForceSensorEvent":
            matrix = self._matrix_payload(data)
            if matrix:
                self.matrix_data_updated.emit(matrix)
            return

        self._emit_sensor_payload(data)

    def _handle_snapshot(self, snapshot: Any) -> None:
        if snapshot is None:
            return
        force_data = getattr(snapshot, "force_sensor", None)
        matrix = self._matrix_payload(force_data)
        if matrix:
            self.matrix_data_updated.emit(matrix)
        for attr in ("angle", "position", "torque", "temperature", "current", "speed", "acceleration", "fault"):
            self._emit_sensor_payload(getattr(snapshot, attr, None))

    def _emit_sensor_payload(self, data: Any) -> None:
        if data is None:
            return
        payload = {}
        sample_perf_ns = time.perf_counter_ns()
        fields = (*SENSOR_DATA_FIELDS, ("positions", "angle"), ("error_codes", "fault"))
        for attr, key in fields:
            value = getattr(data, attr, None)
            if value is not None:
                values = self._to_list(value)
                payload[key] = values
                if key == "angle":
                    sample_wall_time = float(getattr(data, "timestamp", time.time()))
                    self._record_latest_angle_sample(values, sample_perf_ns, sample_wall_time)
                    self.angle_sample_observed.emit(values, sample_perf_ns, sample_wall_time)
        if payload:
            self.sensor_values_updated.emit(payload)

    def _record_latest_angle_sample(
        self,
        values: list[Any],
        sample_perf_ns: int,
        sample_wall_time: float,
    ) -> None:
        with self._angle_sample_lock:
            self._latest_angle_sample = (list(values), sample_perf_ns, sample_wall_time)

    def latest_angle_sample(self) -> tuple[list[Any], int, float] | None:
        with self._angle_sample_lock:
            if self._latest_angle_sample is None:
                return None
            values, sample_perf_ns, sample_wall_time = self._latest_angle_sample
            return list(values), sample_perf_ns, sample_wall_time

    def _read_blocking_source(self, source_name: str) -> None:
        if self.hand is None:
            return
        manager = getattr(self.hand, source_name, None)
        if manager is None or not hasattr(manager, "get_blocking"):
            return
        try:
            data = manager.get_blocking(timeout_ms=self._blocking_timeout_ms(source_name))
        except Exception as exc:
            self._report_reader_error(f"Blocking {source_name} read failed", exc)
            return
        if source_name == "force_sensor":
            matrix = self._matrix_payload(data)
            if matrix:
                self.matrix_data_updated.emit(matrix)
            return
        self._emit_sensor_payload(data)

    def _snapshot_read_interval(self) -> float:
        if not self.poll_intervals:
            return SNAPSHOT_READ_INTERVAL_S
        return max(SNAPSHOT_READ_INTERVAL_S, min(self.poll_intervals.values()))

    def _next_blocking_wait(self, next_due: dict[str, float]) -> float:
        if not next_due:
            return 0.25
        wait_s = min(next_due.values()) - time.monotonic()
        return max(0.01, min(wait_s, 0.25))

    def _blocking_timeout_ms(self, source_name: str) -> int:
        return BLOCKING_TIMEOUTS_BY_NAME.get(source_name, BLOCKING_DEFAULT_TIMEOUT_MS)

    def _report_reader_error(self, prefix: str, exc: Exception) -> None:
        message = f"{prefix}: {exc}"
        if message == self._last_reader_error:
            return
        self._last_reader_error = message
        self.status_updated.emit("warning", message)

    def _matrix_payload(self, data: Any) -> dict[str, Any]:
        if data is None:
            return {}
        matrices = getattr(data, "matrices", None)
        if isinstance(matrices, dict):
            payload = {}
            for name, values in matrices.items():
                # L30's vendor API already uses e.g. ``thumb_matrix``.
                # Normalize both that form and bare ``thumb`` without adding
                # a second suffix, which would make the heatmap ignore it.
                finger = str(name).removesuffix("_matrix")
                if finger == "little":
                    finger = "pinky"
                payload[f"{finger}_matrix"] = values
            return payload
        payload = {}
        for finger, key in (
            ("thumb", "thumb_matrix"),
            ("index", "index_matrix"),
            ("middle", "middle_matrix"),
            ("ring", "ring_matrix"),
            ("pinky", "pinky_matrix"),
        ):
            finger_data = getattr(data, finger, None)
            values = getattr(finger_data, "values", None)
            if values is not None:
                payload[key] = values
        return payload

    def _to_list(self, value: Any) -> list[Any]:
        if hasattr(value, "to_list"):
            return list(value.to_list())
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _joint_command_values(self, positions: list[int]) -> list[float]:
        if self.model == "L30":
            return [int(position) for position in positions]
        joint_names = HAND_CONFIGS[self.model].joint_names
        values = []
        for idx, position in enumerate(positions):
            if idx < len(joint_names) and self._is_non_thumb_abduction(joint_names[idx]):
                position = MAX_JOINT_POSITION - position
            values.append(float(position))
        return values

    @staticmethod
    def _is_non_thumb_abduction(joint_name: str) -> bool:
        lowered = joint_name.lower()
        return "abduction" in lowered and "thumb" not in lowered

    def publish_joint_state(self, positions: list[int], *, force: bool = False) -> dict[str, Any] | None:
        if self.hand is None:
            self.status_updated.emit("error", "Hand SDK is not connected")
            return None
        values = self._joint_command_values(positions)
        key = tuple(int(v) for v in values)
        if not force and key == self._last_positions:
            return {
                "sent": False,
                "skipped": "duplicate",
                "target_values": values,
                "command_key": key,
            }
        baseline_sample = self.latest_angle_sample()
        send_start_ns = time.perf_counter_ns()
        send_start_wall_time = time.time()
        if self.model == "L30" and not self.motion_enabled:
            return {"sent": False, "skipped": "motion_disabled", "target_values": values, "command_key": key}
        try:
            if self.model == "L30":
                self.hand.position.set(values)
            else:
                self.hand.angle.set_angles(values)
            send_end_ns = time.perf_counter_ns()
            self._last_positions = key
            self.status_updated.emit("info", f"Joint state sent: {key}")
            return {
                "sent": True,
                "target_values": values,
                "command_key": key,
                "baseline_sample": baseline_sample,
                "send_start_ns": send_start_ns,
                "send_end_ns": send_end_ns,
                "send_start_wall_time": send_start_wall_time,
            }
        except Exception as exc:
            self.status_updated.emit("error", f"Send failed: {exc}")
            return {
                "sent": False,
                "error": str(exc),
                "target_values": values,
                "command_key": key,
                "baseline_sample": baseline_sample,
                "send_start_ns": send_start_ns,
                "send_end_ns": time.perf_counter_ns(),
                "send_start_wall_time": send_start_wall_time,
            }

    def publish_speed(self, value: int, *, log_success: bool = True) -> None:
        if self.hand is None or not hasattr(self.hand, "speed"):
            self.status_updated.emit("error", "Speed control is not available")
            return
        try:
            if self.model == "L30":
                self.hand.speed.set([int(value)] * self.joint_count)
            else:
                self.hand.speed.set_speeds([float(value)] * self.joint_count)
            if log_success:
                self.status_updated.emit("info", f"Speed set to {value}")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set speed: {exc}")

    def publish_joint_speeds(self, values: list[int], *, log_success: bool = True) -> None:
        if self.hand is None or not hasattr(self.hand, "speed"):
            self.status_updated.emit("error", "Speed control is not available")
            return
        if len(values) != self.joint_count:
            self.status_updated.emit(
                "error",
                f"Speed joint count ({len(values)}) does not match current joint count ({self.joint_count})",
            )
            return
        try:
            speeds = [int(value) for value in values] if self.model == "L30" else [float(value) for value in values]
            if self.model == "L30":
                self.hand.speed.set(speeds)
            else:
                self.hand.speed.set_speeds(speeds)
            if log_success:
                self.status_updated.emit("info", f"Per-joint speeds set: {self._format_joint_values(values)}")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set per-joint speeds: {exc}")

    def publish_torque(self, value: int, *, log_success: bool = True) -> None:
        if self.hand is None or not hasattr(self.hand, "torque"):
            self.status_updated.emit("error", "Torque control is not available")
            return
        try:
            if self.model == "L30":
                self.hand.torque.set_commanded([int(value)] * self.joint_count)
            else:
                self.hand.torque.set_torques([float(value)] * self.joint_count)
            if log_success:
                self.status_updated.emit("info", f"Torque set to {value}")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set torque: {exc}")

    def publish_joint_torques(self, values: list[int], *, log_success: bool = True) -> None:
        if self.hand is None or not hasattr(self.hand, "torque"):
            self.status_updated.emit("error", "Torque control is not available")
            return
        if len(values) != self.joint_count:
            self.status_updated.emit(
                "error",
                f"Torque joint count ({len(values)}) does not match current joint count ({self.joint_count})",
            )
            return
        try:
            torques = [int(value) for value in values] if self.model == "L30" else [float(value) for value in values]
            if self.model == "L30":
                self.hand.torque.set_commanded(torques)
            else:
                self.hand.torque.set_torques(torques)
            if log_success:
                self.status_updated.emit("info", f"Per-joint torques set: {self._format_joint_values(values)}")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set per-joint torques: {exc}")

    def publish_torque_limit(self, value: int, *, log_success: bool = True) -> None:
        if self.model != "L30" or self.hand is None or not self.torque_limit_supported:
            self.status_updated.emit("error", "Torque-limit control is not available for this hand")
            return
        try:
            # The vendor command uses 0..1000, where each unit is 0.1%.
            protocol_value = int(value) * 10
            self.hand.torque_limit.set([protocol_value] * self.joint_count)
            if log_success:
                self.status_updated.emit("info", f"Torque limit set to {value}%")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set torque limit: {exc}")

    def publish_acceleration(self, value: int, *, log_success: bool = True) -> None:
        """Set the same acceleration on every supported motor."""
        if self.hand is None or not hasattr(self.hand, "acceleration"):
            self.status_updated.emit("error", "Acceleration control is not available")
            return
        try:
            if self.model == "L30":
                if not self.acceleration_supported:
                    raise RuntimeError("L30 acceleration is unavailable for this protocol")
                self.hand.acceleration.set([int(value)] * self.joint_count)
            else:
                self.hand.acceleration.set_accelerations([float(value)] * self.joint_count)
            if log_success:
                self.status_updated.emit("info", f"Acceleration set to {value}")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to set acceleration: {exc}")

    def _format_joint_values(self, values: list[int]) -> str:
        joint_names = HAND_CONFIGS[self.model].joint_names
        return ", ".join(f"{name}={value}" for name, value in zip(joint_names, values))

    def clear_faults(self) -> None:
        if self.hand is None or not hasattr(self.hand, "fault"):
            return
        if self.model == "L30":
            self.status_updated.emit("warning", "L30 firmware does not expose a fault-clear command through this SDK")
            return
        try:
            self.hand.fault.clear_faults()
            self.status_updated.emit("info", "Fault clear command sent")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to clear faults: {exc}")

    def read_fault_data(self, timeout_ms: int) -> Any:
        if self.hand is None or not hasattr(self.hand, "fault"):
            raise RuntimeError("Fault status is not available for this hand")
        if self.model == "L30":
            return self.hand.fault.get()
        if not hasattr(self.hand.fault, "get_blocking"):
            raise RuntimeError("Blocking fault read is not available for this hand")

        try:
            self._stop_sensor_readers()
            return self.hand.fault.get_blocking(timeout_ms=timeout_ms)
        finally:
            if self.hand is not None:
                try:
                    self._start_sensor_reader()
                except Exception as exc:
                    self.status_updated.emit("warning", f"Failed to restart sensor reader after fault check: {exc}")

    def shutdown(self) -> None:
        self._stop_sensor_readers()
        if self.hand is not None:
            try:
                self.hand.close()
                self.status_updated.emit("info", "Hand SDK connection closed")
            except Exception as exc:
                self.status_updated.emit("error", f"Failed to close hand: {exc}")
            self.hand = None

    def enable_motion(
        self,
        *,
        speed: int,
        commanded_current: int,
        torque_limit_percent: int | None,
        acceleration: int | None,
    ) -> None:
        if self.model != "L30" or self.hand is None:
            return
        try:
            # Program every displayed setting before enabling, so retained device
            # settings cannot take effect when the motors become enabled.
            speed = max(self.speed_limits[0], min(int(speed), self.speed_limits[1]))
            self.hand.speed.set([speed] * self.joint_count)
            if acceleration is not None and self.acceleration_supported:
                self.hand.acceleration.set([int(acceleration)] * self.joint_count)
            if torque_limit_percent is not None and self.torque_limit_supported:
                protocol_limit = max(0, min(100, int(torque_limit_percent))) * 10
                self.hand.torque_limit.set([protocol_limit] * self.joint_count)
            current = max(self.torque_limits[0], min(int(commanded_current), self.torque_limits[1]))
            self.hand.torque.set_commanded([current] * self.joint_count)
            self.hand.enable_all()
            self.motion_enabled = True
            self.status_updated.emit(
                "warning",
                f"L30 motion enabled: speed={speed}, current={current}; commands now use native 17-joint values",
            )
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to enable L30 motion: {exc}")

    def disable_motion(self) -> None:
        if self.model != "L30" or self.hand is None:
            return
        try:
            self.hand.disable_all()
            self.motion_enabled = False
            self.status_updated.emit("warning", "L30 motion disabled")
        except Exception as exc:
            self.status_updated.emit("error", f"Failed to disable L30 motion: {exc}")


class HandControlGUI(QWidget):
    """Dexterous hand control interface with the old GUI's visual style."""

    status_updated = pyqtSignal(str, str)

    def __init__(self, sdk_manager: HandSdkManager):
        super().__init__()
        self.sdk_manager = sdk_manager
        self.sdk_manager.status_updated.connect(self.update_status)
        self.sdk_manager.matrix_data_updated.connect(self.update_matrix_display)
        self.sdk_manager.sensor_values_updated.connect(self.update_sensor_values)
        self.sdk_manager.angle_sample_observed.connect(self.update_motion_timing_from_angles)
        self.status_updated = self.sdk_manager.status_updated

        self.model = sdk_manager.model
        self.side = sdk_manager.side
        self.hand_config = HAND_CONFIGS[self.model]
        self.cycle_timer: QTimer | None = None
        self.current_action_index = -1
        self.cycle_loop_active = False
        self.cycle_loop_index = -1
        self.cycle_loop_iterations = 0
        self.preset_buttons: list[QPushButton] = []
        self.latest_sensor_values: dict[str, list[Any]] = {}
        self.finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        self.per_joint_speed_sliders: list[QSlider] = []
        self.per_joint_torque_sliders: list[QSlider] = []
        self.latest_touch_matrices: dict[str, Any] = {}
        self.current_torque_values: list[int] = []
        self.auto_grab_running = False
        self.auto_grab_fingers_stopped = {finger: False for finger in self.finger_order}
        self.auto_grab_baseline = {finger: 0 for finger in self.finger_order}
        self.auto_grab_above_count = {finger: 0 for finger in self.finger_order}
        self.auto_grab_debounce_limit = 3
        self.auto_grab_sensor_fail_count = 0
        self.auto_grab_started_at = 0.0
        self.prev_touch: dict[str, list[int]] = {}
        self.touch_history: dict[str, list[list[int]]] = {}
        self.last_slip_time: dict[str, int] = {}
        self.closed_threshold = 10
        self._syncing_per_joint_settings = False
        self.motion_test_active = False
        self.motion_test_session_id = ""
        self.motion_test_records: list[MotionTimingRecord] = []
        self.motion_test_command_index = 0
        self.motion_test_log_path = ""

        self.live_speed_timer = QTimer(self)
        self.live_speed_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_speed_timer.setSingleShot(True)
        self.live_speed_timer.timeout.connect(self._publish_live_speed)

        self.live_torque_timer = QTimer(self)
        self.live_torque_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_torque_timer.setSingleShot(True)
        self.live_torque_timer.timeout.connect(self._publish_live_torque)

        self.live_torque_limit_timer = QTimer(self)
        self.live_torque_limit_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_torque_limit_timer.setSingleShot(True)
        self.live_torque_limit_timer.timeout.connect(self._publish_live_torque_limit)

        self.live_acceleration_timer = QTimer(self)
        self.live_acceleration_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_acceleration_timer.setSingleShot(True)
        self.live_acceleration_timer.timeout.connect(self._publish_live_acceleration)

        self.live_joint_speed_timer = QTimer(self)
        self.live_joint_speed_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_joint_speed_timer.setSingleShot(True)
        self.live_joint_speed_timer.timeout.connect(self._publish_live_joint_speeds)

        self.live_joint_torque_timer = QTimer(self)
        self.live_joint_torque_timer.setInterval(LIVE_COMMAND_INTERVAL_MS)
        self.live_joint_torque_timer.setSingleShot(True)
        self.live_joint_torque_timer.timeout.connect(self._publish_live_joint_torques)

        self.init_ui()
        self.current_torque_values = [self.torque_slider.value()] * len(self.hand_config.joint_names)
        self._update_commanded_torque_display()

        self.auto_grab_timer = QTimer(self)
        self.auto_grab_timer.setInterval(50)
        self.auto_grab_timer.timeout.connect(self._auto_grab_step)

        self.publish_timer = QTimer(self)
        self.publish_timer.setInterval(PUBLISH_INTERVAL_MS)
        self.publish_timer.timeout.connect(self.publish_joint_state)
        self.publish_timer.start()

    def init_ui(self) -> None:
        self.setWindowTitle(f"Realhand Dexterous Hand Control Interface - {self.side} {self.model}")
        self.setMinimumSize(1200, 900)
        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 6px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #165DFF;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                border-radius: 4px;
                background: #CCCCCC;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #165DFF, stop:1 #0E42D2);
                border: 1px solid #5C8AFF;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #F0F0F0; }
            QPushButton:pressed { background-color: #D0D0D0; }
            QPushButton[category="preset"] {
                background-color: #E6F7FF;
                color: #1890FF;
                border-color: #91D5FF;
            }
            QPushButton[category="preset"]:hover { background-color: #B3E0FF; }
            QPushButton[category="action"] {
                background-color: #FFF7E6;
                color: #FA8C16;
                border-color: #FFD591;
            }
            QPushButton[category="danger"] {
                background-color: #FFF1F0;
                color: #F5222D;
                border-color: #FFCCC7;
            }
            QLabel#StatusLabel {
                padding: 5px;
                border-radius: 4px;
            }
            QLabel#StatusInfo {
                background-color: #F0F7FF;
                color: #0066CC;
            }
            QLabel#StatusError {
                background-color: #FFF0F0;
                color: #CC0000;
            }
            QTextEdit#ValueDisplay {
                background-color: #F8F8F8;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            """
        )

        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_joint_control_panel())
        splitter.addWidget(self.create_preset_actions_panel())
        splitter.addWidget(self.create_status_monitor_panel())
        splitter.setSizes([320, 520, 360])
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addWidget(self.create_value_display_panel(), stretch=0)
        self.update_value_display()

    def create_joint_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel(f"Joint Control - {self.model}")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        layout.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        self.sliders_layout = QGridLayout(scroll_content)
        self.sliders_layout.setSpacing(10)
        self.create_joint_sliders()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        return panel

    def create_joint_sliders(self) -> None:
        self.sliders: list[QSlider] = []
        self.slider_labels: list[QLabel] = []
        initial_positions = self.sdk_manager.initial_positions or self.hand_config.init_pos
        for row, (name, value) in enumerate(zip(self.hand_config.joint_names, initial_positions)):
            label = QLabel(f"{name}: {value}")
            label.setMinimumWidth(150)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*self._joint_slider_bounds(row))
            slider.setValue(int(value))
            slider.valueChanged.connect(lambda val, idx=row: self.on_slider_value_changed(idx, val))
            self.sliders_layout.addWidget(label, row, 0)
            self.sliders_layout.addWidget(slider, row, 1)
            self.slider_labels.append(label)
            self.sliders.append(slider)

    def _joint_slider_bounds(self, index: int) -> tuple[int, int]:
        if self.model == "L30" and self.sdk_manager.joint_limits and index < len(self.sdk_manager.joint_limits):
            return self.sdk_manager.joint_limits[index]
        return (0, MAX_JOINT_POSITION)

    def _setting_slider_bounds(self, kind: str) -> tuple[int, int]:
        if self.model == "L30":
            if kind == "speed":
                return self.sdk_manager.speed_limits
            if kind == "acceleration":
                return self.sdk_manager.acceleration_bounds
            if kind == "torque_limit":
                return self.sdk_manager.torque_limit_bounds
            return self.sdk_manager.torque_limits
        return (0, MAX_JOINT_POSITION)

    def _setting_slider_default(self, kind: str) -> int:
        """Return the established displayed default without sending a command."""
        minimum, maximum = self._setting_slider_bounds(kind)
        if self.model == "L30" and kind in {"speed", "torque", "torque_limit", "acceleration"}:
            return max(minimum, min(round(maximum * 0.85), maximum))
        # Preserve the original GUI behavior for existing hands: their global
        # and per-joint setting sliders begin at the maximum value. L30 uses
        # the explicit 85% initialization above instead.
        return maximum

    def create_preset_actions_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        matrix_group = QGroupBox("Finger Touch Heatmap")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_display = MatrixDisplayWidget(rows=12, cols=6, cell_px=12)
        self.matrix_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        matrix_layout.addWidget(self.matrix_display)
        layout.addWidget(matrix_group, stretch=0)

        preset_group = QGroupBox("System Presets")
        preset_layout = QGridLayout(preset_group)
        preset_layout.setSpacing(8)
        self.create_system_preset_buttons(preset_layout)
        preset_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(preset_group, stretch=1)

        actions = QHBoxLayout()
        self.cycle_button = QPushButton("Cycle Preset Actions")
        self.cycle_button.setProperty("category", "action")
        self.cycle_button.clicked.connect(self.on_cycle_clicked)
        actions.addWidget(self.cycle_button)

        self.home_button = QPushButton("Return to Home")
        self.home_button.setProperty("category", "action")
        self.home_button.clicked.connect(self.on_home_clicked)
        actions.addWidget(self.home_button)

        self.stop_button = QPushButton("Stop All Actions")
        self.stop_button.setProperty("category", "danger")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        actions.addWidget(self.stop_button)
        if self.model == "L30":
            self.enable_motion_button = QPushButton("Enable L30 Motion")
            self.enable_motion_button.setProperty("category", "danger")
            self.enable_motion_button.clicked.connect(self.on_enable_l30_motion_clicked)
            actions.addWidget(self.enable_motion_button)
        layout.addLayout(actions)

        return panel

    def create_system_preset_buttons(self, layout: QGridLayout) -> None:
        self.preset_buttons.clear()
        for idx, (name, positions) in enumerate(self.hand_config.preset_actions.items()):
            button = QPushButton(name)
            button.setProperty("category", "preset")
            button.clicked.connect(
                lambda checked, pos=positions, preset_name=name: self.on_preset_action_clicked(
                    pos,
                    preset_name=preset_name,
                )
            )
            row, col = divmod(idx, 2)
            layout.addWidget(button, row, col)
            self.preset_buttons.append(button)

    def create_status_monitor_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Status Monitor")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        layout.addWidget(title)

        quick_group = QGroupBox("Quick Settings")
        quick_layout = QVBoxLayout(quick_group)
        quick_layout.addLayout(self._sensor_read_mode_row())
        quick_layout.addLayout(self._live_slider_row("Speed:", "speed"))
        quick_layout.addLayout(self._live_slider_row("Commanded Current:", "torque"))
        if self.model == "L30" and self.sdk_manager.torque_limit_supported:
            quick_layout.addLayout(self._live_slider_row("Torque Limit:", "torque_limit"))
        if self.model == "O6" or (self.model == "L30" and self.sdk_manager.acceleration_supported):
            quick_layout.addLayout(self._live_slider_row("Acceleration:", "acceleration"))
        layout.addWidget(quick_group)

        realtime_group = QGroupBox("Realtime Sensors (Per Finger)")
        realtime_layout = QGridLayout(realtime_group)
        sensor_keys = ["angle", "torque", "temperature", "current"]
        headers = ["Finger", "Angle", "Torque", "Temp", "Current"]
        if self.model == "O6" or (self.model == "L30" and self.sdk_manager.acceleration_supported):
            sensor_keys.append("acceleration")
            headers.append("Accel")
        for col, text in enumerate(headers):
            realtime_layout.addWidget(QLabel(text), 0, col)
        self.realtime_labels: dict[str, dict[str, QLabel]] = {}
        titles = {
            "thumb": "Thumb",
            "index": "Index",
            "middle": "Middle",
            "ring": "Ring",
            "pinky": "Pinky",
        }
        for row, finger in enumerate(self.finger_order, start=1):
            realtime_layout.addWidget(QLabel(titles[finger]), row, 0)
            self.realtime_labels[finger] = {}
            for col, key in enumerate(sensor_keys, start=1):
                label = QLabel("--")
                realtime_layout.addWidget(label, row, col)
                self.realtime_labels[finger][key] = label
        layout.addWidget(realtime_group)

        tabs = QTabWidget()
        tabs.addTab(self._system_info_tab(), "System Info")
        if self.model in ("L20", "L30"):
            tabs.addTab(self._l20_joint_settings_tab(), f"{self.model} Joint Settings")
        tabs.addTab(self._touch_control_tab(), "Touch Control")
        tabs.addTab(self._status_log_tab(), "Status Log")
        layout.addWidget(tabs)
        return panel

    def _sensor_read_mode_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Sensor read:"))
        combo = QComboBox()
        combo.addItems(list(SENSOR_READ_MODES))
        combo.setCurrentText(self.sdk_manager.sensor_read_mode)
        combo.currentTextChanged.connect(self.on_sensor_read_mode_changed)
        self.sensor_read_mode_combo = combo
        row.addWidget(combo)
        row.addStretch()
        return row

    def _live_slider_row(self, label_text: str, kind: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(*self._setting_slider_bounds(kind))
        slider.setValue(self._setting_slider_default(kind))
        slider.setMinimumWidth(150)
        row.addWidget(slider)
        value_label = QLabel(f"{slider.value()}%" if kind == "torque_limit" else str(slider.value()))
        value_label.setMinimumWidth(30)
        row.addWidget(value_label)
        if kind == "speed":
            self.speed_slider = slider
            self.speed_val_lbl = value_label
            slider.valueChanged.connect(self.on_global_speed_changed)
        elif kind == "torque":
            self.torque_slider = slider
            self.torque_val_lbl = value_label
            slider.valueChanged.connect(self.on_global_torque_changed)
            value_label.setMinimumWidth(48)
        elif kind == "torque_limit":
            self.torque_limit_slider = slider
            self.torque_limit_val_lbl = value_label
            slider.valueChanged.connect(self.on_global_torque_limit_changed)
            value_label.setMinimumWidth(48)
        else:
            self.acceleration_slider = slider
            self.acceleration_val_lbl = value_label
            slider.valueChanged.connect(self.on_global_acceleration_changed)
        row.addStretch()
        return row

    def _l20_joint_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Per-Joint Controls")
        group.setCheckable(True)
        group.setChecked(True)
        group_layout = QVBoxLayout(group)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        control_tabs = QTabWidget()
        control_tabs.addTab(self._per_joint_setting_tab("speed"), "Speed")
        control_tabs.addTab(self._per_joint_setting_tab("torque"), "Torque")
        content_layout.addWidget(control_tabs)
        group_layout.addWidget(content)
        group.toggled.connect(content.setVisible)

        layout.addWidget(group)
        layout.addStretch()
        return widget

    def _per_joint_setting_tab(self, kind: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        grid = QGridLayout(scroll_content)
        grid.setSpacing(8)

        sliders: list[QSlider] = []
        for row, name in enumerate(self.hand_config.joint_names):
            name_label = QLabel(name)
            name_label.setMinimumWidth(115)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*self._setting_slider_bounds(kind))
            slider.setValue(self._setting_slider_default(kind))
            slider.setMinimumWidth(120)
            value_label = QLabel(str(slider.value()))
            value_label.setMinimumWidth(28)
            slider.valueChanged.connect(
                lambda val, setting=kind, label=value_label: self.on_per_joint_setting_changed(setting, label, val)
            )
            grid.addWidget(name_label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
            sliders.append(slider)

        if kind == "speed":
            self.per_joint_speed_sliders = sliders
        else:
            self.per_joint_torque_sliders = sliders

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        return widget

    def _touch_control_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        auto_group = QGroupBox("Touch Auto Grab")
        auto_layout = QVBoxLayout(auto_group)

        auto_settings = QGridLayout()
        auto_settings.addWidget(QLabel("Threshold"), 0, 0)
        self.auto_grab_threshold_spin = QSpinBox()
        self.auto_grab_threshold_spin.setRange(1, 1000)
        self.auto_grab_threshold_spin.setValue(30)
        auto_settings.addWidget(self.auto_grab_threshold_spin, 0, 1)

        auto_settings.addWidget(QLabel("Speed"), 0, 2)
        self.auto_grab_speed_spin = QSpinBox()
        self.auto_grab_speed_spin.setRange(1, 100)
        self.auto_grab_speed_spin.setValue(50)
        auto_settings.addWidget(self.auto_grab_speed_spin, 0, 3)
        auto_layout.addLayout(auto_settings)

        auto_buttons = QHBoxLayout()
        self.auto_grab_start_button = QPushButton("Auto Grab")
        self.auto_grab_start_button.setProperty("category", "action")
        self.auto_grab_start_button.clicked.connect(self.start_auto_grab)
        auto_buttons.addWidget(self.auto_grab_start_button)

        self.auto_grab_stop_button = QPushButton("Stop")
        self.auto_grab_stop_button.setProperty("category", "danger")
        self.auto_grab_stop_button.clicked.connect(self.stop_auto_grab)
        self.auto_grab_stop_button.setEnabled(False)
        auto_buttons.addWidget(self.auto_grab_stop_button)

        self.auto_grab_status_label = QLabel("Ready")
        auto_buttons.addWidget(self.auto_grab_status_label)
        auto_buttons.addStretch()
        auto_layout.addLayout(auto_buttons)

        finger_row = QHBoxLayout()
        self.auto_grab_finger_labels: dict[str, QLabel] = {}
        for finger in self.finger_order:
            label = QLabel(f"{finger[0].upper()}:--")
            label.setMinimumWidth(48)
            finger_row.addWidget(label)
            self.auto_grab_finger_labels[finger] = label
        finger_row.addStretch()
        auto_layout.addLayout(finger_row)
        layout.addWidget(auto_group)

        torque_group = QGroupBox("Commanded Torque")
        torque_layout = QVBoxLayout(torque_group)
        self.commanded_torque_summary_label = QLabel("Command: --")
        torque_layout.addWidget(self.commanded_torque_summary_label)

        torque_finger_row = QHBoxLayout()
        self.commanded_torque_finger_labels: dict[str, QLabel] = {}
        for finger in self.finger_order:
            label = QLabel(f"{finger[0].upper()}:--")
            label.setMinimumWidth(70)
            torque_finger_row.addWidget(label)
            self.commanded_torque_finger_labels[finger] = label
        torque_finger_row.addStretch()
        torque_layout.addLayout(torque_finger_row)

        self.last_torque_boost_label = QLabel("Last boost: none")
        torque_layout.addWidget(self.last_torque_boost_label)
        layout.addWidget(torque_group)

        slip_group = QGroupBox("Slip Detection")
        slip_layout = QVBoxLayout(slip_group)

        slip_settings = QGridLayout()
        slip_settings.addWidget(QLabel("Contact Max >="), 0, 0)
        self.slip_contact_spin = QSpinBox()
        self.slip_contact_spin.setRange(0, 10000)
        self.slip_contact_spin.setValue(5)
        slip_settings.addWidget(self.slip_contact_spin, 0, 1)

        slip_settings.addWidget(QLabel("Mag Delta >="), 0, 2)
        self.slip_mag_spin = QSpinBox()
        self.slip_mag_spin.setRange(0, 1000000)
        self.slip_mag_spin.setValue(50)
        slip_settings.addWidget(self.slip_mag_spin, 0, 3)

        slip_settings.addWidget(QLabel("Loc Delta >="), 1, 0)
        self.slip_loc_spin = QDoubleSpinBox()
        self.slip_loc_spin.setRange(0.0, 100.0)
        self.slip_loc_spin.setDecimals(1)
        self.slip_loc_spin.setSingleStep(0.5)
        self.slip_loc_spin.setValue(2.0)
        slip_settings.addWidget(self.slip_loc_spin, 1, 1)

        slip_settings.addWidget(QLabel("Cooldown ms"), 1, 2)
        self.slip_cooldown_spin = QSpinBox()
        self.slip_cooldown_spin.setRange(0, 60000)
        self.slip_cooldown_spin.setValue(500)
        slip_settings.addWidget(self.slip_cooldown_spin, 1, 3)
        slip_layout.addLayout(slip_settings)

        slip_options = QHBoxLayout()
        self.slip_window_checkbox = QCheckBox("Use N-frame detection")
        slip_options.addWidget(self.slip_window_checkbox)
        slip_options.addWidget(QLabel("Frames"))
        self.slip_window_frames_spin = QSpinBox()
        self.slip_window_frames_spin.setRange(2, 60)
        self.slip_window_frames_spin.setValue(5)
        slip_options.addWidget(self.slip_window_frames_spin)
        self.slip_torque_boost_checkbox = QCheckBox("Boost torque on slip")
        slip_options.addWidget(self.slip_torque_boost_checkbox)
        slip_options.addWidget(QLabel("Step"))
        self.slip_torque_boost_step_spin = QSpinBox()
        self.slip_torque_boost_step_spin.setRange(1, 100)
        self.slip_torque_boost_step_spin.setValue(5)
        slip_options.addWidget(self.slip_torque_boost_step_spin)
        slip_options.addStretch()
        slip_layout.addLayout(slip_options)

        slip_status = QHBoxLayout()
        self.slip_labels: dict[str, QLabel] = {}
        for finger in self.finger_order:
            label = QLabel(f"{finger[0].upper()}:--")
            label.setMinimumWidth(48)
            slip_status.addWidget(label)
            self.slip_labels[finger] = label
        slip_status.addStretch()
        slip_layout.addLayout(slip_status)
        layout.addWidget(slip_group)

        layout.addStretch()
        return widget

    def _system_info_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        conn_group = QGroupBox("Connection Status")
        conn_layout = QVBoxLayout(conn_group)
        self.connection_status = QLabel("Hand SDK Connected")
        self.connection_status.setObjectName("StatusLabel")
        self.connection_status.setObjectName("StatusInfo")
        conn_layout.addWidget(self.connection_status)

        info_group = QGroupBox("Hand Info")
        info_layout = QVBoxLayout(info_group)
        info = (
            f"Hand Type: {self.side}\n"
            f"Joint Model: {self.model}\n"
            f"Serial Number: {self.sdk_manager.serial_number}\n"
            f"Firmware Version: {self.sdk_manager.firmware_version}\n"
            f"Mechanical Version: {self.sdk_manager.mechanical_version}\n"
            f"CAN: {self.sdk_manager.interface_name}\n"
            f"Backend: {self.sdk_manager.interface_type}\n"
            f"Joint Count: {len(self.hand_config.joint_names)}"
        )
        info_label = QLabel(info)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)

        clear_faults_btn = QPushButton("Clear Faults")
        clear_faults_btn.clicked.connect(self.sdk_manager.clear_faults)
        info_layout.addWidget(clear_faults_btn)

        layout.addWidget(conn_group)
        layout.addWidget(info_group)
        layout.addStretch()
        return widget

    def _fault_report_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        output_group = QGroupBox("Fault Check CSV")
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel("Output folder:"), 0, 0)
        self.fault_report_dir_edit = QLineEdit(DEFAULT_FAULT_REPORT_DIR)
        output_layout.addWidget(self.fault_report_dir_edit, 0, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_fault_report_dir)
        output_layout.addWidget(browse_btn, 0, 2)

        output_layout.addWidget(QLabel("Fault timeout ms:"), 1, 0)
        self.fault_timeout_spin = QSpinBox()
        self.fault_timeout_spin.setRange(100, 5000)
        self.fault_timeout_spin.setSingleStep(100)
        self.fault_timeout_spin.setValue(1000)
        output_layout.addWidget(self.fault_timeout_spin, 1, 1)
        layout.addWidget(output_group)

        action_row = QHBoxLayout()
        self.run_fault_report_btn = QPushButton("Run Fault Check and Save CSV")
        self.run_fault_report_btn.setProperty("category", "action")
        self.run_fault_report_btn.clicked.connect(self.run_fault_report)
        action_row.addWidget(self.run_fault_report_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.fault_report_status_label = QLabel("Ready")
        self.fault_report_status_label.setWordWrap(True)
        layout.addWidget(self.fault_report_status_label)

        preview_group = QGroupBox("Last Fault Check")
        preview_layout = QVBoxLayout(preview_group)
        self.fault_report_preview = QTextEdit()
        self.fault_report_preview.setReadOnly(True)
        self.fault_report_preview.setMinimumHeight(220)
        preview_layout.addWidget(self.fault_report_preview)
        layout.addWidget(preview_group)

        layout.addStretch()
        return widget

    def _browse_fault_report_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose fault report folder",
            self.fault_report_dir_edit.text().strip() or DEFAULT_FAULT_REPORT_DIR,
        )
        if path:
            self.fault_report_dir_edit.setText(path)

    def run_fault_report(self) -> None:
        output_dir = self.fault_report_dir_edit.text().strip() or DEFAULT_FAULT_REPORT_DIR
        timeout_ms = self.fault_timeout_spin.value()
        self.run_fault_report_btn.setEnabled(False)
        self._set_fault_report_status("Running fault check...", error=False)
        self.fault_report_preview.clear()

        fault_data = None
        fault_error = ""
        try:
            fault_data = self.sdk_manager.read_fault_data(timeout_ms)
        except Exception as exc:
            fault_error = str(exc)

        try:
            rows, report_path, report_result = self._build_fault_report_rows(
                fault_data=fault_data,
                fault_error=fault_error,
                output_dir=output_dir,
            )
            append_fault_report_rows(report_path, rows)
        except Exception as exc:
            self._set_fault_report_status(f"Fault report save failed: {exc}", error=True)
            QMessageBox.critical(self, "Fault Report Failed", str(exc))
            self.run_fault_report_btn.setEnabled(True)
            return

        self.fault_report_preview.setPlainText(
            self._format_fault_report_preview(rows, report_path, report_result)
        )
        if report_result == "PASS":
            self._set_fault_report_status(f"Fault check PASS. Saved to {report_path}", error=False)
            self.status_updated.emit("info", f"Fault check PASS: {report_path}")
        else:
            self._set_fault_report_status(f"Fault check FAIL. Saved to {report_path}", error=True)
            self.status_updated.emit("warning", f"Fault check FAIL: {report_path}")
        self.run_fault_report_btn.setEnabled(True)

    def _set_fault_report_status(self, message: str, *, error: bool) -> None:
        self.fault_report_status_label.setText(message)
        color = "#B00020" if error else "#1B5E20"
        self.fault_report_status_label.setStyleSheet(f"color: {color};")

    def _build_fault_report_rows(
        self,
        *,
        fault_data: Any | None,
        fault_error: str,
        output_dir: str,
    ) -> tuple[list[dict[str, str]], str, str]:
        read_at = time.time()
        test_id = datetime.fromtimestamp(read_at).strftime("%Y%m%d-%H%M%S")
        serial_number = self.sdk_manager.serial_number
        serial_number = "" if serial_number == "Unavailable" else serial_number
        serial_identity = parse_serial_identity(serial_number)

        device_info = self.sdk_manager.device_info
        firmware_version = self.sdk_manager.firmware_version
        firmware_version = "" if firmware_version == "Unavailable" else firmware_version
        mechanical_version = ""
        pcb_version = ""
        if device_info is not None:
            mechanical_version = str(getattr(device_info, "mechanical_version", ""))
            pcb_version = str(getattr(device_info, "pcb_version", ""))

        fault_timestamp = getattr(fault_data, "timestamp", None)
        fault_timestamp_unix = "" if fault_timestamp is None else f"{float(fault_timestamp):.3f}"
        fault_timestamp_iso = "" if fault_timestamp is None else timestamp_iso(float(fault_timestamp))

        common = {
            "test_id": test_id,
            "read_at_iso": timestamp_iso(read_at),
            "read_at_unix": f"{read_at:.3f}",
            "selected_model": self.model,
            "selected_side": self.side,
            "serial_number": serial_number,
            **serial_identity,
            "firmware_version": firmware_version,
            "mechanical_version": mechanical_version,
            "pcb_version": pcb_version,
            "interface_name": self.sdk_manager.interface_name,
            "interface_type": self.sdk_manager.interface_type,
            "fault_timestamp_iso": fault_timestamp_iso,
            "fault_timestamp_unix": fault_timestamp_unix,
        }

        faults = getattr(fault_data, "faults", None)
        if faults is not None and hasattr(faults, "to_list"):
            fault_codes = list(faults.to_list())
        else:
            fault_codes = list(getattr(fault_data, "error_codes", ()) or ())
        sdk_has_any_fault = (
            bool(faults.has_any_fault())
            if faults is not None and hasattr(faults, "has_any_fault")
            else (any(int(code) != 0 for code in fault_codes) if fault_codes else None)
        )
        if self.model == "L30":
            model_match = True
            side_match = True
            device_info_loaded = bool(self.device_info is not None and firmware_version)
        else:
            model_match = bool(serial_identity["serial_model"]) and serial_identity["serial_model"] == self.model
            side_match = bool(serial_identity["serial_side"]) and serial_identity["serial_side"] == self.side
            device_info_loaded = bool(serial_number and firmware_version and mechanical_version and pcb_version)

        joint_count_matches = bool(fault_codes) and len(fault_codes) == len(self.hand_config.joint_names)
        fault_read_ok = fault_data is not None and not fault_error
        no_faults = sdk_has_any_fault is False
        report_pass = all(
            (
                device_info_loaded,
                model_match,
                side_match,
                fault_read_ok,
                no_faults,
                joint_count_matches,
            )
        )
        report_result = "PASS" if report_pass else "FAIL"

        rows: list[dict[str, str]] = []

        def add_check(name: str, passed: bool, message: str) -> None:
            rows.append(
                {
                    **common,
                    "row_type": "check",
                    "report_result": report_result,
                    "check_name": name,
                    "check_result": "PASS" if passed else "FAIL",
                    "check_message": message,
                }
            )

        rows.append(
            {
                **common,
                "row_type": "summary",
                "report_result": report_result,
                "check_name": "overall_result",
                "check_result": report_result,
                "check_message": "All SDK fault and identity checks passed" if report_pass else "One or more checks failed",
            }
        )

        add_check("device_info_loaded", device_info_loaded, "Device info read during connection")
        add_check(
            "serial_model_matches_selection",
            model_match,
            f"serial_model={serial_identity['serial_model'] or 'unknown'}, selected_model={self.model}",
        )
        add_check(
            "serial_side_matches_selection",
            side_match,
            f"serial_side={serial_identity['serial_side'] or 'unknown'}, selected_side={self.side}",
        )
        add_check("fault_get_blocking", fault_read_ok, fault_error or "SDK fault.get_blocking() returned data")
        add_check(
            "faults_has_any_fault",
            no_faults,
            "SDK faults.has_any_fault() returned false"
            if no_faults
            else f"SDK faults.has_any_fault() returned {sdk_has_any_fault}",
        )
        add_check(
            "fault_joint_count_matches_model",
            joint_count_matches,
            f"fault_codes={len(fault_codes)}, gui_joints={len(self.hand_config.joint_names)}",
        )

        for index, code in enumerate(fault_codes):
            joint_name = (
                self.hand_config.joint_names[index]
                if index < len(self.hand_config.joint_names)
                else f"Joint {index + 1}"
            )
            has_fault = fault_code_has_fault(code)
            names = "; ".join(fault_code_names(code))
            rows.append(
                {
                    **common,
                    "row_type": "joint_fault",
                    "report_result": report_result,
                    "check_name": "joint_fault_code",
                    "check_result": "FAIL" if has_fault else "PASS",
                    "check_message": names,
                    "joint_index": str(index),
                    "joint_name": joint_name,
                    "fault_code_value": fault_code_value(code),
                    "fault_has_fault": str(has_fault),
                    "fault_names": names,
                }
            )

        file_token = safe_filename_token(serial_number or f"{self.model}_{self.side}")
        report_path = os.path.join(output_dir, f"{file_token}_fault_checks.csv")
        return rows, report_path, report_result

    def _format_fault_report_preview(
        self,
        rows: list[dict[str, str]],
        report_path: str,
        report_result: str,
    ) -> str:
        summary = rows[0] if rows else {}
        lines = [
            f"Result: {report_result}",
            f"CSV: {report_path}",
            f"Serial: {summary.get('serial_number', '') or 'Unavailable'}",
            f"Serial model: {summary.get('serial_model', '') or 'Unknown'}",
            f"Serial side: {summary.get('serial_side', '') or 'Unknown'}",
            f"Firmware: {summary.get('firmware_version', '') or 'Unavailable'}",
            f"Mechanical: {summary.get('mechanical_version', '') or 'Unavailable'}",
            "",
            "Checks:",
        ]
        for row in rows:
            if row.get("row_type") == "check":
                lines.append(
                    f"- {row.get('check_result')}: {row.get('check_name')} "
                    f"({row.get('check_message')})"
                )

        fault_rows = [row for row in rows if row.get("row_type") == "joint_fault"]
        if fault_rows:
            lines.extend(["", "Joint faults:"])
            for row in fault_rows:
                lines.append(
                    f"- {row.get('check_result')}: {row.get('joint_name')} "
                    f"code={row.get('fault_code_value')} {row.get('fault_names')}"
                )
        return "\n".join(lines)

    def _status_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.status_log = QLabel("Waiting for system startup...")
        self.status_log.setObjectName("StatusLabel")
        self.status_log.setObjectName("StatusInfo")
        self.status_log.setWordWrap(True)
        self.status_log.setMinimumHeight(300)
        layout.addWidget(self.status_log)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_status_log)
        layout.addWidget(clear_btn)
        return widget

    def create_value_display_panel(self) -> QGroupBox:
        panel = QGroupBox("Joint Value List")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 20, 10, 20)
        self.value_display = QTextEdit()
        self.value_display.setObjectName("ValueDisplay")
        self.value_display.setReadOnly(True)
        self.value_display.setMinimumHeight(60)
        self.value_display.setMaximumHeight(80)
        layout.addWidget(self.value_display)
        return panel

    def on_slider_value_changed(self, index: int, value: int) -> None:
        if 0 <= index < len(self.slider_labels):
            self.slider_labels[index].setText(f"{self.hand_config.joint_names[index]}: {value}")
        self.update_value_display()

    def update_value_display(self) -> None:
        self.value_display.setText(str([slider.value() for slider in self.sliders]))

    def on_preset_action_clicked(self, positions: list[int], *, preset_name: str = "Manual Preset") -> None:
        if len(positions) != len(self.sliders):
            QMessageBox.warning(
                self,
                "Action Mismatch",
                f"Preset action joint count ({len(positions)}) does not match current joint count ({len(self.sliders)})",
            )
            return
        for idx, (slider, position) in enumerate(zip(self.sliders, positions)):
            slider.setValue(int(position))
            self.on_slider_value_changed(idx, int(position))
        self.publish_joint_state(force=True, command_label=preset_name)

    def on_home_clicked(self) -> None:
        if self.model == "L30" and self.sdk_manager.joint_limits:
            # Native L30 Home: flexion/rotation joints at their actual
            # minimum; side/abduction and wrist joints at range midpoint.
            home_positions = []
            for name, (minimum, maximum) in zip(
                self.hand_config.joint_names,
                self.sdk_manager.joint_limits,
            ):
                is_centered = any(
                    token in name.lower() for token in ("side", "abduction", "wrist")
                )
                home_positions.append((minimum + maximum) // 2 if is_centered else minimum)
        else:
            home_positions = self.sdk_manager.initial_positions or self.hand_config.init_pos
        for slider, position in zip(self.sliders, home_positions):
            slider.setValue(int(position))
        self.publish_joint_state(force=True, command_label="Home")
        self.status_updated.emit("info", "Return to Home")

    def on_stop_clicked(self) -> None:
        self._stop_cycle_actions()
        if self.auto_grab_running:
            self.stop_auto_grab()
        if self.motion_test_active:
            self._finish_motion_test("Stopped")
        if self.model == "L30":
            self.sdk_manager.disable_motion()
            if hasattr(self, "enable_motion_button"):
                self.enable_motion_button.setText("Enable L30 Motion")
        self.status_updated.emit("warning", "All actions stopped")

    def on_enable_l30_motion_clicked(self) -> None:
        if self.model != "L30":
            return
        if self.sdk_manager.motion_enabled:
            self.sdk_manager.disable_motion()
            self.enable_motion_button.setText("Enable L30 Motion")
            return
        answer = QMessageBox.warning(
            self,
            "Enable L30 Motion",
            "Confirm the workspace is clear. This allows native L30 position commands.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer == QMessageBox.Yes:
            self.sdk_manager.enable_motion(
                speed=self.speed_slider.value(),
                commanded_current=self.torque_slider.value(),
                torque_limit_percent=(
                    self.torque_limit_slider.value()
                    if self.sdk_manager.torque_limit_supported
                    else None
                ),
                acceleration=(
                    self.acceleration_slider.value()
                    if self.sdk_manager.acceleration_supported
                    else None
                ),
            )
            if self.sdk_manager.motion_enabled:
                self.enable_motion_button.setText("Disable L30 Motion")

    def on_cycle_clicked(self) -> None:
        if not self.hand_config.preset_actions:
            QMessageBox.warning(self, "No Preset Actions", "Current hand model has no preset actions to cycle")
            return
        if self.cycle_timer and self.cycle_timer.isActive():
            self._stop_cycle_actions()
            self.status_updated.emit("info", "Stopped cycling preset actions")
            if self.motion_test_active:
                self._finish_motion_test("Cycle stopped")
            return
        self._start_cycle_actions()

    def _start_cycle_actions(self) -> None:
        self.current_action_index = -1
        self.cycle_loop_active = False
        self.cycle_loop_index = -1
        self.cycle_loop_iterations = 0
        self.cycle_timer = QTimer(self)
        self.cycle_timer.timeout.connect(self.run_next_action)
        self.cycle_timer.start(LOOP_TIME_MS)
        self.cycle_button.setText("Stop Cycling")
        self.status_updated.emit("info", "Started cycling preset actions")
        self.run_next_action()

    def _stop_cycle_actions(self) -> None:
        if self.cycle_timer and self.cycle_timer.isActive():
            self.cycle_timer.stop()
        self.cycle_timer = None
        self.cycle_button.setText("Cycle Preset Actions")
        self.reset_preset_buttons_color()

    def run_next_action(self) -> None:
        if not self.hand_config.preset_actions:
            return
        self.reset_preset_buttons_color()
        name = self._next_cycle_preset_name()
        self.on_preset_action_clicked(self.hand_config.preset_actions[name], preset_name=name)
        preset_index = list(self.hand_config.preset_actions).index(name)
        if 0 <= preset_index < len(self.preset_buttons):
            self.preset_buttons[preset_index].setStyleSheet(
                "background-color: green; color: white; border-color: #91D5FF;"
            )
        self.status_updated.emit("info", f"Running preset action: {name}")

    def _next_cycle_preset_name(self) -> str:
        loop_names = self._cycle_loop_names()
        if self.cycle_loop_active and loop_names:
            self.cycle_loop_index = (self.cycle_loop_index + 1) % len(loop_names)
            name = loop_names[self.cycle_loop_index]
            if self.cycle_loop_index == len(loop_names) - 1:
                self.cycle_loop_iterations += 1
                if self._cycle_loop_limit_reached():
                    self.cycle_loop_active = False
                    self.cycle_loop_index = -1
                    self.cycle_loop_iterations = 0
            return name

        names = list(self.hand_config.preset_actions)
        self.current_action_index = (self.current_action_index + 1) % len(names)
        name = names[self.current_action_index]
        if loop_names and name == loop_names[-1]:
            self.cycle_loop_active = True
            self.cycle_loop_index = len(loop_names) - 1
            self.cycle_loop_iterations = 0
        return name

    def _cycle_loop_names(self) -> list[str]:
        names = [
            name
            for name in self.hand_config.cycle_loop_actions
            if name in self.hand_config.preset_actions
        ]
        return names if len(names) >= 2 else []

    def _cycle_loop_limit_reached(self) -> bool:
        repeats = max(0, int(self.hand_config.cycle_loop_repeats))
        return repeats > 0 and self.cycle_loop_iterations >= repeats

    def reset_preset_buttons_color(self) -> None:
        for button in self.preset_buttons:
            button.setStyleSheet("")
            button.setProperty("category", "preset")
            button.style().unpolish(button)
            button.style().polish(button)

    def publish_joint_state(self, *, force: bool = False, command_label: str = "Manual") -> None:
        positions = [slider.value() for slider in self.sliders]
        result = self.sdk_manager.publish_joint_state(positions, force=force)
        if self.motion_test_active and result and result.get("sent"):
            self._record_motion_command(command_label, result)

    def on_start_test_clicked(self) -> None:
        if self.motion_test_active:
            QMessageBox.information(self, "Motion Test", "A motion timing test is already running.")
            return
        if not self.hand_config.preset_actions:
            QMessageBox.warning(self, "No Preset Actions", "Current hand model has no preset actions to test")
            return
        if self.cycle_timer and self.cycle_timer.isActive():
            self._stop_cycle_actions()
        self._start_motion_test()
        if self.sdk_manager.latest_angle_sample() is None:
            self.status_updated.emit(
                "warning",
                "Motion test started before an angle baseline was available; first command may be marked no_baseline",
            )
        self._start_cycle_actions()

    def on_end_test_clicked(self) -> None:
        if not self.motion_test_active:
            QMessageBox.information(self, "Motion Test", "No motion timing test is running.")
            return
        self._stop_cycle_actions()
        self._finish_motion_test("Ended")

    def _start_motion_test(self) -> None:
        self.motion_test_active = True
        self.motion_test_session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.motion_test_records = []
        self.motion_test_command_index = 0
        self.motion_test_log_path = ""
        self._set_motion_test_buttons(True)
        self.motion_test_summary_label.setText(f"Motion test: running ({self.motion_test_session_id})")
        self.status_updated.emit("info", f"Motion timing test started: {self.motion_test_session_id}")

    def _finish_motion_test(self, reason: str, *, show_dialog: bool = True) -> None:
        if not self.motion_test_active:
            return
        self._close_pending_motion_commands()
        log_path = ""
        if self.motion_test_records:
            log_path = self._write_motion_test_log()
        summary = self._motion_test_summary(reason, log_path)
        self.motion_test_active = False
        self.motion_test_log_path = log_path
        self._set_motion_test_buttons(False)
        self.motion_test_summary_label.setText(summary.replace("\n", " | "))
        self.status_updated.emit("info", summary.replace("\n", " | "))
        if show_dialog:
            QMessageBox.information(self, "Motion Test Complete", summary)

    def _set_motion_test_buttons(self, running: bool) -> None:
        if hasattr(self, "start_test_button"):
            self.start_test_button.setEnabled(not running)
        if hasattr(self, "end_test_button"):
            self.end_test_button.setEnabled(running)

    def _record_motion_command(self, preset_name: str, result: dict[str, Any]) -> None:
        send_start_ns = int(result["send_start_ns"])
        self._set_pending_motion_cutoffs(send_start_ns)
        self.motion_test_command_index += 1

        baseline_sample = result.get("baseline_sample")
        baseline_values: list[Any] | None = None
        baseline_perf_ns: int | None = None
        baseline_wall_time: float | None = None
        if baseline_sample is not None:
            baseline_values, baseline_perf_ns, baseline_wall_time = baseline_sample

        record = MotionTimingRecord(
            session_id=self.motion_test_session_id,
            command_index=self.motion_test_command_index,
            preset_name=preset_name,
            target_values=list(result["target_values"]),
            baseline_values=list(baseline_values) if baseline_values is not None else None,
            baseline_perf_ns=baseline_perf_ns,
            baseline_wall_time=baseline_wall_time,
            send_start_ns=send_start_ns,
            send_end_ns=int(result["send_end_ns"]),
            send_start_wall_time=float(result["send_start_wall_time"]),
        )

        if record.baseline_values is None:
            record.status = "no_baseline"
        elif not self._candidate_motion_joints(record):
            record.status = "no_target_delta"

        self.motion_test_records.append(record)
        self._update_motion_test_running_summary()

    def _set_pending_motion_cutoffs(self, next_command_ns: int) -> None:
        for record in self.motion_test_records:
            if record.next_command_ns is None:
                record.next_command_ns = next_command_ns

    def _close_pending_motion_commands(self) -> None:
        for record in self.motion_test_records:
            if record.status == "pending":
                record.status = "superseded" if record.next_command_ns is not None else "not_detected"

    def update_motion_timing_from_angles(
        self,
        values: Any,
        sample_perf_ns: Any,
        sample_wall_time: Any,
    ) -> None:
        if not self.motion_test_active or not self.motion_test_records:
            return
        sample_ns = int(sample_perf_ns)
        wall_time = float(sample_wall_time)
        angle_values = list(values)

        changed = False
        for record in self.motion_test_records:
            if record.status not in {"pending", "moved"} or sample_ns < record.send_start_ns:
                continue
            if record.next_command_ns is not None and sample_ns >= record.next_command_ns:
                continue
            if record.status == "pending" and record.first_sample_ns is None:
                record.first_sample_ns = sample_ns
                record.first_sample_wall_time = wall_time
                record.first_sample_values = list(angle_values)

            if record.status == "pending":
                moved_indexes = self._moved_joint_indexes(record, angle_values)
                if moved_indexes:
                    record.status = "moved"
                    record.first_motion_ns = sample_ns
                    record.first_motion_wall_time = wall_time
                    record.first_motion_values = list(angle_values)
                    record.moved_joint_indexes = moved_indexes
                    changed = True

            if self._record_motion_progress_thresholds(record, angle_values, sample_ns, wall_time):
                changed = True

        if changed:
            self._update_motion_test_running_summary()

    def _candidate_motion_joints(self, record: MotionTimingRecord) -> list[int]:
        if record.baseline_values is None:
            return []
        count = min(len(record.target_values), len(record.baseline_values))
        indexes = []
        for idx in range(count):
            target = self._numeric_value(record.target_values, idx)
            baseline = self._numeric_value(record.baseline_values, idx)
            if target is None or baseline is None:
                continue
            if abs(target - baseline) > MOTION_TARGET_DELTA_THRESHOLD:
                indexes.append(idx)
        return indexes

    def _moved_joint_indexes(self, record: MotionTimingRecord, current_values: list[Any]) -> list[int]:
        if record.baseline_values is None:
            return []
        moved = []
        for idx in self._candidate_motion_joints(record):
            if idx >= len(current_values):
                continue
            target = self._numeric_value(record.target_values, idx)
            baseline = self._numeric_value(record.baseline_values, idx)
            current = self._numeric_value(current_values, idx)
            if target is None or baseline is None or current is None:
                continue
            target_delta = target - baseline
            observed_delta = current - baseline
            if abs(observed_delta) >= MOTION_START_THRESHOLD and observed_delta * target_delta > 0:
                moved.append(idx)
        return moved

    def _record_motion_progress_thresholds(
        self,
        record: MotionTimingRecord,
        current_values: list[Any],
        sample_ns: int,
        wall_time: float,
    ) -> bool:
        changed = False
        for label, fraction in MOTION_PROGRESS_THRESHOLDS:
            if label in record.progress_ns_by_threshold:
                continue
            progressed_indexes = self._progressed_joint_indexes(record, current_values, fraction)
            if not progressed_indexes:
                continue
            record.progress_ns_by_threshold[label] = sample_ns
            record.progress_wall_time_by_threshold[label] = wall_time
            record.progress_values_by_threshold[label] = list(current_values)
            record.progress_joint_indexes_by_threshold[label] = progressed_indexes
            changed = True
        return changed

    def _comparable_motion_joints(self, record: MotionTimingRecord) -> list[int]:
        if record.baseline_values is None:
            return []
        count = min(len(record.target_values), len(record.baseline_values))
        indexes = []
        for idx in range(count):
            target = self._numeric_value(record.target_values, idx)
            baseline = self._numeric_value(record.baseline_values, idx)
            if target is None or baseline is None:
                continue
            if abs(target - baseline) >= MOTION_COMPARABLE_MIN_TARGET_DELTA:
                indexes.append(idx)
        return indexes

    def _progressed_joint_indexes(
        self,
        record: MotionTimingRecord,
        current_values: list[Any],
        target_fraction: float,
    ) -> list[int]:
        if record.baseline_values is None:
            return []
        progressed = []
        for idx in self._comparable_motion_joints(record):
            if idx >= len(current_values):
                continue
            target = self._numeric_value(record.target_values, idx)
            baseline = self._numeric_value(record.baseline_values, idx)
            current = self._numeric_value(current_values, idx)
            if target is None or baseline is None or current is None:
                continue
            target_delta = target - baseline
            observed_delta = current - baseline
            required_delta = max(MOTION_START_THRESHOLD, abs(target_delta) * target_fraction)
            if abs(observed_delta) >= required_delta and observed_delta * target_delta > 0:
                progressed.append(idx)
        return progressed

    def _joint_names_for_indexes(self, indexes: list[int]) -> list[str]:
        return [
            self.hand_config.joint_names[idx]
            if idx < len(self.hand_config.joint_names)
            else f"Joint {idx + 1}"
            for idx in indexes
        ]

    @staticmethod
    def _numeric_value(values: list[Any], index: int) -> float | None:
        try:
            return float(values[index])
        except (TypeError, ValueError, IndexError):
            return None

    def _update_motion_test_running_summary(self) -> None:
        if not hasattr(self, "motion_test_summary_label"):
            return
        moved_count = sum(1 for record in self.motion_test_records if record.status == "moved")
        comparable_count = sum(
            1
            for record in self.motion_test_records
            if "10pct" in record.progress_ns_by_threshold
        )
        pending_count = sum(
            1
            for record in self.motion_test_records
            if record.status == "pending" and record.next_command_ns is None
        )
        cutoff_count = sum(
            1
            for record in self.motion_test_records
            if record.status == "pending" and record.next_command_ns is not None
        )
        self.motion_test_summary_label.setText(
            f"Motion test: running ({len(self.motion_test_records)} commands, "
            f"{moved_count} nudged, {comparable_count} comparable, "
            f"{pending_count} pending, {cutoff_count} closing)"
        )

    def _write_motion_test_log(self) -> str:
        os.makedirs(DEFAULT_MOTION_TIMING_DIR, exist_ok=True)
        serial_number = self.sdk_manager.serial_number
        serial_token = "" if serial_number == "Unavailable" else serial_number
        file_token = safe_filename_token(serial_token or f"{self.model}_{self.side}")
        path = os.path.join(
            DEFAULT_MOTION_TIMING_DIR,
            f"{file_token}_motion_latency_{self.motion_test_session_id}.csv",
        )
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MOTION_TIMING_COLUMNS)
            writer.writeheader()
            for record in self.motion_test_records:
                writer.writerow(self._motion_record_row(record))
        return path

    def _motion_record_row(self, record: MotionTimingRecord) -> dict[str, str]:
        first_sample_latency = self._elapsed_ms(record.send_start_ns, record.first_sample_ns)
        first_motion_latency = self._elapsed_ms(record.send_start_ns, record.first_motion_ns)
        send_complete_latency = self._elapsed_ms(record.send_end_ns, record.first_motion_ns)
        latency_5pct = self._elapsed_ms(record.send_start_ns, record.progress_ns_by_threshold.get("5pct"))
        latency_10pct = self._elapsed_ms(record.send_start_ns, record.progress_ns_by_threshold.get("10pct"))
        latency_50pct = self._elapsed_ms(record.send_start_ns, record.progress_ns_by_threshold.get("50pct"))
        latency_90pct = self._elapsed_ms(record.send_start_ns, record.progress_ns_by_threshold.get("90pct"))
        baseline_age = self._elapsed_ms(record.baseline_perf_ns, record.send_start_ns)
        sensor_wall_time = (
            record.progress_wall_time_by_threshold.get("10pct")
            or record.first_motion_wall_time
            or record.first_sample_wall_time
        )
        sensor_timestamp_unix = "" if sensor_wall_time is None else f"{sensor_wall_time:.6f}"
        sensor_timestamp_iso = "" if sensor_wall_time is None else timestamp_iso(sensor_wall_time)
        moved_joint_names = [
            self.hand_config.joint_names[idx]
            if idx < len(self.hand_config.joint_names)
            else f"Joint {idx + 1}"
            for idx in record.moved_joint_indexes
        ]
        comparable_joint_indexes = self._comparable_motion_joints(record)
        comparable_joint_names = self._joint_names_for_indexes(comparable_joint_indexes)
        threshold_joint_indexes = record.progress_joint_indexes_by_threshold.get("10pct", [])
        threshold_joint_names = self._joint_names_for_indexes(threshold_joint_indexes)
        serial_number = self.sdk_manager.serial_number
        serial_number = "" if serial_number == "Unavailable" else serial_number
        return {
            "session_id": record.session_id,
            "command_index": str(record.command_index),
            "preset_name": record.preset_name,
            "status": record.status,
            "send_started_at_iso": timestamp_iso(record.send_start_wall_time),
            "send_started_at_unix": f"{record.send_start_wall_time:.6f}",
            "send_duration_ms": f"{(record.send_end_ns - record.send_start_ns) / 1_000_000:.3f}",
            "first_angle_sample_latency_ms": first_sample_latency,
            "first_motion_latency_ms": first_motion_latency,
            "send_complete_to_first_motion_ms": send_complete_latency,
            "comparable_dead_time_ms": latency_10pct,
            "time_to_5pct_target_delta_ms": latency_5pct,
            "time_to_10pct_target_delta_ms": latency_10pct,
            "time_to_50pct_target_delta_ms": latency_50pct,
            "time_to_90pct_target_delta_ms": latency_90pct,
            "baseline_sample_age_ms": baseline_age,
            "sensor_sample_timestamp_iso": sensor_timestamp_iso,
            "sensor_sample_timestamp_unix": sensor_timestamp_unix,
            "moved_joint_indexes": json.dumps(record.moved_joint_indexes, separators=(",", ":")),
            "moved_joint_names": json.dumps(moved_joint_names, separators=(",", ":")),
            "comparable_joint_indexes": json.dumps(comparable_joint_indexes, separators=(",", ":")),
            "comparable_joint_names": json.dumps(comparable_joint_names, separators=(",", ":")),
            "threshold_crossed_joint_indexes": json.dumps(threshold_joint_indexes, separators=(",", ":")),
            "threshold_crossed_joint_names": json.dumps(threshold_joint_names, separators=(",", ":")),
            "target_values": self._json_list(record.target_values),
            "baseline_values": self._json_list(record.baseline_values),
            "first_angle_sample_values": self._json_list(record.first_sample_values),
            "first_motion_values": self._json_list(record.first_motion_values),
            "first_10pct_values": self._json_list(record.progress_values_by_threshold.get("10pct")),
            "model": self.model,
            "side": self.side,
            "serial_number": serial_number,
            "interface_name": self.sdk_manager.interface_name,
            "interface_type": self.sdk_manager.interface_type,
            "sensor_read_mode": self.sdk_manager.sensor_read_mode,
        }

    @staticmethod
    def _elapsed_ms(start_ns: int | None, end_ns: int | None) -> str:
        if start_ns is None or end_ns is None:
            return ""
        return f"{(end_ns - start_ns) / 1_000_000:.3f}"

    @staticmethod
    def _json_list(values: list[Any] | None) -> str:
        if values is None:
            return ""
        return json.dumps(values, separators=(",", ":"))

    def _motion_test_summary(self, reason: str, log_path: str) -> str:
        moved_records = [
            record
            for record in self.motion_test_records
            if record.status == "moved" and record.first_motion_ns is not None
        ]
        first_nudge_latencies = [
            (record.first_motion_ns - record.send_start_ns) / 1_000_000
            for record in moved_records
            if record.first_motion_ns is not None
        ]
        comparable_latencies = [
            (record.progress_ns_by_threshold["10pct"] - record.send_start_ns) / 1_000_000
            for record in self.motion_test_records
            if "10pct" in record.progress_ns_by_threshold
        ]
        lines = [
            f"Motion test {reason.lower()}.",
            f"Commands sent: {len(self.motion_test_records)}",
            f"First nudges detected: {len(moved_records)}",
            f"Comparable movements detected: {len(comparable_latencies)}",
        ]
        if comparable_latencies:
            lines.extend(
                [
                    f"Average dead time (10% target travel): {sum(comparable_latencies) / len(comparable_latencies):.3f} ms",
                    f"Min dead time: {min(comparable_latencies):.3f} ms",
                    f"Max dead time: {max(comparable_latencies):.3f} ms",
                ]
            )
        else:
            lines.append("Average dead time: unavailable")
        if first_nudge_latencies:
            lines.append(
                f"Average first nudge: {sum(first_nudge_latencies) / len(first_nudge_latencies):.3f} ms"
            )
        status_counts: dict[str, int] = {}
        for record in self.motion_test_records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        if status_counts:
            status_text = ", ".join(
                f"{status}={count}" for status, count in sorted(status_counts.items())
            )
            lines.append(f"Statuses: {status_text}")
        if log_path:
            lines.append(f"CSV: {log_path}")
        return "\n".join(lines)

    def on_sensor_read_mode_changed(self, mode: str) -> None:
        self.sdk_manager.set_sensor_read_mode(mode)

    def _start_live_timer(self, timer: QTimer) -> None:
        if not timer.isActive():
            timer.start()

    def on_global_speed_changed(self, value: int) -> None:
        self.speed_val_lbl.setText(str(value))
        if self.live_joint_speed_timer.isActive():
            self.live_joint_speed_timer.stop()
        self._start_live_timer(self.live_speed_timer)

    def _publish_live_speed(self) -> None:
        self.sdk_manager.publish_speed(self.speed_slider.value(), log_success=False)

    def on_global_torque_changed(self, value: int) -> None:
        self.torque_val_lbl.setText(str(value))
        if self.live_joint_torque_timer.isActive():
            self.live_joint_torque_timer.stop()
        self.current_torque_values = [value] * len(self.hand_config.joint_names)
        self._sync_per_joint_torque_sliders()
        self._update_commanded_torque_display()
        self._set_last_torque_boost_text("Last boost: reset", "black")
        self._start_live_timer(self.live_torque_timer)

    def _publish_live_torque(self) -> None:
        value = self.torque_slider.value()
        self.current_torque_values = [value] * len(self.hand_config.joint_names)
        self._sync_per_joint_torque_sliders()
        self._update_commanded_torque_display()
        self.sdk_manager.publish_torque(value, log_success=False)

    def on_global_torque_limit_changed(self, value: int) -> None:
        self.torque_limit_val_lbl.setText(f"{value}%")
        self._start_live_timer(self.live_torque_limit_timer)

    def _publish_live_torque_limit(self) -> None:
        self.sdk_manager.publish_torque_limit(
            self.torque_limit_slider.value(), log_success=False
        )

    def on_global_acceleration_changed(self, value: int) -> None:
        self.acceleration_val_lbl.setText(str(value))
        self._start_live_timer(self.live_acceleration_timer)

    def _publish_live_acceleration(self) -> None:
        self.sdk_manager.publish_acceleration(
            self.acceleration_slider.value(), log_success=False
        )

    def on_per_joint_setting_changed(self, kind: str, value_label: QLabel, value: int) -> None:
        value_label.setText(str(value))
        if self._syncing_per_joint_settings:
            return
        if kind == "speed":
            if self.live_speed_timer.isActive():
                self.live_speed_timer.stop()
            self._start_live_timer(self.live_joint_speed_timer)
            return
        if self.live_torque_timer.isActive():
            self.live_torque_timer.stop()
        self.current_torque_values = [slider.value() for slider in self.per_joint_torque_sliders]
        self._update_commanded_torque_display()
        self._set_last_torque_boost_text("Last boost: reset", "black")
        self._start_live_timer(self.live_joint_torque_timer)

    def _publish_live_joint_speeds(self) -> None:
        if not self.per_joint_speed_sliders:
            return
        values = [slider.value() for slider in self.per_joint_speed_sliders]
        self.sdk_manager.publish_joint_speeds(values, log_success=False)

    def _publish_live_joint_torques(self) -> None:
        if not self.per_joint_torque_sliders:
            return
        values = [slider.value() for slider in self.per_joint_torque_sliders]
        self.current_torque_values = values
        self._update_commanded_torque_display()
        self.sdk_manager.publish_joint_torques(values, log_success=False)

    def _sync_per_joint_torque_sliders(self) -> None:
        if len(self.per_joint_torque_sliders) != len(self.current_torque_values):
            return
        self._syncing_per_joint_settings = True
        try:
            for slider, value in zip(self.per_joint_torque_sliders, self.current_torque_values):
                slider.setValue(int(value))
        finally:
            self._syncing_per_joint_settings = False

    def _update_commanded_torque_display(
        self,
        *,
        boosted_finger: str | None = None,
        boosted_values: list[int] | None = None,
        maxed_out: bool = False,
    ) -> None:
        if not hasattr(self, "torque_val_lbl"):
            return

        values = self._current_joint_torque_values()
        if values:
            unique_values = sorted(set(values))
            self._sync_global_torque_slider_position(values, uniform=len(unique_values) == 1)
            if len(unique_values) == 1:
                self.torque_val_lbl.setText(str(unique_values[0]))
                self.torque_val_lbl.setToolTip("Uniform commanded torque")
            else:
                self.torque_val_lbl.setText("mixed")
                self.torque_val_lbl.setToolTip("Per-joint commanded torques differ; see Touch Control")

        if hasattr(self, "commanded_torque_summary_label") and values:
            self.commanded_torque_summary_label.setText(
                f"Command: min={min(values)} max={max(values)}"
            )

        if hasattr(self, "commanded_torque_finger_labels"):
            finger_map = self._finger_joint_map()
            for finger, label in self.commanded_torque_finger_labels.items():
                indices = [idx for idx in finger_map.get(finger, []) if idx < len(values)]
                if not indices:
                    self._set_label_status(label, f"{finger[0].upper()}:--", "gray")
                    continue
                finger_values = [values[idx] for idx in indices]
                color = "#B26A00" if boosted_finger == finger and not maxed_out else "black"
                text = ",".join(str(value) for value in finger_values)
                self._set_label_status(label, f"{finger[0].upper()}:{text}", color)

        if boosted_finger and hasattr(self, "last_torque_boost_label"):
            if maxed_out:
                self._set_last_torque_boost_text(
                    f"Last boost: {boosted_finger} already at max",
                    "#B26A00",
                )
            else:
                values_text = boosted_values if boosted_values is not None else []
                self._set_last_torque_boost_text(
                    f"Last boost: {boosted_finger} -> {values_text}",
                    "green",
                )

    def _set_last_torque_boost_text(self, text: str, color: str) -> None:
        if not hasattr(self, "last_torque_boost_label"):
            return
        self.last_torque_boost_label.setText(text)
        self.last_torque_boost_label.setStyleSheet(f"color: {color};")

    def _sync_global_torque_slider_position(self, values: list[int], *, uniform: bool) -> None:
        if not hasattr(self, "torque_slider") or not values:
            return
        # Keep the physical slider from showing a stale value after slip torque boost.
        # For mixed per-joint torque values, use the max so lowering the global slider
        # always emits valueChanged and resets every joint to the user's chosen value.
        display_value = values[0] if uniform else max(values)
        was_blocked = self.torque_slider.blockSignals(True)
        try:
            self.torque_slider.setValue(int(display_value))
        finally:
            self.torque_slider.blockSignals(was_blocked)

    def update_matrix_display(self, matrix_data: dict[str, Any]) -> None:
        for key, data in matrix_data.items():
            self.matrix_display.update_matrix_data(key, data)
            finger = self._finger_from_matrix_key(key)
            if finger is not None:
                self.latest_touch_matrices[finger] = data
                self._update_slip_detection(finger, data)

    def start_auto_grab(self) -> None:
        if self.auto_grab_running:
            return
        self._set_auto_grab_start_torque()
        if not self.latest_touch_matrices:
            self.auto_grab_status_label.setText("No touch data")
            self.status_updated.emit("warning", "Auto Grab requires touch matrix data")
            return
        self.auto_grab_running = True
        self.auto_grab_sensor_fail_count = 0
        self.auto_grab_fingers_stopped = {finger: False for finger in self.finger_order}
        self.auto_grab_above_count = {finger: 0 for finger in self.finger_order}
        self.auto_grab_started_at = time.monotonic()
        self.auto_grab_baseline = self._touch_max_by_finger()
        self.auto_grab_start_button.setEnabled(False)
        self.auto_grab_stop_button.setEnabled(True)
        self.auto_grab_status_label.setText("Grabbing...")
        for finger, label in self.auto_grab_finger_labels.items():
            self._set_label_status(label, f"{finger[0].upper()}:--", "black")
        self.auto_grab_timer.start()
        self.status_updated.emit(
            "info",
            f"Auto Grab started; baseline={self.auto_grab_baseline}; "
            f"window={AUTO_GRAB_MAX_DURATION_S:.0f}s",
        )

    def _set_auto_grab_start_torque(self) -> None:
        if self.live_torque_timer.isActive():
            self.live_torque_timer.stop()
        if self.live_joint_torque_timer.isActive():
            self.live_joint_torque_timer.stop()
        self.current_torque_values = [AUTO_GRAB_START_TORQUE] * len(self.hand_config.joint_names)
        self._sync_per_joint_torque_sliders()
        self._update_commanded_torque_display()
        self._set_last_torque_boost_text("Last boost: reset", "black")
        self.sdk_manager.publish_torque(AUTO_GRAB_START_TORQUE, log_success=False)
        self.status_updated.emit("info", f"Auto Grab torque set to {AUTO_GRAB_START_TORQUE}")

    def stop_auto_grab(self) -> None:
        if self.auto_grab_timer.isActive():
            self.auto_grab_timer.stop()
        self.auto_grab_running = False
        self.auto_grab_start_button.setEnabled(True)
        self.auto_grab_stop_button.setEnabled(False)
        self.auto_grab_status_label.setText("Stopped")
        self.status_updated.emit("warning", "Auto Grab stopped")

    def _auto_grab_step(self) -> None:
        if not self.auto_grab_running:
            self.auto_grab_timer.stop()
            return

        touch_data = self._touch_max_by_finger()
        if not touch_data:
            self.auto_grab_sensor_fail_count += 1
            if self.auto_grab_sensor_fail_count >= 3:
                self._finish_auto_grab("Touch read error")
            return
        self.auto_grab_sensor_fail_count = 0

        current_pos = [slider.value() for slider in self.sliders]
        target_pos = self._auto_grab_target_positions()
        if len(target_pos) != len(current_pos):
            self._finish_auto_grab("Target mismatch")
            return

        threshold = self.auto_grab_threshold_spin.value()
        step_size = max(1, self.auto_grab_speed_spin.value() // 10)
        finger_joint_map = self._finger_joint_map()
        changed = False

        for finger, joint_indices in finger_joint_map.items():
            if self.auto_grab_fingers_stopped.get(finger, False):
                continue

            raw_pressure = int(touch_data.get(finger, 0))
            baseline = int(self.auto_grab_baseline.get(finger, 0))
            pressure = max(0, raw_pressure - baseline)
            self._update_auto_grab_finger_label(finger, pressure, threshold)

            if pressure >= threshold:
                self.auto_grab_above_count[finger] = self.auto_grab_above_count.get(finger, 0) + 1
                if self.auto_grab_above_count[finger] >= self.auto_grab_debounce_limit:
                    self.auto_grab_fingers_stopped[finger] = True
                    self._update_auto_grab_finger_stopped(finger, pressure, raw_pressure, baseline)
                continue

            self.auto_grab_above_count[finger] = 0
            for idx in joint_indices:
                if idx >= len(current_pos):
                    continue
                current = current_pos[idx]
                target = target_pos[idx]
                if current == target:
                    continue
                if current > target:
                    current_pos[idx] = max(target, current - step_size)
                else:
                    current_pos[idx] = min(target, current + step_size)
                changed = True

        reached_target = all(
            self.auto_grab_fingers_stopped.get(finger, False)
            or all(
                idx >= len(current_pos) or current_pos[idx] == target_pos[idx]
                for idx in finger_joint_map.get(finger, [])
            )
            for finger in finger_joint_map
        )

        if changed:
            self._set_slider_positions(current_pos)
        stopped_count = sum(1 for stopped in self.auto_grab_fingers_stopped.values() if stopped)
        elapsed_s = time.monotonic() - self.auto_grab_started_at
        if all(self.auto_grab_fingers_stopped.values()):
            self._finish_auto_grab(f"Done ({stopped_count}/5 stopped)")
        elif elapsed_s >= AUTO_GRAB_MAX_DURATION_S:
            target_text = "target reached" if reached_target else "target not reached"
            self._finish_auto_grab(
                f"Done ({stopped_count}/5 stopped, {target_text}, {AUTO_GRAB_MAX_DURATION_S:.0f}s)"
            )

    def _finish_auto_grab(self, status: str) -> None:
        if self.auto_grab_timer.isActive():
            self.auto_grab_timer.stop()
        self.auto_grab_running = False
        self.auto_grab_start_button.setEnabled(True)
        self.auto_grab_stop_button.setEnabled(False)
        self.auto_grab_status_label.setText(status)
        self.status_updated.emit("info", f"Auto Grab {status}")

    def _auto_grab_target_positions(self) -> list[int]:
        for name, positions in self.hand_config.preset_actions.items():
            if name.lower() == "fist":
                return list(positions)
        return [0] * len(self.sliders)

    def _set_slider_positions(self, positions: list[int]) -> None:
        for slider, value in zip(self.sliders, positions):
            slider.setValue(int(value))

    def _touch_max_by_finger(self) -> dict[str, int]:
        return {
            finger: self._get_max_pressure(self.latest_touch_matrices.get(finger))
            for finger in self.finger_order
        }

    def _finger_from_matrix_key(self, key: str) -> str | None:
        if key.endswith("_matrix"):
            return self._finger_key_from_name(key[:-7])
        return self._finger_key_from_name(key)

    def _finger_joint_map(self) -> dict[str, list[int]]:
        mapping = {finger: [] for finger in self.finger_order}
        for idx, name in enumerate(self.hand_config.joint_names):
            finger = self._finger_key_from_name(name)
            if finger is not None:
                mapping[finger].append(idx)
        return mapping

    def _update_auto_grab_finger_label(self, finger: str, pressure: int, threshold: int) -> None:
        label = self.auto_grab_finger_labels.get(finger)
        if label is None:
            return
        if pressure >= threshold:
            color = "red"
        elif pressure >= threshold * 0.7:
            color = "#B26A00"
        else:
            color = "black"
        self._set_label_status(label, f"{finger[0].upper()}:{pressure:3d}", color)

    def _update_auto_grab_finger_stopped(
        self,
        finger: str,
        pressure: int,
        raw_pressure: int,
        baseline: int,
    ) -> None:
        label = self.auto_grab_finger_labels.get(finger)
        if label is not None:
            self._set_label_status(label, f"{finger[0].upper()}:STP", "green")
        self.status_updated.emit(
            "info",
            f"{finger.capitalize()} stopped at pressure {pressure} (raw={raw_pressure}, baseline={baseline})",
        )

    def _update_slip_detection(self, finger: str, touch_data: Any) -> None:
        if not hasattr(self, "slip_labels"):
            return
        label = self.slip_labels.get(finger)
        positions = [slider.value() for slider in self.sliders]
        if not self._finger_is_active(finger, positions):
            self.prev_touch.pop(finger, None)
            self.touch_history.pop(finger, None)
            if label is not None:
                self._set_label_status(label, f"{finger[0].upper()}:OFF", "gray")
            return

        flat = self._flatten_touch(touch_data)
        if not flat:
            self.touch_history.pop(finger, None)
            if label is not None:
                self._set_label_status(label, f"{finger[0].upper()}:--", "black")
            return

        contact_threshold = self.slip_contact_spin.value()
        max_pressure = max(flat)
        if max_pressure < contact_threshold:
            self.touch_history.pop(finger, None)
            if label is not None:
                self._set_label_status(label, f"{finger[0].upper()}:NO", "black")
            return

        if self.slip_window_checkbox.isChecked():
            frames = max(2, self.slip_window_frames_spin.value())
            history = self.touch_history.setdefault(finger, [])
            history.append(flat)
            if len(history) > frames:
                history.pop(0)
            if len(history) < frames:
                if label is not None:
                    self._set_label_status(label, f"{finger[0].upper()}:OK", "green")
                return
            previous = history[0]
            current = history[-1]
            mode = f"{frames}f"
        else:
            previous = self.prev_touch.get(finger)
            self.prev_touch[finger] = flat
            if not previous or len(previous) != len(flat):
                if label is not None:
                    self._set_label_status(label, f"{finger[0].upper()}:OK", "green")
                return
            current = flat
            mode = "1f"

        mag_delta = abs(sum(current) - sum(previous))
        loc_delta = self._touch_location_delta(previous, current)
        now_ms = int(time.time() * 1000)
        last_ms = self.last_slip_time.get(finger, 0)
        if (
            mag_delta >= self.slip_mag_spin.value()
            and loc_delta >= self.slip_loc_spin.value()
            and now_ms - last_ms >= self.slip_cooldown_spin.value()
        ):
            self.last_slip_time[finger] = now_ms
            if label is not None:
                self._set_label_status(label, f"{finger[0].upper()}:SLP", "red")
            self.status_updated.emit(
                "warning",
                f"Slip detected: {finger} (mode={mode}, mag={mag_delta}, loc={loc_delta:.2f})",
            )
            if self.slip_torque_boost_checkbox.isChecked():
                self._boost_finger_torque(finger)
        elif label is not None:
            self._set_label_status(label, f"{finger[0].upper()}:OK", "green")

    def _finger_is_active(self, finger: str, positions: list[int]) -> bool:
        indices = self._finger_joint_map().get(finger, [])
        if not indices:
            return False
        return not all(
            positions[idx] < self.closed_threshold
            for idx in indices
            if idx < len(positions)
        )

    def _boost_finger_torque(self, finger: str) -> None:
        values = self._current_joint_torque_values()
        indices = self._finger_joint_map().get(finger, [])
        if not indices:
            return
        step = max(1, self.slip_torque_boost_step_spin.value())
        changed = False
        for idx in indices:
            if idx >= len(values):
                continue
            boosted = min(MAX_JOINT_POSITION, values[idx] + step)
            if boosted != values[idx]:
                values[idx] = boosted
                changed = True
        if not changed:
            self._update_commanded_torque_display(
                boosted_finger=finger,
                maxed_out=True,
            )
            self.status_updated.emit(
                "info",
                f"Slip torque boost skipped: {finger} already at max torque",
            )
            return
        self.current_torque_values = values
        self._sync_per_joint_torque_sliders()
        self.sdk_manager.publish_joint_torques(values)
        boosted_values = [values[idx] for idx in indices if idx < len(values)]
        self._update_commanded_torque_display(
            boosted_finger=finger,
            boosted_values=boosted_values,
        )
        self.status_updated.emit("info", f"Slip torque boost: {finger} -> {boosted_values}")

    def _current_joint_torque_values(self) -> list[int]:
        joint_count = len(self.hand_config.joint_names)
        if len(self.current_torque_values) != joint_count:
            default_torque = self.torque_slider.value() if hasattr(self, "torque_slider") else MAX_JOINT_POSITION
            self.current_torque_values = [default_torque] * joint_count
        return list(self.current_torque_values)

    def _flatten_touch(self, data: Any) -> list[int]:
        if data is None:
            return []
        try:
            if hasattr(data, "flatten"):
                return [int(value) for value in list(data.flatten())]
            if isinstance(data, list) and data and isinstance(data[0], list):
                flat = []
                for row in data:
                    flat.extend(row)
                return [int(value) for value in flat]
            return [int(value) for value in list(data)]
        except Exception:
            return []

    def _get_max_pressure(self, data: Any) -> int:
        flat = self._flatten_touch(data)
        return max(flat) if flat else 0

    def _touch_location_delta(self, previous: list[int], current: list[int]) -> float:
        previous_loc = self._pressure_centroid(previous)
        current_loc = self._pressure_centroid(current)
        if previous_loc is None or current_loc is None:
            return 0.0
        dx = current_loc[0] - previous_loc[0]
        dy = current_loc[1] - previous_loc[1]
        return (dx * dx + dy * dy) ** 0.5

    def _pressure_centroid(self, flat: list[int]) -> tuple[float, float] | None:
        if not flat:
            return None
        rows = 12
        cols = 6
        total = sum(flat)
        if total <= 0:
            return None
        cx = 0.0
        cy = 0.0
        for idx, value in enumerate(flat[: rows * cols]):
            if value <= 0:
                continue
            row = idx // cols
            col = idx % cols
            cx += col * value
            cy += row * value
        return cx / total, cy / total

    def _set_label_status(self, label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"color: {color};")

    def update_sensor_values(self, payload: dict[str, list[Any]]) -> None:
        self.latest_sensor_values.update(payload)
        sensor_keys = ["angle", "torque", "temperature", "current"]
        if self.model == "O6" or (self.model == "L30" and self.sdk_manager.acceleration_supported):
            sensor_keys.append("acceleration")
        for sensor_key in sensor_keys:
            grouped = self._group_values_by_finger(self.latest_sensor_values.get(sensor_key))
            for finger, values in grouped.items():
                label = self.realtime_labels.get(finger, {}).get(sensor_key)
                if label is not None:
                    label.setText(self._format_cell_value(values))

    def _finger_key_from_name(self, name: str) -> str | None:
        lowered = name.lower()
        if "thumb" in lowered:
            return "thumb"
        if "index" in lowered:
            return "index"
        if "middle" in lowered:
            return "middle"
        if "ring" in lowered:
            return "ring"
        if "pinky" in lowered or "little" in lowered:
            return "pinky"
        return None

    def _group_values_by_finger(self, values: list[Any] | None) -> dict[str, list[Any]]:
        grouped = {finger: [] for finger in self.finger_order}
        if not values:
            return grouped
        for idx, name in enumerate(self.hand_config.joint_names):
            if idx >= len(values):
                break
            finger = self._finger_key_from_name(name)
            if finger:
                grouped[finger].append(values[idx])
        return grouped

    def _format_cell_value(self, values: list[Any] | Any) -> str:
        if values is None:
            return "--"
        if isinstance(values, list):
            if not values:
                return "--"
            return "[" + ", ".join(self._format_scalar(v) for v in values) + "]"
        return self._format_scalar(values)

    def _format_scalar(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.0f}"
        return str(value)

    def update_status(self, status_type: str, message: str) -> None:
        if hasattr(self, "connection_status"):
            if status_type == "error":
                self.connection_status.setText("Hand SDK Error")
                self.connection_status.setObjectName("StatusLabel")
                self.connection_status.setObjectName("StatusError")
            elif "connected" in message.lower():
                self.connection_status.setText("Hand SDK Connected")
                self.connection_status.setObjectName("StatusLabel")
                self.connection_status.setObjectName("StatusInfo")
        if not hasattr(self, "status_log"):
            return
        current_time = time.strftime("%H:%M:%S")
        log_entry = f"[{current_time}] {message}\n"
        current_log = self.status_log.text()
        if len(current_log) > 10000:
            current_log = current_log[-10000:]
        self.status_log.setText(log_entry + current_log)
        self.status_log.setObjectName("StatusLabel")
        self.status_log.setObjectName("StatusError" if status_type == "error" else "StatusInfo")
        self.status_log.style().unpolish(self.status_log)
        self.status_log.style().polish(self.status_log)

    def clear_status_log(self) -> None:
        self.status_log.setText("Log cleared")
        self.status_log.setObjectName("StatusLabel")
        self.status_log.setObjectName("StatusInfo")

    def closeEvent(self, event) -> None:
        if self.motion_test_active:
            self._finish_motion_test("Window closed", show_dialog=False)
        if self.cycle_timer and self.cycle_timer.isActive():
            self.cycle_timer.stop()
        if self.auto_grab_timer and self.auto_grab_timer.isActive():
            self.auto_grab_timer.stop()
        if self.publish_timer and self.publish_timer.isActive():
            self.publish_timer.stop()
        for timer in (
            self.live_speed_timer,
            self.live_torque_timer,
            self.live_acceleration_timer,
            self.live_joint_speed_timer,
            self.live_joint_torque_timer,
        ):
            if timer.isActive():
                timer.stop()
        self.sdk_manager.shutdown()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autodetecting RealHand SDK PyQt5 GUI controller")
    parser.add_argument("--config", default=os.getenv("REALHAND_GUI_CONFIG", DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-config", action="store_true", help="Ignore the GUI config file")
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default=None)
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--interface-type", default=None)
    parser.add_argument("--no-autodetect", action="store_true", help="Skip can0-can3 autodetection")
    parser.add_argument("--no-can-setup", action="store_true", help="Probe without running ip link setup")
    parser.add_argument("--can-bitrate", type=int, default=None, help="SocketCAN bitrate for autodetect setup")
    parser.add_argument("--no-dialog", action="store_true", help="Use CLI/env settings without showing the connection dialog")
    parser.add_argument(
        "--sensor-read-mode",
        choices=(*SENSOR_READ_MODES, *SENSOR_READ_MODE_ALIASES),
        default=None,
        help="Realtime sensor read mode for status display",
    )
    return parser.parse_args(argv)


def load_gui_config(path: str, *, enabled: bool = True) -> dict[str, Any]:
    if not enabled or not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _validated_model(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    if value not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model {value!r}; expected one of {', '.join(MODEL_CLASSES)}")
    return value


def _validated_side(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).lower()
    if value not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return value


def _validated_interface_type(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _validated_sensor_read_mode(value: Any) -> str | None:
    if value is None:
        return None
    value = normalize_sensor_read_mode(value)
    if value not in SENSOR_READ_MODES:
        raise ValueError(
            f"sensor_read_mode must be one of {', '.join(SENSOR_READ_MODES)}"
        )
    return value


def _config_poll_intervals(config: dict[str, Any]) -> dict[str, float]:
    merged = dict(POLL_INTERVALS_BY_NAME)
    raw = config.get("poll_intervals", {})
    if raw is None:
        return merged
    if not isinstance(raw, dict):
        raise ValueError("poll_intervals must be a JSON object mapping sensor names to seconds")
    for key, value in raw.items():
        if value is None:
            merged.pop(str(key), None)
            continue
        interval = float(value)
        if interval <= 0:
            raise ValueError(f"poll interval for {key!r} must be positive")
        merged[str(key)] = interval
    return merged


def build_runtime_options(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model = (
        args.model
        or os.getenv("REALHAND_GUI_MODEL")
        or config.get("model")
        or DEFAULT_MODEL
    )
    side = (
        args.side
        or os.getenv("REALHAND_GUI_SIDE")
        or config.get("side")
        or DEFAULT_SIDE
    )
    interface = (
        args.interface
        or os.getenv("REALHAND_GUI_INTERFACE")
        or config.get("interface")
        or "can0"
    )
    interface_type = (
        args.interface_type
        or os.getenv("REALHAND_GUI_INTERFACE_TYPE")
        or config.get("interface_type")
        or "socketcan"
    )
    can_bitrate = (
        args.can_bitrate
        or os.getenv("REALHAND_GUI_CAN_BITRATE")
        or config.get("can_bitrate")
        or DEFAULT_CAN_BITRATE
    )
    sensor_read_mode = (
        args.sensor_read_mode
        or os.getenv("REALHAND_GUI_SENSOR_READ_MODE")
        or config.get("sensor_read_mode")
        or DEFAULT_SENSOR_READ_MODE
    )
    model = _validated_model(model) or DEFAULT_MODEL
    side = _validated_side(side) or DEFAULT_SIDE
    sensor_read_mode = (
        _validated_sensor_read_mode(sensor_read_mode) or DEFAULT_SENSOR_READ_MODE
    )
    # The L30 CANFD analyser is not a SocketCAN device and cannot use the
    # standard serial-number probe used by the other models.
    if model == "L30":
        if args.interface_type is None and "REALHAND_GUI_INTERFACE_TYPE" not in os.environ and "interface_type" not in config:
            interface_type = "libcanbus"
        if args.interface is None and "REALHAND_GUI_INTERFACE" not in os.environ and "interface" not in config:
            interface = "0"
    show_dialog = bool(config.get("show_dialog", True)) and not args.no_dialog
    autodetect = model != "L30" and bool(config.get("autodetect", True)) and not args.no_autodetect
    setup_can = bool(config.get("setup_can", True)) and not args.no_can_setup
    return {
        "model": model,
        "side": side,
        "interface": str(interface),
        "interface_type": str(interface_type),
        "show_dialog": show_dialog,
        "autodetect": autodetect,
        "setup_can": setup_can,
        "can_bitrate": int(can_bitrate),
        "sensor_read_mode": sensor_read_mode,
        "poll_intervals": _config_poll_intervals(config),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        config = load_gui_config(args.config, enabled=not args.no_config)
        runtime = build_runtime_options(args, config)
    except Exception as exc:
        print(f"Failed to load GUI config: {exc}")
        return 1

    app = QApplication(sys.argv)

    selection = {
        "model": runtime["model"],
        "side": runtime["side"],
        "interface": runtime["interface"],
        "interface_type": runtime["interface_type"],
        "sensor_read_mode": runtime["sensor_read_mode"],
    }

    autodetect_handled = False
    if runtime["autodetect"]:
        hands, messages = autodetect_hands(
            interface_type=runtime["interface_type"],
            bitrate=runtime["can_bitrate"],
            setup_can=runtime["setup_can"],
        )
        # Standard autodetection is SocketCAN-only.  If it found no hand,
        # try the USB CANFD analyser used by L30 before opening manual setup.
        if not hands:
            l30_hands, l30_messages = autodetect_l30_hand()
            hands.extend(l30_hands)
            messages.extend(l30_messages)
        for message in messages:
            print(f"[autodetect] {message}")
        selected_hand, autodetect_handled = choose_detected_hand(hands)
        if selected_hand is not None:
            selection = selection_from_detected_hand(selected_hand)
            print(f"[autodetect] selected {selected_hand.label()}")
        elif hands:
            return 0
        elif runtime["show_dialog"]:
            detail = "\n".join(messages[-8:])
            QMessageBox.warning(
                None,
                "No RealHand Autodetected",
                "No supported RealHand was detected on can0 through can3.\n\n"
                "Manual connection settings will be shown next."
                + (f"\n\nLast probe messages:\n{detail}" if detail else ""),
            )

    if runtime["show_dialog"] and not autodetect_handled:
        dialog = ConnectionDialog(runtime)
        if dialog.exec_() != QDialog.Accepted:
            return 0
        selection = dialog.selection()

    try:
        sdk_manager = HandSdkManager(
            model=selection["model"],
            side=selection["side"],
            interface_name=selection["interface"],
            interface_type=selection["interface_type"],
            poll_intervals=runtime["poll_intervals"],
            sensor_read_mode=selection.get("sensor_read_mode", runtime["sensor_read_mode"]),
        )
    except Exception as exc:
        QMessageBox.critical(None, "Connection Failed", str(exc))
        return 1

    window = HandControlGUI(sdk_manager)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

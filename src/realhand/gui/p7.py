"""Native Qt control interface for a RealHand P7 arm.

This is deliberately independent of ROS2: it connects to the P7's RBot TCP
controller through :class:`realhand.arm.p7.P7`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The P7 GUI requires PyQt5. Install it with `pip install realhand[gui]`."
    ) from exc

from realhand.arm.p7 import P7
from realhand.arm.p7.consts import (
    DEFAULT_ACCELERATION,
    DEFAULT_TCP_HOST,
    DEFAULT_VELOCITY,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    NUM_JOINTS,
)

STYLESHEET = """
QWidget { font-size: 12px; }
QGroupBox { border: 1px solid #cccccc; border-radius: 6px; margin-top: 8px; padding: 8px; }
QGroupBox::title { color: #165dff; font-weight: bold; left: 10px; padding: 0 4px; }
QPushButton { background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background: #e5e7eb; }
QPushButton#danger { background: #fff1f0; border-color: #ffccc7; color: #cf1322; }
QLabel#status { background: #f0f7ff; border-radius: 4px; color: #0050b3; padding: 6px; }
QLabel#status[error="true"] { background: #fff1f0; color: #a8071a; }
QPlainTextEdit { font-family: monospace; background: #fafafa; border: 1px solid #d1d5db; }
QSlider::groove:horizontal { background: #d1d5db; border-radius: 3px; height: 6px; }
QSlider::handle:horizontal { background: #165dff; border-radius: 7px; height: 14px; margin: -4px 0; width: 14px; }
"""


class JointRow(QtWidgets.QWidget):
    """One labelled slider and numeric control with a floating-point range."""

    def __init__(self, label: str, minimum: float, maximum: float, step: float, suffix: str) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        name = QtWidgets.QLabel(label)
        name.setMinimumWidth(110)
        self._scale = 1000
        self.value_box = QtWidgets.QDoubleSpinBox()
        self.value_box.setRange(minimum, maximum)
        self.value_box.setDecimals(3)
        self.value_box.setSingleStep(step)
        self.value_box.setSuffix(f" {suffix}")
        self.value_box.setMinimumWidth(140)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimumWidth(260)
        self._set_slider_range(minimum, maximum)
        layout.addWidget(name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_box)
        layout.addStretch(1)
        self.slider.valueChanged.connect(self._slider_changed)
        self.value_box.valueChanged.connect(self._spin_changed)

    def value(self) -> float:
        return float(self.value_box.value())

    def set_value(self, value: float) -> None:
        self.value_box.setValue(value)

    def set_range(self, minimum: float, maximum: float) -> None:
        """Update the displayed range while keeping the current value valid."""
        self.value_box.setRange(minimum, maximum)
        self._set_slider_range(minimum, maximum)
        self._spin_changed(self.value_box.value())

    def _set_slider_range(self, minimum: float, maximum: float) -> None:
        self.slider.setRange(
            int(round(minimum * self._scale)), int(round(maximum * self._scale))
        )

    def _slider_changed(self, value: int) -> None:
        blocker = QtCore.QSignalBlocker(self.value_box)
        self.value_box.setValue(value / self._scale)
        del blocker

    def _spin_changed(self, value: float) -> None:
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(int(round(value * self._scale)))
        del blocker


class P7ControlPanel(QtWidgets.QWidget):
    """P7 controls modelled after the ROS2 SDK's Arm tab."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.arm: P7 | None = None
        self.joint_rows: list[JointRow] = []
        self.velocity_rows: list[JointRow] = []
        self.acceleration_rows: list[JointRow] = []
        self._build_ui()
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.refresh_state)
        self.poll_timer.start(500)

    def _build_ui(self) -> None:
        self.host_edit = QtWidgets.QLineEdit(DEFAULT_TCP_HOST)
        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["left", "right"])
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_arm)
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_arm)
        self.disconnect_button.setEnabled(False)
        connection = QtWidgets.QHBoxLayout()
        connection.addWidget(QtWidgets.QLabel("Controller host"))
        connection.addWidget(self.host_edit, 1)
        connection.addWidget(QtWidgets.QLabel("Arm"))
        connection.addWidget(self.side_combo)
        connection.addWidget(self.connect_button)
        connection.addWidget(self.disconnect_button)

        tabs = QtWidgets.QTabWidget()
        self.joint_rows = self._add_control_tab(tabs, "Joint targets", -6.283, 6.283, 0.01, "rad")
        self.velocity_rows = self._add_control_tab(tabs, "Velocity", 0, MAX_VELOCITY, 0.05, "rad/s")
        self.acceleration_rows = self._add_control_tab(tabs, "Acceleration", 0, MAX_ACCELERATION, 0.1, "rad/s²")
        for row in self.velocity_rows:
            row.set_value(DEFAULT_VELOCITY)
        for row in self.acceleration_rows:
            row.set_value(DEFAULT_ACCELERATION)

        send_joints = QtWidgets.QPushButton("Move Joints")
        send_joints.clicked.connect(self.move_joints)
        send_velocities = QtWidgets.QPushButton("Set Velocity")
        send_velocities.clicked.connect(self.set_velocities)
        send_accelerations = QtWidgets.QPushButton("Set Acceleration")
        send_accelerations.clicked.connect(self.set_accelerations)
        controls = QtWidgets.QHBoxLayout()
        for button in (send_joints, send_velocities, send_accelerations):
            controls.addWidget(button)
        controls.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        action_specs = (
            ("Enable", "Arm enabled", lambda arm: arm.enable()),
            ("Disable", "Arm disabled", lambda arm: arm.disable()),
            ("Home", "Arm homed", lambda arm: arm.home(blocking=False)),
            ("Reset Errors", "Errors reset", lambda arm: arm.reset_error()),
            ("Emergency Stop", "Emergency stop active", lambda arm: arm.emergency_stop()),
            ("Resume", "Emergency stop released", lambda arm: arm.resume_from_emergency_stop()),
        )
        for label, result, callback in action_specs:
            button = QtWidgets.QPushButton(label)
            if label == "Emergency Stop":
                button.setObjectName("danger")
            button.clicked.connect(lambda _checked=False, result=result, callback=callback: self.run_action(result, callback))
            actions.addWidget(button)
        actions.addStretch(1)

        self.status = QtWidgets.QLabel("Connect to a P7 controller to begin")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.state_text = QtWidgets.QPlainTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMaximumHeight(220)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(connection)
        layout.addWidget(self.status)
        layout.addWidget(tabs, 1)
        layout.addLayout(controls)
        layout.addLayout(actions)
        layout.addWidget(QtWidgets.QLabel("Live arm state and pose"))
        layout.addWidget(self.state_text)

    def _add_control_tab(self, tabs: QtWidgets.QTabWidget, title: str, minimum: float, maximum: float, step: float, suffix: str) -> list[JointRow]:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        rows = []
        for index in range(NUM_JOINTS):
            row = JointRow(f"Joint {index + 1}", minimum, maximum, step, suffix)
            rows.append(row)
            layout.addWidget(row)
        layout.addStretch(1)
        tabs.addTab(container, title)
        return rows

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.status.setText(message)
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def connect_arm(self) -> None:
        self.disconnect_arm(silent=True)
        try:
            self.arm = P7(self.side_combo.currentText(), tcp_host=self.host_edit.text().strip())
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.host_edit.setEnabled(False)
            self.side_combo.setEnabled(False)
            self._set_joint_limits(self.arm.get_joint_limits())
            self._set_status(f"Connected to {self.arm.side} P7 at {self.arm.tcp_host}")
            self.refresh_state()
        except Exception as exc:
            self.arm = None
            self._set_status(f"Connection failed: {exc}", error=True)

    def disconnect_arm(self, *, silent: bool = False) -> None:
        arm, self.arm = self.arm, None
        if arm is not None:
            try:
                arm.close()
            except Exception as exc:
                if not silent:
                    self._set_status(f"Disconnect failed: {exc}", error=True)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.host_edit.setEnabled(True)
        self.side_combo.setEnabled(True)
        if not silent:
            self._set_status("Disconnected")

    def _set_joint_limits(self, limits: list[tuple[float, float]]) -> None:
        """Apply the connected controller's physical joint limits to the sliders."""
        if len(limits) != len(self.joint_rows):
            raise ValueError(f"Expected {len(self.joint_rows)} joint limits, got {len(limits)}")
        for row, (lower, upper) in zip(self.joint_rows, limits):
            row.set_range(lower, upper)

    def _require_arm(self) -> P7 | None:
        if self.arm is None:
            self._set_status("Connect to an arm first", error=True)
        return self.arm

    def run_action(self, success: str, action: Callable[[P7], None]) -> None:
        arm = self._require_arm()
        if arm is None:
            return
        try:
            action(arm)
            self._set_status(success)
            self.refresh_state()
        except Exception as exc:
            self._set_status(f"Command failed: {exc}", error=True)

    def move_joints(self) -> None:
        self.run_action("Joint motion started", lambda arm: arm.move_j([row.value() for row in self.joint_rows], blocking=False))

    def set_velocities(self) -> None:
        self.run_action("Velocity limits applied", lambda arm: arm.set_velocities([row.value() for row in self.velocity_rows]))

    def set_accelerations(self) -> None:
        self.run_action("Acceleration limits applied", lambda arm: arm.set_accelerations([row.value() for row in self.acceleration_rows]))

    def refresh_state(self) -> None:
        arm = self.arm
        if arm is None:
            return
        try:
            state = arm.get_state()
            payload: dict[str, Any] = {
                "pose": state.pose.model_dump(),
                "joint_angles_rad": [round(item.angle, 4) for item in state.joint_angles],
                "joint_velocities_rad_s": [round(item.velocity, 4) for item in state.joint_velocities],
                "joint_torques": [round(item.torque, 4) for item in state.joint_torques or []],
                "temperatures": [round(item.temperature, 1) for item in state.joint_temperatures or []],
                "moving": arm.is_moving(),
            }
            self.state_text.setPlainText(json.dumps(payload, indent=2))
        except Exception as exc:
            self._set_status(f"State read failed: {exc}", error=True)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self.poll_timer.stop()
        self.disconnect_arm(silent=True)
        event.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RealHand P7 Control Interface")
        self.setMinimumSize(760, 680)
        self.setStyleSheet(STYLESHEET)
        self.setCentralWidget(P7ControlPanel(self))


def main() -> None:
    """Start the P7 desktop GUI."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    main()

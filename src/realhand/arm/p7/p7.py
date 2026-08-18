from __future__ import annotations

import functools
import math
import threading
import time
from typing import Any, Literal

import numpy as np

from realhand.arm.common import ControlMode, Pose, State
from realhand.arm.common.model import (
    AccelerationState,
    AngleState,
    TemperatureState,
    TorqueState,
    VelocityState,
)
from realhand.arm.p7.consts import (
    DEFAULT_ACCELERATION,
    DEFAULT_MOVE_TIMEOUT,
    DEFAULT_MOVE_TOLERANCE,
    DEFAULT_TCP_HOST,
    DEFAULT_VELOCITY,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    MIN_ACCELERATION,
    MIN_VELOCITY,
    MOVE_L_DEFAULT_ACCELERATION,
    MOVE_L_DEFAULT_ANGULAR_ACCELERATION,
    MOVE_L_DEFAULT_MAX_ANGULAR_VELOCITY,
    MOVE_L_DEFAULT_MAX_VELOCITY,
    MOVE_L_MAX_ACCELERATION,
    MOVE_L_MAX_ANGULAR_ACCELERATION,
    MOVE_L_MAX_MAX_ANGULAR_VELOCITY,
    MOVE_L_MAX_MAX_VELOCITY,
    NUM_JOINTS,
)
from realhand.exceptions import StateError, TimeoutError, ValidationError
from realhand.motion_timer import MotionTimer

_RBOT_ROBOTS: dict[str, tuple[Any, int]] = {}
_RBOT_ROBOT_LOCK = threading.Lock()


def _load_rbot_symbols():
    from .rbot.rbot_api import RbotArm, RbotEuler, RbotPosition
    from .rbot.rbot_robot import RbotRobot

    return RbotRobot, RbotArm, RbotPosition, RbotEuler


def _acquire_rbot_robot(tcp_host: str):
    with _RBOT_ROBOT_LOCK:
        entry = _RBOT_ROBOTS.get(tcp_host)
        if entry is not None:
            robot, ref_count = entry
            _RBOT_ROBOTS[tcp_host] = (robot, ref_count + 1)
            return robot

        RbotRobot, _, _, _ = _load_rbot_symbols()
        robot = RbotRobot(tcp_host)
        if not robot.connect():
            raise StateError(f"Failed to connect to P7 RBot controller at {tcp_host}")
        _RBOT_ROBOTS[tcp_host] = (robot, 1)
        return robot


def _release_rbot_robot(tcp_host: str) -> None:
    with _RBOT_ROBOT_LOCK:
        entry = _RBOT_ROBOTS.get(tcp_host)
        if entry is None:
            return

        robot, ref_count = entry
        if ref_count > 1:
            _RBOT_ROBOTS[tcp_host] = (robot, ref_count - 1)
            return

        try:
            robot.disconnect()
        finally:
            _RBOT_ROBOTS.pop(tcp_host, None)


def _guard_not_moving(method):
    @functools.wraps(method)
    def wrapper(self: P7, *args, **kwargs):
        if self.is_moving():
            raise StateError("Cannot start new motion while arm is moving.")
        return method(self, *args, **kwargs)

    return wrapper


class P7:
    """RealHand P7 arm interface backed by the RBot TCP controller."""

    def __init__(
        self,
        side: Literal["left", "right"],
        tcp_host: str | None = None,
        *,
        interface_name: str | None = None,
        interface_type: str = "rbot",
        tcp_offset: list[float] | None = None,
        world_frame: Literal["urdf", "maestro"] = "urdf",
        default_velocity: float = DEFAULT_VELOCITY,
        default_acceleration: float = DEFAULT_ACCELERATION,
        enable_on_start: bool = False,
        check_joint_limits: bool = True,
        move_timeout_s: float = DEFAULT_MOVE_TIMEOUT,
        move_tolerance_rad: float = DEFAULT_MOVE_TOLERANCE,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if interface_type != "rbot":
            raise ValueError(f"interface_type must be 'rbot', got {interface_type!r}")
        if world_frame not in ("urdf", "maestro"):
            raise ValueError(
                f"world_frame must be 'urdf' or 'maestro', got {world_frame!r}"
            )

        _, RbotArm, _, _ = _load_rbot_symbols()
        self.side = side
        self.tcp_host = tcp_host or interface_name or DEFAULT_TCP_HOST
        self.interface_name = self.tcp_host
        self.interface_type = interface_type
        self.tcp_offset = list(tcp_offset) if tcp_offset is not None else [0.0, 0.0, 0.0]
        self.world_frame = world_frame
        self.check_joint_limits = check_joint_limits
        self.move_timeout_s = float(move_timeout_s)
        self.move_tolerance_rad = float(move_tolerance_rad)
        self.rbot_arm = RbotArm.LEFT_ARM if side == "left" else RbotArm.RIGHT_ARM
        self.robot = _acquire_rbot_robot(self.tcp_host)

        self._control_mode: ControlMode | None = None
        self._motion_timer = MotionTimer()
        self._closed = False
        self._control_velocities = [float(default_velocity)] * NUM_JOINTS
        self._control_accelerations = [float(default_acceleration)] * NUM_JOINTS
        self._control_angles = self.get_angles()
        self._joint_limits = self._read_joint_limits()

        if enable_on_start:
            self.reset_error()
            self.enable()

    def _read_joint_limits(self) -> list[tuple[float, float]]:
        success, lower, upper = self.robot.get_joint_limit(self.rbot_arm)
        if not success or len(lower) != NUM_JOINTS or len(upper) != NUM_JOINTS:
            success, lower, upper = self.robot.get_default_joint_limit(self.rbot_arm)
        if not success or len(lower) != NUM_JOINTS or len(upper) != NUM_JOINTS:
            lower = [-math.pi] * NUM_JOINTS
            upper = [math.pi] * NUM_JOINTS
        return [
            (min(float(lo), float(hi)), max(float(lo), float(hi)))
            for lo, hi in zip(lower, upper)
        ]

    def start_polling(self, intervals: dict[Any, float] | None = None) -> None:
        return None

    def stop_polling(self) -> None:
        return None

    def is_moving(self) -> bool:
        return self._motion_timer.is_moving()

    def wait_motion_done(self) -> None:
        self._motion_timer.wait_done()

    def set_control_mode(self, mode: ControlMode) -> None:
        if mode is not ControlMode.PP:
            raise ValidationError(f"P7 only supports {ControlMode.PP}")
        self._control_mode = mode

    def enable(self) -> None:
        if not self.robot.enable_arm(self.rbot_arm, True):
            raise StateError(f"Failed to enable P7 {self.side}: {self.robot.get_last_error()}")

    def disable(self) -> None:
        if not self.robot.enable_arm(self.rbot_arm, False):
            raise StateError(f"Failed to disable P7 {self.side}: {self.robot.get_last_error()}")

    def reset_error(self) -> None:
        if not self.robot.clear_errors():
            raise StateError(f"Failed to clear P7 errors: {self.robot.get_last_error()}")

    def emergency_stop(self, enable: bool = True) -> None:
        if not self.robot.emergency_stop(self.rbot_arm, enable):
            action = "emergency stop" if enable else "resume from emergency stop"
            raise StateError(
                f"Failed to {action} P7 {self.side}: {self.robot.get_last_error()}"
            )
        if enable:
            self._motion_timer.cancel()

    def resume_from_emergency_stop(self) -> None:
        self.emergency_stop(False)

    def _validate_joint_values(
        self,
        values: list[float],
        *,
        name: str,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        if len(values) != NUM_JOINTS:
            raise ValueError(f"{name} count must be {NUM_JOINTS}, got {len(values)}")
        if low is None or high is None:
            return
        for i, value in enumerate(values):
            if not (low <= value <= high):
                raise ValidationError(
                    f"Joint {i} {name} {value} out of range [{low}, {high}]"
                )

    def _set_angles(self, angles: list[float], *, check_limits: bool = True) -> None:
        self._validate_joint_values(angles, name="angles")
        if check_limits:
            for i, (angle, (lo, hi)) in enumerate(zip(angles, self._joint_limits)):
                if not (lo <= angle <= hi):
                    raise ValidationError(
                        f"Joint {i} angle {angle:.4f} rad out of range [{lo:.4f}, {hi:.4f}]"
                    )
        if not self.robot.joint_follow(self.rbot_arm, [float(v) for v in angles]):
            raise StateError(
                f"joint_follow failed for P7 {self.side}: {self.robot.get_last_error()}"
            )
        self._control_angles = [float(v) for v in angles]

    def set_velocities(self, velocities: list[float]) -> None:
        self._validate_joint_values(
            velocities, name="velocity", low=MIN_VELOCITY, high=MAX_VELOCITY
        )
        self._control_velocities = [float(v) for v in velocities]

    def set_accelerations(self, accelerations: list[float]) -> None:
        self._validate_joint_values(
            accelerations,
            name="acceleration",
            low=MIN_ACCELERATION,
            high=MAX_ACCELERATION,
        )
        self._control_accelerations = [float(v) for v in accelerations]

    def get_state(self) -> State:
        angles, velocities, efforts, temperatures, timestamp = self._read_arm_state()
        return State(
            pose=self.get_pose(),
            joint_angles=[
                AngleState(angle=angle, timestamp=timestamp) for angle in angles
            ],
            joint_control_angles=[
                AngleState(angle=angle, timestamp=timestamp)
                for angle in self._control_angles
            ],
            joint_velocities=[
                VelocityState(velocity=velocity, timestamp=timestamp)
                for velocity in velocities
            ],
            joint_control_velocities=[
                VelocityState(velocity=velocity, timestamp=timestamp)
                for velocity in self._control_velocities
            ],
            joint_control_acceleration=[
                AccelerationState(acceleration=acceleration, timestamp=timestamp)
                for acceleration in self._control_accelerations
            ],
            joint_torques=[
                TorqueState(torque=effort, timestamp=timestamp) for effort in efforts
            ],
            joint_temperatures=[
                TemperatureState(temperature=temperature, timestamp=timestamp)
                for temperature in temperatures
            ],
        )

    def _read_arm_state(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float], float]:
        state = self.robot.get_state()
        if state is None:
            now = time.time()
            return (
                self._control_angles.copy(),
                [0.0] * NUM_JOINTS,
                [0.0] * NUM_JOINTS,
                [0.0] * NUM_JOINTS,
                now,
            )

        arm_state = state.left_arm if self.side == "left" else state.right_arm
        timestamp = float(arm_state.sec) + float(arm_state.nanosec) / 1_000_000_000.0
        return (
            [float(v) for v in arm_state.get_joints_list()],
            [float(v) for v in arm_state.get_velocities_list()],
            [float(v) for v in arm_state.get_efforts_list()],
            [float(v) for v in arm_state.get_temperatures_list()],
            timestamp,
        )

    def get_angles(self) -> list[float]:
        positions = self.robot.get_joint_positions(self.rbot_arm)
        if positions is None or len(positions) != NUM_JOINTS:
            return getattr(self, "_control_angles", [0.0] * NUM_JOINTS).copy()
        return [float(v) for v in positions]

    def get_control_angles(self) -> list[float]:
        return self._control_angles.copy()

    def get_velocities(self) -> list[float]:
        _, velocities, _, _, _ = self._read_arm_state()
        return velocities

    def get_control_velocities(self) -> list[float]:
        return self._control_velocities.copy()

    def get_control_acceleration(self) -> list[float]:
        return self._control_accelerations.copy()

    def get_torques(self) -> list[float]:
        _, _, efforts, _, _ = self._read_arm_state()
        return efforts

    def get_temperatures(self) -> list[float]:
        _, _, _, temperatures, _ = self._read_arm_state()
        return temperatures

    def get_pose(self) -> Pose:
        pose = self.robot.get_cartesian_pose(self.rbot_arm)
        if pose is None:
            raise StateError(f"Failed to read P7 {self.side} Cartesian pose")
        position, euler = pose
        return Pose(
            x=float(position.x),
            y=float(position.y),
            z=float(position.z),
            rx=float(euler.x),
            ry=float(euler.y),
            rz=float(euler.z),
        )

    def home(self, *, blocking: bool = True) -> None:
        self.move_j([0.0] * NUM_JOINTS, blocking=blocking)

    @_guard_not_moving
    def move_j(self, target_joints: list[float], *, blocking: bool = True) -> None:
        self._validate_joint_values(target_joints, name="angles")
        if self.check_joint_limits:
            for i, (angle, (lo, hi)) in enumerate(zip(target_joints, self._joint_limits)):
                if not (lo <= angle <= hi):
                    raise ValidationError(
                        f"Joint {i} angle {angle:.4f} rad out of range [{lo:.4f}, {hi:.4f}]"
                    )

        target = [float(v) for v in target_joints]
        duration = self._move_duration(
            self.get_angles(),
            target,
            self._control_velocities,
            self._control_accelerations,
        )
        self._motion_timer.start(duration)
        if not self.robot.move_to_joint_target(
            self.rbot_arm,
            target,
            speed=max(self._control_velocities),
            accel=max(self._control_accelerations),
            block=False,
        ):
            self._motion_timer.cancel()
            raise StateError(
                f"move_joint failed for P7 {self.side}: {self.robot.get_last_error()}"
            )
        self._control_angles = target
        if blocking:
            self._wait_until_angles(target, self.move_timeout_s, self.move_tolerance_rad)
            self.wait_motion_done()

    @_guard_not_moving
    def move_p(
        self,
        target_pose: Pose,
        *,
        current_angles: list[float] | None = None,
        blocking: bool = True,
    ) -> None:
        _, _, RbotPosition, RbotEuler = _load_rbot_symbols()
        position = RbotPosition(target_pose.x, target_pose.y, target_pose.z)
        euler = RbotEuler(target_pose.rx, target_pose.ry, target_pose.rz)
        seed = current_angles if current_angles is not None else self.get_angles()
        try:
            target_joints = self.inverse_kinematics(target_pose, current_angles=seed)
            duration = self._move_duration(
                self.get_angles(),
                target_joints,
                self._control_velocities,
                self._control_accelerations,
            )
        except RuntimeError:
            duration = 0.0
        self._motion_timer.start(duration)
        if not self.robot.move_to_pose_target(
            self.rbot_arm,
            position,
            euler,
            speed=max(self._control_velocities),
            accel=max(self._control_accelerations),
            block=blocking,
        ):
            self._motion_timer.cancel()
            raise StateError(
                f"move_pose failed for P7 {self.side}: {self.robot.get_last_error()}"
            )
        if blocking:
            self.wait_motion_done()

    @_guard_not_moving
    def move_l(
        self,
        target_pose: Pose,
        *,
        max_velocity: float = MOVE_L_DEFAULT_MAX_VELOCITY,
        max_angular_velocity: float = MOVE_L_DEFAULT_MAX_ANGULAR_VELOCITY,
        acceleration: float = MOVE_L_DEFAULT_ACCELERATION,
        angular_acceleration: float = MOVE_L_DEFAULT_ANGULAR_ACCELERATION,
        current_pose: Pose | None = None,
        current_angles: list[float] | None = None,
    ) -> None:
        if not (0 < max_velocity <= MOVE_L_MAX_MAX_VELOCITY):
            raise ValidationError(
                f"max_velocity {max_velocity} out of range (0, {MOVE_L_MAX_MAX_VELOCITY}]"
            )
        if not (0 < max_angular_velocity <= MOVE_L_MAX_MAX_ANGULAR_VELOCITY):
            raise ValidationError(
                f"max_angular_velocity {max_angular_velocity} out of range "
                f"(0, {MOVE_L_MAX_MAX_ANGULAR_VELOCITY}]"
            )
        if not (0 < acceleration <= MOVE_L_MAX_ACCELERATION):
            raise ValidationError(
                f"acceleration {acceleration} out of range (0, {MOVE_L_MAX_ACCELERATION}]"
            )
        if not (0 < angular_acceleration <= MOVE_L_MAX_ANGULAR_ACCELERATION):
            raise ValidationError(
                f"angular_acceleration {angular_acceleration} out of range "
                f"(0, {MOVE_L_MAX_ANGULAR_ACCELERATION}]"
            )

        _, _, RbotPosition, RbotEuler = _load_rbot_symbols()
        position = RbotPosition(target_pose.x, target_pose.y, target_pose.z)
        euler = RbotEuler(target_pose.rx, target_pose.ry, target_pose.rz)
        if not self.robot.linear_move_to_pose(
            self.rbot_arm,
            position,
            euler,
            speed=max_velocity,
            accel=acceleration,
            block=True,
        ):
            raise StateError(
                f"move_linear failed for P7 {self.side}: {self.robot.get_last_error()}"
            )
        self._motion_timer.cancel()

    def _wait_until_angles(
        self,
        target_joints: list[float],
        timeout_s: float,
        tolerance_rad: float,
        poll_period_s: float = 0.05,
    ) -> None:
        target = np.asarray(target_joints, dtype=float)
        deadline = time.time() + timeout_s
        last_error = math.inf
        while time.time() < deadline:
            current = np.asarray(self.get_angles(), dtype=float)
            last_error = float(np.max(np.abs(current - target)))
            if last_error <= tolerance_rad:
                return
            time.sleep(poll_period_s)
        raise TimeoutError(
            f"P7 {self.side} joint move timed out; max error {last_error:.4f} rad"
        )

    def forward_kinematics(self, angles: list[float]) -> Pose:
        self._validate_joint_values(angles, name="angles")
        result = self.robot.compute_forward_kinematics(self.rbot_arm, angles)
        if result is None:
            raise RuntimeError("P7 forward kinematics failed")
        position, euler = result
        return Pose(
            x=float(position.x),
            y=float(position.y),
            z=float(position.z),
            rx=float(euler.x),
            ry=float(euler.y),
            rz=float(euler.z),
        )

    def inverse_kinematics(
        self,
        pose: Pose,
        *,
        current_angles: list[float] | None = None,
    ) -> list[float]:
        _, _, RbotPosition, RbotEuler = _load_rbot_symbols()
        seed = current_angles if current_angles is not None else self.get_angles()
        self._validate_joint_values(seed, name="angles")
        position = RbotPosition(pose.x, pose.y, pose.z)
        euler = RbotEuler(pose.rx, pose.ry, pose.rz)
        result = self.robot.compute_inverse_kinematics(
            self.rbot_arm, position, euler, initial_joints=seed
        )
        if result is None:
            raise RuntimeError("P7 inverse kinematics failed to converge")
        return [float(v) for v in result]

    def set_joint_limits(self, limits: list[tuple[float, float]]) -> None:
        if len(limits) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} limits, got {len(limits)}")
        for i, (lower, upper) in enumerate(limits):
            if lower > upper:
                raise ValueError(f"Joint {i} lower limit must be <= upper limit")
        self._joint_limits = [(float(lower), float(upper)) for lower, upper in limits]

    def get_joint_limits(self) -> list[tuple[float, float]]:
        return self._joint_limits.copy()

    def calibrate_zero(self) -> None:
        if not self.robot.set_zero(self.rbot_arm):
            raise StateError(
                f"Failed to calibrate P7 {self.side} zero: {self.robot.get_last_error()}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self.wait_motion_done()
        _release_rbot_robot(self.tcp_host)
        self._closed = True

    def __enter__(self) -> P7:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @staticmethod
    def _trapezoidal_duration(
        distance: float, speed: float, acceleration: float
    ) -> float:
        if distance < 1e-12:
            return 0.0
        v = abs(speed)
        a = abs(acceleration)
        if a < 1e-12 or v < 1e-12:
            return 0.0
        t_acc = v / a
        d_acc = 0.5 * a * t_acc**2
        if 2 * d_acc >= distance:
            return 2.0 * math.sqrt(distance / a)
        t_cruise = (distance - 2 * d_acc) / v
        return 2 * t_acc + t_cruise

    def _move_duration(
        self,
        current_angles: list[float],
        target_angles: list[float],
        control_speeds: list[float],
        control_accelerations: list[float],
    ) -> float:
        return max(
            (
                self._trapezoidal_duration(abs(tgt - cur), v, a)
                for cur, tgt, v, a in zip(
                    current_angles,
                    target_angles,
                    control_speeds,
                    control_accelerations,
                )
            ),
            default=0.0,
        )

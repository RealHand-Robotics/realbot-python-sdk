import os

from realhand import P7
from realhand.exceptions import ValidationError

TCP_HOST = os.environ.get("P7_TCP_HOST", "192.168.10.21")
TARGET_ANGLES = [
    float(value)
    for value in os.environ.get("P7_LEFT_TARGET_ANGLES", "0,0,0,0,0,0,0").split(",")
]
VELOCITY = float(os.environ.get("P7_MOVE_VELOCITY", "0.5"))
ACCELERATION = float(os.environ.get("P7_MOVE_ACCELERATION", "1.0"))


with P7(side="left", tcp_host=TCP_HOST) as arm:
    print("Left P7 joint limits:")
    for index, (lower, upper) in enumerate(arm.get_joint_limits(), start=1):
        print(f"joint {index}: {lower:.6f} to {upper:.6f} rad")

    print(f"\nTarget angles: {TARGET_ANGLES}")

    arm.enable()
    arm.set_velocities([VELOCITY] * 7)
    arm.set_accelerations([ACCELERATION] * 7)

    try:
        arm.move_j(TARGET_ANGLES, blocking=True)
    except ValidationError as exc:
        print(f"Target rejected: {exc}")
        raise

    print("Move complete")

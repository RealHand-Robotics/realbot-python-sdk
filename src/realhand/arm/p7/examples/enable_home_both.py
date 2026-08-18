import os

from realhand import P7

LEFT_TCP_HOST = os.environ.get(
    "P7_LEFT_TCP_HOST", os.environ.get("P7_TCP_HOST", "192.168.10.21")
)
RIGHT_TCP_HOST = os.environ.get("P7_RIGHT_TCP_HOST", LEFT_TCP_HOST)
HOME_VELOCITY = float(os.environ.get("P7_HOME_VELOCITY", "0.5"))
HOME_ACCELERATION = float(os.environ.get("P7_HOME_ACCELERATION", "1.0"))


with (
    P7(side="left", tcp_host=LEFT_TCP_HOST) as left_arm,
    P7(side="right", tcp_host=RIGHT_TCP_HOST) as right_arm,
):
    print("Enabling left and right P7 arms")
    left_arm.enable()
    right_arm.enable()

    left_arm.set_velocities([HOME_VELOCITY] * 7)
    right_arm.set_velocities([HOME_VELOCITY] * 7)
    left_arm.set_accelerations([HOME_ACCELERATION] * 7)
    right_arm.set_accelerations([HOME_ACCELERATION] * 7)

    print("Moving left and right P7 arms to home")
    left_arm.home(blocking=False)
    right_arm.home(blocking=False)

    left_arm.wait_motion_done()
    right_arm.wait_motion_done()
    print("Home command complete")

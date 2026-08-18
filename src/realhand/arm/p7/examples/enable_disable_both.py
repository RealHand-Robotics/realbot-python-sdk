import os
import time

from realhand import P7

LEFT_TCP_HOST = os.environ.get("P7_LEFT_TCP_HOST", os.environ.get("P7_TCP_HOST", "192.168.10.21"))
RIGHT_TCP_HOST = os.environ.get("P7_RIGHT_TCP_HOST", LEFT_TCP_HOST)


with (
    P7(side="left", tcp_host=LEFT_TCP_HOST) as left_arm,
    P7(side="right", tcp_host=RIGHT_TCP_HOST) as right_arm,
):
    print("Disabling left and right P7 arms")
    left_arm.disable()
    right_arm.disable()
    time.sleep(1.0)

    print("Enabling left and right P7 arms")
    left_arm.enable()
    right_arm.enable()
    time.sleep(1.0)

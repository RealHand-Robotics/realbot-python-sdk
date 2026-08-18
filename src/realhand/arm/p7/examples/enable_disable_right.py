import os
import time

from realhand import P7

TCP_HOST = os.environ.get("P7_TCP_HOST", "192.168.10.21")


with P7(side="right", tcp_host=TCP_HOST) as arm:
    print("Disabling right P7 arm")
    arm.disable()
    time.sleep(1.0)

    print("Enabling right P7 arm")
    arm.enable()
    time.sleep(1.0)

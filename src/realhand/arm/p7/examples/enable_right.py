import os

from realhand import P7

TCP_HOST = os.environ.get("P7_TCP_HOST", "192.168.10.21")


with P7(side="right", tcp_host=TCP_HOST) as arm:
    print("Enabling right P7 arm")
    arm.enable()

import os

from realhand import P7

TCP_HOST = os.environ.get("P7_TCP_HOST", "192.168.10.21")


with P7(side="left", tcp_host=TCP_HOST) as arm:
    print("Enabling left P7 arm")
    arm.enable()

    print("Sending emergency stop")
    arm.emergency_stop()

    print("Resuming from emergency stop")
    arm.resume_from_emergency_stop()

    print("Re-enabling after emergency stop resume")
    arm.enable()

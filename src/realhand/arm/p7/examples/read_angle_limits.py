import math
import os

from realhand import P7

LEFT_TCP_HOST = os.environ.get(
    "P7_LEFT_TCP_HOST", os.environ.get("P7_TCP_HOST", "192.168.10.21")
)
RIGHT_TCP_HOST = os.environ.get("P7_RIGHT_TCP_HOST", LEFT_TCP_HOST)


for side, tcp_host in (("left", LEFT_TCP_HOST), ("right", RIGHT_TCP_HOST)):
    with P7(side=side, tcp_host=tcp_host) as arm:
        print(f"\n{side} P7 angle limits")
        for index, (lower, upper) in enumerate(arm.get_joint_limits(), start=1):
            print(
                f"joint {index}: "
                f"{lower:.6f} to {upper:.6f} rad "
                f"({math.degrees(lower):.2f} to {math.degrees(upper):.2f} deg)"
            )

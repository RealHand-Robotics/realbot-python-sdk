"""Print L30 joint names, native limits, and selected protocol."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print("Protocol:", hand.protocol_version)
    for name, limits in hand.joint_ranges().items():
        print(f"{name}: {limits}")

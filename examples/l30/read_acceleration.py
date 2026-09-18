"""Read V6 acceleration values (not supported by V6.2)."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    if not hand.acceleration.supported:
        print("Acceleration readback is unavailable for", hand.protocol_version)
    else:
        print(hand.acceleration.get())

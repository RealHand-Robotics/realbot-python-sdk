"""Read L30 firmware, mechanical version, serial (V6.2 only), and side."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.protocol_version)
    print(hand.info.get())

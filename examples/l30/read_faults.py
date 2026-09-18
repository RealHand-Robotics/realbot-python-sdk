"""Read 17 L30 native fault/error codes."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.fault.get().error_codes)

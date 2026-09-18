"""Read the 17 native L30 speed values."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.speed.get())

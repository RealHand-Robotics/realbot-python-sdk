"""Read the native 17-element L30 position vector."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.position.get())

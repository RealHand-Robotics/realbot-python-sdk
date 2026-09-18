"""Collect read-only data and print the cached L30 snapshot."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    hand.position.get(); hand.speed.get(); hand.temperature.get()
    print(hand.get_snapshot())

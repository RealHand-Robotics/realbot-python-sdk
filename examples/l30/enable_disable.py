"""Explicitly enable, then disable all L30 joints without commanding a position."""
from common import connected_hand, parser

command = parser(__doc__)
args = command.parse_args()
with connected_hand(args) as hand:
    hand.enable_all()
    print("Joints enabled. Disabling them again now.")
    hand.disable_all()

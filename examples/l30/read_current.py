"""Read 17 motor-current feedback values (6.5 mA units)."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.current.get())

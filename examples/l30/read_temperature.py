"""Read 17 motor temperatures in °C."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print(hand.temperature.get())

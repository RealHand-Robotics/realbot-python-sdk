"""Read the five native 12×6 tactile pressure matrices."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    for finger, matrix in hand.force_sensor.get().items():
        print(f"{finger}: {matrix}")

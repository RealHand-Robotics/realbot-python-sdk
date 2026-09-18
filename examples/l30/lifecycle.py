"""Demonstrate context-managed open/close and is_closed()."""
from common import connected_hand, parser

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    print("Open:", not hand.is_closed())
print("The context manager closed the connection.")

"""Send the L30 emergency-stop command. Use only for an actual emergency."""
from common import connected_hand, parser

command = parser(__doc__)
args = command.parse_args()
with connected_hand(args) as hand:
    hand.emergency_stop()
    print("Emergency-stop command sent.")

"""Set the same native speed on all 17 joints (does not enable motors)."""
from common import connected_hand, parser

command = parser(__doc__)
command.add_argument("--value", type=int, default=75, help="V6 range: 0..150")
args = command.parse_args()
with connected_hand(args) as hand:
    hand.speed.set([args.value] * hand.JOINT_COUNT)
    print("Speed programmed:", args.value)

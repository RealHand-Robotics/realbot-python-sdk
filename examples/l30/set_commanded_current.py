"""Set commanded motor current on all joints; this is not a force command."""
from common import connected_hand, parser

command = parser(__doc__)
command.add_argument("--value", type=int, default=100, help="V6 current units: -2047..2047, 6.5 mA each")
args = command.parse_args()
with connected_hand(args) as hand:
    hand.torque.set_commanded([args.value] * hand.JOINT_COUNT)
    print("Commanded current programmed:", args.value)

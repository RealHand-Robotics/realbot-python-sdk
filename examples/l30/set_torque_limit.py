"""Set the V6 motor-effort limit as a normal 0..100 percentage."""
from common import connected_hand, parser

command = parser(__doc__)
command.add_argument("--percent", type=int, default=50, choices=range(0, 101), metavar="0..100")
args = command.parse_args()
with connected_hand(args) as hand:
    if not hand.torque_limit.supported:
        raise SystemExit(f"Torque-limit commands are unavailable for {hand.protocol_version}")
    hand.torque_limit.set([args.percent * 10] * hand.JOINT_COUNT)
    print(f"Torque limit programmed: {args.percent}%")

"""Set V6 acceleration on all joints (does not enable motors)."""
from common import connected_hand, parser

command = parser(__doc__)
command.add_argument("--value", type=int, default=20, help="V6 range: 0..254; avoid 0 unless firmware behavior is known")
args = command.parse_args()
with connected_hand(args) as hand:
    if not hand.acceleration.supported:
        raise SystemExit(f"Acceleration commands are unavailable for {hand.protocol_version}")
    hand.acceleration.set([args.value] * hand.JOINT_COUNT)
    print("Acceleration programmed:", args.value)

"""Enable and send one explicit native 17-element position vector."""
from common import connected_hand, parser

command = parser(__doc__)
command.add_argument("--positions", nargs=17, type=int, metavar="P", required=True, help="17 native values within joint_metadata.py limits")
args = command.parse_args()
with connected_hand(args) as hand:
    hand.enable_all()
    try:
        hand.position.set(args.positions)
        print("Position command accepted.")
    finally:
        hand.disable_all()

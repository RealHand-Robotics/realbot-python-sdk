"""Stream position and tactile events for five seconds, without enabling motors."""
from time import monotonic

from common import connected_hand, parser
from realhand.hand.l30 import SensorSource

args = parser(__doc__).parse_args()
with connected_hand(args) as hand:
    hand.start_polling({SensorSource.POSITION: 0.1, SensorSource.FORCE_SENSOR: 0.25})
    queue = hand.stream()
    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            print(queue.get(timeout=0.5))
        except Exception:
            pass
    hand.stop_stream()
    hand.stop_polling()

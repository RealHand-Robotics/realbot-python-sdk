"""Shared safe command-line helpers for L30 examples."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from realhand import L30


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--side", choices=("left", "right"), default=os.environ.get("L30_SIDE", "left"))
    result.add_argument("--canfd-id", type=int, default=int(os.environ.get("CANFD_ID", "0")))
    return result


@contextmanager
def connected_hand(args: argparse.Namespace):
    """Open the vendor transport without enabling motors."""
    with L30(side=args.side, canfd_id=args.canfd_id, interface_type="libcanbus") as hand:
        yield hand


"""Privileged installer for the bundled L30 metal-CANFD analyser support."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ASSET_DIRECTORY = "vendor/l30/metal_canfd_analyzer"
LIBRARY_ARCHIVE = "libcanbus(ubuntu22).tar"
UDEV_RULE = "99-canfd.rules"
LIBRARY_DIRECTORY = Path("/usr/local/lib")
UDEV_RULE_DIRECTORY = Path("/etc/udev/rules.d")


def _require_root() -> None:
    if os.name != "posix" or os.geteuid() != 0:
        raise PermissionError("Run this command with sudo: sudo realhand-install-l30-canfd")


def _require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required system command not found: {name}")
    return path


def install(*, library_only: bool = False) -> None:
    """Install the bundled L30 library and, unless skipped, its udev rule."""
    _require_root()
    tar = _require_command("tar")
    ldconfig = _require_command("ldconfig")
    udevadm = None if library_only else _require_command("udevadm")

    asset_directory = Path(__file__).resolve().parent / ASSET_DIRECTORY
    archive = asset_directory / LIBRARY_ARCHIVE
    rule = asset_directory / UDEV_RULE
    if not archive.is_file() or not rule.is_file():
        raise RuntimeError("Bundled L30 CANFD support files are missing from the realhand installation")

    LIBRARY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    subprocess.run([tar, "-xvf", str(archive), "-C", str(LIBRARY_DIRECTORY)], check=True)
    subprocess.run([ldconfig], check=True)

    if library_only:
        print("Installed the L30 libcanbus library. Host udev configuration was skipped.")
        return

    UDEV_RULE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rule, UDEV_RULE_DIRECTORY / UDEV_RULE)
    subprocess.run([udevadm, "control", "--reload-rules"], check=True)
    subprocess.run([udevadm, "trigger"], check=True)

    print("Installed L30 libcanbus and the CANFD udev rule. Unplug and reconnect the analyser.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install bundled L30 metal-CANFD analyser support")
    parser.add_argument(
        "--library-only",
        action="store_true",
        help="install libcanbus but do not install or reload the host-owned udev rule",
    )
    parser.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if os.geteuid() != 0 and not args._elevated:
            command = ["sudo", sys.executable, str(Path(__file__).resolve()), "--_elevated"]
            if args.library_only:
                command.append("--library-only")
            return subprocess.run(command, check=False).returncode
        install(library_only=args.library_only)
    except (FileNotFoundError, PermissionError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

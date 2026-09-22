#!/usr/bin/env bash
# Install bundled L30 metal-CANFD analyser support from a cloned SDK checkout.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${REPOSITORY_ROOT}/src/realhand/l30_canfd_install.py" "$@"

# L30 metal CANFD analyser files

This directory contains the vendor files needed by the blue/black metal L30
CANFD analyser on 64-bit Ubuntu 22.04 or 24.04:

- `libcanbus(ubuntu22).tar` — vendor CANFD transport library for Ubuntu 22.04
  and 24.04
- `99-canfd.rules` — udev permissions rule for the analyser

The top-level SDK README contains the supported installation commands. These
files are intentionally not used for the transparent USB CANFD adaptor, which
enumerates as SocketCAN instead.

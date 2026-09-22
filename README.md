# Realhand Arm & Hand Python SDK

Pure Python SDK with GUI for Realhand dexterous hands and robotic arms.

> **Note:** This project is under active development. APIs may change between versions.

## 🚀 Get Started
This Python SDK supports RealHand dexterous hands O6, L6, L20, L20 Lite, L25, and L30, plus the A7, A7 Lite, and P7 robotic arms.

We provide detailed tutorial.
🧪 [Quick start tutorial](https://realhand-robotics.github.io/realbot-python-sdk-document/)

## 📦 Installation

```bash
# pip
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git

# uv
uv add "realhand @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"

```

### Optional desktop GUI

Install the `gui` extra to use the hand-control GUI, including L30
autodetection:

```bash
# pip
python3 -m pip install "realhand[gui] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
realhand-control-gui

# uv
uv add "realhand[gui] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
uv run realhand-control-gui
```

### L30 metal CANFD analyser

The blue/black metal CANFD analyser uses the vendor `libcanbus` transport, not
SocketCAN. The bundled 64-bit Ubuntu 22.04 library archive is also the
supported archive for Ubuntu 24.04. It and the required udev rule are in
`src/realhand/vendor/l30/metal_canfd_analyzer/`. After either a pip or uv
installation, install them with:

```bash
# After pip or uv installation
realhand-install-l30-canfd

# From a cloned SDK checkout (no Python-package install required)
./scripts/install_l30_canfd.sh
```

The command prompts for administrator permission, extracts the vendor library
to `/usr/local/lib`, runs `ldconfig`, installs and reloads the udev rule, and
then asks you to reconnect the CANFD analyser. This adapter does **not** create
`can0`, so do not configure it with `ip link`; use the L30 default
`interface_type="libcanbus"` and `canfd_id=0` instead.

For a Docker container, install only the library because the udev daemon and
USB permissions belong to the host:

```bash
realhand-install-l30-canfd --library-only
```

The bundled library is for 64-bit Ubuntu 22.04 and 24.04. The matching files
for other platforms remain available in the vendor L30 SDK.

### A7 and A7 Lite users

A7 and A7 Lite require Pinocchio for kinematics. Install the `kinetix` extra:

```bash
# pip
pip install "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"

# uv
uv add "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
```

P7 uses the built-in RBot TCP controller interface and does not require the `kinetix` extra.

### P7 desktop GUI

The optional native PyQt5 GUI connects directly to the RBot controller; it does
not require ROS 2 or a container. It starts disconnected and does not enable or
move the arm until you connect and send a command.

```bash
python3 -m pip install "realhand[gui] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
realhand-p7-gui
```

## L30 quick example

The L30 uses its native 17-element integer motor-position vector.

```python
from realhand import L30

with L30(side="right", canfd_id=0) as hand:
    # Confirm the hand's workspace is clear before enabling motion.
    hand.enable_all()
    hand.position.set([0] * 17)  # native L30 home position

    print(hand.position.get())
    print(hand.info.get().serial_number)
    print(hand.fault.get().error_codes)
    hand.disable_all()
```

For safety, connecting an L30 does not enable its joints. `position.set()` is
blocked until `enable_all()` succeeds. Use only values in the connected hand's
native per-joint ranges.

The L30 also provides cached snapshots and a polling-backed event stream:

```python
from realhand.hand.l30 import SensorSource

hand.start_polling({
    SensorSource.POSITION: 1 / 30,
    SensorSource.TEMPERATURE: 1.0,
    SensorSource.FORCE_SENSOR: 1 / 15,
})
snapshot = hand.get_snapshot()

for event in hand.stream():
    print(event)
```

Standalone examples for every public L30 operation are in `examples/l30/`.
Examples that change configuration, enable joints, move joints, or send an
emergency stop execute directly.

## P7 quick example

```python
from realhand import P7

arm = P7(
    side="left",
    interface_name="192.168.10.21",
    interface_type="rbot",
    world_frame="urdf",
)

arm.enable()
arm.move_j([-0.2, 0.1, 0.2, -0.2, 0.0, 0.0, 0.0], blocking=False)
arm.emergency_stop()
arm.resume_from_emergency_stop()
arm.enable()
arm.home(blocking=False)
```

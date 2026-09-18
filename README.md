# Realhand Arm & Hand Python SDK

Pure Python SDK with GUI for Realhand dexterous hands and robotic arms.

> **Note:** This project is under active development. APIs may change between versions.

## 🚀 Get Started
This Python SDK supports RealHand dexterous hands O6, L6, L20, L20 Lite, L25, and L30, plus the A7, A7 Lite, and P7 robotic arms.

### L30 quick example (metal CANFD analyser)

The L30 uses its native 17-element motor-position vector. Install the
matching vendor `libcanbus` library under `/usr/local/lib` before use. The L30
API is direct Python and does not require ROS 2.

#### Install libcanbus (Ubuntu 22.04, metal USB CANFD analyser)

The blue/black metal CANFD analyser uses the vendor `libcanbus` transport, not
SocketCAN. From the directory that contains the vendor-supplied
`libcanbus(ubuntu22).tar` and `99-canfd.rules` files, run:

```bash
sudo tar -xvf "libcanbus(ubuntu22).tar" -C /usr/local/lib/
sudo ldconfig

sudo install -m 644 99-canfd.rules /etc/udev/rules.d/99-canfd.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the CANFD analyser after installing the udev rule. This
adapter does **not** create `can0`, so do not configure it with `ip link`; use
the L30 default `interface_type="libcanbus"` and `canfd_id=0` instead.

```python
from realhand import L30

with L30(side="right", canfd_id=0) as hand:
    # Confirm the hand's workspace is clear before enabling motion.
    hand.enable_all()
    hand.speed.set([50] * 17)

    # Send a position only after confirming the 17 values are safe for the
    # connected hand and its surroundings.
    # hand.position.set([...])

    print(hand.position.get())  # read-only
    print(hand.info.get().serial_number)
    print(hand.fault.get().error_codes)
    hand.disable_all()
```

For safety, connecting an L30 does not enable its joints. `position.set()` is
blocked until `enable_all()` succeeds, and it rejects any value outside the
connected hand's native per-joint range.

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

Standalone examples for every public L30 operation are in
[`examples/l30`](examples/l30/README.md). Examples that change configuration,
enable joints, move joints, or send an emergency stop execute directly.

We provide detailed tutorial.
🧪 [Quick start tutorial](https://realhand-robotics.github.io/realbot-python-sdk-document/)

## 📦 Installation

```bash
# pip
pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git

# uv
uv add "realhand @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"

```


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
pip install -e ".[gui]"
realhand-p7-gui
```

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

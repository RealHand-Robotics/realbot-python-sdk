# Realhand Arm & Hand Python SDK

Pure Python SDK with GUI for Realhand dexterous hands and robotic arms.

> **Note:** This project is under active development. APIs may change between versions.

## 🚀 Get Started
This Python SDK supports RealHand dexterous hands O6, L6, L20, L20 Lite, and L25, plus the A7 Lite and P7 robotic arms.

We provide detailed tutorial.
🧪 [Quick start tutorial](https://realhand-robotics.github.io/realbot-python-sdk-document/)

## 📦 Installation

```bash
# pip
pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk-test.git

# uv
uv add "realhand @ git+https://github.com/RealHand-Robotics/realbot-python-sdk-test.git"

```


### A7 Lite users

A7 Lite requires Pinocchio for kinematics. Install the `kinetix` extra:

```bash
# pip
pip install "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk-test.git"

# uv
uv add "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk-test.git"
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

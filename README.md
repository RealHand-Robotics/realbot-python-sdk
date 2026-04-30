# Realhand Arm & Hand Python SDK

Pure Python SDK with GUI for [Realhand](https://github.com/RealHand-Robotics/Realhand-python-sdk) dexterous hands and robotic arms.

[![PyPI](https://img.shields.io/pypi/v/realhand)](https://pypi.org/project/realhand/)
[![GitHub](https://img.shields.io/badge/github-RealHand--Robotics%2Frealbot--python--sdk-blue)](https://github.com/RealHand-Robotics/realbot-python-sdk)

> **Note:** This project is under active development. APIs may change between versions.

## 🚀 Get Started
This python sdk supports Realhand dexterous hand O6, L6, L20 and 7 axises arm A7, A7 Lite.
We provide detailed tutorial
🧪 [Quick start tutorial](https://realhand-robotics.github.io/realbot-python-sdk-document/)

## 📦 Installation

```bash
# pip
pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git

# uv
uv add "realhand @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"

```


### Arm users

Arms (A7 / A7 Lite) require Pinocchio for kinematics. Install the `kinetix` extra:

```bash
# pip
pip install "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"

# uv
uv add "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
```


> **Windows users:** Pinocchio does not support pip on Windows. Use `conda install pinocchio -c conda-forge` instead.


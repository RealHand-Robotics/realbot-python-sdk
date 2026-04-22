# Realhand Arm & Hand Python SDK

Pure Python SDK for [Realhand](https://realhand.cn) dexterous hands and robotic arms.

[![PyPI](https://img.shields.io/pypi/v/realhand)](https://pypi.org/project/realhand/)
[![GitHub](https://img.shields.io/badge/github-RealHand--Robotics%2Frealbot--python--sdk-blue)](https://github.com/RealHand-Robotics/realbot-python-sdk)

> **Note:** This project is under active development. APIs may change between versions.

[中文文档](README_zh.md)

## Installation

```bash
# pip
pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git
```


### Arm users

Arms (A7 / A7 Lite) require Pinocchio for kinematics. Install the `kinetix` extra:

```bash
# pip
pip install "realhand[kinetix] @ git+https://github.com/RealHand-Robotics/realbot-python-sdk.git"
```


> **Windows users:** Pinocchio does not support pip on Windows. Use `conda install pinocchio -c conda-forge` instead.

# CS 528 Final Project: DIY Controller

This project adapts raw sensor data into gamepad input. As a proof of concept, it uses three IMU sensors to play Mario Kart 8.

## Getting Started

### Installation
Install dependencies with your preferred Python package manager:
- **uv (Recommended)**: `uv sync`
- **pip**: `pip install .`

*(Note: Xbox gamepad emulation requires the `vgamepad` package and the kernel-level ViGEmBus driver installed).*

### Running the Program
Start the gesture recognition script from within the active ESP-IDF virtual environment:
- **uv**: `uv run gesture_recognition.py`
- **Python**: `python gesture_recognition.py`

## Contributors:
- [Aryan](https://github.com/aryand15)
- [Oleg](https://github.com/olegpolin)
- [Cameron](https://github.com/proulxdev)

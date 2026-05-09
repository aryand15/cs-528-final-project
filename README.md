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

### 3D Printing the Enclosure
`ESP32S3_IMU_Enclosure_v3.step` contains the body and lid for the ESP32 S3 and IMU. We printed it with generic PLA with a standard 0.4mm nozzle and 0.2mm layer thickness. The lid should be printed with its outside/top surface on the bed to help support the edge's overhang. The IMU slots into the pegs for consistent orientation. The ESP32 and IMU can be hot glued or taped in place. Here is the [OnShape link](https://cad.onshape.com/documents/f8cecf56b9558fafa9d3d539/w/0b9b4e573ed9d98d951dcd60/e/f5b858c83c0ce33ba65c5a7a?renderMode=0&uiState=69ffabcd2c0495f1a9bc1804) if you would like to modify it.

## Contributors:
- [Aryan](https://github.com/aryand15)
- [Oleg](https://github.com/olegpolin)
- [Cameron](https://github.com/proulxdev)

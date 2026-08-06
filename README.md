# SpatialSense

SpatialSense is a real-time spatial perception prototype for Apple Silicon that combines:

- **YOLO26 (MLX)** for object detection
- **Depth Anything (CoreML)** for monocular depth estimation
- **Binaural audio cues** for obstacle awareness on headphones

It renders a live camera feed with detections, depth visualization, a 3×3 proximity grid, and optional spatial audio output.

## Features

- Real-time object detection with **YOLO26n** (fast) or **YOLO26s** (higher accuracy)
- Depth estimation with **Depth Anything V2/V3** CoreML models on Apple Neural Engine
- 3×3 scene proximity grid (left/center/right × top/mid/low)
- Binaural warning cues mapped by object position and perceived proximity
- Side-by-side visual output: RGB detections + depth colormap

## Requirements

- macOS on Apple Silicon (M-series)
- Python 3.10+
- Webcam or Continuity Camera
- Headphones (recommended for spatial audio mode)

## Installation

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download model assets:

```bash
python setup_models.py
```

This creates a local `models/` directory with all required YOLO and depth model files.

## Quick Start

Run with defaults:

```bash
python main.py
```

Then press **`q`** to quit.

## Model Options

### YOLO model

- `-n` → YOLO26n (default, fastest)
- `-s` → YOLO26s (higher accuracy, slower)

### Depth model

Use `--depth` with one of:

- `v2s` → Depth-Anything-V2-Small (default)
- `v2b` → Depth-Anything-V2-Base
- `v3s` → Depth-Anything-V3-Small
- `v3b` → Depth-Anything-V3-Base

## Usage Examples

```bash
python main.py                     # YOLO26n + DA-V2-Small (defaults)
python main.py -s                  # YOLO26s
python main.py --depth v2b         # DA-V2-Base
python main.py -s --depth v3b      # YOLO26s + DA-V3-Base
python main.py --no-audio          # disable binaural output
python main.py --camera 0          # use MacBook webcam
```

## CLI Arguments

- `-n` / `-s`: choose YOLO model variant
- `--depth {v2s,v2b,v3s,v3b}`: choose depth model
- `--no-audio`: disable audio engine
- `--camera N`: select camera index (default `1`)

## Troubleshooting

- **Camera does not open**: try another camera index (`--camera 0`, `--camera 1`, etc.).
- **Low FPS**: use `-n` and `--depth v2s` for best speed.
- **No audio output**: check output device/headphones and ensure `--no-audio` is not set.
- **Model load errors**: rerun `python setup_models.py` and verify `models/` exists.

## License

This project is licensed under the terms in [LICENSE](LICENSE).

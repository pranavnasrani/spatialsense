"""
SpatialSense — Real-time object detection + depth estimation on Apple Silicon.

Usage:
    python main.py                     # YOLO26n + DA-V2-Small (defaults)
    python main.py -s                  # YOLO26s (higher accuracy)
    python main.py --depth v2b         # DA-V2-Base (better depth)
    python main.py -s --depth v3b      # YOLO26s + DA-V3-Base (best quality)

YOLO flags:
    -n    YOLO26n — 2.7M params, ~8ms   (default)
    -s    YOLO26s — 9.5M params, ~15ms  (+40% mAP)

Depth flags (--depth):
    v2s   Depth-Anything-V2-Small  — 24.8M, ~20ms   (default, Apple official)
    v2b   Depth-Anything-V2-Base   — 97.5M, ~60ms   (sharper edges)
    v3s   Depth-Anything-V3-Small  — 80M,   ~25ms   (latest gen)
    v3b   Depth-Anything-V3-Base   — 120M,  ~60ms   (latest gen, best quality)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import time
import cv2
import numpy as np
from PIL import Image
import coremltools as ct
from yolo26mlx import YOLO
from audio_engine import AudioEngine
from audio_types import AudioFrame, DetectedObject

# ── CLI Argument Parsing ──────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="SpatialSense: real-time detection + depth on Apple Silicon",
)

yolo_group = parser.add_mutually_exclusive_group()
yolo_group.add_argument(
    "-n", action="store_const", const="n", dest="yolo", default="n",
    help="Use YOLO26n (nano, default — fast)",
)
yolo_group.add_argument(
    "-s", action="store_const", const="s", dest="yolo",
    help="Use YOLO26s (small — higher accuracy)",
)

parser.add_argument(
    "--depth", choices=["v2s", "v2b", "v3s", "v3b"], default="v2s",
    help="Depth model: v2s (default), v2b, v3s, v3b",
)

parser.add_argument(
    "--no-audio", action="store_true",
    help="Disable binaural audio output (useful when debugging visuals)",
)

parser.add_argument(
    "--camera", type=int, default=1, metavar="N",
    help="Camera index: 0 = MacBook webcam, 1 = iPhone Continuity Camera (default: 1)",
)

args = parser.parse_args()

# ── Model Configuration ───────────────────────────────────────────────────────

YOLO_CONFIG = {
    "n": {"path": "models/yolo26n.npz", "name": "YOLO26n"},
    "s": {"path": "models/yolo26s.npz", "name": "YOLO26s"},
}

# Depth estimation is run every DEPTH_FRAME_INTERVAL frames.
# Kept uniform across all models for fair comparison.
DEPTH_FRAME_INTERVAL = 2

DEPTH_CONFIG = {
    "v2s": {
        "path": "models/DepthAnythingV2SmallF16.mlpackage",
        "name": "DA-V2-Small",
        "input_size": (518, 392),   # (width, height) for cv2.resize
        "invert": False,            # V2: higher value = closer
    },
    "v2b": {
        "path": "models/DepthAnythingV2BaseF16.mlpackage",
        "name": "DA-V2-Base",
        "input_size": (518, 518),
        "invert": False,
    },
    "v3s": {
        "path": "models/DepthAnythingV3_small_504.mlpackage",
        "name": "DA-V3-Small",
        "input_size": (504, 504),
        "invert": True,             # V3: lower value = closer (inverted)
    },
    "v3b": {
        "path": "models/DepthAnythingV3_base_504.mlpackage",
        "name": "DA-V3-Base",
        "input_size": (504, 504),
        "invert": True,
    },
}

yolo_cfg = YOLO_CONFIG[args.yolo]
depth_cfg = DEPTH_CONFIG[args.depth]

# Minimum YOLO confidence threshold for displaying a detection
CONF_THRESHOLD = 0.4

# ── Model Loading ──────────────────────────────────────────────────────────────

print(f"Loading {yolo_cfg['name']} MLX model...")
yolo_model = YOLO(yolo_cfg["path"])

print(f"Loading {depth_cfg['name']} CoreML model (ANE)...")
depth_model = ct.models.MLModel(depth_cfg["path"])

# Retrieve the CoreML model's expected input name so we can call predict() correctly
depth_input_name = depth_model.get_spec().description.input[0].name
print(f"  → depth model input key: '{depth_input_name}'")
print(f"  → depth input size: {depth_cfg['input_size'][0]}×{depth_cfg['input_size'][1]}")
print(f"  → depth frame interval: every {DEPTH_FRAME_INTERVAL} frames")
print(f"  → depth invert: {depth_cfg['invert']}")
print(f"Models loaded. Starting capture...")

# ── Audio Engine Initialization ───────────────────────────────────────────────

audio_engine: AudioEngine | None = None
if not args.no_audio:
    audio_engine = AudioEngine()
    audio_engine.start()

WINDOW_TITLE = f"SpatialSense | {yolo_cfg['name']} (MLX) + {depth_cfg['name']} (ANE)"

# ── Video Capture Setup ────────────────────────────────────────────────────────

cap = cv2.VideoCapture(args.camera)
frame_count = 0
depth_colormap = None
depth_map = None

# ── Performance Metrics ───────────────────────────────────────────────────────

# Exponential moving average (EMA) smoothing factor — avoids jittery numbers
EMA_ALPHA = 0.1
yolo_ms_ema = 0.0
depth_ms_ema = 0.0
fps_ema = 0.0
last_depth_ms = 0.0       # Raw ms for the most recent depth inference

try:
    while cap.isOpened():
        frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape

        # A. YOLO Detection (every frame) — MLX / Apple Metal GPU
        t0 = time.perf_counter()
        yolo_results = yolo_model.predict(frame, conf=CONF_THRESHOLD)
        yolo_ms = (time.perf_counter() - t0) * 1000
        yolo_ms_ema = EMA_ALPHA * yolo_ms + (1 - EMA_ALPHA) * yolo_ms_ema

        # B. Depth Estimation (every DEPTH_FRAME_INTERVAL frames) — CoreML / ANE
        if depth_map is None or frame_count % DEPTH_FRAME_INTERVAL == 0:
            # Resize to the exact fixed input size the CoreML model expects
            small_bgr = cv2.resize(frame, depth_cfg["input_size"])
            pil_img = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))

            # Run on Neural Engine via CoreML
            t0 = time.perf_counter()
            coreml_out = depth_model.predict({depth_input_name: pil_img})
            depth_ms = (time.perf_counter() - t0) * 1000
            last_depth_ms = depth_ms
            depth_ms_ema = EMA_ALPHA * depth_ms + (1 - EMA_ALPHA) * depth_ms_ema

            # CoreML returns a dict; the depth output may be:
            #   - a PIL Image (mode 'F' = float32, or 'L' = uint8)
            #   - a numpy array (depending on coremltools version / output spec)
            raw_depth = next(iter(coreml_out.values()))

            if isinstance(raw_depth, Image.Image):
                # PIL Image → convert to float32 numpy array
                raw_depth = np.array(raw_depth, dtype=np.float32)
            else:
                # Already a numpy array — ensure float32 and squeeze to 2-D
                raw_depth = np.array(raw_depth, dtype=np.float32).squeeze()

            # V3 models output "distance-like" depth (closer = smaller value).
            # Invert so that closer = higher value, matching V2 convention and
            # producing warm colours for nearby objects in INFERNO colormap.
            if depth_cfg["invert"]:
                raw_depth = raw_depth.max() - raw_depth

            depth_map = cv2.resize(raw_depth, (w, h))

            # Render visual colormap
            depth_visual = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

        # C. Compute 3×3 Spatial Grid (Left, Center, Right × Top, Middle, Bottom)
        # Useful for low-latency binaural audio spatialization / HRTF
        grid_matrix = np.zeros((3, 3), dtype=np.float32)
        if depth_map is not None:
            cell_h = h // 3
            cell_w = w // 3

            for row in range(3):
                for col in range(3):
                    y_start, y_end = row * cell_h, (row + 1) * cell_h
                    x_start, x_end = col * cell_w, (col + 1) * cell_w

                    cell_crop = depth_map[y_start:y_end, x_start:x_end]
                    if cell_crop.size > 0:
                        # Use 90th percentile (closest objects after normalization/inversion)
                        # or median to represent sector distance
                        grid_matrix[row, col] = float(np.percentile(cell_crop, 85))

        # D. Draw 3×3 Grid HUD & Overlays on Camera Feed
        if depth_map is not None:
            cell_h = h // 3
            cell_w = w // 3

            # Draw 3x3 grid lines
            for i in range(1, 3):
                cv2.line(frame, (i * cell_w, 0), (i * cell_w, h), (100, 100, 100), 1)
                cv2.line(frame, (0, i * cell_h), (w, i * cell_h), (100, 100, 100), 1)

            # Labels for sectors
            col_names = ["Left", "Center", "Right"]
            row_names = ["Top", "Mid", "Low"]

            # Overlay sector depth metrics
            max_depth_val = np.max(grid_matrix) if np.max(grid_matrix) > 0 else 1.0
            for row in range(3):
                for col in range(3):
                    val = grid_matrix[row, col]
                    norm_val = val / max_depth_val  # 0.0 (far) to 1.0 (close)

                    # Color coding: Green (safe/far) -> Yellow -> Red (close/hazard)
                    color = (
                        int(0),
                        int(255 * (1.0 - norm_val)),
                        int(255 * norm_val),
                    )

                    cx = col * cell_w + 10
                    cy = row * cell_h + 30
                    text = f"{row_names[row]}-{col_names[col]}: {val:.0f}"

                    cv2.putText(
                        frame,
                        text,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # E. Extract Spatial Coordinates & Overlay Depth for YOLO (if present)
        detected_objects: list[DetectedObject] = []

        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy        # (N, 4)
            confs = boxes.conf       # (N,)
            classes = boxes.cls      # (N,)

            for i in range(len(boxes)):
                conf = float(confs[i])
                if conf <= CONF_THRESHOLD or depth_map is None:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                cls_id = int(classes[i])
                label = yolo_model.names[cls_id]

                crop = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                median_depth = np.median(crop) if crop.size > 0 else 0.0

                # Collect spatial metadata for audio engine
                cx_norm   = ((x1 + x2) / 2.0) / w
                cy_norm   = ((y1 + y2) / 2.0) / h
                area_frac = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
                detected_objects.append(DetectedObject(
                    label=label,
                    cx=cx_norm,
                    cy=cy_norm,
                    depth=median_depth,
                    bbox_area_frac=area_frac,
                ))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} | Depth: {median_depth:.1f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # F. Push spatial data to binaural audio engine
        if audio_engine is not None:
            audio_engine.push(AudioFrame(
                objects=detected_objects,
                grid_matrix=grid_matrix,
                frame_w=w,
                frame_h=h,
            ))

        # D. Compute frame-level FPS
        frame_ms = (time.perf_counter() - frame_start) * 1000
        fps = 1000.0 / max(frame_ms, 0.1)
        fps_ema = EMA_ALPHA * fps + (1 - EMA_ALPHA) * fps_ema if fps_ema > 0 else fps

        # E. Draw performance metrics overlay (top-left of detection view)
        metrics = [
            f"FPS: {fps_ema:.1f}",
            f"YOLO: {yolo_ms_ema:.1f}ms ({yolo_cfg['name']})",
            f"Depth: {depth_ms_ema:.1f}ms ({depth_cfg['name']})",
            f"Frame: {frame_ms:.1f}ms",
        ]
        y_offset = 20
        for line in metrics:
            # Black background for readability
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (8, y_offset - th - 4), (14 + tw, y_offset + 4), (0, 0, 0), -1)
            cv2.putText(
                frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )
            y_offset += 22

        # F. Display Dual Stream Output
        if depth_colormap is not None:
            combined = np.hstack((frame, depth_colormap))
            cv2.imshow(WINDOW_TITLE, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always release the camera and close windows, even on exceptions or Ctrl-C
    if audio_engine is not None:
        audio_engine.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Capture released.")

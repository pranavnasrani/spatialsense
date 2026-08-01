import warnings
warnings.filterwarnings("ignore")  # Suppress SSL/deprecations

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ── Constants ──────────────────────────────────────────────────────────────────
# Depth-Anything-V2-Small native resolution is 518×518.
# Using the native 518×518 resolution for maximum depth accuracy.
DEPTH_INPUT_SIZE = (518, 518)

# Depth estimation is run every N frames to maintain high FPS.
# The last computed depth map is reused on skipped frames.
DEPTH_FRAME_INTERVAL = 2

# Minimum YOLO confidence threshold for displaying a detection
CONF_THRESHOLD = 0.4

# ── Device Setup ───────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running pipeline on device: {device}")

# ── Model Loading ──────────────────────────────────────────────────────────────
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

model_id = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"Loading depth model ({model_id})...")
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
depth_model.eval()
print("Models loaded. Starting capture...")

# ── Video Capture Setup ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_count = 0
depth_colormap = None
depth_map = None

try:
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w, _ = frame.shape

            # A. YOLO Detection (every frame)
            yolo_results = yolo_model.predict(source=frame, device=device, verbose=False)[0]

            # B. Depth Estimation (every DEPTH_FRAME_INTERVAL frames)
            # depth_map is None on the first frame, so inference always runs initially;
            # thereafter it runs every Nth frame and the last map is reused in between.
            if depth_map is None or frame_count % DEPTH_FRAME_INTERVAL == 0:
                small_frame = cv2.resize(frame, DEPTH_INPUT_SIZE)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                inputs = processor(images=rgb_small, return_tensors="pt").to(device)
                outputs = depth_model(**inputs)

                # Interpolate depth back to native frame resolution
                predicted_depth = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_map = predicted_depth.cpu().numpy()

                # Render visual colormap
                depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

            # C. Extract Spatial Coordinates & Overlay Depth
            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD and depth_map is not None:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]

                    # Crop depth array safely to box bounds
                    crop = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    # Depth is relative (unitless model output), not metric distance
                    median_depth = np.median(crop) if crop.size > 0 else 0.0

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} | Rel. Depth: {median_depth:.1f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # D. Display Dual Stream Output
            if depth_colormap is not None:
                combined = np.hstack((frame, depth_colormap))
                cv2.imshow("SpatialSense High-FPS Feed", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Always release the camera and close windows, even on exceptions or Ctrl-C
    cap.release()
    cv2.destroyAllWindows()
    print("Capture released.")
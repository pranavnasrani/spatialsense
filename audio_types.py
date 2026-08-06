"""
audio_types.py — Shared dataclasses for the SpatialSense audio pipeline.

The vision thread populates these and pushes them into the AudioEngine queue.
No ML/audio dependencies here — keeps the import graph clean.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectedObject:
    """One YOLO detection with spatial metadata derived from the depth map."""
    label: str
    cx: float             # Normalised bbox centre X  ∈ [0, 1]  (0=left, 1=right)
    cy: float             # Normalised bbox centre Y  ∈ [0, 1]  (0=top,  1=bottom)
    depth: float          # Median depth value at bbox (model-relative units)
    bbox_area_frac: float # bbox_area / frame_area     ∈ [0, 1]


@dataclass
class AudioFrame:
    """Snapshot of one video frame's spatial data, consumed by AudioEngine."""
    objects: list         # list[DetectedObject] — all detections this frame
    grid_matrix: np.ndarray  # shape (3, 3) — proximity values per sector
    frame_w: int          # Frame width  in pixels (for reference)
    frame_h: int          # Frame height in pixels (for reference)

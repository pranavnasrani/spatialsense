"""
setup_models.py — One-time model download for SpatialSense.

Run once before starting the app:
    python setup_models.py

Downloads:
  YOLO (MLX — Apple GPU via Metal):
    1. yolo26n.npz   — YOLO26n weights (webAI-Official/yolo26n-mlx)
    2. yolo26s.npz   — YOLO26s weights (webAI-Official/yolo26s-mlx)

  Depth-Anything (CoreML — Apple Neural Engine):
    3. DepthAnythingV2SmallF16.mlpackage  — DA-V2-Small (apple/coreml-depth-anything-v2-small)
    4. DepthAnythingV2BaseF16.mlpackage   — DA-V2-Base  (mrgnw/depth-anything-v2-coreml)
    5. DepthAnythingV3_small_504.mlpackage — DA-V3-Small (mlboydaisuke/Depth-Anything-3-Small-CoreML)
    6. DepthAnythingV3_base_504.mlpackage  — DA-V3-Base  (mlboydaisuke/Depth-Anything-3-Base-CoreML)
"""

import os
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YOLO models (MLX .npz weights)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOLO_MODELS = [
    ("yolo26n.npz", "webAI-Official/yolo26n-mlx", "yolo26n.npz"),
    ("yolo26s.npz", "webAI-Official/yolo26s-mlx", "yolo26s.npz"),
]

for local_name, repo_id, filename in YOLO_MODELS:
    dest = MODELS_DIR / local_name
    if dest.exists():
        print(f"[✓] {local_name} already present")
    else:
        print(f"[↓] Downloading {local_name} ({repo_id})...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODELS_DIR))
        print(f"[✓] {local_name} saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Depth-Anything V2 models (CoreML .mlpackage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. DA-V2-Small — Apple official (snapshot_download for .mlpackage directory)
DA_V2S_DEST = MODELS_DIR / "DepthAnythingV2SmallF16.mlpackage"
if DA_V2S_DEST.exists():
    print("[✓] DA-V2-Small already present")
else:
    print("[↓] Downloading DA-V2-Small (apple/coreml-depth-anything-v2-small)...")
    snapshot_download(
        repo_id="apple/coreml-depth-anything-v2-small",
        allow_patterns=["DepthAnythingV2SmallF16.mlpackage/**"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print("[✓] DA-V2-Small saved")

# 2. DA-V2-Base — mrgnw community (tar.gz, needs extraction)
DA_V2B_DEST = MODELS_DIR / "DepthAnythingV2BaseF16.mlpackage"
if DA_V2B_DEST.exists():
    print("[✓] DA-V2-Base already present")
else:
    print("[↓] Downloading DA-V2-Base (mrgnw/depth-anything-v2-coreml)...")
    tar_path = hf_hub_download(
        repo_id="mrgnw/depth-anything-v2-coreml",
        filename="DepthAnythingV2BaseF16.mlpackage.tar.gz",
        local_dir=str(MODELS_DIR),
    )
    print("    Extracting tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(MODELS_DIR))
    # Clean up the tar.gz after extraction
    os.remove(tar_path)
    print("[✓] DA-V2-Base saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Depth-Anything V3 models (CoreML .mlpackage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3. DA-V3-Small
DA_V3S_DEST = MODELS_DIR / "DepthAnythingV3_small_504.mlpackage"
if DA_V3S_DEST.exists():
    print("[✓] DA-V3-Small already present")
else:
    print("[↓] Downloading DA-V3-Small (mlboydaisuke/Depth-Anything-3-Small-CoreML)...")
    snapshot_download(
        repo_id="mlboydaisuke/Depth-Anything-3-Small-CoreML",
        allow_patterns=["DepthAnythingV3_small_504.mlpackage/**"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print("[✓] DA-V3-Small saved")

# 4. DA-V3-Base
DA_V3B_DEST = MODELS_DIR / "DepthAnythingV3_base_504.mlpackage"
if DA_V3B_DEST.exists():
    print("[✓] DA-V3-Base already present")
else:
    print("[↓] Downloading DA-V3-Base (mlboydaisuke/Depth-Anything-3-Base-CoreML)...")
    snapshot_download(
        repo_id="mlboydaisuke/Depth-Anything-3-Base-CoreML",
        allow_patterns=["DepthAnythingV3_base_504.mlpackage/**"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print("[✓] DA-V3-Base saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[✓] All models ready. Usage:")
print("    python main.py -n              # YOLO26n (default) + DA-V2-Small (default)")
print("    python main.py -s              # YOLO26s")
print("    python main.py --depth v2b     # DA-V2-Base")
print("    python main.py -s --depth v3b  # YOLO26s + DA-V3-Base")

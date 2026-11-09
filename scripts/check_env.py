"""Check that the local Mac environment can run the E0/E1 prototype steps.

Run from the repository root:  conda activate affordance3d && python scripts/check_env.py

Checks: package versions; DINOv2 ViT-S/14 patch features on MPS match CPU;
Open3D point-cloud operations; SceneFun3D toolkit imports. Exits 1 on failure.
"""

import importlib.metadata as md
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_DIR = REPO_ROOT / "third_party" / "scenefun3d"

DINO_ID = "facebook/dinov2-small"  # ViT-S/14, Apache-2.0
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
INPUT_SIDE = 448  # plan's starting resolution -> 32x32 = 1024 patch tokens
EXPECTED_TOKENS = 1 + (INPUT_SIDE // 14) ** 2  # CLS + patches
EXPECTED_DIM = 384
MIN_TOKEN_COSINE = 0.999  # MPS vs CPU agreement per token
TIMED_RUNS = 5

PACKAGES = ["torch", "torchvision", "transformers", "numpy", "open3d", "opencv-python",
            "trimesh", "viser", "pyviz3d", "moviepy", "plotly"]


def report_versions() -> None:
    print(f"python {sys.version.split()[0]}")
    for name in PACKAGES:
        print(f"  {name:<14} {md.version(name)}")


def dino_forward(model: torch.nn.Module, pixels: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    """Return patch tokens on CPU and mean warm per-frame latency in seconds."""
    model = model.to(device)
    x = pixels.to(device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    with torch.inference_mode():
        model(pixel_values=x)  # warm-up (kernel compilation on MPS)
        sync()
        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            out = model(pixel_values=x).last_hidden_state
        sync()
    return out.float().cpu(), (time.perf_counter() - start) / TIMED_RUNS


def check_dinov2() -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).eval()
    pixels = torch.randn(1, 3, INPUT_SIDE, INPUT_SIDE, generator=torch.Generator().manual_seed(0))

    cpu_tokens, cpu_s = dino_forward(model, pixels, "cpu")
    assert cpu_tokens.shape == (1, EXPECTED_TOKENS, EXPECTED_DIM), f"unexpected shape {cpu_tokens.shape}"
    print(f"  DINOv2 tokens {tuple(cpu_tokens.shape)}  cpu {cpu_s * 1000:.0f} ms/frame")

    if not torch.backends.mps.is_available():
        print("  MPS unavailable -> E1 feature extraction will run on CPU")
        return
    mps_tokens, mps_s = dino_forward(model, pixels, "mps")
    cosine = torch.nn.functional.cosine_similarity(cpu_tokens, mps_tokens, dim=-1).min().item()
    print(f"  mps {mps_s * 1000:.0f} ms/frame  min token cosine vs cpu {cosine:.6f}")
    assert cosine >= MIN_TOKEN_COSINE, "MPS features disagree with CPU; use CPU for E1"


def check_open3d() -> None:
    import open3d as o3d

    points = np.random.default_rng(0).uniform(-1.0, 1.0, size=(50_000, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = cloud.voxel_down_sample(voxel_size=0.05)
    down.estimate_normals()
    assert down.has_normals() and 0 < len(down.points) < len(points)
    print(f"  open3d voxel downsample {len(points)} -> {len(down.points)} points, normals ok")


def check_scenefun3d_toolkit() -> None:
    if not TOOLKIT_DIR.is_dir():
        raise FileNotFoundError(f"SceneFun3D toolkit not found at {TOOLKIT_DIR}")
    sys.path.insert(0, str(TOOLKIT_DIR))
    import data_downloader.download_utils.download_data  # noqa: F401  (moviepy)
    import utils.viz  # noqa: F401  (pyviz3d)
    from utils.data_parser import DataParser  # noqa: F401
    from utils.fusion_util import PointCloudToImageMapper  # noqa: F401
    print("  SceneFun3D toolkit: DataParser, fusion, viz and downloader import ok")


def main() -> int:
    report_versions()
    checks = [("DINOv2 on MPS/CPU", check_dinov2), ("Open3D", check_open3d),
              ("SceneFun3D toolkit", check_scenefun3d_toolkit)]
    failures = []
    for name, check in checks:
        print(f"[{name}]")
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - diagnostic script: report every failing check, not just the first
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append(name)
    print("ALL CHECKS PASSED" if not failures else f"FAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

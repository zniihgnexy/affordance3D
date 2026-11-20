"""scripts/fit_gaussians_batch.py: the video list, shards and the prediction check (CPU; gsplat is not imported)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fit_gaussians_batch", Path(__file__).resolve().parents[1] / "scripts" / "fit_gaussians_batch.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
ROOTS = {"primary": "data/scenefun3d", "revisit": "data/scenefun3d_revisit"}


def test_video_list_puts_the_primary_video_first_with_its_root():
    pairs = [["2", "c"], ["1", "b"], ["1", "a"], ["9", "z"]]
    out = B.video_list(["1", "2"], pairs, {"1": "b", "2": "c"}, ROOTS)
    assert [(r["visit"], r["video"], r["role"]) for r in out] == [("1", "b", "primary"), ("1", "a", "revisit"), ("2", "c", "primary")]
    assert out[1]["data_root"] == "data/scenefun3d_revisit"


def test_shards_cover_every_item_once():
    items = list(range(11))
    parts = [B.shard(items, f"{i}/4") for i in range(4)]
    assert sorted(x for p in parts for x in p) == items
    with pytest.raises(ValueError):
        B.shard(items, "4/4")


def test_judge_uses_medians_and_failures():
    found = [{"fit_s": 600, "test_psnr_db": 24.0, "test_ssim": 0.8, "test_depth_median_m": 0.02, "file_mb": 200, "peak_gpu_gb": 3,
              "gaussians": 10},
             {"fit_s": 1200, "test_psnr_db": 21.0, "test_ssim": 0.7, "test_depth_median_m": 0.05, "file_mb": 300, "peak_gpu_gb": 4,
              "gaussians": 20}]
    out = B.judge(found, failed=5, expect={"fit_min": [8, 25], "psnr_db_min": 22, "depth_median_m_max": 0.03, "failures_max": 4})
    assert out["G2-P1 median fit minutes"] == {"value": 15.0, "held": True}
    assert out["G2-P2 median held-out PSNR"]["held"] is True and out["G2-P2 median held-out PSNR"]["value"] == 22.5
    assert out["G2-P3 median depth error m"]["held"] is False
    assert out["G2-P4 videos fitted"] == {"value": "2 of 7", "held": False}

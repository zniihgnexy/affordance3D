"""Stage 2 (week-2 notebook 8.6): fit the 3D Gaussian map of every video of the validation scenes, the phase-1 video
under data/scenefun3d and the second-visit videos under data/scenefun3d_revisit, with the settings and steps of
scripts/fit_gaussians.py; sharded over GPU queue jobs, then summarised on the CPU.

    python scripts/fit_gaussians_batch.py --config configs/gaussians/fit_val82.yaml --shard 0/6     (GPU queue only)
    python scripts/fit_gaussians_batch.py --config configs/gaussians/fit_val82.yaml --stage summarise

Per video, <outputs>/<name>/ gets <visit>_<video>.npz, <visit>_<video>_views.png and <visit>_<video>.json (the same
measures as the trial); a video whose .json exists is skipped, so a stopped job resumes. A failing video writes
<visit>_<video>.failed.json and the job goes on. summarise writes summary.json: per-video rows, medians over videos,
failures, and each prediction of the config marked held or not.
"""
import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location("fit_gaussians", REPO_ROOT / "scripts" / "fit_gaussians.py")
FG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FG)  # the trial's loading, fitting and saving steps, unchanged

from affordance3d.device import default_device  # noqa: E402
from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402

MEASURES = ("fit_s", "test_psnr_db", "test_ssim", "test_depth_median_m", "file_mb", "peak_gpu_gb", "gaussians")


def rows(path: Path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text().split()[1:]]


def video_list(visits: list[str], pairs: list[list[str]], primary: dict[str, str], roots: dict[str, str]) -> list[dict]:
    """Every video of the given visits, the phase-1 (primary) one first, with its role and data root."""
    order = {visit: i for i, visit in enumerate(visits)}
    out = [{"visit": v, "video": vid, "role": "primary" if primary.get(v) == vid else "revisit",
            "data_root": roots["primary" if primary.get(v) == vid else "revisit"]}
           for v, vid, *_ in pairs if v in order]
    return sorted(out, key=lambda r: (order[r["visit"]], r["role"] != "primary", r["video"]))


def shard(items: list, spec: str) -> list:
    """The i-th of n interleaved parts of `items`, for spec "i/n"."""
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {spec}: need 0 <= i < n")
    return items[i::n]


def videos(config: dict) -> list[dict]:
    lists = REPO_ROOT / config["lists"]
    visits = (REPO_ROOT / config["scenes"]).read_text().split()
    primary = {v: vid for v, vid in rows(lists / "train_val_set_only_one_video.csv")}
    return video_list(visits, rows(lists / "train_val_set.csv"), primary, config["data_roots"])


def fit_one(config: dict, item: dict, folder: Path, device: str) -> None:
    stem = f"{item['visit']}_{item['video']}"
    if (folder / f"{stem}.json").exists():
        return
    tic = time.time()
    try:
        frames = FG.load_frames({**config, "data_root": item["data_root"]}, item["visit"], item["video"], config["frame_stride"])
        load_s = round(time.time() - tic, 1)
        params, stats = FG.fit(frames, config, device)
        row = {**item, "load_s": load_s, **stats, "file_mb": FG.save(params, frames, config, folder, stem)}
        (folder / f"{stem}.json").write_text(json.dumps(row, indent=1))
        (folder / f"{stem}.failed.json").unlink(missing_ok=True)  # an earlier failure of this video is superseded
        print(json.dumps(row), flush=True)
        del params
    except (RuntimeError, ValueError, OSError, KeyError, IndexError) as error:  # one video must not stop the job
        (folder / f"{stem}.failed.json").write_text(json.dumps({**item, "error": repr(error), "trace": traceback.format_exc()}))
        print(f"FAILED {stem}: {error!r}", flush=True)
    finally:
        import torch
        torch.cuda.empty_cache()


def medians(found: list[dict]) -> dict:
    return {k: float(np.median([r[k] for r in found if r.get(k) is not None])) for k in MEASURES if found}


def judge(found: list[dict], failed: int, expect: dict) -> dict:
    m = medians(found)
    return {"G2-P1 median fit minutes": {"value": round(m["fit_s"] / 60, 1),
                                         "held": expect["fit_min"][0] <= m["fit_s"] / 60 <= expect["fit_min"][1]},
            "G2-P2 median held-out PSNR": {"value": round(m["test_psnr_db"], 2), "held": m["test_psnr_db"] >= expect["psnr_db_min"]},
            "G2-P3 median depth error m": {"value": round(m["test_depth_median_m"], 3),
                                           "held": m["test_depth_median_m"] <= expect["depth_median_m_max"]},
            "G2-P4 videos fitted": {"value": f"{len(found)} of {len(found) + failed}", "held": failed <= expect["failures_max"]}}


def summarise(config: dict, folder: Path) -> dict:
    stems = [f"{v['visit']}_{v['video']}" for v in videos(config)]
    found = [json.loads((folder / f"{s}.json").read_text()) for s in stems if (folder / f"{s}.json").exists()]
    failed = [s for s in stems if (folder / f"{s}.failed.json").exists() and not (folder / f"{s}.json").exists()]
    missing = len(stems) - len(found) - len(failed)
    summary = {"videos": len(stems), "fitted": len(found), "failed": failed, "missing": missing,
               "medians": medians(found), "by_role": {role: medians([r for r in found if r["role"] == role]) for role in ("primary", "revisit")},
               "low_psnr": [f"{r['visit']}_{r['video']}" for r in found if r["test_psnr_db"] < config["expect"]["psnr_db_min"]],
               "predictions": judge(found, len(failed) + missing, config["expect"]) if found else {}, "rows": found}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    parser.add_argument("--shard", default="", help="i/n: fit the i-th of n parts (GPU)")
    parser.add_argument("--stage", choices=["fit", "summarise"], default="fit")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, f"{args.stage} {args.shard}".strip(), REPO_ROOT)
    if args.stage == "summarise":
        summary = summarise(config, folder)
        print(json.dumps({k: summary[k] for k in ("videos", "fitted", "failed", "missing", "medians", "predictions")}, indent=1))
        return
    device = default_device()
    if not device.startswith("cuda"):
        raise SystemExit("gsplat needs CUDA: submit with scripts/remote/queue_fit_gaussians_batch.sh")
    items = shard(videos(config), args.shard) if args.shard else videos(config)
    for item in items:
        fit_one(config, item, folder, device)


if __name__ == "__main__":
    main()

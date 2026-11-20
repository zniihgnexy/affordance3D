"""Fit a 3D Gaussian map to the phone video of a scene and time it (the trial asked for on 2026-11-20, week-2 notebook
7.11, question 3): ARKit poses and LiDAR depth as a robot would have them, Gaussians started from the depth, colour and
depth losses, gsplat's default densification. Needs gsplat and a CUDA device: run it through the GPU queue only
(scripts/remote/queue_fit_gaussians.sh), which sets AFFORDANCE3D_DEVICE=cuda.

    python scripts/fit_gaussians.py --config configs/gaussians/fit_trial.yaml

Writes under <outputs>/<name>/: <visit>.npz (the Gaussians, float16), <visit>_views.png (held-out photos, renders and
rendered depth) and summary.json (per scene: frames, Gaussians, fit seconds, peak GPU memory, test PSNR and SSIM, depth
error against LiDAR, file size; then each prediction of the config marked held or not).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.device import default_device  # noqa: E402
from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402
from affordance3d.mem import gaussian_map as G  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402

PARAMS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def load_frames(config: dict, visit: str, video: str, stride: int) -> list:
    """Every `stride`-th hires frame with its ARKit pose (in the scan frame) and its LiDAR depth."""
    scene = P.load_scene(REPO_ROOT / config["data_root"], visit, video)
    step = config["pixel_step"]
    return [P.input_frame(scene, t, P.lidar_depth(scene, t, step), step) for t in scene.vid.rgb_times[::stride]]


def tensors(frames: list, device: str) -> dict:
    return {"rgb": torch.tensor(np.stack([f.rgb for f in frames]), device=device).float() / 255.0,
            "depth": torch.tensor(np.stack([f.depth_m for f in frames]), device=device),
            "viewmats": torch.tensor(np.stack([np.linalg.inv(f.cam_to_world) for f in frames]), device=device).float(),
            "Ks": torch.tensor(np.stack([f.intrinsics for f in frames]), device=device).float()}


def render(params, data: dict, i: int, sh_degree: int):
    from gsplat import rasterization
    height, width = data["rgb"].shape[1:3]
    colours = torch.cat([params["sh0"], params["shN"]], dim=1)
    out, _, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                 torch.sigmoid(params["opacities"]), colours, data["viewmats"][i:i + 1],
                                 data["Ks"][i:i + 1], width, height, sh_degree=sh_degree, render_mode="RGB+ED",
                                 packed=False)
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], info


def fit(frames: list, config: dict, device: str) -> tuple[dict, dict]:
    from gsplat.strategy import DefaultStrategy
    train, test = G.split_frames(len(frames), config["test_every"])
    init = config["init"]
    points, colours = G.coloured_points([frames[i] for i in train], init["pixel_stride"])
    points, colours = G.voxel_mean(points, colours, init["voxel_m"])
    start = G.initial_gaussians(points, colours, init["voxel_m"], init["opacity"], config["sh_degree"])
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in start.items()})
    scale = G.scene_scale([frames[i] for i in train])
    lr = {**config["lr"], "means": config["lr"]["means"] * scale}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr[k], "name": k}], eps=1e-15) for k in PARAMS}
    strategy = DefaultStrategy(**config["strategy"])
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scale)
    data = tensors(frames, device)
    loss_cfg, rng = config["loss"], np.random.default_rng(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tic = time.time()
    for step in range(config["iterations"]):
        i = int(rng.choice(train))
        rgb, depth, info = render(params, data, i, min(step // 1000, config["sh_degree"]))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        photo = data["rgb"][i]
        l1 = (rgb - photo).abs().mean()
        d_ssim = 1.0 - G.ssim(rgb.permute(2, 0, 1)[None], photo.permute(2, 0, 1)[None])
        errors = G.depth_error(depth, data["depth"][i])
        loss = (1 - loss_cfg["ssim_weight"]) * l1 + loss_cfg["ssim_weight"] * d_ssim
        if errors.numel():
            loss = loss + loss_cfg["depth_weight"] * errors.mean()
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    torch.cuda.synchronize()
    stats = {"frames": len(frames), "train_frames": len(train), "test_frames": len(test), "start_gaussians": len(points),
             "gaussians": int(params["means"].shape[0]), "fit_s": round(time.time() - tic, 1),
             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "scene_scale_m": round(scale, 2)}
    return params, {**stats, **evaluate(params, data, test, config)}


@torch.no_grad()
def evaluate(params, data: dict, test: list[int], config: dict) -> dict:
    psnrs, ssims, errors = [], [], []
    for i in test:
        rgb, depth, _ = render(params, data, i, config["sh_degree"])
        photo = data["rgb"][i]
        psnrs.append(G.psnr(rgb, photo))
        ssims.append(float(G.ssim(rgb.permute(2, 0, 1)[None], photo.permute(2, 0, 1)[None])))
        errors.append(G.depth_error(depth, data["depth"][i]).cpu())
    errors = torch.cat(errors)
    return {"test_psnr_db": round(float(np.mean(psnrs)), 2), "test_ssim": round(float(np.mean(ssims)), 3),
            "test_depth_median_m": round(float(errors.median()), 3) if errors.numel() else None}


@torch.no_grad()
def save(params, frames: list, config: dict, folder: Path, visit: str) -> float:
    """The Gaussians as float16 (returns the file size in MB) and a figure of held-out views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = folder / f"{visit}.npz"
    np.savez_compressed(path, **{k: params[k].detach().cpu().numpy().astype(np.float16) for k in PARAMS})
    data = tensors(frames, params["means"].device)
    _, test = G.split_frames(len(frames), config["test_every"])
    shown = test[:: max(1, len(test) // config["figure_views"])][: config["figure_views"]]
    fig, axes = plt.subplots(len(shown), 3, figsize=(10, 3.4 * len(shown)), squeeze=False)
    for row, i in enumerate(shown):
        rgb, depth, _ = render(params, data, i, config["sh_degree"])
        for ax, image, title in zip(axes[row], (data["rgb"][i].cpu(), rgb.cpu(), depth.cpu()),
                                    ("held-out photo", "rendered", "rendered depth (m)")):
            ax.imshow(image, cmap="viridis" if image.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / f"{visit}_views.png", dpi=110)
    plt.close(fig)
    return round(path.stat().st_size / 2**20, 1)


def predictions(rows: list[dict], expect: dict) -> dict:
    val = next(r for r in rows if r["role"] == "validation")
    return {"G-P1 fit minutes": {"value": round(val["fit_s"] / 60, 1), "held": expect["fit_min"][0] <= val["fit_s"] / 60 <= expect["fit_min"][1]},
            "G-P2 test PSNR": {"value": val["test_psnr_db"], "held": val["test_psnr_db"] >= expect["psnr_db_min"]},
            "G-P3 file MB": {"value": val["file_mb"], "held": val["file_mb"] <= expect["file_mb_max"]},
            "G-P4 depth median m": {"value": val["test_depth_median_m"],
                                    "held": val["test_depth_median_m"] is not None and val["test_depth_median_m"] <= expect["depth_median_m_max"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    device = default_device()
    if not device.startswith("cuda"):
        raise SystemExit("gsplat needs CUDA: submit with scripts/remote/queue_fit_gaussians.sh (sets AFFORDANCE3D_DEVICE=cuda)")
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "fit", REPO_ROOT)
    torch.manual_seed(config["seed"])
    rows = []
    for scene in config["scenes"]:
        tic = time.time()
        frames = load_frames(config, scene["visit"], scene["video"], scene["frame_stride"])
        load_s = round(time.time() - tic, 1)
        params, stats = fit(frames, config, device)
        rows.append({**scene, "load_s": load_s, **stats, "file_mb": save(params, frames, config, folder, scene["visit"])})
        print(json.dumps(rows[-1]), flush=True)
        del params
        torch.cuda.empty_cache()
    summary = {"scenes": rows, "predictions": predictions(rows, config["expect"]),
               "gpu": torch.cuda.get_device_name(0)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["predictions"], indent=1))


if __name__ == "__main__":
    main()

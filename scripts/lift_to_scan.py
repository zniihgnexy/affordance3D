"""Lift the chosen masks to the laser scan, as fixed in the discussion draft, section 27.2 (CPU).

    python scripts/lift_to_scan.py --visit 421254 [--masks <folder>] [--out <folder>] [--rel-tol 0.25] [--memory-arm p4t_sam3]

For every arm of runs/scenefun3d_eval/box_masks/<visit>.npz (scripts/run_box_masks.py) and every task: the mask, made
twice as large by nearest neighbour, is lifted with affordance3d.mem.scan_lifting on the frame's camera at its native
resolution, using the poses registered to the scan (hires_poses) and the depth rendered from it (hires_depth), as Fun3DU
and UniFunc3D do. A frame needs its pose within 0.05 s and depth of the same timestamp; otherwise the task is empty
("no camera"). Two upper bounds of the lifting go the same way with the annotated target as the mask: on the frame the
memory arm (p4t_sam3) chose ("oracle_chosen") and on the keyframe that shows most target points ("oracle_best").

Writes, under --out (default runs/scenefun3d_eval/): lifted/<arm>/<visit>.json (the Fun3DU format: desc_id and cropped-scan indices),
official/<arm>/ (the official submission format: run-length masks over the full scan with SAM 3's score as confidence)
and lifted/causes_<visit>.json (why a prediction is empty, per arm; for oracle_chosen, "no box" means the memory arm
gave no usable frame).
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (imported quietly before the pipeline, which needs it)

from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from affordance3d.mem import scan_lifting as L  # noqa: E402
from affordance3d.mem import scenefun3d_eval as SE  # noqa: E402

EVAL = REPO_ROOT / "runs" / "scenefun3d_eval"
MEMORY_ARM = "p4t_sam3"
NATIVE_FRAMES_KEPT = 8
CAUSES = ("no box", "no camera", "no point")  # no box: the model chose no number or its answer could not be read


class Cameras:
    """The scan-registered frames of one scene at native and working resolution, loaded once each."""

    def __init__(self, scene: P.Scene) -> None:
        self.scene, self.native, self.working = scene, {}, {}

    def usable(self, t: str) -> bool:
        index = self.scene.gt_index
        return t in index.matched and t in index.depth

    def frame(self, t: str, native: bool):
        cache = self.native if native else self.working
        if t not in cache:
            if native and len(cache) >= NATIVE_FRAMES_KEPT:  # a native frame holds tens of MB: keep only the latest few
                cache.pop(next(iter(cache)))
            cache[t] = P.gt_frames(self.scene, [t], pixel_step=1 if native else P.PIXEL_STEP)[0]
        return cache[t]


def lift(points: np.ndarray, cameras: Cameras, t: str, working_mask: np.ndarray, rel_tol: float) -> np.ndarray:
    """Cropped-scan indices kept by the working-resolution mask of frame t."""
    frame = cameras.frame(t, native=True)
    factor = frame.image_hw[0] // working_mask.shape[0]
    if (working_mask.shape[0] * factor, working_mask.shape[1] * factor) != frame.image_hw:
        raise SystemExit(f"frame {t}: mask {working_mask.shape} does not scale to {frame.image_hw}")
    return np.flatnonzero(L.lift_mask(points, frame, L.upsample(working_mask, factor), rel_tol))


def task_parts(scene: P.Scene) -> list[tuple[int, ...]]:
    return [tuple(i for i, element in enumerate(scene.elements) if element.annot_id in annot_ids) for _, annot_ids in scene.tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--masks", type=Path, default=EVAL / "box_masks", help="folder of <visit>.npz from run_box_masks.py")
    parser.add_argument("--out", type=Path, default=EVAL)
    parser.add_argument("--rel-tol", type=float, default=L.REL_DEPTH_TOL)
    parser.add_argument("--memory-arm", default=MEMORY_ARM)
    args = parser.parse_args()
    visit = args.visit
    result = json.loads((REPO_ROOT / "runs" / "mem_phase1" / f"{visit}.json").read_text())
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", visit, result["video"])
    kept_full = np.flatnonzero(scene.crop == 1)
    points, n_full = scene.scan_xyz[kept_full], len(scene.crop)
    cameras = Cameras(scene)
    with np.load(args.masks / f"{visit}.npz") as saved:
        masks = {key: saved[key] for key in saved.files}
    desc_ids = [str(d) for d in masks["desc_ids"]]
    if len(desc_ids) != len(scene.tasks):
        raise SystemExit(f"{visit}: {len(desc_ids)} masks for {len(scene.tasks)} tasks")
    arms = sorted({key.split("/")[0] for key in masks if "/" in key})
    predictions, causes = {}, {}
    for arm in arms:
        hw = tuple(int(v) for v in masks[f"{arm}/image_hw"])
        rows, counts = [], dict.fromkeys(CAUSES, 0)
        for i, desc_id in enumerate(desc_ids):
            status, t, score = str(masks[f"{arm}/status"][i]), str(masks[f"{arm}/frames"][i]), float(masks[f"{arm}/scores"][i])
            cause = "no box" if status == "none" else "no camera" if not cameras.usable(t) else None
            indices = np.empty(0, dtype=np.int64)
            if cause is None:
                mask = np.unpackbits(masks[f"{arm}/masks"][i], count=hw[0] * hw[1]).astype(bool).reshape(hw)
                indices = lift(points, cameras, t, mask, args.rel_tol)
                cause = None if indices.size else "no point"
            if cause:
                counts[cause] += 1
            rows.append((desc_id, indices, score, t if cause in (None, "no point") else None))
        predictions[arm], causes[arm] = rows, counts
    # the two upper bounds of the lifting itself (27.2, item 5): the annotated target as the mask
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    working = [cameras.frame(t, native=False) if cameras.usable(t) else None for t in steady]
    chosen_frames = [row[3] for row in predictions[args.memory_arm]] if args.memory_arm in predictions else [None] * len(desc_ids)
    for name in ("oracle_chosen", "oracle_best"):
        rows, counts = [], dict.fromkeys(CAUSES, 0)
        for i, (desc_id, parts) in enumerate(zip(desc_ids, task_parts(scene))):
            target = np.concatenate([scene.part_points(p) for p in parts]) if parts else np.zeros((0, 3))
            if name == "oracle_chosen":
                t = chosen_frames[i]
            else:
                usable = [k for k, frame in enumerate(working) if frame is not None]
                best = L.most_visible(target, [working[k] for k in usable], args.rel_tol) if len(target) else None
                t = steady[usable[best]] if best is not None else None
            indices = np.empty(0, dtype=np.int64)
            if t is None or not cameras.usable(t):
                counts["no camera" if t is not None else "no box"] += 1
            else:
                region = L.region_mask(target, cameras.frame(t, native=False), args.rel_tol)
                indices = lift(points, cameras, t, region, args.rel_tol) if region.any() else indices
                if not indices.size:
                    counts["no point"] += 1
            rows.append((desc_id, indices, 1.0, t))
        predictions[name], causes[name] = rows, counts
    for arm, rows in predictions.items():
        folder = args.out / "lifted" / arm
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{visit}.json").write_text(json.dumps([{"desc_id": d, "pred_mask": idx.tolist()} for d, idx, _, _ in rows]))
        official = args.out / "official" / arm
        for desc_id, idx, score, _ in rows:
            full = np.zeros(n_full, dtype=bool)
            full[kept_full[idx]] = True
            SE.write_task(official, visit, desc_id, [full] if idx.size else [], [score] if idx.size else [], n_full)
    (args.out / "lifted" / f"causes_{visit}.json").write_text(json.dumps(causes, indent=1))
    print(visit, json.dumps(causes), flush=True)


if __name__ == "__main__":
    main()

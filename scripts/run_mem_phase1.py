"""Phase 1 on one scene: where should a memory put its tokens? (specification: discussion draft, section 21)

    python scripts/run_mem_phase1.py --visit 421254                 # one scene, everything except the model-reading step
    python scripts/run_mem_phase1.py --visit 421254 --vlm           # also let the vision-language model pick the part

Writes runs/mem_phase1/<visit>.json. The affordance map is ORACLE (projected annotations without part ids), so
every number is an upper bound. Runs on the CPU unless AFFORDANCE3D_DEVICE=cuda is set by a queue job.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.features import DinoV2Extractor
from affordance3d.mem import fusion, keyframes, policies
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from affordance3d.mem.actionable import distance_to_part
from affordance3d.mem.encoders import VLM_REVISION, ImageTextEncoder, VisionLanguageModel
from affordance3d.scenefun3d_partial import N_KEYFRAMES, benchmark_video
from report import mem_steps_helpers as H
from report.mem_demo_helpers import contact_error_cm, scaled_mono_depths, scan_view, stand_free_in_scan

LISTS = REPO_ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists"
MONO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
TOP_KEYFRAMES, PATCH, MAX_MARKS = 3, 14, 40
P4_BUDGETS, OTHER_BUDGETS = (0.25, 0.5, 1.0), (0.5, 1.0, 2.0, 4.0, 8.0)
VOXELS_M = (0.01, 0.02, 0.05)
NEAR_PART_M = 0.5  # fused points further than this from the single-frame estimate cannot belong to the part


def part_actions_of(scene: P.Scene) -> list[int]:
    return [T.ACTIONS.index(e.label) if e.label in T.ACTIONS else T.NO_ACTION for e in scene.elements]


def parts_seen(labels: list[np.ndarray], n_parts: int) -> int:
    most = np.array([[(lab == p).sum() for lab in labels] for p in range(n_parts)]).max(axis=1)
    return int((most >= P.MIN_PART_PIXELS).sum())


def part_centres(labels: list[np.ndarray], parts: tuple[int, ...]) -> dict[int, list[tuple[int, int]]]:
    """For each keyframe that shows a target part: the centre pixel (u, v) of every target part in it (scoring only)."""
    centres: dict[int, list[tuple[int, int]]] = {}
    for k, lab in enumerate(labels):
        for part in parts:
            rows_cols = np.argwhere(lab == part)
            if len(rows_cols) >= P.MIN_PART_PIXELS:
                v, u = np.median(rows_cols, axis=0)
                centres.setdefault(k, []).append((int(u), int(v)))
    return centres


class CropEncoder:
    """Image-text features of crops, each crop encoded once."""

    def __init__(self, encoder: ImageTextEncoder, frames: list) -> None:
        self._encoder, self._frames, self._known = encoder, frames, {}
        self.crops_encoded = 0

    def __call__(self, candidates: list[policies.Candidate]) -> np.ndarray:
        new = [c for c in dict.fromkeys((c.keyframe, c.crop) for c in candidates) if c not in self._known]
        if new:
            crops = [self._frames[k].rgb[top:bottom, left:right] for k, (left, top, right, bottom) in new]
            self._known.update(zip(new, self._encoder.images(crops)))
            self.crops_encoded += len(new)
        return np.array([self._known[(c.keyframe, c.crop)] for c in candidates])


def evaluate(name: str, budget: float, seed: int, candidates: list[policies.Candidate], vectors: np.ndarray, queries: np.ndarray,
             keyframe_vectors: np.ndarray, tasks: list, centres: list[dict], image_hw: tuple[int, int]) -> dict:
    """Retrieval scores of one way of placing tokens (an empty candidate list = whole keyframes only)."""
    hit1, hit3, on_part = [], [], []
    for query, task, centre in zip(queries, tasks, centres):
        scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
        first = S.first_hit(scores.ranking("combined" if candidates else "semantic"), set(task.keyframes))
        hit1.append(int(first is not None and first <= 1))
        hit3.append(int(first is not None and first <= 3))
        if candidates:
            best = candidates[int(np.argmax(scores.token_scores))]
            on_part.append(int(any(policies.covers(best, pixel) for pixel in centre.get(best.keyframe, []))))
    return {"policy": name, "budget": budget, "seed": seed, "tokens": len(candidates), "hit1": hit1, "hit3": hit3, "on_part": on_part,
            "area_share": policies.median_area_share(candidates, image_hw)}


def search_results(memory: T.L2Memory, frames: list, encoder: ImageTextEncoder, keyframe_vectors: np.ndarray, tasks: list,
                   centres: list[dict], queries: np.ndarray, seeds: int, max_budget: float,
                   extra_pool: list[policies.Candidate] | None) -> tuple[list[dict], dict, int]:
    """Retrieval scores of every policy and budget; also the token sets (with features) the model-reading step uses."""
    image_hw, n_keyframes = frames[0].image_hw, len(frames)
    embed = CropEncoder(encoder, frames)
    afford = policies.affordance_candidates([t.region for t in memory.tokens], image_hw)
    known = {(c.keyframe, c.crop): t.semantic for c, t in zip(afford, memory.tokens)}
    embed._known.update(known)  # the memory already holds these features

    def run(name: str, budget: float, seed: int, candidates: list[policies.Candidate]) -> dict:
        vectors = embed(candidates) if candidates else np.zeros((0, keyframe_vectors.shape[1]))
        return evaluate(name, budget, seed, candidates, vectors, queries, keyframe_vectors, tasks, centres, image_hw)

    results = [run("P0 whole keyframes", 0.0, 0, [])]
    reading_sets = {"P4 affordance regions": (afford, embed(afford))}
    for budget in P4_BUDGETS:
        for seed in range(1 if budget == 1.0 else seeds):
            kept = policies.subsample(afford, max(1, round(budget * len(afford))), np.random.default_rng(seed))
            results.append(run("P4 affordance regions", budget, seed, kept))
    for shape in policies.GRID_SHAPES:
        cells = policies.grid_candidates(n_keyframes, image_hw, shape if image_hw[0] >= image_hw[1] else shape[::-1])
        if len(cells) <= max_budget * len(afford) * 1.5:
            results.append(run(f"P1 uniform grid {shape[0]}x{shape[1]}", len(cells) / len(afford), 0, cells))
    for budget in (b for b in OTHER_BUDGETS if b <= max_budget):
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            n = round(budget * len(afford))
            results.append(run("P2 random places", budget, seed, policies.random_candidates(afford, n, n_keyframes, image_hw, rng)))
            if extra_pool:
                generic = policies.size_matched(extra_pool, afford, n, rng)
                results.append(run("P3 generic regions", budget, seed, generic))
                if budget == 1.0 and seed == 0:
                    reading_sets["P3 generic regions"] = (generic, embed(generic))
    return results, reading_sets, embed.crops_encoded


def geometry_results(scene: P.Scene, memory: T.L2Memory, times: list[str], frames: list, labels: list[np.ndarray],
                     truth: np.ndarray) -> dict:
    """Contact points read three ways, scored against the scan: L2 read-out, single evidence frame per true part, fused cloud."""
    view = scan_view(scene)

    def summary(entries: list) -> dict:
        errors = np.array([contact_error_cm(scene, e) for e in entries])
        return {"parts": len(entries), "contact_cm": errors.round(2).tolist(),
                "stand_free": [bool(stand_free_in_scan(e, view)) for e in entries]}

    start = time.time()
    l2 = H.readout_memory(scene, memory, frames, truth)
    read_ms = 1000 * (time.time() - start) / max(len(l2.entries), 1)
    explicit = P.build_memory(scene, times, frames, labels)
    out = {"l2_readout": summary(list(l2.entries)), "single_frame": summary(list(explicit.entries)), "read_ms_per_part": read_ms, "fused": {}}
    for voxel in VOXELS_M:
        start = time.time()
        cloud = fusion.fuse_cloud(frames, voxel)
        build_s = time.time() - start
        errors, start = [], time.time()
        for entry in explicit.entries:
            masks = [lab == entry.part if (lab == entry.part).sum() >= P.MIN_PART_PIXELS else None for lab in labels]
            near = cloud[np.linalg.norm(cloud - entry.contact, axis=1) < NEAR_PART_M]
            contact = fusion.fused_contact(near, frames, masks) if len(near) else None
            errors.append(None if contact is None else round(100 * distance_to_part(contact, scene.part_points(entry.part)), 2))
        out["fused"][str(voxel)] = {"contact_cm": errors, "points": len(cloud), "bytes": fusion.cloud_bytes(cloud), "build_s": build_s,
                                    "read_ms_per_part": 1000 * (time.time() - start) / max(len(explicit.entries), 1)}
    return out


def storage_bytes(scene: P.Scene, times: list[str], frames: list, action_maps: list[np.ndarray], memory: T.L2Memory) -> dict:
    def png(image: np.ndarray) -> int:
        return len(cv2.imencode(".png", image)[1])

    return {"keyframe_images": sum(Path(scene.vid.rgb_path(t)).stat().st_size for t in times),
            "depth_maps": sum(png((f.depth_m * 1000).astype(np.uint16)) for f in frames),
            "affordance_maps": sum(png((a + 1).astype(np.uint8)) for a in action_maps),
            "token_features": memory.feature_bytes(), "tokens": len(memory.tokens)}


def reading_results(reader: H.CachedReader, candidates: list[policies.Candidate], vectors: np.ndarray, frames: list,
                    keyframe_vectors: np.ndarray, queries: np.ndarray, tasks: list, centres: list[dict]) -> dict:
    """The model picks a numbered region in the three best keyframes of each task; right = the region is on the target part."""
    right, seconds, shown = [], [], []
    for query, task, centre in zip(queries, tasks, centres):
        scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
        top = [int(k) for k in scores.ranking()[:TOP_KEYFRAMES]]
        inside = sorted((i for i, c in enumerate(candidates) if c.keyframe in top), key=lambda i: -scores.token_scores[i])[:MAX_MARKS]
        kept = [candidates[i] for i in sorted(inside)]
        images, marks = H.mark_boxes(frames, [(c.keyframe, c.box) for c in kept], top)
        shown.append(len(marks))
        if not marks:
            right.append(0)
            continue
        prompt = S.mark_prompt(task.text, len(marks))
        reply = reader.ask(H.reading_key(VLM_REVISION, prompt, top, [kept[i].box for i in marks]), images, prompt)
        mark = S.parse_mark(reply["answer"], len(marks))
        chosen = None if mark is None else kept[marks[mark - 1]]
        right.append(int(chosen is not None and any(policies.covers(chosen, pixel) for pixel in centre.get(chosen.keyframe, []))))
        seconds.append(reply["seconds"])
    return {"right_part": right, "seconds": seconds, "marks_shown": shown}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--video", default=None, help="default: the video the benchmark file list gives for the scene")
    parser.add_argument("--keyframes", default="steady", choices=keyframes.KINDS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-budget", type=float, default=8.0)
    parser.add_argument("--vlm", action="store_true", help="also run the model-reading step (slow on the CPU)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    video = args.video or benchmark_video(LISTS, args.visit)
    started = time.time()
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, video)
    times = keyframes.scene_keyframes(scene, args.keyframes, N_KEYFRAMES)
    mono = scaled_mono_depths(scene, times, MONO_ID, args.out / "mono" / video)
    frames = [P.input_frame(scene, t, d) for t, d in zip(times, mono)]
    no_lidar = [k for k, t in enumerate(times) if not (P.lidar_depth(scene, t) > 0).any()]  # depth scale from the other keyframes
    labels = P.label_maps(scene, P.gt_frames(scene, times))  # ORACLE with part ids: scoring only
    actions = part_actions_of(scene)
    action_maps = [T.action_map(lab, actions) for lab in labels]
    image_hw = frames[0].image_hw
    result = {"visit": args.visit, "video": video, "keyframes": args.keyframes, "n_keyframes": len(times), "image_hw": list(image_hw),
              "parts": len(scene.elements), "parts_seen": {args.keyframes: parts_seen(labels, len(scene.elements))},
              "keyframes_without_lidar": no_lidar}
    other = "uniform" if args.keyframes == "steady" else "steady"
    other_times = keyframes.scene_keyframes(scene, other, N_KEYFRAMES)
    result["parts_seen"][other] = parts_seen(P.label_maps(scene, P.gt_frames(scene, other_times)), len(scene.elements))

    write_start = time.time()
    dino = DinoV2Extractor(input_hw=(image_hw[0] // PATCH * PATCH, image_hw[1] // PATCH * PATCH))
    grids = [dino(frame.rgb).grid for frame in frames]
    encoder = ImageTextEncoder()
    keyframe_vectors = encoder.images([frame.rgb for frame in frames])
    memory = T.write_memory(times, frames, action_maps, grids, encoder.images, scene.up)
    result["write_s"] = time.time() - write_start
    if not memory.tokens:
        raise SystemExit(f"{args.visit}: no affordance region in the keyframes")
    truth = H.truth_parts(memory, labels, len(scene.elements))
    link = H.linking_scores(memory, truth).iloc[0]
    result["memory"] = {"tokens": len(memory.tokens), "parts": memory.n_parts, "true_parts_with_token": int(link["true parts with a token"]),
                        "linked_exactly": link["parts linked exactly right"], "mixed_regions": H.mixed_regions(memory, labels)}
    tasks_all = H.tasks_of(scene, labels)
    tasks = [task for task in tasks_all if task.keyframes]
    result["tasks"] = {"all": len(tasks_all), "answerable": len(tasks), "texts": [task.text for task in tasks]}
    regions_file = args.out / "regions" / f"{args.visit}.json"
    pool = None
    if regions_file.exists():  # made by scripts/make_generic_regions.py for the same keyframes
        saved = json.loads(regions_file.read_text())
        if saved["keyframes"] != args.keyframes or saved["n_keyframes"] != len(times):
            raise SystemExit(f"{regions_file}: made for other keyframes")
        pool = [policies.Candidate(k, (left, top, right, bottom), T.context_box((left, top, right, bottom), image_hw))
                for k, left, top, right, bottom in saved["regions"]]
    result["generic_regions"] = None if pool is None else len(pool)
    queries = encoder.texts([task.text for task in tasks])
    centres = [part_centres(labels, task.parts) for task in tasks]
    result["search"], reading_sets, result["crops_encoded"] = search_results(memory, frames, encoder, keyframe_vectors, tasks, centres,
                                                                             queries, args.seeds, args.max_budget, pool)
    result["geometry"] = geometry_results(scene, memory, times, frames, labels, truth)
    result["storage"] = storage_bytes(scene, times, frames, action_maps, memory)
    if args.vlm:
        reader = H.CachedReader(args.out / f"vlm_answers_{args.visit}.json", VisionLanguageModel)
        result["reading"] = {name: reading_results(reader, candidates, vectors, frames, keyframe_vectors, queries, tasks, centres)
                             for name, (candidates, vectors) in reading_sets.items()}
    result["total_s"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.visit}.json").write_text(json.dumps(result, indent=1))
    one = next(r for r in result["search"] if r["policy"].startswith("P4") and r["budget"] == 1.0)
    zero = result["search"][0]
    print(f"{args.visit}: {len(tasks)} tasks, {len(memory.tokens)} tokens; first keyframe right: whole frames "
          f"{np.mean(zero['hit1']):.0%}, affordance tokens {np.mean(one['hit1']):.0%}; {result['total_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

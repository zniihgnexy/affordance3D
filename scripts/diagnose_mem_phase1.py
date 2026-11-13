"""Phase-1 diagnostics (discussion draft, section 23): where does the gain of affordance tokens come from?

    python scripts/diagnose_mem_phase1.py --features       # encode once per scene on the CPU (about an hour for 30 scenes)
    python scripts/diagnose_mem_phase1.py                  # score every variant from the saved features

Reads the scene results in runs/mem_phase1/, writes runs/mem_phase1/diagnostics/<visit>.npz and
runs/mem_phase1/diagnostics.json. Nothing here changes the phase-1 results or the G1 rule.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.mem import keyframes, policies
from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H
from report.mem_demo_helpers import scaled_mono_depths

MONO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
SEEDS = (0, 1, 2)
BUDGETS = (1.0, 2.0, 4.0, 8.0)
GRID = (2, 1)  # the grid closest to budget 1 that ran in every scene
WEIGHTS, EMPTY_TERMS = (0.25, 0.5, 1.0, 2.0), (0.0, -1.0, -3.0)
RUN_NAMES = {"P0": "P0", "P4": "P4", "P2": "P2", "P3": "P3", "P1": f"P1 uniform grid {GRID[0]}x{GRID[1]}"}


def token_sets(frames: list, action_maps: list, pool: list) -> dict[str, list[policies.Candidate]]:
    """The token places that are scored: the run's own draws, and the two sets that copy P4's keyframes."""
    image_hw = frames[0].image_hw
    sets = {}
    for seed in SEEDS:
        drawn = R.token_sets(frames, action_maps, pool, seed)
        sets[f"P2 s{seed}"], sets[f"P3 s{seed}"], sets["P4"] = drawn[A.P2], drawn[A.P3], drawn[A.P4]
        rng = np.random.default_rng(2000 + seed)
        sets[f"P2 same keyframes s{seed}"] = D.random_in_same_keyframes(sets["P4"], image_hw, rng)
        sets[f"P3 same keyframes s{seed}"] = D.generic_in_same_keyframes(pool, sets["P4"], rng)
    sets["P1"] = policies.grid_candidates(len(frames), image_hw, GRID if image_hw[0] >= image_hw[1] else GRID[::-1])
    return sets


def empty_keyframes(frames: list, afford: list, pool: list, seeds: int = 5) -> dict[str, float]:
    """Keyframes left without a token by the run's draws, per policy and budget (mean over seeds); no encoding needed."""
    n_keyframes, image_hw = len(frames), frames[0].image_hw
    counts: dict[str, list[int]] = {}
    for budget in BUDGETS:
        for seed in range(seeds):
            rng, n = np.random.default_rng(R.RUNNER_SEED + seed), round(budget * len(afford))
            for name, drawn in (("P2", policies.random_candidates(afford, n, n_keyframes, image_hw, rng)),
                                ("P3", policies.size_matched(pool, afford, n, rng))):
                counts.setdefault(f"{name} at {budget:g}x", []).append(n_keyframes - len({c.keyframe for c in drawn}))
    empty = {name: float(np.mean(values)) for name, values in counts.items()}
    empty["P4 at 1x"] = float(n_keyframes - len({c.keyframe for c in afford}))
    return empty


def features(result: dict, runs: Path, encoder: object) -> dict:
    """Everything the scoring needs for one scene: vectors of keyframes, queries and token sets, and the ground truth."""
    visit = result["visit"]
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", visit, result["video"])
    times = keyframes.scene_keyframes(scene, result["keyframes"], result["n_keyframes"])
    mono = scaled_mono_depths(scene, times, MONO_ID, runs / "mono" / result["video"])
    frames = [P.input_frame(scene, t, d) for t, d in zip(times, mono)]
    labels = P.label_maps(scene, P.gt_frames(scene, times))  # ORACLE with part ids: scoring only
    actions = [T.ACTIONS.index(e.label) if e.label in T.ACTIONS else T.NO_ACTION for e in scene.elements]
    action_maps = [T.action_map(label, actions) for label in labels]
    pool = R.generic_pool(runs / "regions" / f"{visit}.json", frames[0].image_hw)
    sets = token_sets(frames, action_maps, pool)
    tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
    if [task.text for task in tasks] != result["tasks"]["texts"]:
        raise SystemExit(f"{visit}: the tasks differ from those of the run")
    crops = list(dict.fromkeys((c.keyframe, c.crop) for candidates in sets.values() for c in candidates))
    vectors = dict(zip(crops, encoder.images([frames[k].rgb[top:bottom, left:right] for k, (left, top, right, bottom) in crops])))
    arrays = {"keyframes": encoder.images([frame.rgb for frame in frames]), "queries": encoder.texts([task.text for task in tasks])}
    for name, candidates in sets.items():
        arrays[f"vectors/{name}"] = np.array([vectors[(c.keyframe, c.crop)] for c in candidates]).reshape(len(candidates), -1)
        arrays[f"keyframe/{name}"] = np.array([c.keyframe for c in candidates], dtype=np.int64)
        arrays[f"boxes/{name}"] = np.array([c.box for c in candidates], dtype=np.int64).reshape(len(candidates), 4)
    truth = {"relevant": [sorted(task.keyframes) for task in tasks],
             "centres": [{str(k): pixels for k, pixels in R.target_pixels(labels, task.parts).items()} for task in tasks],
             "empty_keyframes": empty_keyframes(frames, sets["P4"], pool), "crops_encoded": len(crops)}
    return {**arrays, "truth": np.array(json.dumps(truth))}


def make_features(results: list[dict], runs: Path, out: Path) -> None:
    from affordance3d.mem.encoders import ImageTextEncoder

    out.mkdir(parents=True, exist_ok=True)
    encoder = ImageTextEncoder()
    for result in results:
        file = out / f"{result['visit']}.npz"
        if file.exists():
            continue
        start = time.time()
        saved = features(result, runs, encoder)
        np.savez_compressed(file.with_suffix(".tmp.npz"), **saved)
        file.with_suffix(".tmp.npz").rename(file)
        print(f"{result['visit']}: {json.loads(str(saved['truth']))['crops_encoded']} crops, {time.time() - start:.0f} s", flush=True)


def candidates_of(scene: dict, name: str) -> list[policies.Candidate]:
    return [policies.Candidate(int(k), tuple(int(v) for v in box), (0, 0, 0, 0))
            for k, box in zip(scene[f"keyframe/{name}"], scene[f"boxes/{name}"])]


def hits(scene: dict, name: str | None, weight: float = 1.0, empty_z: float = S.EMPTY_KEYFRAME_Z, shortlist: bool = False) -> np.ndarray:
    """Per task: 1 when the first keyframe is right. `name` is a token set; None = whole keyframes;
    `shortlist` = whole keyframes plus the one bit per keyframe taken from the keyframes of the P4 tokens."""
    keyframe_vectors, relevant = scene["keyframes"], scene["truth"]["relevant"]
    has_token = np.zeros(len(keyframe_vectors), dtype=bool)
    has_token[scene["keyframe/P4"]] = True
    out = []
    for query, right in zip(scene["queries"], relevant):
        if shortlist:
            scores = D.shortlist_scores(query, keyframe_vectors, has_token, weight, empty_z)
        elif name is None:
            scores = D.keyframe_scores(query, keyframe_vectors, np.zeros((0, keyframe_vectors.shape[1])), [])
        else:
            scores = D.keyframe_scores(query, keyframe_vectors, scene[f"vectors/{name}"], scene[f"keyframe/{name}"], weight, empty_z)
        out.append(D.first_right(scores, right))
    return np.array(out, dtype=float)


def over_seeds(scene: dict, name: str, **settings: float) -> np.ndarray:
    return np.mean([hits(scene, f"{name} s{seed}", **settings) for seed in SEEDS], axis=0)


def with_misses(scene: dict, share: float, empty_z: float = S.EMPTY_KEYFRAME_Z, seeds: int = 5) -> np.ndarray:
    """P4 with only a random `share` of its tokens, as the run subsamples them (an affordance map that misses parts)."""
    vectors, token_keyframe, out = scene["vectors/P4"], scene["keyframe/P4"], []
    for seed in range(seeds):
        keep = np.sort(np.random.default_rng(seed).permutation(len(vectors))[:max(1, round(share * len(vectors)))])
        thinned = {**scene, "vectors/thinned": vectors[keep], "keyframe/thinned": token_keyframe[keep]}
        out.append(hits(thinned, "thinned", empty_z=empty_z))
    return np.mean(out, axis=0)


def median_area(scene: dict, name: str) -> float:
    boxes = scene[f"boxes/{name}"]
    return float(np.median((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))) if len(boxes) else float("nan")


def covered(scene: dict, name: str) -> np.ndarray:
    """Per task: 1 when one of the boxes the model-reading step would show lies on the target part."""
    candidates, out = candidates_of(scene, name), []
    for query, centres in zip(scene["queries"], scene["truth"]["centres"]):
        shown = D.shown_candidates(query, scene["keyframes"], scene[f"vectors/{name}"], candidates)
        out.append(int(any(policies.covers(c, pixel) for c in shown for pixel in centres.get(str(c.keyframe), []))))
    return np.array(out, dtype=float)


def triple(values: tuple[float, float, float]) -> dict:
    return {"rate": values[0], "low": values[1], "high": values[2]}


def repeats_the_run(scenes: list[dict], results: list[dict]) -> dict:
    """How many task outcomes of the GPU run the CPU features reproduce (seeds 0 to 2, budget 1)."""
    same = {}
    for name, prefix in RUN_NAMES.items():
        match = total = 0
        for scene, result in zip(scenes, results):
            for seed in (SEEDS if name in ("P2", "P3") else (0,)):
                stored = next(run["hit1"] for run in result["search"] if run["policy"].startswith(prefix) and run["seed"] == seed
                              and (name in ("P0", "P1") or abs(run["budget"] - 1.0) < 1e-9))
                ours = hits(scene, None if name == "P0" else f"{name} s{seed}" if name in ("P2", "P3") else name)
                match, total = match + int((ours == np.array(stored)).sum()), total + len(stored)
        same[name] = f"{match} of {total}"
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--features", action="store_true", help="encode the scenes (slow); without it, score the saved features")
    args = parser.parse_args()
    results = A.scored(A.load_results(args.runs))
    out = args.runs / "diagnostics"
    if args.features:
        make_features(results, args.runs, out)
        return
    scenes = []
    for result in results:
        with np.load(out / f"{result['visit']}.npz") as saved:
            scenes.append({**{key: saved[key] for key in saved.files if key != "truth"}, "truth": json.loads(str(saved["truth"]))})

    arms = {"P0 whole keyframes": [hits(s, None) for s in scenes], "P4 affordance regions": [hits(s, "P4") for s in scenes],
            "P3 generic regions": [over_seeds(s, "P3") for s in scenes], "P2 random places": [over_seeds(s, "P2") for s in scenes],
            "P1 uniform grid 2x1": [hits(s, "P1") for s in scenes],
            "whole keyframes + which keyframes hold an affordance region": [hits(s, None, shortlist=True) for s in scenes],
            "P4, no penalty for keyframes without a token": [hits(s, "P4", empty_z=0.0) for s in scenes],
            "P3, no penalty for keyframes without a token": [over_seeds(s, "P3", empty_z=0.0) for s in scenes],
            "P2, no penalty for keyframes without a token": [over_seeds(s, "P2", empty_z=0.0) for s in scenes],
            "P3 in the keyframes of P4": [over_seeds(s, "P3 same keyframes") for s in scenes],
            "P2 in the keyframes of P4": [over_seeds(s, "P2 same keyframes") for s in scenes]}
    pairs = (("P4 affordance regions", "whole keyframes + which keyframes hold an affordance region"),
             ("whole keyframes + which keyframes hold an affordance region", "P0 whole keyframes"),
             ("P4 affordance regions", "P3 in the keyframes of P4"), ("P4 affordance regions", "P2 in the keyframes of P4"),
             ("P3 in the keyframes of P4", "whole keyframes + which keyframes hold an affordance region"),
             ("P4, no penalty for keyframes without a token", "P3, no penalty for keyframes without a token"),
             ("P4, no penalty for keyframes without a token", "P0 whole keyframes"),
             ("P3, no penalty for keyframes without a token", "P0 whole keyframes"))
    sweep = {}
    for weight in WEIGHTS:
        for empty_z in EMPTY_TERMS:
            setting = {"weight": weight, "empty_z": empty_z}
            rows = {"P4": [hits(s, "P4", **setting) for s in scenes], "P1": [hits(s, "P1", **setting) for s in scenes],
                    "P3": [over_seeds(s, "P3", **setting) for s in scenes], "P2": [over_seeds(s, "P2", **setting) for s in scenes],
                    "shortlist": [hits(s, None, shortlist=True, **setting) for s in scenes],
                    "P3 same keyframes": [over_seeds(s, "P3 same keyframes", **setting) for s in scenes],
                    "P2 same keyframes": [over_seeds(s, "P2 same keyframes", **setting) for s in scenes]}
            sweep[f"weight {weight:g}, empty {empty_z:g}"] = rows
    swept = ("P4", "P1", "P2", "P3", "shortlist", "P3 same keyframes", "P2 same keyframes")
    best = {name: max(sweep, key=lambda setting: A.pooled(sweep[setting][name])) for name in swept}
    whole = arms["P0 whole keyframes"]
    misses = {f"P4 with {share:g} of its tokens, empty {empty_z:g}": [with_misses(s, share, empty_z) for s in scenes]
              for share in (0.25, 0.5) for empty_z in (S.EMPTY_KEYFRAME_Z, 0.0)}
    stored = {share: A.pooled([A.scene_rates(result, A.P4, share) for result in results]) for share in (0.25, 0.5)}
    chance = np.array([D.chance_rates(right, np.isin(np.arange(len(s["keyframes"])), s["keyframe/P4"]))
                       for s in scenes for right in s["truth"]["relevant"]])
    empty_names = scenes[0]["truth"]["empty_keyframes"]
    summary = {
        "scenes": len(scenes), "tasks": int(sum(len(s["queries"]) for s in scenes)), "seeds": list(SEEDS),
        "repeats_the_run": repeats_the_run(scenes, results),
        "first_keyframe_right": {name: triple(A.bootstrap(per_scene)) for name, per_scene in arms.items()},
        "differences": {f"{first} minus {second}": triple(A.paired_difference(arms[first], arms[second])) for first, second in pairs},
        "sweep": {setting: {name: A.pooled(per_scene) for name, per_scene in rows.items()} for setting, rows in sweep.items()},
        "best_setting": best,
        "at_best_settings": {f"P4 (default) minus {name} at its best setting": triple(A.paired_difference(arms["P4 affordance regions"],
                                                                                                             sweep[best[name]][name]))
                             for name in swept[1:]},
        "same_setting": {setting: {f"P4 minus {name}": triple(A.paired_difference(rows["P4"], rows[name]))
                                   for name in ("P3", "P3 same keyframes")} for setting, rows in sweep.items()},
        "misses": {name: {**triple(A.bootstrap(per_scene)), "minus P0": triple(A.paired_difference(per_scene, whole))}
                   for name, per_scene in misses.items()},
        "misses_repeat_the_run": {f"{share:g}": [stored[share], A.pooled(misses[f"P4 with {share:g} of its tokens, empty -3"])]
                                  for share in (0.25, 0.5)},
        "median_region_area_px": {name: float(np.median([median_area(s, key) for s in scenes]))
                                  for name, key in (("P4", "P4"), ("P3 same keyframes", "P3 same keyframes s0"),
                                                    ("P2 same keyframes", "P2 same keyframes s0"), ("P3", "P3 s0"))},
        "area_ratio_same_keyframes_to_p4": float(np.median([median_area(s, "P3 same keyframes s0") / median_area(s, "P4") for s in scenes])),
        "chance": {"a random keyframe": float(chance[:, 0].mean()), "a random keyframe among those with an affordance region": float(chance[:, 1].mean())},
        "a_shown_box_is_on_the_target": {name: triple(A.bootstrap([covered(s, key) for s in scenes]))
                                         for name, key in (("P4 affordance regions", "P4"), ("P3 generic regions", "P3 s0"))},
        "keyframes_without_a_token_of_40": {name: float(np.mean([s["truth"]["empty_keyframes"][name] for s in scenes])) for name in empty_names},
        "tokens_per_scene": {name: float(np.mean([len(s[f"keyframe/{name}"]) for s in scenes]))
                             for name in ("P4", "P3 s0", "P3 same keyframes s0", "P2 same keyframes s0", "P1")},
    }
    (args.runs / "diagnostics.json").write_text(json.dumps(summary, indent=1))
    print("repeats the run:", summary["repeats_the_run"])
    for name, row in summary["first_keyframe_right"].items():
        print(f"  {name}: {100 * row['rate']:.1f} [{100 * row['low']:.1f}, {100 * row['high']:.1f}]")
    for name, row in {**summary["differences"], **summary["at_best_settings"]}.items():
        print(f"  {name}: {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    print("  best settings:", best)
    for name in ("P3", "P3 same keyframes"):
        rows = [(setting, row[f"P4 minus {name}"]) for setting, row in summary["same_setting"].items()]
        low, high = min(rows, key=lambda r: r[1]["rate"]), max(rows, key=lambda r: r[1]["rate"])
        print(f"  same setting, P4 minus {name}: from {100 * low[1]['rate']:+.1f} [{100 * low[1]['low']:+.1f}, {100 * low[1]['high']:+.1f}] "
              f"({low[0]}) to {100 * high[1]['rate']:+.1f} ({high[0]}); intervals above 0 in "
              f"{sum(r[1]['low'] > 0 for r in rows)} of {len(rows)} settings")
    for name, row in summary["misses"].items():
        print(f"  {name}: {100 * row['rate']:.1f}; minus P0 {100 * row['minus P0']['rate']:+.1f} "
              f"[{100 * row['minus P0']['low']:+.1f}, {100 * row['minus P0']['high']:+.1f}]")
    print("  misses repeat the run (stored, here):", {k: [round(100 * v, 1) for v in pair] for k, pair in summary["misses_repeat_the_run"].items()})
    print("  median region area (px):", summary["median_region_area_px"], "ratio", round(summary["area_ratio_same_keyframes_to_p4"], 2))
    print("  chance:", {name: round(value, 3) for name, value in summary["chance"].items()})
    print("  a shown box is on the target:", {name: round(row["rate"], 3) for name, row in summary["a_shown_box_is_on_the_target"].items()})
    print("  keyframes without a token:", {name: round(value, 1) for name, value in summary["keyframes_without_a_token_of_40"].items()})


if __name__ == "__main__":
    main()

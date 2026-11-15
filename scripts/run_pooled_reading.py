"""Phase 2b end to end (discussion draft, section 26.2): with the keyframes of several scenes pooled, does the function
index lead the model to the right part more often than semantic frame choice, and more so as the pool grows?

    python scripts/run_pooled_reading.py --visit 421254 --model 8b

The target scene's keyframes are pooled with those of K - 1 other scenes, K = 1, 4, 16. The other scenes are the first
draw that scripts/pooled_retrieval.py made for this target and K: its seed-0 generator is replayed over all its pool
sizes (1, 2, 4, 8, 16) and targets, so the draws are the same. Both arms read the same annotated part map (the P4 regions
of every pooled scene) with the same numbered boxes and the neutral prompt; only the three keyframes differ:

    semantic        the three pooled keyframes most like the task as whole keyframes; the annotated regions on them,
                    in memory order, at most 40, numbered (arm c4 of scripts/run_mem_controls.py)
    function index  the three best pooled keyframes by the phase-1 P4 score over the pooled memory; the best-scoring
                    regions there, at most 40, numbered (arm p4n)

The pooled memory holds its scenes in the order of their visit ids, not the target first, so that "memory order" (the
cap of 40 in the semantic arm, ties) says nothing about which scene is the target. A pick is right when the chosen box's
centre lies on a target part of the target scene, widened by 8 pixels; a box on another scene's keyframe is wrong.
Labels (ground truth, for scoring only) are made for the target scene alone; another scene's keyframe is loaded only
when it is shown.

Writes runs/mem_phase1/pooled_reading/<visit>.json after each K: per K, arm and task the reading's record, the draws used
and the memory's scenes. The model's answers are cached in runs/mem_phase1/pooled_reading/answers_<visit>.json, keyed by
what the model is shown (the same question in two arms or at two pool sizes is asked once; the model decodes greedily),
so a stopped run resumes where it stopped. --model 4b writes to runs/mem_phase1/pooled_reading_4b/ instead. Summarised
by scripts/aggregate_pooled_reading.py.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.frames import Frame
from affordance3d.mem import grounding as G
from affordance3d.mem import keyframes
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from affordance3d.mem import policies
from affordance3d.mem import search as S
from affordance3d.mem.encoders import VLMS, VisionLanguageModel
from affordance3d.mem.inputs import index_video
from affordance3d.mem.labels import BACKGROUND
from affordance3d.scenefun3d import index_sequence
from report import mem_steps_helpers as H


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _script("run_mem_controls")  # the reading step of arms p4n and c4
PR = _script("pooled_retrieval")  # the draws of phase 2b

POOL_SIZES = (1, 4, 16)
ARMS = ("semantic", "function index")
SEED = 0  # the generator of scripts/pooled_retrieval.py
KEY_ARM = "pooled"  # the answer cache key leaves the arm out: the same images, boxes and prompt get one answer
OUT_FOLDERS = {"8b": "pooled_reading", "4b": "pooled_reading_4b"}
VERSION = "pooled-reading-v1"  # change it when the choice of keyframes or regions changes


# ---------------------------------------------------------------- the draws of phase 2b

def retrieval_draws(n_scenes: int, seed: int = SEED) -> dict[int, list[list[list[int]]]]:
    """Every draw of scripts/pooled_retrieval.py, made as it makes them: one generator, the pool sizes in turn, the
    targets in scene order, DRAWS draws each (none for K = 1). Per size and target, the other scenes' indices per draw."""
    rng = np.random.default_rng(seed)
    draws: dict[int, list[list[list[int]]]] = {}
    for k in PR.POOL_SIZES:
        per_target = []
        for target in range(n_scenes):
            others = [i for i in range(n_scenes) if i != target]
            made = [[]] if k == 1 else [rng.choice(others, size=k - 1, replace=False) for _ in range(PR.DRAWS)]
            per_target.append([[int(i) for i in draw] for draw in made])
        draws[k] = per_target
    return draws


def first_draws(n_scenes: int, target: int, sizes: Sequence[int] = POOL_SIZES) -> dict[int, list[int]]:
    """The other scenes pooled with `target` at each size: the first draw phase 2b made (in its draw order)."""
    every = retrieval_draws(n_scenes)
    unknown = sorted(set(sizes) - set(every))
    if unknown:
        raise ValueError(f"pool sizes {unknown} were not drawn by scripts/pooled_retrieval.py")
    return {k: every[k][target][0] for k in sizes}


def memory_order(target: int, others: Sequence[int]) -> list[int]:
    """The scenes of one pooled memory in visit order (the scored scenes are sorted by visit): nothing marks the target."""
    members = sorted({target, *others})
    if len(members) != 1 + len(others):
        raise ValueError(f"a draw repeats a scene or holds the target: {target}, {list(others)}")
    return members


# ---------------------------------------------------------------- the pooled memory and the two arms

def pool(scenes: Mapping[int, dict], members: Sequence[int]) -> dict:
    """One memory from several scenes (diagnostics as saved_memory returns them): the keyframe vectors and region vectors
    stacked, the P4 regions as candidates on pooled keyframe indices, and for each pooled keyframe its scene and its own
    index there."""
    offsets = np.cumsum([0, *(len(scenes[m]["keyframes"]) for m in members)])
    candidates = [policies.Candidate(int(k) + int(offset), tuple(int(v) for v in box), (0, 0, 0, 0))
                  for m, offset in zip(members, offsets) for k, box in zip(scenes[m]["keyframe"], scenes[m]["boxes"])]
    return {"keyframes": np.concatenate([scenes[m]["keyframes"] for m in members]),
            "vectors": np.concatenate([scenes[m]["vectors"] for m in members]),
            "candidates": candidates,
            "scene": np.concatenate([np.full(len(scenes[m]["keyframes"]), m, dtype=np.int64) for m in members]),
            "local": np.concatenate([np.arange(len(scenes[m]["keyframes"]), dtype=np.int64) for m in members])}


Choice = tuple[list[int], list[policies.Candidate]]


def semantic_choice(query: np.ndarray, memory: dict) -> Choice:
    """c4 on the pooled memory: the three keyframes most like the task as whole keyframes, and the annotated regions on
    them in memory order (at most 40); no region is ranked."""
    shown = RUN.best_keyframes(query, memory["keyframes"])
    return shown, RUN.annotated_on(shown, memory["candidates"])


def function_choice(query: np.ndarray, memory: dict) -> Choice:
    """p4n on the pooled memory, as run_mem_controls.numbered_choice chooses: the three best keyframes by the P4 score,
    the best-scoring regions on them (at most 40), passed on in memory order."""
    candidates = memory["candidates"]
    scores = S.score_keyframes(query, memory["keyframes"], memory["vectors"], [c.keyframe for c in candidates],
                               [0] * len(candidates))
    top = [int(k) for k in scores.ranking()[:RUN.TOP_KEYFRAMES]]
    inside = sorted((i for i, c in enumerate(candidates) if c.keyframe in top), key=lambda i: -scores.token_scores[i])
    return top, [candidates[i] for i in sorted(inside[:RUN.MAX_MARKS])]


CHOICES: dict[str, Callable[[np.ndarray, dict], Choice]] = {"semantic": semantic_choice, "function index": function_choice}


# ---------------------------------------------------------------- frames, names and labels by pooled keyframe

class SceneFrames:
    """One scene's keyframes: named "<visit>/<time>", loaded when first shown and kept."""

    def __init__(self, visit: str, times: Sequence[str], load: Callable[[str], Frame]) -> None:
        self.visit, self.times, self._load, self._frames = visit, list(times), load, {}

    def name(self, k: int) -> str:
        return f"{self.visit}/{self.times[k]}"

    def frame(self, k: int) -> Frame:
        if k not in self._frames:
            self._frames[k] = self._load(self.times[k])
        return self._frames[k]


class ByIndex:
    """Indexable by pooled keyframe, each item made on demand (the reading step only indexes frames and labels)."""

    def __init__(self, item: Callable[[int], object]) -> None:
        self._item = item

    def __getitem__(self, k: int) -> object:
        return self._item(int(k))


class PooledView:
    """What run_mem_controls.numbered_reading reads, by pooled keyframe: frames, names and labels. Only the target scene
    has labels; another scene's keyframe gets an empty label map, so no box on it can be right."""

    def __init__(self, memory: dict, sources: Mapping[int, SceneFrames], target: int, labels: Sequence[np.ndarray]) -> None:
        self._scene, self._local, self._sources, self._target, self._labels = memory["scene"], memory["local"], sources, target, labels
        self.times = [sources[int(s)].name(int(k)) for s, k in zip(self._scene, self._local)]
        self.frames = ByIndex(lambda k: self._sources[int(self._scene[k])].frame(self.local(k)))
        self.labels = ByIndex(self._label)

    def local(self, k: int) -> int:
        return int(self._local[k])

    def on_target(self, k: int) -> bool:
        return int(self._scene[k]) == self._target

    def _label(self, k: int) -> np.ndarray:
        if self.on_target(k):
            return self._labels[self.local(k)]
        return np.full(self.frames[k].image_hw, BACKGROUND, dtype=np.int32)


def shown_facts(shown: list[int], kept: list[policies.Candidate], task: H.Task, view: PooledView) -> dict:
    """What a choice put in front of the model (descriptive): target keyframes among those shown, whether one of them shows
    a target part (the retrieval hit of section 26), whether one of the numbered boxes is right (the reading's ceiling)."""
    on_target = [k for k in shown if view.on_target(k)]
    return {"target_keyframes_shown": len(on_target),
            "top3_right": any(view.local(k) in task.keyframes for k in on_target),
            "a_box_right": any(G.hits_part(c.box, view.labels[c.keyframe], task.parts) for c in kept if view.on_target(c.keyframe))}


def read_task(reader: H.CachedReader, arm: str, task: H.Task, query: np.ndarray, memory: dict, view: PooledView) -> dict:
    """One arm on one task: choose keyframes and regions, let the model pick a number, score the pick on the target."""
    shown, kept = CHOICES[arm](query, memory)
    row = RUN.numbered_reading(reader, KEY_ARM, task.text, view.frames, view.times, shown, kept, view.labels, task.parts)
    chosen_scene = None if row["frame"] is None else row["frame"].split("/", 1)[0]
    return {**row, **shown_facts(shown, kept, task, view), "chosen_scene": chosen_scene}


# ---------------------------------------------------------------- loading

def scored_scenes(runs: Path) -> list[dict]:
    """The phase-1 results phase 2b drew from, in its order; checked against its summary when that is there."""
    results = A.scored(A.load_results(runs))
    file = runs / "pooled_retrieval.json"
    if file.exists():
        retrieval = json.loads(file.read_text())
        ours = (len(results), sum(len(result["tasks"]["texts"]) for result in results))
        if (retrieval["scenes"], retrieval["tasks"]) != ours:
            raise SystemExit(f"{runs} holds {ours[0]} scenes and {ours[1]} tasks; scripts/pooled_retrieval.py drew from "
                             f"{retrieval['scenes']} and {retrieval['tasks']}: the draws would differ")
    return results


def saved_memory(runs: Path, visit: str) -> dict:
    """Keyframe vectors, task queries and the P4 regions of one scene, as the phase-1 diagnostics saved them."""
    with np.load(runs / "diagnostics" / f"{visit}.npz") as saved:
        return {"keyframes": saved["keyframes"], "queries": saved["queries"], "vectors": saved["vectors/P4"],
                "keyframe": saved["keyframe/P4"], "boxes": saved["boxes/P4"]}


def light_scene(data: Path, visit: str, video: str) -> P.Scene:
    """What showing another scene's keyframes needs (its video and frame index), without its scan or annotations."""
    empty = np.zeros((0, 3))
    return P.Scene(vid=index_video(data, visit, video), gt_index=index_sequence(data, visit, video), scan_xyz=empty,
                   scan_rgb=empty, crop=np.zeros(0, dtype=bool), elements=(), motions={}, tasks=())


def scene_frames(scene: P.Scene, result: dict) -> SceneFrames:
    """A scene's keyframes as phase 1 chose them; only the colour image is loaded (the reading needs no depth)."""
    times = keyframes.scene_keyframes(scene, result["keyframes"], result["n_keyframes"])
    return SceneFrames(result["visit"], times, lambda t: RUN.rgb_frames(scene, [t])[0])


def target_scene(data: Path, result: dict) -> tuple[SceneFrames, list[np.ndarray], list[H.Task]]:
    """The target's keyframes, its label maps (ground truth, for scoring only) and its tasks, checked against phase 1."""
    scene = P.load_scene(data, result["visit"], result["video"])
    frames = scene_frames(scene, result)
    labels = P.label_maps(scene, P.gt_frames(scene, frames.times))
    tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
    if [task.text for task in tasks] != result["tasks"]["texts"]:
        raise SystemExit(f"{result['visit']}: the tasks differ from those of the phase-1 run")
    return frames, labels, tasks


@dataclass(frozen=True)
class Setup:
    visits: list[str]  # the scored scenes, in the order of the draws
    target: int
    draws: dict[int, list[int]]  # per K, the other scenes in draw order
    members: dict[int, list[int]]  # per K, the memory's scenes in memory order
    saved: dict[int, dict]
    sources: dict[int, SceneFrames]
    labels: list[np.ndarray]  # the target's, per keyframe
    tasks: list[H.Task]
    queries: np.ndarray
    video: str


def prepare(visit: str, runs: Path, data: Path) -> Setup:
    """Everything one target needs: its draws, the saved memories of every scene it is pooled with, and their frames."""
    results = scored_scenes(runs)
    visits = [result["visit"] for result in results]
    if visit not in visits:
        raise SystemExit(f"{visit} is not one of the {len(visits)} scored scenes in {runs}")
    target = visits.index(visit)
    draws = first_draws(len(visits), target)
    members = {k: memory_order(target, others) for k, others in draws.items()}
    needed = sorted({m for scenes in members.values() for m in scenes})
    saved = {m: saved_memory(runs, visits[m]) for m in needed}
    frames, labels, tasks = target_scene(data, results[target])
    sources = {m: scene_frames(light_scene(data, visits[m], results[m]["video"]), results[m]) for m in needed if m != target}
    sources[target] = frames
    for m in needed:
        if len(sources[m].times) != len(saved[m]["keyframes"]):
            raise SystemExit(f"{visits[m]}: {len(sources[m].times)} keyframes, but the diagnostics hold {len(saved[m]['keyframes'])}")
    if len(saved[target]["queries"]) != len(tasks):
        raise SystemExit(f"{visit}: {len(tasks)} tasks, but the diagnostics hold {len(saved[target]['queries'])} queries")
    return Setup(visits, target, draws, members, saved, sources, labels, tasks, saved[target]["queries"], results[target]["video"])


# ---------------------------------------------------------------- the run

def new_record(setup: Setup, model: str) -> dict:
    visits = setup.visits
    return {"visit": visits[setup.target], "video": setup.video, "model": model, "version": VERSION,
            "controls_version": RUN.CONTROLS_VERSION, "seed": SEED, "scenes": visits, "tasks": [task.text for task in setup.tasks],
            "draws": {str(k): [visits[i] for i in others] for k, others in setup.draws.items()},
            "memory": {str(k): [visits[i] for i in scenes] for k, scenes in setup.members.items()},
            "per_size": {}, "complete": False}


def write_record(file: Path, record: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    temporary.replace(file)


def read_size(reader: H.CachedReader, setup: Setup, k: int) -> dict[str, list[dict]]:
    """Both arms on every task of the target, with the memory of pool size `k`."""
    memory = pool(setup.saved, setup.members[k])
    view = PooledView(memory, setup.sources, setup.target, setup.labels)
    return {arm: [read_task(reader, arm, task, query, memory, view) for task, query in zip(setup.tasks, setup.queries)]
            for arm in ARMS}


def run_visit(visit: str, runs: Path, data: Path, out: Path, make_model: Callable[[], object], model: str) -> dict:
    """The whole check for one target scene; the record is written after each pool size."""
    started = time.time()
    setup = prepare(visit, runs, data)
    reader = H.CachedReader(out / f"answers_{visit}.json", make_model)
    record = new_record(setup, model)
    for k in POOL_SIZES:
        record["per_size"][str(k)] = read_size(reader, setup, k)
        write_record(out / f"{visit}.json", record)
        rates = ", ".join(f"{arm} {np.mean([row['hit'] for row in rows]):.0%}" for arm, rows in record["per_size"][str(k)].items())
        print(f"{visit} K = {k}: {rates} of {len(setup.tasks)} tasks right", flush=True)
    record = {**record, "complete": True, "total_s": round(time.time() - started, 1)}
    write_record(out / f"{visit}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--model", choices=sorted(VLMS), default="8b")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    parser.add_argument("--out", type=Path, help="folder of the records and answers (default: <runs>/pooled_reading for 8b)")
    args = parser.parse_args()
    model_id, revision = VLMS[args.model]
    out = args.out or args.runs / OUT_FOLDERS[args.model]
    run_visit(args.visit, args.runs, args.data, out, lambda: VisionLanguageModel(model_id=model_id, revision=revision),
              f"{model_id}@{revision}")


if __name__ == "__main__":
    main()

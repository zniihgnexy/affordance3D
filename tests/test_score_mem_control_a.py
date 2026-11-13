import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_mem_control_a.py"
spec = importlib.util.spec_from_file_location("score_mem_control_a", SCRIPT)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def _unit(rows: list[list[float]]) -> np.ndarray:
    rows = np.array(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_memories_holds_every_generic_region_and_every_grid_level_and_scores_alike() -> None:
    scene = {"keyframes": _unit([[1, 0.1], [0.2, 1], [0.9, 0.4]]), "queries": _unit([[0, 1]]), "truth": {"relevant": [[1]]},
             "vectors/P4": _unit([[0, 1]]), "keyframe/P4": np.array([1])}
    features = {"pool/vectors": _unit([[1, 0], [0.1, 1], [1, 0.2]]), "pool/keyframe": np.array([0, 1, 2]),
                "grid 1x1/vectors": _unit([[1, 0], [0, 1], [1, 1]]), "grid 1x1/keyframe": np.array([0, 1, 2]),
                "grid 2x1/vectors": _unit([[1, 0]] * 6), "grid 2x1/keyframe": np.repeat(np.arange(3), 2)}

    arms = score.memories(scene, features)

    assert set(arms) == {"P0", "P4", "a: every generic region", "a: every grid level"}
    assert len(arms["a: every grid level"][0]) == 9 and len(arms["P0"][0]) == 0
    for vectors, keyframe in arms.values():
        assert score.first_right(scene, vectors, keyframe).tolist() == [1.0]

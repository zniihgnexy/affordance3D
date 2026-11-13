import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from affordance3d.mem import phase1_stats as A
from affordance3d.mem import policies as W
from report import mem_phase1_helpers as R
from tests.conftest import TEST_HW, frame_from_depth

HIT = {"P0": [0, 0, 1, 0], "P2": [0, 1, 1, 0], "P3": [1, 1, 0, 0], "P4": [1, 1, 1, 0]}


def _run(policy: str, budget: float, hit1: list[int], seed: int = 0, tokens: int = 10, on_part: list[int] | None = None) -> dict:
    return {"policy": policy, "budget": budget, "seed": seed, "tokens": tokens, "hit1": hit1, "hit3": hit1,
            "on_part": hit1 if on_part is None else on_part, "area_share": 0.004}


def _scene(visit: str, flip: bool = False) -> dict:
    """One scene's result with four tasks; `flip` swaps the outcomes of P3 and P4."""
    hit = dict(HIT, P3=HIT["P4"], P4=HIT["P3"]) if flip else HIT
    search = [_run("P0 whole keyframes", 0.0, hit["P0"], tokens=0, on_part=[]),
              _run("P1 uniform grid 2x1", 0.8, hit["P2"], tokens=8),
              *[_run("P2 random places", budget, hit["P2"]) for budget in (0.5, 1.0, 2.0)],
              *[_run("P3 generic regions", budget, hit["P3"]) for budget in (0.5, 1.0, 2.0)],
              _run("P4 affordance regions", 0.5, hit["P2"], tokens=5), _run("P4 affordance regions", 1.0, hit["P4"])]
    fused = {"0.01": {"contact_cm": [1.0, 5.0], "points": 100, "bytes": 3_000_000, "build_s": 2.0, "read_ms_per_part": 4.0}}
    return {"visit": visit, "n_keyframes": 40, "tasks": {"all": 5, "answerable": 4}, "memory": {"tokens": 10},
            "generic_regions": 300, "search": search,
            "geometry": {"read_ms_per_part": 1.5, "fused": fused}}


def _summary(scenes: int = 2) -> dict:
    row = {"rate": 0.5, "low": 0.4, "high": 0.6}
    first = {"P0": row, "P2": row, "P3": row, "P4": {"rate": 0.75, "low": 0.6, "high": 0.9},
             **{f"P4 minus {name}": {"rate": 0.25, "low": 0.05, "high": 0.4} for name in ("P0", "P2", "P3")}}
    geometry = {"single_frame": {"parts": 4, "read": 4, "median_cm": 2.6, "within_3cm": 0.5, "within_3cm_of_all": 0.5, "stand_free": 0.9},
                "l2_readout": {"parts": 5, "read": 4, "median_cm": 3.1, "within_3cm": 0.5, "within_3cm_of_all": 0.4, "stand_free": 0.8},
                "fused 0.01 m": {"parts": 4, "read": 3, "median_cm": 1.8, "within_3cm": 2 / 3, "within_3cm_of_all": 0.5,
                                 "bytes_per_scene": 3_000_000.0, "build_s": 2.0}}
    reading = {"P4 affordance regions": {**row, "seconds_per_query": 0.9}, "P3 generic regions": {**row, "seconds_per_query": 1.1},
               "P4 minus P3": {"rate": 0.0, "low": -0.1, "high": 0.1}}
    return {"scenes": scenes, "visits": [f"42{index}" for index in range(scenes)], "scenes_run": scenes,
            "scenes_without_tasks": [], "missing": {}, "tasks": 8, "tasks_all": 10, "parts": 40,
            "parts_seen": {"steady": 30, "uniform": 28}, "first_keyframe_right": first, "reading": reading, "geometry": geometry,
            "storage_bytes_per_scene": {"keyframe_images": 14e6, "depth_maps": 12e6, "affordance_maps": 1e5, "token_features": 2e5,
                                        "tokens": 96.0}, "write_s_per_scene": 30.0}


def _write_runs(folder: Path, scenes: int, summary_scenes: int | None) -> None:
    for index in range(scenes):
        (folder / f"42{index}.json").write_text(json.dumps(_scene(f"42{index}")))
    (folder / "vlm_answers_420.json").write_text("{}")
    if summary_scenes is not None:
        (folder / "summary.json").write_text(json.dumps(_summary(summary_scenes)))


def test_load_runs_returns_the_scored_scenes_and_their_summary(tmp_path: Path) -> None:
    _write_runs(tmp_path, scenes=2, summary_scenes=2)

    results, summary = R.load_runs(tmp_path)

    assert [result["visit"] for result in results] == ["420", "421"] and summary["scenes"] == 2


def test_load_runs_refuses_a_missing_or_outdated_summary(tmp_path: Path) -> None:
    _write_runs(tmp_path, scenes=2, summary_scenes=None)
    with pytest.raises(FileNotFoundError, match="aggregate_mem_phase1"):
        R.load_runs(tmp_path)

    (tmp_path / "summary.json").write_text(json.dumps(_summary(scenes=1)))
    with pytest.raises(ValueError, match="again"):
        R.load_runs(tmp_path)

    other_scenes = {**_summary(scenes=2), "visits": ["420", "429"]}  # as many scenes, but not the ones that are there
    (tmp_path / "summary.json").write_text(json.dumps(other_scenes))
    with pytest.raises(ValueError, match="again"):
        R.load_runs(tmp_path)


def test_scene_differences_subtract_the_other_policy_scene_by_scene() -> None:
    results = [_scene("1"), _scene("2", flip=True)]

    against_generic = R.scene_differences(results, A.P3)
    against_whole = R.scene_differences(results, A.P0)

    assert against_generic.tolist() == pytest.approx([0.25, -0.25])  # P4 3 of 4 and P3 2 of 4, then the reverse
    assert against_whole.tolist() == pytest.approx([0.5, 0.25])  # whole keyframes have no budget to match


def test_scene_differences_are_nan_where_a_policy_was_not_run() -> None:
    without_generic = _scene("3")
    without_generic = {**without_generic, "search": [run for run in without_generic["search"] if not run["policy"].startswith("P3")]}

    differences = R.scene_differences([_scene("1"), without_generic], A.P3)

    assert differences[0] == pytest.approx(0.25) and np.isnan(differences[1])


def test_typical_scene_is_the_one_with_the_median_difference() -> None:
    level = _scene("3")
    level = {**level, "search": [{**run, "hit1": HIT["P4"]} if run["policy"].startswith("P3") else run for run in level["search"]]}
    results = [_scene("1"), _scene("2", flip=True), level]  # differences +0.25, -0.25, 0

    assert R.typical_scene(results, min_tasks=1) == "3"
    assert R.typical_scene(results, min_tasks=10) == "3"  # no scene has ten tasks: all scenes are eligible
    big = {**_scene("4"), "tasks": {"all": 12, "answerable": 12}}
    assert R.typical_scene([*results, big], min_tasks=10) == "4"  # the only scene with enough tasks


def test_token_table_reports_the_best_token_rate_and_region_size_at_budget_one() -> None:
    table = R.token_table([_scene("1"), _scene("2")])

    assert list(table.index) == ["P4 affordance regions", "P3 generic regions", "P2 random places"]
    assert table.loc["P4 affordance regions", "best token is on the target part (%)"] == 75.0
    assert table.loc["P3 generic regions", "tokens per scene"] == 10.0
    assert table.loc["P2 random places", "median region area (% of the image)"] == pytest.approx(0.4)


def test_geometry_table_gives_storage_and_build_time_only_to_the_fused_cloud() -> None:
    table = R.geometry_table([_scene("1"), _scene("2")], _summary())

    fused = table.loc["fused cloud, 1 cm voxels"]
    single = table.loc["one keyframe, read now (true part ids)"]
    assert fused["extra storage per scene (MB)"] == 3.0 and fused["build time per scene (s)"] == 2.0
    assert fused["read time per part (ms)"] == 4.0 and fused["within 3 cm (% of read)"] == pytest.approx(66.7)
    assert single["extra storage per scene (MB)"] == 0.0 and np.isnan(single["read time per part (ms)"])
    assert table.loc["one keyframe, read now (parts linked by the memory)", "read time per part (ms)"] == 1.5


def test_tables_of_data_rates_and_costs_keep_the_summary_numbers() -> None:
    summary = _summary()

    data = R.data_table([_scene("1"), _scene("2")], summary)["value"]
    rates = R.rates_table(summary["first_keyframe_right"])
    cost = R.cost_table(summary)["value"]

    assert data["tasks: all / answerable (the part is visible in a keyframe)"] == "10 / 8"
    assert data["of these, visible in the steady keyframes / in evenly spaced keyframes"] == "30 / 28"
    assert rates.loc["P4 minus P3"].tolist() == [25.0, 5.0, 40.0]
    assert cost["token features (MB per scene)"] == 0.2
    assert cost["model reading, P3 generic regions (s per query, median)"] == 1.1


def test_turn_rate_table_compares_the_two_keyframe_choices_over_all_scenes(tmp_path: Path) -> None:
    saved = {"steady": ["1.0", "3.0"], "uniform": ["2.0", "4.0"], "turn_rate_deg_s": {"1.0": 2.0, "2.0": 30.0, "3.0": 4.0, "4.0": 50.0}}
    results = [{"visit": visit, "video": f"v{visit}"} for visit in ("420", "421")]
    for result in results:
        folder = tmp_path / result["visit"] / result["video"]
        folder.mkdir(parents=True)
        (folder / "keyframes.json").write_text(json.dumps(saved))

    table = R.turn_rate_table(tmp_path, results)

    steady, uniform = table.loc["steady keyframes (used)"], table.loc["evenly spaced keyframes (for comparison)"]
    assert steady["keyframes"] == 4 and steady["median turn rate (deg/s)"] == 3.0 and steady["fastest (deg/s)"] == 4.0
    assert uniform["median turn rate (deg/s)"] == 40.0 and uniform["fastest (deg/s)"] == 50.0


def _actions_with_two_regions() -> np.ndarray:
    actions = np.full(TEST_HW, -1, dtype=np.int8)
    actions[4:10, 6:14] = 0
    actions[30:36, 40:44] = 2
    return actions


def test_token_sets_draw_what_the_runner_draws_for_the_seed() -> None:
    frames = [frame_from_depth(np.full(TEST_HW, 2.0), index=i) for i in range(3)]
    action_maps = [_actions_with_two_regions(), np.full(TEST_HW, -1, dtype=np.int8), _actions_with_two_regions()]
    pool = [W.Candidate(k, (x, 2, x + 6, 8), (0, 0, 20, 20)) for k in range(3) for x in range(0, 50, 5)]

    sets = R.token_sets(frames, action_maps, pool, seed=1)

    afford = sets[A.P4]
    rng = np.random.default_rng(R.RUNNER_SEED + 1)  # the runner: random places first, generic regions next, one generator
    assert [(c.keyframe, c.box) for c in afford] == [(0, (6, 4, 14, 10)), (0, (40, 30, 44, 36)), (2, (6, 4, 14, 10)), (2, (40, 30, 44, 36))]
    assert sets[A.P2] == W.random_candidates(afford, 4, 3, TEST_HW, rng)
    assert sets[A.P3] == W.size_matched(pool, afford, 4, rng) and len(sets[A.P3]) == 4
    assert len(sets[A.P1]) == 3 * 1  # 4 tokens over 3 keyframes: the 1x1 grid is the closest


def test_first_pick_takes_the_keyframe_of_the_best_token_and_falls_back_to_whole_keyframes() -> None:
    keyframe_vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    candidates = [W.Candidate(0, (0, 0, 4, 4), (0, 0, 8, 8)), W.Candidate(2, (5, 5, 9, 9), (0, 0, 8, 8)),
                  W.Candidate(2, (1, 1, 3, 3), (0, 0, 8, 8))]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    query = np.array([0.0, 1.0])

    with_tokens = R.first_pick(query, keyframe_vectors, candidates, vectors, relevant={2})
    without = R.first_pick(np.array([1.0, 0.0]), keyframe_vectors, [], np.zeros((0, 2)), relevant={2})

    assert with_tokens == R.Pick(keyframe=2, box=(5, 5, 9, 9), right=True)
    assert without == R.Pick(keyframe=0, box=None, right=False)


def test_choose_examples_prefers_tasks_that_separate_the_policies() -> None:
    def picks(p2: bool, p3: bool, p4: bool) -> dict[str, R.Pick]:
        return {A.P2: R.Pick(0, None, p2), A.P3: R.Pick(0, None, p3), A.P4: R.Pick(0, None, p4)}

    tasks = [picks(True, True, True), picks(False, True, True), picks(True, False, True), picks(True, True, False)]

    assert R.choose_examples(tasks, n=3) == [2, 1, 3]  # right where generic regions fail, where random places fail, then a miss
    assert R.choose_examples([picks(True, True, True)] * 2, n=3) == [0]  # nothing separates them: one plain example


def test_target_pixels_are_the_centres_of_the_target_parts_per_keyframe() -> None:
    labels = [np.full(TEST_HW, -1), np.full(TEST_HW, -1)]
    labels[1][10:14, 20:26] = 3
    labels[1][0, 0] = 5  # one pixel is not a visible part

    assert R.target_pixels(labels, parts=(3, 5)) == {1: [(22, 11)]}


def _diagnostics() -> dict:
    row = {"rate": 0.5, "low": 0.4, "high": 0.6}
    step = {"rate": 0.1, "low": 0.02, "high": 0.2}
    setting = {"P4": 0.66, "P3": 0.47, "P2": 0.39, "P1": 0.5, "shortlist": 0.55}
    return {"first_keyframe_right": {arm[0]: row for arm in R.ATTRIBUTION_ARMS}, "differences": {s[0]: step for s in R.ATTRIBUTION_STEPS},
            "sweep": {"weight 1, empty -3": setting, "weight 1, empty 0": {**setting, "P4": 0.631}}}


def test_attribution_figure_and_sweep_table_show_the_follow_up_check(tmp_path: Path) -> None:
    (tmp_path / "diagnostics.json").write_text(json.dumps(_diagnostics()))

    diagnostics = R.load_diagnostics(tmp_path)
    figure, table = R.attribution_figure(diagnostics), R.sweep_table(diagnostics)

    bars = figure.axes[0].patches
    assert len(figure.axes) == 2 and len(bars) == 5 and [bool(bar.get_hatch()) for bar in bars] == [False, False, True, True, False]
    assert table.loc["weight 1, empty 0", "P4 affordance regions"] == 63.1 and table.loc["weight 1, empty -3", "one bit per keyframe"] == 55.0
    plt.close(figure)
    with pytest.raises(FileNotFoundError, match="diagnose_mem_phase1"):
        R.load_diagnostics(tmp_path / "nowhere")


def test_figures_refuse_a_summary_without_anything_to_compare() -> None:
    only_affordance = {**_summary(), "first_keyframe_right": {"P4": {"rate": 0.75, "low": 0.6, "high": 0.9}}}
    scene = _scene("1")
    scene = {**scene, "search": [run for run in scene["search"] if run["policy"].startswith("P4")]}

    with pytest.raises(ValueError, match="other policy"):
        R.headline_figure(only_affordance)
    with pytest.raises(ValueError, match="another policy"):
        R.per_scene_figure([scene])


def test_figures_are_built_from_results_and_summary() -> None:
    results, summary = [_scene("1"), _scene("2", flip=True)], _summary()
    frames = [frame_from_depth(np.full(TEST_HW, 2.0), index=i) for i in range(2)]
    sets = {A.P1: W.grid_candidates(2, TEST_HW, (1, 1)), A.P4: [W.Candidate(1, (2, 2, 9, 9), (0, 0, 12, 12))]}
    pick = {A.P0: R.Pick(0, None, False), A.P4: R.Pick(1, (2, 2, 9, 9), True)}

    with_bit = R.budget_figure(results, bit={"rate": 0.55, "low": 0.5, "high": 0.6})
    assert len(with_bit.axes[0].get_lines()) == len(R.budget_figure(results).axes[0].get_lines()) + 1  # one more line: the bit arm
    figures = [R.headline_figure(summary), R.reading_figure(summary), R.budget_figure(results), R.per_scene_figure(results),
               R.policies_figure(frames, sets, keyframes_shown=[0, 1]),
               R.picks_figure(frames, ["open the drawer"], [pick], [{1: [(5, 5)]}])]

    assert [len(figure.axes) for figure in figures] == [2, 2, 1, 3, 4, 2]
    assert len(figures[3].axes[0].patches) == 2  # one bar per scene
    for figure in figures:
        plt.close(figure)

# Experiments: index and commands

Every experiment of the memory direction, with the command that reproduces it, what it reads and writes, and the section
of `docs/mem/mem分支_问题定义与讨论稿_2026-11-10.md` that fixed its plan before it ran. Run every command from the
repository root on the GPU server. GPU work goes only through the shared queue: set `AFFORDANCE3D_QUEUE` to the queue
folder (the one that holds `pending/`). Results live under `runs/` (not versioned). Data: `scripts/remote/fetch_data.sh`;
models: `scripts/remote/fetch_models.py`.

New experiments are driven by a config file under `configs/` and one entrypoint. `affordance3d/experiment.py` gives each
run its own folder with the resolved config and the commit; a changed setting needs a new config name. The older
experiments below predate this and are driven by command-line options; they are converted when they are next changed.

## Current

| Experiment | Section | Command | Writes |
|---|---|---|---|
| Public evaluation on SceneFun3D split0 (Fun3DU protocol; paired comparison with UniFunc3D 8B) | 27 | `python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage queue`, then `--stage lift`, `--stage score [--official]`; `--stage status` shows progress | `runs/public_eval/<name>/` |
| S0: per-part affordance vectors by rules on one scene, scored by the SceneFun3D Task 3 paper definition (CPU, oracle masks and labels; runs on the Mac or the server, with the same results) | proposal 2026-11-18, section 7 (`docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md`) | `python scripts/run_part_vectors.py --config configs/part_vectors/s0_421069.yaml` | `runs/part_vectors/<name>/` |
| S0b: the S0 test unchanged on three validation scenes chosen in advance for their many doors, pooled; plus a post-hoc door-hinge diagnostic (CPU, oracle masks and labels; runs on the Mac or the server, with the same results; scenes fetched with `scripts/download_scenefun3d_keyframes.py`) | proposal 2026-11-18, section 8 | `python scripts/run_part_vectors.py --config configs/part_vectors/s0b_val3.yaml`; `python scripts/diagnose_door_hinges.py --config configs/part_vectors/s0b_val3.yaml`; label-vs-motion counts: `python scripts/count_label_motions.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt` | `runs/part_vectors/<name>/` (per scene in `<visit>/`, pooled `summary.json`, `door_hinges.json`) |
| S0c: a door rule fixed before the run (axis = gravity, origin = far end of the surface's level line; rule sets `gravity`, `oracle_gravity`) on three further validation scenes chosen in advance, with predictions P1–P4 in the config (CPU, oracle masks and labels; runs on the server) | proposal 2026-11-18, section 9 | `python scripts/run_part_vectors.py --config configs/part_vectors/s0c_gravity_val3.yaml`; `python scripts/diagnose_door_hinges.py --config configs/part_vectors/s0c_gravity_val3.yaml` | `runs/part_vectors/<name>/` (per scene in `<visit>/`, pooled `summary.json`, `door_hinges.json`) |
| Part shapes per label: how many annotated parts are elongated, so that a grasp axis means something (descriptive; annotation and laser scan only, CPU) | proposal 2026-11-18, section 10 | `python scripts/count_part_shapes.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt` | prints only |
| Scoring check against UniFunc3D's released predictions (must equal their paper) | 27.1 | `python -I scripts/scenefun3d_eval.py fun3du --pred_dir third_party/unifunc3d/predictions/8b_val --reference third_party/unifunc3d/reference_results/8b_val.json --fingerprints third_party/unifunc3d/datasets/val.json --out runs/scenefun3d_eval/unifunc3d_8b_val.json` | that file |
| Official SceneFun3D scorer, ground truth and sanity check | 27.1 | `python -m eval.affordance_grounding.prepare_gt_val_data ...` (in `third_party/scenefun3d`, see `scripts/scenefun3d_eval.py`); `python -I scripts/scenefun3d_eval.py oracle --mode elements --out <dir>` then `score --pred_dir <dir>` | `runs/scenefun3d_eval/` |

## Earlier (results recorded in the discussion draft)

| Experiment | Section | Command | Writes |
|---|---|---|---|
| Phase 1: write policies P0 to P4 on 30 scenes | 21, 22 | `bash scripts/remote/queue_phase1.sh <scenes file>` (runs `scripts/run_mem_phase1.py --visit <v> --vlm`); `python scripts/aggregate_mem_phase1.py` | `runs/mem_phase1/<visit>.json`, `summary.json` |
| Phase-1 diagnostics (where the gain comes from) | 23 | `python scripts/diagnose_mem_phase1.py --features`, then `python scripts/diagnose_mem_phase1.py` | `runs/mem_phase1/diagnostics/`, `diagnostics.json` |
| Controls without the memory (c1 to c4, b, d, c+), 4B and 8B | 24.1 to 24.8 | `bash scripts/remote/queue_controls.sh <scenes file> [jobs] [priority] [tag] [arms] [model]` (runs `scripts/run_mem_controls.py`); `python scripts/aggregate_mem_controls.py [--model 8b]` | `runs/mem_phase1/controls[_8b]/`, `controls[_8b]_summary.json` |
| Control a (generic regions, scored on the CPU) | 24.7 | `python scripts/score_mem_control_a.py` | `runs/mem_phase1/control_a.json` |
| Off-the-shelf tool maps (SAM 3) and their reading arms | 24.9 | `bash scripts/remote/queue_tool_maps.sh <scenes file> sam3` and `... sam3_tasks` (runs `scripts/run_tool_maps.py`); tool arms via `queue_controls.sh ... p4t_sam3,c4t_sam3,p4tp_sam3,sam3q 8b`; `python scripts/score_tool_maps.py`; `python scripts/aggregate_mem_controls.py --model 8b` | `runs/mem_phase1/tool_maps/`, `tool_maps_summary.json`, `controls_8b_summary.json` |
| Retrieval by action type (CPU) | 24.10 | `python scripts/score_action_key.py` | `runs/mem_phase1/action_key.json` |
| Frame ceiling (CPU) | 24 | `python scripts/count_frame_ceiling.py` | `runs/mem_phase1/frame_ceiling.json` |
| B-0: events from HD-EPIC annotations (B stopped, round 10) | 25 | `python scripts/hdepic_b0_events.py` | `runs/hdepic/b0_events.json` |
| C-0: actionable records (CPU) | 25 | `python scripts/build_action_records.py` | `runs/mem_c0/` |
| C-1: how to act on the part, 4B then 8B (full run parked, 2026-11-16) | 25.5 | `python scripts/run_mem_c1.py --visit <v> [--model 8b]` (queue jobs in `parked/`); `python scripts/aggregate_mem_c1.py [--model 8b]` | `runs/mem_c1/` |
| 2b: several scenes pooled in one memory (shelved, 2026-11-16) | 26 | `bash scripts/remote/queue_pooled_reading.sh <scenes file> ... 8b` (runs `scripts/run_pooled_reading.py`); `python scripts/aggregate_pooled_reading.py` | `runs/mem_phase1/pooled_reading/` |

## Notebooks

Builders in `notebooks/builders/` write each notebook from scratch (see its README); long builders keep their cells in
`<builder>_parts/part_NN.py`. `bash scripts/run_notebooks.sh` executes every committed notebook into
`runs/notebook_check/`.

# Notebook builders

**One notebook per week (London time, Monday to Sunday).** Since 2026-11-20 the repository keeps exactly one notebook per
week under `report/`:

| week | notebook | how it is made |
|---|---|---|
| 1: 9 to 15 November | `report/week1_2026-11-09_to_11-15.ipynb` | `python scripts/merge_notebooks.py --config configs/report/week1.yaml`: the ten notebooks of that week, merged with their saved outputs (nothing re-run), read from commit `3f2baae` |
| 2: 16 to 22 November | `report/week2_2026-11-16_to_11-22.ipynb` | `python notebooks/builders/build_report_2026_11_20_week2.py report/week2_2026-11-16_to_11-22.ipynb`, then `jupyter nbconvert --to notebook --execute --inplace` on it |

A week's content goes into the notebook of the week it covers (the week-1 summary was generated on Monday 16 November and
opens the week-1 notebook). Notes from the project lead go into the "✏️ 你的批注" cell of the current week's notebook;
before rebuilding a week's notebook, carry any text written there into its builder first.

## The builders

Each script writes one notebook from scratch (cells only, no outputs); run it, then execute the notebook:

```bash
python notebooks/builders/build_report_2026_11_20_week2.py report/week2_2026-11-16_to_11-22.ipynb
jupyter nbconvert --to notebook --execute --inplace report/week2_2026-11-16_to_11-22.ipynb
```

Long builders keep their cells in `<builder>_parts/part_NN.py` (each under 500 lines); the builder runs the parts in order, in its own namespace, exactly as when they were one file.

The week-1 builders still work: each writes its part as a separate notebook at the path you give it (the table shows the
path it had, which is also the path the merge config reads from commit `3f2baae`). To change a week-1 part, rebuild and
execute it at that path from its own folder (`bash scripts/run_notebooks.sh <path>` runs it there), commit it, point
`source_commit` in `configs/report/week1.yaml` at that commit, merge again and remove the separate file.

| builder | week | part (original path) |
|---|---|---|
| `build_report_2026_11_16_weekly.py` | 1 | 1. weekly summary (`report/2026-11-16_weekly_summary.ipynb`) |
| `build_report_2026_11_09.py` | 1 | 2. datasets report (`report/2026-11-09_datasets_report.ipynb`) |
| `build_01_step_by_step_demo.py` | 1 | 3. E0/E1 step-by-step demo (`notebooks/01_step_by_step_demo.ipynb`) |
| `build_02_scenefun3d_e0.py` | 1 | 4. E0 on SceneFun3D (`notebooks/02_scenefun3d_e0.ipynb`) |
| `build_mem_demo.py` | 1 | 5. mem pipeline demo (`report/2026-11-10_mem_pipeline_demo.ipynb`) |
| `build_mem_s1_01.py` | 1 | 6. S1-01 (`notebooks/mem/S1_01_video_inputs_to_actionable_affordance.ipynb`) |
| `build_mem_s1_02.py` | 1 | 7. S1-02 (`notebooks/mem/S1_02_humanoid_approach.ipynb`) |
| none (written by hand) | 1 | 8. pipeline working document (`notebooks/mem/pipeline.ipynb`) |
| `build_mem_pipeline_examples.py` | 1 | 9. pipeline examples (`notebooks/mem/pipeline_examples.ipynb`) |
| `build_mem_phase1_results.py` | 1 | 10. phase-1 results (`notebooks/mem/phase1_results.ipynb`) |
| `build_report_2026_11_20_week2.py` | 2 | the whole week-2 notebook |

Week-1 parts and what they need when re-run:
- The pipeline working document (part 8) was text only and edited by hand; it has no builder. Its examples (part 9) need the two frozen models fetched by `scripts/remote/fetch_models.py`.
- The phase-1 results (part 10) show results computed elsewhere: the per-scene files in `runs/mem_phase1/` come from GPU queue jobs (`scripts/remote/queue_phase1.sh`) and are summarised by `scripts/aggregate_mem_phase1.py`; the notebook itself runs on the CPU in a few minutes.
- The weekly summary (part 1) only reads result files under `runs/` (`runs/mem_phase1/`, `runs/hdepic/`, `runs/mem_c0/`, `runs/mem_c1/preview/`); it runs on the CPU in under a minute and checks every number its text quotes.

The week-2 notebook (concise English report with charts since 2026-11-23; chart code in `report/week2_report.py`) reads the part-vector runs in `runs/part_vectors/` (S0, S0b, S0c with their door diagnoses, and the demo run `demo_421069_gravity`), the public evaluation in `runs/public_eval/public_eval_split0_8b/`, and the two videos `report/assets/week2_*.mp4` made by `scripts/make_week2_media.py`; it runs on the CPU and checks the numbers its text quotes.

`bash scripts/run_notebooks.sh` executes the current week's notebook into `runs/notebook_check/` without touching the committed copy; pass paths to run others.

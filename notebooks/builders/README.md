# Notebook builders

Each script writes one notebook from scratch (cells only, no outputs); run it, then execute the notebook:

```bash
python notebooks/builders/build_mem_demo.py report/2026-11-10_mem_pipeline_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace report/2026-11-10_mem_pipeline_demo.ipynb
```

| builder | notebook |
|---|---|
| `build_01_step_by_step_demo.py` | `notebooks/01_step_by_step_demo.ipynb` |
| `build_02_scenefun3d_e0.py` | `notebooks/02_scenefun3d_e0.ipynb` |
| `build_report_2026_11_09.py` | `report/2026-11-09_datasets_report.ipynb` |
| `build_mem_s1_01.py` | `notebooks/mem/S1_01_video_inputs_to_actionable_affordance.ipynb` |
| `build_mem_s1_02.py` | `notebooks/mem/S1_02_humanoid_approach.ipynb` |
| `build_mem_demo.py` | `report/2026-11-10_mem_pipeline_demo.ipynb` |
| `build_mem_pipeline_examples.py` | `notebooks/mem/pipeline_examples.ipynb` |

`notebooks/mem/pipeline.ipynb` has no builder: it is a text-only working document that is edited by hand (one cell per pipeline block plus a notes cell for the project lead), so never regenerate it. Its examples live in `notebooks/mem/pipeline_examples.ipynb`, which does have a builder and needs the two frozen models fetched by `scripts/remote/fetch_models.py`.

`bash scripts/run_notebooks.sh` executes every committed notebook into `runs/notebook_check/` without touching the committed copies; use it to confirm a machine is set up.

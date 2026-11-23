# Cells of build_report_2026_11_20_week2.py, part 6 (appendix: commands and the number checks).

md(r"""
---
# Appendix

## A. Commands (repository root, on the server; GPU steps only through the queue)

| What | Command |
| --- | --- |
| Part vectors (S0, S0b, S0c, demo, T3) | `python scripts/run_part_vectors.py --config configs/part_vectors/<s0_421069 / s0b_val3 / s0c_gravity_val3 / demo_421069_gravity / t3_val30>.yaml` |
| Public evaluation, Task 2 | `python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage <queue / lift / score --official>` |
| Stage 0: Task 2 losses, OpenEQA | `python scripts/diagnose_task2_losses.py --config configs/stage0/task2_losses.yaml`; `python scripts/openeqa_fit.py --config configs/stage0/openeqa_fit.yaml --stage <count / correct>` |
| Overwrite simulation | `python scripts/simulate_overwrite.py --config configs/stage0/overwrite_sim.yaml` |
| Label vs motion type | `python scripts/count_label_motions.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt` |
| Gaussian trial (GPU queue) | `bash scripts/remote/setup_gaussians.sh` once; `bash scripts/remote/queue_fit_gaussians.sh configs/gaussians/fit_trial.yaml 100 11000 3` |
| Stage 2: second-visit videos | `python scripts/download_revisit_videos.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt` |
| Stage 2: SAM 3, Gaussian maps (GPU queue) | `bash scripts/remote/queue_tool_maps_revisit.sh configs/stage2/revisit_videos.txt 2`; `bash scripts/remote/queue_fit_gaussians_batch.sh configs/gaussians/fit_val82.yaml 6`, then `python scripts/fit_gaussians_batch.py --config configs/gaussians/fit_val82.yaml --stage summarise` |
| Videos; interactive 3D file | `python scripts/make_week2_media.py --config configs/report/week2_media.yaml`; `python scripts/make_week2_rrd.py --config configs/report/week2_rrd.yaml` |
| This notebook | `python notebooks/builders/build_report_2026_11_20_week2.py report/week2_2026-11-16_to_11-22.ipynb`, then `jupyter nbconvert --to notebook --execute --inplace` |

**Documents:** the proposal and every discussion of the week (Chinese): `docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md`;
understanding gaps: `docs/mem/理解差异记录.md`; experiment index: `EXPERIMENTS.md`. The earlier, longer Chinese version of
this notebook is in git history (commit b5b1267).

## B. Number checks

Each number quoted in the text was recomputed from the result files in the cell that shows it; a mismatch stops the
notebook.
""")

code(r"""
print(f"{len(CHECKS)} numbers checked, all match the result files:")
print("; ".join(CHECKS))
""")

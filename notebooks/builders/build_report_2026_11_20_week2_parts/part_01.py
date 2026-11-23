# Cells of build_report_2026_11_20_week2.py, part 1 (title, summary, setup, timeline, decision log): run in order by that
# builder, with md() and code() in scope.

md(r"""
# Week 2 · 16–22 November 2026 · part-level functional memory

One notebook per London week. Every number below is read from result files under `runs/` and re-checked in the appendix.
Long discussions (in Chinese) stay in the [proposal document](../docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md)
(sections 9–17); week 1 is in [the week-1 notebook](week1_2026-11-09_to_11-15.ipynb).

## Summary
- **The memory now stores an interpretable 3D affordance vector per part**: contact point, normal, motion type, direction or
  axis, axis origin and grasp axis, in world coordinates a robot can take with one rigid transform.
  Taking door axes from gravity raised the axis accuracy of doors from 68% to 84% (S0c, type given).
- **Public SceneFun3D Task 2 (find the part):** the memory with SAM 3 part maps scores AP25 24.9, against 44.0 for
  UniFunc3D 8B. The same memory with annotated part maps scores 43.1. It beats a VLM reading the raw video by +5.6
  (+10.3 on small parts) while reading 3 frames instead of 40.
- **Stage 0 (five CPU-only tests):**
  - Task 3 (how the part moves) is the most likely headline: given the part, the vectors are above the only published
    reference, and the motion type is the largest error, which is why each part will store what it is mounted on.
  - OpenEQA is auxiliary only: about 15% of its relevant questions are about parts.
- **The memory form was redesigned with you:** one 3D Gaussian map per room, with training-free features, the frames that
  saw each part and when it was last confirmed, plus the interpretable part vectors. A new video overwrites only where it
  sees clearly and the geometry changed; letting the newest view decide would erase 10–17% of found parts even in a
  scene that did not change.
- **Framework v4 frozen (20 Nov).** A trial Gaussian fit of one validation video took 14.7 min, with 25.4 dB held-out
  PSNR and 1.6 cm median depth error.
- **Stage 2 started:**
  - SAM 3 part maps of the 52 second-visit videos are done;
  - Gaussian maps of the 82 validation videos are fitting (section 5).
- **Demo media redrawn as game-style interaction hints**, plus an interactive 3D file you can turn yourself (section 1).
""")

code(r"""
# setup: paths, the demo scene (421069) with its keyframes and vectors, and the number checks of the appendix
import contextlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "affordance3d").is_dir())  # the repository (run from inside it)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the notebook)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, Video, display

import count_label_motions as CLM
import run_part_vectors as RPV
from affordance3d.experiment import load_config
from report import week2_helpers as H
from report import week2_report as WR

get_ipython().run_line_magic("matplotlib", "inline")  # run_part_vectors switches matplotlib to Agg when imported

pd.set_option("display.max_colwidth", 80)
FIG = ROOT / "docs" / "mem" / "figures"
ASSETS = ROOT / "report" / "assets"
DEMO = load_config(ROOT / "configs" / "part_vectors" / "demo_421069_gravity.yaml")
scene, times, frames, labels = RPV.load_inputs(DEMO, str(DEMO["visit"]), str(DEMO["video"]))
demo_rows = H.run_json("demo_421069_gravity", "parts.json")
truths = {p: RPV.truth_of(scene, p) for p in {r["part"] for r in demo_rows}}
media = json.loads((ROOT / "runs" / "report_media" / "week2_media_game" / "media.json").read_text())
CHECKS = []


def check(name, got, want, tol=0.05):
    # every number the text quotes is checked against the result files (listed in the appendix)
    assert abs(got - want) <= tol, f"{name}: text says {want}, files say {got}"
    CHECKS.append(name)
    return name, want


s0c = H.run_json("s0c_gravity_val3")
off = {rule: s0c["variants"][f"lidar/{rule}/best"]["off_part_rotations"] for rule in ("oracle_kind", "oracle_gravity")}
check("S0c door axis right, type given %", round(off["oracle_kind"]["ma"]), 68)
check("S0c door axis right, gravity axis %", round(off["oracle_gravity"]["ma"]), 84)
print(f"scene 421069: {len(times)} keyframes, {len(truths)} parts with a vector")
""")

code(r"""
fig = WR.timeline([
    ("16–17 Nov  public Task 2 vs UniFunc3D 8B; code: one config, one command", "tested"),
    ("18 Nov  block-by-block review; 29 papers; new pipeline: a 3D vector per part", "done"),
    ("18–20 Nov  S0, S0b, S0c: rules + depth give 3D vectors; door axis = gravity", "tested"),
    ("20 Nov  stage 0: five CPU tests; decisions re-analysed", "tested"),
    ("20 Nov  memory form: one Gaussian map per room; trial fit", "done"),
    ("20 Nov  framework v4 frozen; stage 2 queued", "done"),
    ("next  'mounted on' prompt plan, lifting, revisit test, Task 3 scorer", "next"),
])
display(fig)
plt.close(fig)
""")

md(r"""
## Decision log

| Date | Question | Your choice |
| --- | --- | --- |
| 16 Nov | Code rules | files ≤ 500 lines, shared modules, one config and one command per experiment |
| 18 Nov | Main idea | M1–M6: interpretable 3D vector, helps choose keyframes, vector + depth → action, two heads, no training, few resources |
| 20 Nov | Selling points and benchmarks | parked as future directions; the final benchmarks are chosen by the results |
| 20 Nov | Experiment order | cheap tests → freeze the framework → run end to end → benchmark fit → you pick → targeted improvements |
| 20 Nov | Notebooks | one per London week |
| 20 Nov | VLM "mounted on what?" with SAM 3 door panels (decision 1) | fold into stage 2; plan shown to you first |
| 20 Nov | Store "mounted on" per part (decision 2) | yes |
| 20 Nov | Grasp axis, contact region, new metrics (decision 4) | add, in stage 2 |
| 20 Nov | Behind UniFunc3D on Task 2 (decision 5) | accept for now; GPU catch-up once the pipeline is settled |
| 20 Nov | Memory form (decisions 3 and 6) | one map per room; implicit features below, interpretable part vectors above |
| 20 Nov | Overwrite rule | conditional: visible, clear, and the geometry changed |
| 20 Nov | Second-visit test | download the other 52 validation videos (done) |
| 20 Nov | Map representation | Gaussians as the main map, a voxel map as the control; stage 2 decides |
| 20 Nov | Meaning of "no training" | no network training; per-scene reconstruction is allowed |
| 20 Nov | Trial fit card | GPU 3 for that one job |
| 20 Nov | Framework v4 freeze; stage-2 queue | approved; stage-2 jobs in front of your own jobs, on the default cards |
| 23 Nov | This report | concise, English, as visual as possible |
""")

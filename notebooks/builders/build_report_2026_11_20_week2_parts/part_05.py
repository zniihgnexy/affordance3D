# Cells of build_report_2026_11_20_week2.py, part 5 (section 4: memory form and related work; section 5: stage 2;
# section 6: main-idea check; the notes cell).

md(r"""
---
# 4. Memory form: one Gaussian map per room

**What others store:**
- **Text graphs:** an LLM reads captions. A wrong caption, or a part never detected, cannot be recovered, because the
  pixels are gone. On OpenEQA, GPT-4 reading ConceptGraphs captions scores 36.5, close to a blind LLM at 33.5, while
  GPT-4V reading frames scores 55.3.
- **Image memories:** these re-read stored views and lose small parts at their stored resolution.
- **SceneFun3D methods:** these re-read the whole video for every question.
- **Dense feature maps:** these mostly stay at object level, and rarely update across visits.

**The gap:** no published system keeps one training-free, room-scale map that a new video is relocalised into, that is
updated locally below object level, and that stores how each part moves.

The detailed tables, checked against arXiv, are in the proposal document, sections 15–16.
""")

code(r"""
fig = WR.positioning()
display(fig)
plt.close(fig)
""")

md(r"""
---
# 5. Stage 2: run end to end (started 20 Nov)

**Data:** the 30 validation scenes have 82 videos, the phase-1 video of each scene plus 52 second visits (keyframes only,
3.1 GB).

**GPU jobs:** queued in front of your own jobs, on the default cards 0 1 4 5 6 7.
- **SAM 3 part maps** of the 52 second-visit videos.
- **Gaussian maps** of all 82 videos, with the trial's settings.
  - On the first pass, 64 were fitted and 18 failed with CUDA out-of-memory. Our jobs used 3–4 GB, but other users'
    processes of about 17 GB each filled the shared cards.
  - That breaks the prediction "at most 4 failures" (G2-P4) on the first pass. The 18 were re-queued with the same
    settings, waiting for 12 GB of free memory before starting.
  - The cell below shows the state at build time.
""")

code(r"""
spec = importlib.util.spec_from_file_location("fit_gaussians_batch", ROOT / "scripts" / "fit_gaussians_batch.py")
FGB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FGB)
CFG = load_config(ROOT / "configs" / "gaussians" / "fit_val82.yaml")
OUT = ROOT / CFG["outputs"] / CFG["name"]
stems = [f"{v['visit']}_{v['video']}" for v in FGB.videos(CFG)]
rows = [json.loads((OUT / f"{s}.json").read_text()) for s in stems if (OUT / f"{s}.json").exists()]
failed = [s for s in stems if (OUT / f"{s}.failed.json").exists() and not (OUT / f"{s}.json").exists()]
sam3 = sorted((ROOT / "runs" / "stage2" / "tool_maps" / "sam3").glob("*_*.json"))
sam3_done = sum(len(json.loads(p.read_text())["keyframes"]) == 40 for p in sam3)
display(Markdown(f"**At build time:** SAM 3 maps of second-visit videos finished: **{sam3_done} of 52**. "
                 f"Gaussian maps: **{len(rows)} fitted**, **{len(failed)} failed so far and re-queued**, "
                 f"{len(stems) - len(rows) - len(failed)} not reached yet (of {len(stems)})."))
fig = WR.stage2_fits(rows)
display(fig)
plt.close(fig)
pred = FGB.judge(rows, len(stems) - len(rows), CFG["expect"]) if rows else {}
display(pd.DataFrame([[k, v["value"], "held" if v["held"] else "not held"] for k, v in pred.items()],
                     columns=["prediction (over the videos fitted so far)", "value", ""]).style.hide(axis="index"))
check("revisit videos listed", len(stems) - 30, 52)
""")

md(r"""
**Next in stage 2** (each comparison is written down with its predictions before it runs):
1. **"Mounted on what?" (W4):** the VLM prompt and its scoring, shown to you before it runs.
2. **Lifting features onto the Gaussians:** training-free, in the style of Occam's LGS or LUDVIG. The voxel control uses
   the same five operations.
3. **Second visit:** relocalisation with AnyLoc and Reloc3r, then the conditional overwrite, checked by rendering and by
   depth.
4. **Task 3 scorer:** our own, following the public definitions, plus one run under AffordBot's protocol.
5. **Comparisons:**
   - retrieval of the target part, including the 41 parts SAM 3 never found;
   - VLM picks on rendered close-ups vs the best keyframes, on the 44 "chose another part" tasks;
   - update errors on the second visit;
   - cost.

---
# 6. Main-idea check

| Main idea | This week | Verdict |
| --- | --- | --- |
| M1 interpretable 3D vector, robot-ready | part vectors in world coordinates; tested on 30 scenes | conforms |
| M2 the vector helps choose keyframes | each map element records the frames that saw it; close-ups rendered on demand | conforms |
| M3 affordance + depth → action vector | rules on LiDAR depth; the type comes from "mounted on" (stage 2) | conforms |
| M4 two heads (open QA, action vector) | action head built; QA head interface frozen, not built | partly |
| M5 off-the-shelf models + rules, no training | no network trained; per-scene Gaussian fitting counts as reconstruction (your 20 Nov ruling) | conforms |
| M6 minimal resources | stage 0 CPU only; about 15 GPU-min and 0.2 GB per video for a map | conforms |

No drift this week.
""")

md(r"""
## ✏️ Your notes

Write comments and updates for this week here; I read them and answer point by point.

(empty)
""")

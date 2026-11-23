# Cells of build_report_2026_11_20_week2.py, part 4 (sections 3.3-3.6: part vectors, OpenEQA, overwrite, Gaussian trial).

md(r"""
## 3.3 Part vectors on 30 validation scenes (stage 0, test T3)

**Setup:** the part is given (annotated masks). The scores ask whether the memory's vector gets the motion right:
- **+M:** the type is right;
- **+MA:** the type and the axis or direction are right (within 15°);
- **+MAO:** in addition, the origin is right (within 0.25 m).

The 471 parts are those the keyframes show; 78 more never appear in them.

**Readings:**
- **Main rule:** type from the label plus a gravity axis gives +MA 48.8 and +MAO 38.4.
- **Type given:** the same rule reaches 71.5 and 53.9. The type is the largest error, which is why each part will store
  what it is mounted on.
- **Doors and windows:** the axis direction is right for 82.5% of them, but the hinge position only for 24.5%. That needs
  the door panel (SAM 3 door masks, stage 2).
- **Geometry does not settle the type:** guessing "door or drawer" from geometry is worse than the label alone (+M 68.4
  for the main rule vs 74.3 for label only).
""")

code(r"""
t3 = H.run_json("t3_val30", "summary.json")
fig = WR.vector_rules(t3)
display(fig)
plt.close(fig)
g, og = t3["variants"]["lidar/gravity/best"], t3["variants"]["lidar/oracle_gravity/best"]
check("T3 scored", t3["counts"]["scored"], 471)
check("T3 not seen", t3["counts"]["not_seen"], 78)
check("T3 gravity +MA", g["all"]["ma"], 48.8)
check("T3 gravity +MAO", g["all"]["mao"], 38.4)
check("T3 type given +MA", og["all"]["ma"], 71.5)
check("T3 type given +MAO", og["all"]["mao"], 53.9)
check("T3 door axis", og["off_part_rotations"]["ma"], 82.5)
check("T3 door origin", og["off_part_rotations"]["mao"], 24.5)
check("T3 gravity +M", g["all"]["m"], 68.4)
check("T3 label-only +M", t3["variants"]["lidar/c1/best"]["all"]["m"], 74.3)
""")

md(r"""
**Why the label alone cannot settle the type.** In the 30 validation scenes, one part label covers parts that slide
and parts that turn. Even choosing each label's majority type gets at most 80.3% of the parts right.
""")

code(r"""
visits = (ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists" / "val_scenes.txt").read_text().split()
pairs = Counter(pair for visit in visits for pair in CLM.label_motions(ROOT / "data" / "scenefun3d", visit))
fig = H.label_motion_figure(pairs)
display(fig)
plt.close(fig)
best = sum(max(pairs[(lab, "trans")], pairs[(lab, "rot")]) for lab in {lab for lab, _ in pairs})
check("label-only ceiling %", round(100 * best / sum(pairs.values()), 1), 80.3)
""")

md(r"""
**What a robot would do with the memory (demo scene 421069).** Each vector is turned into the path of the contact point:
- slide 30 cm for parts that push or pull;
- turn 60° about the axis for parts that swing;
- turn a full circle for knobs.

Orange follows the memory and green the annotation. Two parts per kind are chosen in advance (those in the most
keyframes), so right and wrong cases both appear.
""")

code(r"""
examples = H.pick_examples(demo_rows, truths, DEMO["main"])
fig = H.motion_paths_figure(examples, truths, frames["lidar"])
display(fig)
plt.close(fig)
""")

md(r"""
## 3.4 OpenEQA: does it ask about parts? (stage 0, test T5)

Questions were keyword-flagged, then a hand-read sample of 60 corrected the shares. Only object-state questions ("is the
door open?") often need a part. Over the three focus categories about 15% of questions do, so OpenEQA stays an
auxiliary check for the QA head.
""")

code(r"""
t5 = json.loads((S0 / "openeqa_fit" / "summary.json").read_text())
fig = WR.openeqa_share(t5["table"]["category"], t5["focus_categories"])
display(fig)
plt.close(fig)
check("OpenEQA focus share corrected %", round(100 * t5["focus_corrected_share"], 1), 15.1)
""")

md(r"""
## 3.5 Why "newest overwrites" needs a condition (overwrite simulation)

**Setup:** one video in which nothing changes. How often does SAM 3 find a part that is in view, and how many found
parts would a "one view decides" rule erase?

**Reading:** a part in view is found on a single keyframe only about two times in three. "This frame did not see it"
therefore mostly means a missed detection, not a change. That is why the chosen rule overwrites only where the view is
clear and the geometry has changed.
""")

code(r"""
ow = json.loads((S0 / "overwrite_sim" / "summary.json").read_text())
fig = WR.overwrite_bars(ow)
display(fig)
plt.close(fig)
check("overwrite ever found", ow["ever_found"], 360)
check("overwrite per-frame found %", round(100 * ow["per_frame_found_rate"], 1), 66.2)
check("overwrite last view erases %", round(100 * ow["erased"]["last_view"]["all"], 1), 14.4)
check("overwrite last view erases, seen 2+ %", round(100 * ow["erased"]["last_view"]["shown_twice_or_more"], 1), 17.2)
check("overwrite largest view erases %", round(100 * ow["erased"]["largest_view"]["all"], 1), 10.3)
""")

md(r"""
## 3.6 Trial fit of the Gaussian map (stage 1, on GPU 3 as you asked)

**Inputs:** ARKit poses and LiDAR depth, as a robot has them; every 8th frame is held out and scored.

**Validation scene 421254** (below, photo | render | rendered depth): 80 frames.
- It fits in 14.7 min with 3.7 GB peak GPU memory.
- On held-out views the PSNR is 25.4 dB and the median depth error 1.6 cm.
- Three of the four predictions held. The file is 201.5 MB, 1.5 MB over its bound.

**Demo scene 421069:** a large room with sparse frames reaches only 20.3 dB and shows floaters. Rendered close-ups are
therefore reported separately by coverage in stage 2.
""")

code(r"""
fit = json.loads((ROOT / "runs" / "gaussians" / "fit_trial" / "summary.json").read_text())
val, demo = fit["scenes"]
display(Image(filename=str(ROOT / "runs" / "gaussians" / "fit_trial" / f"{val['visit']}_views.png"), width=760))
check("trial fit minutes", round(val["fit_s"] / 60, 1), 14.7)
check("trial peak GPU GB", round(val["peak_gpu_gb"], 1), 3.7)
check("trial frames", val["frames"], 80)
check("trial PSNR", round(val["test_psnr_db"], 1), 25.4)
check("trial depth error cm", round(100 * val["test_depth_median_m"], 1), 1.6)
check("trial file MB", round(val["file_mb"], 1), 201.5)
check("demo PSNR", round(demo["test_psnr_db"], 1), 20.3)
""")

# Cells of build_report_2026_11_20_week2.py, part 3 (section 2: frozen pipeline v4; section 3.1-3.2: Task 2 results).

md(r"""
---
# 2. Pipeline v4 (frozen 20 Nov)

Orange is the claim, blue are frozen public models, and dashed blocks are not built yet. What is frozen is each block's
input and output; how a block works inside can still change.
""")

code(r"""
display(Image(filename=str(FIG / "proposal_2026-11-18" / "proposal_v4.png"), width=950))
""")

md(r"""
| Block | Input → output | Uses now | Status |
| --- | --- | --- | --- |
| W1 keyframes | video → 40 keyframes | poses, sharpness | built |
| W2 part masks | keyframes → part masks and part names | SAM 3, 8 part names | built |
| W3 depth, poses | keyframes → LiDAR depth, camera poses (scan frame) | phone LiDAR, ARKit | built |
| W4 type and host | part → motion type, mounted on (door / drawer / panel / lid) | label prior; VLM question | stage 2 |
| W5 affordance vector | part, W3, W4 → contact, normal, region, grasp axis, type, axis, origin (world) | rules | prototype |
| W6 scene map | video → one map per room: Gaussians, lifted features, visibility, last-confirmed time, part records | gsplat, SigLIP 2; voxel control | trial fit done |
| W7 revisit | new video, old map → pose, changed regions, updated map | AnyLoc + Reloc3r; render or depth check | stage 2 |
| R1–R3 read | question → object and action → candidates and best or rendered views → VLM picks the part | LLM, map query, VLM | R3 built |
| Heads | part → world action vector (grasp, then motion); question → free-text answer | part record; VLM | action head built |

The map offers five operations, shared by the Gaussian map and the voxel control: build, update, query, views, part.
""")

md(r"""
---
# 3. Results

## 3.1 Public SceneFun3D Task 2: find the part (445 validation tasks)

The same scorer is used for every arm; UniFunc3D's numbers come from its public predictions. With SAM 3 part maps the
memory is 19.1 points behind UniFunc3D 8B; with annotated part maps it is level. The upper bounds show what a perfect
mask would give on the frame the model chose, and on the best frame.
""")

code(r"""
PE = ROOT / "runs" / "public_eval" / "public_eval_split0_8b"
pe = json.loads((PE / "summary.json").read_text())
cmp = {p.stem: json.loads(p.read_text()) for p in sorted((PE / "compare").glob("*.json"))}
ref = cmp["p4t_sam3_vs_reference"]["first_minus_second"]
unifunc = pe["means"]["p4t_sam3"]["AP25"] - ref["AP25"]["rate"]  # the reference's own score on the same tasks
scores = {"raw video, 40 frames (VLM)": pe["means"]["c3"]["AP25"], "ours: memory, SAM 3 part maps (3 frames)": pe["means"]["p4t_sam3"]["AP25"],
          "memory, annotated part maps": pe["means"]["p4n"]["AP25"], "UniFunc3D 8B": unifunc,
          "upper bound: perfect mask, chosen frame": pe["means"]["oracle_chosen"]["AP25"],
          "upper bound: perfect mask, best frame": pe["means"]["oracle_best"]["AP25"]}
fig = WR.task2_bars(scores)
display(fig)
plt.close(fig)
check("raw video AP25", round(pe["means"]["c3"]["AP25"], 1), 19.3)
check("memory SAM 3 AP25", round(pe["means"]["p4t_sam3"]["AP25"], 1), 24.9)
check("memory annotated AP25", round(pe["means"]["p4n"]["AP25"], 1), 43.1)
check("UniFunc3D 8B AP25", round(unifunc, 1), 44.0)
check("memory minus UniFunc3D AP25", round(ref["AP25"]["rate"], 1), -19.1)
check("memory minus raw video AP25", round(cmp["p4t_sam3_vs_c3"]["first_minus_second"]["AP25"]["rate"], 1), 5.6)
check("oracle_chosen AP25", round(pe["means"]["oracle_chosen"]["AP25"], 1), 51.5)
check("oracle_best AP25", round(pe["means"]["oracle_best"]["AP25"], 1), 88.1)
""")

md(r"""
## 3.2 Where Task 2 is lost, and where the memory wins (stage 0, tests T1 and T2)

- **Losses:** of the 105 tasks that annotated maps get right and SAM 3 maps get wrong, the VLM chose another part in 44,
  SAM 3 never found the part in 41, and the lifted mask missed in 19. Reading is the largest loss, then the part maps;
  misses differ by 51 points between part labels.
- **Wins:** the memory beats the raw video most on small parts and in scenes with look-alike parts, the task the memory
  is built for.
""")

code(r"""
S0 = ROOT / "runs" / "stage0"
t1 = json.loads((S0 / "task2_losses" / "summary.json").read_text())
fig = WR.stage0_losses(t1)
display(fig)
plt.close(fig)
check("T1 hit", t1["stages_memory_arm"]["hit"], 111)
check("T1 lost to annotated", t1["lost_to_annotated"]["tasks"], 105)
check("T1 lost: not chosen", t1["lost_to_annotated"]["stages"]["not_chosen"], 44)
check("T1 lost: never found", t1["lost_to_annotated"]["stages"]["never_found"], 41)
check("T1 lost: mask", t1["lost_to_annotated"]["stages"]["lost_in_mask"], 19)
check("T1 label spread", round(100 * t1["predictions"]["T1-P3"]["spread"]), 51)
check("T2 all", round(t1["memory_minus_raw"]["all"]["diff"], 1), 5.6)
check("T2 small", round(t1["memory_minus_raw"]["small"]["diff"], 1), 10.3)
""")

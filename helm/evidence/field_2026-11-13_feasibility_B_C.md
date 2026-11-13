# Feasibility scout: B (questions about past actions, HD-EPIC) and C (action-ready answers, SceneFun3D)

Date: 2026-11-13. Desk check only: no video, pose, twin or mask data downloaded, no GPU, no contact with the project server.
Labels used: **[measured]** = I observed it today; **[project-measured]** = a number already measured by the project on the server; **[claimed]** = stated by a source, not checked; **[extrapolated]** = my estimate from measured numbers; **[unverified]** = open.

Deviation to note: besides HEAD requests, I read about 1.8 MB of published *text* (the downloader's md5 manifest 80 KB, the annotations README 19 KB, the seven VQA question files 1.66 MB from GitHub, nine `durations.txt` files 5.5 KB from data.bris). No video, pose, twin, hand-mask or movement-annotation bytes were fetched. Files are in `scouts/web/`.

---

## 1. Verdict

- **B: testable after named work.** (i) PI go for one kitchen's videos (P02: 10.24 GB) and for GPU time outside the approved phase-1 spec; (ii) a CPU-only step that checks every question can be tied to annotated events (needs ~30 MB of GitHub annotations); (iii) a PI ruling on whether Aria MPS wrist/palm tracking counts as an allowed input. That ruling only matters for the non-ORACLE event step. Main warning: the ORACLE event memory is built from the same annotations that generated the object-motion answers, so its score is high by construction. It is a ceiling and nothing more.
- **C: testable now.** The data for 30 val scenes is already on the server. It needs a script that turns the ground truth into action records (CPU) and a VLM pass (GPU queue, outside the phase-1 spec, so PI go). Main warning: two things measured on one local scene. The direction at the contact point is ±the surface normal for most parts. ORACLE labels almost fix the motion type. So "direction" scores mostly test normal estimation and the push/pull sign. Only hinged parts (axis, hinge side) and contact-on-part separate the methods.

## 2. Data, by what it has to teach

| what must be learned or measured | resource | what it provides | what is still missing | access status (2026-11-13) |
|---|---|---|---|---|
| Past-action questions and answers | HD-EPIC VQA, `hd-epic-annotations/vqa-benchmark/*.json` (GitHub) | 7 prototypes × (200–500) = 2,400 questions **[measured]**, all 5-way multiple choice, all single-video, inputs = video id + `<TIME>` + `<BBOX>` in the text, `correct_idx` public | per-prototype published baselines (paper Fig. 11 not read) | downloaded and parsed (only the 7 files) |
| Raw input for the "VLM reads the video" arm | HD-EPIC `Videos/PXX/*.mp4` on data.bris (mirror on OneDrive without VRS) | 156 mp4, 41.32 h total **[measured from durations.txt]**, 123.56 GB (decimal) **[measured, HEAD]**; 30 fps; capture 1408×1408 **[claimed, paper]** | codec/mp4 resolution **[unverified]**; download speed from the server **[unknown]** | entry point reachable (all 506 HEAD → 200) |
| Event tokens, ORACLE (object moves) | `scene-and-object-movements/assoc_info.json` 2.9 MB + `mask_info.json` 7.6 MB (GitHub) | per video: object association → tracks with `time_segment` → mask IDs; per mask: frame number, `3d_location`, bbox, fixture ID (e.g. `P01_cupboard.009`) **[claimed, README]**; 19.9K tracks **[claimed]** | README warns masks and boxes "may be inconsistent". No public fixture-ID → name mapping found (answers use names such as "counter left from sink") | entry point reachable (tree listing); not downloaded |
| Event tokens, ORACLE (actions, open/close) | `HD_EPIC_Narrations.pkl` 17.5 MB (GitHub) | start/end time, verbs, nouns, main actions, hands **[claimed]** | **no fixture ID**. The paper resolved open/close to a fixture through gaze; no per-fixture open/close file found in the repo | reachable; not downloaded |
| Pick-up/put-down with 3D position | `eye-gaze-priming/priming_info.json` 8.8 MB | start/end frame + 3D location per object movement **[claimed]** | — | reachable; not downloaded |
| Device pose aligned to mp4 (input per charter) | "Intermediate data" (OneDrive, Jan 2026) or `SLAM-and-Gaze/PXX/SLAM/multi/*.zip` | per-frame device-to-world, calibration, gaze **[claimed]**; SLAM zips also hold dense point clouds | intermediate-data size unknown (OneDrive returned 403 to the automated fetch); SLAM for P02 is 19.28 GB **[measured]** | intermediate: claimed by its site; SLAM: reachable |
| Hand signal for non-ORACLE event detection | `SLAM-and-Gaze/PXX/GAZE_HAND/*.zip` | MPS gaze + wrist/palm positions **[claimed, data.bris readme]**; P02 ≈ 0.02 GB **[measured]** | which frame the hand data is in (device vs world) **[unverified]**; whether it is an allowed input (PI) | reachable |
| Fixture geometry/names | `Digital-Twin/blenders.zip` 1.44 GB, one zip for all kitchens **[measured]** | Blender twin with fixtures **[claimed]** | whether names like "counter left from sink" are stored **[unverified]**. Under the charter, a reconstruction is evaluation/training only | reachable |
| 2D hand masks | `Hands-Masks/*.zip` 2.08 GB, all kitchens, 3 zips **[measured]** | 7.7M masks, 11K manual **[claimed]** | — | reachable |
| Part masks + motion parameters (C) | SceneFun3D `annotations.json`, `motions.json`, laser scan; 30 val scenes on server | per part: affordance label (9 + exclude), motion type rot/trans, `motion_dir`, `motion_origin_idx`, `motion_viz_orient` | contact point, hand pose, approach side/stance, motion range, state, effect target. Official Task-3 eval page "coming soon" (as recorded 2026-11-10, not rechecked) | local train scene 421069: **actually ran** (CPU probe below); val scenes: claimed by project docs, not checked today |

### HD-EPIC per kitchen [measured: HEAD Content-Length, durations.txt, VQA files]

"Past-action" = the 6 prototypes below, i.e. the 2,400 minus `fixture_location` (500). `fixture_location` asks about static spatial layout ("where is the boiler relative to my gaze"), which belongs with A.

| P | videos | hours | mp4 GB (all) | SLAM GB | past-action Q | videos those Qs use | their hours | their GB | median video min |
|---|---|---|---|---|---|---|---|---|---|
| P01 | 27 | 5.09 | 18.98 | 23.74 | 278 | 27 | 5.09 | 18.98 | 7.9 |
| P02 | 19 | 4.58 | 12.01 | 19.28 | **242** | 15 | 3.94 | **10.24** | 9.0 |
| P03 | 21 | 7.15 | 18.84 | 26.66 | 381 | 21 | 7.15 | 18.84 | 15.2 |
| P04 | 18 | 4.62 | 14.15 | 61.07 | 198 | 17 | 4.46 | 13.74 | 16.5 |
| P05 | 26 | 3.45 | 11.63 | 57.57 | 237 | 25 | 3.44 | 11.61 | 4.8 |
| P06 | 11 | 4.09 | 13.32 | 45.64 | 127 | 11 | 4.09 | 13.32 | 19.8 |
| P07 | 9 | 3.59 | **8.47** | 47.00 | 133 | 8 | 3.06 | **7.38** | 28.4 |
| P08 | 12 | 4.04 | 10.29 | 45.07 | 113 | 11 | 3.98 | 10.16 | 19.2 |
| P09 | 13 | 4.71 | 15.87 | 47.79 | 191 | 13 | 4.71 | 15.87 | 17.6 |
| all | 156 | 41.32 | 123.56 | 373.83 | 1,900 | | | | |

Shared files (not split by kitchen): Digital-Twin 1.44 GB, Hands-Masks 2.08 GB, GAZE_HAND 0.70 GB in total.

Past-action questions per prototype in P02 / P07: fixture interaction counting 28 / 19; object contents retrieval 12 / 22; object location 76 / 14; movement counting 30 / 14; movement itinerary 72 / 41; stationary object localisation 24 / 23.

- **Smallest by bytes: P07** (8.47 GB for all videos; 7.38 GB for the 8 its questions use; 133 questions).
- **Recommended: P02.** 242 questions for 10.24 GB (15 videos, fetchable by `--video-id`). P02 also has the smallest SLAM (19.28 GB) and ~20 MB of GAZE_HAND. Two of the three COLMAP-aligned videos are P02. With 242 paired questions the 95% CI of a paired accuracy difference is about ±8 pp, against ±11 pp for P07 **[extrapolated, assuming ~40% discordant pairs]**.
- P05 is cheap but its videos are short (median 4.8 min), so there is little memory pressure.

### Answer-prior floors [measured, from the public answers, all kitchens]

| prototype | floor |
|---|---|
| fixture interaction counting | picking the answer most common elsewhere scores **40.3%** ("0" is correct 40% of the time) |
| object movement counting | the same leave-one-out prior pick scores **56.5%** |
| other prototypes | floor ≈ chance (most common answer 0.5–15.5%) |

`correct_idx` positions are roughly uniform (16–24% each). Any arm that defaults to small counts will look good. Every result must be read per prototype against these floors.

### Licence

- data.bris `readme.txt` and the project page: **CC BY-NC 4.0** [claimed].
- data.bris CKAN metadata shows "CC BY 4.0", a discrepancy. Follow the stricter licence.
- Research use is allowed by the charter. No licence request is needed. Object masks are on Dropbox (size not checked).

### Published reference points [claimed, not reproduced]

| result | 3D perception | object motion |
|---|---|---|
| HD-EPIC paper: Gemini Pro on video | 32.5 | 20.8 |
| HD-EPIC paper: LLaVA-Video 7B (≤128 frames) | 27.3 | 18.9 |
| HD-EPIC paper: Gemini Pro given ground-truth narrations as text (Table A4) | — | 38.5 |
| Human | 93.8 | 92.7 |
| EgoAdapt (arXiv 2605.24500, preprint; Qwen3-VL-8B on raw video, no annotations reported) with option-permutation voting | 64.87 | 61.63 |
| EgoAdapt without the voting | 41.39 | 46.73 |

The narration-as-text row is an existing "annotation memory" ceiling. **The raw-video opponent can be much stronger than the 2025 baselines.**

## 3. Allowed inputs and evaluation-only information

**B: test-time inputs**
- mp4 RGB frames.
- The question text, the choices, and the `<BBOX>`/`<TIME>` inside it.
- Device pose (charter, round 2).
- *Pending a PI ruling:* MPS wrist/palm positions and gaze. They are device-derived like pose, but the charter names only pose.

**B: evaluation-only**
- `correct_idx`.
- `assoc_info`, `mask_info`, narrations and priming. These are the source of the answers, allowed only in arms labelled ORACLE.
- The digital twin and the SLAM dense clouds. Both are reconstructions; under the charter they serve only as training material and evaluation truth.
- The answer distribution. The prior floor above is a diagnostic only.

**B: places where evaluation-only information could leak into an input**
1. Choosing frames for the "timestamps-only" or raw-video arms from annotated event times. Those arms must sample uniformly or by input-only criteria.
2. Learning the fixture-ID → name mapping from VQA answers.
3. Labelling detected events with digital-twin fixture positions or IDs.
4. Matching the query box to `mask_info` to identify the object. This is allowed in the ORACLE arm only.
5. Default answers that ride the priors ("0", "3").
6. Tuning option-permutation voting or prompts on the scored questions.
7. Pretraining contamination: the VQA files have been public since 2025, and the training data of Qwen3-VL is unknown. A text-only arm partly checks this.

**C: test-time inputs**
- RGB keyframes.
- LiDAR-scaled monocular depth.
- Device pose.
- Task sentence (as in phase 1).

**C: evaluation-only**
- `annotations.json` part indices, `motions.json`.
- Laser scan and its normals.
- Rendered high-resolution depth (hires_depth) and the registered poses. In phase 1 these were used only to build the ORACLE affordance map and to score.

**C: places where evaluation-only information could leak into an input**
1. The ORACLE affordance label almost fixes the motion type. In 421069, every label maps to a single type, so "type accuracy" is free in ORACLE arms.
2. Using laser-scan normals instead of normals from the depth map.
3. Choosing the frame by ground-truth part visibility. Allowed only in a labelled diagnostic sub-arm.
4. Using the ground-truth motion origin to decide the hinge side.

## 4. Compute

**Measured**
- On a Mac M1 (CPU), one SceneFun3D scene (8.88 M laser points, 37 motions):
  - parse and per-part geometry: 7.5 s wall, 0.90 GB peak RSS;
  - script: `scouts/scripts/motion_geometry_probe.py`.
- From the Mac, HEAD on 506 HD-EPIC files took 60 s. This says nothing about the server's download speed.

**Project-measured (phase 1)**
- Qwen3-VL-4B, 3 frames: ≈1.3 s per question on a shared A6000.

**Extrapolated, not measured**

*C*
- Ground-truth derivation for 30 val scenes: ≈4 min CPU.
- VLM pass over 390 tasks × 2 models (4B, 8B): ≈0.3–1 GPU-h.

*B (P02)*
- Download of 10.24 GB:

  | speed | time |
  |---|---|
  | 20 MB/s | ≈9 min |
  | 1 MB/s | ≈2.8 h |
  | 0.3 MB/s | ≈9.5 h |

- Pre-extracting frames at 1 fps, 448 px, from 3.94 h of 1408² video: ≈20–40 min CPU decode. Not measured: there is no ffmpeg on the Mac.
- VLM time per question with 32 frames at 448 px (≈3–6K visual tokens):
  - 4B: ≈2–5 s;
  - 8B: ≈4–9 s.
- All arms: 242 questions × 5 arms × 2 models ≈ 0.7–2.5 GPU-h, ×5 if options are permuted five times.
- Peak memory ≈10–20 GB (4B) and ≈18–28 GB (8B), which fits a 48 GB A6000.
- Replace these with the measured per-question time of the c3 arm (40 frames, Qwen3-VL-4B) now running on the server.

## 5. Probe ladder

### B (one kitchen, P02; thresholds fixed now)

**B-0. Engineering check.** CPU only; ~30 MB of GitHub annotations; no video.
- Question: can each of the 242 questions be tied to annotated events?
- Work:
  - resolve each `<BBOX>`@`<TIME>` to an association in `mask_info` (IoU ≥ 0.5, ±15 frames);
  - for fixture counting, list the open/close narrations in the video;
  - re-derive the answers for movement counting and stationary localisation with rule solvers on `assoc_info`, using the paper's rules (tracks ≥1.1 s apart, ≤20 cm displacement).
- Continue if ≥90% of questions resolve and the rule solvers reproduce ≥90% of the ground-truth answers.
- Otherwise this is a data failure: frame/time alignment, or the box inconsistencies the README warns about.
- Supports no statement about the method.

**B-1. Can events be found without annotations?** CPU; GAZE_HAND ~20 MB + intermediate poses (size unknown); needs the PI ruling on MPS hands.
- Detect interaction moments from wrist/palm speed minima and dwell. Each event = (t, 3D hand position).
- Score against the annotated pick-up/put-down frames and 3D locations (evaluation-only): recall within ±1 s and 30 cm.
- Continue to B-2 as planned if recall ≥ 60% at precision ≥ 50%.
- If lower, the non-ORACLE event tokens need an RGB detector (GPU). Then B is costly and the PI should weigh it against A and C before B-2.
- This is the cheapest step that could change "B alongside A and C".

**B-2. Headroom and event structure.** P02 videos (10.24 GB); GPU queue.

| arm | information it gets | what it is for |
|---|---|---|
| B-text | question + choices only | prior/contamination floor |
| B0 raw video | 32 frames uniform over the whole video + the query frame with the box drawn, timestamps as text | the "traditional VLM" opponent (R8); report frames and seconds per question |
| B1 timestamps-only memory | K = 16 frames picked by input-only rules (uniform, or our keyframe selector on pose), each with its timestamp | plain alternative at equal budget to B3 |
| B3n ORACLE narrations | ground-truth narration text with times (≤2K tokens) + the same K = 16 frames | less-circular ceiling (published Gemini 38.5 on motion) |
| B3 ORACLE events | annotated events as text (object, time segment, from/to fixture ID, 3D position; ≤2K tokens) + K = 16 evidence frames at event starts and ends | diagnosis-only ceiling; circular for object-motion prototypes |

- Same model (Qwen3-VL-4B, then 8B), same prompt, same answer parsing in every arm.
- Metric: 5-way accuracy per prototype and pooled. Paired bootstrap 95% CI. Every number reported next to the prior floors.
- Continue if both hold:
  - B3 − B0 lower CI ≥ 10 pp (headroom);
  - B3 − B1 lower CI ≥ 5 pp (event structure adds something beyond timestamps).
- If a difference is > 10 pp but its CI is too wide, record it as "unclear" and add P07 (+7.38 GB). Do not reinterpret.

**B-3** (only after B-1 and B-2 pass). Memory built from the B-1 detected events (B4).
- Report the fraction of the B3 − B1 gain that B4 keeps.
- The G3-style pass bar: ≥ 50%.

### C (30 val scenes, 390 answerable tasks)

**C-0. Engineering check.** CPU.
- Build per-task action records: part, label, type; signed direction for translations (official viewer convention: `inwards` → −`motion_dir`); axis and origin for rotations; hinge side projected into the keyframe.
- Visually check the sign conventions on 10 parts. The rotation-axis sign may encode the opening sense, but this is undocumented.
- Continue if:
  - ≥ 90% of tasks have motions for every target part;
  - the projected direction is ≥ 20 px long for ≥ 80% of tasks;
  - the scorer gives 100% to the ground truth itself.

**C-1. Is there headroom on action fields?** GPU ≈0.3–1 h. Each arm uses the frame picked by c3 or P4 respectively, plus a diagnostic sub-arm on the ground-truth-best frame.

| arm | information it gets | what it is for |
|---|---|---|
| V plain VLM | the chosen frame | JSON: point, label (9), push/pull/rotate, 2D arrow, hinge side |
| G geometry rule | V's point + V's label; direction from the depth-map normal by a type rule | plain alternative at equal input |
| M our memory | P4 ORACLE region → contact, ORACLE label → type; direction from the same depth rule | the system; labelled as an ORACLE ceiling |
| O | ground-truth part + motion | scorer check only (must score 100%) |

- Metrics:
  - contact-on-part (point inside the projected part, +8 px);
  - label and type accuracy;
  - push/pull sign for translations;
  - axis orientation class + hinge-side accuracy for rotations;
  - median 3D angle error after lifting with depth;
  - joint "action-ready" success;
  - AP25+T/+TD-style analogues, comparable with AffordBot's 18.3/10.8 and Fun3DU+motion's 11.5/4.0 [claimed, NeurIPS 2025].
- C keeps a separate claim only if both hold:
  - M − V joint success lower CI ≥ 10 pp;
  - given correct contact, M − V on direction fields ≥ 5 pp.

**Measured diagnostic (C, train scene 421069, 37 motions; not evidence)**

| part type | angle between motion direction/axis and the local surface normal |
|---|---|
| translations (press, plug) and knobs | median 1.4–7.5° (key_press 31°, n = 2) |
| door/oven hinges | ≈ 89°; origin 0.26–0.31 m from the part |

- `hook_turn` is mixed: 2 handle-twist axes and 3 door/window hinges.
- A type-conditioned normal rule gets **29/37 within 20°**.
- 23/24 task sentences have motions for every target part.

## 6. Outcomes, interpreted in advance

| result | meaning | next |
|---|---|---|
| B3 − B0 lower CI < 5 pp (no headroom) | raw-video VLM already answers what annotations know | stop elaborating B; keep B0 as a comparison row |
| B3 ≫ B0 but B1 ≈ B3 | gain is from keeping timestamped frames, not from events | event tokens not supported; B reduces to keyframe memory |
| gain only when B3 gets more frames/tokens than B1 | hypothesis not supported | rerun at an equal token budget before any claim |
| B-0 < 90% resolvable or rule solvers < 90% | data failure | fix alignment/mapping before any method work |
| B-1 recall < 60% | non-ORACLE events need a GPU detector | PI weighs B's cost against A and C |
| B-text ≥ B0, or gains only on the counting prototypes | answer priors or contamination | report against the floors; use option permutation in all arms |
| B0 (8B) ≥ 45% on object motion | opponent as strong as EgoAdapt without voting | headroom judged against that |
| C: V within 5 pp of M on direction, given correct contact | C collapses into A (localisation) | report C as a metric of A |
| C: G ≈ M | direction comes from depth geometry, not from memory | the claim narrows to "memory supplies contact + label" |
| C-0: sign conventions inconsistent | data failure for sense/hinge | score axis orientation only |

## 7. Costly actions (none started; all need the PI's go)

1. P02 videos: 10.24 GB (15 files; all P02 = 12.01 GB). Fallback P07: 7.38 GB.
2. GitHub annotations: about 37 MB in total.
   - `assoc_info`, `mask_info`, narrations and priming: ≈37 MB together.
   - Small, but outside this desk check.
3. Pose source:
   - Intermediate data: size unknown.
   - Avoid P02 SLAM (19.28 GB).
4. GAZE_HAND P02: ≈20 MB, together with the input ruling on MPS hands.
5. Optional later: digital twin 1.44 GB (evaluation-only); hand masks 2.08 GB.
6. GPU outside the phase-1 spec:
   - B-2 ≈ 0.7–2.5 GPU-h (×5 with permutations);
   - C-1 ≈ 0.3–1 GPU-h.
7. A one-file speed test from the server to data.bris (e.g. a 14.5 KB GAZE_HAND zip, then one ~0.5 GB mp4).

## 8. Not checked

**Downloads and tests**
- No video, pose, SLAM, twin, hand-mask or movement-annotation bytes were downloaded.
- No download speed was tested from the server.
- No VLM was run, and frame decoding was not timed.

**Files and contents not inspected**
- mp4 codec and resolution.
- The twin's fixture names.
- The format and frame of the intermediate data and of GAZE_HAND.
- `fine_grained_action_localization.json` (9.3 MB). It may hold "when did I …" past-action questions.
- The paper's per-prototype baselines (Fig. 11).

**Not verified**
- EgoAdapt numbers: preprint, not verified.
- The rotation-axis sign semantics.
- Motion coverage in the 30 val scenes; only the train scene 421069 was measured.
- The Codabench test protocol.

## Sources (accessed 2026-11-13)

**HD-EPIC**
- Project page: https://hd-epic.github.io/site/
- data.bris record: https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47
- data.bris readme: https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/readme.txt
- Downloader and md5 manifest: https://github.com/hd-epic/hd-epic-downloader
- Annotations and VQA: https://github.com/hd-epic/hd-epic-annotations
- Paper: https://arxiv.org/html/2502.04144v2

**Other HD-EPIC VQA results**
- EgoAdapt: https://arxiv.org/html/2605.24500v1
- 2025 challenge report: https://arxiv.org/abs/2506.08553

**SceneFun3D and C**
- SceneFun3D annotation docs: https://scenefun3d.github.io/documentation/dataset/annotations/
- SceneFun3D toolkit (`examples/show_annotations.py`): https://github.com/SceneFun3D/scenefun3d
- AffordBot: https://arxiv.org/html/2511.10017v1

**Local files**
- `data/scenefun3d/421069/`
- `docs/mem/mem分支_问题定义与讨论稿_2026-11-10.md` (§20.5, §20.7, §24)
- `helm/rounds/R8.md`
- `helm/LEDGER.md`

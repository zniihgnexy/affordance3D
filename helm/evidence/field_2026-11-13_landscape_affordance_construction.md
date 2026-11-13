# Landscape: building part-level, action-typed affordance maps from video for a write-once / search-many index

Scout record, 2026-11-13. Targeted scoping review, not exhaustive. Cutoff: arXiv listings and web results visible on 2026-11-13.

Reading grades in this file follow the helm's request: **A** = full text read; **B** = method and experiment sections read (here: arXiv HTML or PDF text, read through targeted extraction of method, protocol and result tables); **C** = abstract only. No work was reproduced. A C-grade reading cannot show that a capability is absent.

Three things are kept apart below: **[paper]** = what a paper states; **[synthesis]** = my reading across papers; **[to test]** = something the project could check. Numbers are compared only inside one protocol; SceneFun3D numbers differ strongly between papers' own reproductions (see Evidence E7).

---

## Part 1 - Assessment

### 1. Verdict and positioning

- The combination "query-agnostic + part-level + action-typed" already exists, but **only on laser-scan point clouds** (SceneFun3D Task 1: Mask3D-F, and the ECCV 2026 challenge entry TopoField on the VOLT backbone) and, in a weaker form, **from posed RGB-D frames with a trained 2D detector** (FunGraph, 7 action classes, 2D mAP50 26.7). So "a query-agnostic, part-level, action-typed affordance map" cannot be claimed as new by itself.
- The strong SceneFun3D methods of the last 12 months (Fun3DU, TASA, AffordBot, T-FunS3D, UniFunc3D, AFFORDMEM, ThinkAfford) are all **per-query**: each instruction triggers its own VLM pass over tens of views, at 37-332 s per query where reported. None builds a reusable part index from video and none is evaluated as an index for retrieval.
- No work found has, together: video-only input (no scan point cloud), part granularity, a closed action vocabulary, one entry per physical part across frames, a cheap single write pass, and **recall of all parts in the scene** as the reported target. The nearest partial matches are FunGraph (closest overall), ThinkAfford's proposal stage (closest on "high recall, query-free proposals", but conditioned on an affordance type and lifted onto the scan), OpenFunGraph (training-free, query-agnostic, but no action types and a loose recall metric), and EPIC-Aff (egocentric, action-typed, query-agnostic map, but hotspot-level, not part-level).

### 2. Terms (used the same way throughout)

| Term | Meaning here |
| --- | --- |
| Affordance map | Per-frame or per-scene output saying where one can act and with which action |
| Part-level | Granularity of SceneFun3D "functional interactive elements": handles, knobs, buttons, switches, sockets, foot pedals; smaller than objects |
| Action-typed | Each region carries an action label from a closed set (e.g. SceneFun3D's 9: rotate, key_press, tip_push, hook_pull, pinch_pull, hook_turn, foot_push, plug_in, unplug) or a motion type (rotation / translation) |
| Query-agnostic (write-time) | The map is produced before, and independently of, any user instruction; the same map serves every later query |
| Per-query | The method needs the instruction (or a parsed object / affordance name) to run its detection or segmentation |
| Cross-frame consistent | The same physical part gets one identity across the frames in which it is seen |
| Recall-oriented | The reported target includes the fraction of all ground-truth parts in the scene that appear in the output |

### 3. Landscape (one table per line of work)

Column "Q-agn?" = query-agnostic at write time. "Trained?" = any task-specific training or fine-tuning.

#### 3a. Functional-part detection and segmentation in 3D indoor scenes (SceneFun3D line)

| Work | Venue / first posted | Input | Output | Trained? | Q-agn? | Numbers (as stated; protocol in brackets) | What it implies or rules out for us | Grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SceneFun3D benchmark, baselines Mask3D-F, SoftGroup-F, OpenMask3D-F | CVPR 2024 (CVF open access; no arXiv version found) | Laser-scan point cloud (XYZ, RGB, normals) | Task 1: instance masks of all functional elements + one of 9 affordance labels; Task 2: mask for a task description; Task 3: motion type, axis, origin | Yes (Mask3D-F, SoftGroup-F) | Task 1 yes; Task 2 no | Task 1 [test]: Mask3D-F AP 7.9 / AP50 18.3 / AP25 26.6; SoftGroup-F 3.6 / 8.4 / 17.2; LERF 4.8 / 12.3 / 18.1. Task 2: OpenMask3D 0.0 / 0.0 (AP50 / AP25); LERF 4.9 / 11.3; OpenMask3D-F 8.0 / 17.5. Task 3: Mask3D-FM AP25 26.6, +motion type 23.8, +axis 9.8, +origin 7.9. Dataset: 14,867 element annotations, 710 scenes | **Task 1 is already the query-agnostic, part-level, action-typed target**, but on scans. Shows motion type is cheap to add once the mask is right (26.6 -> 23.8) while axis is hard | B |
| TopoField (decoder on VOLT) | arXiv 2026-10-27; ECCV 2026 SceneFun3D Challenge tech report, 2nd place | Laser scans, 1 cm voxels | 9-class instance masks with confidence | Yes (VOLT trained on SceneFun3D) | Yes (no language query described; track not named in the paper) | Official test: AP 28.3 / AP50 41.4 / AP25 47.6. Validation (30 scenes, 561 instances): some candidate with IoU >= 0.5 exists for 67.2% of instances; top-1 20.0%, top-3 38.3%, top-5 46.3%; val AP 39.67. No runtime | Strongest query-agnostic number found. Its own diagnostic splits **candidate recall** from **ranking**: the same split matters for an index. Scan-only | B |
| VOLT (Volume Transformer) | ECCV 2026 (arXiv 2026-04-21) | Point cloud | General 3D backbone | Yes | n/a | Abstract reports no SceneFun3D numbers | Backbone behind TopoField | C |
| FunGraph | IROS 2025 (arXiv 2025-03-10) | Posed RGB-D frames | 3D scene graph with functional-element nodes; 7 of the 9 SceneFun3D classes (plug_in, unplug dropped); VLM refines names | Yes (RT-DETR / YOLOv11 trained on SceneFun3D labels projected to 2D) | Graph is built from the frames and then queried by GPT-4o (inferred; not stated explicitly) | 2D detection [val, mAP50]: GroundingDINO zero-shot 0.7, YOLO-World 0.0, YOLOv11 22.4, RT-DETR 26.7. 3D: AP 5.9 / AP50 16.0 / AP25 30.3 (authors say not comparable with Mask3D-F 7.9 / 18.3 / 26.6). Task grounding (99 queries): AP25 33.3 vs ConceptGraphs 0.0. No runtime | **Closest existing construction method** to "CA": query-free, part-level, action-typed, from frames. Weak point is 2D recall; zero-shot detectors are near zero on these parts | B |
| OpenFunGraph | CVPR 2025 (arXiv 2025-03-24) | Posed RGB-D | Functional 3D scene graph: objects, interactive elements (free-text descriptions), functional edges | No (RAM++, GroundingDINO, GPT-4, LLaVA) | Appears so (graph built, then queried; not stated) | SceneFun3D interactive-element node R@3 71.0 vs Open3DSG* 54.4, ConceptGraphs* 6.6; FunGraph3D 44.4. Recall counts a hit at **any non-zero 3D IoU** plus a top-3 label match. Prompting GroundingDINO with "object. part" tags: node R@3 59.3 -> 73.0 (SceneFun3D) | Training-free query-agnostic writer exists; no action types; its recall metric is loose and cannot be compared with IoU-thresholded recall | B |
| FunFact | arXiv 2026-04-04 | Posed RGB-D | Probabilistic functional scene graph over an object- and part-centric map | Foundation models + factor graph | Yes | Abstract: improves node and relation recall, better calibration; numbers not read | Calibrated confidences could matter for an index | C |
| Fun3DU | CVPR 2025 (arXiv 2024-11-25) | Point cloud + posed views + task text | Mask for one task | No (LLM, Molmo, SAM) | No | Split0 AP50 16.9 / AP25 33.3 / mIoU 15.2 (as reported, cited in AFFORDMEM and UniFunc3D tables). Reproductions: AP25 23.4 (T-FunS3D), 12.56 (ThinkAfford protocol). Time: 118.4 s / task on A100 (authors, quoted by T-FunS3D); 167 s on A40 (T-FunS3D); about 332 s / query and 82 min / scene on 8 H100 (UniFunc3D) | Per-query baseline; the reproduction spread is a warning on cross-paper numbers | C (numbers from other papers' tables) |
| UniFunc3D | NeurIPS 2026 (arXiv 2026-03-24) | SceneFun3D video frames; masks evaluated on the scan | Mask for one task | No (Qwen3-VL 8B / 30B, SAM3) | No | Split0: 8B AP50 23.82 / AP25 44.04 / mIoU 20.92; 30B 31.24 / 51.01 / 24.30. Split1 30B: 21.32 / 35.76 / 17.09. Time: about 53 s / query (8B), 99 s (30B); 16 / 26 min per scene; 8 H100. Round 1 samples 64 frames at 512x384, K = 4 iterations; whether this is cached across queries is not stated. Limitation stated: parts under 0.1% of image area | Per-query SOTA among training-free methods; it is the SceneFun3D-side version of "a VLM reads the video for every question" | B |
| AFFORDMEM | arXiv 2026-05-12 | Posed RGB-D + a cross-scene bank of annotated example images | Mask for one task | No (Qwen3-VL-32B, SAM3) | Cross-scene bank reusable; in-scene graph timing not stated | Split0 AP50 20.13 / AP25 41.66 / AR25 50.62 / mIoU 17.32; split1 16.3 / 32.7. No runtime. States small parts can be missed (qualitative) | "Memory" here helps the per-query grounder; it is not a scene index | B |
| TASA | arXiv 2025-11-12 | Point cloud + multi-view RGB + task | Mask for one task | Partly (Point Transformer refinement) | No | mAP 23.2 / AP50 26.9 / AP25 28.6 / mIoU 19.7 (split not named in the paper). 37.61 s / sample vs Fun3DU 130.26 s on A800, K = 10 frames | Frame selection is where its speed comes from | B |
| T-FunS3D | arXiv 2026-06-04 | Point cloud + posed RGB-D | Object scene graph built once; part masks per task | No (Mask3D class-agnostic, FG-CLIP, Qwen3-14B, Molmo-7B-D, SAM) | Objects yes; parts no | Val mAP 8.1 / AP50 17.8 / AP25 34.5; AR25 46.9 (Fun3DU 46.7). 78 s / query vs Fun3DU 167 s (A40); per scene 12 s + 580 s | Shows the "index objects once, find parts per query" split; parts still cost a VLM pass per query | B |
| AffordBot | NeurIPS 2025 (arXiv 2025-11-13) | Point cloud rendered to surround views + MLLM | Element location + motion type + motion axis for one instruction | Listed as training-based by AFFORDMEM / UniFunc3D | No | Split0 AP50 20.91 / AP25 24.76 / mIoU 14.42 (AFFORDMEM table) | Only per-query method found that also outputs motion type and axis | C |
| ThinkAfford | arXiv 2026-08-11 | K = 25 RGB views; laser scan used to lift masks | Stage 1 (APG): heatmap proposals from learnable affordance prompts, without object or part names; Stage 2: VLM selects proposals for the instruction | Yes (prompts + decoder on 200 SceneFun3D scenes; GRPO for the selector) | APG does not see the instruction but is conditioned on the parsed affordance type | Its protocol, val: AP 4.42 / AP50 10.69 / AP25 25.46; reproduced Fun3DU 12.56 AP25, AffordBot 10.70. APG coverage R@25 77.5%, R@50 50.6% per query, under an "oracle best-view union" | **Closest to "write for recall, read for precision"**, inside a per-query pipeline. The 26.9-point gap between R@25 and R@50 says parts are found but loosely bounded | B |
| SynthFun3D (action-guided generation of functionality segmentation data) | CVPR 2026 GenRecon3D workshop (arXiv 2025-11-28) | Synthetic scenes from PartNet-Mobility + Objaverse | Training data; LoRA-tuned Gemma3-4B pointer inside Fun3DU | Yes | No | Val AP25 1.12 (real data only) -> 12.81 (real + synthetic); mAP +2.2 | A data route for training a part pointer; absolute numbers low | B |
| OpenTrack3D | arXiv 2025-12-03 | RGB-D stream | Cross-view consistent open-vocabulary instance proposals, online, mesh-free | Partly | Proposals yes | Abstract claims SOTA on several benchmarks including SceneFun3D; numbers not read | A mechanism for cross-frame consistency without a mesh | C |
| SceneFunRI | arXiv 2026-05-14 | SceneFun3D-derived 2D tasks | Location of functional objects that are not visible | Benchmark | n/a | Best baseline Gemini 3 Flash CAcc@75 15.20 | Some targets are never visible; recall from views has a ceiling | C |
| Search3D | RA-L (arXiv 2024-09-27) | RGB-D scene | Hierarchical open-vocabulary 3D representation (parts, objects, materials) | Partly | Representation built once, then searched | Abstract only | Scene-scale part search exists, evaluated on MultiScan / ScanNet++ parts, not functional parts | C |
| Articulate3D / USDNet | ICCV 2025 per prior record (arXiv 2024-12-02) | Point cloud | Part segmentation + motion attributes | Yes | Yes | Abstract only | Query-agnostic articulated / interactable parts with motion, on scans | C |
| OPDMulti | arXiv 2023-03-24 (venue not confirmed here) | Single image | Openable parts + motion parameters, multiple objects | Yes | Yes | Abstract: "remains challenging" | Single-image, query-agnostic, motion-typed openable parts | C |
| IAAO | CVPR 2025 | Multi-view posed images per scene state; 3DGS | Affordance masks and labels, articulation | Per-scene optimization + foundation models | Field built once, queried with CLIP text | Not read | Per-scene optimization; contrary to a cheap write | C (abstract + method excerpt) |

#### 3b. 2D affordance grounding (AGD20K line) and VLM-based affordance / pointing

| Work | Venue / first posted | Input | Output | Trained? | Q-agn? | Numbers | Implication | Grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cross-View-AG, AGD20K | CVPR 2022 | Exocentric + egocentric images; test: egocentric image + affordance label | Heatmap; 36 affordance classes | Weakly supervised | Per affordance label | AGD20K seen KLD 1.538 / SIM 0.334 / NSS 0.927 (from WSAG-PLSP table) | Object-centric images, not scenes; label given at test time | C (+table) |
| LOCATE | CVPR 2023 | Same | Heatmap | Weakly supervised | Per label | Seen 1.226 / 0.401 / 1.177; unseen 1.405 / 0.372 / 1.157 (same table) | Part prototypes transferred from exocentric use | C (+table) |
| WSAG-PLSP (part-level semantic priors) | ICLR 2025 (arXiv 2025-05-30) | Same | Heatmap | Pseudo-labels from an off-the-shelf part segmenter via an affordance -> part-name map | Per label (open-vocabulary text) | Seen 0.890 / 0.510 / 1.547; unseen 1.153 / 0.437 / 1.418 | Part segmentation is a strong prior for affordance regions | B |
| OOAL | arXiv 2023-11-29 (venue not confirmed here) | Image | Affordance segmentation; one example per base category | Yes | Per label | Abstract only | Data-light affordance learning | C |
| INTRA | arXiv 2024-09-10 (venue not confirmed here) | Exocentric images only | Text-conditioned affordance map | Weakly supervised | Per text | Abstract only | Removes paired ego / exo requirement | C |
| AffordanceLLM | arXiv 2024-01-12 | Image + text | Affordance map | Fine-tuned VLM | Per query | Abstract reports gains on AGD20K, no number read | VLM knowledge improves generalization | C |
| AffordanceSAM | ACM MM 2026 (arXiv 2025-04-22) | Image + action | Mask | Yes | Per action | Abstract: SOTA on AGD20K | SAM backbone for affordance | C |
| Affordance-R1 | arXiv 2025-08-08 | Image + instruction | Mask via reasoning | GRPO-trained | Per query | Abstract only | RL reasoning for affordance | C |
| RAGNet / AffordanceNet | ICCV 2025 (arXiv 2025-07-31) | Image + instruction | Affordance map | Trained (273k images, 180 categories) | Per query | Abstract only | Large data incl. egocentric domain; grasp-oriented | C |
| Affogato / Espresso | ECCV 2026 (arXiv 2025-06-13) | Image or 3D + text | Heatmap | Trained on 750K auto-generated pairs | Per query | Abstract only | Automated data at scale | C |
| UAD | arXiv 2025-06-10 | Image + instruction | Affordance map | Distilled decoder on frozen features | Per task | Abstract only | Distillation from VLMs without labels | C |
| InstructPart | ACL 2025 (arXiv 2025-05-23) | Image + instruction | Part mask | Benchmark + fine-tuned baseline | Per query | Abstract: fine-tuning doubles baseline | Task-oriented part segmentation is hard for VLMs | C |
| EgoAfford / EgoLens | arXiv 2026-08-05 | Egocentric image + tabletop task | Masks of up to 3 action components | 3B MLLM | Per task | Abstract only | Tabletop, not fixtures | C |
| RoboPoint | arXiv 2024-06-15 (venue not confirmed here) | Image + instruction | Points | Instruction-tuned VLM | Per query | +21.8% spatial-affordance accuracy over GPT-4o (abstract) | Pointing models are per query | C |
| Molmo / PixMo | arXiv 2024-09-25 | Image + text | Points (pointing data) | Yes | Per query; can be asked to point at every instance of a class | Abstract only | The pointer inside Fun3DU, T-FunS3D, TASA | C |
| PointArena | arXiv 2025-05-15 | Pointing benchmark | - | - | - | Molmo-72B best (abstract) | Pointing quality differs by model | C |
| SAM 3 | arXiv 2025-11-20 | Image / video + concept prompt | All matching instances, masks, video tracks | Yes | Per concept; a fixed part vocabulary can be run once per video | Abstract: doubles prior systems on concept segmentation | Used by UniFunc3D and AFFORDMEM; a candidate query-free writer component (untested on SceneFun3D parts in any paper found) | C |
| Part Grounding, Not Action Knowledge | arXiv 2026-09-01 | Images of 19 articulated objects, 8 VLMs | Action label | Evaluation study | n/a | Naming the target part raises action accuracy from 0.158-0.474 to 0.684-0.947; push recall from 0-1/8 to 7-8/8 | **[paper]** the bottleneck is finding the part, not knowing the action. **[synthesis]** action typing at write time is likely easier than part recall | C |
| VLPart; PartGLEE; OV-PARTS; PartCLIPSeg | arXiv 2023-05; ECCV 2024; NeurIPS 2023 D&B; NeurIPS 2024 | Images | Open-vocabulary object-part segmentation | Yes | Per vocabulary | Abstracts only; evaluated on Pascal-Part, ADE20K-Part, PartImageNet | Part vocabularies come from objects (wheel, leg), rarely from fixtures (socket, switch) | C |
| Find3D | arXiv 2024-11-20 | Object point cloud + text | Part segmentation | Yes | Per text | Abstract only | Object-level 3D parts | C |

#### 3c. Learning affordances or interaction hotspots from human video

| Work | Venue / first posted | Input | Output | Trained? | Q-agn? | Numbers | Implication | Grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Interaction Hotspots | arXiv 2018-12-11 (venue not re-checked) | Video for training; image or video at test | Hotspot map per afforded action | Weakly supervised | Yes (maps for each action) | Abstract only | Origin of "learn where to act from watching" | C |
| Ego-Topo | CVPR 2020 | Egocentric video | Topological map of interaction zones + likely activities | Yes | Yes | Abstract only | Environment affordance at **zone** level; cross-kitchen linking | C |
| VRB | CVPR 2023 | Human videos for training; image at test | Contact points + post-contact trajectory | Yes | Yes | Abstract only | Where and how, but no action vocabulary and no parts | C |
| EPIC-Aff | ICCV 2023 | Egocentric frames; labels from narrations + VISOR masks + COLMAP poses | Multi-label per-pixel affordance map (20 or 43 verb classes); predictions accumulated into a 3D map | Yes | Yes (all classes at once) | Easy set: DeepLab-v3 (asymmetric loss) 42.3 mIoU, Mask R-CNN 40.1 mIoU / 59.3 mAP, hotspot model 17.5 mIoU; complex set 31.1 mIoU | **Closest egocentric, action-typed, query-agnostic map**; regions are interaction hotspots "not aligned with object borders", not parts; relies on SfM | B |
| 3DOI | ICCV 2023 | Single image + a query point | Movable or not, rotation / translation, 2D axis, push / pull, contact, mask, depth | Yes | Needs a point | Abstract only | Output fields close to an index entry, but point-prompted | C |
| Robo-ABC | arXiv 2024-01-15 | Affordance memory of contact points from human video | Contact points mapped to new objects | No | Per object | +31.6% retrieval accuracy (abstract) | Retrieval from an affordance memory, object-level | C |
| Learning Precise Affordances from Egocentric Videos | ICCV 2025 (arXiv 2024-08-19) | Egocentric video -> automatic labels | Affordance segmentation | Yes | Per object / label | +13.8% mIoU; 77.1% grasping over 179 trials (abstract) | Auto-labelling from egocentric video at object scale | C |
| HRP | RSS 2024 | Human videos | Pretrained representation using hand, object, contact labels | Yes | n/a | Abstract only | Affordance labels as pretraining signal | C |
| AFF-ttention / STAformer | arXiv 2024-06-03 (venue not confirmed here) | Egocentric video | Next-active object; environment affordance memory; hotspots | Yes | Yes | Up to +45% relative top-5 mAP on Ego4D STA (abstract) | A per-scene "persistent memory of interactions" already exists, for anticipation | C |
| Ego-SAG / VSAD | arXiv 2024-09-29 | Egocentric interaction video + 3D scene | 3D affordance region for that interaction | Yes | Per interaction video | mAP 10.7 (prior project record) | Interaction video as the query, not as the writer | C |
| VideoAfford | arXiv 2026-02-10 | HOI videos + object point cloud | 3D object affordance | MLLM trained | Per query | VIDA: 38K videos, 16 types (abstract) | Object-level | C |
| VLAff | IROS 2026 (arXiv 2026-08-05) | Egocentric video -> dataset; image + instruction at test | Heatmap, grasp, trajectory | Yes | Per instruction | 204K episodes (abstract) | Object-centric actionable affordance | C |
| DIV-FF | arXiv 2025-03-11 | One egocentric video per scene | Persistent / dynamic / actor feature fields with image- and video-language features | Per-scene optimization | Field built once, text-queried | Affordance segmentation 20.7 mIoU on about 700 masks (from an HTML excerpt) | Video-language features capture action context better than CLIP; costly per-scene field | C (+excerpt) |
| EgoFun3D | arXiv 2026-04-13 | Egocentric interaction video | Receptor / effector parts, articulation, function template | Off-the-shelf pipeline | Per video | 517 videos (abstract v2) | Part function from interaction, object scale | C |
| ArtiSG; FunRec; Pandora; ArtiPoint | Preprint 2025-12; CVPR 2026; BMVC 2025; CoRL 2025 | Human demonstrations / egocentric interaction | Articulated parts, axes, scene graphs | Mixed | Yes | ArtiSG part recall 88.5 with demonstrations vs 55.8 vision-only (prior record) | Interaction raises recall of articulated parts; reused from prior record, not re-read | C / prior record |

#### 3d. Hand-object interaction and contact detection in egocentric video

| Work | Venue / first posted | Input | Output | Trained? | Q-agn? | Numbers | Implication | Grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100DOH | CVPR 2020 | Frames | Hand box, side, contact state, contacted-object box | Yes | Yes | 100K annotated frames (abstract) | Off-the-shelf contact cue; object box, not part | C |
| EgoHOS | ECCV 2022 | Frames | Hand and interacting-object masks, contact boundaries | Yes | Yes | 11,243 labelled images (abstract) | Pixel contact boundary can localize the touched part | C |
| VISOR | NeurIPS 2022 D&B | EPIC-KITCHENS video | Hand / active-object masks and relations | Benchmark | n/a | 272K manual masks, 67K hand-object relations (abstract) | Training and evaluation data for contact | C |
| Hand-Object Contact Transition Detection | ACCV 2026 (arXiv 2026-11-10) | Egocentric video | Touch and release timestamps | Yes | Yes | 4,021 videos, 8,498 touch / 8,077 release labels (abstract) | Gives "before contact / after contact" frames | C |
| EgoLoc (temporal interaction localization) | arXiv 2025-08-17 | Egocentric video | Contact / separation timestamps | Zero-shot VLM | Yes | Abstract only | Same, training-free | C |
| Hands23 | NeurIPS 2023 | Frames | Hands, contact, grasp, second objects | Yes | Yes | Prior record; title not found on arXiv | - | prior record |

#### 3e. Write-once memories compared with a VLM reading the video (context for the system claim)

| Work | Venue / first posted | Granularity | Claim | Grade |
| --- | --- | --- | --- | --- |
| VL-KnG | arXiv 2025-10-01 | Objects (with an affordance attribute in text) | Graph built once from monocular egocentric video; accuracy "competitive with frontier VLMs" at substantially lower latency on OpenEQA, NaVQA, WalkieKnowledge (abstract) | C |
| AMEGO; R4DSG; Ledger | prior records | Objects / HOI tracklets | Query without re-reading the video | prior record |
| MEMORA | EMNLP 2026 (arXiv 2026-07-15) | Environment / entity / activity stores | Memory-grounded planning gains up to 16.6% | C |

### 3f. Property matrix for the nearest works [synthesis]

Y = yes; P = partly; N = no; ? = not stated in what I read.

| Work | Q-agn write | Part-level | Action-typed | Video / frames only (no scan) | One entry per part across frames | Cheap write (no per-scene optimization, no per-node LLM calls) | Reports scene-level part recall | Evaluated as a retrieval index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SceneFun3D Mask3D-F | Y | Y | Y (9) | N (scan) | Y (3D) | Y | P (AP) | N |
| TopoField / VOLT | Y | Y | Y (9) | N (scan) | Y (3D) | P (13 forward passes) | Y (67.2% any-candidate at IoU 0.5) | N |
| FunGraph | Y? | Y | Y (7) | Y (posed RGB-D) | Y (3D fusion) | Y (detector) + VLM naming | P (2D mAP) | P (GPT-4o over graph JSON, 99 queries) |
| OpenFunGraph | Y? | Y | N (free text) | Y (posed RGB-D) | Y (3D fusion) | N (GPT-4, LLaVA per node) | P (R@3 at IoU > 0) | P (QA demos) |
| ThinkAfford APG | P (type-conditioned) | Y | Y (type) | N (scan lifting) | Y (3D voting) | ? | Y (R@25 77.5%, oracle) | N |
| UniFunc3D / Fun3DU / TASA / AFFORDMEM / T-FunS3D | N | Y | N (AffordBot: Y) | N (scan for evaluation) | Y (3D voting) | N (37-332 s per query) | P (AR) | N |
| EPIC-Aff | Y | N (hotspots) | Y (20 / 43) | P (needs SfM) | P (accumulated 3D map) | Y | N (mIoU) | N |
| Ego-Topo | Y | N (zones) | Y (activities) | Y | Y (zones) | Y | N | N |
| AGD20K line, VLM pointers | N (per label / query) | P (object parts) | Y (label is the input) | Y (single image) | N | Y per call | N | N |

**[synthesis] Where a new construction method could be new:** the empty cell is the conjunction of columns 1-8: a query-free writer over egocentric or handheld frames, at part level, with the 9-class action vocabulary, that fuses detections of the same part across frames without a scan point cloud, is optimized and reported for **recall of all parts in the scene** (with precision handled at read time), and is evaluated by what it does for retrieval. Each single property has a precedent; I found no work with all of them. Specific gaps within that:
1. **Recall-first writing is measured but not targeted.** TopoField and ThinkAfford report candidate recall as a diagnostic; no method is trained or designed with scene-level part recall under a fixed write budget as its objective.
2. **No scan-free, part-level, action-typed construction from video is evaluated on SceneFun3D parts** except FunGraph's 2D detector (mAP50 26.7) and OpenFunGraph (no action types).
3. **Cross-frame identity of tiny parts without a point cloud** is not evaluated in any work read (all strong methods vote on the scan).
4. **Interaction as a write-time signal for small, non-articulated parts** (buttons, switches, sockets) has no quantitative result found; ArtiSG / FunRec evidence is for articulated parts.
5. **The cost of per-query methods is large and reported** (UniFunc3D about 53-99 s per query on 8 H100; T-FunS3D 78 s; TASA 37.6 s), so a write-once index has a measurable efficiency target on the same benchmark.

**[to test]** Whether a fixed part vocabulary run once per video with SAM 3 or Molmo, then typed with a VLM, reaches FunGraph-level or better recall on SceneFun3D frames; the literature found gives no number for this (zero-shot GroundingDINO / YOLO-World were near zero in FunGraph, but SAM 3 and Molmo were not tested there).

### 4. Data and platforms

| Resource | What it actually provides | Fit | Gap or access status (checked 2026-11-13 unless noted) |
| --- | --- | --- | --- |
| SceneFun3D | 710 scenes; 14,867 functional elements with 9 affordance labels; motion annotations (type, axis, origin) for 14,279; task descriptions; iPad video + laser scan | Primary; part-level, action-typed ground truth | Project says 30 val scenes already on the server (R8); test-set evaluation via challenge server not checked |
| SceneFun3D challenge (ECCV 2026, OpenSUN3D workshop) | Leaderboard for functionality segmentation | A public number for query-agnostic segmentation (TopoField 2nd place) | Leaderboard page and 1st-place entry not found; "claimed by a tech report" only |
| FunGraph3D | 14 scenes, laser scan + iPad RGB-D + egocentric interaction videos, functional graph labels | Cross-dataset test of a construction method | Prior record: HF, no request needed; not re-checked |
| EPIC-Aff | 38,876 egocentric images, multi-label affordance masks (20 / 43 classes) | Action-typed egocentric training signal (hotspot level) | Access not checked |
| AGD20K; InstructPart; RAGNet; Affogato-750K | Object-centric affordance images / part instructions | Pretraining or auxiliary data; not scene fixtures | Access not checked |
| Articulate3D; OPDMulti | Articulated / openable parts with motion | Motion-typed parts; scans or single images | Access not checked |
| HD-EPIC; VISOR; EgoHOS; ContactTransition | Egocentric interaction video with hands, active objects, contact times | Interaction cue for writing; HD-EPIC for "what was done" questions | HD-EPIC open per prior record; others not checked |
| SynthFun3D generator | Synthetic scenes with functional-part masks from action descriptions | Training data route | Code availability not checked |

### 5. Unsettled

1. Who won the ECCV 2026 SceneFun3D challenge and with what method and numbers; whether any entry worked from video without the scan.
2. Recall of off-the-shelf SAM 3 / Molmo / Qwen3-VL pointing when run once per frame with a fixed part vocabulary on SceneFun3D frames. Not reported anywhere found.
3. Whether UniFunc3D's 64-frame round can be cached across queries (would change the per-query cost comparison).
4. Why reproductions of Fun3DU on SceneFun3D differ so much (AP25 33.3 reported, 23.4 and 12.56 reproduced); a new method must fix one protocol and re-run baselines.
5. OpenFunGraph and FunGraph write cost per scene (not reported).
6. Whether interaction cues help find small non-articulated parts (buttons, sockets).
7. Venues of OOAL, INTRA, RoboPoint, OPDMulti, AFF-ttention and Interaction Hotspots were not re-checked in this pass.

---

## Part 2 - Search and evidence record

### 1. Scope

- **Question:** which methods build part-level, action-typed affordance maps from egocentric or handheld RGB(-D) video or images without re-processing the scene per query; which gaps a new construction method could fill when the map indexes a write-once / search-many memory.
- **Included:** SceneFun3D-line 3D functional-part methods (2024-2026, plus any newer found), 2D affordance grounding and VLM affordance / pointing (emphasis 2024-2026), human-video affordance learning, egocentric hand-object contact detection, open-vocabulary part segmentation (sampled), and three write-once memory analogs for context.
- **Excluded:** robot policy learning from affordances, grasp synthesis, object-level 3D affordance on single point clouds (LASO, OpenAD, PAVLM, 3D-AffordanceLLM, SeqAffordSplat, VoxAfford: titles seen, not read), 4D event memories (covered in the 2026-11-10 project survey).
- **Reused, not re-read:** ArtiSG, FunRec, Pandora, ArtiPoint, Ego-SAG numbers, Hands23, AMEGO, R4DSG, Ledger, Affordance RAG, Memory Over Maps, TASA's Eq. (4) issue (from the 2026-11-02 / 2026-11-10 / 2026-11-11 project records).
- **Cutoff:** 2026-11-13. This is a targeted scoping review, not an exhaustive one.

### 2. Query log (as typed)

**arXiv API (export.arxiv.org, via a small script; first http call returned an empty body, https worked):**
- id_list batches: 2411.16310, 2603.23478, 2511.11702, 2605.11616, 2511.10017, 2503.07909, 2503.19199, 2512.24845, 2603.28732, 2409.19650 | 2605.14704, 2610.37218, 2606.05975, 2604.03696, 2512.03532, 2608.10981 | 2203.09905, 2303.09665, 2311.17776, 2409.06210, 2505.24103, 2608.04533, 2401.06341, 2508.06206, 2507.23734, 2506.09284, 2504.15650, 2505.18291, 2506.12009, 2609.13225, 2406.10721, 2409.17146, 2511.16719 | 1812.04558, 2001.04583, 2304.08488, 2309.02120, 2305.09664, 2408.10123, 2602.09638, 2302.03292, 2401.07487, 2407.18911, 2006.06669, 2208.03826, 2209.13064, 2611.13486, 2508.12349, 2406.01194 | 2305.11173, 2407.16696, 2310.05107, 2406.11384, 2409.18431, 2411.13550, 2303.14087, 2412.01398, 2406.07549 | 2604.19609, 2409.11870, 2503.08344, 2604.11038, 2402.15321 | 2510.01483, 2607.14252, 2608.05215, 2505.09990, 2507.23772
- search_query strings: `abs:SceneFun3D`; `ti:SceneFun3D` (did not return the SceneFun3D paper; no arXiv version found); `abs:VOLT AND abs:point` (noise); `abs:"functional interactive elements"`; `abs:"functionality segmentation"`; `abs:"affordance grounding" AND abs:egocentric`; `abs:"open-vocabulary affordance"`; `abs:affordance AND abs:pointing AND abs:"vision-language"`; `ti:affordance AND abs:"multimodal large language model" AND abs:segmentation`; title lookups `ti:AffordanceLLM`, `ti:"Affordance-R1"`, `ti:RAGNet`, `ti:"Unsupervised Affordance Distillation"`, `ti:AffordanceSAM`, `ti:"One-Shot Open Affordance Learning"`, `ti:InstructPart`, `ti:"Robo-ABC"`, `ti:"SAM 3" AND ti:concepts`, `ti:Molmo AND ti:PixMo`, `ti:"Understanding 3D Object Interaction from a Single Image"`, `ti:"Multi-label affordance mapping"`, `ti:"Affordances from Human Videos as a Versatile Representation"`, `ti:"Grounded Human-Object Interaction Hotspots"`, `ti:"Ego-Topo"`; `abs:"interaction hotspots"`; `abs:affordance AND abs:"egocentric video" AND abs:contact`; `abs:affordance AND abs:"human videos" AND abs:"contact point"`; `abs:"environment affordance" AND abs:egocentric`; `abs:affordance AND abs:"egocentric videos" AND abs:scene AND abs:map`; `ti:"Fine-Grained Egocentric Hand-Object Segmentation"`; `ti:"Richer 2D Understanding of Hands at Scale"` (no result); `ti:"Understanding Human Hands in Contact at Internet Scale"`; `ti:"EPIC-KITCHENS VISOR"`; `abs:"hand-object" AND abs:contact AND abs:egocentric AND abs:detection`; `ti:HRP AND ti:affordances`; `abs:"active object" AND abs:egocentric AND abs:segmentation`; `ti:"Going Denser with Open-Vocabulary Part Segmentation"`; `ti:PartGLEE`; `ti:"OV-PARTS"`; `ti:"Understanding Multi-Granularity for Open-Vocabulary Part Segmentation"`; `ti:Search3D`; `ti:"Find Any Part in 3D"`; `ti:OPDMulti`; `ti:Articulate3D`; `abs:"part segmentation" AND abs:"open-vocabulary" AND abs:scene AND abs:3D`; `abs:"interactable" AND abs:parts AND abs:"single image" AND abs:openable` (no result); `abs:OpenSUN3D`; `abs:"SceneFun3D" AND abs:challenge`; `ti:"Volume Transformer"`; `ti:SpotLight AND abs:affordance`; `ti:"DIV-FF"`; `abs:affordance AND abs:"feature field" AND abs:egocentric`; `abs:"affordance map" AND abs:video AND abs:retrieval`; `abs:affordance AND abs:memory AND abs:egocentric AND abs:part`; `abs:"interactive elements" AND abs:video`; `abs:"functional parts" AND abs:egocentric`; `abs:affordance AND abs:"query-agnostic"` (no result); `abs:"affordance proposals"`; `ti:PointArena`; `abs:"interaction hotspot" AND abs:scene AND abs:3D` (no result).

**WebSearch (standard):** `SceneFun3D functionality segmentation challenge winner technical report point transformer 2025`; `ECCV 2026 SceneFun3D challenge results functionality segmentation first place` (neither found a leaderboard).

**Exa search:** "paper detecting all interactive elements (handles, knobs, buttons, switches) in egocentric video of indoor scenes with action types, class-agnostic, 2025 2026"; "scene-level affordance feature field from RGB-D video, open-vocabulary, queried with action verbs, 3D Gaussian splatting or neural field, interactive parts"; "2026 arXiv: build affordance memory from egocentric video once, then answer many queries by retrieving part-level affordance regions instead of re-running a VLM over the whole video".

**Full-text fetches (WebFetch on arXiv HTML, targeted extraction):** 2610.37218v1 (TopoField), 2606.05975v1 (T-FunS3D), 2608.10981v1 (ThinkAfford), 2605.11616v1 (AFFORDMEM), 2511.11702v1 (TASA), 2603.23478v2 (UniFunc3D, first 100k of 127k characters), 2503.07909v2 (FunGraph), 2511.23230v2 (SynthFun3D, first 100k of 130k characters), 2503.19199 (OpenFunGraph), 2505.24103 (WSAG-PLSP, first 100k of 129k characters), 2309.02120v1 (EPIC-Aff).
**PDF:** SceneFun3D CVPR 2024 open-access PDF (WebFetch returned HTTP 403; downloaded with curl into its own folder, read with pdftotext: task definitions, dataset statistics, baselines, result tables).

**Sources failed or not used:** CVF PDF via WebFetch (403); Google Scholar, Semantic Scholar, OpenReview and the SceneFun3D challenge page were not queried; GitHub repositories not checked.

### 3. Selection

Results from arXiv search, title lookups, web and Exa searches were merged by arXiv ID. A work was kept if it (a) produces affordance or functional-part outputs on scenes, frames or video, or (b) is a component that the SceneFun3D methods use (Molmo, SAM 3), or (c) supplies a training or contact signal from egocentric video. Within the SceneFun3D line every 2024-2026 method found was kept. In the 2D line, I kept the AGD20K anchors (Cross-View-AG, LOCATE), one 2025 table to fix one protocol (WSAG-PLSP), and VLM-based methods from 2024-2026; older object-level datasets were dropped. Hit counts were not counted.

### 4. Key evidence

| ID | Claim | Primary source | Supports / does not support |
| --- | --- | --- | --- |
| E1 | A query-agnostic, part-level, action-typed task already exists and has trained baselines on laser scans (Mask3D-F AP25 26.6, test) | SceneFun3D, CVPR 2024, Sec. 3 and Table 2 | Does not support calling a query-agnostic action-typed part map new by itself |
| E2 | The best query-agnostic number found is AP25 47.6 (test) on scans, with 67.2% any-candidate coverage at IoU 0.5 on val | TopoField, arXiv 2610.37218 | Supports "recall of candidates is the limiting quantity, even with scans" |
| E3 | Zero-shot detectors are near zero on SceneFun3D functional elements in 2D (GroundingDINO mAP50 0.7, YOLO-World 0.0); a trained RT-DETR reaches 26.7 | FunGraph, Table I | Supports "a writer from frames needs training or a stronger prompting scheme"; does not test SAM 3 or Molmo |
| E4 | A training-free writer from posed RGB-D finds 71.0% of interactive elements at R@3 with any non-zero IoU | OpenFunGraph, node evaluation | Supports feasibility of a query-free writer; metric too loose to show part-accurate recall |
| E5 | Per-query methods cost 37.6 s (TASA, A800), 78 s (T-FunS3D, A40), 53-99 s (UniFunc3D, 8 H100), 118-332 s (Fun3DU, by source) per query | TASA Table 2; T-FunS3D Table III; UniFunc3D Table 7 | Supports an efficiency target for a write-once index on the same benchmark |
| E6 | A proposal stage without the instruction (but given the affordance type) covers 77.5% of queries at 3D IoU 0.25 and 50.6% at 0.5 | ThinkAfford, Table 3 | Supports "write for recall, read for precision" as viable; coverage measured under an oracle best-view union |
| E7 | Fun3DU's AP25 on SceneFun3D is 33.3 as reported, 23.4 and 12.56 in two later reproductions | AFFORDMEM / UniFunc3D tables; T-FunS3D Table I; ThinkAfford Table 1 | Supports re-running baselines under one protocol; cross-paper numbers are not comparable |
| E8 | Once the target part is named, VLMs choose the right action much more often (0.158-0.474 -> 0.684-0.947) | Part Grounding, Not Action Knowledge, abstract | Supports "action typing is easier than part recall" (C-grade, 19 objects) |
| E9 | An egocentric, query-agnostic, multi-label action map can be built and accumulated in 3D (42.3 mIoU, 20 classes) but its regions are hotspots, not parts | EPIC-Aff, ICCV 2023 | Supports a precedent for action-typed egocentric maps; does not cover part granularity |
| E10 | Mask3D-FM loses little AP25 when motion type must also be right (26.6 -> 23.8), and much when the axis must be right (9.8) | SceneFun3D, motion table | Supports adding motion type at write time; axis is a separate, harder problem |
| E11 | A write-once object-level graph from egocentric video answers questions with accuracy competitive with frontier VLMs at lower latency | VL-KnG, abstract | Supports the system-level comparison idea at object level only (C-grade) |

### 5. Not done

- Did not read full texts end to end (no A grades); B grades are targeted extractions of method and result sections through a fetch tool, which may miss caveats in parts not asked about.
- Did not read UniFunc3D, SynthFun3D, WSAG-PLSP beyond the first 100k characters of their HTML.
- Did not download any dataset, model or code; did not run anything; did not check dataset licences or access beyond prior project records.
- Did not find or read the ECCV 2026 SceneFun3D leaderboard or first-place report.
- Did not verify venues for OOAL, INTRA, RoboPoint, OPDMulti, AFF-ttention, Interaction Hotspots, Molmo.
- Did not read object-level 3D affordance works (LASO, OpenAD, PAVLM, 3D-AffordanceLLM, SeqAffordSplat, VoxAfford, HAMMER) beyond titles; they are excluded and not listed as evidence.
- Did not query Semantic Scholar, Google Scholar or OpenReview.

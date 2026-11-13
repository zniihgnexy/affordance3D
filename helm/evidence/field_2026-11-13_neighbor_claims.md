# Neighbour audit: R8 claims CA, CB, CC, CD

Date: 2026-11-13. Scout: neighbor-scout. Inputs read: `helm/rounds/R8.md`, `helm/CHARTER.md`, `docs/mem/mem分支_文献数据仿真调研_2026-11-10.md`, `docs/research/跨视角Affordance检索_文献调研与立项评估_2026-11-02.md`, plus the T-FunS3D/TASA notes in `helm/LEDGER.md`, `helm/REVIEW.md` and `helm/rounds/R7.md`.

**Reading grades (the caller's scale).** A = full text read. B = method section or experiment tables read (excerpts through HTML or PDF). C = abstract or project page only. No paper was read in full in this session, so there are **no A grades**. A grade C reading cannot show that a capability is absent from a paper.

**Verdict scale.** The caller's scale (new / partly taken / taken) is given first. The scout scale follows in brackets: already done / weak / baseline material / open, originality unconfirmed / test first.

"Nothing found" in this file means nothing was found by the queries listed at the end, on 2026-11-13.

---

## 1. Claim audit

### CA: a new method that builds part-level affordance maps with action types from egocentric or handheld video, written once before any question

| Nearest work (first posted) | What it already does | Grade |
|---|---|---|
| FunGraph, arXiv 2503.07909 (2025-03-10, IROS 2025) | Input is posed RGB-D frames. A 2D detector of functional elements is trained on **SceneFun3D annotations projected into the iPad frames**, with the 9 SceneFun3D affordance labels (hook pull, rotate, key press…). A VLM refines the labels. Detections are fused into a 3D scene graph with object→part "has-part" edges carrying the affordance label. The graph is built once, and GPT-4o searches its JSON for each task. | B |
| OP3DSG, arXiv 2606.29786 (2026-06-29, ECCV 2026) | Input is a posed RGB-D sequence. Per-frame part detection is guided by an object→part knowledge base, followed by part-aware 3D fusion. An LLM agent assigns one of 9 affordance labels to each node, and the graph also has functional relations. Evaluated on its UniGraph3D benchmark (FunGraph3D plus post-processed SceneFun3D). It does not predict motion. | B |
| KeySG, arXiv 2510.01049 (2025-10-01, preprint) | Input is posed RGB-D. Keyframes are chosen by clustering camera poses. A VLM proposes functional-element tags from the keyframes, and LLMDet with SAM segments them as sub-nodes of objects. There is a hierarchical multimodal RAG over keyframes. It has no explicit store of action labels. Evaluated on FunGraph3D. | B |
| Interaction-derived maps: EPIC-Aff (arXiv 2309.02120, 2023); FunRec (arXiv 2604.05621, CVPR 2026); Ego-SAG (arXiv 2409.19650) | Affordance labels or masks extracted from hand interaction in egocentric video. FunRec mentions "hand-guided affordance mapping" as an application. These maps are at the level of movable parts or regions, not knobs and buttons at SceneFun3D scale. | C |

**What remains distinct:** (i) the map stays in keyframe image space (keyframe flags plus part tokens) with no fused 3D map; (ii) an objective of coverage and recall under a memory budget, rather than 3D segmentation accuracy. Neither is a construction *method*. All three routes R8 names for a new method have a neighbour. Multi-frame aggregation: FunGraph, OP3DSG, KeySG. Hand-contact cues: EPIC-Aff, FunRec, ArtiSG (project note). Labelling with a VLM and then organising: FunGraph label refinement, the OP3DSG affordance agent. The most natural trained writer, a 2D part-plus-affordance detector trained on SceneFun3D projections, is FunGraph's recipe.

**Verdict:** partly taken [weak]. It can become new only once a specific mechanism exists that is absent from the three graph papers.

**Evidence needed:** on the same frames, part recall and action-type accuracy against a FunGraph-style detector (whose code is public) and against OP3DSG and KeySG. Recall must reach the level that the planned miss-and-false-detection simulation says is needed.

### CB: search by function, using a memory indexed by the affordance maps (keyframe flags, part tokens, action types) and searched for each question, instead of a VLM reading the whole video

| Nearest work | What it already does | Grade |
|---|---|---|
| FunGraph (2025-03) | A function-aware memory of part nodes with an affordance attribute, written once and searched per task by an LLM. On SceneFun3D the authors argue that a scene graph is a better representation than retrieving directly from the point cloud. | B |
| KeySG (2025-10) | Keyframes plus functional-element nodes, with hierarchical retrieval per query, then an LLM. | B |
| T-FunS3D, arXiv 2606.05975 (2026-06-04, ICRA 2026) | A scene graph built once per scene from the point cloud and posed RGB-D, with the **top-k views of each instance cached**. For each task a VLM (Molmo with SAM) finds the part. It is object-level indexed, not action-typed. | B |
| AFFORDMEM, arXiv 2605.11616 (2026-05-12); Affordance RAG, arXiv 2512.18987 (2025-12-22, RA-L) | AFFORDMEM keeps a cross-scene exemplar bank keyed by interaction label and an in-scene graph. Affordance RAG keeps an object-level pick/place affordance memory with hierarchical retrieval. | B / C (A in the 2026-11-02 project note) |

**What remains distinct:** the action type serves as a retrieval key at part granularity over keyframes. There is also a controlled finding: at equal budget, affordance tokens beat generic region tokens (gate G1). These queries, and the earlier reviewer search in REVIEW.md, found no published equivalent of that finding.

**Verdict:** partly taken [weak as a general idea; the controlled finding is open, originality unconfirmed].

**Evidence needed:** with *predicted* maps, an ablation that removes the action-type key, and a comparison with FunGraph and T-FunS3D on the same frames.

### CC: more accurate and faster than a VLM that reads the raw video for each question, counting total cost over many questions on one scene

| Nearest work | What it already does | Grade |
|---|---|---|
| Video-EMem "Write Once, Query Many", arXiv 2610.24537 (2026-10-17) | A query-independent writer and a per-query reader on generic long video. It beats Qwen3-VL-30B given 2,048 frames by +2.6 on average and has about 19× lower per-query latency than an agentic method. | B |
| VL-KnG, arXiv 2510.01483 (2025-10-01) | A persistent graph built once from monocular egocentric video. Its accuracy is competitive with frontier VLMs, and query latency is decoupled from video length. | C |
| MemTree3D, arXiv 2608.18009 (2026-08-18) | Reusable memory that picks keyframes for 3D QA without searching every frame per query. It improves over uniform-sampling GPT-4o by 17.4% on OpenEQA. | B |
| T-FunS3D (2026-06) on SceneFun3D | Built once at about 592 s per scene, then 78 s per query. Fun3DU needs about 1,920 s plus 167 s per query (A40). | B |
| UniFunc3D, arXiv 2603.23478 (2026-03-24, NeurIPS 2026) | Effectively the "VLM reads the video" opponent on SceneFun3D: Qwen3-VL, 64 frames × 4 passes per query. It is state of the art among training-free methods. | B |

**What remains distinct:** only the domain result, i.e. part-level functional grounding plus action answers on SceneFun3D, against a raw-video VLM and against T-FunS3D, with a cost curve that includes the write cost.

**Verdict:** partly taken [the paradigm is already done; the domain result is baseline material on its own].

**Evidence needed:** a cost-versus-number-of-questions curve that crosses below both the raw-video VLM and T-FunS3D, using the same answer model.

### CD: especially strong on action-related questions

**(A) Questions that name an action.**
- "Part Grounding, Not Action Knowledge", arXiv 2609.13225 (2026-09). Grade B. VLM affordance errors come from **finding the part**. Naming the part raises action accuracy by 0.32–0.63 across 8 models. This *predicts* CD-A's mechanism; it does not test a memory.
- GroundBench, arXiv 2609.13308 (2026-09-10). Grade C.
- AffordBot, arXiv 2511.10017 (NeurIPS 2025). Grade B.
- SceneFun3D, whose tasks carry the 9 labels. Grade B.

No work was found that reports accuracy per action type for a functional memory against a raw-video VLM.

Verdict: new [open, originality unconfirmed, and **test first**: split the existing 390 tasks by the ground-truth affordance label].

**(C) Answers that can be acted on (where to touch, which direction to move).**
- AffordBot (2025-11). Grade B. For each instruction it outputs a mask, a motion type and a discretised axis, on SceneFun3D.
- AFUN, arXiv 2606.02551 (2026-06-01). Grade B. Outputs a mask, a contact point and a 3D post-contact motion curve.
- UniRAP, arXiv 2610.29768 (2026-10). Grade C. Outputs a part affordance and a 4-DoF interaction pose.
- AssistQ, arXiv 2203.04203 (2022). Grade C. Button-level answers.
- GroundBench reports that giving a VLM the joint type and axis never improved its action accuracy.

Verdict: partly taken [weak]. What remains distinct is serving the motion type and direction from a write-once memory.

**(B) Past actions in long video.**
- AMEGO (ECCV 2024). Grade B. A query-agnostic memory of hand-object interactions that answers more than 20K queries without reprocessing.
- EgoSG, arXiv 2606.25842 (2026-06-24). Grade C. Evaluated on HD-EPIC VQA.
- Imprint, arXiv 2607.00696 (2026-07-01). Grade C.
- Embodied VideoAgent, GESTO, EGG. Grade C, from the project note of 2026-11-10.

Verdict: taken at object and fixture level [already done]. At part level it is open, but no dataset exists: HD-EPIC is fixture-level, and SceneFun3D contains no human actions.

---

## 2. What cannot be claimed (and the accurate wording)

| Avoid | Neighbour | Accurate wording |
|---|---|---|
| "first affordance / functional memory" | FunGraph, OP3DSG, KeySG, Affordance RAG, AFFORDMEM | "a part-level memory indexed by action type over keyframes, written once from posed video" |
| "first to build the scene once and answer many functional queries" | T-FunS3D | "…and, unlike T-FunS3D, it keeps no fused 3D map and reads k keyframes per question" (only if the cost comparison holds) |
| "memory beats a VLM reading the whole video while being faster" (as a general finding) | Video-EMem, VL-KnG, MemTree3D | "on part-level functional grounding (SceneFun3D), at the same answer model…" |
| "VLMs lack action knowledge" | 2609.13225 | "VLMs fail at finding the part; a part index supplies it" |
| "first to answer with motion type and direction" | AffordBot, AFUN, SceneFun3D Task 3 | "serves motion type and direction from memory without per-query 3D processing" |
| "first memory of past actions in egocentric video" | AMEGO, EgoSG, Embodied VideoAgent | drop it, or restrict to part level once data exists |
| "no point cloud needed" as the distinction | R7 / V-29 (the inputs fuse into a point cloud in about 2 s) | "keeps no fused map; the memory is keyframes plus part tokens" |

## 3. Crowding: works from the last 12 months, newest first

| Date | Work | Covers | Leaves uncovered |
|---|---|---|---|
| 2026-10-27 | TopoField (ECCV 2026 SceneFun3D challenge, arXiv 2610.37218), C | point-cloud functionality segmentation, AP 28.3 | video, memory, queries |
| 2026-10 | UniRAP (2610.29768), C | part affordance and interaction pose, per image | memory, scene scale |
| 2026-10-17 | Video-EMem (2610.24537), B | write once, query many; beats a raw-video VLM on accuracy and latency | part level, action, 3D scenes |
| 2026-09-14 | MessyMem (2609.15976, CoRL 2026), C | 3DSG plus properties and outcomes learned by interacting, plus keyframes | built from robot interaction, not passive video; robot tasks |
| 2026-09 | Part Grounding (2609.13225), B; GroundBench (2609.13308), C | the VLM failure on affordance is part grounding | memory, video |
| 2026-08-18 | MemTree3D (2608.18009), B | reusable memory picks keyframes for 3D QA | functional parts, action types |
| 2026-07-01 | Imprint (2607.00696), C | interaction records in long egocentric QA | parts, 3D |
| 2026-06-29 | OP3DSG (2606.29786), B | part nodes with 9 affordance labels, from posed RGB-D | motion, QA against a VLM, keyframe evidence |
| 2026-06-29 | Analytic concept-centric memory (2606.29774), C | part, affordance and transition memory for manipulation (robotics) | passive video; it is a robot paper |
| 2026-06-24 | EgoSG (2606.25842), C | temporal scene graphs on HD-EPIC VQA | parts, affordance |
| 2026-06-04 | T-FunS3D (2606.05975), B | build once on SceneFun3D, faster per query | action-typed index, comparison with a raw-video VLM |
| 2026-06-01 | AFUN (2606.02551), B | where and how (mask, contact point, motion curve) from one RGB-D view | memory, multi-view, video |
| 2026-05-12 | AFFORDMEM (2605.11616), B | exemplar memory plus scene graph for SceneFun3D | per-query processing, action index |
| 2026-03-24 | UniFunc3D (2603.23478), B | raw-video VLM grounding, state of the art among training-free methods | reuse across queries, cost |
| 2025-12-22 | Affordance RAG (2512.18987), C | object-level affordance memory and retrieval | parts |

## 4. Distinctness from the sibling project

Per CHARTER §3, the sibling `affordance-vla` trains VLA policies. The project memory index in context also describes it as feeding an episode-level affordance state to a VLA. I did not access that repository.

- **CA** overlaps with the sibling if the same video-to-affordance-map builder also produces the sibling's affordance state. If CA is the method contribution of both papers, one branch becomes a variant of the other. The separation is in the output and the evaluation: this paper scores the map by memory recall and retrieval or QA accuracy, never by policy success.
- **CD-C** (where to touch, which direction) is the claim closest to the sibling. Its output is an action-ready interface, which is exactly what a VLA would consume, and it sits near the fence line "training VLA or robot control is outside". Keep it as an answer *format*, scored against annotations (SceneFun3D motion direction, part extent), with no executed actions.
- **CB, CC, CD-A and CD-B** are memory-side and do not overlap with VLA training.

## 5. Standing on its own (application)

**What the system lets someone do.** Walk through a home once with a handheld camera. Afterwards, ask many questions such as "how do I turn on the oven light" or "which knob controls the back burner". Each answer comes back as the frame, the part to touch and the kind of motion, after a small VLM looks at a few frames rather than the whole recording.

**Closest systems, and what each cannot do:**
- **T-FunS3D:** needs a point cloud and runs a VLM part search on each query (about 78 s). Its index is object-level, not action-typed.
- **FunGraph and OP3DSG:** fuse everything into a 3D graph, return no keyframe evidence, give no motion direction, and are not compared with a raw-video VLM.
- **UniFunc3D:** reprocesses the video for every query.
- **AssistQ:** works on instructional videos, not the user's own space.
- **Video-EMem:** handles generic long video and has no part or action grounding.

**CA as an algorithm** cannot yet be stated in two sentences without naming a skeleton, because the mechanism is not designed. Today it leans on the FunGraph recipe (a detector trained on projected SceneFun3D labels, then fusion). With that recipe swapped out, only the keyframe-space storage choice remains.

## 6. Most defensible combination and the result that would strengthen it

The most defensible combination is CB + CD-A, with CC as the cost axis. Stated as one controlled finding: a part-level index keyed by action type and written once from video lets a small VLM read only k keyframes; the advantage over a raw-video VLM is concentrated on questions that name an action; and the amortized cost falls below both the raw-video VLM and T-FunS3D after N questions per scene. Recent results point the same way: 2609.13225 locates the VLM bottleneck in finding the part, which is what the index supplies.

CA alone is the weakest part. CD-B and CD-C are respectively taken, or partly taken and drifting toward the sibling.

**Single result that would strengthen it most:** with *predicted* (non-oracle) maps on the SceneFun3D validation split, per-action-type accuracy against the Qwen3-VL raw-video baseline. The gain should be clearly larger on action-named tasks than on the rest, and removing the action-type key should remove that extra gain. If the gain only appears with oracle maps, the paper rests on CA, the least defensible claim.

The cheap pre-check is CPU-only: split the existing results of run c and of P4 by ground-truth affordance label.

---

## 7. Search record

**Sources used:** WebSearch (standard and extended), Exa web search, WebFetch on arXiv abstract and HTML pages. Exa and WebFetch returned highlights and summaries produced by a summariser, which is why method-level readings are graded B, not A.

**Failed:** none during the search. The Codex MCP server was not connected; it was not needed.

**Queries as typed:**

WebSearch:
1. `egocentric video memory indexed by affordance part-level retrieval question answering 2026` (extended)
2. `"affordance" memory retrieval keyframes "instead of" video-language model full video functional query efficiency arXiv 2026` (standard)

Exa:
1. `paper that builds a functional affordance memory from egocentric or handheld video once and retrieves keyframes and interactive parts (handles, knobs, buttons) for language queries, compared against a VLM that reads the whole video`
2. `SceneFun3D functionality segmentation new method 2026 arXiv video frames training-free motion type prediction`
3. `detect interactive functional elements (handles, knobs, buttons, switches) with affordance labels in every frame of an indoor RGB video and build a scene-level affordance map, query-agnostic`
4. `vision-language model predicts motion type and motion axis direction and contact point for functional parts in 3D indoor scenes, actionable affordance grounding`
5. `memory-based retrieval versus feeding the whole long video to a VLM: accuracy and amortized cost when many questions are asked about the same video, build index once`
6. `memory of interactive parts and their affordances for embodied question answering, retrieve keyframes by function (what can be pulled, pressed, turned), 2026 arXiv`
7. `benchmark of questions about past hand-object interactions in long egocentric video: which drawer did I open, which switch did I press, fixture interaction, object movement`
8. `VLMs fail at fine-grained functional part grounding in video for action verbs like press, rotate, pull; per-affordance-type breakdown of grounding accuracy`
9. `AR assistant answers how to operate appliances in a home (which knob to turn, where to press) from a previously recorded walkthrough video using a stored affordance memory`
10. `AFUN: Towards an Affordance Foundation Model for Functionality Understanding arXiv`
11. `training-free functionality segmentation SceneFun3D from posed RGB video without point cloud, reusable per-scene memory, runtime per task`

WebFetch (arXiv abs/html): 2606.05975, 2605.11616, 2603.23478, 2609.15976, 2503.07909, 2606.29786 (abs and html), 2510.01049 (abs and html), 2512.18987, 2001.04583, 2606.25842, 2606.29774, 2610.24537, 2607.00696.

**Selection:** works were kept if they (a) build part-level or affordance-labelled memory from posed video, (b) compare reusable memory with per-query VLM reading, (c) output action or motion for parts, or (d) answer past-interaction questions over long egocentric video. Pure single-object affordance segmentation work (VideoAfford, VLAff, Li 2024) was noted but not audited.

**Not checked:**
- No paper was read in full (no A grades).
- Not searched: Google Scholar, Semantic Scholar, OpenReview (NeurIPS 2026, ICLR 2027 submissions), CVF or ECVA 2026 proceedings, or other ECCV 2026 SceneFun3D challenge reports besides TopoField.
- ReToken was seen only through a blog post and is not cited.
- Embodied VideoAgent, GESTO and EGG were taken from the 2026-11-10 project note, not re-read.
- OP3DSG's efficiency appendix was not read.
- For T-FunS3D, two runtime tables were seen (arXiv: 78 s per query; IROS-workshop PDF: 150 s per query); the version difference was not resolved.
- Whether AffordBot or UniFunc3D report results per affordance class was not checked.

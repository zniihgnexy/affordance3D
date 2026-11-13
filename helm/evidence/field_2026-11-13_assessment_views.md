# Outside views for the round-10 assessment (2026-11-13)

Two independent agents, dispatched together: a venue reviewer (files only, no access to the helm's reasoning) and a neighbour scout (three candidate central claims, last eighteen months). Their issues are registered in `helm/REVIEW.md` (V-64 to V-73). Reading levels: abstract / method / tables / full; nothing below was read in full. Figures the scouts reported but the helm did not check are marked "not checked by the helm".

## Venue reviewer (NeurIPS 2027 main track, use-inspired)

- Verdict: below the bar at the validate stage. NeurIPS 2027 guidance not published; the 2026 Use-Inspired guidance (neurips.cc/Conferences/2026/ReviewerGuidelines, read 2026-11-13) asks whether the use case is real and meaningful, for task, metrics and methods that fit it, for comparison "against any commonly used non-ML approaches as well as baseline ML methods", and for analysis of why the choices work.
- It read the repository as it stood before the 8B record (§24.8) and the action-key record (§24.10) were pushed.

## Neighbour scout: three candidate claims

| Claim | Nearest works (reading level) | Verdict |
|---|---|---|
| K1: a part map written once, shown to a VLM as numbered boxes on a few retrieved frames | Set-of-Mark 2310.11441 and MOKA 2403.03174 (method); AffordBot 2511.10017, NeurIPS 2025 (method): numbered candidate boxes tagged with affordance type on rendered views, needs a point cloud; ThinkAfford 2608.10981 (method): numbered proposals on video views, per query, trained; 3D-Mem 2411.17735, CVPR 2025 (tables): a few retrieved snapshots beat a multi-frame VLM, object level; VL-KnG 2510.01483 (tables): pays back its build after about 10 queries | weak: the gain comes from numbered part candidates, a known mechanism; the cost argument exists at object level |
| K2: a builder that finds more parts per unit of write cost | Functional-SLAM 2609.07497 (tables): recall at 3 and FPS reported as two points; a keyframe-only variant 2.45× faster with lower recall; HHOpenFunGraph 2605.15753 (tables): element recall 70.1 on SceneFun3D, 58.2 on FunGraph3D (metric definition and poses not checked by the helm; not comparable with our per-part recall as is); OpenFunGraph 2503.19199 (tables) | weak |
| K3: a functional memory beats semantic retrieval more as the record gets longer or holds more similar parts | CapMem 2609.17688 and MERIT 2608.07663 (tables): memory gains grow with video length in generic video QA; 2610.31284 (tables): retrieval distractors from repeated rooms and objects; AffordMem 2605.11616 (method); OpenGround 2512.23020 (tables): grounders given true candidates collapse when many same-label parts are present | test first: the part-level version was not found (not an exhaustive search) |

Benchmarks the scout reported for K3 (not checked by the helm): OpenTarget (2512.23020): 7,724 queries, 70 part classes, 8.35 same-label instances per scene, built on ScanNet++ and Articulate3D, single rooms, no splits; FunGraph3D: 14 scenes, 201 interactive elements; Articulate3D: 280 scenes. HD-EPIC figures are in `field_2026-11-13_feasibility_B_C.md`.

Related work to watch: SpatialMem 2601.14895 (object-level egocentric memory, evidence-frame retrieval at an equal frame budget, no parts).

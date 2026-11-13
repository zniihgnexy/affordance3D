# Landscape: "traditional VLM" baselines for horizontal alignment (SceneFun3D-type grounding, HD-EPIC VQA, long egocentric video)

Scout record, 2026-11-13. Targeted scoping review, not exhaustive. Cutoff: what was reachable on 2026-11-13.
Three things kept apart below: **[paper]** = what a paper states; **[synthesis]** = my reading across papers; **[to test]** = something we would have to run ourselves.

Reading grades. **A**: method and experiment sections (or the relevant table plus its protocol paragraph) read. **B**: abstract or project page only. **C**: only a search snippet or table excerpt returned by a search tool; numbers not checked against the paper's own page. Every reading went through a fetch tool that converts the page; the numbers below were requested verbatim and, where noted, cross-checked across two or three papers. None of these grades means reproduced.

---

## Part 1 — Assessment

### 1. Verdict and positioning

- **SceneFun3D (tasks A/C).** [paper] No surveyed paper has a plain "VLM reads uniformly sampled frames, then answers with a frame and a box" baseline as a main-table row. Every method uses a VLM *inside* a pipeline (Molmo-7B, Qwen3-VL-8B/30B/32B, Qwen2.5-VL-72B) and reports 3D mask metrics. The nearest plain-VLM rows are ablations: UniFunc3D's one-stage Qwen3-VL-8B (+SAM3) and AffordMem's frame-by-frame Qwen3-VL (+SAM3) diagnostic. [synthesis] Our 2D proxy ("chosen frame and box land on the part") has no published counterpart. Horizontal alignment on SceneFun3D therefore means lifting to 3D and reporting AP50/AP25/mIoU on split0 (the 30 validation scenes), and also copying the runtime per query that three papers report.
- **HD-EPIC VQA (task B).** [paper] Plain "VLM reads the video" rows exist on the full 26.6K-question set: Gemini 1.5 Pro, LLaVA-Video-7B, LongVA-7B, VideoLLaMA2-7B, Qwen2.5-VL-7B/32B, with per-prototype numbers for Fixture Interaction Counting and the three Object Motion prototypes. No plain Qwen3-VL row is published. The 2026 challenge entry that uses a frozen Qwen3-VL-8B (EgoAdapt, 67.22) wraps it in a heavy inference pipeline and gives no plain baseline.
- **Long video in general.** [synthesis] In 2026 papers the default open-weight baseline is Qwen3-VL-8B-Instruct, with InternVL3.5-8B second. Qwen2.5-VL-7B and LLaVA-Video-7B remain as legacy anchors. Commercial references are Gemini 2.5 Pro and GPT-5. Frame budgets vary widely, from 32 to 768 frames. No surveyed paper states a dollar cost.

### 2. Terms (used the same way throughout)

| Term | Meaning here |
| --- | --- |
| plain VLM baseline | Per question, a VLM receives uniformly sampled raw frames (or the video) plus the question and answers directly. No stored index or memory, and no extra detector except a segmenter that turns its point or box into a mask. |
| pipeline VLM | A VLM used as one component of a multi-stage method (task parsing, pointing, verification). |
| split0 / split1 | SceneFun3D evaluation splits used since Fun3DU: split0 = 30 validation scenes; split1 = 200 training scenes. |
| protocol | The question set, frame budget, resolution and evaluator behind one number. Numbers are compared only within one protocol. |
| FIC / FL / OL / OCR | HD-EPIC 3D-Perception prototypes: Fixture Interaction Counting, Fixture Location, Object Location, Object Contents Retrieval. |
| OMI / OMC / SOL | HD-EPIC Object-Motion prototypes: Object Movement Itinerary, Object Movement Counting, Stationary Object Localization. |

### 3. Full record

#### 3a. SceneFun3D-type task-driven functional part grounding

Metric columns are AP50 / AP25 / mIoU on **split0** unless stated otherwise.

| paper | benchmark | models and sizes (role) | frames / resolution | numbers | open / commercial | grade |
| --- | --- | --- | --- | --- | --- | --- |
| **Fun3DU** (arXiv 2411.16310, Nov 2024; CVPR 2025) | SceneFun3D split0, split1 | Llama-3.1-9B 4-bit (task parsing); OWLv2 base + RobustSAM (context object); Molmo-7B-D-0924 (points to the part); SAM | ~1,800 RGB-D frames per scene; scores all views, keeps top V̂ = 50 | split0 own: mAP 7.6, AP50 16.9, AP25 33.3, mIoU 15.2. split1: 6.1 / 12.6 / 23.1 / 11.5. Ablation "w/o VLM" (OWLv2 only): AP50 4.5, AP25 13.9, mIoU 6.3. Time per task on one A100, excluding the LLM and context steps: 118.4 s at V̂=50 (28.4 s at V̂=2; 237.6 s at V̂=200) | all open | A |
| **TASA** (arXiv 2511.11702, 2025-11-12; AAAI 2026) | SceneFun3D (split not labelled in Table 1) | "Qwen" (size not given in the text read) for concept extraction and checks; Molmo + SAM; CLIP for frame selection; trained 3D refinement (Point Transformer) | top-K = 10 frames of ~100 high-res RGB-D images | mAP 23.2, AP50 26.9, AP25 28.6, mIoU 19.7. Time per task on A800: 37.61 s vs Fun3DU 130.26 s | open | A |
| **AffordBot** (arXiv 2511.10017, 2025-11-13; NeurIPS 2025) | SceneFun3D (split ambiguous, see note) | Qwen2.5-VL-72B on 4× A800, CoT prompting; Mask3D fine-tuned | 8 rendered surround views, 680×680, from the point cloud (no raw video) | own table: mIoU 14.0, AP 15.5, AP50 20.0, AP25 23.3. MLLM swap inside AffordBot (AP/AP50/AP25): LLaVA-v1.6-34B 10.6/14.2/16.9; Qwen2.5-VL-72B 15.5/20.0/23.3; GPT-4o 16.5/22.1/28.9; GPT-o1 24.8/30.3/33.4. No runtime or cost given | open; GPT-4o and o1 only in the ablation | A |
| **AffordMem** (arXiv 2605.11616, 2026-05-12; venue not stated) | SceneFun3D split0, split1 | Qwen3-VL-32B (parsing, grounding, selection); SAM3 | memory recalls K = 20 masked exemplar frames; per-query frame count not stated | split0: AP50 20.13, AP25 41.66, mIoU 17.32; split1: 16.3 / 32.7 / 14.0. **Plain-ish diagnostic, Fun3DU-style Qwen3-VL + SAM3 frame by frame: AP50 7.14, AP25 17.86.** Direct SAM3 text prompting: AP50 14.80, AP25 30.45. One A100 80 GB; no runtime given | open | A |
| **T-FunS3D** (arXiv 2606.05975, 2026-06-04; ICRA 2026 per project page and poster) | SceneFun3D val (split0), train (split1) | Qwen3-14B, thinking off (task parsing); FG-CLIP; Molmo-7B-D; SAM; Mask3D | 1 in 5 video frames; top-5 views per instance | split0: mAP 8.1, AP50 17.8, AP25 34.5, mIoU 15.7. Its reproduction of Fun3DU: 4.4 / 10.3 / 23.4 / 12.0. split1: 7.1 / 13.6 / 24.9 / 11.7. **Per query on A40: T-FunS3D 78 s, Fun3DU 167 s.** One-time scene preprocessing: 12 s + 580 s | open | A |
| **UniFunc3D** (arXiv 2603.23478, first posted 2026-03-24, v2 2026-10-05; NeurIPS 2026 per project page) | SceneFun3D split0, split1 | Qwen3-VL-8B or Qwen3-VL-30B-A3B (single MLLM for grounding and verification); SAM3. Qwen2.5-VL-72B in an ablation | Round 1: N = 64 frames at 512×384, K = 4 sampling iterations; Round 2: a window of 2Δ+1 frames at native resolution | split0: 8B 23.82 / 44.04 / 20.92; 30B 31.24 / 51.01 / 24.30. split1: 8B 16.24 / 29.02 / 14.23; 30B 21.32 / 35.76 / 17.09. **One-stage plain Qwen3-VL-8B + SAM3, split0: high-res AP50 4.27, AP25 7.64, mIoU 3.11; low-res 7.42 / 14.38 / 7.35** (frame count not stated). **Runtime per query, 8× H100 platform: Fun3DU ~332 s, UniFunc3D-8B ~53 s, UniFunc3D-30B ~99 s; per scene ~82 / ~16 / ~26 min.** "Can still run on a single H100" | open | A |
| **ThinkAfford** (arXiv 2608.10981, 2026-08-11; venue not stated) | SceneFun3D val, "official evaluator" | Qwen-VL 8B (version stated inconsistently in the paper); APG trained; VPAR trained with GRPO | K = 25 views | AP 4.42, AP50 10.69, AP25 25.46. **Its reproductions under the official evaluator: Fun3DU 1.66 / 4.55 / 12.56 (AP/AP50/AP25); AffordBot 3.90 / 6.37 / 10.70; TASA 0.05 / 0.22 / 1.50** | open | A (table and setup) |
| **SceneFunRI** (arXiv 2605.14704, 2026-05-14; benchmark) | 2D localization of *invisible* functional objects in SceneFun3D frames (855 instances) | Gemini 3 Flash; Qwen3.5 0.8B–122B; InternVL3/3.5 1B–38B; Gemma 3 4B–27B; spatial models (MultiHopSpatial 4B, SenseNova-SI 8B, RoboBrain2.5 8B) | **single RGB image** plus instruction; resolution not stated | CAcc@75: Gemini 3 Flash 15.20; Qwen3.5-4B 10.99; Qwen3.5-9B 11.11; Qwen3.5-122B 12.63; InternVL3-8B 4.09; gemma-3-27b 7.60; human 66.43 | open and commercial | A (table) |
| **TopoField** (arXiv 2610.37218, 2026-10-27; ECCV 2026 SceneFun3D Challenge, 2nd place) | SceneFun3D functional segmentation (no task text; point cloud) | point-cloud model (VOLT-based); no VLM | — | AP 28.286, AP50 41.383, AP25 47.558 (30 val scenes) | — | B |

Notes on 3a.
- [paper] UniFunc3D and AffordMem list "TASA-72B" and "AffordBot-72B" on split0: TASA 26.9 / 28.6 / 19.7; AffordBot 20.91 / 24.76 / 14.42 (AR50 18.99, AR25 22.84). The TASA text I read names only "Qwen", so the 72B attribution comes from the later papers.
- [synthesis] AffordBot's own table lists Fun3DU at 6.1 / 12.6 / 23.1 / 11.5, which are Fun3DU's **split1** numbers. Later papers say AffordBot was trained on split1 and list it only on split0, with different values. The split behind AffordBot's own table is unclear.
- [synthesis] The same method gets very different numbers in different papers. Fun3DU split0 AP50 is 16.9 in its own paper, 10.3 in T-FunS3D's reproduction and 4.55 under ThinkAfford's "official evaluator". Cross-paper SceneFun3D numbers are therefore not on one protocol. Compare within one table only, or rerun the baselines under our own evaluator.

#### 3b. HD-EPIC VQA

Category columns: Recipe / Ingredient / Nutrition / Fine-grained Action / 3D Perception / Object Motion / Gaze / Avg. FIC, OMI, OMC and SOL are prototypes (see Terms).

| paper | benchmark / protocol | models and sizes | frames / resolution | numbers | open / commercial | grade |
| --- | --- | --- | --- | --- | --- | --- |
| **HD-EPIC** (arXiv 2502.04144, 2025-02-06; CVPR 2025) | full VQA, 26,550 questions, 30 prototypes | Llama-3.2-90B-Vision (blind, text only); VideoLLaMA2-7B-16F; LongVA-7B-DPO; LLaVA-Video-7B-Qwen2; Gemini Pro = gemini-1.5-pro-001 | Gemini: 1 fps, 768×768, videos up to 6000 s at normal speed, longer ones sped up to fit 2M context. LongVA: 8 fps, ≤400 frames, 384×284. VideoLLaMA2: 16 frames, 336×336. LLaVA-Video: ≤128 frames, otherwise as LongVA (implied) | Table 2. VideoLLaMA2 30.8 25.7 32.7 27.2 25.7 28.5 21.2 / 27.4. LongVA 29.6 30.8 33.7 30.7 32.9 22.7 24.5 / 29.3. LLaVA-Video 36.3 33.5 38.7 43.0 27.3 18.9 29.3 / 32.4. Gemini Pro 60.5 46.2 34.7 39.6 32.5 20.8 28.7 / 37.6. Blind Gemini Pro 26.7; blind Llama 26.5. Human 90.3 (Motion 92.7). Abstract says Gemini Pro "only achieves 38.5%"; Table 2 says 37.6. Paper: context caching was "necessary … to keep costs tractable" (no dollar figure) | open; Gemini commercial | A |
| **HelloWorld** (arXiv 2601.10228, 2026-01-15; CVPR 2025 EgoVis workshop, 2nd place in the 2025 challenge) | full VQA | fine-tuned Qwen2.5-VL-7B-Instruct (theirs); baselines Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-32B-Instruct | theirs: ≤768 frames, 3136–846720 total pixels per video; baseline settings not stated | Qwen2.5-VL-7B 40.6 35.8 32.0 37.3 35.0 23.9 29.7 / 33.5. Qwen2.5-VL-32B 59.0 37.0 33.0 40.3 35.6 19.8 33.6 / 36.9. Theirs 41.6. **Per prototype, VL2 / LongVA / LLaVA-V / Gemini / Q7B / Q32B / theirs.** FIC 17.7 / 32.3 / 16.3 / 35.3 / 26.3 / 24.7 / 29.0. FL 18.8 / 26.6 / 21.8 / 20.8 / 27.0 / 26.2 / 34.2. OL 31.0 / 41.2 / 30.6 / 32.4 / 40.0 / 46.3 / 49.8. OMI 11.0 / 10.2 / 9.8 / 18.0 / 11.2 / 7.8 / 14.2. OMC 44.0 / 34.5 / 20.0 / 13.0 / 38.0 / 22.0 / 44.5. SOL 30.5 / 23.5 / 27.0 / 31.5 / 22.5 / 29.5 / 31.0. No GPU or runtime reported | open | A |
| **DeepFrames** (challenge report R1, arXiv 2506.08553; HD-EPIC 2025 challenge) | full VQA; ablations on 50 questions per prototype | Gemini 2.0 Flash (graph building and answering) | 1 fps, 480p; segments ≤400 s; videos standardized to a 2400 s window (sped up when longer) | Video-only Gemini 2.0 Flash: Recipe 40.75, Ingr. 45.17, Nutr. 33.67, Action 48.05, 3D 29.38, **Motion 17.53**, Gaze 30.45, Overall 35.00. SceneNet (scene graph, no video): 3D 41.84, **Motion 30.17**, Action 31.97, Overall 37.96. Final ensemble 44.21. EgoAdapt's table lists it as the best 2025 entry | commercial | A |
| **Ricoh** (challenge report R3, 2025) | full 26,550 questions, zero-shot | Qwen2.5-VL 7B / 32B / 72B; LLaVA-Video 7B / 72B | 72B limited to 10240×28×28 total pixels; frame-rate test at 1/2/5 fps with videos capped at 32 s | Avg: Qwen2.5-VL-7B 28.66; Qwen2.5-VL-32B+72B (split run) 32.14; LLaVA-Video-7B 30.19; LLaVA-Video-72B 32.34. "Inference speed is very slow" for 32B/72B on their hardware | open | A (short report, fully read) |
| **EgoSG / "Graph it first!"** (arXiv 2606.25842, v1 2026-06-24, v2 2026-07-02; venue not stated) | HD-EPIC subset: 1,250 questions, 5 categories (Nutrition and Gaze excluded), 50 per prototype | Gemini 2.0 Flash; Qwen2.5-VL 3B/7B; InternVL3 2B/8B/14B; VideoLLaMA3 2B/7B | 1 fps. Max frames: Gemini 3,900; Qwen2.5-VL 512; VideoLLaMA3 180; InternVL3 32. Resolution not stated | Columns Recipe / Ingr. / Action / 3D / Object Motion / Overall, raw video vs EgoSG. Gemini Flash 2.0: 51.75 / 34.00 / 46.00 / 39.00 / 26.67 / 39.48 → EgoSG 61.50 / 37.33 / 38.00 / 44.00 / 32.67 / 42.70. Qwen2.5-VL-7B: 37.25 / 30.67 / 32.00 / 25.50 / **38.00** / 32.68 → 46.25 / 27.00 / 34.50 / 37.00 / 34.67 / 35.88. InternVL3-8B: 35.00 / 29.00 / 40.00 / 31.00 / **20.00** / 31.00 → 42.25 / 18.67 / 33.00 / 38.50 / 34.00 / 33.28. **Cost: graph generation ≈5.7 s per 1-min clip, done once and amortized over ~22 questions per video; a single question can be slower than video-only inference** (runtime shown only as a figure) | open and commercial | A |
| **EgoReasoner** (arXiv 2603.06561, Mar 2026; venue not stated) | the 6 HD-EPIC "4D" prototypes (FIC, FL, OL, OMC, OMI, SOL); videos **clipped to 2–5 min** around the question | Gemini 2.5 Pro; Qwen2.5-VL 3B/7B; VideoLLaMA2/3; LongVA; LLaVA-Video; EgoThinker; VideoChat-R1; theirs (Qwen2.5-VL-3B with SFT and GRPO) | ≤128 frames at 256×256 (theirs; stated as applied to all baselines) | Avg / FIC / FL / OL / OMC / OMI / SOL. Gemini 2.5 Pro 49.2 / 36.3 / 40.4 / 63.6 / 45.5 / 60.0 / 49.5. Qwen2.5-VL-7B 25.7 / 22.7 / 22.6 / 34.0 / 33.0 / 20.2 / 22.0. Qwen2.5-VL-3B 27.0 / 31.0 / 18.4 / 37.8 / 39.0 / 16.2 / 19.5. VideoLLaMA3-7B 26.0. Theirs 37.5. Human 91.1 | open and commercial | A |
| **EgoAdapt** (arXiv 2605.24500, 2026-05-23; CVPR 2026 HD-EPIC challenge report) | full VQA; test or val split not stated | **Qwen3-VL-8B, frozen**, with category routing, option-likelihood scoring and test-time consistency (5 option permutations for ambiguous questions) | per-category frame budgets and fps; values not given | 88.00 / 68.17 / 71.67 / 64.03 / 64.87 / **61.63** / 52.15 / **67.22**. Without test-time adaptation: 3D 41.39, Motion 46.73, overall 57.18. **No plain Qwen3-VL-8B row.** 8× A100; no per-question time | open | A |
| **HIPPO** (arXiv 2605.29402, 2026-05-28; challenge solution) | full VQA | Gemini 3.1 Pro (offline semantic evidence, query parsing); WeDetect-Large-Uni detector; answering MLLM not named | coarse pass 0.1 fps on 600 s chunks; fine pass 1 fps on 60 s chunks; visual evidence at 1 fps | overall 65.8; FIC 46.0, OMI 49.0, OMC 51.0, SOL 58.0. Claims fewer input frames than direct sampling (not quantified). **Its "DeepFrames" column repeats HelloWorld's per-prototype numbers** (e.g. FIC 29.0, OMI 14.2), so that column is unreliable | commercial plus open detector | A |

Cross-check. The per-prototype baseline values for VideoLLaMA2, LongVA, LLaVA-Video and Gemini Pro (FIC, FL, OL, OMI, OMC, SOL) agree across HelloWorld, HIPPO and EgoReasoner. [synthesis] EgoReasoner copies those original full-video values into a table whose own rows use clipped videos, so its table mixes two protocols.

#### 3c. Long-video QA, retrieval and episodic memory: what 2026 papers use as plain-VLM baselines

| paper | benchmark | models and sizes | frames / resolution | numbers | open / commercial | grade |
| --- | --- | --- | --- | --- | --- | --- |
| **WorldMM** (arXiv 2512.02425, 2025-12-02; CVPR 2026) | EgoLifeQA, Ego-R1 Bench, HippoVlog, LVBench, Video-MME (L) | base models Qwen3-VL-8B, Gemini 2.5 Pro, GPT-5; memory agents; WorldMM-8B (Qwen3-VL-8B) and WorldMM-GPT | **base models: uniformly 0.5 fps, capped at 768 frames**; resolution not stated | EgoLifeQA / Ego-R1 / HippoVlog / LVBench / VMME-L / Avg. Qwen3-VL-8B 38.6 / 35.7 / 74.4 / 48.3 / 61.0 / 51.6. Gemini 2.5 Pro 46.4 / 46.7 / 72.0 / 57.0 / 55.7 / 55.6. GPT-5 48.6 / 46.3 / 75.7 / 60.4 / 74.3 / 61.1. WorldMM-8B 56.4 / 52.0 / 69.7 / 55.4 / 66.0 / 59.9. Latency appears only in a figure: long-video LLMs show "significantly higher inference latency" | open and commercial | A |
| **Ego-R1** (arXiv 2506.13654, 2025-06-16; TPAMI 2026) | Video-MME long, EgoSchema, EgoLifeQA, Ego-R1 Bench | LongVA-7B, LLaVA-Video-7B, LLaVA-OneVision-7B, InternVideo2.5-8B, Gemini-1.5-Pro; agents | **64 uniform frames for 7B models**; 512 for InternVideo2.5 and Gemini; LLaVA-OV at 1 fps; agents 1024 frames | VMME-L / EgoSchema / EgoLifeQA / Ego-R1. LLaVA-Video-7B 61.5 / 57.3 / 36.4 / 29.0. LLaVA-OV-7B 60.0 / 60.1 / 30.8 / 31.6. InternVideo2.5-8B 53.4 / 63.9 / 33.0 / 34.0. Gemini-1.5-Pro 67.4 / 72.2 / 36.9 / 38.3. TPAMI abstract adds Gemini-3.1-Pro 53.7 on Ego-R1 Bench | open and commercial | A |
| **MyEgo** (arXiv 2604.01966; CVPR 2026) | personalized egocentric VideoQA, 541 videos (9.2 min average) | Qwen3-VL-8B Instruct and Thinking, Qwen2.5-VL-7B, InternVL3/3.5-8B, LLaVA-Video, LLaVA-OV, MiniCPM-V 4.5, LongVA, LongVU; GPT-5; Gemini-2.5 Pro | **32 uniform frames, ≤1 fps** (LongVA and LongVU 128). Native resolution for Qwen; 448² for InternVL; GPT-5 480p; Gemini 720p | MC-2 / MC-5 / OE avg. Qwen3-VL-8B-Instruct 55.0 / 36.6 / 36.4. InternVL3.5-8B 54.6 / 36.6 / 29.9. Qwen2.5-VL-7B 55.1 / 32.3 / 34.3. GPT-5 66.4 / 53.7 / 46.1. Gemini-2.5 Pro 61.8 / 45.5 / 40.9. Paper: 128 frames did not help over 32 | open and commercial | A |
| **TaRO** (arXiv 2606.09248, 2026-06-08) | Ego4D NLQ, zero-shot (also TACoS) | Qwen3-VL-8B-Instruct; Time-R1 and VideoChat-R1.5 (Qwen2.5-VL-7B) | 2 fps, ~3584 video-token cap (stated for training; evaluation setting unclear) | R1@0.3 / 0.5 / 0.7. Qwen3-VL-8B 5.10 / 2.94 / 1.52. Time-R1 6.48 / 3.29 / 1.49. TaRO (Qwen3-VL-8B) 13.80 / 7.86 / 4.17 | open | A (table) |
| **Edge episodic memory** (arXiv 2602.22455, v2 2026-05-13) | QAEgo4D-Closed, 500 MC questions | Qwen3-VL 2B/4B/8B Instruct (clip describer plus text-only answerer); prior Gemini-based and LLaVA-OV results | describer 2 fps on 15 s clips; answering reads text memory only | 8B/8B accuracy 54.40; Gemini-based (prior work) 56.00. **Query TTFT on L40S 48 GB: Qwen3-VL-2B 0.20 s, 4B 0.49 s, 8B 0.88 s.** Describer peak memory, 8B at batch 2: 23.03 GB; 9.31 s per clip | open | A |
| **InternVideo3** (arXiv 2606.12195, 2026-06-10) | Video-MME, LVBench, EgoSchema, MLVU, others | baseline rows Qwen3-VL-8B, InternVL3.5-8B | settings not read | Qwen3-VL-8B: Video-MME 71.4, LVBench 58.0, EgoSchema 69.8, MLVU 57.6. InternVL3.5-8B: 66.0, 43.4, 58.6, 53.2. Not stated whether rerun or copied | open | C (rows only) |
| **Qwen3-VL technical report** (arXiv 2511.21631, v2 2025-11-27) | Video-MME, MLVU, LVBench and others | Qwen3-VL family | eval: ≤2048 frames, 2 fps (4 fps for Charades-STA), ≤224K video tokens, 640 tokens per frame | video needle-in-a-haystack (235B-A22B only): 100% up to 30 min (1 fps), 99.5% up to ~1M tokens (~2 h) with YaRN. Small-model numbers not extracted | open | B |
| **LSD-Bench supplement** (ICCV 2025) | LSD-Bench | Qwen2.5-VL-7B, one GPU (type not seen) | 256 vs 768 frames | 50.1% at 22.4 s per query (256 frames); 52.5% at 40.3 s (768 frames) | open | C |
| **MemStream** (arXiv 2602.18434, 2026-02-20) | CG-Bench, LVBench, VideoMME-long; RVS-Ego streaming | Qwen2.5-VL-7B | 0.5 fps to 128 frames (~17K tokens) or 512 frames (~68K) | uniform 128 frames: LVBench 41.51, VMME-L 55.67; 512 frames: 40.03, 52.11 (more frames scored lower). ReKV streaming: 2.8 s latency, 29 GB GPU; KV cache for 1 h at 0.5 fps is 10.3–13.2 GB | open | C |
| **LongEgoRefer supplement** (Springer 2026) | long egocentric referring expressions | Gemini 2.5 Pro, GPT-5, Qwen3-VL-32B, InternVL3.5-38B | Gemini 1 fps at **258 tokens per frame (66 in low-res mode beyond 1 h)**; GPT-5 ≤500 frames, long side 512 px; open models ≤128 frames, short side ~400 px | mvIoU (20–40 min): GPT-5 9.46, Gemini 2.5 Pro 5.50, Qwen3-VL-32B 1.43 | open and commercial | C |

### 4. Data and platforms: which open-weight models fit on one 48 GB GPU in bf16

[synthesis] Weights only, computed as parameter count × 2 bytes. Parameter counts are from the Hugging Face model API, checked 2026-11-13. Long-video inference also needs room for vision activations and the KV cache: for Qwen2.5-VL-7B, MemStream gives 10.3–13.2 GB of KV cache per hour of video at 0.5 fps and 29 GB total in its streaming setup.

| model (HF id) | params | bf16 weights | one 48 GB card | used as baseline in |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct | 2.13 B | ~4.3 GB | yes | edge paper |
| Qwen3-VL-4B-Instruct | 4.44 B | ~8.9 GB | yes | edge paper; our R8 plan |
| **Qwen3-VL-8B-Instruct** | 8.77 B | ~17.5 GB | yes; measured 23 GB peak on L40S 48 GB (edge paper) | UniFunc3D, WorldMM, MyEgo, TaRO, EgoAdapt, InternVideo3 |
| Qwen3-VL-30B-A3B-Instruct | 31.07 B | ~62 GB | **no** (needs two cards or quantization) | UniFunc3D-30B (H100) |
| Qwen3-VL-32B-Instruct | 33.36 B | ~67 GB | **no** | AffordMem (A100 80 GB), LongEgoRefer |
| Qwen2.5-VL-7B-Instruct | 8.29 B | ~16.6 GB | yes | HD-EPIC challenge entries, EgoSG, EgoReasoner, MyEgo |
| Qwen2.5-VL-32B / 72B | 33.45 B / 73.41 B | ~67 / ~147 GB | **no** | HelloWorld baseline (32B); AffordBot, TASA (72B, per others) |
| InternVL3.5-8B | 8.53 B | ~17.1 GB | yes | MyEgo, InternVideo3 |
| InternVL3.5-14B / InternVL3-14B | 15.12 B | ~30.2 GB | yes, ~18 GB left over | EgoSG (InternVL3-14B, 32 frames) |
| LLaVA-Video-7B-Qwen2 | 8.03 B | ~16.1 GB | yes | HD-EPIC, Ego-R1, MyEgo |
| VideoLLaMA3-7B | 8.04 B | ~16.1 GB | yes | EgoSG, EgoReasoner |
| Molmo-7B-D-0924 | 8.02 B (stored in F32) | ~16 GB in bf16 | yes | Fun3DU, T-FunS3D |
| Gemma-3-12B / 27B | 12.19 B / 27.43 B | ~24.4 / ~54.9 GB | yes / **no** | SceneFunRI |
| Qwen3.5-4B / 9B / 27B / 35B-A3B (released Feb 2026; tagged image-text-to-text) | 4.66 / 9.65 / 27.78 / 35.95 B | ~9.3 / ~19.3 / ~55.6 / ~71.9 GB | yes / yes / **no** / **no** | SceneFunRI (single image). Video support not checked |

### 5. Commercial APIs: who used what, and cost

| model | where used as a baseline | input setting stated | cost stated? |
| --- | --- | --- | --- |
| Gemini 1.5 Pro (gemini-1.5-pro-001) | HD-EPIC (37.6); Ego-R1 (512 frames) | 1 fps, 768×768, up to 2M context | No dollar figure. HD-EPIC says context caching was needed "to keep costs tractable" for repeated questions on one video; ~0.2% of questions were blocked by the API |
| Gemini 2.0 Flash | DeepFrames (video-only 35.00), EgoSG | 1 fps, 480p, ≤45 min per call (DeepFrames) | no |
| Gemini 2.5 Pro | EgoReasoner (49.2 on 4D prototypes, clipped protocol), WorldMM, MyEgo, LongEgoRefer | 258 tokens per frame at 1 fps; 66 in low-res mode (LongEgoRefer) | no |
| GPT-5 | WorldMM, MyEgo, LongEgoRefer | ≤500 frames, long side 512 (LongEgoRefer); 480p (MyEgo) | no |
| GPT-4o / o1 | only as the reasoning MLLM inside AffordBot (ablation) | 8 renders, 680×680 | no |
| Gemini 3 Flash / 3.1 Pro | SceneFunRI (single image); HIPPO (pipeline); Ego-R1 TPAMI (53.7 on Ego-R1 Bench) | — | no |

[synthesis] No surveyed paper gives a dollar cost per question. A cost estimate would need current list prices (not checked) multiplied by the stated token counts, e.g. 258 tokens per frame × 1 fps × video length.

### 6. Unsettled

1. Whether our 390 SceneFun3D tasks equal the full split0 task set used by Fun3DU, UniFunc3D, AffordMem and T-FunS3D. Not checked.
2. The SceneFun3D evaluator: which evaluator each paper used, and why ThinkAfford's "official evaluator" reproductions are 2–4× lower than the self-reported numbers. Not resolved. Rerunning baselines under our own evaluator may be the only clean alignment.
3. UniFunc3D's one-stage plain-VLM ablation does not state its frame count, so it cannot be matched exactly.
4. No published plain Qwen3-VL (any size) row exists on full HD-EPIC VQA. Settings for HelloWorld's Qwen2.5-VL-7B/32B baseline rows were not stated.
5. HD-EPIC numbers come from different question sets (full 26,550; EgoSG's 1,250; EgoReasoner's clipped six prototypes; DeepFrames' 50 per prototype in ablations) and are not comparable across those sets.
6. Whether EgoAdapt's 67.22 and HIPPO's 65.8 are test-server numbers. Neither report says.
7. Plain-VLM latency on HD-EPIC-length videos (15–30 min) at a stated frame budget on a 48 GB card: not found. Closest is LSD-Bench (Qwen2.5-VL-7B, 22.4 s at 256 frames, GPU unknown; grade C).

---

## Part 2 — Search and evidence record

### 1. Scope

- **Question (fixed before search):** Which "traditional VLM" baselines, and in which settings, recent papers report on SceneFun3D-type grounding, HD-EPIC VQA and long egocentric video QA, so our comparison can be aligned with published tables. Also: per-query latency or compute, and which open models fit on a 48 GB GPU in bf16.
- **Included:** SceneFun3D functionality-segmentation papers from 2024-11 to 2026-10. HD-EPIC VQA papers and challenge reports for 2025 and 2026. 2025–2026 long-video and egocentric memory papers that report plain-VLM baselines with frame budgets.
- **Excluded or not covered:** EgoSchema leaderboards beyond the rows above; Video-MME and LVBench leaderboards; streaming-only benchmarks; robotics manipulation; image-only affordance benchmarks other than SceneFunRI. Official challenge leaderboards (Codabench) were not opened.
- **Cutoff:** 2026-11-13. This is a targeted scoping review, not an exhaustive one.
- **Reused from earlier project records:** `docs/mem/mem分支_文献数据仿真调研_2026-11-10.md` already had EgoSG (InternVL3-8B 20.0 → 34.0 on object motion), HD-EPIC (Gemini Pro 20.8 on motion) and AffordBot (AP 15.5). These were re-read here and extended. New in this record: per-prototype FIC/OMI/OMC/SOL tables, the 2025/2026 challenge entries, UniFunc3D's and AffordMem's plain-VLM ablations, runtime per query, and the 2026 long-video baseline settings.

### 2. Query log (as typed)

WebSearch:
1. `HD-EPIC VQA benchmark results Gemini Pro LLaVA-Video Qwen2-VL per category object motion fixture`
2. `Fun3DU functional understanding segmentation 3D scenes SceneFun3D Molmo LLM baseline`
3. `UniFunc3D SceneFun3D functional part grounding`
4. `T-FunS3D task-driven functional segmentation 3D arXiv`
5. `TASA task-aware 3D functionality segmentation SceneFun3D Qwen2.5-VL-72B arXiv`
6. `SceneFun3D functionality segmentation 2026 training-free MLLM new state of the art arXiv`
7. `HD-EPIC VQA challenge 2025 technical report winner Gemini 2.5 Qwen2.5-VL accuracy`
8. `"HD-EPIC" VQA Qwen3-VL accuracy 2026`
9. `Qwen3-VL technical report arXiv video needle-in-a-haystack frames evaluation settings Video-MME LVBench`
10. `EgoSchema Video-MME LVBench 2026 baseline Qwen3-VL-8B InternVL3.5-8B frames 64 128 long video agent latency per query`
11. `Ego4D NLQ zero-shot Qwen2.5-VL-7B R1@0.3 multimodal LLM temporal grounding long egocentric video 2025`
12. `egocentric long video memory retrieval vs uniform frame sampling VLM baseline latency per query seconds EgoLife EgoLifeQA Qwen2.5-VL 2026`

Exa web_search:
13. `paper evaluating on SceneFun3D task-driven functional interactive element grounding with multimodal LLM, 2026`
14. `WorldMM dynamic multimodal memory agent long egocentric video question answering EgoLifeQA arXiv`
15. `Ego-R1 chain-of-tool-thought ultra-long egocentric video reasoning benchmark baselines Gemini GPT-4o Qwen2.5-VL frames`
16. `2026 paper evaluates on HD-EPIC VQA benchmark object motion fixture interaction with Qwen3-VL or Gemini 2.5 baseline`
17. `long video question answering memory index vs uniform frame sampling Qwen2.5-VL-7B latency seconds per query table GPU A100 2025 2026`
18. `long egocentric video question answering API cost per question USD Gemini 2.5 Pro GPT-5 tokens frames reported in paper`

Pages fetched and read (WebFetch): arXiv HTML pages for 2502.04144 (v2, both halves); 2601.10228; 2603.23478 (main part); 2606.05975; 2511.10017 (abs and HTML); 2511.11702; 2605.11616; 2411.16310v5; 2605.14704; 2608.10981; 2606.25842; 2512.02425 (both halves); 2606.09248; 2602.22455; 2605.24500; 2605.29402; 2511.21631 (both halves); 2606.12195 (main part); the hd-epic.github.io pages. Challenge PDFs HD-EPIC-R1 and HD-EPIC-R3 were downloaded and converted with pdftotext into `scout_vlm_pdfs/`.

Hugging Face model API (`https://huggingface.co/api/models/<id>`, parameter totals only): 20 model ids listed in section 4.

Failures: the CVF supplemental PDF for WorldMM returned HTTP 403, so the arXiv HTML was used instead. HD-EPIC-R1/R3 PDFs could not be read through WebFetch and were converted locally. The Qwen3-VL report's small-model video table was not extracted. The HD-EPIC paper's per-prototype figure (Fig. 11) is an image only; per-prototype values were taken from three later papers that agree with each other.

### 3. Selection

Results from the 18 queries were merged by paper id. A paper was kept if it (a) reports on SceneFun3D, HD-EPIC VQA, or an egocentric long-video benchmark, and (b) names the VLM, its size and its frame or input setting. A paper reporting latency or cost was also kept. Hits were not counted. Priority went to the last 18 months, with the HD-EPIC paper (Feb 2025) and Fun3DU (Nov 2024) kept as protocol anchors. The cap was about 25 works; 26 are listed.

### 4. Key evidence

| claim | primary source | supports / does not support |
| --- | --- | --- |
| No SceneFun3D paper has a plain-VLM main-table row | Fun3DU, TASA, AffordBot, AffordMem, T-FunS3D, UniFunc3D, ThinkAfford tables | supports (all read at grade A) |
| Plain one-stage Qwen3-VL-8B + SAM3 is far below pipelines on split0 (AP25 7.64 / 14.38 vs 44.04 for UniFunc3D-8B) | UniFunc3D Table 2 | supports |
| Frame-by-frame Qwen3-VL + SAM3 gives AP50 7.14, AP25 17.86 | AffordMem Table 2 | supports |
| Runtime per query: Fun3DU ~332 s, UniFunc3D-8B ~53 s (8× H100); T-FunS3D 78 s vs Fun3DU 167 s (A40); TASA 37.61 s vs Fun3DU 130.26 s (A800) | UniFunc3D Table 7; T-FunS3D Table III; TASA Table 2 | supports. Hardware and timing scope differ, so compare within each paper only |
| Cross-paper SceneFun3D numbers are not on one protocol | Fun3DU 16.9 vs T-FunS3D repro 10.3 vs ThinkAfford repro 4.55 (AP50) | supports |
| HD-EPIC plain-VLM per-prototype numbers exist for FIC and OMI/OMC/SOL | HelloWorld Table 2; same values in HIPPO and EgoReasoner | supports |
| A plain Qwen3-VL row exists on HD-EPIC | EgoAdapt (Qwen3-VL-8B used only inside its pipeline) | does not support: none found |
| Structured memory beats raw video on HD-EPIC object motion | EgoSG (InternVL3-8B 20.00 → 34.00; Qwen2.5-VL-7B 38.00 → 34.67); DeepFrames (Gemini 2.0 Flash 17.53 → 30.17) | mixed: supports for most models, not for Qwen2.5-VL-7B on the EgoSG subset |
| Write-once memory amortizes over many questions | EgoSG (≈5.7 s per 1-min clip, ~22 questions per video) | supports, from a figure plus text; per-query seconds not in the text |
| Qwen3-VL-8B is the default 2026 open baseline | WorldMM, MyEgo, TaRO, EgoAdapt, InternVideo3, edge paper | supports (synthesis across six papers) |
| A dollar cost per question is stated | all surveyed papers | does not support: none states one |

### 5. Not done

- Nothing was run or installed; no model was downloaded. The resource statuses in section 4 are "claimed by its HF model card/API", checked 2026-11-13, not "actually ran".
- Not opened: official HD-EPIC Codabench leaderboards (2025, 2026); the ECCV 2026 SceneFun3D challenge leaderboard; the Qwen3-VL report's small-model video table; the appendices of InternVideo3 and UniFunc3D (Δ value, one-stage frame count); the HD-EPIC per-prototype figure.
- Not checked: commercial API price lists; whether Qwen3.5 supports video input; GPU type behind the LSD-Bench timing; the Ricoh report's per-category figure (image only).
- C-grade entries (InternVideo3 rows, LSD-Bench, MemStream, LongEgoRefer) were not verified against the papers' own pages.

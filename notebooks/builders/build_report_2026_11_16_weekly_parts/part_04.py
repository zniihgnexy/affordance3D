# Cells of build_report_2026_11_16_weekly.py, part 4 of 4: run in order by that builder, with md() and code() in scope.

code(r"""
COMMITS = '''
11-09 15:19 d70db17 chore: add conda requirements, pytest setup and environment check
11-09 15:19 28144fc feat: add RGB-D geometry, DINOv2 features and voxel memory prototype
11-09 15:19 fe7d4d0 feat: add SceneFun3D loader, downloader and E0 alignment checks
11-09 15:19 8a03511 feat: add dataset report for the 9 November group meeting
11-09 15:19 53511fd docs: describe the local prototype in the README
11-10 11:44 77cec64 docs: start mem branch notes on 4D action-aware environment memory
11-10 12:04 d073bbc docs: record the Mac/remote compute split for SMPL-X visualisation
11-10 12:06 4dacf72 docs: note the A6000 server is Linux and where SMPL-X files go
11-10 12:45 0ec7c41 feat: add video-first inputs and actionable affordance helpers for S1
11-10 14:05 c0e9885 feat: add SMPL-X body model, humanoid placement and a small renderer
11-10 14:05 03eedb5 docs: add S1 notebooks (video inputs to humanoid display) and record results
11-10 14:21 01d6039 feat: add the S1 pipeline module, approach animation and demo helpers
11-10 14:21 cc3e364 docs: add a step-by-step demo notebook of the mem pipeline
11-10 15:08 5695b9d docs: record that this repository now carries only the memory direction
11-11 00:29 1464f34 feat: add server setup scripts and never pick CUDA automatically
11-11 00:29 a7f0a9e docs: describe the move to the A6000 server
11-11 01:28 2325adb chore: keep the notebook builders in the repository and add a run-all check
11-11 18:09 01081db docs: list the pipeline links to refine one by one (round-3 agenda)
11-11 18:51 1f85f2b docs: add the pipeline working notebook (idea, requirements and gaps per block)
11-11 20:06 2a818f6 docs: add the overview v2 proposal, related work and two new blocks to the pipeline notebook
11-11 22:25 60d03e7 docs: record the round-3 decisions and add the storage-form section (implicit memory) to the pipeline notebook
11-12 01:12 856ecbb feat: L2 memory (part tokens, keyframe search, VLM reading) with one worked example per pipeline block
11-12 01:12 628b79d docs: rewrite the pipeline document for the L2 memory and record the round-4 decisions
11-12 01:28 dabdf62 chore: track the report videos in git (third-party preview images stay out)
11-12 01:42 90502b5 docs: record round 5 and add the research assessment and roadmap (proposal)
11-12 19:39 2fb6472 docs: record round 6 and fix the phase-1 experiment specification before running it
11-12 23:32 f7698c5 docs: add the helm charter and ledger for the memory direction
11-12 23:58 2f1fe62 feat: phase-1 tooling (write policies, reconstruction baseline, keyframes-only download, per-scene runner)
11-13 00:18 280e2d3 docs: update the current mainline in the master-control document (P-1) and record the review-loop settings
11-13 00:35 69ae5c2 docs: record the contribution type and a candidate venue in the helm charter
11-13 07:07 9f23088 feat: phase-1 results tooling, follow-up diagnostics and two input edge-case fixes
11-13 07:07 856a4c4 docs: record the phase-1 results (provisional) and fix the plan of the follow-up checks before running them
11-13 08:15 8f0c789 feat: extend the phase-1 follow-up check and show it in the results notebook
11-13 08:15 53f5d99 docs: record the follow-up check, revise the phase-1 record after three review passes and open round 7
11-13 19:09 62b4f74 feat: pipeline flowchart and single-file brief for stage reports; split the phase-1 gain as flag only, tokens only, both
11-13 19:09 532f0f6 docs: rewrite the round-7 brief as one plain document with the pipeline flowchart and the contribution map
11-13 19:34 317f8fc docs: explain question 1 of round 7 in detail and add a reply guide
11-13 19:54 07d1099 docs: record the project lead's answers to round 7 and the passing of gate G1
11-13 20:04 cf57d75 docs: open round 8: restate the contribution the project lead described, for confirmation
11-13 20:10 f8f1afc feat: controls of round 7 (a vision-language model without the memory) and their plan, fixed before the run
11-13 20:15 003f211 docs: record the project lead's answer on the action direction (A and C, with B alongside)
11-13 20:20 a3fd78c fix: define the part in every prompt of the round-7 controls, after the pilot and before the full run
11-13 20:29 7af92f9 fix: longer answers for the box-drawing arms of the round-7 controls, before the full run
11-13 20:34 87325af docs: record the answers to round 8 and propose charter changes P-2
11-13 21:39 dc2d02f docs: amend the record of the first control batch after review, and fix two CPU checks before they run
11-13 21:49 e527cc4 docs: record the action-type test and the miss and false-alarm simulation, and the first review of round 9
11-13 21:56 5e6412c feat: second batch of the round-7 controls (b, d, c+ with the 4B model, a), planned before the run
11-13 21:58 d4c3f65 docs: plan a stricter false-alarm rule for the miss and false-alarm simulation, before computing it
11-13 22:03 78a8a06 docs: record the stricter false-alarm check, the second review of round 9 and the helm's decision D-8
11-13 22:21 560b14e docs: round 9, results of control c and the miss simulation, the field survey, and where the paper's contribution should sit
11-13 23:02 7bd63c6 docs: record the second batch of the round-7 controls (a, b, d, c+ with the 4B model)
11-13 23:54 2c9f822 docs: record the answers to round 9 (option B, the checks approved in the order the lead set)
11-14 00:01 3be0633 feat: controls c4, c1' and c2' and the 8B rerun of round 9, planned before the run
11-14 00:03 e5900e2 docs: plan the first steps of B (HD-EPIC) and C (action records) approved in round 9, before they run
11-14 00:10 72f5caa docs: propose charter changes P-3 after the answers to round 9 (replaces most of P-2)
11-14 00:19 5170ede docs: apply charter changes P-3 as approved by the lead (all but moving event memory inside the fence)
11-14 00:22 d622a45 docs: record the desk check of off-the-shelf tools for the tool-map check (round 9, question 2)
11-14 00:28 f1c9a50 feat: B-0 and B-1 on HD-EPIC kitchen P02 (round 9), and a skip option for the model fetcher
11-14 00:39 f26fca4 feat: C-0 action records on the 30 SceneFun3D validation scenes (round 9)
11-14 00:53 9b318fd docs: record the lead's answers on B-0, B-1 and C-1, and fix the amended plans of B-1 and C-1 before they run
11-14 00:55 33ae23e docs: plan the tool-map check of round 9 (question 2), before anything of it runs
11-14 01:01 052a4a1 docs: record the 4B results of the round-9 controls: c4 is close to P4, so the pre-set rule stops for discussion
11-14 01:05 1914526 feat: plan and code of the action-key retrieval check (upper bound, CPU), before it runs
11-14 01:24 f396558 feat: C-1 arms (plain VLM, geometry rule, memory, ground truth) as planned in section 25.5
11-14 01:34 e3b3d81 feat: B-1 with device poses read from the SLAM archives (trajectory files only): fails its bar
11-14 01:36 0e0d05c feat: code of the tool-map check (round 9, question 2), with rule 2 corrected before any tool ran
11-14 01:38 b96c02d fix: the tool-check test reads the recall gain under the rule's key (follows D-9)
11-14 03:15 6aa8fa0 docs: record the action-key retrieval check: no visible gain, and even the true type adds under 5 points
11-14 04:41 1f40f11 docs: record the 8B results of the round-9 controls: c4 is still close to P4, and the 8B model changes little
11-14 04:52 90cb176 docs: amend the C-1 plan before its full run: an arm M4 separates the memory from the annotation (D-10)
11-14 05:00 e7f448e docs: round-10 assessment (proposal), its outside views, and the 8B timings
11-14 05:02 937035b feat: C-1 arm M4 (c4's choice: annotated boxes on frames picked without the memory), per D-10
11-14 07:11 8938a1f docs: record the answers to the round-10 assessment, and amend the stop rule of the tool-map check before it runs
11-14 07:15 a624571 feat: plan and code of phase 2b (pooled-scene retrieval, CPU), before it runs
11-14 07:19 29e7aa7 docs: propose charter changes P-4 after the round-10 assessment was adopted
11-14 07:20 3b4490e docs: record phase 2b: the function index beats semantic retrieval by a steady 12 to 18 points, but the gap does not grow with pool size
11-15 16:31 ae8119e docs: record the answer after phase 2b, and plan its end-to-end version before it runs
11-15 16:54 60cc291 feat: end-to-end version of phase 2b (section 26.2): the 8B model reads numbered annotated boxes on pooled memories
11-16 18:28 adb3d5f docs: record the queue position the lead chose for the two small jobs
'''
commits = pd.DataFrame([dict(zip(("日期", "时间", "提交", "说明"), line.split(" ", 3))) for line in COMMITS.strip().splitlines()])
display(commits.groupby("日期").size().rename("提交数").to_frame().T)
print(f"{len(commits)} commits from 2026-11-09 to 2026-11-16 (commit timestamps; the last one is after the week)")
display(commits.set_index(["日期", "时间"]))
""")

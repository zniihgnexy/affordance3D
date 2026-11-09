# Video-Based Environment Memory and Affordance

This repository preserves the research background, proposed pipeline and experimental plans for a project on building environment memory from video.

**Current starting point:** the environment is the primary subject; affordance is a learnable feature layer within its memory. Search is one way to read that memory, and VLA or robot action is a possible later application. The methods and experiments described here are research plans, not validated results.

## Start here

1. [Research handoff and first experiments](docs/research/实验前研究交接_2026-11-06.md) — current direction, constraints and E0–E3 test stages.
2. [Pipeline and experimental route](docs/pipeline.md) — current architecture, planned comparisons and historical versions.
3. [Literature and research background](docs/literature.md) — reviews, source records and related work.
4. [Datasets and simulation resources](docs/data-and-simulation.md) — candidate data, platform information and resource constraints.
5. [Project scope and maintenance rules](docs/项目总控.md) — overall boundaries and the procedure for changing them.

The project scope records the confirmed 3 November Pipeline v3 and its earlier emphasis on demonstration search. Read it alongside the 6 November handoff, which records the later environment-centred clarification for the initial prototype; the handoff does not itself amend the project scope.

## Local prototype (Mac)

- **Environment:** [`requirements.txt`](requirements.txt), installed with conda + pip (commands at the top of the file); check it with `python scripts/check_env.py`.
- **Step-by-step demo:** [`notebooks/01_step_by_step_demo.ipynb`](notebooks/01_step_by_step_demo.ipynb) walks through the E0/E1 mechanics (projection, alignment checks, frozen DINOv2 features, voxel memory, readout) on Open3D's small Redwood RGB-D sample. It is an engineering demo, not SceneFun3D and not an affordance result.
- **E0 on SceneFun3D:** [`notebooks/02_scenefun3d_e0.ipynb`](notebooks/02_scenefun3d_e0.ipynb) checks data, orientation, depth units and cross-frame alignment on scene 421069; the run is recorded in [the E0 run record](docs/research/实验记录_E0_SceneFun3D_421069_2026-11-06.md). Download scenes with `python scripts/download_scenefun3d.py --visit <id> --video <id>` (ranged, resumable, CRC-checked).
- **Dataset report (group meeting, 2026-11-09):** [`report/2026-11-09_datasets_report.ipynb`](report/2026-11-09_datasets_report.ipynb) covers the candidate datasets, a recap of Labs 1–2, and one scene each from SceneFun3D and EPIC-KITCHENS + VISOR (walkthrough video, 3D view, query examples). Its helpers are in `report/`; before re-running it, fetch the EPIC video with `bash scripts/download_epic_p28_101.sh` and the official preview images with `bash scripts/download_report_images.sh`.
- **Code and tests:** `affordance3d/` (frames, geometry, features, memory, SceneFun3D loader, E0 checks, viz); run `python -m pytest`.

See the [documentation index](docs/README.md) for additional background. Dataset and platform descriptions are not proof of access, installation or compatibility; individual resources and their licences should be checked before use.

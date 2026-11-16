"""The public evaluation of section 27.2, end to end from one config: GPU stages through the queue, lifting, scoring.

    python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage queue [--scenes 421254]
    python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage lift
    python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage score [--official]
    python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage status
    ... --set compare.seed=1          (any setting; the run folder then must be new, see affordance3d.experiment)

Run from the repository root. Everything goes to <outputs>/<name>/ (the run folder), which records the resolved config
and the commit (resolved_config.json) and every stage run (stages.log):

    queue   writes queue jobs (GPU; $AFFORDANCE3D_QUEUE/pending) that run per scene, in order:
              scripts/run_tool_maps.py --tool sam3_tasks --all-tasks      SAM 3 with every task text (tool_maps/sam3_tasks_all/)
              scripts/run_mem_controls.py --all-tasks                     every arm answers every task (controls_<model>_all/)
              scripts/run_box_masks.py --out <run>/box_masks              SAM 3 masks of the chosen boxes
            the first two are shared caches under runs/mem_phase1/ (their answers are reused, never redone)
    lift    scripts/lift_to_scan.py per scene (CPU): <run>/lifted/<arm>/, <run>/official/<arm>/, causes
    score   scripts/scenefun3d_eval.py (CPU): <run>/scores/<arm>.json (Fun3DU protocol), <run>/compare/*.json (paired,
            read by the config's margins), and with --official the official scorer: <run>/official_scores/<arm>.json
    status  what is done per scene
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402

FIXED_POSE_GAP_S = 0.05  # affordance3d.scenefun3d.index_sequence matches poses within this gap; not a free setting
ORACLES = ("oracle_chosen", "oracle_best")


def scenes_of(config: dict, chosen: str) -> list[str]:
    every = [line.strip() for line in (REPO_ROOT / config["scenes"]).read_text().splitlines() if line.strip()]
    wanted = [s for s in chosen.split(",") if s] if chosen else every
    if set(wanted) - set(every):
        raise SystemExit(f"scenes not in {config['scenes']}: {sorted(set(wanted) - set(every))}")
    return wanted


def queue(config: dict, run: Path, scenes: list[str]) -> None:
    q = config["queue"]
    folder = Path(os.environ.get(q["folder_env"], "")) / "pending"
    if not folder.is_dir():
        raise SystemExit(f"set {q['folder_env']} to the queue folder (the one that holds pending/)")
    todo = [v for v in scenes if not (run / "box_masks" / f"{v}.npz").exists()]
    model, arms = config["reading"]["model"], ",".join(config["reading"]["arms"])
    for j in range(q["jobs"]):
        visits = todo[j::q["jobs"]]
        if not visits:
            continue
        job = folder / f"{q['priority']}_{config['name']}_{j}.sh"
        if job.exists():
            raise SystemExit(f"{job} is already queued")
        steps = (f'for v in {" ".join(visits)}; do\n'
                 f'  python scripts/run_tool_maps.py --visit $v --tool sam3_tasks --all-tasks || {{ echo "FAILED $v sam3_tasks"; continue; }}\n'
                 f'  python scripts/run_mem_controls.py --visit $v --model {model} --all-tasks --arms {arms} || {{ echo "FAILED $v controls"; continue; }}\n'
                 f'  python scripts/run_box_masks.py --visit $v --arms {arms} --out {run / "box_masks"} || echo "FAILED $v masks"\n'
                 'done\n')
        job.with_suffix(".tmp").write_text(
            f"#Q minmb={q['minmb']} share={q['share']}\nset -uo pipefail\ncd {REPO_ROOT}\nsource scripts/remote/env.sh\n"
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AFFORDANCE3D_DEVICE=cuda HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8\n" + steps)
        job.with_suffix(".tmp").rename(job)
        print(f"queued {job.name}: {len(visits)} scenes")


def lift(config: dict, run: Path, scenes: list[str]) -> None:
    for v in scenes:
        if not (run / "box_masks" / f"{v}.npz").exists():
            print(f"{v}: no masks yet")
            continue
        subprocess.run([sys.executable, "scripts/lift_to_scan.py", "--visit", v, "--masks", str(run / "box_masks"), "--out", str(run),
                        "--rel-tol", str(config["lifting"]["rel_depth_tol"]), "--memory-arm", config["reading"]["memory_arm"]],
                       cwd=REPO_ROOT, check=True)


def score(config: dict, run: Path, official: bool) -> None:
    script = [sys.executable, "-I", "scripts/scenefun3d_eval.py"]
    arms = [*config["reading"]["arms"], *ORACLES]
    for arm in arms:
        subprocess.run([*script, "fun3du", "--pred_dir", str(run / "lifted" / arm), "--out", str(run / "scores" / f"{arm}.json")],
                       cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        if official and arm in config["reading"]["arms"]:
            out = subprocess.run([*script, "score", "--pred_dir", str(run / "official" / arm)], cwd=REPO_ROOT, check=True,
                                 capture_output=True, text=True).stdout
            (run / "official_scores").mkdir(exist_ok=True)
            (run / "official_scores" / f"{arm}.json").write_text(out)
    c = config["compare"]
    pairs = [(arm, REPO_ROOT / c["reference"], f"{arm}_vs_reference") for arm in config["reading"]["arms"]]
    pairs.append((config["reading"]["memory_arm"], run / "lifted" / "c3", f"{config['reading']['memory_arm']}_vs_c3"))
    for arm, other, label in pairs:
        subprocess.run([*script, "compare", "--first", str(run / "lifted" / arm), "--second", str(other),
                        "--margins", json.dumps(c["margins"]), "--resamples", str(c["resamples"]), "--seed", str(c["seed"]),
                        "--out", str(run / "compare" / f"{label}.json")], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    summary = {arm: json.loads((run / "scores" / f"{arm}.json").read_text())["means"] for arm in arms}
    verdicts = {p.stem: json.loads(p.read_text())["verdict"] for p in sorted((run / "compare").glob("*.json"))}
    (run / "summary.json").write_text(json.dumps({"means": summary, "verdicts": verdicts}, indent=1))
    print(json.dumps({"means": {a: {m: round(v, 2) for m, v in s.items()} for a, s in summary.items()}, "verdicts": verdicts}, indent=1))


def status(config: dict, run: Path, scenes: list[str]) -> None:
    model = config["reading"]["model"]
    checks = {"sam3_tasks_all": lambda v: REPO_ROOT / "runs/mem_phase1/tool_maps/sam3_tasks_all" / f"{v}.json",
              f"controls_{model}_all": lambda v: REPO_ROOT / f"runs/mem_phase1/controls_{model}_all" / f"{v}.json",
              "box_masks": lambda v: run / "box_masks" / f"{v}.npz",
              "lifted": lambda v: run / "lifted" / config["reading"]["memory_arm"] / f"{v}.json"}
    for name, path in checks.items():
        done = [v for v in scenes if path(v).exists()]
        print(f"{name}: {len(done)} of {len(scenes)} scenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("queue", "lift", "score", "status"), required=True)
    parser.add_argument("--scenes", default="", help="comma-separated visits (default: every scene of the config)")
    parser.add_argument("--official", action="store_true", help="score: also run the official scorer (slow)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting of the config")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config["lifting"]["max_pose_gap_s"] != FIXED_POSE_GAP_S:
        raise SystemExit(f"lifting.max_pose_gap_s is fixed at {FIXED_POSE_GAP_S} s by the scene index; it cannot be set here")
    run = run_folder(config, REPO_ROOT)
    scenes = scenes_of(config, args.scenes)
    if args.stage != "status":
        log_stage(run, args.stage, REPO_ROOT)
    if args.stage == "queue":
        queue(config, run, scenes)
    elif args.stage == "lift":
        lift(config, run, scenes)
    elif args.stage == "score":
        score(config, run, args.official)
    else:
        status(config, run, scenes)


if __name__ == "__main__":
    main()

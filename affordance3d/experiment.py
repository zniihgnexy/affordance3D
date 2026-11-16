"""Experiment bookkeeping shared by every configuration-driven entrypoint: load a YAML config with explicit overrides,
and keep one output folder per experiment that records the resolved config and the code version.

    config = load_config(Path("configs/public_eval/split0_8b.yaml"), ["compare.seed=1"])
    folder = run_folder(config, REPO_ROOT)   # <config["outputs"]>/<config["name"]>/, with resolved_config.json

A run folder is never silently reused for a different experiment: if it already holds a resolved config that differs
(code version and time aside), run_folder stops. Each stage that runs appends its command and commit to stages.log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

RESOLVED = "resolved_config.json"


def _set(config: dict, dotted: str, value: object) -> None:
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise SystemExit(f"override {dotted}: {key} is not a section of the config")
        node = node[key]
    if keys[-1] not in node:
        raise SystemExit(f"override {dotted}: no such setting in the config")
    node[keys[-1]] = value


def load_config(path: Path, overrides: list[str] = ()) -> dict:
    """The YAML config at `path` with `key.sub=value` overrides applied (values parsed as YAML); unknown keys stop."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        dotted, _, raw = item.partition("=")
        if not raw:
            raise SystemExit(f"override {item!r}: expected key=value")
        _set(config, dotted.strip(), yaml.safe_load(raw))
    config["_source"] = {"config": str(path), "overrides": list(overrides)}
    return config


def git_commit(root: Path) -> str:
    """The commit of the code that runs, with "+dirty" when tracked files differ from it."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout
    return (head or "unknown") + ("+dirty" if dirty.strip() else "")


def run_folder(config: dict, root: Path) -> Path:
    """<outputs>/<name>/ for this config: created with resolved_config.json, or reused only when that file matches."""
    folder = root / config["outputs"] / config["name"]
    record = folder / RESOLVED
    if record.exists():
        if json.loads(record.read_text())["config"] != config:
            raise SystemExit(f"{folder} holds a different experiment; give this config another name")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"config": config, "commit": git_commit(root), "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                                     indent=1, ensure_ascii=False))
    return folder


def log_stage(folder: Path, stage: str, root: Path) -> None:
    """Append the stage, the command line and the code version to <folder>/stages.log."""
    with (folder / "stages.log").open("a", encoding="utf-8") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{stage}\t{git_commit(root)}\t{' '.join(sys.argv)}\n")

"""Stage 0, test T5 (proposal section 13; configs/stage0/openeqa_fit.yaml): how many OpenEQA questions are about
functional parts. Text only: the public question file, no scene data, no model.

    python scripts/openeqa_fit.py --config configs/stage0/openeqa_fit.yaml --stage count     # flag, count, draw the sample
    python scripts/openeqa_fit.py --config configs/stage0/openeqa_fit.yaml --stage correct   # after the sample is read

count: downloads the file once (the config's source to its local path), flags each question by the config's part and
use words (whole words, ignoring case; a trailing "s" allowed on part words), counts flags per category and per source
(ScanNet / HM3D, from the episode path), and draws sample_per_side flagged and unflagged questions with the seed into
<outputs>/<name>/sample.json, each with "relevant": null to be filled by hand against the config's criterion.
correct: reads the filled sample, estimates the instrument's precision (relevant among flagged) and the share of
relevant questions among unflagged ones, and corrects every share: f * precision + (1 - f) * missed. Writes summary.json.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402

FIELDS = ("question", "answer", "category", "episode_history")


def matcher(part_words: list[str], use_words: list[str]) -> re.Pattern:
    parts = [re.escape(w) + "s?" for w in part_words]
    uses = [r"\s+".join(re.escape(x) for x in w.split()) for w in use_words]
    return re.compile(r"\b(?:" + "|".join(parts + uses) + r")\b", re.IGNORECASE)


def source_of(item: dict) -> str:
    path = item.get("episode_history", "").lower()
    return "scannet" if "scannet" in path else "hm3d" if "hm3d" in path else "other"


def load_questions(config: dict) -> list[dict]:
    local = REPO_ROOT / config["local"]
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(config["source"], timeout=60) as response:  # noqa: S310 (fixed https URL from the config)
            local.write_bytes(response.read())
    items = json.loads(local.read_text(encoding="utf-8"))
    missing = [f for f in FIELDS if any(f not in item for item in items)]
    if missing:
        raise SystemExit(f"{local}: some questions lack {missing}")
    return items


def counts(items: list[dict], flagged: list[bool]) -> dict:
    table: dict[str, dict] = {}
    for item, flag in zip(items, flagged, strict=True):
        for key in (("all", "all"), ("category", item["category"]), ("source", source_of(item)),
                    ("category_source", f"{item['category']} | {source_of(item)}")):
            cell = table.setdefault(key[0], {}).setdefault(key[1], {"questions": 0, "flagged": 0})
            cell["questions"] += 1
            cell["flagged"] += int(flag)
    for group in table.values():
        for cell in group.values():
            cell["share"] = cell["flagged"] / cell["questions"]
    return table


def stage_count(config: dict, folder: Path) -> None:
    items = load_questions(config)
    pattern = matcher(config["part_words"], config["use_words"])
    hits = [sorted({m.lower() for m in pattern.findall(item["question"])}) for item in items]
    flagged = [bool(h) for h in hits]
    rng = np.random.default_rng(config["seed"])
    sample = []
    for side in (True, False):
        pool = [i for i, f in enumerate(flagged) if f == side]
        for i in sorted(rng.choice(pool, size=min(config["sample_per_side"], len(pool)), replace=False).tolist()):
            sample.append({"index": i, "flagged": side, "words": hits[i], "category": items[i]["category"],
                           "source": source_of(items[i]), "question": items[i]["question"], "answer": items[i]["answer"],
                           "relevant": None})
    table = counts(items, flagged)
    (folder / "counts.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    (folder / "sample.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False))
    print(json.dumps({k: table[k] for k in ("all", "category", "source")}, indent=1))
    print(f"sample of {len(sample)} written to {folder / 'sample.json'}; fill 'relevant' by hand, then --stage correct")


def stage_correct(config: dict, folder: Path) -> None:
    sample = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
    if any(row["relevant"] is None for row in sample):
        raise SystemExit("sample.json still has questions without a judgement")
    precision = float(np.mean([row["relevant"] for row in sample if row["flagged"]]))
    missed = float(np.mean([row["relevant"] for row in sample if not row["flagged"]]))
    table = json.loads((folder / "counts.json").read_text(encoding="utf-8"))
    for group in table.values():
        for cell in group.values():
            cell["corrected"] = cell["share"] * precision + (1 - cell["share"]) * missed
    focus = {name: cell for name, cell in table["category"].items()
             if any(word in name.lower() for word in config["focus_categories"])}
    questions = sum(c["questions"] for c in focus.values())
    corrected = sum(c["corrected"] * c["questions"] for c in focus.values()) / questions
    reading = ("candidate for the QA head, part subset as a headline" if corrected >= 0.30 else
               "part subset on the side; OpenEQA as a whole shows that nothing gets worse" if corrected >= 0.10 else
               "no-regression check only")
    flagged_all = table["all"]["all"]["share"]
    top = max(table["category"], key=lambda k: table["category"][k]["share"])
    summary = {"precision": precision, "relevant_among_unflagged": missed, "focus_categories": sorted(focus),
               "focus_corrected_share": corrected, "reading": reading, "table": table,
               "predictions": {"flagged_share_10_to_25": {"held": 0.10 <= flagged_all <= 0.25, "share": flagged_all},
                               "highest_is_object_state": {"held": "state" in top.lower(), "highest": top}}}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("count", "correct"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, args.stage, REPO_ROOT)
    (stage_count if args.stage == "count" else stage_correct)(config, folder)


if __name__ == "__main__":
    main()

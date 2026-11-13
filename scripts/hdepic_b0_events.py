"""B-0: can the past-action questions of one HD-EPIC kitchen be tied to annotated events? (discussion draft, 25.1)

    python scripts/hdepic_b0_events.py     # reads data/hd-epic/annotations, writes runs/hdepic/b0_events.json

CPU only; annotations only, no video. The six past-action prototypes (all but fixture location) of one kitchen:
- every question box (`<BBOX>` at its first `<TIME>`) is tied to an object association when a mask of that video
  lies within 15 frames and overlaps the box with IoU >= 0.5 (fixture-interaction counting is tied to the
  narrations instead: the video's narrations whose verb includes the asked "open"/"close" are listed);
- movement counting and stationary-object localisation are answered again by rule from the tied association:
  the count is its number of tracks; the stationary answer is the one choice from which the object stays still
  for at least the asked duration after its first movement (no answer if zero or several choices qualify).
Rule: continue if >= 90% of the questions are tied AND the rules give the correct answer for >= 90% of the
counting and stationary questions (untied questions count as wrong); otherwise it is a data failure.

    python scripts/hdepic_b0_events.py --swap-box-axes     # post-hoc repair run, writes runs/hdepic/b0_events_swapped.json

The repair run was added after the first run failed: the question boxes and the `mask_info` boxes hold the same
numbers with x and y exchanged. It reads each question box as (top, left, bottom, right); it is not the
pre-registered verdict.
"""

import argparse
import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d import hdepic as H

PROTOTYPES = (
    "3d_perception_fixture_interaction_counting",
    "3d_perception_object_contents_retrieval",
    "3d_perception_object_location",
    "object_motion_object_movement_counting",
    "object_motion_object_movement_itinerary",
    "object_motion_stationary_object_localization",
)
FIXTURE_COUNTING, COUNTING, STATIONARY, ITINERARY = PROTOTYPES[0], PROTOTYPES[3], PROTOTYPES[5], PROTOTYPES[4]
RULE_PROTOTYPES = (COUNTING, STATIONARY)
MIN_TIED = 0.90
MIN_REPRODUCED = 0.90


def kitchen_questions(vqa_dir: Path, kitchen: str) -> list[tuple[str, str, dict]]:
    """(prototype, question id, question) for the questions of one kitchen, in file order."""
    rows = []
    for prototype in PROTOTYPES:
        for question_id, question in json.loads((vqa_dir / f"{prototype}.json").read_text()).items():
            if question["inputs"]["video 1"]["id"].startswith(f"{kitchen}-"):
                rows.append((prototype, question_id, question))
    return rows


def check_fixture_counting(question: dict, narrations: list[dict]) -> dict:
    """List the open/close narrations of the video; tied if there is at least one with the asked verb."""
    verb = H.fixture_verb(question["question"])
    listed = H.narrations_with_verb(narrations, verb) if verb else []
    answer = question["choices"][question["correct_idx"]]
    return {"verb": verb, "tied": bool(listed), "narrations": len(listed), "answer": answer,
            "answer_within_listed": answer.isdigit() and int(answer) <= len(listed),
            "listed": [[round(r["start_timestamp"], 2), round(r["end_timestamp"], 2), r["narration"]] for r in listed]}


def solve_counting(question: dict, assoc: H.Association, masks: dict[str, H.Mask]) -> dict:
    """The rule count against the correct answer, and whether the object passes the paper's retention filter."""
    count = H.movement_count(assoc)
    answer = question["choices"][question["correct_idx"]]
    return {"rule_answer": count, "answer": answer, "reproduced": str(count) == answer.strip(),
            "retention_filter": H.passes_retention_filter(assoc, masks)}


def solve_stationary(question: dict, assoc: H.Association) -> dict:
    """The one qualifying choice against the correct answer."""
    required = H.required_static_seconds(question["question"])
    choice_times = [H.times_in(choice)[0] for choice in question["choices"]]
    qualifying = H.qualifying_choices(assoc, choice_times, required) if required is not None else []
    rule_answer = qualifying[0] if len(qualifying) == 1 else None
    return {"required_s": required, "qualifying": qualifying, "rule_answer": rule_answer,
            "answer": question["correct_idx"], "reproduced": rule_answer == question["correct_idx"],
            "static_intervals": [[round(a, 2), None if b == float("inf") else round(b, 2)]
                                 for a, b in H.static_intervals(assoc)]}


def check_question(prototype: str, question: dict, video: dict, swap_box_axes: bool = False) -> dict:
    """Tie one question to the annotations and, for the two rule prototypes, answer it again."""
    record: dict = {"prototype": prototype, "video": question["inputs"]["video 1"]["id"]}
    if prototype == FIXTURE_COUNTING:
        record.update(check_fixture_counting(question, video["narrations"]))
        return record
    query = H.query_of(question)
    if query is None:
        record.update({"tied": False, "reason": "no box or time in the question"})
        return record
    if swap_box_axes:
        query = dataclasses.replace(query, box=H.swap_axes(query.box))
    resolution = H.resolve(query, video["masks"], video["owners"])
    record.update({"query_s": query.seconds, "tied": resolution.assoc_id is not None, "assoc": resolution.assoc_id,
                   "iou": round(resolution.iou, 3), "frame_gap": resolution.frame_gap,
                   "candidate_assocs": resolution.candidate_assocs})
    if resolution.assoc_id is None:
        if prototype in RULE_PROTOTYPES:
            record["reproduced"] = False
        return record
    assoc = video["assocs"][resolution.assoc_id]
    record.update({"object": assoc.name, "tracks": len(assoc.tracks)})
    if prototype == COUNTING:
        record.update(solve_counting(question, assoc, video["mask_by_id"]))
    elif prototype == STATIONARY:
        record.update(solve_stationary(question, assoc))
    elif prototype == ITINERARY:  # descriptive only: moves in the correct answer against the tied object's tracks
        record["answer_hops"] = H.itinerary_hops(question["choices"][question["correct_idx"]])
    return record


def load_videos(data: Path, videos: set[str]) -> dict[str, dict]:
    """Masks, associations and narrations of the given videos."""
    mask_info = json.loads((data / "scene-and-object-movements" / "mask_info.json").read_text())
    assoc_info = json.loads((data / "scene-and-object-movements" / "assoc_info.json").read_text())
    table = H.load_pickled_table(data / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    table = table[table.video_id.isin(videos)]
    narrations: dict[str, list[dict]] = defaultdict(list)
    for row in table[["video_id", "start_timestamp", "end_timestamp", "narration", "verbs"]].to_dict("records"):
        narrations[row["video_id"]].append(row)
    result = {}
    for video in sorted(videos):
        masks, assocs = H.masks_of(mask_info, video), H.associations_of(assoc_info, video)
        result[video] = {"masks": masks, "mask_by_id": {m.mask_id: m for m in masks}, "assocs": assocs,
                         "owners": H.mask_owners(assocs), "narrations": narrations[video]}
    return result


def share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarise(records: list[dict]) -> dict:
    """Per-prototype and overall rates, and the verdict by the fixed rule."""
    per_prototype = {}
    for prototype in PROTOTYPES:
        rows = [r for r in records if r["prototype"] == prototype]
        entry = {"questions": len(rows), "tied": sum(r["tied"] for r in rows)}
        if prototype in RULE_PROTOTYPES:
            entry["reproduced"] = sum(r.get("reproduced", False) for r in rows)
            entry["reproduced_among_tied"] = share(entry["reproduced"], entry["tied"])
        if prototype == COUNTING:
            entry["retention_filter"] = dict(Counter(str(r.get("retention_filter")) for r in rows if r["tied"]))
        if prototype == STATIONARY:
            entry["qualifying_choices"] = dict(Counter(len(r["qualifying"]) for r in rows if r["tied"]))
        if prototype == FIXTURE_COUNTING:
            entry["answer_within_listed"] = sum(r["answer_within_listed"] for r in rows)
        if prototype == ITINERARY:
            entry["hops_equal_tracks"] = sum(r.get("answer_hops") == r.get("tracks") for r in rows if r["tied"])
        if prototype != FIXTURE_COUNTING:
            entry["ambiguous"] = sum(r.get("candidate_assocs", 0) > 1 for r in rows)
        per_prototype[prototype] = entry
    total, tied = len(records), sum(r["tied"] for r in records)
    rule_rows = [r for r in records if r["prototype"] in RULE_PROTOTYPES]
    reproduced = sum(r.get("reproduced", False) for r in rule_rows)
    tied_share, reproduced_share = share(tied, total), share(reproduced, len(rule_rows))
    passed = (tied_share or 0) >= MIN_TIED and (reproduced_share or 0) >= MIN_REPRODUCED
    return {"questions": total, "tied": tied, "tied_share": tied_share, "rule_questions": len(rule_rows),
            "reproduced": reproduced, "reproduced_share": reproduced_share,
            "reproduced_among_tied": share(reproduced, sum(r["tied"] for r in rule_rows)),
            "thresholds": {"tied": MIN_TIED, "reproduced": MIN_REPRODUCED},
            "verdict": "continue" if passed else "data failure", "per_prototype": per_prototype}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "hd-epic" / "annotations")
    parser.add_argument("--kitchen", default="P02")
    parser.add_argument("--out", type=Path, default=None, help="default runs/hdepic/b0_events[_swapped].json")
    parser.add_argument("--swap-box-axes", action="store_true",
                        help="post-hoc repair run: read question boxes as (top, left, bottom, right)")
    args = parser.parse_args()
    args.out = args.out or REPO_ROOT / "runs" / "hdepic" / f"b0_events{'_swapped' if args.swap_box_axes else ''}.json"
    questions = kitchen_questions(args.data / "vqa-benchmark", args.kitchen)
    videos = load_videos(args.data, {q["inputs"]["video 1"]["id"] for _, _, q in questions})
    records = [{"id": question_id, **check_question(prototype, question, videos[question["inputs"]["video 1"]["id"]],
                                                    args.swap_box_axes)}
               for prototype, question_id, question in questions]
    summary = summarise(records)
    summary["run"] = "post-hoc repair (question box axes swapped)" if args.swap_box_axes else "pre-registered"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kitchen": args.kitchen, "summary": summary, "questions": records}, indent=1))
    for prototype, entry in summary["per_prototype"].items():
        print(f"{prototype:48s} {entry}")
    print(f"tied {summary['tied']}/{summary['questions']} = {summary['tied_share']}; rules reproduce "
          f"{summary['reproduced']}/{summary['rule_questions']} = {summary['reproduced_share']} "
          f"(among tied {summary['reproduced_among_tied']}); verdict: {summary['verdict']}")
    print(f"wrote {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")


if __name__ == "__main__":
    main()

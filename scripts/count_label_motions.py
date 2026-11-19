"""How often each affordance label moves by translation or rotation, from the annotation files alone (no frames).

    python scripts/count_label_motions.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt

Reads <visit>_annotations.json and <visit>_motions.json of every listed scene (fetch them first, e.g. with
affordance3d.scenefun3d_download.download_scene and the assets "annotations" and "motions"). Prints, per label, the
parts with a motion annotation split by motion type, and the share of parts a label-only rule gets right at best
(each label taking its more common type): the ceiling of any rule that reads the motion type from the label.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def label_motions(root: Path, visit: str) -> list[tuple[str, str]]:
    """(label, motion type) of every part of the scene that has a motion annotation; the label exclude is left out."""
    folder = root / visit
    labels = {a["annot_id"]: a["label"] for a in json.loads((folder / f"{visit}_annotations.json").read_text())["annotations"]}
    motions = json.loads((folder / f"{visit}_motions.json").read_text())["motions"]
    return [(labels[m["annot_id"]], m["motion_type"]) for m in motions if labels.get(m["annot_id"], "exclude") != "exclude"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=Path, required=True, help="text file with one visit id per line")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    args = parser.parse_args()
    visits = args.scenes.read_text().split()
    pairs = Counter(pair for visit in visits for pair in label_motions(args.root, visit))
    labels = sorted({label for label, _ in pairs}, key=lambda label: -(pairs[(label, "trans")] + pairs[(label, "rot")]))
    print(f"{len(visits)} scenes, {sum(pairs.values())} parts with a motion")
    print(f"{'label':12s} {'trans':>6s} {'rot':>6s} {'rot %':>6s}")
    for label in labels:
        trans, rot = pairs[(label, "trans")], pairs[(label, "rot")]
        print(f"{label:12s} {trans:6d} {rot:6d} {100 * rot / (trans + rot):6.1f}")
    best = sum(max(pairs[(label, "trans")], pairs[(label, "rot")]) for label in labels)
    print(f"label-only rule, best possible motion type: {best}/{sum(pairs.values())} = {100 * best / sum(pairs.values()):.1f}%")


if __name__ == "__main__":
    main()

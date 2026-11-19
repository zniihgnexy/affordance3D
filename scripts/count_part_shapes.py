"""How elongated each annotated part is, per affordance label, from the annotation and laser-scan files alone (no frames).

    python scripts/count_part_shapes.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt

For every part (label other than exclude, at least MIN_POINTS laser-scan points, with or without a motion annotation)
it takes the span of the part's points along their longest and second-longest axis (5th to 95th percentile) and
prints, per label, the share of parts at least ELONGATED times as long as wide and the median length and width. A grasp
axis (the closing direction across a part) only means something for elongated parts such as bar handles (proposal
2026-11-18, section 10).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.scenefun3d import read_annotations, read_laser_scan  # noqa: E402

MIN_POINTS = 20
ELONGATED = 2.0


def spans(points: np.ndarray) -> tuple[float, float]:
    """Length and width: the 5th-to-95th percentile span along the longest and the second-longest principal axis."""
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    length, width = (float(np.subtract(*np.percentile(centred @ vectors[:, k], [95, 5]))) for k in (2, 1))
    return length, width


def label_spans(root: Path, visit: str) -> list[tuple[str, tuple[float, float]]]:
    folder = root / visit
    xyz, _ = read_laser_scan(folder / f"{visit}_laser_scan.ply")
    return [(e.label, spans(xyz[e.indices])) for e in read_annotations(folder / f"{visit}_annotations.json")
            if e.label != "exclude" and len(e.indices) >= MIN_POINTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=Path, required=True, help="text file with one visit id per line")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    args = parser.parse_args()
    by_label = defaultdict(list)
    for visit in args.scenes.read_text().split():
        for label, shape in label_spans(args.root, visit):
            by_label[label].append(shape)
    print(f"{'label':12s} {'parts':>5s} {'elongated':>9s} {'length cm':>9s} {'width cm':>8s}")
    total = elongated = 0
    for label, shapes in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        shapes = np.array(shapes)
        long = shapes[:, 0] >= ELONGATED * np.maximum(shapes[:, 1], 1e-6)
        total, elongated = total + len(shapes), elongated + int(long.sum())
        print(f"{label:12s} {len(shapes):5d} {100 * long.mean():8.0f}% {100 * np.median(shapes[:, 0]):9.1f} {100 * np.median(shapes[:, 1]):8.1f}")
    print(f"all: {elongated}/{total} parts at least {ELONGATED:g} times as long as wide")


if __name__ == "__main__":
    main()

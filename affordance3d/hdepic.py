"""Readers and rule solvers for the public HD-EPIC annotations (checks B-0 and B-1, discussion draft section 25).

The annotations come from the HD-EPIC GitHub repository (CC BY-NC 4.0) and stay on the data disk. Questions are
the VQA-benchmark files; an object seen in a question (`<BBOX ...>` at `<TIME ...>`) is tied to an object
"association" of `scene-and-object-movements` (a named object and its tracks, one track per movement). The rules
for the answers follow the HD-EPIC paper (arXiv 2502.04144, appendix on the VQA prototypes).
"""

from __future__ import annotations

import io
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

FPS = 30  # MP4 frame rate; mask frame numbers count from 0
MAX_FRAME_GAP = 15  # a question's box matches a mask at most this many frames away ...
MIN_IOU = 0.5  # ... that overlaps it at least this much
MIN_TRACK_GAP_S = 1.1  # the paper keeps objects for movement counting only if consecutive tracks are this far apart ...
MAX_SHIFT_M = 0.2  # ... and the object moves at most this far between the end of one track and the start of the next
TIME_TOLERANCE_S = 0.1  # slack on the edges of a static interval (times are given to the millisecond)

Box = tuple[float, float, float, float]  # left, top, right, bottom in pixels

TIME_PATTERN = re.compile(r"<TIME (\d+):(\d+):(\d+(?:\.\d+)?) video \d+>")
BOX_PATTERN = re.compile(r"<BBOX ([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)>")
STATIC_PATTERN = re.compile(r"more than (\d+(?:\.\d+)?) seconds")
FIXTURE_VERB_PATTERN = re.compile(r"\bdid I (open|close)\b")
VERB_STEM = {"open": "open", "close": "clos"}  # the paper: narrations whose verb "includes" open or close

# The narrations file is a pickled pandas table. Unpickling runs code, so only the globals it was found to need
# (listed with pickletools) may be loaded.
PICKLE_GLOBALS = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
}


@dataclass(frozen=True)
class Query:
    """The object a question points at: a box in one frame of one video."""

    video: str
    box: Box
    seconds: float

    @property
    def frame(self) -> int:
        return round(self.seconds * FPS)


@dataclass(frozen=True)
class Mask:
    """One annotated object box (`mask_info.json`)."""

    mask_id: str
    frame: int
    box: Box
    location: tuple[float, float, float] | None
    fixture: str | None


@dataclass(frozen=True)
class Track:
    """One movement of an object, from pick-up to put-down."""

    track_id: str
    start: float
    end: float
    masks: tuple[str, ...]


@dataclass(frozen=True)
class Association:
    """All movements of one object in one video, ordered by start time."""

    assoc_id: str
    name: str
    tracks: tuple[Track, ...]


@dataclass(frozen=True)
class Resolution:
    """Which association a query box was tied to, if any."""

    assoc_id: str | None
    iou: float
    frame_gap: int | None
    candidate_assocs: int  # distinct associations among the masks that pass both thresholds


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        return super().find_class(module, name)


def load_pickled_table(path: Path) -> pd.DataFrame:
    """Load a pickled pandas table, refusing any global outside `PICKLE_GLOBALS`."""
    return _RestrictedUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def parse_time(hours: str, minutes: str, seconds: str) -> float:
    """Seconds from the three fields of a `<TIME hh:mm:ss.fff video n>` tag."""
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def times_in(text: str) -> list[float]:
    """Every `<TIME ...>` of a text, in seconds, in order."""
    return [parse_time(*match.groups()) for match in TIME_PATTERN.finditer(text)]


def first_box(text: str) -> Box | None:
    """The first `<BBOX left top right bottom>` of a text."""
    match = BOX_PATTERN.search(text)
    return tuple(float(value) for value in match.groups()) if match else None


def query_of(question: dict) -> Query | None:
    """The first box and the first time of a question, in its first video; None if either is missing."""
    box, times = first_box(question["question"]), times_in(question["question"])
    if box is None or not times:
        return None
    return Query(question["inputs"]["video 1"]["id"], box, times[0])


def swap_axes(box: Box) -> Box:
    """The same box with x and y exchanged: (top, left, bottom, right) read as (left, top, right, bottom)."""
    return box[1], box[0], box[3], box[2]


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (left, top, right, bottom)."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def masks_of(mask_info: dict, video: str) -> list[Mask]:
    """The annotated boxes of one video."""
    masks = []
    for mask_id, entry in mask_info.get(video, {}).items():
        location = entry.get("3d_location")
        fixture = entry.get("fixture")
        masks.append(Mask(mask_id, int(entry["frame_number"]), tuple(float(v) for v in entry["bbox"]),
                          tuple(float(v) for v in location) if location else None,
                          None if fixture in (None, "Null") else fixture))
    return masks


def associations_of(assoc_info: dict, video: str) -> dict[str, Association]:
    """The object associations of one video, each with its tracks ordered by start time."""
    result = {}
    for assoc_id, entry in assoc_info.get(video, {}).items():
        tracks = sorted((Track(t["track_id"], float(t["time_segment"][0]), float(t["time_segment"][1]),
                               tuple(t["masks"])) for t in entry["tracks"]), key=lambda t: (t.start, t.end))
        result[assoc_id] = Association(assoc_id, entry["name"], tuple(tracks))
    return result


def mask_owners(associations: dict[str, Association]) -> dict[str, str]:
    """Association id of every mask id that belongs to a track."""
    return {mask_id: assoc.assoc_id for assoc in associations.values() for track in assoc.tracks
            for mask_id in track.masks}


def resolve(query: Query, masks: list[Mask], owners: dict[str, str], max_frame_gap: int = MAX_FRAME_GAP,
            min_iou: float = MIN_IOU) -> Resolution:
    """Tie a query box to the association of the best-overlapping mask within `max_frame_gap` frames.

    Only masks that belong to an association count. The best mask has the highest IoU, then the smallest gap.
    """
    candidates = []  # (-iou, gap, association)
    for mask in masks:
        gap = abs(mask.frame - query.frame)
        if gap > max_frame_gap or mask.mask_id not in owners:
            continue
        overlap = box_iou(query.box, mask.box)
        if overlap >= min_iou:
            candidates.append((-overlap, gap, owners[mask.mask_id]))
    if not candidates:
        return Resolution(None, 0.0, None, 0)
    best = min(candidates)
    return Resolution(best[2], -best[0], best[1], len({c[2] for c in candidates}))


def movement_count(assoc: Association) -> int:
    """How many times the object changed location: its number of tracks ("distinct movements" in the paper)."""
    return len(assoc.tracks)


def _end_location(track: Track, masks: dict[str, Mask], last: bool) -> tuple[float, float, float] | None:
    """3D location of the latest (`last`) or earliest mask of a track that has one."""
    located = [masks[m] for m in track.masks if m in masks and masks[m].location is not None]
    if not located:
        return None
    return (max if last else min)(located, key=lambda m: m.frame).location


def passes_retention_filter(assoc: Association, masks: dict[str, Mask], min_gap_s: float = MIN_TRACK_GAP_S,
                            max_shift_m: float = MAX_SHIFT_M) -> bool | None:
    """The paper's rule for keeping an object in movement counting; None if a needed 3D location is missing.

    Consecutive tracks must be at least `min_gap_s` apart, and the object may move at most `max_shift_m` between
    the end of one track and the start of the next.
    """
    for before, after in zip(assoc.tracks, assoc.tracks[1:]):
        if after.start - before.end < min_gap_s:
            return False
    for before, after in zip(assoc.tracks, assoc.tracks[1:]):
        put, picked = _end_location(before, masks, last=True), _end_location(after, masks, last=False)
        if put is None or picked is None:
            return None
        if math.dist(put, picked) > max_shift_m:
            return False
    return True


def static_intervals(assoc: Association) -> list[tuple[float, float]]:
    """Intervals in which the object stays still after its first movement; the last one never ends."""
    tracks = assoc.tracks
    gaps = [(before.end, after.start) for before, after in zip(tracks, tracks[1:])]
    return gaps + [(tracks[-1].end, math.inf)] if tracks else []


def qualifying_choices(assoc: Association, choice_times: list[float], min_static_s: float,
                       tolerance_s: float = TIME_TOLERANCE_S) -> list[int]:
    """Indices of the choice times from which the object stays still for at least `min_static_s` seconds.

    A choice qualifies if it falls, after the first movement, inside a static interval that lasts at least
    `min_static_s` more seconds (the paper's positive interval runs from the start of the longest static
    interval to its end minus the required duration).
    """
    result = []
    for index, time in enumerate(choice_times):
        for start, end in static_intervals(assoc):
            if start - tolerance_s <= time <= end and end - time >= min_static_s - tolerance_s:
                result.append(index)
                break
    return result


def required_static_seconds(question_text: str) -> float | None:
    """The X of "remain static for more than X seconds"."""
    match = STATIC_PATTERN.search(question_text)
    return float(match.group(1)) if match else None


def fixture_verb(question_text: str) -> str | None:
    """"open" or "close", as asked by a fixture-interaction-counting question."""
    match = FIXTURE_VERB_PATTERN.search(question_text)
    return match.group(1) if match else None


def narrations_with_verb(rows: list[dict], verb: str) -> list[dict]:
    """The narrations of which some verb includes `verb` ("open" or "close"), in time order."""
    stem = VERB_STEM[verb]
    chosen = [row for row in rows if any(stem in v for v in row["verbs"])]
    return sorted(chosen, key=lambda row: row["start_timestamp"])


def itinerary_hops(answer: str) -> int:
    """Number of moves in an itinerary answer ("from A to B, then from B to C" has two)."""
    return len([part for part in answer.split(", then ") if part.strip().startswith("from ")])

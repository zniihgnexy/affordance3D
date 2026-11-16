"""Geometry read at query time (pipeline blocks 7, 8, 13) and the storage table, for notebooks/mem/pipeline_examples.ipynb.

Moved unchanged from report/mem_steps_helpers.py, which re-exports it.
"""

from __future__ import annotations

import io
import pickle
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance3d.frames import Frame
from affordance3d.mem import pipeline as P
from affordance3d.mem import tokens as T
from report.mem_demo_helpers import ScanView, _draw_entry, contact_error_cm, stand_free_in_scan


# ---------------------------------------------------------------- blocks 7, 8, 13: geometry read at query time

def readout_entry(scene: P.Scene, memory: T.L2Memory, readout: T.Readout, truth_part: int) -> P.MemoryEntry:
    """A read-out in the shape the display and scoring helpers expect.

    `truth_part` (ground truth) is used only to score the contact point and to draw the oracle motion.
    """
    token, element = memory.tokens[readout.token], scene.elements[truth_part]
    seen = tuple(memory.times[memory.tokens[i].region.keyframe] for i in memory.tokens_of(int(memory.part_of[readout.token])))
    return P.MemoryEntry(truth_part, element.annot_id, T.ACTIONS[token.region.action], scene.task_of(element.annot_id),
                         memory.times[token.region.keyframe], readout.pixel, token.region.pixels, seen, readout.contact,
                         readout.normal, readout.stand, scene.motions.get(element.annot_id))


def readout_memory(scene: P.Scene, memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray) -> P.Memory:
    """Every part of the L2 memory read out from its evidence token, for the entry figure and the scoreboard."""
    entries = []
    for part in range(memory.n_parts):
        token = memory.evidence(part)
        readout = T.read_geometry(memory, token, frames, scene.up)
        if readout is not None:
            entries.append(readout_entry(scene, memory, readout, int(truth[token])))
    return P.Memory(memory.times, tuple(entries), memory.floor, np.zeros((0, 3)))


def readout_figure(scene: P.Scene, titles: Sequence[str], entries: Sequence[P.MemoryEntry], frames: dict[str, Frame],
                   view: ScanView) -> plt.Figure:
    """Geometry read for each chosen token, drawn on its evidence keyframe: whole frame above, zoom below."""
    fig, axes = plt.subplots(2, len(entries), figsize=(3.5 * len(entries), 8.0), squeeze=False)
    for column, (title, entry) in enumerate(zip(titles, entries)):
        frame = frames[entry.keyframe]
        for axis in axes[:, column]:
            axis.imshow(frame.rgb)
            _draw_entry(axis, scene, entry, frame, stand_free_in_scan(entry, view))
            axis.axis("off")
        u, v = entry.pixel
        axes[1, column].set_xlim(u - 90, u + 90)
        axes[1, column].set_ylim(v + 90, v - 90)
        axes[0, column].set_title(title, fontsize=8)
        axes[1, column].set_title(f"zoom · {entry.action} · evidence keyframe {entry.keyframe} s\n"
                                  f"contact {contact_error_cm(scene, entry):.1f} cm from the part it marks", fontsize=8)
    fig.suptitle("red: contact point   cyan: surface normal   yellow: part motion (ORACLE)   ring: stand point", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    return fig


def _png_bytes(image: np.ndarray) -> int:
    return len(cv2.imencode(".png", image)[1])


def storage_table(scene: P.Scene, scan_path: Path, keyframes: Sequence[str], frames: Sequence[Frame],
                  action_maps: Sequence[np.ndarray], memory: T.L2Memory, explicit: P.Memory) -> pd.DataFrame:
    """Measured sizes: what the L2 memory keeps, next to the explicit entries of v1 and the laser scan."""
    index_bytes = len(pickle.dumps([(t.time_s, t.region) for t in memory.tokens] + [memory.part_of]))
    poses = np.array([np.concatenate([f.cam_to_world.ravel(), f.intrinsics.ravel()]) for f in frames])
    buffer = io.BytesIO()
    np.save(buffer, poses)
    rows = [("L2 memory", "keyframe images (as in the dataset)", sum(Path(scene.vid.rgb_path(t)).stat().st_size for t in keyframes)),
            ("L2 memory", "depth maps (16-bit PNG, millimetres)", sum(_png_bytes((f.depth_m * 1000).astype(np.uint16)) for f in frames)),
            ("L2 memory", "affordance maps (PNG)", sum(_png_bytes((a + 1).astype(np.uint8)) for a in action_maps)),
            ("L2 memory", "camera poses, intrinsics", buffer.getbuffer().nbytes),
            ("L2 memory", f"part tokens: features of {len(memory.tokens)} tokens (16-bit)", memory.feature_bytes()),
            ("L2 memory", "part tokens: thin index", index_bytes),
            ("for comparison", f"v1 explicit entries ({len(explicit.entries)} parts)", len(pickle.dumps(explicit.entries))),
            ("for comparison", "laser scan of the room (ground truth, never an input)", scan_path.stat().st_size)]
    table = pd.DataFrame(rows, columns=["group", "what", "bytes"])
    table["size"] = [f"{b / 1e6:.2f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB" for b in table["bytes"]]
    return table.drop(columns="bytes").set_index(["group", "what"])

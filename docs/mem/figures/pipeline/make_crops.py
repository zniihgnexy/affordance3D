"""Make the two real crops of the pipeline figure from one correct P4 answer (CPU; run from the repository root).

    python docs/mem/figures/pipeline/make_crops.py [output folder]

Reads phase-1 outputs under runs/ (not versioned): the scene, its steady keyframes, the P4 regions and the
first P4 reading of task 14 in scene 421254, "Open the top left drawer of the cabinet located to the left
of the TV", which picks the right handle.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (imported quietly before the pipeline, which needs it)

from affordance3d.mem import keyframes
from affordance3d.mem import pipeline as P
from report import mem_steps_helpers as H

VISIT, TASK_INDEX = "421254", 14
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/mem/figures/pipeline")
INPUT_SCALE = 0.75  # the whole keyframe is shown small, so a smaller file loses nothing in print
ZOOM = (300, 255, 620, 755)  # the two cabinets at full resolution, so the numbered marks stay legible
CHOSEN_RGB, CHOSEN_PAD, CHOSEN_WIDTH = (0, 112, 192), 6, 5  # the box the model chose, drawn in blue

OUT.mkdir(parents=True, exist_ok=True)
result = json.loads(Path(f"runs/mem_phase1/{VISIT}.json").read_text())
record = json.loads(Path(f"runs/mem_phase1/controls/{VISIT}.json").read_text())
row = record["arms"]["p4n"][TASK_INDEX]
if not row["hit"]:
    raise SystemExit("the example reading no longer picks the right part; choose another task")
scene = P.load_scene(Path("data/scenefun3d"), VISIT, result["video"])
steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
frames = [P.input_frame(scene, t, np.zeros(P.frame_hw(scene, t, P.PIXEL_STEP), dtype=np.float32)) for t in steady]
with np.load(f"runs/mem_phase1/diagnostics/{VISIT}.npz") as saved:
    p4_keyframe, p4_boxes = saved["keyframe/P4"], saved["boxes/P4"]
shown = [steady.index(t) for t in row["keyframes_shown"]]
k = steady.index(row["frame"])
regions = [(int(kk), tuple(int(v) for v in b)) for kk, b in zip(p4_keyframe, p4_boxes) if int(kk) in shown]
images, _ = H.mark_boxes(frames, regions, shown)

keyframe = Image.fromarray(frames[k].rgb)
keyframe = keyframe.resize((int(keyframe.width * INPUT_SCALE), int(keyframe.height * INPUT_SCALE)), Image.LANCZOS)
keyframe.save(OUT / "pipeline_input.jpg", quality=92)

answer = Image.fromarray(np.asarray(images[shown.index(k)]).copy())
left, top, right, bottom = row["box"]
ImageDraw.Draw(answer).rectangle([left - CHOSEN_PAD, top - CHOSEN_PAD, right + CHOSEN_PAD, bottom + CHOSEN_PAD],
                                 outline=CHOSEN_RGB, width=CHOSEN_WIDTH)
answer.crop(ZOOM).save(OUT / "pipeline_answer_zoom.jpg", quality=92)
print("task:", record["tasks"][TASK_INDEX], "| frame", row["frame"], "| marks", row["marks"], "| written to", OUT)

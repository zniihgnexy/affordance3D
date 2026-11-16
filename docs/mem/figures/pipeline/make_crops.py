"""Make the two real crops of the pipeline figure from one correct answer (CPU; run from the repository root).

    python docs/mem/figures/pipeline/make_crops.py [output folder]

Reads outputs under runs/ (not versioned): the scene, its steady keyframes, the SAM 3 part map and the Qwen3-VL-8B
reading of tool P4 (arm p4t_sam3, discussion draft section 24.9) for task 14 in scene 421254, "Open the top left drawer
of the cabinet located to the left of the TV", which picks the right handle. The numbered boxes are rebuilt as that
reading showed them (every SAM 3 box on the three keyframes shown, in the map's order), and the rebuilt numbering is
checked against the box the model chose.
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
from affordance3d.mem import tool_maps as TM
from report import mem_steps_helpers as H

VISIT, TASK_INDEX, ARM, TOOL = "421254", 14, "p4t_sam3", "sam3"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/mem/figures/pipeline")
INPUT_SCALE = 0.75  # the whole keyframe is shown small, so a smaller file loses nothing in print
ZOOM = (300, 255, 620, 755)  # the two cabinets at full resolution, so the numbered marks stay legible
CHOSEN_RGB, CHOSEN_PAD, CHOSEN_WIDTH = (0, 112, 192), 6, 5  # the box the model chose, drawn in blue

OUT.mkdir(parents=True, exist_ok=True)
result = json.loads(Path(f"runs/mem_phase1/{VISIT}.json").read_text())
record = json.loads(Path(f"runs/mem_phase1/controls_8b/{VISIT}.json").read_text())
row = record["arms"][ARM][TASK_INDEX]
if not row["hit"]:
    raise SystemExit("the example reading no longer picks the right part; choose another task")
scene = P.load_scene(Path("data/scenefun3d"), VISIT, result["video"])
steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
frames = [P.input_frame(scene, t, np.zeros(P.frame_hw(scene, t, P.PIXEL_STEP), dtype=np.float32)) for t in steady]

tool_map = TM.read_tool_map(Path(f"runs/mem_phase1/tool_maps/{TOOL}/{VISIT}.json"))
candidates = TM.candidates_of(TM.tool_detections(tool_map), frames[0].image_hw)
top = [steady.index(t) for t in row["keyframes_shown"]]
kept = [c for c in candidates if c.keyframe in top]
images, marks = H.mark_boxes(frames, [(c.keyframe, c.box) for c in kept], top)
if len(kept) != row["marks"] or len(marks) != row["marks"]:
    raise SystemExit("the reading did not number every box on its keyframes; the numbering cannot be rebuilt without scores")
chosen = kept[marks[row["image"] - 1]]
if list(chosen.box) != row["box"] or steady[chosen.keyframe] != row["frame"]:
    raise SystemExit("the rebuilt numbering does not match the box the model chose")
k = chosen.keyframe

keyframe = Image.fromarray(frames[k].rgb)
keyframe = keyframe.resize((int(keyframe.width * INPUT_SCALE), int(keyframe.height * INPUT_SCALE)), Image.LANCZOS)
keyframe.save(OUT / "pipeline_input.jpg", quality=92)

answer = Image.fromarray(np.asarray(images[top.index(k)]).copy())
left, top_edge, right, bottom = row["box"]
ImageDraw.Draw(answer).rectangle([left - CHOSEN_PAD, top_edge - CHOSEN_PAD, right + CHOSEN_PAD, bottom + CHOSEN_PAD],
                                 outline=CHOSEN_RGB, width=CHOSEN_WIDTH)
answer.crop(ZOOM).save(OUT / "pipeline_answer_zoom.jpg", quality=92)
print(f"task: {record['tasks'][TASK_INDEX]} | frame {row['frame']} | {row['marks']} marks, chose {row['image']} | written to {OUT}")

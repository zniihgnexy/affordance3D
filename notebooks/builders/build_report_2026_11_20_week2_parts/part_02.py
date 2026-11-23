# Cells of build_report_2026_11_20_week2.py, part 2 (section 1: the two videos and the interactive 3D memory).

md(r"""
---
# 1. Demo: the memory of one scene (421069, kitchen and living room)

**What is drawn:**
- **Colour:** the part, coloured by action: blue pull, orange push or press, purple turn, green open a door or window.
- **Arrow:** the memory's vector, sliding back and forth once every 1.6 s.
- **Tag:** the object (named from its task text) and the action.

The highlights stay on the parts while the camera moves, because the memory lives in 3D world coordinates, not on any
one image: written once, readable from any view.

**Caveats:**
- **A demo, not a test:** part masks and names come from the annotation; the formal test is section 3.
- **The memory's own errors show:**
  - the oven doors get a straight "pull" arrow, but they flip down;
  - the tap arcs are about 80° off;
  - 4 of the 6 hob knobs without a task have their axis within 15°.
""")

code(r"""
display(Video(str(ASSETS / "week2_walkthrough_421069.mp4"), embed=True, width=540, html_attributes="controls loop muted playsinline"))
w = media["walkthrough"]
print(f"walkthrough: {w['frames']} frames; at least one memory part in view in {w['frames_with_parts']} of them")
check("walkthrough frames", w["frames"], 1025)
turns = [r for r in demo_rows if (r["depth"], r["rule"], r["frames"]) == ("lidar", "gravity", "best") and r["vector"] and r["label"] == "rotate"]
taps, knobs = [r for r in turns if "tap" in r["task"].lower()], [r for r in turns if not r["task"]]
check("tap axis error, degrees (smaller of the two)", round(min(r["score"]["axis_deg"] for r in taps)), 80)
check("knobs without a task", len(knobs), 6)
check("knob axes within 15 degrees", sum(r["score"]["axis_deg"] <= 15 for r in knobs), 4)
""")

code(r"""
display(Video(str(ASSETS / "week2_memory_orbit_421069.mp4"), embed=True, width=720, html_attributes="controls loop muted playsinline"))
o = media["orbit"]
print(f"orbit: {o['frames']} frames over the laser scan, ceiling removed; median {o['parts_drawn_median']:.0f} of {media['parts']} parts in view")
""")

md(r"""
## 1.1 Interactive 3D memory: `report/assets/week2_memory_421069.rrd`

The same content as the orbit video, as a Rerun file you can turn, zoom and switch layer by layer. The arrows play on
the timeline `loop`, and the ceiling layer is hidden until you tick it in the entity tree.

**How to open it, either way:**
- **Install the viewer:** `pip install rerun-sdk==0.43.2`, then from the repository `rerun report/assets/week2_memory_421069.rrd`.
- **No install:** open https://app.rerun.io/version/0.43.2/ and drag the file into the page.

A warning that ffmpeg is missing is harmless: the file holds no video.

Below are three fixed views of the same geometry (drawn here from the exporter's own functions, orthographic and x-ray:
parts behind walls are shown on top), then an optional cell that opens the file inside this notebook.
""")

code(r"""
from report import week2_interaction as G
from report import week2_media as M
from report import week2_rrd as W

RRD = load_config(ROOT / "configs" / "report" / "week2_rrd.yaml")
style = G.Style.from_config(load_config(ROOT / RRD["style_config"])["walkthrough"]["interaction"])
parts = M.load_parts(ROOT / DEMO["outputs"] / DEMO["name"], RRD["variant"], truths)
xyz, rgb, below = W._scan_layers(scene, RRD["scan"]["cut_above_floor_m"])
rotation, offset = W.world_frame(xyz[below], scene.up)
keep = np.flatnonzero(below)[W.downsample(int(below.sum()), 400_000, RRD["seed"])]
up = rotation.T @ np.array([0.0, 0.0, 1.0])
glyphs = []
for part, (action, category, swing) in G.part_actions(parts, style).items():
    p = next(q for q in parts if q.part == part)
    title = G.object_name(p.task, style.objects.get(p.label, p.label.replace("_", " ")), style.max_words)
    line = G.glyph_at(p, swing, category, 1.0, p.normal if p.normal is not None else up, style)
    glyphs.append({"points": W.to_world(scene.part_points(part), rotation, offset), "line": W.to_world(line, rotation, offset),
                   "colour": np.array(style.colours[category]) / 255.0, "tag": W.label_text(title, action)})
fig = WR.memory_snapshots(W.to_world(xyz[keep], rotation, offset), rgb[keep], glyphs,
                          [("top-down, with tags", 0, 90), ("oblique, one side", 35, 38), ("oblique, other side", 215, 38)])
display(fig)
plt.close(fig)
rrd = json.loads((ROOT / "runs" / "report_media" / "week2_rrd" / "rrd.json").read_text())
print(f"{rrd['bytes'] / 1e6:.1f} MB; {rrd['parts']} parts; {rrd['world/scan/room']:,} room points + {rrd['world/scan/ceiling']:,} ceiling points")
check("rrd parts", rrd["parts"], 29)
check("rrd size MB", round(rrd["bytes"] / 1e6), 38)
""")

code(r"""
# Optional: the same file inside this notebook. Needs `pip install rerun-sdk==0.43.2 rerun-notebook==0.43.2` in the
# Jupyter environment; set SHOW_INLINE = True and run this cell by hand. It is off when the notebook is built, so that the
# 38 MB file is not stored in the notebook.
SHOW_INLINE = False
if SHOW_INLINE:
    from rerun_notebook import Viewer as RrdViewer

    viewer = RrdViewer(width="auto", height=600)
    display(viewer)
    viewer.block_until_ready()
    viewer.send_rrd((ASSETS / "week2_memory_421069.rrd").read_bytes())
else:
    print("inline viewer off (set SHOW_INLINE = True to open the .rrd here)")
""")

"""Display helpers for report/2026-11-10_mem_pipeline_demo.ipynb (mem branch).

The pipeline itself lives in affordance3d.mem.pipeline; this module only draws and scores it.
Anything that touches the laser scan here is ground truth: display or evaluation, never an input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree

from affordance3d.frames import Frame
from affordance3d.geometry import project, subsample_frame
from affordance3d.mem import pipeline as P
from affordance3d.mem.actionable import distance_to_part, floor_height, is_free
from affordance3d.mem.body import JOINT, BodyModel
from affordance3d.mem.depth_models import MonoDepth, depth_errors
from affordance3d.mem.humanoid import PosedHumanoid, approach_frames, pose_humanoid
from affordance3d.mem.labels import BACKGROUND, projected_axis
from affordance3d.mem.render import look_at, overlay_body, rasterize, splat, to_camera
from affordance3d.mem.tokens import ACTIONS as ACTION_NAMES
from affordance3d.scenefun3d import evenly_spaced

ACTIONS = list(ACTION_NAMES)
ACTION_COLOUR = dict(zip(ACTIONS, plt.get_cmap("tab10").colors))
ON_PART_M = 0.03
GT_UP = np.array([0.0, 0.0, 1.0])
CHECK_HW = (480, 600)
CHECK_K = np.array([[470.0, 0, 299.5], [0, 470.0, 239.5], [0, 0, 1]])
CHECK_AZIMUTHS_DEG = (-50, 50, -25, 25, -75, 75, 0)
COLLISION_M, NEAR_EFFECTOR_M = 0.02, 0.15


# ---------------------------------------------------------------- inputs

def pick_keyframes(scene: P.Scene, n: int) -> list[str]:
    """Evenly spaced in time, never chosen by labels (only frames that also have ground truth for scoring)."""
    times = [t for t in scene.vid.rgb_times if t in scene.gt_index.depth]
    return [times[i] for i in evenly_spaced(len(times), n)]


def scaled_mono_depths(scene: P.Scene, times: Sequence[str], model_id: str, cache_dir: Path) -> list[np.ndarray]:
    """Monocular depth at the working resolution, rescaled per frame by the device's LiDAR (cached on disk).

    A frame in which the device marks no LiDAR pixel as reliable takes the median scale of the other frames.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [t for t in times if not (cache_dir / f"{t}.npy").exists()]
    if missing:
        model = MonoDepth(model_id)
        for t in missing:
            depth = model.predict(iio.imread(scene.vid.rgb_path(t)), focal_px=float(scene.vid.intrinsics(t)[0, 0]))
            np.save(cache_dir / f"{t}.npy", depth[::P.PIXEL_STEP, ::P.PIXEL_STEP].astype(np.float16))
        del model
    return P.lidar_scaled_together([np.load(cache_dir / f"{t}.npy").astype(np.float32) for t in times],
                                   [P.lidar_depth(scene, t) for t in times])


def inputs_figure(scene: P.Scene, t: str, mono: np.ndarray, gt_frame: Frame) -> plt.Figure:
    lidar, gt = P.lidar_depth(scene, t), gt_frame.depth_m
    vmax = float(np.percentile(gt[gt > 0], 98))

    def error_cm(depth: np.ndarray) -> float:
        return 100 * depth_errors(depth, gt)["median_abs_m"]

    panels = [(gt_frame.rgb, "INPUT · RGB frame", None),
              (np.where(lidar > 0, lidar, np.nan), f"INPUT · iPad LiDAR depth (256×192)\nmedian error {error_cm(lidar):.1f} cm", "viridis"),
              (mono, f"INPUT · monocular depth × LiDAR scale\nmedian error {error_cm(mono):.1f} cm", "viridis"),
              (np.where(gt > 0, gt, np.nan), "GROUND TRUTH · depth rendered from the scan\n(scoring only)", "viridis")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6))
    for axis, (image, title, cmap) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=vmax if cmap else None)
        if cmap:
            plt.colorbar(shown, ax=axis, fraction=0.046, label="m")
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ground-truth scan, for display and scoring

@dataclass(frozen=True)
class ScanView:
    points: np.ndarray  # 1 cm voxels of the cropped laser scan
    colours: np.ndarray
    tree: cKDTree
    occupied: np.ndarray  # 5 cm voxels, for the free-space score
    floor: float


def scan_view(scene: P.Scene) -> ScanView:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene.scan_xyz[scene.crop]))
    cloud.colors = o3d.utility.Vector3dVector(scene.scan_rgb[scene.crop].astype(np.float64))
    fine = cloud.voxel_down_sample(0.01)
    points = np.asarray(fine.points)
    return ScanView(points, np.asarray(fine.colors), cKDTree(points),
                    np.asarray(cloud.voxel_down_sample(0.05).points), floor_height(points[:, 2]))


def contact_error_cm(scene: P.Scene, entry: P.MemoryEntry) -> float:
    return 100 * distance_to_part(entry.contact, scene.part_points(entry.part))


def stand_free_in_scan(entry: P.MemoryEntry, view: ScanView) -> bool:
    return is_free(entry.stand.position, view.occupied, GT_UP, max_points=2)


# ---------------------------------------------------------------- labels and memory entries

def _axis_ends(scene: P.Scene, entry: P.MemoryEntry, length: float = 0.15) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space segment that shows the part's motion: a push/pull arrow, or the rotation axis."""
    if entry.motion is None:
        return None
    direction = np.array(entry.motion["motion_dir"], dtype=float)
    direction /= np.linalg.norm(direction)
    if entry.motion["motion_type"] == "trans":
        return entry.contact, entry.contact + length * direction
    origin = scene.scan_xyz[entry.motion["motion_origin_idx"]]
    return origin - 0.8 * length * direction, origin + 0.8 * length * direction


def labels_figure(scene: P.Scene, frames: Sequence[Frame], labels: Sequence[np.ndarray], times: Sequence[str],
                  n: int = 4) -> plt.Figure:
    order = sorted(range(len(frames)), key=lambda i: -len(set(np.unique(labels[i])) - {BACKGROUND}))[:n]
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 5.6))
    for axis, i in zip(axes, order):
        overlay = (frames[i].rgb * 0.55).astype(np.uint8)
        parts = sorted(set(np.unique(labels[i])) - {BACKGROUND})
        for p in parts:
            overlay[labels[i] == p] = (np.array(ACTION_COLOUR[scene.elements[p].label]) * 255).astype(np.uint8)
        axis.imshow(overlay)
        for p in parts:
            m = scene.motions.get(scene.elements[p].annot_id)
            if m is not None:
                start, end = projected_axis(scene.scan_xyz[m["motion_origin_idx"]], np.array(m["motion_dir"]), frames[i], 0.15)
                axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5})
        axis.set_title(f"t = {times[i]} s · {len(parts)} parts", fontsize=9)
        axis.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", color=ACTION_COLOUR[a], label=a) for a in ACTIONS]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def entries_table(scene: P.Scene, memory: P.Memory, view: ScanView) -> pd.DataFrame:
    rows = [{"action": e.action, "task (the effect, from SceneFun3D)": e.task or "–", "best keyframe": f"{e.keyframe} s",
             "seen in": f"{len(e.seen_in)} keyframes", "motion": {"rot": "rotates", "trans": "slides"}.get(
                 (e.motion or {}).get("motion_type"), "–"),
             "contact error": f"{contact_error_cm(scene, e):.1f} cm",
             "stand point": "free" if stand_free_in_scan(e, view) else "blocked in scan"} for e in memory.entries]
    return pd.DataFrame(rows).sort_values(["action", "task (the effect, from SceneFun3D)"]).reset_index(drop=True)


def _draw_entry(axis: plt.Axes, scene: P.Scene, entry: P.MemoryEntry, frame: Frame, ring_ok: bool) -> None:
    u, v = entry.pixel
    start, normal_end = projected_axis(entry.contact, entry.normal, frame, 0.12)
    axis.annotate("", xy=normal_end, xytext=start, arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 1.8})
    ends = _axis_ends(scene, entry, 0.12)
    if ends is not None:
        uv, _, inside = project(np.stack(ends), frame)
        if inside.all():
            axis.annotate("", xy=uv[1], xytext=uv[0], arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 1.8})
    axis.plot(u, v, "o", color="red", ms=5, mec="white", mew=0.6)
    angles = np.linspace(0, 2 * np.pi, 40)
    side = np.cross(scene.up, entry.stand.facing)
    ring = entry.stand.position + 0.2 * (np.outer(np.cos(angles), entry.stand.facing) + np.outer(np.sin(angles), side))
    ring_uv, _, inside = project(ring, frame)
    if inside.any():
        axis.plot(ring_uv[inside, 0], ring_uv[inside, 1], color="lime" if ring_ok else "red", lw=1.5)


def entries_figure(scene: P.Scene, memory: P.Memory, frames: dict[str, Frame], view: ScanView) -> plt.Figure:
    """One entry per action type (the one seen with most pixels): whole keyframe above, zoom below."""
    shown = [max((e for e in memory.entries if e.action == a), key=lambda e: e.pixels)
             for a in ACTIONS if any(e.action == a for e in memory.entries)]
    fig, axes = plt.subplots(2, len(shown), figsize=(2.55 * len(shown), 7.0))
    for column, entry in enumerate(shown):
        frame = frames[entry.keyframe]
        for row in range(2):
            axis = axes[row, column]
            axis.imshow(frame.rgb)
            _draw_entry(axis, scene, entry, frame, stand_free_in_scan(entry, view))
            if row:
                axis.set_xlim(entry.pixel[0] - 80, entry.pixel[0] + 80)
                axis.set_ylim(entry.pixel[1] + 80, entry.pixel[1] - 80)
            axis.axis("off")
        axes[0, column].set_title(f"{entry.action}\n{(entry.task or '(no task text)')[:30]}", fontsize=8)
    fig.suptitle("red: contact point   cyan: surface normal   yellow: part motion (ORACLE)   ring: stand point", fontsize=9)
    fig.tight_layout()
    return fig


def memory_3d_figure(scene: P.Scene, memory: P.Memory, view: ScanView, n_points: int = 40000) -> go.Figure:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(view.points), min(n_points, len(view.points)), replace=False)
    colours = [f"rgb({r},{g},{b})" for r, g, b in (view.colours[idx] * 255).astype(int)]
    path = np.array([P.input_camera(scene, t)[1][:3, 3] for t in scene.vid.rgb_times[::5]])
    traces = [
        go.Scatter3d(x=view.points[idx, 0], y=view.points[idx, 1], z=view.points[idx, 2], mode="markers",
                     marker={"size": 1.2, "color": colours}, hoverinfo="skip", name="laser scan (ground truth, display only)"),
        go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", line={"color": "orange", "width": 4},
                     hoverinfo="skip", name="camera path (ARKit, input)"),
    ]
    xs, ys, zs = [], [], []
    for e in memory.entries:
        for a, b in ((e.stand.position, e.stand.position + 1.0 * scene.up), (e.stand.position + 1.0 * scene.up, e.contact)):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"color": "black", "width": 3}, hoverinfo="skip",
                               name="stand point → contact"))
    for action in ACTIONS:
        entries = [e for e in memory.entries if e.action == action]
        if not entries:
            continue
        contacts = np.array([e.contact for e in entries])
        r, g, b = (np.array(ACTION_COLOUR[action]) * 255).astype(int)
        traces.append(go.Scatter3d(x=contacts[:, 0], y=contacts[:, 1], z=contacts[:, 2], mode="markers",
                                   marker={"size": 6, "color": f"rgb({r},{g},{b})", "line": {"color": "black", "width": 1}},
                                   text=[e.task or "(no task text)" for e in entries],
                                   hovertemplate=f"{action}<br>%{{text}}<extra></extra>", name=action))
    figure = go.Figure(traces)
    figure.update_layout(height=640, margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"x": 0, "y": 1, "itemsizing": "constant"},
                         scene={"aspectmode": "data", "xaxis_visible": False, "yaxis_visible": False, "zaxis_visible": False,
                                    "camera": {"eye": {"x": 0.9, "y": -1.2, "z": 1.3}}})
    return figure


# ---------------------------------------------------------------- retrieval

def retrieval_figure(scene: P.Scene, memory: P.Memory, queries: Sequence[str], frames: dict[str, Frame],
                     labels: dict[str, np.ndarray], view: ScanView) -> plt.Figure:
    fig, axes = plt.subplots(2, len(queries), figsize=(3.3 * len(queries), 7.6))
    for column, query in enumerate(queries):
        found = P.retrieve(memory.entries, scene.tasks, query)
        for axis in axes[:, column]:
            axis.axis("off")
        if not found:
            axes[0, column].set_title(f'"{query}"\nnot in memory', fontsize=8)
            continue
        entry = max(found, key=lambda e: e.pixels)
        frame = frames[entry.keyframe]
        image = (frame.rgb * 0.6).astype(np.uint8)
        for e in found:
            if e.keyframe == entry.keyframe:
                image[labels[e.keyframe] == e.part] = (np.array(ACTION_COLOUR[e.action]) * 255).astype(np.uint8)
        for row in range(2):
            axes[row, column].imshow(image)
            _draw_entry(axes[row, column], scene, entry, frame, stand_free_in_scan(entry, view))
        u, v = entry.pixel
        axes[1, column].set_xlim(u - 90, u + 90)
        axes[1, column].set_ylim(v + 90, v - 90)
        motion = {"rot": "rotates", "trans": "slides"}.get((entry.motion or {}).get("motion_type"), "no motion label")
        axes[0, column].set_title(f'"{query[:44]}"', fontsize=8)
        axes[1, column].set_title(f"{entry.action} · {motion} · {len(found)} part(s)\nevidence: keyframe {entry.keyframe} s, "
                                  f"seen in {len(entry.seen_in)}", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- humanoid

@dataclass(frozen=True)
class Shot:
    query: str
    entry: P.MemoryEntry
    human: PosedHumanoid
    time: str  # display frame
    frame: Frame  # display frame at the working resolution, with input depth
    in_view: int  # of the three body points (feet, hips, head) inside the display frame


def make_shots(scene: P.Scene, memory: P.Memory, model: BodyModel, queries: Sequence[str], mono_id: str,
               cache_dir: Path, stride: int = 3, min_in_view: int = 2) -> list[Shot]:
    """Pose a humanoid for each query and pick, from the whole video, the frame that shows it best (inputs only)."""
    pairs = []
    for query in queries:
        found = P.retrieve(memory.entries, scene.tasks, query)
        if found:
            pairs.append((query, max(found, key=lambda e: e.pixels)))
    return shots_for(scene, pairs, memory.floor, model, mono_id, cache_dir, stride, min_in_view)


def shots_for(scene: P.Scene, pairs: Sequence[tuple[str, P.MemoryEntry]], floor: float, model: BodyModel, mono_id: str,
              cache_dir: Path, stride: int = 3, min_in_view: int = 2) -> list[Shot]:
    """A humanoid for each (query, entry) pair, in the video frame that shows it best (inputs only)."""
    candidates = scene.vid.rgb_times[::stride]
    chosen = []
    for query, entry in pairs:
        scores = {t: P.display_score(scene, entry, t) for t in candidates}
        best = max(scores, key=lambda t: scores[t][0])
        if scores[best][1] >= min_in_view:
            chosen.append((query, entry, best, scores[best][1]))
    depths = scaled_mono_depths(scene, [t for _, _, t, _ in chosen], mono_id, cache_dir)
    shots = []
    for (query, entry, t, in_view), depth in zip(chosen, depths):
        human = pose_humanoid(model, entry.contact, entry.normal, entry.stand.position, entry.stand.facing,
                              floor, scene.up, entry.action)
        shots.append(Shot(query, entry, human, t, P.input_frame(scene, t, depth), in_view))
    return shots


def effector_position(human: PosedHumanoid) -> np.ndarray:
    if human.effector == "right foot":
        return human.joints[JOINT["right_foot"]]
    side = human.effector.split()[0]
    return 0.5 * (human.joints[JOINT[f"{side}_wrist"]] + human.joints[JOINT[f"{side}_middle1"]])


def shots_table(scene: P.Scene, shots: Sequence[Shot], view: ScanView) -> pd.DataFrame:
    rows = []
    for shot in shots:
        human = shot.human
        away = np.linalg.norm(human.vertices - human.target, axis=1) >= NEAR_EFFECTOR_M
        hits = view.tree.query(human.vertices[away])[0] < COLLISION_M
        rows.append({"query": shot.query, "acts with": f"{human.effector}, {human.posture}",
                     "hand/foot to true part": f"{100 * distance_to_part(effector_position(human), scene.part_points(shot.entry.part)):.1f} cm",
                     "feet above true floor": f"{100 * (human.vertices[:, 2].min() - view.floor):+.1f} cm",
                     "body inside scan geometry": f"{hits.mean():.1%}"})
    return pd.DataFrame(rows)


def _check_camera(shot: Shot, view: ScanView, up: np.ndarray, distance: float, height: float) -> np.ndarray:
    centre = 0.5 * (shot.entry.contact + shot.entry.stand.position) + 0.4 * up
    back = -shot.entry.stand.facing

    def blocked(eye: np.ndarray) -> int:
        samples = eye + np.linspace(0.0, 1.0, 30)[:, None] * (centre - eye)
        keep = np.linalg.norm(samples - centre, axis=1) > 0.9
        return sum(len(n) for n in view.tree.query_ball_point(samples[keep], 0.15))

    eyes = []
    for degrees in CHECK_AZIMUTHS_DEG:
        angle = np.radians(degrees)
        eyes.append(centre + distance * (back * np.cos(angle) + np.cross(up, back) * np.sin(angle)) + height * up)
    return look_at(min(eyes, key=blocked), centre, up)


def _check_background(scene: P.Scene, shot: Shot, view: ScanView, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The laser scan around the part seen from the check camera; the true part is magenta."""
    near = view.tree.query_ball_point(shot.entry.contact, 3.5)
    part = scene.part_points(shot.entry.part)
    points = np.vstack([view.points[near], part])
    colours = np.vstack([view.colours[near], np.tile([1.0, 0.0, 1.0], (len(part), 1))])
    image, depth = splat(to_camera(points, pose), colours, CHECK_K, CHECK_HW, radius_px=1)
    return (image * 255).astype(np.uint8), np.where(np.isfinite(depth), depth, 0.0)


def _body_layer(vertices: np.ndarray, model: BodyModel, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray,
                pose: np.ndarray, margin: float) -> np.ndarray:
    body_depth, shade = rasterize(to_camera(vertices, pose), model.faces, intrinsics, rgb.shape[:2])
    return overlay_body(rgb, depth, body_depth, shade, margin_m=margin, alpha=0.88)


def _mark(image: np.ndarray, scene: P.Scene, entry: P.MemoryEntry, intrinsics: np.ndarray, pose: np.ndarray,
          motion: bool) -> None:
    camera = Frame(0, image, np.zeros(image.shape[:2], np.float32), intrinsics, pose)
    uv, _, inside = project(entry.contact[None], camera)
    if inside[0]:
        cv2.circle(image, tuple(np.rint(uv[0]).astype(int).tolist()), 4, (255, 0, 0), -1, cv2.LINE_AA)
    ends = _axis_ends(scene, entry) if motion else None
    if ends is not None:
        uv, _, inside = project(np.stack(ends), camera)
        if inside.all():
            a, b = (tuple(np.rint(p).astype(int).tolist()) for p in uv)
            cv2.arrowedLine(image, a, b, (255, 230, 0), 2, cv2.LINE_AA, tipLength=0.3)


def humanoid_figure(scene: P.Scene, shots: Sequence[Shot], model: BodyModel, view: ScanView) -> plt.Figure:
    """Final pose of every shot: pasted into the video frame (top) and checked inside the laser scan (bottom)."""
    fig, axes = plt.subplots(2, len(shots), figsize=(3.6 * len(shots), 7.9), gridspec_kw={"height_ratios": [1.65, 1]},
                             squeeze=False)
    for column, shot in enumerate(shots):
        frame = shot.frame
        top = _body_layer(shot.human.vertices, model, frame.rgb, frame.depth_m, frame.intrinsics, frame.cam_to_world, 0.05)
        _mark(top, scene, shot.entry, frame.intrinsics, frame.cam_to_world, motion=True)
        pose = _check_camera(shot, view, scene.up, distance=2.4, height=0.9)
        background, depth = _check_background(scene, shot, view, pose)
        bottom = _body_layer(shot.human.vertices, model, background, depth, CHECK_K, pose, 0.03)
        _mark(bottom, scene, shot.entry, CHECK_K, pose, motion=False)
        for axis, image in zip(axes[:, column], (top, bottom)):
            axis.imshow(image)
            axis.axis("off")
        axes[0, column].set_title(f'"{shot.query[:40]}"\n{shot.entry.action} · {shot.human.effector} · {shot.human.posture}',
                                  fontsize=8)
        axes[1, column].set_title("check in the laser scan (magenta = true part)", fontsize=8)
    fig.tight_layout()
    return fig


def render_approach_video(scene: P.Scene, memory: P.Memory, model: BodyModel, shots: Sequence[Shot], view: ScanView,
                          out_path: Path, fps: int = 12, n_walk: int = 18, n_reach: int = 12, n_hold: int = 12) -> None:
    """Each shot: walk up, reach, hold. Left: the video frame with the humanoid; right: the same inside the scan."""
    header, left_hw = 48, (480, 360)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out_path), (left_hw[1] + CHECK_HW[1], left_hw[0] + header), fps=fps,
                                         codec="libx264", pix_fmt_out="yuv420p", macro_block_size=16,
                                         output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    for shot in shots:
        small = subsample_frame(shot.frame, 2)
        start = P.approach_start(shot.entry, small.cam_to_world[:3, 3], scene.up)
        bodies = approach_frames(model, shot.human, shot.entry.stand.position, start, memory.floor, scene.up, n_walk, n_reach)
        bodies += [bodies[-1]] * n_hold
        check_pose = _check_camera(shot, view, scene.up, distance=3.2, height=1.1)
        background, check_depth = _check_background(scene, shot, view, check_pose)
        for k, vertices in enumerate(bodies):
            holding = k >= len(bodies) - n_hold
            left = _body_layer(vertices, model, small.rgb, small.depth_m, small.intrinsics, small.cam_to_world, 0.05)
            _mark(left, scene, shot.entry, small.intrinsics, small.cam_to_world, motion=holding)
            right = _body_layer(vertices, model, background, check_depth, CHECK_K, check_pose, 0.03)
            _mark(right, scene, shot.entry, CHECK_K, check_pose, motion=False)
            canvas = np.zeros((left_hw[0] + header, left_hw[1] + CHECK_HW[1], 3), np.uint8)
            canvas[header:, :left_hw[1]] = left
            canvas[header:, left_hw[1]:] = right
            cv2.putText(canvas, f"Query: {shot.query}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            stage = "acts" if holding else "reaches" if k >= n_walk else "walks up"
            cv2.putText(canvas, f"{shot.entry.action} with the {shot.human.effector} | {stage} | left: video frame, "
                                f"right: check in the laser scan", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (190, 190, 190), 1, cv2.LINE_AA)
            writer.send(np.ascontiguousarray(canvas))
    writer.close()


# ---------------------------------------------------------------- scores

def scoreboard(scene: P.Scene, memories: dict[str, P.Memory], view: ScanView) -> pd.DataFrame:
    rows = []
    for name, memory in memories.items():
        errors = np.array([contact_error_cm(scene, e) for e in memory.entries])
        rows.append({"depth used to build the memory": name, "parts in memory": f"{len(memory.entries)} of {len(scene.elements)}",
                     "contact error (median)": f"{np.median(errors):.1f} cm",
                     f"contact on the part (≤ {100 * ON_PART_M:.0f} cm)": f"{np.mean(errors <= 100 * ON_PART_M):.0%}",
                     "stand point free in the scan": f"{np.mean([stand_free_in_scan(e, view) for e in memory.entries]):.0%}"})
    return pd.DataFrame(rows).set_index("depth used to build the memory")

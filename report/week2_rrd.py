"""Interactive 3D file (Rerun .rrd) of the week-2 memory of one scene, made by scripts/make_week2_rrd.py: the content of
the orbit video in its game style (report/week2_interaction.py: action colours, names from the task text, the same
arrows, memory vectors only), to turn, zoom and switch layers freely in the Rerun viewer.

Entity tree (gravity along +Z, floor at z = 0, the scan centred on the origin):
  world/scan/room                         the cropped laser scan below the cut height, downsampled, RGB
  world/scan/ceiling                      the scan above the cut (ceiling, wall tops); hidden at first
  world/parts/<nn_object_action>/points   the part's scan points in its action colour
                                /label    one point at the contact carrying the tag "object ▶ action"
                                /rest     the arrow at rest, faint (a straight arrow, or an arc and its head)
                                /axis     rotations only: the axis (through a knob; the hinge of a door or window)
                                /motion   the same arrow sliding back and forth on the timeline "loop"
  world/camera/path                       the walkthrough's camera centres
  world/camera/key_<k>                    a few of those cameras as frustums, without images
Display only: part masks and names come from the annotation, vectors from the part-vector run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from affordance3d.mem import pipeline as P
from affordance3d.scenefun3d import read_pincam
from report import week2_interaction as G

# ---------------------------------------------------------------- pure helpers (tested)


def gravity_rotation(up: np.ndarray) -> np.ndarray:
    """Rotation matrix that turns the direction `up` into +Z (the shortest turn; half a turn about X if up is -Z)."""
    z = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    v, c = np.cross(z, [0.0, 0.0, 1.0]), float(z[2])
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def world_frame(xyz: np.ndarray, up: np.ndarray, floor_percentile: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(R, offset) with p' = R p - offset: gravity along +Z, the floor (a low percentile of heights) at z = 0, and the
    scan's horizontal median at x = y = 0."""
    rotation = gravity_rotation(up)
    turned = xyz @ rotation.T
    return rotation, np.array([np.median(turned[:, 0]), np.median(turned[:, 1]), np.percentile(turned[:, 2], floor_percentile)])


def to_world(points: np.ndarray, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T - offset


def downsample(n: int, max_points: int, seed: int) -> np.ndarray:
    """Sorted indices of at most `max_points` of `n` points, drawn without replacement with the seed (all if n fits)."""
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))


def label_text(name: str, action: str) -> str:
    return f"{name} ▶ {action}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def loop_times(period_s: float, steps: int, loops: int) -> list[tuple[float, float]]:
    """(time in seconds, arrow phase) for `steps` samples per back-and-forth, over `loops` periods."""
    return [(k * period_s / steps, 0.5 - 0.5 * np.cos(2 * np.pi * k / steps)) for k in range(steps * loops)]


def arrow_parts(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """A glyph polyline as (line strip or None, arrowhead origin, arrowhead vector): a straight arrow is one Arrows3D,
    an arc is a strip with a short arrow over its last two segments."""
    if len(points) == 2:
        return None, points[0], points[1] - points[0]
    return points, points[-3], points[-1] - points[-3]


def axis_strip(part, swing: bool, half_twist_m: float, half_swing_m: float) -> np.ndarray | None:
    """The rotation axis as a segment through the point of the axis nearest the contact; None for a translation."""
    if part.kind != "rot" or part.axis is None:
        return None
    axis = part.axis / np.linalg.norm(part.axis)
    foot = part.origin + ((part.contact - part.origin) @ axis) * axis
    half = half_swing_m if swing else half_twist_m
    return np.stack([foot - half * axis, foot + half * axis])


# ---------------------------------------------------------------- logging


def _scan_layers(scene: P.Scene, cut_above_floor_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = scene.crop.astype(bool) if scene.crop.dtype == bool or scene.crop.max() <= 1 else np.ones(len(scene.scan_xyz), bool)
    xyz, rgb = scene.scan_xyz[keep], (np.clip(scene.scan_rgb[keep], 0, 1) * 255).astype(np.uint8)
    height = xyz @ (scene.up / np.linalg.norm(scene.up))
    return xyz, rgb, height <= np.percentile(height, 2) + cut_above_floor_m


def _log_glyph(rec: rr.RecordingStream, path: str, points: np.ndarray, colour: Sequence[int], radius: float) -> None:
    strip, origin, vector = arrow_parts(points)
    if strip is not None:
        rec.log(f"{path}/arc", rr.LineStrips3D([strip], colors=[colour], radii=radius))
    rec.log(f"{path}/head" if strip is not None else path,
            rr.Arrows3D(origins=[origin], vectors=[vector], colors=[colour], radii=radius))


def blueprint(name: str) -> rrb.Blueprint:
    """One 3D view of the whole world with the ceiling layer hidden; the entity tree open; the loop playing."""
    return rrb.Blueprint(
        rrb.Spatial3DView(origin="/world", name=name, overrides={"world/scan/ceiling": rrb.EntityBehavior(visible=False)}),
        rrb.BlueprintPanel(state="expanded"), rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded", timeline="loop", play_state="playing", loop_mode="all"), collapse_panels=False)


def export(scene: P.Scene, parts: Sequence, style: G.Style, cfg: dict, out_path: Path, name: str) -> dict:
    """Write the .rrd file; returns what was logged (entity -> number of points or items)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(cfg["application_id"])
    rec.save(out_path, default_blueprint=blueprint(name))
    s, p, a, c = cfg["scan"], cfg["parts"], cfg["animation"], cfg["camera"]
    xyz, rgb, below = _scan_layers(scene, s["cut_above_floor_m"])
    rotation, offset = world_frame(xyz[below], scene.up)
    world = lambda points: to_world(points, rotation, offset).astype(np.float32)  # noqa: E731
    stats = {}
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for layer, mask, limit in (("room", below, s["max_points"]), ("ceiling", ~below, s["max_points_above_cut"])):
        chosen = np.flatnonzero(mask)[downsample(int(mask.sum()), limit, cfg["seed"])]
        rec.log(f"world/scan/{layer}", rr.Points3D(world(xyz[chosen]), colors=rgb[chosen], radii=s["radius_m"]), static=True)
        stats[f"world/scan/{layer}"] = len(chosen)
    actions, up = G.part_actions(parts, style), rotation.T @ np.array([0.0, 0.0, 1.0])
    paths = {}
    for part in parts:
        action, category, swing = actions[part.part]
        colour = style.colours[category]
        title = G.object_name(part.task, style.objects.get(part.label, part.label.replace("_", " ")), style.max_words)
        path = paths[part.part] = f"world/parts/{part.part:02d}_{slug(title)}_{slug(action)}"
        points = scene.part_points(part.part)
        rec.log(f"{path}/points", rr.Points3D(world(points), colors=[colour], radii=p["radius_m"]), static=True)
        rec.log(f"{path}/label", rr.Points3D(world(part.contact[None]), colors=[colour], radii=p["label_radius_m"],
                                             labels=[label_text(title, action)], show_labels=True), static=True)
        normal = part.normal if part.normal is not None else up
        _log_glyph(rec, f"{path}/rest", world(G.glyph_at(part, swing, category, 0.0, normal, style)), (*colour, p["rest_alpha"]),
                   p["line_radius_m"])
        axis = axis_strip(part, swing, p["axis_half_m_twist"], p["axis_half_m_swing"])
        if axis is not None:
            rec.log(f"{path}/axis", rr.LineStrips3D([world(axis)], colors=[(*colour, p["rest_alpha"])], radii=p["line_radius_m"] / 2),
                    static=True)
        stats[f"{path}/points"] = len(points)
    for time_s, phase in loop_times(style.period_s, a["steps_per_loop"], a["loops"]):
        rec.set_time("loop", duration=time_s)
        for part in parts:
            action, category, swing = actions[part.part]
            normal = part.normal if part.normal is not None else up
            _log_glyph(rec, f"{paths[part.part]}/motion", world(G.glyph_at(part, swing, category, phase, normal, style)),
                       style.colours[category], p["line_radius_m"] * 1.5)
    stats.update(_log_cameras(rec, scene, world, rotation, offset, c))
    rec.flush()
    rec.disconnect()
    stats["bytes"] = out_path.stat().st_size
    return stats


def _log_cameras(rec: rr.RecordingStream, scene: P.Scene, world, rotation: np.ndarray, offset: np.ndarray, c: dict) -> dict:
    index = scene.gt_index
    times = index.depth_times
    poses = [index.poses[index.matched[t][0]] for t in times]
    rec.log("world/camera/path", rr.LineStrips3D([world(np.array([pose[:3, 3] for pose in poses]))], colors=[(90, 90, 90)],
                                                  radii=c["path_radius_m"]), static=True)
    picks = np.linspace(0, len(times) - 1, c["keyframes"]).round().astype(int)
    for k, i in enumerate(picks):
        intrinsics, (width, height) = read_pincam(index.pincam[times[i]])
        pose = poses[i]
        rec.log(f"world/camera/key_{k}", rr.Transform3D(translation=to_world(pose[:3, 3][None], rotation, offset)[0],
                                                        mat3x3=rotation @ pose[:3, :3]), static=True)
        rec.log(f"world/camera/key_{k}", rr.Pinhole(image_from_camera=intrinsics, resolution=[width, height],
                                                    camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=c["frustum_m"]),
                static=True)
    return {"world/camera/path": len(times), "world/camera/key_*": len(picks)}

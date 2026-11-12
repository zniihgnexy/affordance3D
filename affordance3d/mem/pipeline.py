"""The S1 pipeline as reusable steps: scene, input frames, labels, memory entries, retrieval, display frame.

Same rules as the S1 notebooks: memory entries are computed from inputs only (RGB, LiDAR or
monocular depth, ARKit poses); the laser scan, its rendered depth and the COLMAP poses make the
training labels and the evaluation. Coordinates are the Faro frame throughout, with input cameras
mapped there by the ground-truth Faro->ARKit registration, so that predictions can be scored
directly; the input-side gravity is `Scene.up`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import backproject, project, transform_points
from affordance3d.mem.actionable import (
    BODY_RADIUS_M,
    FOOT_BODY_RADIUS_M,
    FOOT_STAND_DISTANCE_M,
    STAND_DISTANCE_M,
    Stand,
    floor_height,
    plan_stand,
    surface_normal,
)
from affordance3d.mem.inputs import VideoInputs, index_video
from affordance3d.mem.labels import part_label_map
from affordance3d.scenefun3d import (
    FunctionalElement,
    SequenceIndex,
    index_sequence,
    load_frames,
    read_annotations,
    read_laser_scan,
    read_pincam,
)

PIXEL_STEP = 2
LABEL_DEPTH_TOL_M = 0.05
MIN_PART_PIXELS = 10
NORMAL_WINDOW_PX = 24
MIN_NORMAL_POINTS = 3  # fewer points do not define a plane
OBSERVED_STRIDE_PX = 8
VOXEL_M = 0.05
VISIBLE_DEPTH_M = 0.15


@dataclass(frozen=True)
class Scene:
    vid: VideoInputs  # test-time inputs
    gt_index: SequenceIndex  # COLMAP poses + rendered depth (ground truth)
    scan_xyz: np.ndarray
    scan_rgb: np.ndarray
    crop: np.ndarray
    elements: tuple[FunctionalElement, ...]
    motions: dict
    tasks: tuple[tuple[str, tuple[str, ...]], ...]  # (description, annot_ids)

    @property
    def arkit_to_faro(self) -> np.ndarray:
        return np.linalg.inv(self.vid.faro_to_arkit)

    @property
    def up(self) -> np.ndarray:
        """Gravity-up from the ARKit world (+z), expressed in the Faro frame."""
        return self.arkit_to_faro[:3, :3] @ np.array([0.0, 0.0, 1.0])

    def part_points(self, part: int) -> np.ndarray:
        return self.scan_xyz[self.elements[part].indices]

    def task_of(self, annot_id: str) -> str:
        return next((text for text, ids in self.tasks if annot_id in ids), "")


def load_scene(data_root: Path, visit: str, video: str) -> Scene:
    folder = data_root / visit
    scan_xyz, scan_rgb = read_laser_scan(folder / f"{visit}_laser_scan.ply")
    motions = {m["annot_id"]: m for m in json.loads((folder / f"{visit}_motions.json").read_text())["motions"]}
    tasks = tuple((d["description"], tuple(d["annot_id"]))
                  for d in json.loads((folder / f"{visit}_descriptions.json").read_text())["descriptions"])
    return Scene(index_video(data_root, visit, video), index_sequence(data_root, visit, video), scan_xyz, scan_rgb,
                 np.load(folder / f"{visit}_crop_mask.npy"), tuple(read_annotations(folder / f"{visit}_annotations.json")),
                 motions, tasks)


def input_camera(scene: Scene, t: str, pixel_step: int = PIXEL_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsics at the working resolution and the ARKit camera pose, expressed in the Faro frame."""
    intrinsics = np.diag([1.0 / pixel_step, 1.0 / pixel_step, 1.0]) @ scene.vid.intrinsics(t)
    return intrinsics, scene.arkit_to_faro @ scene.vid.arkit_pose(t)


def frame_hw(scene: Scene, t: str, pixel_step: int = PIXEL_STEP) -> tuple[int, int]:
    """Working image size (rows, columns) of frame `t`; videos are stored upright as 1440x1920 or 1920x1440."""
    name = f"{scene.vid.folder.name}_{t}.pincam"
    width, height = read_pincam(scene.vid.folder / "hires_wide_intrinsics" / name)[1]
    return height // pixel_step, width // pixel_step


def input_frame(scene: Scene, t: str, depth: np.ndarray, pixel_step: int = PIXEL_STEP) -> Frame:
    rgb = iio.imread(scene.vid.rgb_path(t))[::pixel_step, ::pixel_step]
    intrinsics, pose = input_camera(scene, t, pixel_step)
    return Frame(0, rgb, depth.astype(np.float32), intrinsics, pose)


def gt_frames(scene: Scene, times: Sequence[str], pixel_step: int = PIXEL_STEP) -> list[Frame]:
    """Ground-truth cameras (COLMAP pose, rendered depth): for labels and evaluation only."""
    return load_frames(scene.gt_index, list(times), cw_turns=0, pixel_step=pixel_step)


def lidar_depth(scene: Scene, t: str, pixel_step: int = PIXEL_STEP) -> np.ndarray:
    return scene.vid.lidar_depth(t, frame_hw(scene, t, pixel_step))


def lidar_scale(depth: np.ndarray, lidar: np.ndarray) -> float | None:
    """Median LiDAR / prediction ratio of one frame; None when no pixel has both (the device can mark a whole frame unreliable)."""
    both = (lidar > 0) & (depth > 0)
    return float(np.median(lidar[both] / depth[both])) if both.any() else None


def lidar_scaled(depth: np.ndarray, lidar: np.ndarray) -> np.ndarray:
    """Monocular depth rescaled by the median LiDAR / prediction ratio of the same frame (an input)."""
    scale = lidar_scale(depth, lidar)
    if scale is None:
        raise ValueError("no pixel has both LiDAR and predicted depth")
    return depth * scale


def lidar_scaled_together(depths: Sequence[np.ndarray], lidars: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The frames of one video, each rescaled by its own LiDAR; a frame without a usable LiDAR pixel takes the
    median scale of the other frames (the scale of a monocular model drifts little within one video)."""
    scales = [lidar_scale(depth, lidar) for depth, lidar in zip(depths, lidars, strict=True)]
    known = [scale for scale in scales if scale is not None]
    if not known:
        raise ValueError("no frame has a pixel with both LiDAR and predicted depth")
    shared = float(np.median(known))
    return [depth * (shared if scale is None else scale) for depth, scale in zip(depths, scales)]


def label_maps(scene: Scene, frames: Sequence[Frame]) -> list[np.ndarray]:
    points = [scene.part_points(p) for p in range(len(scene.elements))]
    return [part_label_map(points, frame, LABEL_DEPTH_TOL_M) for frame in frames]


@dataclass(frozen=True)
class MemoryEntry:
    """One functional part in memory: where it was seen and how to act on it (all from inputs)."""

    part: int
    annot_id: str
    action: str
    task: str
    keyframe: str  # keyframe where the part was seen with most pixels
    pixel: tuple[int, int]  # contact pixel in that keyframe
    pixels: int
    seen_in: tuple[str, ...]  # every keyframe where the part was seen
    contact: np.ndarray
    normal: np.ndarray  # supporting surface around the part
    stand: Stand
    motion: dict | None  # oracle motion annotation (type, direction, origin) until step 7 predicts it


@dataclass(frozen=True)
class Memory:
    keyframes: tuple[str, ...]
    entries: tuple[MemoryEntry, ...]
    floor: float  # along Scene.up
    occupied: np.ndarray  # observed points, voxel-downsampled


def voxel_points(points: np.ndarray, size: float = VOXEL_M) -> np.ndarray:
    """One point per occupied voxel."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(size).points)


def contact_and_normal(frame: Frame, labels: np.ndarray, part: int) -> tuple[tuple[int, int], np.ndarray, np.ndarray] | None:
    """Contact pixel, contact point and surrounding surface normal of the pixels labelled `part`, from one frame.

    The normal is that of the surface around the part (a handle sticks out of it). A part that fills the whole
    window around its contact pixel has no surroundings there: then the window's own surface is used.
    """
    rows_cols = np.argwhere(labels == part)
    centre = np.median(rows_cols, axis=0)
    v, u = rows_cols[np.argmin(np.linalg.norm(rows_cols - centre, axis=1))]
    depth = frame.depth_m[rows_cols[:, 0], rows_cols[:, 1]]
    depth = depth[depth > 0]
    if not len(depth):
        return None
    ray = np.linalg.inv(frame.intrinsics) @ np.array([u, v, 1.0])
    contact = transform_points(frame.cam_to_world, (ray * np.median(depth))[None])[0]
    top, left = max(v - NORMAL_WINDOW_PX, 0), max(u - NORMAL_WINDOW_PX, 0)
    window = frame.depth_m[top:v + NORMAL_WINDOW_PX, left:u + NORMAL_WINDOW_PX]
    off_part = labels[top:v + NORMAL_WINDOW_PX, left:u + NORMAL_WINDOW_PX] != part
    around = (window > 0) & off_part
    ys, xs = np.nonzero(around if around.sum() >= MIN_NORMAL_POINTS else window > 0)
    if len(xs) < MIN_NORMAL_POINTS:
        return None
    pixels = np.stack([xs + left, ys + top], axis=1)
    rays = np.linalg.inv(frame.intrinsics) @ np.vstack([pixels.T, np.ones(len(pixels))])
    neighbourhood = transform_points(frame.cam_to_world, (rays * window[ys, xs]).T)
    return (int(u), int(v)), contact, surface_normal(neighbourhood, frame.cam_to_world[:3, 3])


def build_memory(scene: Scene, keyframes: Sequence[str], frames: Sequence[Frame],
                 labels: Sequence[np.ndarray]) -> Memory:
    """Write one entry per functional part seen in at least MIN_PART_PIXELS pixels of some keyframe."""
    observed = np.concatenate([backproject(f, stride=OBSERVED_STRIDE_PX)[0] for f in frames])
    floor, occupied = floor_height(observed @ scene.up), voxel_points(observed)
    counts = np.array([[(lab == p).sum() for lab in labels] for p in range(len(scene.elements))])
    entries = []
    for part, element in enumerate(scene.elements):
        if counts[part].max() < MIN_PART_PIXELS:
            continue
        best = int(np.argmax(counts[part]))
        found = contact_and_normal(frames[best], labels[best], part)
        if found is None:
            continue
        pixel, contact, normal = found
        foot = element.label == "foot_push"
        stand = plan_stand(contact, normal, scene.up, floor, frames[best].cam_to_world[:3, 3], occupied,
                           distance=FOOT_STAND_DISTANCE_M if foot else STAND_DISTANCE_M, max_points=2,
                           radius=FOOT_BODY_RADIUS_M if foot else BODY_RADIUS_M)
        entries.append(MemoryEntry(part, element.annot_id, element.label, scene.task_of(element.annot_id),
                                   keyframes[best], pixel, int(counts[part, best]),
                                   tuple(t for t, c in zip(keyframes, counts[part]) if c > 0), contact, normal,
                                   stand, scene.motions.get(element.annot_id)))
    return Memory(tuple(keyframes), tuple(entries), floor, occupied)


def retrieve(entries: Sequence[MemoryEntry], tasks: Sequence[tuple[str, Sequence[str]]],
             query: str) -> list[MemoryEntry]:
    """ORACLE retrieval: a query that is one of the scene's task descriptions returns the parts it names.

    Placeholder for step 6 (feature search); it only uses the annotated description -> part links.
    """
    wanted = {annot for text, ids in tasks if text.strip().lower() == query.strip().lower() for annot in ids}
    return [e for e in entries if e.annot_id in wanted]


def display_score(scene: Scene, entry: MemoryEntry, t: str, pixel_step: int = PIXEL_STEP) -> tuple[float, int]:
    """How well frame `t` shows the part and the person standing at it (inputs only); -1 if the part is hidden."""
    intrinsics, pose = input_camera(scene, t, pixel_step)
    frame = Frame(0, np.zeros((1, 1, 3), np.uint8), np.zeros(frame_hw(scene, t, pixel_step), np.float32), intrinsics, pose)
    body = entry.stand.position + np.outer([0.1, 0.9, 1.6], scene.up)
    uv, z, inside = project(np.vstack([entry.contact, body]), frame)
    if not inside[0]:
        return -1.0, 0
    u, v = np.rint(uv[0]).astype(int)
    observed = lidar_depth(scene, t, pixel_step)[v, u]
    if observed <= 0 or abs(observed - z[0]) > VISIBLE_DEPTH_M:
        return -1.0, 0
    distance = np.linalg.norm(entry.contact - pose[:3, 3])
    in_view = int(inside[1:].sum())
    return in_view + (1.0 if 1.5 <= distance <= 4.0 else 0.0) - 0.1 * abs(distance - 2.5), in_view


def approach_start(entry: MemoryEntry, camera_centre: np.ndarray, up: np.ndarray, max_walk_m: float = 1.5) -> np.ndarray:
    """Where the person starts walking: from the stand point towards the camera, at most `max_walk_m` away."""
    towards = camera_centre - entry.stand.position
    towards = towards - (towards @ up) * up
    length = np.linalg.norm(towards)
    if length < 1e-6:
        return entry.stand.position - max_walk_m * entry.stand.facing
    return entry.stand.position + towards / length * min(max_walk_m, 0.6 * length)

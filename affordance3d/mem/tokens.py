"""L2 memory: part tokens with a thin explicit index, written from per-keyframe affordance maps.

A region is a connected patch of one action type in one keyframe's affordance map. Each region becomes a
part token: two feature vectors (a self-supervised one pooled over the region and an image-text one of the
crop around it) plus a small index (time, keyframe, pixel and depth in that keyframe). Tokens are linked
across keyframes by feature similarity and rough position; the part ids of the annotations are never used.

Geometry is not stored. Contact point, surface normal and stand point are read from the evidence keyframe
when a query needs them (`read_geometry`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from affordance3d.features import l2_normalize
from affordance3d.frames import Frame
from affordance3d.geometry import backproject, transform_points
from affordance3d.mem.actionable import (
    BODY_RADIUS_M,
    FOOT_BODY_RADIUS_M,
    FOOT_STAND_DISTANCE_M,
    STAND_DISTANCE_M,
    Stand,
    floor_height,
    plan_stand,
)
from affordance3d.mem.pipeline import MIN_PART_PIXELS, OBSERVED_STRIDE_PX, contact_and_normal, voxel_points

ACTIONS = ("rotate", "key_press", "tip_push", "hook_pull", "pinch_pull", "hook_turn", "foot_push", "plug_in", "unplug")
NO_ACTION = -1
NO_REGION = 0
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
FILL_PX = 7  # projected scan points are sparse; gaps up to this many pixels inside a part are filled
LINK_MIN_SIMILARITY = 0.5
LINK_MAX_DISTANCE_M = 0.3
CONTEXT_SCALE = 3.0
CONTEXT_MIN_SIDE_PX = 96
READ_NEIGHBOURS = 3  # keyframes on each side of the evidence keyframe used for the free-space check
FEATURE_DTYPE = np.float16  # how features are counted when the memory size is reported


@dataclass(frozen=True)
class Region:
    """A patch of one action type in one keyframe, located only relative to that keyframe."""

    keyframe: int  # index of the keyframe in the memory
    component: int  # id of this patch in the keyframe's component map
    action: int  # index into ACTIONS
    box: tuple[int, int, int, int]  # left, top, right, bottom (exclusive), in pixels
    pixel: tuple[int, int]  # (u, v) of the region pixel closest to its centre
    pixels: int
    depth_m: float  # median depth of the region in its keyframe


@dataclass(frozen=True)
class PartToken:
    region: Region
    time_s: float
    visual: np.ndarray  # self-supervised feature pooled over the region, unit length
    semantic: np.ndarray  # image-text feature of the crop around the region, unit length


@dataclass(frozen=True)
class L2Memory:
    """Observation layer (times, component maps; images, depth and poses stay with the keyframes),
    part layer (tokens and the part each belongs to) and one scene-level number (floor height)."""

    times: tuple[str, ...]
    components: tuple[np.ndarray, ...]  # per keyframe: region id per pixel, NO_REGION elsewhere
    tokens: tuple[PartToken, ...]
    part_of: np.ndarray  # part id of every token
    floor: float  # along the gravity direction

    @property
    def n_parts(self) -> int:
        return int(self.part_of.max()) + 1 if len(self.part_of) else 0

    def tokens_of(self, part: int) -> np.ndarray:
        return np.flatnonzero(self.part_of == part)

    def evidence(self, part: int) -> int:
        """The token that shows a part with most pixels."""
        members = self.tokens_of(part)
        return int(members[np.argmax([self.tokens[i].region.pixels for i in members])])

    def feature_bytes(self) -> int:
        return sum(t.visual.size + t.semantic.size for t in self.tokens) * np.dtype(FEATURE_DTYPE).itemsize


@dataclass(frozen=True)
class Readout:
    """Geometry of one token, computed at query time from its evidence keyframe."""

    token: int
    pixel: tuple[int, int]
    contact: np.ndarray
    normal: np.ndarray
    stand: Stand


def action_map(part_labels: np.ndarray, part_actions: Sequence[int], fill_px: int = FILL_PX) -> np.ndarray:
    """The affordance map of one keyframe: an action type per pixel, with the part identities dropped.

    The labels are projected scan points, so a part can be a cloud of separate pixels; each action's pixels
    are closed with a `fill_px` square to give the dense patch a predicted map would have.
    """
    lookup = np.append(np.asarray(part_actions, dtype=np.int8), NO_ACTION)  # label -1 (background) -> NO_ACTION
    sparse = lookup[part_labels]
    if fill_px <= 1:
        return sparse
    dense, kernel = sparse.copy(), np.ones((fill_px, fill_px), dtype=np.uint8)
    for action in np.unique(sparse[sparse != NO_ACTION]):
        closed = cv2.morphologyEx((sparse == action).astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        dense[closed & (sparse == NO_ACTION)] = action
    return dense


def _region(mask: np.ndarray, depth_m: np.ndarray, keyframe: int, component: int, action: int) -> Region | None:
    rows_cols = np.argwhere(mask)
    depth = depth_m[mask]
    depth = depth[depth > 0]
    if not len(depth):
        return None
    centre = np.median(rows_cols, axis=0)
    v, u = rows_cols[np.argmin(np.linalg.norm(rows_cols - centre, axis=1))]
    (top, left), (bottom, right) = rows_cols.min(axis=0), rows_cols.max(axis=0) + 1
    return Region(keyframe, component, action, (int(left), int(top), int(right), int(bottom)), (int(u), int(v)),
                  len(rows_cols), float(np.median(depth)))


def find_regions(actions: np.ndarray, depth_m: np.ndarray, keyframe: int,
                 min_pixels: int = MIN_PART_PIXELS) -> tuple[np.ndarray, tuple[Region, ...]]:
    """Connected patches of one action type: the component map and one Region per patch with valid depth."""
    components = np.full(actions.shape, NO_REGION, dtype=np.int32)
    regions: list[Region] = []
    for action in np.unique(actions[actions != NO_ACTION]):
        labelled, count = ndimage.label(actions == action, structure=EIGHT_CONNECTED)
        sizes = np.bincount(labelled.ravel(), minlength=count + 1)
        for local in np.flatnonzero(sizes[1:] >= min_pixels) + 1:
            mask = labelled == local
            region = _region(mask, depth_m, keyframe, len(regions) + 1, int(action))
            if region is not None:
                components[mask] = region.component
                regions.append(region)
    return components, tuple(regions)


def region_position(region: Region, frame: Frame) -> np.ndarray:
    """World position of a region from its own keyframe alone: its pixel, its depth and that keyframe's pose."""
    u, v = region.pixel
    ray = np.linalg.inv(frame.intrinsics) @ np.array([u, v, 1.0])
    return transform_points(frame.cam_to_world, (ray * region.depth_m)[None])[0]


def pool_patches(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unit-length mean of the patch features under a pixel mask; a patch counts by how much of it is covered."""
    grid_h, grid_w = grid.shape[:2]
    weights = cv2.resize(mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    if weights.sum() <= 0:
        raise ValueError("the mask is empty")
    pooled = (grid * weights[..., None]).sum(axis=(0, 1)) / weights.sum()
    return l2_normalize(pooled[None])[0]


def context_box(box: tuple[int, int, int, int], image_hw: tuple[int, int], scale: float = CONTEXT_SCALE,
                min_side: int = CONTEXT_MIN_SIDE_PX) -> tuple[int, int, int, int]:
    """Square crop around a region, wide enough to show what the part belongs to, kept inside the image."""
    left, top, right, bottom = box
    side = min(round(max(scale * max(right - left, bottom - top), min_side)), *image_hw)
    x0 = int(np.clip((left + right) // 2 - side // 2, 0, image_hw[1] - side))
    y0 = int(np.clip((top + bottom) // 2 - side // 2, 0, image_hw[0] - side))
    return x0, y0, x0 + side, y0 + side


def link_tokens(visual: np.ndarray, positions: np.ndarray, actions: Sequence[int], keyframes: Sequence[int],
                min_similarity: float = LINK_MIN_SIMILARITY, max_distance_m: float = LINK_MAX_DISTANCE_M) -> np.ndarray:
    """Part id for every token, taken in the given (time) order.

    A token joins the nearest earlier part of the same action type in which some token both looks alike
    (similarity at least `min_similarity`) and lies within `max_distance_m`; otherwise it starts a new part.
    Position decides between candidates because neighbouring parts of one kind look the same. A part keeps
    all its tokens, nothing is averaged, and two regions of the same keyframe never share a part.
    """
    part_of = np.full(len(visual), -1)
    for i in range(len(visual)):
        best, best_distance = -1, max_distance_m
        for part in range(part_of.max() + 1):
            members = np.flatnonzero(part_of[:i] == part)
            if actions[members[0]] != actions[i] or any(keyframes[m] == keyframes[i] for m in members):
                continue
            alike = members[(visual[members] @ visual[i]) >= min_similarity]
            if not len(alike):
                continue
            distance = float(np.linalg.norm(positions[alike] - positions[i], axis=1).min())
            if distance <= best_distance:
                best, best_distance = part, distance
        part_of[i] = best if best >= 0 else part_of.max() + 1
    return part_of


def write_memory(times: Sequence[str], frames: Sequence[Frame], action_maps: Sequence[np.ndarray],
                 visual_grids: Sequence[np.ndarray], embed_images: Callable[[list[np.ndarray]], np.ndarray],
                 up: np.ndarray, start_s: float | None = None) -> L2Memory:
    """Write the L2 memory of one video: regions of every keyframe, their two features, and the links between them.

    `start_s` is the time that counts as zero for the tokens (default: the first keyframe).
    """
    if not len(times):
        raise ValueError("at least one keyframe is needed")
    found = [find_regions(actions, frame.depth_m, k) for k, (actions, frame) in enumerate(zip(action_maps, frames))]
    regions = [region for _, per_frame in found for region in per_frame]
    crops = []
    for region in regions:
        x0, y0, x1, y1 = context_box(region.box, frames[region.keyframe].image_hw)
        crops.append(frames[region.keyframe].rgb[y0:y1, x0:x1])
    semantic = embed_images(crops) if crops else []
    start = float(times[0]) if start_s is None else start_s
    tokens = tuple(PartToken(region, float(times[region.keyframe]) - start,
                             pool_patches(visual_grids[region.keyframe], found[region.keyframe][0] == region.component),
                             vector) for region, vector in zip(regions, semantic))
    positions = np.array([region_position(r, frames[r.keyframe]) for r in regions]).reshape(-1, 3)
    visual = np.array([t.visual for t in tokens]).reshape(len(tokens), visual_grids[0].shape[-1])
    part_of = link_tokens(visual, positions, [r.action for r in regions], [r.keyframe for r in regions])
    observed = np.concatenate([backproject(f, stride=OBSERVED_STRIDE_PX)[0] for f in frames])
    return L2Memory(tuple(times), tuple(components for components, _ in found), tokens, part_of,
                    floor_height(observed @ up))


def read_geometry(memory: L2Memory, token: int, frames: Sequence[Frame], up: np.ndarray,
                  neighbours: int = READ_NEIGHBOURS) -> Readout | None:
    """Contact point, surface normal and stand point of one token, computed now and not stored.

    Contact and normal come from the evidence keyframe alone. Whether the stand point is free is checked
    against the points seen in that keyframe and its `neighbours` keyframes before and after.
    """
    if not 0 <= token < len(memory.tokens):
        raise IndexError(f"token {token} is not in a memory of {len(memory.tokens)} tokens")
    region = memory.tokens[token].region
    frame = frames[region.keyframe]
    found = contact_and_normal(frame, memory.components[region.keyframe], region.component)
    if found is None:
        return None
    pixel, contact, normal = found
    nearby = frames[max(region.keyframe - neighbours, 0):region.keyframe + neighbours + 1]
    occupied = voxel_points(np.concatenate([backproject(f, stride=OBSERVED_STRIDE_PX)[0] for f in nearby]))
    foot = ACTIONS[region.action] == "foot_push"
    stand = plan_stand(contact, normal, up, memory.floor, frame.cam_to_world[:3, 3], occupied,
                       distance=FOOT_STAND_DISTANCE_M if foot else STAND_DISTANCE_M, max_points=2,
                       radius=FOOT_BODY_RADIUS_M if foot else BODY_RADIUS_M)
    return Readout(token, pixel, contact, normal, stand)

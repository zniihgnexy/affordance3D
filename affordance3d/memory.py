"""Sparse voxel memory of fused visual features with per-frame provenance (E1).

This is the plain "visual history memory" baseline (ConceptFusion-style averaging):
it stores where surfaces are, what their averaged frozen features look like, and which
frames observed them. It contains no affordance supervision. Averaging is only valid
for a static scene; changed object states must not be merged into one voxel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.features import PatchFeatures, l2_normalize
from affordance3d.frames import Frame
from affordance3d.geometry import backproject, project, visible_mask

KEY_BITS = 21  # bits per axis when packing a voxel index into one int64
KEY_OFFSET = 1 << (KEY_BITS - 1)


@dataclass(frozen=True)
class Observation:
    """Surface samples written by one frame: position, colour and patch feature."""

    frame_index: int
    xyz: np.ndarray  # (N, 3) world points
    rgb: np.ndarray  # (N, 3) colours in [0, 1]
    features: np.ndarray  # (N, C) feature of the patch each sample's pixel falls in


@dataclass(frozen=True)
class VoxelMemory:
    voxel_size: float
    codes: np.ndarray  # (M,) packed voxel indices, sorted
    xyz: np.ndarray  # (M, 3) mean position of the samples in each voxel
    rgb: np.ndarray  # (M, 3) mean colour
    features: np.ndarray  # (M, C) unit-normalised mean feature
    seen_by: np.ndarray  # (M, F) True where frame f observed voxel m
    frame_ids: np.ndarray  # (F,) frame index of each seen_by column

    @property
    def n_voxels(self) -> int:
        return len(self.codes)

    @property
    def first_seen(self) -> np.ndarray:
        return self.frame_ids[self.seen_by.argmax(axis=1)]

    @property
    def last_seen(self) -> np.ndarray:
        last_column = self.seen_by.shape[1] - 1 - self.seen_by[:, ::-1].argmax(axis=1)
        return self.frame_ids[last_column]

    @property
    def nbytes(self) -> int:
        arrays = (self.codes, self.xyz, self.rgb, self.features, self.seen_by)
        return sum(array.nbytes for array in arrays)


@dataclass(frozen=True)
class Coverage:
    """The plan's coverage classes for one current view.

    `history_only` (remembered, not visible now) has two causes, kept apart because only
    the first is "outside the current view": `out_of_view`, and `occluded`, which also
    catches pixels with no valid current depth, depth noise, and silhouette voxels whose
    averaged position falls between surfaces.
    """

    visible_now: np.ndarray  # (M,) remembered and visible in the current frame
    out_of_view: np.ndarray  # (M,) remembered, outside the current image or behind the camera
    occluded: np.ndarray  # (M,) remembered, inside the image but depth disagrees or is missing
    new_in_current: np.ndarray  # (N,) current samples in voxels the memory never observed

    @property
    def history_only(self) -> np.ndarray:
        return self.out_of_view | self.occluded


def observe(frame: Frame, features: PatchFeatures, stride: int) -> Observation:
    xyz, uv = backproject(frame, stride)
    rgb = frame.rgb[uv[:, 1], uv[:, 0]].astype(np.float32) / 255.0
    return Observation(frame.index, xyz, rgb, features.at_pixels(uv))


def voxel_codes(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = np.floor(xyz / voxel_size).astype(np.int64) + KEY_OFFSET
    if (keys < 0).any() or (keys >= 1 << KEY_BITS).any():
        raise ValueError("points exceed the voxel index range; use a larger voxel size")
    return (keys[:, 0] << (2 * KEY_BITS)) | (keys[:, 1] << KEY_BITS) | keys[:, 2]


def _group_mean(values: np.ndarray, groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.zeros((len(counts), values.shape[1]), dtype=np.float64)
    np.add.at(sums, groups, values)
    return sums / counts[:, None]


def fuse(observations: Sequence[Observation], voxel_size: float) -> VoxelMemory:
    """Build memory from observations; pass only a time prefix to avoid reading the future."""
    if not observations:
        raise ValueError("fuse needs at least one observation")
    xyz = np.concatenate([obs.xyz for obs in observations])
    column = np.repeat(np.arange(len(observations)), [len(obs.xyz) for obs in observations])
    codes, groups, counts = np.unique(voxel_codes(xyz, voxel_size), return_inverse=True, return_counts=True)

    seen_by = np.zeros((len(codes), len(observations)), dtype=bool)
    seen_by[groups, column] = True
    return VoxelMemory(
        voxel_size=voxel_size,
        codes=codes,
        xyz=_group_mean(xyz, groups, counts),
        rgb=_group_mean(np.concatenate([obs.rgb for obs in observations]), groups, counts).astype(np.float32),
        features=l2_normalize(_group_mean(np.concatenate([obs.features for obs in observations]), groups, counts)).astype(np.float32),
        seen_by=seen_by,
        frame_ids=np.array([obs.frame_index for obs in observations]),
    )


def contains(memory: VoxelMemory, xyz: np.ndarray) -> np.ndarray:
    """True where a point falls in a voxel the memory has observed."""
    return np.isin(voxel_codes(xyz, memory.voxel_size), memory.codes)


def cosine_scores(memory: VoxelMemory, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of every remembered voxel feature to one query feature."""
    return memory.features @ l2_normalize(query.astype(np.float32))


def coverage(memory: VoxelMemory, current: Frame, current_obs: Observation, depth_tol: float) -> Coverage:
    _, _, inside = project(memory.xyz, current)
    visible_now = visible_mask(memory.xyz, current, depth_tol)
    return Coverage(visible_now, ~inside, inside & ~visible_now, ~contains(memory, current_obs.xyz))

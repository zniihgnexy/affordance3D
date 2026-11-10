"""Minimal SMPL-X body model: shape and pose blend shapes plus linear blend skinning, in torch.

Follows the official `smplx` package (lbs.py) for the parts we use: body shape (betas), the
55-joint pose in axis-angle (global orient, 21 body joints, jaw, two eyes, 2 x 15 hand joints),
relaxed hands (`hands_mean`, as smplx's default flat_hand_mean=False) and a root translation.
Model files come from the SMPL-X website under its non-commercial licence; they stay in
data/body_models/ and must not be committed or embedded in published outputs.

Canonical body frame: the person faces +z, up is +y, +x is the person's left.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

N_JOINTS = 55
N_BODY_JOINTS = 21
JOINT = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, "left_knee": 4, "right_knee": 5, "spine2": 6,
         "left_ankle": 7, "right_ankle": 8, "spine3": 9, "left_foot": 10, "right_foot": 11, "neck": 12,
         "left_collar": 13, "right_collar": 14, "head": 15, "left_shoulder": 16, "right_shoulder": 17,
         "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
         "left_middle1": 28, "right_middle1": 43}


@dataclass(frozen=True)
class BodyModel:
    v_template: torch.Tensor  # (V, 3)
    shapedirs: torch.Tensor  # (V, 3, n_betas)
    posedirs: torch.Tensor  # (V, 3, 9 * (N_JOINTS - 1))
    j_regressor: torch.Tensor  # (N_JOINTS, V)
    parents: tuple[int, ...]  # parent joint of each joint, -1 for the root
    weights: torch.Tensor  # (V, N_JOINTS) skinning weights
    faces: np.ndarray  # (F, 3)
    hands_mean: torch.Tensor  # (90,) relaxed left + right hand pose


def load_smplx(path: Path, n_betas: int = 10) -> BodyModel:
    data = np.load(path, allow_pickle=True)
    parents = data["kintree_table"][0].astype(np.int64)
    parents[0] = -1

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(data[name], dtype=np.float32))

    return BodyModel(tensor("v_template"), tensor("shapedirs")[:, :, :n_betas], tensor("posedirs"),
                     tensor("J_regressor"), tuple(int(p) for p in parents), tensor("weights"),
                     np.asarray(data["f"], dtype=np.int64),
                     torch.cat([tensor("hands_meanl"), tensor("hands_meanr")]))


def rodrigues(rotvecs: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices (differentiable, safe at zero)."""
    angle = torch.linalg.norm(rotvecs + 1e-8, dim=1, keepdim=True)
    axis = rotvecs / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    k = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=1).view(-1, 3, 3)
    sin, cos = torch.sin(angle)[:, :, None], torch.cos(angle)[:, :, None]
    return torch.eye(3, dtype=rotvecs.dtype)[None] + sin * k + (1 - cos) * (k @ k)


def full_pose(model: BodyModel, global_orient: torch.Tensor, body_pose: torch.Tensor,
              hand_pose: torch.Tensor | None = None) -> torch.Tensor:
    """(55, 3) axis-angle pose: root, 21 body joints, jaw and eyes at rest, relaxed hands unless given."""
    hands = model.hands_mean if hand_pose is None else hand_pose + model.hands_mean
    face = torch.zeros(9, dtype=body_pose.dtype)
    return torch.cat([global_orient.reshape(3), body_pose.reshape(N_BODY_JOINTS * 3), face, hands]).view(N_JOINTS, 3)


def forward(model: BodyModel, pose: torch.Tensor, betas: torch.Tensor | None = None,
            transl: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Posed vertices (V, 3) and joints (55, 3) for a (55, 3) axis-angle pose."""
    shape = model.v_template
    if betas is not None:
        shape = shape + torch.einsum("vcb,b->vc", model.shapedirs, betas)
    joints = model.j_regressor @ shape
    rotations = rodrigues(pose)
    pose_feature = (rotations[1:] - torch.eye(3)).reshape(-1)
    posed = shape + torch.einsum("vcp,p->vc", model.posedirs, pose_feature)

    parents = model.parents
    relative = torch.cat([joints[:1], joints[1:] - joints[list(parents[1:])]])
    local = torch.cat([torch.cat([rotations, relative[:, :, None]], dim=2),
                       torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(N_JOINTS, 1, 4)], dim=1)
    chain = [local[0]]
    for joint in range(1, N_JOINTS):
        chain.append(chain[parents[joint]] @ local[joint])
    world = torch.stack(chain)
    posed_joints = world[:, :3, 3]
    skin = world.clone()
    skin[:, :3, 3] = world[:, :3, 3] - (world[:, :3, :3] @ joints[:, :, None])[:, :, 0]
    per_vertex = (model.weights @ skin.reshape(N_JOINTS, 16)).reshape(-1, 4, 4)
    vertices = (per_vertex[:, :3, :3] @ posed[:, :, None])[:, :, 0] + per_vertex[:, :3, 3]
    offset = torch.zeros(3) if transl is None else transl
    return vertices + offset, posed_joints + offset

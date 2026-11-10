"""Place an SMPL-X humanoid at a stand point, facing a part, and reach the contact point.

Kinematic only (no physics): a base posture (standing, or squatting for low parts), the
body turned to face the part with feet on the floor, then a short gradient-based IK on the
spine and one arm (or one leg for foot_push) so the palm (or foot) reaches the target.
Joint signs below were checked on the SMPL-X neutral model (canonical: faces +z, up +y,
+x is the person's left): hip flexion is -x, knee flexion +x, right arm down +z at the
shoulder, right elbow flexion +y, forward spine bend +x; the left side mirrors y and z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from affordance3d.mem.body import JOINT, N_BODY_JOINTS, BodyModel, forward, full_pose

SQUAT_BELOW_M = 0.55  # contacts lower than this above the floor are reached squatting
TOUCH_OFFSET_M = 0.03  # aim the palm/foot slightly in front of the surface
IK_STEPS, IK_LR, IK_REGULARISER = 300, 0.08, 0.001  # tuned on synthetic targets: <= 1.6 cm left
HINGE_PENALTY = 10.0  # elbows and knees bend about one axis only


def _body(changes: dict[str, tuple[float, float, float]]) -> np.ndarray:
    pose = np.zeros((N_BODY_JOINTS, 3))
    for name, rotvec in changes.items():
        pose[JOINT[name] - 1] = rotvec
    return pose


ARM_DOWN = {"right": _body({"right_shoulder": (0, 0, 1.2), "right_elbow": (0, 0.2, 0)}),
            "left": _body({"left_shoulder": (0, 0, -1.2), "left_elbow": (0, -0.2, 0)})}
ARM_FORWARD = {"right": _body({"right_shoulder": (0, 1.2, 0.5), "right_elbow": (0, 0.5, 0)}),
               "left": _body({"left_shoulder": (0, -1.2, -0.5), "left_elbow": (0, -0.5, 0)})}
SQUAT = _body({"left_hip": (-1.6, 0, 0), "right_hip": (-1.6, 0, 0), "left_knee": (2.1, 0, 0), "right_knee": (2.1, 0, 0),
               "left_ankle": (-0.5, 0, 0), "right_ankle": (-0.5, 0, 0), "spine1": (0.25, 0, 0)})


def base_posture(reaching: str | None, squat: bool) -> np.ndarray:
    """Arms down, the reaching arm (if any) already forward, optionally squatting."""
    arms = [ARM_FORWARD[side] if side == reaching else ARM_DOWN[side] for side in ("right", "left")]
    return arms[0] + arms[1] + (SQUAT if squat else 0.0)
ARM_CHAIN = {"right": ("spine1", "spine2", "spine3", "right_collar", "right_shoulder", "right_elbow", "right_wrist"),
             "left": ("spine1", "spine2", "spine3", "left_collar", "left_shoulder", "left_elbow", "left_wrist")}
LEG_CHAIN = ("right_hip", "right_knee", "right_ankle")
HINGES = {"right_elbow": 1, "left_elbow": 1, "right_knee": 0}  # joint -> the one axis it may bend about


@dataclass(frozen=True)
class PosedHumanoid:
    vertices: np.ndarray  # (V, 3) world
    joints: np.ndarray  # (55, 3) world
    effector: str  # "right hand", "left hand" or "right foot"
    posture: str  # "standing" or "squatting"
    target: np.ndarray
    reach_error_m: float  # distance left between the palm/foot and the target after IK


def canonical_to_world(facing: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation taking the body's canonical axes (left, up, front) = (x, y, z) to the world."""
    front = facing - (facing @ up) * up
    front /= np.linalg.norm(front)
    left = np.cross(up, front)
    return np.stack([left, up, front], axis=1)


def feet_on_floor(vertices: np.ndarray, pelvis: np.ndarray, stand: np.ndarray, floor_z: float,
                  up: np.ndarray) -> np.ndarray:
    """Translation that puts the pelvis above the stand point and the lowest vertex on the floor."""
    horizontal = (stand - pelvis) - ((stand - pelvis) @ up) * up
    return horizontal + (floor_z - np.min(vertices @ up)) * up


def _effector(joints: torch.Tensor, effector: str) -> torch.Tensor:
    if effector == "right foot":
        return joints[JOINT["right_foot"]]
    side = effector.split()[0]
    return 0.5 * (joints[JOINT[f"{side}_wrist"]] + joints[JOINT[f"{side}_middle1"]])


def pose_humanoid(model: BodyModel, contact: np.ndarray, normal: np.ndarray, stand: np.ndarray,
                  facing: np.ndarray, floor_z: float, up: np.ndarray, action: str,
                  betas: torch.Tensor | None = None) -> PosedHumanoid:
    """Face the part from the stand point and reach the contact with the hand (foot for foot_push)."""
    rotation = canonical_to_world(facing, up)
    left = rotation[:, 0]
    target = contact + TOUCH_OFFSET_M * normal
    height = contact @ up - floor_z
    posture = "squatting" if height < SQUAT_BELOW_M and action != "foot_push" else "standing"
    if action == "foot_push":
        effector, chain, side = "right foot", LEG_CHAIN, None
    else:
        side = "left" if (contact - stand) @ left > 0.05 else "right"
        effector, chain = f"{side} hand", ARM_CHAIN[side]
    base = base_posture(side, posture == "squatting")

    orient = torch.tensor(Rotation.from_matrix(rotation).as_rotvec(), dtype=torch.float32)
    base_t = torch.tensor(base, dtype=torch.float32)
    vertices, joints = forward(model, full_pose(model, orient, base_t), betas)
    transl = torch.tensor(feet_on_floor(vertices.numpy(), joints[0].numpy(), stand, floor_z, up), dtype=torch.float32)

    indices = [JOINT[name] - 1 for name in chain]
    delta = torch.zeros(len(indices), 3, requires_grad=True)
    goal = torch.tensor(target, dtype=torch.float32)
    optimiser = torch.optim.Adam([delta], lr=IK_LR)
    hinge_mask = torch.ones(len(indices), 3)
    for k, name in enumerate(chain):
        if name in HINGES:
            hinge_mask[k, HINGES[name]] = 0.0
    for _ in range(IK_STEPS):
        body = base_t.clone()
        body[indices] = body[indices] + delta
        _, joints = forward(model, full_pose(model, orient, body), betas, transl)
        loss = ((_effector(joints, effector) - goal) ** 2).sum() + IK_REGULARISER * (delta ** 2).sum() \
            + HINGE_PENALTY * ((delta * hinge_mask) ** 2).sum() * IK_REGULARISER
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        body = base_t.clone()
        body[indices] = body[indices] + delta
        vertices, joints = forward(model, full_pose(model, orient, body), betas, transl)
        error = float(torch.linalg.norm(_effector(joints, effector) - goal))
    return PosedHumanoid(vertices.numpy(), joints.numpy(), effector, posture, target, error)

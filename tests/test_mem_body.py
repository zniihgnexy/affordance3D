import numpy as np
import torch

from affordance3d.mem import body
from affordance3d.mem.body import BodyModel, forward, rodrigues


def _toy_model() -> BodyModel:
    """55 joints on a line along +y; vertex k is rigidly attached to joint k (one vertex per joint)."""
    n = body.N_JOINTS
    template = torch.stack([torch.zeros(n), torch.arange(n, dtype=torch.float32), torch.zeros(n)], dim=1)
    parents = (-1, *range(n - 1))  # a single chain 0 -> 1 -> ... -> 54
    return BodyModel(template, torch.zeros(n, 3, 1), torch.zeros(n, 3, 9 * (n - 1)), torch.eye(n), parents,
                     torch.eye(n), np.zeros((0, 3), np.int64), torch.zeros(90))


def test_rodrigues_turns_x_into_y_for_a_quarter_turn_about_z() -> None:
    rotation = rodrigues(torch.tensor([[0.0, 0.0, np.pi / 2]]))[0]

    assert torch.allclose(rotation @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_rest_pose_returns_the_template_plus_translation() -> None:
    model = _toy_model()

    vertices, joints = forward(model, torch.zeros(body.N_JOINTS, 3), transl=torch.tensor([1.0, 0.0, 0.0]))

    assert torch.allclose(vertices, model.v_template + torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(joints, model.v_template + torch.tensor([1.0, 0.0, 0.0]))


def test_rotating_a_joint_moves_its_descendants_about_it_and_leaves_ancestors() -> None:
    model = _toy_model()
    pose = torch.zeros(body.N_JOINTS, 3)
    pose[2] = torch.tensor([0.0, 0.0, -np.pi / 2])  # joint 2 at y=2 turns its children from +y towards +x

    vertices, _ = forward(model, pose)

    assert torch.allclose(vertices[:3], model.v_template[:3])
    assert torch.allclose(vertices[3], torch.tensor([1.0, 2.0, 0.0]), atol=1e-5)
    assert torch.allclose(vertices[5], torch.tensor([3.0, 2.0, 0.0]), atol=1e-5)

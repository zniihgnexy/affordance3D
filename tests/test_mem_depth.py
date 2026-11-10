import numpy as np

from affordance3d.mem.depth_models import depth_errors, with_true_focal


def test_depth_errors_use_only_pixels_with_ground_truth_and_a_prediction() -> None:
    gt = np.array([[1.0, 2.0], [0.0, 4.0]])
    pred = np.array([[1.1, 2.2], [9.0, 0.0]])

    errors = depth_errors(pred, gt)

    assert errors["pixels"] == 2
    assert np.isclose(errors["abs_rel"], 0.1)
    assert np.isclose(errors["median_abs_m"], 0.15)
    assert errors["delta1"] == 1.0


def test_depth_scales_with_the_focal_length_used_to_predict_it() -> None:
    depth = np.array([2.0, 4.0])

    assert np.allclose(with_true_focal(depth, predicted_focal=1000.0, true_focal=1500.0), [3.0, 6.0])

import numpy as np
import pytest

from affordance3d.mem import scenefun3d_eval as SE


@pytest.mark.parametrize("bits", [[0, 1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0], [1]])
def test_rle_round_trips_with_one_based_starts(bits: list[int]) -> None:
    mask = np.array(bits, dtype=bool)

    counts = SE.rle_encode(mask)

    assert SE.rle_decode(counts, len(mask)).tolist() == mask.tolist()
    if bits == [0, 1, 1, 0, 1]:
        assert counts == "2 2 5 1"  # start at point 2 (1-based), 2 points; start at 5, 1 point


def test_write_task_lists_every_mask_with_its_confidence(tmp_path) -> None:
    masks = [np.array([1, 0, 0, 1], dtype=bool), np.array([0, 1, 0, 0], dtype=bool)]

    task_file = SE.write_task(tmp_path, "421254", "abc", masks, [0.9, 0.2], n_points=4)

    lines = [line.split() for line in task_file.read_text().splitlines()]
    assert [score for _, score in lines] == ["0.900000", "0.200000"]
    assert SE.rle_decode((tmp_path / lines[0][0]).read_text(), 4).tolist() == [True, False, False, True]


def test_a_task_without_a_prediction_gets_one_empty_mask(tmp_path) -> None:
    task_file = SE.write_task(tmp_path, "421254", "abc", [], [], n_points=3)

    (path, score), = [line.split() for line in task_file.read_text().splitlines()]
    assert score == "0.000000" and not SE.rle_decode((tmp_path / path).read_text(), 3).any()
    with pytest.raises(ValueError):
        SE.write_task(tmp_path, "421254", "abc", [np.zeros(2, dtype=bool)], [1.0], n_points=3)


def test_element_masks_follow_the_task_order(tmp_path) -> None:
    annotations = [{"annot_id": "a", "indices": [0, 2]}, {"annot_id": "b", "indices": [3]}]

    masks = SE.element_masks(5, annotations, ["b", "a"])

    assert [m.nonzero()[0].tolist() for m in masks] == [[3], [0, 2]]


def test_scan_points_reads_the_vertex_count_from_the_header(tmp_path) -> None:
    ply = tmp_path / "scan.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\nproperty float x\nend_header\n\x00\x01")

    assert SE.scan_points(ply) == 1234


def test_fun3du_thresholds_are_the_float32_values_of_the_historical_code() -> None:
    assert SE.FUN3DU_THRESHOLDS.dtype == np.float32
    assert [float(x) for x in SE.FUN3DU_THRESHOLDS] == [0.5, 0.550000011920929, 0.6000000238418579, 0.6499999761581421,
                                                        0.699999988079071, 0.75, 0.800000011920929, 0.8500000238418579,
                                                        0.8999999761581421, 0.949999988079071]


def test_cropped_indices_keep_the_original_order_and_drop_points_outside_the_crop() -> None:
    crop = np.array([1, 0, 1, 1, 0, 1])

    assert SE.cropped_indices(crop, np.array([5, 0, 1, 3, 3])).tolist() == [0, 2, 3]  # full 0, 3, 5 -> cropped 0, 2, 3


def test_task_ground_truth_is_the_union_of_its_elements_without_exclude_or_missing_ids() -> None:
    crop = np.array([1, 1, 1, 1, 0])
    annotations = [{"annot_id": "a", "indices": [0, 1], "label": "rotate"}, {"annot_id": "b", "indices": [1, 4], "label": "tip_push"},
                   {"annot_id": "x", "indices": [2], "label": "exclude"}]

    assert SE.task_ground_truth(crop, annotations, ["a", "b", "x", "missing"]).tolist() == [0, 1]


def test_fun3du_scores_threshold_precision_and_recall_per_task() -> None:
    gt = np.array([0, 1, 2, 3])

    tight = SE.fun3du_scores(gt, np.array([1]))  # precision 1, recall 0.25
    loose = SE.fun3du_scores(gt, np.array([3, 4, 5, 6, 7, 8, 9, 10]))  # precision 0.125, recall 0.25
    empty = SE.fun3du_scores(gt, np.array([], dtype=np.int64))

    assert tight["AP25"] == tight["AP50"] == tight["mAP"] == 1.0 and tight["AR25"] == 1.0 and tight["AR50"] == 0.0
    assert tight["mIoU"] == pytest.approx(0.25)
    assert loose["AP25"] == 0.0 and loose["AR25"] == 1.0 and loose["mIoU"] == pytest.approx(1 / 11)
    assert all(value == 0.0 for value in empty.values())
    assert SE.fun3du_means([tight, empty])["AP25"] == pytest.approx(50.0)


@pytest.mark.parametrize("ap25, miou, verdict", [((1.0, -4.0, 6.0), (0.0, -2.0, 2.0), "comparable"),
                                                 ((-3.0, -6.0, 0.5), (0.0, -2.0, 2.0), "unclear"),
                                                 ((-8.0, -11.0, -6.0), (0.0, -2.0, 2.0), "behind"),
                                                 ((0.0, -3.0, 3.0), (-4.0, -6.0, -3.0), "behind"),
                                                 ((-14.0, -18.0, -11.0), (0.0, -2.0, 2.0), "clearly behind"),
                                                 ((0.0, -3.0, 3.0), (-7.0, -9.0, -5.5), "clearly behind")])
def test_compare_verdict_reads_ap25_and_miou_by_their_own_margins(ap25, miou, verdict) -> None:
    assert SE.compare_verdict({"AP25": ap25, "mIoU": miou}) == verdict

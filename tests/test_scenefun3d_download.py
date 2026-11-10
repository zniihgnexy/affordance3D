import pytest

from affordance3d.scenefun3d_download import asset_files, asset_url, fill_range, plan_ranges, remaining_range


def test_a_partly_received_range_resumes_after_its_last_byte() -> None:
    assert remaining_range(start=100, end=199, have=0) == (100, 199)
    assert remaining_range(start=100, end=199, have=40) == (140, 199)
    assert remaining_range(start=100, end=199, have=100) is None


def fake_server(data: bytes, failures: list[bool], max_bytes: int):
    """Range fetcher stub: each call either fails (False) or serves up to `max_bytes` of the range."""
    calls = iter(failures)

    def fetch(lo: int, hi: int, piece) -> bool:
        if not next(calls, True):
            return False
        piece.write_bytes(data[lo:min(hi + 1, lo + max_bytes)])
        return True

    return fetch


def test_fill_range_resumes_through_failures_and_partial_answers(tmp_path) -> None:
    data = bytes(range(100))
    part = tmp_path / "f.part40"
    # Three failures in total but never two in a row: passes only if progress resets the stall count.
    fetch = fake_server(data, failures=[False, True, False, True, False, True], max_bytes=7)

    fill_range(40, 59, part, fetch, pause_s=0.0, max_stalls=2)

    assert part.read_bytes() == data[40:60]


def test_fill_range_gives_up_after_consecutive_stalls_only(tmp_path) -> None:
    part = tmp_path / "f.part0"
    always_failing = fake_server(b"x" * 10, failures=[False] * 100, max_bytes=10)
    with pytest.raises(RuntimeError, match="no progress"):
        fill_range(0, 9, part, always_failing, pause_s=0.0, max_stalls=5)


def test_more_bytes_than_requested_is_an_error() -> None:
    with pytest.raises(ValueError, match="more bytes"):
        remaining_range(start=0, end=9, have=11)


def test_visit_assets_follow_the_official_layout() -> None:
    (scan,) = asset_files("421069", "42444490", "laser_scan_5mm")
    assert scan.remote == "421069/421069_laser_scan.ply"
    assert scan.local == "421069/421069_laser_scan.ply"
    assert not scan.is_zip


def test_video_zip_assets_are_extracted_in_the_video_folder() -> None:
    (depth,) = asset_files("421069", "42444490", "hires_depth")
    assert depth.remote == "421069/42444490/hires_depth.zip"
    assert depth.local == "421069/42444490/hires_depth.zip"
    assert depth.is_zip


def test_transform_keeps_the_official_local_rename() -> None:
    (transform,) = asset_files("1", "2", "transform")
    assert transform.remote == "1/2/2_refined_transform.npy"
    assert transform.local == "1/2/2_transform.npy"


def test_unknown_assets_are_rejected() -> None:
    with pytest.raises(ValueError, match="vid_mp4"):
        asset_files("1", "2", "vid_mp4")


def test_ranges_continue_from_the_bytes_already_present() -> None:
    assert plan_ranges(have=10, total=35, chunk=10) == [(10, 19), (20, 29), (30, 34)]
    assert plan_ranges(have=35, total=35, chunk=10) == []


def test_lowres_assets_come_from_arkitscenes_but_land_in_the_scenefun3d_layout() -> None:
    (depth,) = asset_files("421069", "42444490", "lowres_depth")
    assert depth.server == "arkitscenes"
    assert depth.remote == "42444490/lowres_depth.zip"  # ARKitScenes raw layout has no visit folder
    assert depth.local == "421069/42444490/lowres_depth.zip"
    assert depth.is_zip


def test_lowres_poses_keep_the_official_local_rename() -> None:
    (poses,) = asset_files("421069", "42444490", "lowres_poses")
    assert (poses.server, poses.remote, poses.local) == ("arkitscenes", "42444490/lowres_wide.traj",
                                                         "421069/42444490/lowres_poses.traj")


def test_urls_use_the_scenefun3d_split_or_the_arkitscenes_fold() -> None:
    (scan,) = asset_files("421069", "42444490", "laser_scan_5mm")
    (depth,) = asset_files("421069", "42444490", "lowres_depth")
    assert asset_url(scan, split="train", arkit_fold="Training") == \
        "https://cvg-data.inf.ethz.ch/scenefun3d/v1/train/421069/421069_laser_scan.ply"
    assert asset_url(depth, split="train", arkit_fold="Validation") == \
        "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42444490/lowres_depth.zip"

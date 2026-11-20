"""scripts/download_revisit_videos.py: which videos count as the second visit, and the shared scene files."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "download_revisit_videos", Path(__file__).resolve().parents[1] / "scripts" / "download_revisit_videos.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_other_videos_skip_the_primary_and_other_scenes():
    pairs = [["1", "a"], ["1", "b"], ["1", "c"], ["2", "d"], ["3", "e"], ["3", "f"]]
    assert R.other_videos(["1", "3"], pairs, {"1": "a", "3": "f"}) == [("1", "b"), ("1", "c"), ("3", "e")]


def test_scene_files_are_linked_not_copied(tmp_path):
    primary, root = tmp_path / "primary", tmp_path / "revisit"
    (primary / "7").mkdir(parents=True)
    (primary / "7" / "7_annotations.json").write_text("{}")
    R.link_scene_files(primary, root, "7")
    link = root / "7" / "7_annotations.json"
    assert link.is_symlink() and link.resolve() == (primary / "7" / "7_annotations.json").resolve()
    assert not (root / "7" / "7_laser_scan.ply").exists()  # missing in the primary root: nothing to link

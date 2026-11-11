import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "fetch_models.py"
spec = importlib.util.spec_from_file_location("fetch_models", SCRIPT)
fetch_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_models)


def test_digest_of_uses_sha256_for_lfs_files_and_the_git_blob_hash_otherwise(tmp_path: Path) -> None:
    file = tmp_path / "weights.bin"
    file.write_bytes(b"hello")

    lfs = fetch_models.digest_of(file, {"lfs": {"sha256": "published"}})
    plain = fetch_models.digest_of(file, {"blobId": "published"})

    assert lfs == ("published", hashlib.sha256(b"hello").hexdigest())
    assert plain == ("published", hashlib.sha1(b"blob 5\0hello").hexdigest())  # what `git hash-object` gives


@pytest.mark.parametrize(("name", "safe"), [("config.json", True), ("onnx/model.onnx", True), ("", False),
                                            ("/etc/passwd", False), ("../outside", False), ("a/../../b", False)])
def test_safe_name_keeps_files_inside_the_model_folder(name: str, safe: bool) -> None:
    assert fetch_models.safe_name(name) is safe


@pytest.mark.parametrize("spec_text", ["org/name", "org/name@main", "../x/name@" + "a" * 40, "org/name@" + "g" * 40])
def test_fetch_refuses_anything_but_a_repository_pinned_to_a_full_commit_hash(spec_text: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        fetch_models.fetch(spec_text, tmp_path)

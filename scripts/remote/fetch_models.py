#!/usr/bin/env python3
"""Fetch model repositories through the ModelScope mirror and check every file against the Hugging Face hub.

The hub is slow from the lab server, the mirror is not. Files come from the mirror; their hashes are compared
with the ones the hub publishes for the pinned revision (sha256 for LFS files, git blob sha1 for the rest), so
a file that differs from the original is rejected.

    python scripts/remote/fetch_models.py                 # the models of the L2 memory examples
    python scripts/remote/fetch_models.py org/name@sha    # any other repository
    python scripts/remote/fetch_models.py org/name@sha --skip=sam3.pt   # leave out files matching a pattern

A skipped file is neither fetched nor checked; the folder is marked verified for the files that are there.

Models land in $AFFORDANCE3D_MODELS/<org>/<name> (see scripts/remote/env.sh).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_MODELS = (
    "google/siglip2-base-patch16-384@f775b65a79762255128c981547af89addcfe0f88",
    "Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17",
)
HUB_API = "https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
MIRROR_FILE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"
CHUNK_BYTES = 1 << 22
TIMEOUT_S = 60
SKIP = (".gitattributes", "README.md")
SPEC = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")  # org/name@full commit hash
VERIFIED_MARKER = ".verified"  # read by affordance3d.mem.encoders.model_source


def safe_name(name: str) -> bool:
    """True for a relative file name that stays inside the model folder."""
    return bool(name) and not name.startswith("/") and ".." not in Path(name).parts


def published_files(repo: str, revision: str) -> list[dict]:
    with urllib.request.urlopen(HUB_API.format(repo=repo, revision=revision), timeout=TIMEOUT_S) as response:
        payload = json.load(response)
    files = [f for f in payload.get("siblings", []) if f["rfilename"] not in SKIP]
    if payload.get("sha") != revision or not files:
        raise SystemExit(f"{repo}: the hub did not return the files of revision {revision}")
    return files


def digest_of(path: Path, published: dict) -> tuple[str, str]:
    """(expected, actual) hash of a file: sha256 for LFS files, git blob sha1 otherwise."""
    lfs = published.get("lfs")
    hasher = hashlib.sha256() if lfs else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            hasher.update(chunk)
    return (lfs["sha256"] if lfs else published["blobId"]), hasher.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
    partial.replace(target)


def split_arguments(arguments: list[str]) -> tuple[list[str], tuple[str, ...]]:
    """(model specs, file-name patterns to leave out) from the command line."""
    skips = tuple(a.split("=", 1)[1] for a in arguments if a.startswith("--skip="))
    return [a for a in arguments if not a.startswith("--skip=")], skips


def fetch(spec: str, root: Path, skips: tuple[str, ...] = ()) -> None:
    if not SPEC.match(spec):
        raise SystemExit(f"{spec!r}: expected org/name@<40-character commit hash>")
    repo, revision = spec.split("@")
    marker = root / repo / VERIFIED_MARKER
    marker.unlink(missing_ok=True)
    for published in published_files(repo, revision):
        name = published["rfilename"]
        if any(fnmatch.fnmatch(name, pattern) for pattern in skips):
            print(f"skip {repo}/{name}", flush=True)
            continue
        if not safe_name(name):
            raise SystemExit(f"{repo}: refusing the file name {name!r}")
        target = root / repo / name
        if not target.exists():
            download(MIRROR_FILE.format(repo=repo, path=urllib.parse.quote(name)), target)
        expected, actual = digest_of(target, published)
        if expected != actual:
            target.unlink()
            raise SystemExit(f"{repo}/{name}: hash {actual} differs from the published {expected}; file removed")
        print(f"ok  {repo}/{name}  ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    marker.write_text(revision + "\n")


def main() -> None:
    if "AFFORDANCE3D_MODELS" not in os.environ:
        raise SystemExit("AFFORDANCE3D_MODELS is not set; source scripts/remote/env.sh first")
    root = Path(os.environ["AFFORDANCE3D_MODELS"])
    specs, skips = split_arguments(sys.argv[1:])
    for spec in specs or DEFAULT_MODELS:
        fetch(spec, root, skips)


if __name__ == "__main__":
    main()

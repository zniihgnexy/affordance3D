"""SceneFun3D asset downloader that copes with the server slowing long-lived connections.

Same URLs and local layout as the official toolkit downloader
(third_party/scenefun3d/data_downloader), but each file is fetched in fixed-size byte
ranges over fresh connections (a few at a time), resumed from any partial `.tmp`,
size-checked, and zips are CRC-tested (`unzip -t`) before being extracted and deleted.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

BASE_URL = "https://cvg-data.inf.ethz.ch/scenefun3d/v1"
CHUNK_BYTES = 8 * 2**20
WORKERS = 4  # concurrent connections, similar to common download managers
ATTEMPT_TIMEOUT_S = 120  # a stalled connection is dropped; bytes already received are kept
MAX_STALLS = 30  # consecutive attempts without any new byte before giving up (progress resets it)
RETRY_PAUSE_S = 5.0  # pause after an attempt without progress, growing with each further stall
MAX_PAUSE_S = 60.0
HTTP_PARTIAL_CONTENT = "206"

VISIT_FILES = {
    "laser_scan_5mm": "{visit}_laser_scan.ply",
    "crop_mask": "{visit}_crop_mask.npy",
    "annotations": "{visit}_annotations.json",
    "descriptions": "{visit}_descriptions.json",
    "motions": "{visit}_motions.json",
}
VIDEO_ZIPS = ("hires_wide", "hires_depth", "hires_wide_intrinsics")
DEFAULT_ASSETS = ("laser_scan_5mm", "crop_mask", "annotations", "descriptions",
                  "hires_wide", "hires_wide_intrinsics", "hires_depth", "hires_poses")


@dataclass(frozen=True)
class AssetFile:
    remote: str  # path below <BASE_URL>/<split>/
    local: str  # path below the local data root
    is_zip: bool


def asset_files(visit: str, video: str, asset: str) -> list[AssetFile]:
    """Files making up one asset, with the official remote and local paths."""
    if asset in VISIT_FILES:
        name = VISIT_FILES[asset].format(visit=visit)
        return [AssetFile(f"{visit}/{name}", f"{visit}/{name}", False)]
    if asset in VIDEO_ZIPS:
        path = f"{visit}/{video}/{asset}.zip"
        return [AssetFile(path, path, True)]
    if asset == "hires_poses":
        path = f"{visit}/{video}/hires_poses.traj"
        return [AssetFile(path, path, False)]
    if asset == "transform":
        return [AssetFile(f"{visit}/{video}/{video}_refined_transform.npy", f"{visit}/{video}/{video}_transform.npy", False)]
    raise ValueError(f"unsupported asset {asset!r}")


def plan_ranges(have: int, total: int, chunk: int) -> list[tuple[int, int]]:
    """Inclusive byte ranges still missing after the first `have` bytes."""
    return [(start, min(start + chunk, total) - 1) for start in range(have, total, chunk)]


def content_length(url: str) -> int:
    head = subprocess.run(["curl", "-sIL", "--max-time", "30", url], capture_output=True, text=True, check=True).stdout
    sizes = [int(line.split()[1]) for line in head.splitlines() if line.lower().startswith("content-length:")]
    if not sizes:
        raise RuntimeError(f"no content-length for {url}")
    return sizes[-1]


def remaining_range(start: int, end: int, have: int) -> tuple[int, int] | None:
    """Inclusive byte range still needed for [start, end] after `have` bytes; None when complete."""
    expected = end - start + 1
    if have > expected:
        raise ValueError(f"received {have} bytes, more bytes than the {expected} requested")
    return None if have == expected else (start + have, end)


def _curl_range(url: str, lo: int, hi: int, piece: Path) -> bool:
    """Fetch bytes [lo, hi] into `piece`; True only for a 206 answer (a 200 would be the whole file)."""
    result = subprocess.run(["curl", "-s", "--max-time", str(ATTEMPT_TIMEOUT_S), "-r", f"{lo}-{hi}",
                             "-o", str(piece), "-w", "%{http_code}", url], capture_output=True, text=True,
                            check=False)  # timeouts and errors are expected: the HTTP code decides
    return result.stdout.strip() == HTTP_PARTIAL_CONTENT and piece.exists()


def fill_range(start: int, end: int, part: Path, fetch: Callable[[int, int, Path], bool],
               pause_s: float = RETRY_PAUSE_S, max_stalls: int = MAX_STALLS) -> Path:
    """Fill `part` with bytes [start, end], resuming inside the range after stalls, timeouts or errors.

    Only consecutive attempts that add no bytes count towards `max_stalls`; each one waits a
    little longer before retrying, so a briefly failing server does not exhaust the retries.
    """
    piece = part.with_name(part.name + ".piece")
    stalls = 0
    while (needed := remaining_range(start, end, part.stat().st_size if part.exists() else 0)) is not None:
        if stalls >= max_stalls:
            raise RuntimeError(f"{part.name}: no progress in {max_stalls} consecutive attempts")
        received = 0
        if fetch(needed[0], needed[1], piece):
            received = piece.stat().st_size
            with part.open("ab") as out, piece.open("rb") as data:
                shutil.copyfileobj(data, out)
        piece.unlink(missing_ok=True)
        stalls = 0 if received else stalls + 1
        if not received:
            time.sleep(min(pause_s * stalls, MAX_PAUSE_S))
    return part


def download_file(url: str, dest: Path, chunk: int = CHUNK_BYTES, workers: int = WORKERS) -> None:
    """Download `url` to `dest` in byte ranges, keeping any bytes already in `<dest>.tmp`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = content_length(url)
    tmp = dest.with_name(dest.name + ".tmp")
    have = tmp.stat().st_size if tmp.exists() else 0
    ranges = plan_ranges(have, total, chunk)
    print(f"{dest.name}: {have / 1e6:.0f}/{total / 1e6:.0f} MB present, {len(ranges)} ranges to fetch", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        fetch = partial(_curl_range, url)
        parts = list(pool.map(lambda r: fill_range(r[0], r[1], tmp.with_name(f"{tmp.name}.part{r[0]}"), fetch), ranges))
    with tmp.open("ab") as out:
        for part in parts:
            with part.open("rb") as piece:
                shutil.copyfileobj(piece, out)
            part.unlink()
    if tmp.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: assembled {tmp.stat().st_size} bytes, expected {total}")
    tmp.rename(dest)


def extract_zip(zip_path: Path) -> None:
    """CRC-test every member, extract next to the zip, then delete it (like the official tool)."""
    subprocess.run(["unzip", "-tq", str(zip_path)], check=True)
    subprocess.run(["unzip", "-oq", str(zip_path), "-d", str(zip_path.parent)], check=True)
    zip_path.unlink()


def download_scene(root: Path, visit: str, video: str, assets: tuple[str, ...] = DEFAULT_ASSETS,
                   split: str = "train") -> None:
    for asset in assets:
        for item in asset_files(visit, video, asset):
            local = root / item.local
            extracted = local.with_suffix("") if item.is_zip else None
            if (extracted is not None and extracted.is_dir()) or (extracted is None and local.exists()):
                print(f"exists: {item.local}", flush=True)
                continue
            if not local.exists():
                download_file(f"{BASE_URL}/{split}/{item.remote}", local)
            if item.is_zip:
                extract_zip(local)
            print(f"done: {item.local}", flush=True)

"""Read a few members of a remote zip through HTTP range requests, without downloading the archive.

A video of a dataset can be a zip of a thousand frames of which a memory needs forty. `zipfile` only
needs a seekable file, so `RangeReader` serves its reads from byte ranges fetched on demand.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SMALL_READ_BYTES = 1 << 16  # what an uncached read fetches at least (zip directory, local headers)
HEADER_SLACK_BYTES = 512  # a member's local header: fixed part, name and extra field
TIMEOUT_S = 120
ATTEMPTS = 8
RETRY_PAUSE_S = 4.0
HTTP_PARTIAL_CONTENT = 206

Fetch = Callable[[int, int], bytes]  # bytes of the inclusive range [first, last]


class RangeReader(io.RawIOBase):
    """A read-only seekable file of `size` bytes; missing bytes are fetched and kept."""

    def __init__(self, size: int, fetch: Fetch, small_read: int = SMALL_READ_BYTES) -> None:
        super().__init__()
        self._size, self._fetch, self._small_read = size, fetch, small_read
        self._position = 0
        self._ranges: list[tuple[int, bytes]] = []  # (first byte, data)
        self._fetched = [0]  # shared with clones

    @property
    def fetched_bytes(self) -> int:
        """Bytes fetched so far by this reader and its clones."""
        return self._fetched[0]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = min(max(base + offset, 0), self._size)
        return self._position

    def clone(self) -> RangeReader:
        """Another reader of the same file that shares the bytes fetched so far (for use in another thread)."""
        other = RangeReader(self._size, self._fetch, self._small_read)
        other._ranges, other._fetched = self._ranges, self._fetched
        return other

    def prefetch(self, first: int, last: int) -> None:
        """Fetch [first, last] in one request, so that later reads inside it need none."""
        first, last = max(first, 0), min(last, self._size - 1)
        if first <= last and self._cached(first, last - first + 1) is None:
            data = self._fetch(first, last)
            self._fetched[0] += len(data)
            self._ranges.append((first, data))

    def _cached(self, position: int, count: int) -> bytes | None:
        for first, data in self._ranges:
            if first <= position and position + count <= first + len(data):
                return data[position - first:position - first + count]
        return None

    def read(self, count: int = -1) -> bytes:
        available = self._size - self._position
        count = available if count < 0 else min(count, available)
        if count == 0:
            return b""
        if self._cached(self._position, count) is None:
            self.prefetch(self._position, self._position + max(count, self._small_read) - 1)
        data = self._cached(self._position, count)
        self._position += count
        return data


def http_range(url: str, first: int, last: int, attempts: int = ATTEMPTS) -> bytes:
    """Bytes [first, last] of `url`; retried, and refused unless the server answers with exactly that range."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                data = response.read()
                if response.status == HTTP_PARTIAL_CONTENT and len(data) == last - first + 1:
                    return data
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(RETRY_PAUSE_S * (attempt + 1))
    raise RuntimeError(f"{url}: bytes {first}-{last} could not be fetched in {attempts} attempts")


def member_names(reader: RangeReader) -> list[str]:
    """Names of the files in the archive (fetches only the directory at its end)."""
    reader.prefetch(reader._size - reader._small_read * 4, reader._size - 1)
    with zipfile.ZipFile(reader) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


def _extract_one(reader: RangeReader, name: str, dest: Path) -> bool:
    target = dest / name
    if target.exists():
        return False
    with zipfile.ZipFile(reader.clone()) as archive:  # the directory is already cached: no request
        info = archive.getinfo(name)
        own = archive.fp
        own.prefetch(info.header_offset, info.header_offset + len(name.encode()) + HEADER_SLACK_BYTES + info.compress_size)
        data = archive.read(info)  # zipfile checks the CRC
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return True


def extract_members(reader: RangeReader, names: Iterable[str], dest: Path, workers: int = 1) -> int:
    """Extract the named members below `dest` (each CRC-checked); returns how many were written.

    `reader` must already hold the archive directory (call `member_names` first).
    """
    names = list(dict.fromkeys(names))
    if workers <= 1:
        return sum(_extract_one(reader, name, dest) for name in names)
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(lambda name: _extract_one(reader, name, dest), names))

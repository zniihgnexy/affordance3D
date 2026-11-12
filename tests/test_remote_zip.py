import io
import zipfile
from pathlib import Path

import pytest

from affordance3d import remote_zip as R


def _archive(n: int = 40, size: int = 20_000) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for i in range(n):
            archive.writestr(f"frames/f_{i:03d}.bin", bytes([i]) * size)
    return buffer.getvalue()


def _reader(data: bytes, calls: list[tuple[int, int]]) -> R.RangeReader:
    def fetch(first: int, last: int) -> bytes:
        calls.append((first, last))
        return data[first:last + 1]

    return R.RangeReader(len(data), fetch, small_read=4096)


def test_member_names_reads_only_the_end_of_the_archive() -> None:
    data, calls = _archive(), []

    reader = _reader(data, calls)
    names = R.member_names(reader)

    assert names[0] == "frames/f_000.bin" and len(names) == 40
    assert all(first > len(data) // 2 for first, _ in calls)  # nothing from the first half was fetched


def test_extract_members_fetches_little_more_than_the_wanted_files(tmp_path: Path) -> None:
    data, calls = _archive(), []
    reader = _reader(data, calls)
    R.member_names(reader)

    written = R.extract_members(reader, ["frames/f_003.bin", "frames/f_031.bin"], tmp_path, workers=2)
    again = R.extract_members(reader, ["frames/f_003.bin"], tmp_path)

    assert written == 2 and again == 0  # an existing file is not fetched again
    assert (tmp_path / "frames" / "f_031.bin").read_bytes() == bytes([31]) * 20_000
    assert not (tmp_path / "frames" / "f_000.bin").exists()
    assert reader.fetched_bytes < len(data) // 5  # 2 of 40 files, plus the directory


def test_range_reader_serves_reads_seeks_and_the_end_of_file() -> None:
    data, calls = bytes(range(256)) * 40, []
    reader = _reader(data, calls)

    reader.seek(100)
    first = reader.read(10)
    reader.seek(-5, io.SEEK_END)
    tail = reader.read()

    assert first == data[100:110] and tail == data[-5:] and reader.read(3) == b""
    assert reader.tell() == len(data)


def test_a_corrupted_member_is_rejected(tmp_path: Path) -> None:
    data = bytearray(_archive(n=3, size=5000))
    data[200] ^= 0xFF  # flip one byte inside the first member

    reader = _reader(bytes(data), [])
    R.member_names(reader)
    with pytest.raises(zipfile.BadZipFile):
        R.extract_members(reader, ["frames/f_000.bin"], tmp_path)

"""Read zip archives over HTTP using range requests, without writing them to disk.

zipfile only needs a seekable file object. fsspec provides one backed by HTTP range requests,
so listing members fetches just the archive's central directory and reading a member fetches
only that member's compressed bytes.
"""

import re
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import fsspec

TRANSACTIONS_RE = re.compile(r"(?:^|/)transactions_(\d{6})\.csv$")
CALENDAR_RE = re.compile(r"(?:^|/)time\.csv$")
HTTP_BLOCK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class Member:
    name: str
    kind: Literal["transactions", "calendar"]
    shop_week: int | None
    compressed_size: int
    size: int
    crc: int


def classify(info: zipfile.ZipInfo) -> Member | None:
    """Return a Member for files we ingest, or None for directories and anything else."""
    if info.is_dir():
        return None
    if match := TRANSACTIONS_RE.search(info.filename):
        kind, week = "transactions", int(match.group(1))
    elif CALENDAR_RE.search(info.filename):
        kind, week = "calendar", None
    else:
        return None
    return Member(info.filename, kind, week, info.compress_size, info.file_size, info.CRC)


@contextmanager
def open_archive(url: str) -> Generator[zipfile.ZipFile]:
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb", block_size=HTTP_BLOCK_SIZE) as handle, zipfile.ZipFile(handle) as zf:
        yield zf


def list_members(archive: zipfile.ZipFile) -> list[Member]:
    members = [m for info in archive.infolist() if (m := classify(info)) is not None]
    return sorted(members, key=lambda m: (m.kind, m.shop_week or 0))


def read_member(archive: zipfile.ZipFile, member: Member) -> bytes:
    data = archive.read(member.name)
    if len(data) != member.size:
        raise OSError(f"{member.name}: read {len(data)} bytes, expected {member.size}")
    return data

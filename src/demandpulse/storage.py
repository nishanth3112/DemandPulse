"""Key/value byte storage with local-disk and S3 backends.

Pipeline code only talks to `Store`, so tests and offline work run against a local directory
while real runs write to the private S3 bucket.
"""

from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import boto3
from botocore.exceptions import ClientError

from demandpulse.config import Settings


class Store(Protocol):
    def exists(self, key: str) -> bool: ...
    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...


def _validate_key(key: str) -> str:
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _validate_key(key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run never leaves a half-written object.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def list_keys(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        return sorted(k for k in keys if k.startswith(prefix) and not k.endswith(".tmp"))


class S3Store:
    def __init__(self, bucket: str, client: Any) -> None:
        self.bucket = bucket
        self.client = client

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=_validate_key(key))
        except ClientError as err:
            if err.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=_validate_key(key))
        return response["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        # S3 puts are atomic: readers see the whole object or none of it.
        self.client.put_object(Bucket=self.bucket, Key=_validate_key(key), Body=data)

    def list_keys(self, prefix: str = "") -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)


def open_store(settings: Settings) -> Store:
    if settings.s3_bucket:
        session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
        return S3Store(settings.s3_bucket, session.client("s3"))
    if settings.local_data_root:
        return LocalStore(settings.local_data_root)
    raise ValueError(
        "No storage configured: set DEMANDPULSE_S3_BUCKET or DEMANDPULSE_LOCAL_DATA_ROOT"
    )

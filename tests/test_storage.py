import io

import pytest
from botocore.exceptions import ClientError

from demandpulse.config import Settings
from demandpulse.storage import LocalStore, S3Store, open_store


class FakeS3Client:
    """In-memory stand-in implementing the boto3 calls S3Store uses."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                contents = [{"Key": k} for k in client.objects if k.startswith(Prefix)]
                return [{"Contents": contents}] if contents else [{}]

        return Paginator()


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path)
    return S3Store("bucket", FakeS3Client())


def test_write_read_exists_round_trip(store):
    assert not store.exists("a/b.bin")
    store.write_bytes("a/b.bin", b"hello")
    assert store.exists("a/b.bin")
    assert store.read_bytes("a/b.bin") == b"hello"


def test_overwrite_replaces_content(store):
    store.write_bytes("k", b"one")
    store.write_bytes("k", b"two")
    assert store.read_bytes("k") == b"two"


def test_list_keys_filters_by_prefix_and_is_sorted(store):
    for key in ["bronze/b", "bronze/a", "silver/x"]:
        store.write_bytes(key, b"1")
    assert store.list_keys("bronze/") == ["bronze/a", "bronze/b"]
    assert store.list_keys("missing/") == []


@pytest.mark.parametrize("key", ["", "/abs", "../escape", "a/../../b"])
def test_invalid_keys_are_rejected(store, key):
    with pytest.raises(ValueError, match="Invalid storage key"):
        store.exists(key)


def test_local_store_leaves_no_tmp_files(tmp_path):
    local = LocalStore(tmp_path)
    local.write_bytes("x/y", b"data")
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["y"]


def test_open_store_prefers_configured_backend(tmp_path):
    assert isinstance(open_store(Settings(local_data_root=tmp_path)), LocalStore)
    with pytest.raises(ValueError, match="No storage configured"):
        open_store(Settings())

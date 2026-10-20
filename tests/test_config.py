import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from demandpulse.config import Settings
from demandpulse.logging_setup import configure_logging


def test_defaults_when_env_is_empty():
    settings = Settings.from_env({})
    assert settings.aws_profile == "demandpulse"
    assert settings.aws_region == "us-east-1"
    assert settings.s3_bucket is None


def test_s3_bucket_becomes_storage_root():
    settings = Settings.from_env({"DEMANDPULSE_S3_BUCKET": "my-bucket"})
    assert settings.storage_root == "s3://my-bucket"


def test_local_root_becomes_storage_root(tmp_path):
    settings = Settings.from_env({"DEMANDPULSE_LOCAL_DATA_ROOT": str(tmp_path)})
    assert settings.local_data_root == Path(tmp_path)
    assert settings.storage_root == str(tmp_path)


def test_empty_strings_are_treated_as_unset():
    settings = Settings.from_env({"DEMANDPULSE_S3_BUCKET": "", "AWS_PROFILE": ""})
    assert settings.s3_bucket is None
    assert settings.aws_profile == "demandpulse"


def test_both_storage_targets_are_rejected():
    with pytest.raises(ValidationError, match="not both"):
        Settings.from_env({"DEMANDPULSE_S3_BUCKET": "b", "DEMANDPULSE_LOCAL_DATA_ROOT": "/tmp/x"})


def test_storage_root_requires_a_target():
    with pytest.raises(ValueError, match="No storage configured"):
        _ = Settings.from_env({}).storage_root


def test_configure_logging_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1

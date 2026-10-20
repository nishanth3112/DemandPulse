import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class Settings(BaseModel):
    """Runtime settings read from environment variables.

    Credentials are never stored here; boto3 resolves them from the named AWS profile.
    """

    model_config = ConfigDict(frozen=True)

    aws_profile: str = "demandpulse"
    aws_region: str = "us-east-1"
    s3_bucket: str | None = None
    local_data_root: Path | None = None

    @model_validator(mode="after")
    def _not_both_storage_targets(self) -> "Settings":
        if self.s3_bucket and self.local_data_root:
            raise ValueError(
                "Set either DEMANDPULSE_S3_BUCKET or DEMANDPULSE_LOCAL_DATA_ROOT, not both"
            )
        return self

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        values: dict[str, str] = {}
        for field, var in (
            ("aws_profile", "AWS_PROFILE"),
            ("aws_region", "AWS_REGION"),
            ("s3_bucket", "DEMANDPULSE_S3_BUCKET"),
            ("local_data_root", "DEMANDPULSE_LOCAL_DATA_ROOT"),
        ):
            # Empty strings (as in .env.example) mean "unset".
            if env.get(var):
                values[field] = env[var]
        return cls.model_validate(values)

    @property
    def storage_root(self) -> str:
        if self.s3_bucket:
            return f"s3://{self.s3_bucket}"
        if self.local_data_root:
            return str(self.local_data_root)
        raise ValueError(
            "No storage configured: set DEMANDPULSE_S3_BUCKET or DEMANDPULSE_LOCAL_DATA_ROOT"
        )

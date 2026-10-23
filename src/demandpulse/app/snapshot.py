"""Read-only access to the local snapshot the dashboard runs on.

The dashboard never talks to S3 and holds no credentials: `demandpulse.app.export` copies a fixed
set of small tables out of the lake once, and the app reads only that folder through `Snapshot`.
This module imports nothing from the storage or warehouse layers, and a test keeps it that way.
The manifest is written last by the export, so a folder without one is an unfinished export and
`Snapshot` refuses to open it.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_APP_DIR = Path("data/app")
APP_DATA_ENV = "DEMANDPULSE_APP_DATA"
MANIFEST = "snapshot.json"


class Snapshot:
    """Read-only access to an exported snapshot folder, with each table read at most once."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / MANIFEST).is_file():
            raise FileNotFoundError(
                f"No dashboard data in {self.root}. Run `demandpulse app export` first."
            )
        self.manifest: dict[str, Any] = json.loads((self.root / MANIFEST).read_text())
        self._tables: dict[str, pl.DataFrame] = {}

    def has(self, name: str) -> bool:
        return name in self.manifest["tables"]

    def table(self, name: str) -> pl.DataFrame:
        if name not in self._tables:
            if not self.has(name):
                raise KeyError(f"Snapshot has no table {name!r}")
            self._tables[name] = pl.read_parquet(self.root / f"{name}.parquet")
        return self._tables[name]

    def json(self, name: str) -> dict[str, Any]:
        path = self.root / f"{name}.json"
        if not path.is_file():
            raise KeyError(f"Snapshot has no document {name!r}")
        return json.loads(path.read_text())

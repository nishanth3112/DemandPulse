from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "config" / "sources.yaml"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["sample", "full"]
    url: str


def load_sources(path: Path = DEFAULT_REGISTRY) -> dict[str, Source]:
    raw = yaml.safe_load(path.read_text())
    sources = [Source.model_validate(item) for item in raw["sources"]]
    by_name = {s.name: s for s in sources}
    if len(by_name) != len(sources):
        raise ValueError(f"Duplicate source names in {path}")
    return by_name


def get_source(name: str, path: Path = DEFAULT_REGISTRY) -> Source:
    sources = load_sources(path)
    if name not in sources:
        raise KeyError(f"Unknown source {name!r}. Available: {', '.join(sorted(sources))}")
    return sources[name]

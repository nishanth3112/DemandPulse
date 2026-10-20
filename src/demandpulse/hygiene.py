"""Repository hygiene guard: fail if tracked files look like data dumps or contain secrets."""

import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 1_000_000
FORBIDDEN_SUFFIXES = {".parquet", ".zip", ".gz", ".duckdb", ".pkl", ".joblib", ".pt", ".ckpt"}
SECRET_PATTERNS = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS secret access key assignment": re.compile(
        r"aws_secret_access_key\s*[=:]\s*\S{20,}", re.IGNORECASE
    ),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return [root / name for name in out.split("\0") if name]


def check_file(path: Path, root: Path) -> list[str]:
    rel = path.relative_to(root)
    problems: list[str] = []
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append(f"{rel}: forbidden file type {path.suffix}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        problems.append(f"{rel}: {size} bytes exceeds {MAX_FILE_BYTES}")
    # Binary files and the lockfile are not scanned for text patterns.
    if size <= MAX_FILE_BYTES and path.name != "uv.lock":
        text = path.read_bytes().decode("utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{rel}: possible {label}")
    return problems


def check_repo(root: Path) -> list[str]:
    problems: list[str] = []
    for path in tracked_files(root):
        if path.is_file():
            problems.extend(check_file(path, root))
    return problems


def main() -> int:
    problems = check_repo(Path.cwd())
    for problem in problems:
        print(f"HYGIENE: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("Repository hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

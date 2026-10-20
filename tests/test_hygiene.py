import subprocess
from pathlib import Path

from demandpulse.hygiene import MAX_FILE_BYTES, check_repo

# Built by concatenation so this file does not itself trip the guard.
FAKE_ACCESS_KEY = "AKIA" + "A" * 16
FAKE_SECRET_LINE = "aws_secret_access_key" + " = " + "x" * 30


def make_repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_clean_repo_passes(tmp_path):
    repo = make_repo(tmp_path, {"a.py": b"print('hi')\n", "tests/fixtures/s.csv": b"a,b\n1,2\n"})
    assert check_repo(repo) == []


def test_flags_data_file_types(tmp_path):
    repo = make_repo(tmp_path, {"data/week.parquet": b"PAR1"})
    assert any("forbidden file type" in p for p in check_repo(repo))


def test_flags_oversized_files(tmp_path):
    repo = make_repo(tmp_path, {"big.txt": b"x" * (MAX_FILE_BYTES + 1)})
    assert any("exceeds" in p for p in check_repo(repo))


def test_flags_aws_access_key(tmp_path):
    repo = make_repo(tmp_path, {"config.txt": f"key={FAKE_ACCESS_KEY}\n".encode()})
    assert any("AWS access key id" in p for p in check_repo(repo))


def test_flags_secret_assignment(tmp_path):
    repo = make_repo(tmp_path, {"creds": FAKE_SECRET_LINE.encode()})
    assert any("secret access key" in p for p in check_repo(repo))


def test_flags_private_key_block(tmp_path):
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    repo = make_repo(tmp_path, {"id_rsa": header.encode()})
    assert any("private key" in p for p in check_repo(repo))

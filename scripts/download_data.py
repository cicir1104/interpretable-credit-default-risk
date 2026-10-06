"""Download and verify the official UCI workbook."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
EXPECTED_ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
EXPECTED_WORKBOOK_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
WORKBOOK_NAME = "default of credit card clients.xls"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "default_of_credit_card_clients.zip"
WORKBOOK_PATH = RAW_DIR / WORKBOOK_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: expected {expected}, got {actual}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        temporary = ARCHIVE_PATH.with_suffix(".download")
        urllib.request.urlretrieve(DATASET_URL, temporary)
        temporary.replace(ARCHIVE_PATH)
    require_hash(ARCHIVE_PATH, EXPECTED_ARCHIVE_SHA256)

    if not WORKBOOK_PATH.exists():
        with zipfile.ZipFile(ARCHIVE_PATH) as archive:
            names = archive.namelist()
            if names != [WORKBOOK_NAME]:
                raise RuntimeError(f"Unexpected archive contents: {names}")
            with archive.open(WORKBOOK_NAME) as source, WORKBOOK_PATH.open("wb") as target:
                target.write(source.read())
    require_hash(WORKBOOK_PATH, EXPECTED_WORKBOOK_SHA256)

    print(f"Verified {ARCHIVE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Verified {WORKBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

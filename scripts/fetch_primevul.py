#!/usr/bin/env python3
"""Fetch and verify the exact PrimeVul main splits used by CWE119Bench."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/raw/primevul_main"
HF_REVISION = "4fd7158322872d711e90f091dbd8673ef32cb1be"
HF_BASE_URL = (
    "https://huggingface.co/datasets/colin/PrimeVul/resolve/"
    f"{HF_REVISION}"
)
FILES = {
    "primevul_train.jsonl": {
        "size": 470_918_175,
        "sha256": "5cadee7f3383124cc26ef6e67c8dd76c321e5461c94bef399a2f03a28767e6f5",
    },
    "primevul_valid.jsonl": {
        "size": 63_564_497,
        "sha256": "7f0b4d05b063c20d5cf57478128538b8c0c36354e8def7f2d4febe6773b9dc7c",
    },
    "primevul_test.jsonl": {
        "size": 66_064_689,
        "sha256": "c9dcab5ee897da36eccf0dbb3b510de2dab66a7dec681af29a7a28c8a53331c4",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_error(path: Path, expected: dict[str, int | str]) -> str | None:
    if not path.is_file():
        return "missing"
    actual_size = path.stat().st_size
    if actual_size != expected["size"]:
        return f"size {actual_size}, expected {expected['size']}"
    actual_sha = sha256_file(path)
    if actual_sha != expected["sha256"]:
        return f"SHA-256 {actual_sha}, expected {expected['sha256']}"
    return None


def download(url: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CWE119Bench-reproducibility-fetch/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without downloading missing files.",
    )
    parser.add_argument(
        "--replace-invalid",
        action="store_true",
        help="Replace an existing file only when it fails size or SHA-256 validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for filename, expected in FILES.items():
        path = args.output_dir / filename
        error = validation_error(path, expected)
        if error is None:
            print(f"OK {filename}")
            continue

        if args.verify_only:
            failures.append(f"{filename}: {error}")
            continue

        if path.exists() and not args.replace_invalid:
            failures.append(
                f"{filename}: {error}. Pass --replace-invalid to replace this file"
            )
            continue

        print(f"Downloading {filename} from pinned PrimeVul snapshot {HF_REVISION}")
        download(f"{HF_BASE_URL}/{filename}?download=true", path)
        error = validation_error(path, expected)
        if error is not None:
            failures.append(f"{filename}: downloaded file failed validation: {error}")
        else:
            print(f"OK {filename}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Verified {len(FILES)} exact PrimeVul split files in {args.output_dir}")


if __name__ == "__main__":
    main()

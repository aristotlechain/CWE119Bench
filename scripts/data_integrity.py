#!/usr/bin/env python3
"""Create or verify the canonical raw and generated-data integrity manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/data_integrity_manifest.csv"
SCOPES = ("raw", "clean", "noise")
EXPECTED_COUNTS = {"raw": 3, "clean": 110, "noise": 525}
RAW_FILES = (
    "data/raw/primevul_main/primevul_train.jsonl",
    "data/raw/primevul_main/primevul_valid.jsonl",
    "data/raw/primevul_main/primevul_test.jsonl",
)
GENERATED_DIRS = {
    "clean": "data/processed/cwe_119_imbalance",
    "noise": "data/processed/cwe_119_noise_ratio_preserved",
}
FIELDNAMES = (
    "scope",
    "relative_path",
    "normalization",
    "content_bytes",
    "sha256",
)


def exact_bytes(path: Path) -> bytes:
    return path.read_bytes()


def normalized_metadata_bytes(path: Path) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("generated_at_utc", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content(path: Path, normalization: str) -> bytes:
    if normalization == "exact":
        return exact_bytes(path)
    if normalization == "json_without_generated_at_utc":
        return normalized_metadata_bytes(path)
    raise ValueError(f"Unsupported normalization: {normalization}")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_paths(root: Path, scopes: Iterable[str]) -> list[tuple[str, Path, str]]:
    selected: list[tuple[str, Path, str]] = []
    for scope in scopes:
        if scope == "raw":
            selected.extend((scope, root / relative, "exact") for relative in RAW_FILES)
            continue

        generated_root = root / GENERATED_DIRS[scope]
        if not generated_root.is_dir():
            raise FileNotFoundError(generated_root)
        for path in sorted(generated_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".jsonl":
                normalization = "exact"
            elif path.name == "metadata.json":
                normalization = "json_without_generated_at_utc"
            else:
                raise ValueError(f"Unexpected generated-data file: {path}")
            selected.append((scope, path, normalization))
    return selected


def create_manifest(root: Path, manifest: Path, overwrite: bool) -> None:
    if manifest.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite canonical manifest: {manifest}. Pass --overwrite"
        )

    rows: list[dict[str, str | int]] = []
    counts = {scope: 0 for scope in SCOPES}
    for scope, path, normalization in canonical_paths(root, SCOPES):
        payload = content(path, normalization)
        rows.append(
            {
                "scope": scope,
                "relative_path": path.relative_to(root).as_posix(),
                "normalization": normalization,
                "content_bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
        counts[scope] += 1

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected file counts: {counts}. Expected {EXPECTED_COUNTS}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} canonical entries to {manifest}")


def read_manifest(manifest: Path, scopes: set[str]) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            raise ValueError(
                f"Unexpected manifest columns: {reader.fieldnames}. Expected {FIELDNAMES}"
            )
        return [row for row in reader if row["scope"] in scopes]


def verify_manifest(root: Path, manifest: Path, scopes: set[str]) -> None:
    rows = read_manifest(manifest, scopes)
    expected_rows = sum(EXPECTED_COUNTS[scope] for scope in scopes)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Manifest contains {len(rows)} selected entries. Expected {expected_rows}"
        )

    failures: list[str] = []
    counts = {scope: 0 for scope in scopes}
    for row in rows:
        scope = row["scope"]
        counts[scope] += 1
        path = root / row["relative_path"]
        if not path.is_file():
            failures.append(f"missing: {row['relative_path']}")
            continue
        try:
            payload = content(path, row["normalization"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable: {row['relative_path']}: {exc}")
            continue
        if len(payload) != int(row["content_bytes"]):
            failures.append(
                f"size mismatch: {row['relative_path']}: "
                f"{len(payload)} != {row['content_bytes']}"
            )
            continue
        actual_sha = sha256(payload)
        if actual_sha != row["sha256"]:
            failures.append(
                f"hash mismatch: {row['relative_path']}: "
                f"{actual_sha} != {row['sha256']}"
            )

    if failures:
        for failure in failures[:25]:
            print(f"ERROR {failure}", file=sys.stderr)
        if len(failures) > 25:
            print(f"ERROR ... {len(failures) - 25} more failures", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"Verified {len(rows)} canonical entries under {root} "
        f"for scopes {', '.join(sorted(scopes))}: {counts}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--scope",
        nargs="+",
        choices=(*SCOPES, "all"),
        default=["all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    if args.command == "create":
        create_manifest(root, manifest, args.overwrite)
        return

    requested = set(args.scope)
    scopes = set(SCOPES) if "all" in requested else requested
    verify_manifest(root, manifest, scopes)


if __name__ == "__main__":
    main()

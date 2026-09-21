#!/usr/bin/env python3
"""Fetch and hash only the pinned public PyTorch CodeBERT snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "microsoft/codebert-base"
REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
FILES = (
    "config.json", "pytorch_model.bin", "vocab.json", "merges.txt",
    "tokenizer_config.json", "special_tokens_map.json", "README.md",
)


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/codebert.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["model_name"] != MODEL or config["model_revision"] != REVISION:
        raise SystemExit("Model/revision differs from the pinned study snapshot.")
    cache = (PROJECT_ROOT / config["model_cache_dir"]).resolve()
    expected = (PROJECT_ROOT / config["local_model_path"]).resolve()
    from huggingface_hub import HfApi, snapshot_download

    remote = None
    if not args.verify_only:
        remote = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
        if remote.sha != REVISION:
            raise SystemExit("Remote model identity does not match the pinned commit.")
    path = Path(snapshot_download(
        MODEL, revision=REVISION, cache_dir=cache, allow_patterns=list(FILES),
        token=False, local_files_only=args.verify_only,
    )).absolute()
    if path != expected:
        raise SystemExit(f"Unexpected snapshot path: {path} != {expected}")
    remote_files = {file.rfilename: file for file in remote.siblings} if remote else {}
    records = []
    for name in FILES:
        file = path / name
        if not file.is_file():
            raise SystemExit(f"Missing expected snapshot artifact: {name}")
        digest = file_sha256(file)
        metadata = remote_files.get(name)
        if metadata:
            if metadata.size != file.stat().st_size:
                raise SystemExit(f"Remote artifact size mismatch: {name}")
            lfs = getattr(metadata, "lfs", None)
            if lfs and lfs.sha256 != digest:
                raise SystemExit(f"Remote LFS SHA256 mismatch: {name}")
        records.append({"name": name, "bytes": file.stat().st_size, "sha256": digest})
    receipt = cache.parent / "pinned_model_receipt.json"
    if args.verify_only:
        if not receipt.is_file():
            raise SystemExit("Missing download/remote-verification receipt.")
        saved = json.loads(receipt.read_text())
        if (saved["model_name"] != MODEL or saved["model_revision"] != REVISION
                or saved["files"] != records
                or resolve_path(saved.get("snapshot_path", "")) != path):
            raise SystemExit("Cached snapshot differs from the verified download receipt.")
    else:
        receipt.write_text(json.dumps({
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL, "model_revision": REVISION,
            "remote_revision_match": True,
            "pytorch_weights_verified_against_remote_lfs_sha256": True,
            "snapshot_path": portable_path(path), "files": records,
        }, indent=2) + "\n")
    print(json.dumps({"model_name": MODEL, "model_revision": REVISION,
                      "snapshot_path": portable_path(path), "files": len(records),
                      "bytes": sum(row["bytes"] for row in records),
                      "receipt": str(receipt), "verified": True}, indent=2))


if __name__ == "__main__":
    main()

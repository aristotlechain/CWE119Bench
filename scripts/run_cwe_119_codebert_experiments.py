#!/usr/bin/env python3
"""Fine-tune pinned local CodeBERT on the preserved CWE-119 experiment matrix.

Test data are inaccessible to this execution path unless --evaluate-tests is
explicitly supplied. Existing classical-model scripts and outputs are reused as
read-only definitions of metrics and prediction identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from run_cwe_119_model_experiments import (
    CONDITIONS, DEFAULT_TIE_BREAK_METRICS, SEEDS, TEST_RATIOS,
    TRAIN_VALID_RATIOS, compute_metrics, load_jsonl, ratio_dir_name,
    selection_metric_key,
)
from rerun_cwe_119_selected_true_pr_auc import (
    PREDICTION_FIELDS, sha256_file, stable_row_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
MODEL_NAME = "codebert_class_weight_balanced"
SCORE_FIELDS = [*PREDICTION_FIELDS, "selected_epoch", "learning_rate"]
PATH_FIELDS = (
    "local_model_path", "model_cache_dir", "token_cache_dir", "noise_dir", "test_dir",
)
MODEL_FILES = {"config.json", "pytorch_model.bin", "vocab.json", "merges.txt",
               "tokenizer_config.json", "special_tokens_map.json", "README.md"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def portable_command() -> list[str]:
    values = [Path(sys.executable).name, *sys.argv]
    command: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_absolute():
            try:
                value = str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
            except ValueError:
                value = path.name
        command.append(value)
    return command


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(float(value), ".17g")
                             if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def labels_from_rows(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    # Do not fall back from noisy labels to reference labels.
    labels = np.asarray([row[field] for row in rows])
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{field} must contain nonempty binary labels")
    return labels.astype(np.int64)


def balanced_weights(labels: np.ndarray) -> tuple[list[int], list[float]]:
    counts = np.bincount(labels, minlength=2)
    if not len(labels) or np.any(counts == 0):
        raise ValueError("Both observed learner-label classes are required")
    return counts.tolist(), (len(labels) / (2 * counts)).tolist()


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(scores) != len(labels) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid positive-class probabilities")
    return compute_metrics(labels, (scores >= 0.5).astype(np.int64), scores)


def selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    return selection_metric_key(metrics, "mcc", DEFAULT_TIE_BREAK_METRICS)


def validate_config(config: dict[str, Any]) -> None:
    required = {*PATH_FIELDS, "schema_version", "model_name", "model_revision", "epochs",
                "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay",
                "warmup_ratio", "max_grad_norm", "max_length", "model_seed", "dataset_seeds",
                "train_ratios", "test_ratios", "conditions", "mixed_precision", "num_workers",
                "torch_threads", "deterministic", "decision_threshold"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if config["schema_version"] != 1 or config["model_name"] != "microsoft/codebert-base" or config["model_revision"] != MODEL_REVISION:
        raise ValueError("Unsupported model or schema: use the pinned study model")
    for key in ("epochs", "train_batch_size", "eval_batch_size", "torch_threads"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_length"] != 512 or config["model_seed"] != 0 or config["decision_threshold"] != 0.5:
        raise ValueError("Approved token limit, model seed, and threshold are 512, 0, and 0.5")
    if config["mixed_precision"] != "bf16" or config["num_workers"] != 0 or config["deterministic"] is not True:
        raise ValueError("Approved execution uses deterministic BF16 and zero loader workers")
    if not 0 <= config["warmup_ratio"] <= 1 or config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    for key, allowed in (("train_ratios", TRAIN_VALID_RATIOS), ("test_ratios", TEST_RATIOS),
                         ("dataset_seeds", SEEDS), ("conditions", CONDITIONS)):
        if not config[key] or len(config[key]) != len(set(config[key])) or not set(config[key]) <= set(allowed):
            raise ValueError(f"Invalid or duplicated {key}")


def code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name) for name in (
        Path(__file__).name, "run_cwe_119_model_experiments.py", "rerun_cwe_119_selected_true_pr_auc.py")}


def verify_pinned_artifacts(config: dict[str, Any]) -> dict[str, str]:
    """Bind actual bytes to the independently downloaded pinned-model receipt."""
    snapshot = resolve_path(config["local_model_path"])
    receipt_path = resolve_path(config["model_cache_dir"]).parent / "pinned_model_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("model_name") != config["model_name"] or
            receipt.get("model_revision") != MODEL_REVISION or
            receipt.get("remote_revision_match") is not True or
            receipt.get("pytorch_weights_verified_against_remote_lfs_sha256") is not True or
            resolve_path(receipt.get("snapshot_path", "")) != snapshot or
            snapshot.name != MODEL_REVISION):
        raise ValueError("Pinned model receipt does not establish the study snapshot")
    files = receipt.get("files", [])
    if len(files) != len(MODEL_FILES) or {item.get("name") for item in files} != MODEL_FILES:
        raise ValueError("Pinned model receipt has an unexpected artifact domain")
    # Extra loader-preferred files could bypass the verified .bin or tokenizer.
    if {path.name for path in snapshot.iterdir() if path.is_file()} != MODEL_FILES:
        raise ValueError("Pinned snapshot contains missing or unverified loader artifacts")
    observed = {}
    for item in files:
        path = snapshot / item["name"]
        digest = sha256_file(path)
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise ValueError(f"Pinned model/tokenizer artifact checksum mismatch: {path}")
        observed[item["name"]] = digest
    return observed


def checked_outputs(group_dir: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("output_sha256", {})
    required = {"metrics.csv", "prediction_scores.csv", "validation_history.csv",
                "selected_model/config.json", "selected_model/model.safetensors"}
    if not required <= outputs.keys():
        raise ValueError(f"Incomplete saved output hashes: {group_dir}")
    for name, expected in outputs.items():
        path = group_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Saved output checksum mismatch: {path}")


class TokenCache:
    """SQLite text-only cache. Learner/reference labels never enter the cache."""
    def __init__(self, directory: Path, tokenizer: Any, config: dict[str, Any]):
        self.tokenizer = tokenizer
        self.settings = {"model_name": config["model_name"], "revision": config["model_revision"],
                         "max_length": config["max_length"], "truncation_side": "right",
                         "add_special_tokens": True, "input_field": "func", "schema_version": 1,
                         "tokenizer_artifact_sha256": {name: digest for name, digest in config.get("model_artifact_sha256", {}).items()
                                                       if name not in {"pytorch_model.bin", "README.md"}}}
        directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(directory / (content_hash(self.settings) + ".sqlite3"))
        self.connection.execute("CREATE TABLE IF NOT EXISTS texts (sha TEXT PRIMARY KEY, ids TEXT NOT NULL, original_length INTEGER NOT NULL)")

    def encode(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
        texts = [row.get("func") or "" for row in rows]
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Function inputs must be strings")
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        records: dict[str, tuple[list[int], int]] = {}
        unknown: dict[str, str] = {}
        for text_sha, text in zip(hashes, texts):
            if text_sha in records or text_sha in unknown:
                continue
            entry = self.connection.execute("SELECT ids, original_length FROM texts WHERE sha=?", (text_sha,)).fetchone()
            if entry is None:
                unknown[text_sha] = text
            else:
                records[text_sha] = (json.loads(entry[0]), int(entry[1]))
        pending = list(unknown.items())
        for offset in range(0, len(pending), 256):
            chunk = pending[offset:offset + 256]
            chunk_texts = [text for _, text in chunk]
            full = self.tokenizer(chunk_texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
            bounded = self.tokenizer(chunk_texts, add_special_tokens=True, truncation=True,
                                     max_length=self.settings["max_length"], padding=False)["input_ids"]
            for (text_sha, _), original_ids, ids in zip(chunk, full, bounded):
                if len(ids) > self.settings["max_length"]:
                    raise ValueError("Tokenizer exceeded the token budget")
                records[text_sha] = (ids, len(original_ids))
                self.connection.execute("INSERT INTO texts VALUES (?, ?, ?)",
                                        (text_sha, json.dumps(ids, separators=(",", ":")), len(original_ids)))
            self.connection.commit()
        encoded = [{"input_ids": records[key][0], "attention_mask": [1] * len(records[key][0])} for key in hashes]
        lengths = [records[key][1] for key in hashes]
        truncated = sum(length > self.settings["max_length"] for length in lengths)
        return encoded, {"n_rows": len(rows), "truncated_rows": truncated,
                         "truncation_rate": truncated / len(rows) if rows else 0.0,
                         "max_original_tokens": max(lengths, default=0),
                         "mean_original_tokens": float(np.mean(lengths)) if lengths else 0.0,
                         "tokenizer_settings_sha256": content_hash(self.settings)}

    def close(self) -> None:
        self.connection.close()


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(torch: Any, loader_generator: Any) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader_generator": loader_generator.get_state()}


def restore_rng(torch: Any, loader_generator: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    loader_generator.set_state(state["loader_generator"].cpu())


def make_loader(torch: Any, encoded: list[dict[str, list[int]]], labels: np.ndarray,
                tokenizer: Any, batch_size: int, generator: Any, shuffle: bool) -> Any:
    from transformers import DataCollatorWithPadding
    features = [{**encoding, "labels": int(label)} for encoding, label in zip(encoded, labels)]
    return torch.utils.data.DataLoader(features, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, pin_memory=True, generator=generator,
                                       collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))


def score_model(torch: Any, model: Any, loader: Any) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to("cuda", non_blocking=True) for key, value in batch.items() if key != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Expected two classification logits")
            chunks.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0)


def load_model(torch: Any, config: dict[str, Any], path: Path, *, pretrained: bool) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification
    seed_everything(torch, config["model_seed"])
    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(path), num_labels=2, local_files_only=True, trust_remote_code=False,
        output_loading_info=True,
    )
    missing_encoder = [key for key in info["missing_keys"] if not key.startswith("classifier.")]
    if missing_encoder or info["mismatched_keys"] or info.get("error_msgs"):
        raise ValueError(f"Pretrained encoder did not load intact: {info}")
    if not pretrained and info["missing_keys"]:
        raise ValueError(f"Selected checkpoint has missing weights: {info}")
    new_head_keys = {"classifier.dense.bias", "classifier.dense.weight",
                     "classifier.out_proj.bias", "classifier.out_proj.weight"}
    if pretrained and set(info["missing_keys"]) != new_head_keys:
        raise ValueError(f"Expected an entirely new classification head: {info}")
    allowed_unused = {"pooler.dense.bias", "pooler.dense.weight",
                      "roberta.pooler.dense.bias", "roberta.pooler.dense.weight"}
    if pretrained and not set(info["unexpected_keys"]) <= allowed_unused:
        raise ValueError(f"Unexpected checkpoint tensors: {info}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    info["classifier_initialization_sha256"] = hashlib.sha256(b"".join(
        parameter.detach().cpu().numpy().tobytes() for parameter in model.classifier.parameters())).hexdigest()
    return model.to("cuda"), info


def save_selected_model(model: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".atomic_model"
    temporary.mkdir(exist_ok=True)
    model.save_pretrained(temporary, safe_serialization=True)
    for path in temporary.iterdir():
        os.replace(path, directory / path.name)
    temporary.rmdir()


def output_hashes(directory: Path) -> dict[str, str]:
    names = ["metrics.csv", "prediction_scores.csv", "validation_history.csv", "selected_hyperparameters.csv"]
    names.extend(str(path.relative_to(directory)) for path in sorted((directory / "selected_model").glob("*")) if path.is_file())
    return {name: sha256_file(directory / name) for name in names if (directory / name).is_file()}


def meta_row(config: dict[str, Any], group: dict[str, Any], epoch: int,
             stage: str, eval_ratio: str) -> dict[str, Any]:
    return {"run_name": config["run_name"], "model": MODEL_NAME, "model_family": "codebert",
            **group, "stage": stage, "eval_ratio": eval_ratio, "split": "valid" if stage == "validation" else "test",
            "hyperparameter_name": "learning_rate", "hyperparameter_value": config["learning_rate"],
            "learning_rate": config["learning_rate"], "selected_epoch": epoch,
            "selected_hyperparams": True}


def prediction_rows(config: dict[str, Any], group: dict[str, Any], epoch: int,
                    stage: str, eval_ratio: str, rows: list[dict[str, Any]],
                    scores: np.ndarray) -> list[dict[str, Any]]:
    meta = meta_row(config, group, epoch, stage, eval_ratio)
    meta.pop("selected_hyperparams")
    result = []
    for row_number, (row, score) in enumerate(zip(rows, scores), start=1):
        item = {**meta, "row_number": row_number, "row_sha256": stable_row_sha256(row)}
        for name in ("idx", "func_hash", "commit_id", "cve", "file_hash", "source_primevul_file",
                     "benchmark_ratio", "benchmark_seed", "benchmark_row_role"):
            item[name] = row.get(name, "")
        item.update(binary_label=int(row["binary_label"]), predicted_label=int(score >= 0.5),
                    positive_class_score=float(score), score_type="positive_class_probability",
                    decision_threshold=0.5)
        result.append({name: item[name] for name in SCORE_FIELDS})
    return result


def environment_record(torch: Any) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "scikit-learn", "numpy", "pandas", "scipy", "lightgbm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "command": portable_command()}


def verify_inputs(manifest: dict[str, Any]) -> None:
    for name, expected in manifest["input_sha256"].items():
        path = resolve_path(manifest["input_paths"][name])
        if sha256_file(path) != expected:
            raise ValueError(f"Input checksum changed: {path}")


def evaluate_and_finish(torch: Any, config: dict[str, Any], group_dir: Path,
                        manifest: dict[str, Any], tokenizer: Any, cache: TokenCache,
                        evaluate_tests: bool) -> dict[str, Any]:
    group, epoch = manifest["group"], manifest["selected_epoch"]
    checkpoint_hashes = {str(path.relative_to(group_dir)): sha256_file(path)
                         for path in sorted((group_dir / "selected_model").glob("*")) if path.is_file()}
    if manifest.get("selected_checkpoint_sha256") not in (None, checkpoint_hashes):
        raise ValueError("Selected checkpoint changed before evaluation")
    manifest.update(status="evaluating", selected_checkpoint_sha256=checkpoint_hashes)
    atomic_json(group_dir / "manifest.json", manifest)
    selected, _ = load_model(torch, config, group_dir / "selected_model", pretrained=False)
    metrics, predictions = [], []
    evaluations = [("validation", group["train_ratio"], resolve_path(manifest["input_paths"]["valid"]))]
    if evaluate_tests:
        evaluations += [("test", ratio, resolve_path(config["test_dir"]) / ratio_dir_name(ratio) /
                         f"seed_{group['seed']}" / "test.jsonl") for ratio in config["test_ratios"]]
    for stage, ratio, path in evaluations:
        input_key = "valid" if stage == "validation" else "test:" + ratio
        observed_hash = sha256_file(path)
        previous_hash = manifest["input_sha256"].get(input_key)
        if previous_hash is not None and previous_hash != observed_hash:
            raise ValueError(f"Evaluation input checksum changed: {path}")
        manifest["input_paths"][input_key] = portable_path(path)
        manifest["input_sha256"][input_key] = observed_hash
        rows = load_jsonl(path)
        labels = labels_from_rows(rows, "binary_label")
        encoded, stats = cache.encode(rows)
        manifest["truncation_stats"]["validation" if stage == "validation" else input_key] = stats
        loader = make_loader(torch, encoded, labels, tokenizer, config["eval_batch_size"],
                             torch.Generator().manual_seed(0), False)
        scores = score_model(torch, selected, loader)
        if stage == "validation":
            expected_scores = np.asarray(manifest["selected_validation_scores"], dtype=np.float64)
            difference = float(np.max(np.abs(scores - expected_scores)))
            if difference > 1e-6:
                raise ValueError(f"Checkpoint reload changed validation probabilities by {difference}")
            manifest["checkpoint_reload_max_abs_score_difference"] = difference
            manifest["checkpoint_reload_tolerance"] = 1e-6
        metrics.append({**meta_row(config, group, epoch, stage, ratio),
                        "train_rows": manifest["train_rows"], "valid_rows": manifest["valid_rows"],
                        "elapsed_seconds": manifest["elapsed_seconds"], **probability_metrics(labels, scores)})
        predictions.extend(prediction_rows(config, group, epoch, stage, ratio, rows, scores))
    history = manifest["validation_history"]
    for row in history:
        row["selected"] = row["epoch"] == epoch
    atomic_csv(group_dir / "validation_history.csv", history)
    atomic_csv(group_dir / "metrics.csv", metrics)
    atomic_csv(group_dir / "prediction_scores.csv", predictions, SCORE_FIELDS)
    atomic_csv(group_dir / "selected_hyperparameters.csv", [{
        "run_name": config["run_name"], "model": MODEL_NAME, "model_family": "codebert", **group,
        "hyperparameter_name": "learning_rate", "hyperparameter_value": config["learning_rate"],
        "learning_rate": config["learning_rate"], "selected_epoch": epoch,
        "training_epochs_completed": manifest["training_epochs_completed"],
        "selection_metric": "mcc", "tie_break_metrics": "/".join(DEFAULT_TIE_BREAK_METRICS),
        "selection_key": json.dumps(selection_key(manifest["best_metrics"])),
    }])
    verify_inputs(manifest)
    manifest["input_sha256_after"] = dict(manifest["input_sha256"])
    manifest.update(status="complete", evaluation_status="tests_complete" if evaluate_tests else "validation_only",
                    completed_at=utc_now(), output_sha256=output_hashes(group_dir))
    atomic_json(group_dir / "manifest.json", manifest)
    del selected
    torch.cuda.empty_cache()
    return manifest


def run_group(torch: Any, config: dict[str, Any], output: Path, group: dict[str, Any],
              tokenizer: Any, cache: TokenCache, hashes: dict[str, Any],
              *, resume: bool, evaluate_tests: bool) -> dict[str, Any]:
    group_dir = output / "groups" / ratio_dir_name(group["train_ratio"]) / f"seed_{group['seed']}" / group["condition"]
    manifest_path = group_dir / "manifest.json"
    condition_path = resolve_path(config["noise_dir"]) / ratio_dir_name(group["train_ratio"]) / f"seed_{group['seed']}" / group["condition"]
    inputs = {name: condition_path / (name + ".jsonl") for name in ("train", "valid")}
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    group_provenance = {**hashes, "group": group, "training_input_sha256": input_hashes}
    provenance_hash = content_hash(group_provenance)
    active_path = output / "active_epoch_checkpoint.pt"
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Group already exists. Explicitly use --resume: {group_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provenance_sha256") != provenance_hash:
            raise ValueError(f"Configuration, source code, or inputs changed: {group_dir}")
        verify_inputs(manifest)
        if manifest["status"] == "evaluating":
            if manifest["training_epochs_completed"] != config["epochs"]:
                raise ValueError("Evaluation resume has an incomplete training history")
            return evaluate_and_finish(torch, config, group_dir, manifest, tokenizer, cache, evaluate_tests)
        if manifest["status"] == "complete":
            checked_outputs(group_dir, manifest)
            from validate_cwe_119_codebert_results import validate_group
            validate_group(group_dir, allow_partial=True)
            if not evaluate_tests or manifest["evaluation_status"] == "tests_complete":
                print(f"Verified completed group: {group}", flush=True)
                return manifest
            return evaluate_and_finish(torch, config, group_dir, manifest, tokenizer, cache, True)
    else:
        manifest = {"schema_version": 1, "status": "running", "evaluation_status": "not_evaluated",
                    "group": group, **hashes, "provenance_sha256": provenance_hash,
                    "effective_config": config, "protocol": config, "started_at": utc_now(),
                    "input_paths": {name: portable_path(path) for name, path in inputs.items()},
                    "input_sha256": input_hashes, "training_seed": config["model_seed"],
                    "target_epochs": config["epochs"], "training_epochs_completed": 0,
                    "validation_history": [], "truncation_stats": {}}
        group_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    start = perf_counter()
    train_rows, valid_rows = (load_jsonl(inputs[name]) for name in ("train", "valid"))
    train_labels, valid_labels = labels_from_rows(train_rows, "noisy_label"), labels_from_rows(valid_rows, "binary_label")
    counts, weights = balanced_weights(train_labels)
    train_encoded, train_stats = cache.encode(train_rows)
    valid_encoded, valid_stats = cache.encode(valid_rows)
    manifest.update(train_rows=len(train_rows), valid_rows=len(valid_rows), noisy_label_counts={"0": counts[0], "1": counts[1]},
                    class_weights=weights, truncation_stats={"train": train_stats, "validation": valid_stats})
    model, loading_info = load_model(torch, config, resolve_path(config["local_model_path"]), pretrained=True)
    manifest["pretrained_loading_info"] = loading_info
    train_generator = torch.Generator().manual_seed(config["model_seed"])
    train_loader = make_loader(torch, train_encoded, train_labels, tokenizer, config["train_batch_size"], train_generator, True)
    valid_loader = make_loader(torch, valid_encoded, valid_labels, tokenizer, config["eval_batch_size"], torch.Generator().manual_seed(0), False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device="cuda"))
    best_state, best_key, best_metrics, best_epoch, best_scores = None, None, None, None, None
    history, first_epoch, previous_elapsed = [], 1, 0.0
    if active_path.exists():
        state = torch.load(active_path, map_location="cpu", weights_only=False)
        if state["provenance_sha256"] != provenance_hash:
            # A completed previous group can leave its active checkpoint after a crash.
            previous_manifest = Path(state["group_dir"]) / "manifest.json"
            if not previous_manifest.exists() or json.loads(previous_manifest.read_text())["status"] != "complete":
                raise ValueError("Another unfinished group owns the active checkpoint. Resume it first")
        else:
            if not resume:
                raise ValueError("An epoch checkpoint exists. --resume is required")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            restore_rng(torch, train_generator, state["rng"])
            best_state, best_metrics, best_epoch = state["best_model"], state["best_metrics"], state["best_epoch"]
            best_scores = np.asarray(state["best_scores"], dtype=np.float64)
            best_key, history = selection_key(best_metrics), state["history"]
            first_epoch, previous_elapsed = state["epoch"] + 1, state["elapsed_seconds"]
            manifest["resume_from_epoch"] = state["epoch"]
            print(f"Resuming {group} after epoch {state['epoch']}", flush=True)
        del state
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, config["epochs"] + 1):
        epoch_start = perf_counter()
        model.train()
        train_losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to("cuda", non_blocking=True)
            inputs_gpu = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs_gpu).logits
                loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {group}, epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach()))
        scores = score_model(torch, model, valid_loader)
        metrics = probability_metrics(valid_labels, scores)
        key = selection_key(metrics)
        if best_key is None or key > best_key:
            best_key, best_metrics, best_epoch, best_scores = key, metrics, epoch, scores
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({**group, "epoch": epoch, "learning_rate": config["learning_rate"],
                        "mean_train_loss": float(np.mean(train_losses)), "epoch_elapsed_seconds": perf_counter() - epoch_start,
                        "selected": False, **metrics})
        elapsed = previous_elapsed + perf_counter() - start
        state = {"provenance_sha256": provenance_hash, "group_dir": str(group_dir), "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": rng_state(torch, train_generator), "best_model": best_state, "best_metrics": best_metrics,
                 "best_epoch": best_epoch, "best_scores": best_scores.tolist(), "history": history, "elapsed_seconds": elapsed}
        temporary = active_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, active_path)
        del state
        manifest.update(training_epochs_completed=epoch, elapsed_seconds=elapsed,
                        validation_history=history, selected_epoch=best_epoch, best_metrics=best_metrics,
                        selected_validation_scores=best_scores.tolist(), optimizer_total_steps=total_steps,
                        optimizer_warmup_steps=warmup_steps,
                        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
        atomic_json(manifest_path, manifest)
        print(f"{group} epoch {epoch}/{config['epochs']}: loss={np.mean(train_losses):.5f}, MCC={metrics['mcc']:.5f}, best={best_epoch}", flush=True)
    if best_state is None:
        raise ValueError("No epoch completed. Cannot export selected checkpoint")
    model.load_state_dict(best_state)
    model.config.codebert_selected_epoch = best_epoch
    model.config.codebert_learning_rate = config["learning_rate"]
    model.config.codebert_protocol_sha256 = manifest["protocol_sha256"]
    save_selected_model(model, group_dir / "selected_model")
    # Commit the complete train/validation selection record before reloading it.
    manifest.update(training_epochs_completed=config["epochs"], elapsed_seconds=previous_elapsed + perf_counter() - start,
                    validation_history=history, selected_epoch=best_epoch, best_metrics=best_metrics,
                    selected_validation_scores=best_scores.tolist(), optimizer_total_steps=total_steps,
                    optimizer_warmup_steps=warmup_steps,
                    selected_checkpoint={"epoch": best_epoch, "learning_rate": config["learning_rate"],
                                         "protocol_sha256": manifest["protocol_sha256"]})
    atomic_json(manifest_path, manifest)
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    manifest = evaluate_and_finish(torch, config, group_dir, manifest, tokenizer, cache, evaluate_tests)
    if active_path.exists():
        active_path.unlink()
    return manifest


def aggregate(output: Path, manifest: dict[str, Any]) -> None:
    metrics, predictions, selections, histories, completed = [], [], [], [], []
    for path in sorted((output / "groups").glob("ratio_*/seed_*/*/manifest.json")):
        group_manifest = json.loads(path.read_text(encoding="utf-8"))
        if group_manifest["status"] != "complete":
            continue
        if group_manifest["protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError(f"Aggregate refuses a mixed protocol: {path}")
        checked_outputs(path.parent, group_manifest)
        completed.append({**group_manifest["group"], "evaluation_status": group_manifest["evaluation_status"],
                          "manifest_path": str(path.relative_to(output)), "manifest_sha256": sha256_file(path)})
        metrics.extend(read_csv(path.parent / "metrics.csv"))
        predictions.extend(read_csv(path.parent / "prediction_scores.csv"))
        selections.extend(read_csv(path.parent / "selected_hyperparameters.csv"))
        histories.extend(read_csv(path.parent / "validation_history.csv"))
    if metrics:
        atomic_csv(output / "metrics.csv", metrics)
        atomic_csv(output / "prediction_scores.csv", predictions, SCORE_FIELDS)
        atomic_csv(output / "selected_hyperparameters.csv", selections)
        atomic_csv(output / "validation_history.csv", histories)
    manifest.update(completed_groups=completed, completed_group_count=len(completed),
                    metric_row_count=len(metrics), prediction_row_count=len(predictions), updated_at=utc_now())
    manifest["output_sha256"] = {name: sha256_file(output / name) for name in
                                 ("metrics.csv", "prediction_scores.csv", "selected_hyperparameters.csv", "validation_history.csv") if (output / name).exists()}
    atomic_json(output / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+")
    parser.add_argument("--dataset-seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-tests", action="store_true", help="Explicitly enable test-set access after checkpoint selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    for argument, key in ((args.ratios, "train_ratios"), (args.dataset_seeds, "dataset_seeds"), (args.conditions, "conditions")):
        if argument is not None:
            config[key] = argument
    output = resolve_path(args.output_dir)
    config["run_name"] = output.name
    validate_config(config)
    config["model_artifact_sha256"] = verify_pinned_artifacts(config)
    # Run domains belong to provenance. The explicit test-access switch can be
    # added to an unchanged validation-only run without restarting training.
    hashes = {"config_sha256": content_hash(config), "protocol_sha256": content_hash(config),
              "code_sha256": code_hashes(), "model_artifact_sha256": config["model_artifact_sha256"]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output exists. --resume is required")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in hashes.items()):
            raise ValueError("Cannot resume after changes to configuration, source code, or run domains")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory must be empty for a new run")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "status": "running", "started_at": utc_now(), **hashes,
                    "config_path": portable_path(config_path), "effective_config": config, "protocol": config,
                    "declared_domains": {"train_ratios": config["train_ratios"], "seeds": config["dataset_seeds"],
                                         "conditions": config["conditions"], "test_ratios": config["test_ratios"]},
                    "matrix": {"train_ratios": config["train_ratios"], "seeds": config["dataset_seeds"],
                               "conditions": config["conditions"], "test_ratios": config["test_ratios"]},
                    "selection_metrics": ["mcc", *DEFAULT_TIE_BREAK_METRICS], "completed_groups": []}
        atomic_json(manifest_path, manifest)
        atomic_json(output / "effective_config.json", config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Approved run requires a CUDA GPU with BF16 support")
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local_model = resolve_path(config["local_model_path"])
    if local_model.name != MODEL_REVISION:
        raise ValueError("Local model directory must identify the pinned snapshot SHA")
    tokenizer = AutoTokenizer.from_pretrained(str(local_model), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    manifest.update(environment=environment_record(torch), status="running",
                    evaluate_tests=args.evaluate_tests,
                    evaluation_scope="tests" if args.evaluate_tests else "validation_only",
                    diagnostic=config["epochs"] != 5)
    atomic_json(manifest_path, manifest)
    cache = TokenCache(resolve_path(config["token_cache_dir"]), tokenizer, config)
    try:
        for ratio in config["train_ratios"]:
            for seed in config["dataset_seeds"]:
                for condition in config["conditions"]:
                    group = {"train_ratio": ratio, "seed": seed, "condition": condition}
                    run_group(torch, config, output, group, tokenizer, cache, hashes,
                              resume=args.resume, evaluate_tests=args.evaluate_tests)
                    aggregate(output, manifest)
        manifest["status"] = "complete"
        aggregate(output, manifest)
    except BaseException as error:
        manifest.update(status="partial", interrupted_at=utc_now(), error=f"{type(error).__name__}: {error}")
        aggregate(output, manifest)
        raise
    finally:
        cache.close()


if __name__ == "__main__":
    main()

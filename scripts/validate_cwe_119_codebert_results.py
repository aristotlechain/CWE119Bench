#!/usr/bin/env python3
"""Independently audit CodeBERT scores, selected epochs, provenance, and aggregates.

This module deliberately imports neither the neural runner nor its metric helpers.
It reconstructs outcomes from exported probabilities and canonical JSONL records.
Partial/pilot validation can never certify the full 175-group paper experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
MODEL_FILES = {"config.json", "pytorch_model.bin", "vocab.json", "merges.txt",
               "tokenizer_config.json", "special_tokens_map.json", "README.md"}
TRAIN_RATIOS = ("60/40", "70/30", "75/25", "80/20", "90/10")
TEST_RATIOS = ("50/50", *TRAIN_RATIOS)
SEEDS = (1, 2, 3, 4, 5)
CONDITIONS = ("clean", "random_flip_10", "random_flip_20", "random_fp_10",
              "random_fp_20", "heuristic_fp_10", "heuristic_fp_20")
SELECTION_METRICS = ("mcc", "f1", "recall", "precision", "balanced_accuracy")
INTEGER_METRICS = ("n_rows", "positive_rows", "benign_rows", "tn", "fp", "fn", "tp")
FLOAT_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                 "mcc", "roc_auc", "average_precision", "pr_auc")
IDENTITY_FIELDS = ("idx", "func_hash", "commit_id", "cve", "file_hash",
                   "source_primevul_file", "benchmark_ratio", "benchmark_seed",
                   "benchmark_row_role")
TOP_OUTPUTS = ("metrics.csv", "selected_hyperparameters.csv", "prediction_scores.csv",
               "validation_history.csv", "manifest.json")
METRIC_TOLERANCE = 1e-10


class ValidationError(ValueError):
    """An artifact does not satisfy its declared scientific contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def integer(value: Any, context: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context}: expected integer, got {value!r}") from exc
    require(math.isfinite(number) and number.is_integer(), f"{context}: invalid integer")
    return int(number)


def number(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context}: expected number, got {value!r}") from exc
    require(math.isfinite(result), f"{context}: non-finite number")
    return result


def close(observed: Any, expected: float, context: str, tolerance: float = METRIC_TOLERANCE) -> None:
    actual = number(observed, context)
    require(abs(actual - expected) <= tolerance, f"{context}: {actual} differs from {expected}")


def sha256(path: Path, cache: dict[Path, str] | None = None) -> str:
    path = path.resolve()
    if cache is not None and path in cache:
        return cache[path]
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    if cache is not None:
        cache[path] = result
    return result


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def stable_row_sha256(row: dict[str, Any]) -> str:
    identity = [row.get(field) for field in
                ("source_primevul_file", "idx", "func_hash", "commit_id", "cve", "file_hash")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"{path}: absent or duplicate column names")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"{path}: malformed CSV row")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing canonical input: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(rows and all(isinstance(row, dict) for row in rows), f"{path}: empty/invalid JSONL")
    return rows


def reconstruct_metrics(labels: Any, probabilities: Any) -> dict[str, int | float]:
    y_true = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    require(y_true.ndim == scores.ndim == 1 and len(y_true) == len(scores) and len(scores) > 0,
            "Metric reconstruction: empty or mismatched arrays")
    require(set(np.unique(y_true)) == {0, 1}, "Metric reconstruction requires both truth classes")
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            "Probabilities must be finite and in [0,1]")
    prediction = (scores >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true, scores, pos_label=1, drop_intermediate=False)
    return {
        "n_rows": len(y_true), "positive_rows": int(y_true.sum()),
        "benign_rows": int((y_true == 0).sum()), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp), "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
    }


def metric_fixture() -> dict[str, float]:
    metrics = reconstruct_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    difference = abs(metrics["pr_auc"] - metrics["average_precision"])
    require(difference > 1e-3, "AP-versus-PR-AUC fixture unexpectedly aliases metrics")
    return {"average_precision": metrics["average_precision"], "pr_auc": metrics["pr_auc"],
            "absolute_difference": difference}


def check_metrics(row: dict[str, Any], reconstructed: dict[str, Any], context: str) -> None:
    for metric in INTEGER_METRICS:
        require(metric in row and integer(row[metric], f"{context}.{metric}") == reconstructed[metric],
                f"{context}: inconsistent {metric}")
    for metric in FLOAT_METRICS:
        require(metric in row, f"{context}: missing {metric}")
        close(row[metric], reconstructed[metric], f"{context}.{metric}")


def group_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["train_ratio"]), integer(row["seed"], "seed"), str(row["condition"])


def cell_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (*group_key(row), str(row["stage"]), str(row["eval_ratio"]))


def resolve_input_root(value: Any, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def protocol_value(protocol: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in protocol:
            return protocol[name]
        if name in protocol.get("training", {}):
            return protocol["training"][name]
    return default


def test_ratios(protocol: dict[str, Any]) -> tuple[str, ...]:
    values = protocol.get("test_ratios", protocol.get("matrix", {}).get("test_ratios", TEST_RATIOS))
    require(values and len(set(values)) == len(values) and set(values).issubset(TEST_RATIOS),
            "Invalid declared test-ratio domain")
    return tuple(values)


def selection_epoch(history: list[dict[str, Any]], target_epochs: int) -> int:
    epochs = [integer(row["epoch"], "validation_history.epoch") for row in history]
    require(len(epochs) == target_epochs and set(epochs) == set(range(1, target_epochs + 1)),
            "Validation history has missing or duplicate epochs")
    return integer(max(history, key=lambda row: (
        *(number(row[metric], f"history.{metric}") for metric in SELECTION_METRICS),
        -integer(row["epoch"], "history.epoch")))["epoch"], "selected epoch")


def check_hashes(manifest: dict[str, Any], group_dir: Path,
                 inputs: dict[str, Path], cache: dict[Path, str]) -> None:
    recorded_inputs = manifest.get("input_sha256", {})
    require(set(recorded_inputs) == set(inputs), "Input checksum domain does not match evaluated inputs")
    for name, path in inputs.items():
        require(recorded_inputs[name] == sha256(path, cache), f"Input checksum mismatch: {name}")
        if name in manifest.get("input_paths", {}):
            recorded = resolve_input_root(manifest["input_paths"][name], path)
            require(recorded == path.resolve(), f"Noncanonical input path: {name}")
    recorded_after = manifest.get("input_sha256_after", recorded_inputs)
    require(recorded_after == recorded_inputs, "Input checksums changed during training")
    outputs = manifest.get("output_sha256", {})
    required = {"metrics.csv", "prediction_scores.csv", "validation_history.csv", "selected_model/config.json"}
    require(required.issubset(outputs), "Manifest omits required output/checkpoint checksums")
    weights = [name for name in outputs if name.startswith("selected_model/") and
               (name.endswith(".safetensors") or name.endswith(".bin"))]
    require(weights, "Manifest omits selected model weights")
    for name, recorded in outputs.items():
        path = (group_dir / name).resolve()
        require(path.is_relative_to(group_dir.resolve()), f"Unsafe output checksum path: {name}")
        expected = recorded.get("sha256") if isinstance(recorded, dict) else recorded
        require(expected == sha256(path, cache), f"Output checksum mismatch: {name}")
    code = manifest.get("code_sha256", {})
    known = {"runner": "run_cwe_119_codebert_experiments.py",
             "frozen_metrics": "run_cwe_119_model_experiments.py",
             "frozen_scores": "rerun_cwe_119_selected_true_pr_auc.py"}
    require(code, "Manifest omits source-code checksums")
    for name, recorded in code.items():
        path_name = manifest.get("code_paths", {}).get(name, known.get(name, name))
        path = Path(path_name)
        if not path.is_absolute():
            path = PROJECT_ROOT / "scripts" / path
        require(recorded == sha256(path, cache), f"Source-code checksum mismatch: {name}")
    artifacts = manifest["protocol"].get("model_artifact_sha256")
    if manifest["protocol"].get("model_name") == "microsoft/codebert-base":
        require(isinstance(artifacts, dict) and set(artifacts) == MODEL_FILES,
                "Pinned CodeBERT artifact domain is incomplete or unexpected")
    if artifacts is not None:
        require(isinstance(artifacts, dict) and artifacts, "Empty model artifact binding")
        require(manifest.get("model_artifact_sha256") == artifacts, "Model artifact binding differs from protocol")
        local_model = resolve_input_root(manifest["protocol"].get("local_model_path"), PROJECT_ROOT / ".cache/codebert/model_cache")
        hub = resolve_input_root(manifest["protocol"].get("model_cache_dir"), PROJECT_ROOT / ".cache/codebert/model_cache/hub")
        for name, recorded in artifacts.items():
            require(isinstance(name, str) and name in MODEL_FILES and Path(name).name == name,
                    "Unsafe model artifact filename")
            logical_path = local_model / name
            require(logical_path.is_relative_to(local_model), "Unsafe model artifact path")
            path = logical_path.resolve()
            # Hugging Face snapshots legitimately symlink each named artifact
            # to ../../blobs inside the declared hub cache. Accept that layout
            # while rejecting symlinks outside both declared storage roots.
            require(path.is_relative_to(local_model) or path.is_relative_to(hub),
                    "Unsafe model artifact symlink target")
            require(recorded == sha256(path, cache), f"Pinned model artifact checksum mismatch: {name}")
        if manifest["protocol"].get("model_name") == "microsoft/codebert-base":
            require(local_model.name == MODEL_REVISION and manifest["protocol"].get("model_revision") == MODEL_REVISION,
                    "Pinned model snapshot/revision mismatch")
            require({path.name for path in local_model.iterdir() if path.is_file()} == MODEL_FILES,
                    "Pinned model directory contains unverified loader artifacts")
            receipt = read_json(hub.parent / "pinned_model_receipt.json")
            require(receipt.get("model_name") == "microsoft/codebert-base"
                    and receipt.get("model_revision") == MODEL_REVISION
                    and receipt.get("remote_revision_match") is True
                    and receipt.get("pytorch_weights_verified_against_remote_lfs_sha256") is True
                    and resolve_input_root(receipt.get("snapshot_path", ""), local_model) == local_model.resolve(),
                    "Pinned model receipt does not match verified protocol")
            receipt_files = receipt.get("files", [])
            require(len(receipt_files) == len(MODEL_FILES)
                    and {record.get("name") for record in receipt_files} == MODEL_FILES,
                    "Pinned model receipt artifact domain differs")
            for record in receipt_files:
                require(record.get("sha256") == artifacts[record["name"]], "Model receipt checksum differs from protocol")
                size = record.get("size_bytes", record.get("bytes"))
                if size is not None:
                    require(integer(size, "model receipt size") == (local_model / record["name"]).stat().st_size,
                            "Model receipt byte count differs")


def validate_group(group_dir: Path, *, noise_dir: Path | None = None,
                   test_dir: Path | None = None, allow_partial: bool = True,
                   hash_cache: dict[Path, str] | None = None) -> dict[str, Any]:
    """Audit one completed group without requiring Torch or loading model weights."""
    group_dir = Path(group_dir).resolve()
    manifest = read_json(group_dir / "manifest.json")
    require(manifest.get("status") == "complete", f"Incomplete group: {group_dir}")
    protocol = manifest.get("protocol")
    require(isinstance(protocol, dict), "Group manifest omits protocol")
    require(manifest.get("protocol_sha256") == payload_sha256(protocol), "Protocol hash mismatch")
    require(manifest.get("config_sha256") == manifest["protocol_sha256"], "Effective configuration hash mismatch")
    if "effective_config" in manifest:
        require(manifest["effective_config"] == protocol, "Effective configuration differs from protocol")
    ratio, seed, condition = group_key(manifest["group"])
    require(ratio in TRAIN_RATIOS and seed in SEEDS and condition in CONDITIONS,
            "Group is outside the declared factor domains")
    require(group_dir.parts[-3:] == (f"ratio_{ratio.replace('/', '_')}", f"seed_{seed}", condition),
            "Group directory differs from declared factors")
    roots = protocol.get("inputs", {})
    noise_dir = Path(noise_dir) if noise_dir else resolve_input_root(
        protocol.get("noise_dir", roots.get("noise_dir")), PROJECT_ROOT / "data/processed/cwe_119_noise_ratio_preserved")
    test_dir = Path(test_dir) if test_dir else resolve_input_root(
        protocol.get("test_dir", roots.get("test_dir")), PROJECT_ROOT / "data/processed/cwe_119_imbalance")
    directory = f"ratio_{ratio.replace('/', '_')}/seed_{seed}/{condition}"
    inputs = {"train": noise_dir / directory / "train.jsonl",
              "valid": noise_dir / directory / "valid.jsonl"}
    evaluation_status = manifest.get("evaluation_status")
    require(evaluation_status in {"validation_only", "tests_complete"}, "Invalid evaluation status")
    ratios = test_ratios(protocol)
    if evaluation_status == "tests_complete":
        inputs.update({f"test:{value}": test_dir / f"ratio_{value.replace('/', '_')}/seed_{seed}/test.jsonl"
                       for value in ratios})
    else:
        require(allow_partial, "Validation-only group requires --allow-partial")
    cache = hash_cache if hash_cache is not None else {}
    check_hashes(manifest, group_dir, inputs, cache)
    train = read_jsonl(inputs["train"])
    noisy_counts = Counter(integer(row.get("noisy_label"), "train.noisy_label") for row in train)
    require(set(noisy_counts) == {0, 1}, "Training learner labels must contain both binary classes")
    require(manifest.get("noisy_label_counts") == {str(key): noisy_counts[key] for key in (0, 1)},
            "Training learner-label counts disagree with manifest")
    weights = manifest.get("class_weights", [])
    require(len(weights) == 2, "Missing binary class weights")
    for cls in (0, 1):
        close(weights[cls], len(train) / (2 * noisy_counts[cls]), f"class_weights[{cls}]", 1e-6)
    require(integer(manifest.get("training_seed"), "training_seed") == 0,
            "Training seed differs from the fixed study initialization seed")
    target_epochs = integer(manifest.get("target_epochs"), "target_epochs")
    protocol_epochs = integer(protocol_value(protocol, "max_epochs", "epochs", default=5), "protocol epochs")
    require(0 < target_epochs <= protocol_epochs, "Invalid target epoch budget")
    require(integer(manifest.get("training_epochs_completed"), "completed epochs") == target_epochs,
            "Group has not completed its declared epoch budget")
    if not allow_partial:
        require(target_epochs == protocol_epochs == 5, "Group did not complete the study's five epochs")
    history = read_csv(group_dir / "validation_history.csv")
    selected_epoch = selection_epoch(history, target_epochs)
    require(integer(manifest.get("selected_epoch"), "manifest selected epoch") == selected_epoch,
            "Selected epoch violates validation history/tie-breaks")
    learning_rate = number(protocol_value(protocol, "learning_rate", default=2e-5), "learning rate")
    checkpoint = manifest.get("selected_checkpoint", {})
    model_config = read_json(group_dir / "selected_model/config.json")
    selected_metadata = checkpoint or {
        "epoch": model_config.get("codebert_selected_epoch"),
        "learning_rate": model_config.get("codebert_learning_rate"),
        "protocol_sha256": model_config.get("codebert_protocol_sha256"),
    }
    require(integer(selected_metadata.get("epoch"), "checkpoint epoch") == selected_epoch,
            "Checkpoint epoch metadata differs from selection")
    close(selected_metadata.get("learning_rate"), learning_rate, "checkpoint learning rate")
    require(selected_metadata.get("protocol_sha256") == manifest["protocol_sha256"],
            "Checkpoint protocol metadata differs")
    for field, expected in (("codebert_selected_epoch", selected_epoch),
                            ("codebert_learning_rate", learning_rate),
                            ("codebert_protocol_sha256", manifest["protocol_sha256"])):
        if field in model_config:
            require(model_config[field] == expected, f"Model configuration selection metadata differs: {field}")
    if "id2label" in model_config:
        require(set(model_config["id2label"]) == {"0", "1"}, "Selected model does not have two label indices")
    reload_tolerance = number(manifest.get("checkpoint_reload_tolerance", 1e-6), "reload tolerance")
    require(0 <= reload_tolerance <= 1e-6, "Checkpoint reload tolerance exceeds 1e-6")
    reload_difference = number(manifest.get("checkpoint_reload_max_abs_score_difference"), "reload difference")
    require(0 <= reload_difference <= reload_tolerance, "Checkpoint reload scores failed reproduction")

    canonical = {("validation", ratio): read_jsonl(inputs["valid"])}
    if evaluation_status == "tests_complete":
        canonical.update({("test", value): read_jsonl(inputs[f"test:{value}"]) for value in ratios})
    score_rows = read_csv(group_dir / "prediction_scores.csv")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in score_rows:
        require(group_key(row) == (ratio, seed, condition) and row.get("model_family") == "codebert",
                "Prediction group/model metadata mismatch")
        require(row.get("hyperparameter_name") == "learning_rate", "Incorrect neural hyperparameter name")
        close(row.get("hyperparameter_value"), learning_rate, "score hyperparameter value")
        key = row["stage"], row["eval_ratio"]
        require(key in canonical, f"Unexpected prediction cell: {key}")
        cells.setdefault(key, []).append(row)
    require(set(cells) == set(canonical), "Missing evaluation score cell")
    metric_rows = read_csv(group_dir / "metrics.csv")
    indexed_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        key = row["stage"], row["eval_ratio"]
        require(key in canonical and key not in indexed_metrics, "Missing, extra, or duplicate metric cell")
        require(group_key(row) == (ratio, seed, condition) and row.get("model_family") == "codebert",
                "Metric group/model metadata mismatch")
        if "hyperparameter_name" in row:
            require(row["hyperparameter_name"] == "learning_rate", "Incorrect metric hyperparameter name")
            close(row.get("hyperparameter_value"), learning_rate, "metric hyperparameter value")
        if "selected_hyperparams" in row:
            require(row["selected_hyperparams"].lower() == "true", "Metric does not describe selected checkpoint")
        indexed_metrics[key] = row
    require(set(indexed_metrics) == set(canonical), "Missing metric cell")
    counts = {"validation_cells": 0, "test_cells": 0, "validation_score_rows": 0, "test_score_rows": 0}
    for (stage, eval_ratio), records in canonical.items():
        exported = cells[(stage, eval_ratio)]
        require(len(exported) == len(records), f"{stage}/{eval_ratio}: score row count mismatch")
        labels, scores = [], []
        for ordinal, (row, source) in enumerate(zip(exported, records), 1):
            require(integer(row.get("row_number"), "row_number") == ordinal,
                    "Prediction row order/number is not canonical")
            require(row.get("row_sha256") == stable_row_sha256(source), "Prediction source row hash mismatch")
            for field in IDENTITY_FIELDS:
                require(row.get(field) == csv_value(source.get(field, "")), f"Prediction identity differs: {field}")
            truth = integer(source.get("binary_label"), "canonical binary_label")
            require(integer(row.get("binary_label"), "export binary_label") == truth,
                    "Prediction reference label differs from canonical JSONL")
            score = number(row.get("positive_class_score"), "positive_class_score")
            require(0 <= score <= 1, "Probability outside [0,1]")
            require(integer(row.get("predicted_label"), "predicted_label") == int(score >= 0.5),
                    "Predicted label differs from >=0.5 probability rule")
            require(row.get("score_type") == "positive_class_probability" and
                    number(row.get("decision_threshold"), "decision_threshold") == 0.5,
                    "Incorrect score type/threshold")
            require(row.get("split") == ("valid" if stage == "validation" else "test"), "Incorrect split")
            require(integer(row.get("selected_epoch"), "score selected_epoch") == selected_epoch,
                    "Prediction selected epoch differs")
            close(row.get("learning_rate"), learning_rate, "score learning rate")
            labels.append(truth)
            scores.append(score)
        reconstructed = reconstruct_metrics(labels, scores)
        metric = indexed_metrics[(stage, eval_ratio)]
        check_metrics(metric, reconstructed, f"{stage}/{eval_ratio}")
        require(integer(metric.get("selected_epoch"), "metric selected_epoch") == selected_epoch,
                "Metric selected epoch differs")
        close(metric.get("learning_rate"), learning_rate, "metric learning rate")
        require(metric.get("split") == ("valid" if stage == "validation" else "test"), "Metric split differs")
        if stage == "validation":
            selected_history = next(row for row in history if integer(row["epoch"], "history epoch") == selected_epoch)
            # Audit epoch-selection arithmetic against the original selected
            # probabilities. Checkpoint reload is a separate numerical check.
            before_reload = manifest.get("selected_validation_scores", scores)
            require(len(before_reload) == len(scores), "Selected validation score count differs")
            before_metrics = reconstruct_metrics(labels, before_reload)
            check_metrics(selected_history, before_metrics, "selected validation history")
            difference = float(np.max(np.abs(np.asarray(before_reload, dtype=np.float64) - scores)))
            close(reload_difference, difference, "recorded checkpoint reload difference")
            require(difference <= reload_tolerance, "Exported validation probabilities failed reload check")
        counts[f"{stage}_cells"] += 1
        counts[f"{stage}_score_rows"] += len(records)
    for row in history:
        require(integer(row.get("n_rows"), "history n_rows") == len(canonical[("validation", ratio)]),
                "Validation history row count differs from canonical data")
        truth_positive = sum(integer(record["binary_label"], "validation binary label")
                             for record in canonical[("validation", ratio)])
        require(integer(row.get("positive_rows"), "history positive_rows") == truth_positive
                and integer(row.get("benign_rows"), "history benign_rows") == len(canonical[("validation", ratio)]) - truth_positive,
                "Validation history reference-class counts differ")
        for metric in FLOAT_METRICS:
            number(row.get(metric), f"history.{metric}")
        if "selected" in row:
            require(row["selected"].lower() in {"true", "false"}, "Invalid history selected flag")
            require((row["selected"].lower() == "true") == (integer(row["epoch"], "history epoch") == selected_epoch),
                    "Validation history selected flag differs from selected epoch")
        if "learning_rate" in row:
            close(row["learning_rate"], learning_rate, "history learning rate")
        if "train_ratio" in row:
            require(group_key(row) == (ratio, seed, condition), "Validation history group metadata differs")
    return {"status": "full_group_passed" if evaluation_status == "tests_complete" and target_epochs == protocol_epochs == 5 else "partial_passed",
            "group": {"train_ratio": ratio, "seed": seed, "condition": condition},
            "selected_epoch": selected_epoch, "learning_rate": learning_rate,
            "evaluation_status": evaluation_status, "target_epochs": target_epochs,
            "protocol_sha256": manifest["protocol_sha256"], "code_sha256": manifest["code_sha256"], "counts": counts,
            "_metrics": metric_rows, "_predictions": score_rows, "_history": history}


def aggregate_key(row: dict[str, Any], kind: str) -> tuple[Any, ...]:
    if kind == "validation_history.csv":
        return (*group_key(row), integer(row["epoch"], "history epoch"))
    key = cell_key(row)
    return (*key, integer(row["row_number"], "row number")) if kind == "prediction_scores.csv" else key


def row_digest(row: dict[str, Any]) -> str:
    return payload_sha256(row)


def audit_aggregate(path: Path, expected: dict[tuple[Any, ...], str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    require(path.is_file(), f"Missing aggregate file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = aggregate_key(row, path.name)
            require(key not in seen and key in expected, f"{path.name}: duplicate or unexpected aggregate key")
            require(row_digest(row) == expected[key], f"{path.name}: aggregate row differs from audited group")
            seen.add(key)
    require(seen == set(expected), f"{path.name}: missing aggregate rows")


def declared_group_keys(manifest: dict[str, Any]) -> set[tuple[str, int, str]]:
    protocol = manifest.get("protocol", {})
    matrix = manifest.get("matrix", protocol.get("matrix", protocol))
    ratios = matrix.get("train_ratios", matrix.get("ratios", TRAIN_RATIOS))
    seeds = matrix.get("seeds", matrix.get("dataset_seeds", SEEDS))
    conditions = matrix.get("conditions", CONDITIONS)
    for values, allowed, name in ((ratios, TRAIN_RATIOS, "ratios"), (seeds, SEEDS, "seeds"),
                                  (conditions, CONDITIONS, "conditions")):
        require(values and len(values) == len(set(values)) and set(values).issubset(allowed),
                f"Invalid declared {name} domain")
    for field, values in (("train_ratios", ratios), ("dataset_seeds", seeds), ("conditions", conditions)):
        if field in protocol:
            require(set(protocol[field]) == set(values), f"Protocol/declaration mismatch: {field}")
    if "declared_domains" in manifest:
        require(manifest["declared_domains"] == matrix, "Declared domains differ from matrix")
    if "test_ratios" in matrix:
        require(tuple(matrix["test_ratios"]) == test_ratios(protocol), "Test-ratio declaration differs from protocol")
    return set(itertools.product(ratios, seeds, conditions))


def validate_results(results_dir: Path, *, allow_partial: bool = False,
                     noise_dir: Path | None = None, test_dir: Path | None = None) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    top = read_json(results_dir / "manifest.json")
    require(top.get("status") in {"partial", "running", "complete"}, "Invalid top-level run status")
    require(isinstance(top.get("protocol"), dict), "Run manifest omits protocol")
    require(top.get("protocol_sha256") == payload_sha256(top["protocol"]), "Run protocol checksum mismatch")
    require(top.get("config_sha256") == top["protocol_sha256"], "Run effective-configuration checksum mismatch")
    if "effective_config" in top:
        require(top["effective_config"] == top["protocol"], "Run effective configuration differs")
    effective_config_path = results_dir / "effective_config.json"
    if effective_config_path.exists():
        require(payload_sha256(read_json(effective_config_path)) == top["config_sha256"],
                "Saved effective configuration checksum mismatch")
    if "model_artifact_sha256" in top["protocol"]:
        require(top.get("model_artifact_sha256") == top["protocol"]["model_artifact_sha256"],
                "Run model artifact binding differs from protocol")
    intended = declared_group_keys(top)
    full_domain = set(itertools.product(TRAIN_RATIOS, SEEDS, CONDITIONS))
    cache: dict[Path, str] = {}
    audited: list[dict[str, Any]] = []
    unfinished = []
    expected = {name: {} for name in ("metrics.csv", "prediction_scores.csv", "validation_history.csv")}
    completed_manifests = {}
    for manifest_path in sorted((results_dir / "groups").glob("ratio_*/seed_*/*/manifest.json")):
        metadata = read_json(manifest_path)
        key = group_key(metadata["group"])
        require(key in intended, "Group outside declared matrix")
        require(not any(group_key(record["group"]) == key for record in audited), "Duplicate completed group")
        if metadata.get("status") != "complete":
            unfinished.append({"group": metadata["group"], "status": metadata.get("status")})
            require(allow_partial, "Run contains incomplete groups. Use --allow-partial")
            continue
        record = validate_group(manifest_path.parent, noise_dir=noise_dir, test_dir=test_dir,
                                allow_partial=allow_partial, hash_cache=cache)
        if top.get("protocol_sha256"):
            require(record["protocol_sha256"] == top["protocol_sha256"], "Group protocol differs from run protocol")
        if "code_sha256" in top:
            require(record["code_sha256"] == top["code_sha256"], "Group source code differs from run source code")
        completed_manifests[key] = (manifest_path, sha256(manifest_path, cache))
        for filename, private_key in (("metrics.csv", "_metrics"), ("prediction_scores.csv", "_predictions"),
                                      ("validation_history.csv", "_history")):
            for row in record[private_key]:
                row_key = aggregate_key(row, filename)
                require(row_key not in expected[filename], f"Duplicate group {filename} row")
                expected[filename][row_key] = row_digest(row)
            del record[private_key]
        audited.append(record)
    require(audited, "No completed groups to validate")
    completed = {group_key(record["group"]) for record in audited}
    missing = sorted(intended - completed)
    require(allow_partial or not missing, "Missing declared groups. Use --allow-partial for progress/pilots")
    for filename, rows in expected.items():
        audit_aggregate(results_dir / filename, rows)
    if "completed_groups" in top:
        recorded_complete = top["completed_groups"]
        require(isinstance(recorded_complete, list) and len(recorded_complete) == len(completed),
                "Completed-group manifest count differs from audited groups")
        seen_completed = set()
        for record in recorded_complete:
            key = group_key(record)
            require(key in completed_manifests and key not in seen_completed, "Invalid/duplicate completed-group record")
            path, digest = completed_manifests[key]
            require(record.get("manifest_path") == path.relative_to(results_dir).as_posix()
                    and record.get("manifest_sha256") == digest, "Completed-group manifest provenance differs")
            seen_completed.add(key)
    if "completed_group_count" in top:
        require(integer(top["completed_group_count"], "completed_group_count") == len(completed),
                "Declared completed group count differs")
    selected = read_csv(results_dir / "selected_hyperparameters.csv")
    selected_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in selected:
        key = group_key(row)
        require(key in completed and key not in selected_by_key, "Duplicate/extra selected configuration")
        require(row.get("model_family") == "codebert", "Selected configuration has wrong model family")
        selected_by_key[key] = row
    require(set(selected_by_key) == completed, "Missing selected configuration")
    for record in audited:
        row = selected_by_key[group_key(record["group"])]
        require(integer(row["selected_epoch"], "selected epoch") == record["selected_epoch"],
                "Aggregate selected epoch differs from audited group")
        close(row["learning_rate"], record["learning_rate"], "aggregate learning rate")
    checked = {filename: sha256(results_dir / filename, cache) for filename in TOP_OUTPUTS}
    outputs = top.get("output_sha256", {})
    require(set(TOP_OUTPUTS) - {"manifest.json"} == set(outputs), "Missing/extra aggregate output checksums")
    for name, recorded in outputs.items():
        require(name in checked and name != "manifest.json", "Unexpected/self-referential aggregate output hash")
        actual = recorded.get("sha256") if isinstance(recorded, dict) else recorded
        require(actual == checked[name], f"Aggregate checksum mismatch: {name}")
    counts = {key: sum(record["counts"][key] for record in audited)
              for key in ("validation_cells", "test_cells", "validation_score_rows", "test_score_rows")}
    if "metric_row_count" in top:
        require(integer(top["metric_row_count"], "metric_row_count") == counts["validation_cells"] + counts["test_cells"],
                "Aggregate metric count differs from audited cells")
    if "prediction_row_count" in top:
        require(integer(top["prediction_row_count"], "prediction_row_count") == counts["validation_score_rows"] + counts["test_score_rows"],
                "Aggregate score count differs from audited rows")
    full = (top["status"] == "complete" and intended == full_domain and not missing and not unfinished
            and all(record["status"] == "full_group_passed" for record in audited)
            and counts == {"validation_cells": 175, "test_cells": 1050,
                           "validation_score_rows": 16555, "test_score_rows": 98700})
    if full:
        protocol = top["protocol"]
        require(protocol.get("model_name") == "microsoft/codebert-base"
                and protocol.get("model_revision") == MODEL_REVISION
                and protocol.get("model_artifact_sha256"), "Full certification requires pinned CodeBERT artifact provenance")
        require(integer(protocol.get("max_length"), "max_length") == 512
                and integer(protocol.get("model_seed"), "model_seed") == 0
                and number(protocol.get("decision_threshold"), "protocol threshold") == 0.5,
                "Full certification requires the study token/seed/threshold settings")
        require({"run_cwe_119_codebert_experiments.py", "run_cwe_119_model_experiments.py",
                 "rerun_cwe_119_selected_true_pr_auc.py"}.issubset(top.get("code_sha256", {})),
                "Full certification requires all scientific source-code hashes")
    require(allow_partial or full, "Artifact is not the complete 175-group/five-epoch experiment")
    return {"status": "full_passed" if full else "partial_passed", "report_version": "codebert_validation_v1",
            "results_dir": str(results_dir), "completed_groups": len(audited), "declared_groups": len(intended),
            **counts, "counts": counts, "missing_groups": [list(key) for key in missing],
            "unfinished_groups": unfinished, "evaluation_scope": "full_matrix" if full else "partial_or_pilot",
            "metric_tolerance": METRIC_TOLERANCE, "checkpoint_reload_tolerance": 1e-6,
            "protocol_sha256": top["protocol_sha256"], "config_sha256": top["config_sha256"],
            "metric_fixture": metric_fixture(), "checked_file_sha256": checked, "groups": audited}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Audit complete pilot/progress groups without full certification")
    parser.add_argument("--noise-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Default: RESULTS_DIR/validation_report.json")
    args = parser.parse_args()
    output = args.output or args.results_dir / "validation_report.json"
    try:
        report = validate_results(args.results_dir, allow_partial=args.allow_partial,
                                  noise_dir=args.noise_dir, test_dir=args.test_dir)
    except (ValidationError, KeyError, OSError, ValueError, TypeError) as exc:
        report = {"status": "failed", "report_version": "codebert_validation_v1", "error": str(exc),
                  "results_dir": str(args.results_dir.resolve())}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"CodeBERT audit failed: {exc}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "completed_groups", "validation_cells", "test_cells")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independently validate the selected-model true-PR-AUC reruns.

This validator intentionally does not import either experiment runner. It
reconstructs all metrics from the retained per-example scores, checks every
score row against the corresponding benchmark JSONL row, compares metrics
unaffected by the PR-AUC correction with the frozen original runs, and emits
versioned combined result tables only after every validation gate passes.

The default profile is the complete four-model experiment:

* 700 selected model configurations.
* 700 validation metric cells and 66,220 validation score rows.
* 4,200 test metric cells and 394,800 test score rows.

Existing result and report files are read-only. The report directory must not
already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results/model_experiments"
DEFAULT_NOISE_DIR = PROJECT_ROOT / "data/processed/cwe_119_noise_ratio_preserved"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data/processed/cwe_119_imbalance"
REPORT_VERSION = "cwe119_true_pr_auc_validation_v1"

TRAIN_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
TEST_RATIOS = ["50/50", "60/40", "70/30", "75/25", "80/20", "90/10"]
SEEDS = [1, 2, 3, 4, 5]
CONDITIONS = [
    "clean",
    "random_flip_10",
    "random_flip_20",
    "random_fp_10",
    "random_fp_20",
    "heuristic_fp_10",
    "heuristic_fp_20",
]

RUNS = {
    "logistic_regression": {
        "source": "logreg_balanced_ratio_preserved_noise_mcc_selected_v1",
        "new": "logreg_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "hyperparameter": "C",
        "score_type": "positive_class_probability",
        "decision_threshold": ">=0.5",
    },
    "linear_svm": {
        "source": "linear_svm_balanced_ratio_preserved_noise_mcc_selected_v1",
        "new": "linear_svm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "hyperparameter": "C",
        "score_type": "decision_function",
        "decision_threshold": ">0 (model.predict used, exact zero is class 0)",
    },
    "random_forest": {
        "source": "random_forest_balanced_ratio_preserved_noise_mcc_selected_v1",
        "new": "random_forest_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "hyperparameter": "min_samples_leaf",
        "score_type": "positive_class_probability",
        "decision_threshold": ">=0.5",
    },
    "lightgbm": {
        "source": "lightgbm_balanced_ratio_preserved_noise_mcc_selected_v1",
        "new": "lightgbm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "hyperparameter": "num_leaves",
        "score_type": "positive_class_probability",
        "decision_threshold": ">=0.5",
    },
}

INTEGER_METRICS = [
    "n_rows",
    "positive_rows",
    "benign_rows",
    "tn",
    "fp",
    "fn",
    "tp",
]
FLOAT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
    "pr_auc",
]
OLD_REPRODUCTION_FLOAT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
]
IDENTITY_FIELDS = [
    "idx",
    "func_hash",
    "commit_id",
    "cve",
    "file_hash",
    "source_primevul_file",
    "benchmark_ratio",
    "benchmark_seed",
    "benchmark_row_role",
]
REQUIRED_SCORE_FIELDS = {
    "run_name",
    "model",
    "model_family",
    "train_ratio",
    "seed",
    "condition",
    "stage",
    "eval_ratio",
    "split",
    "hyperparameter_name",
    "hyperparameter_value",
    "row_number",
    "row_sha256",
    *IDENTITY_FIELDS,
    "binary_label",
    "predicted_label",
    "positive_class_score",
    "score_type",
    "decision_threshold",
}
REQUIRED_METRIC_FIELDS = {
    "run_name",
    "model",
    "model_family",
    "hyperparameter_name",
    "train_ratio",
    "seed",
    "condition",
    "stage",
    "eval_ratio",
    "split",
    "C",
    "selected_hyperparams",
    *INTEGER_METRICS,
    *FLOAT_METRICS,
}

EXPECTED_CONFIGS_PER_MODEL = 175
EXPECTED_VALIDATION_CELLS_PER_MODEL = 175
EXPECTED_TEST_CELLS_PER_MODEL = 1050
EXPECTED_VALIDATION_SCORES_PER_MODEL = 16_555
EXPECTED_TEST_SCORES_PER_MODEL = 98_700
EXPECTED_ALL_SCORES_PER_MODEL = 115_255

EXPECTED_CONFIGS = 700
EXPECTED_VALIDATION_CELLS = 700
EXPECTED_TEST_CELLS = 4_200
EXPECTED_VALIDATION_SCORES = 66_220
EXPECTED_TEST_SCORES = 394_800
EXPECTED_ALL_SCORES = 461_020


class ValidationFailure(RuntimeError):
    """Raised when any audit invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = sha256_file(resolved)
    return cache[resolved]


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing benchmark JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationFailure(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            require(isinstance(row, dict), f"Non-object JSON at {path}:{line_number}")
            rows.append(row)
    require(rows, f"Empty benchmark JSONL: {path}")
    return rows


def write_csv_exact(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def selected(value: str) -> bool:
    return value.strip().lower() == "true"


def as_int(value: str, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"Expected integer for {context}. Observed {value!r}") from exc


def as_float(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"Expected float for {context}. Observed {value!r}") from exc
    require(math.isfinite(result), f"Non-finite float for {context}: {value!r}")
    return result


def canonical_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value)


def stable_row_sha256(row: dict[str, Any]) -> str:
    material = json.dumps(
        [
            row.get("source_primevul_file"),
            row.get("idx"),
            row.get("func_hash"),
            row.get("commit_id"),
            row.get("cve"),
            row.get("file_hash"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def metric_fixture() -> dict[str, float]:
    """Prove that average precision and trapezoidal PR-AUC are not aliases."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    average_precision = float(average_precision_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score,
        pos_label=1,
        drop_intermediate=False,
    )
    pr_auc = float(auc(recall, precision))
    require(
        math.isclose(average_precision, 5.0 / 6.0, abs_tol=1e-15),
        f"AP fixture failed: {average_precision}",
    )
    require(
        math.isclose(pr_auc, 19.0 / 24.0, abs_tol=1e-15),
        f"Trapezoidal PR-AUC fixture failed: {pr_auc}",
    )
    require(
        not math.isclose(average_precision, pr_auc, abs_tol=1e-15),
        "Metric fixture failed because AP and trapezoidal PR-AUC are aliased.",
    )
    return {
        "average_precision": average_precision,
        "trapezoidal_pr_auc": pr_auc,
        "absolute_difference": abs(average_precision - pr_auc),
    }


def reconstruct_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, int | float]:
    require(len(y_true) > 0, "Cannot reconstruct metrics from an empty score group.")
    require(
        set(np.unique(y_true).tolist()) == {0, 1},
        "Every evaluated group must contain both binary classes.",
    )
    require(
        set(np.unique(y_pred).tolist()).issubset({0, 1}),
        "Predictions must be binary.",
    )
    require(np.isfinite(y_score).all(), "A score group contains non-finite values.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denominator == 0 else float((tp * tn - fp * fn) / denominator)
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        y_score,
        pos_label=1,
        drop_intermediate=False,
    )
    return {
        "n_rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "benign_rows": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": mcc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
    }


def metric_key(row: dict[str, str]) -> tuple[str, int, str, str, str]:
    return (
        row["train_ratio"],
        as_int(row["seed"], "metrics.seed"),
        row["condition"],
        row["stage"],
        row["eval_ratio"],
    )


def config_key(row: dict[str, str]) -> tuple[str, int, str]:
    return (
        row["train_ratio"],
        as_int(row["seed"], "configuration.seed"),
        row["condition"],
    )


def expected_config_keys() -> list[tuple[str, int, str]]:
    return [
        (ratio, seed, condition)
        for ratio in TRAIN_RATIOS
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def expected_metric_keys() -> tuple[
    list[tuple[str, int, str, str, str]],
    list[tuple[str, int, str, str, str]],
]:
    validation = [
        (ratio, seed, condition, "validation_selected_reproduction", ratio)
        for ratio, seed, condition in expected_config_keys()
    ]
    tests = [
        (ratio, seed, condition, "test", test_ratio)
        for ratio, seed, condition in expected_config_keys()
        for test_ratio in TEST_RATIOS
    ]
    return validation, tests


def ratio_dir_name(ratio: str) -> str:
    return "ratio_" + ratio.replace("/", "_")


def compare_reconstructed_metrics(
    stored: dict[str, str],
    reconstructed: dict[str, int | float],
    *,
    context: str,
    tolerance: float,
) -> float:
    maximum = 0.0
    for name in INTEGER_METRICS:
        observed = as_int(stored[name], f"{context}.{name}")
        expected = int(reconstructed[name])
        require(observed == expected, f"{context}: {name} is {observed}, expected {expected}")
    for name in FLOAT_METRICS:
        observed = as_float(stored[name], f"{context}.{name}")
        expected = float(reconstructed[name])
        difference = abs(observed - expected)
        maximum = max(maximum, difference)
        require(
            difference <= tolerance,
            f"{context}: {name} differs by {difference:.17g} (limit {tolerance:g})",
        )
    return maximum


def compare_old_reproduction(
    old: dict[str, str],
    new: dict[str, str],
    *,
    context: str,
    tolerance: float,
) -> float:
    maximum = 0.0
    for name in INTEGER_METRICS:
        old_value = as_int(old[name], f"{context}.old_{name}")
        new_value = as_int(new[name], f"{context}.new_{name}")
        require(old_value == new_value, f"{context}: reproduced {name} changed")
    for name in OLD_REPRODUCTION_FLOAT_METRICS:
        old_value = as_float(old[name], f"{context}.old_{name}")
        new_value = as_float(new[name], f"{context}.new_{name}")
        difference = abs(old_value - new_value)
        maximum = max(maximum, difference)
        require(
            difference <= tolerance,
            f"{context}: reproduced {name} differs by {difference:.17g} "
            f"(limit {tolerance:g})",
        )
    return maximum


def index_new_metrics(
    path: Path,
    *,
    model_family: str,
    run_name: str,
    hyperparameter_name: str,
) -> tuple[list[str], list[dict[str, str]], dict[tuple[str, int, str, str, str], dict[str, str]]]:
    fieldnames, rows = read_csv(path)
    missing = sorted(REQUIRED_METRIC_FIELDS - set(fieldnames))
    require(not missing, f"{path} is missing metric columns: {missing}")
    expected_validation, expected_tests = expected_metric_keys()
    expected_keys = set(expected_validation + expected_tests)
    require(
        len(rows) == EXPECTED_VALIDATION_CELLS_PER_MODEL + EXPECTED_TEST_CELLS_PER_MODEL,
        f"{path}: expected 1,225 metric rows, observed {len(rows)}",
    )
    index: dict[tuple[str, int, str, str, str], dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        context = f"{path}:{row_number}"
        require(row["run_name"] == run_name, f"{context}: unexpected run_name")
        require(row["model_family"] == model_family, f"{context}: unexpected model_family")
        require(
            row["hyperparameter_name"] == hyperparameter_name,
            f"{context}: unexpected hyperparameter_name",
        )
        require(selected(row["selected_hyperparams"]), f"{context}: setting is not selected")
        key = metric_key(row)
        require(key not in index, f"{path}: duplicate metric cell {key}")
        index[key] = row
        if row["stage"] == "test":
            require(row["split"] == "test", f"{context}: test cell has wrong split")
        else:
            require(
                row["stage"] == "validation_selected_reproduction",
                f"{context}: unexpected stage {row['stage']!r}",
            )
            require(row["split"] == "valid", f"{context}: validation cell has wrong split")
            require(row["eval_ratio"] == row["train_ratio"], f"{context}: invalid validation ratio")
        for name in INTEGER_METRICS:
            as_int(row[name], f"{context}.{name}")
        for name in FLOAT_METRICS:
            as_float(row[name], f"{context}.{name}")
    require(set(index) == expected_keys, f"{path}: metric-factor domain is not the full design")
    return fieldnames, rows, index


def index_source_metrics(
    path: Path,
    *,
    model_family: str,
) -> tuple[
    dict[tuple[str, int, str], dict[str, str]],
    dict[tuple[str, int, str, str], dict[str, str]],
]:
    fieldnames, rows = read_csv(path)
    missing = sorted(REQUIRED_METRIC_FIELDS - set(fieldnames))
    require(not missing, f"{path} is missing source metric columns: {missing}")
    validation: dict[tuple[str, int, str], dict[str, str]] = {}
    tests: dict[tuple[str, int, str, str], dict[str, str]] = {}
    for row in rows:
        if row["model_family"] != model_family or not selected(row["selected_hyperparams"]):
            continue
        key = config_key(row)
        if row["stage"] == "validation_grid" and row["split"] == "valid":
            require(key not in validation, f"{path}: duplicate selected validation cell {key}")
            validation[key] = row
        elif row["stage"] == "test" and row["split"] == "test":
            test_key = (*key, row["eval_ratio"])
            require(test_key not in tests, f"{path}: duplicate selected test cell {test_key}")
            tests[test_key] = row
    expected_configs = set(expected_config_keys())
    expected_tests = {(*key, ratio) for key in expected_configs for ratio in TEST_RATIOS}
    require(set(validation) == expected_configs, f"{path}: source selected settings are incomplete")
    require(set(tests) == expected_tests, f"{path}: source selected test cells are incomplete")
    return validation, tests


def validate_selected_hyperparameters(
    path: Path,
    *,
    model_family: str,
    hyperparameter_name: str,
    new_metrics: dict[tuple[str, int, str, str, str], dict[str, str]],
    source_validation: dict[tuple[str, int, str], dict[str, str]],
) -> dict[tuple[str, int, str], float]:
    fieldnames, rows = read_csv(path)
    required = {
        "model",
        "model_family",
        "train_ratio",
        "seed",
        "condition",
        "hyperparameter_name",
        "hyperparameter_value",
        "source_validation_mcc",
        "source_metrics",
    }
    require(required.issubset(fieldnames), f"{path}: selected-setting columns are incomplete")
    require(len(rows) == EXPECTED_CONFIGS_PER_MODEL, f"{path}: expected 175 selected settings")
    values: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = config_key(row)
        require(key not in values, f"{path}: duplicate selected setting {key}")
        require(row["model_family"] == model_family, f"{path}: wrong selected model family")
        require(
            row["hyperparameter_name"] == hyperparameter_name,
            f"{path}: wrong selected hyperparameter name for {key}",
        )
        value = as_float(row["hyperparameter_value"], f"{path}:{key}.hyperparameter")
        source = source_validation[key]
        require(
            value == as_float(source["C"], f"{path}:{key}.source_C"),
            f"{path}: selected hyperparameter changed for {key}",
        )
        require(row["model"] == source["model"], f"{path}: selected model name changed for {key}")
        require(
            as_float(row["source_validation_mcc"], f"{path}:{key}.source_mcc")
            == as_float(source["mcc"], f"{path}:{key}.source_metric_mcc"),
            f"{path}: selected source MCC changed for {key}",
        )
        new_validation = new_metrics[(*key, "validation_selected_reproduction", key[0])]
        require(
            value == as_float(new_validation["C"], f"{path}:{key}.new_C"),
            f"{path}: selected hyperparameter does not match new validation row for {key}",
        )
        for test_ratio in TEST_RATIOS:
            new_test = new_metrics[(*key, "test", test_ratio)]
            require(
                value == as_float(new_test["C"], f"{path}:{key}:{test_ratio}.new_C"),
                f"{path}: test hyperparameter changed for {(*key, test_ratio)}",
            )
        values[key] = value
    require(set(values) == set(expected_config_keys()), f"{path}: selected-setting domain is incomplete")
    return values


def validate_checksum_pair(
    before_path: Path,
    after_path: Path,
    *,
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    before_fields, before_rows = read_csv(before_path)
    after_fields, after_rows = read_csv(after_path)
    required = {"path", "size_bytes", "sha256"}
    require(required.issubset(before_fields), f"{before_path}: checksum columns are incomplete")
    require(before_fields == after_fields, "Input checksum table schemas differ before/after")
    require(before_rows == after_rows, "One or more rerun inputs changed during model fitting")
    index: dict[str, dict[str, str]] = {}
    for row in before_rows:
        recorded_path = row["path"]
        require(recorded_path not in index, f"Duplicate input checksum path: {recorded_path}")
        path = resolve_recorded_path(recorded_path)
        require(path.is_file(), f"Recorded rerun input no longer exists: {path}")
        expected_size = as_int(row["size_bytes"], f"{before_path}:{recorded_path}.size")
        require(path.stat().st_size == expected_size, f"Input size changed after rerun: {path}")
        actual_hash = cached_sha256(path, hash_cache)
        require(actual_hash == row["sha256"], f"Input hash changed after rerun: {path}")
        index[recorded_path] = row
    return {"file_count": len(index), "before_after_identical": True, "current_files_match": True}


def validate_manifest(
    path: Path,
    *,
    model_family: str,
    run_name: str,
    source_metrics_path: Path,
    output_dir: Path,
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    manifest = read_json(path)
    require(manifest.get("status") == "complete", f"Incomplete rerun manifest: {path}")
    require(manifest.get("model_family") == model_family, f"{path}: wrong model family")
    require(manifest.get("run_name") == run_name, f"{path}: wrong run name")
    require(
        manifest.get("source_metrics_sha256") == cached_sha256(source_metrics_path, hash_cache),
        f"{path}: source metrics checksum does not match",
    )
    factors = manifest.get("factors")
    require(isinstance(factors, dict), f"{path}: missing factor manifest")
    require(factors.get("train_ratios") == TRAIN_RATIOS, f"{path}: train ratios are incomplete")
    require(factors.get("test_ratios") == TEST_RATIOS, f"{path}: test ratios are incomplete")
    require(factors.get("seeds") == SEEDS, f"{path}: seeds are incomplete")
    require(factors.get("conditions") == CONDITIONS, f"{path}: conditions are incomplete")
    counts = manifest.get("counts")
    require(isinstance(counts, dict), f"{path}: missing count manifest")
    expected_counts = {
        "selected_model_fits": EXPECTED_CONFIGS_PER_MODEL,
        "validation_metric_rows": EXPECTED_VALIDATION_CELLS_PER_MODEL,
        "test_metric_rows": EXPECTED_TEST_CELLS_PER_MODEL,
        "validation_prediction_rows": EXPECTED_VALIDATION_SCORES_PER_MODEL,
        "test_prediction_rows": EXPECTED_TEST_SCORES_PER_MODEL,
        "all_prediction_rows": EXPECTED_ALL_SCORES_PER_MODEL,
        "reproduction_check_rows": EXPECTED_VALIDATION_CELLS_PER_MODEL + EXPECTED_TEST_CELLS_PER_MODEL,
    }
    for name, expected in expected_counts.items():
        require(counts.get(name) == expected, f"{path}: manifest {name} is not {expected}")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), f"{path}: missing output checksum manifest")
    expected_files = [
        "input_checksums.csv",
        "input_checksums_after.csv",
        "selected_hyperparameters.csv",
        "metrics.csv",
        "prediction_scores.csv",
        "reproduction_checks.csv",
    ]
    for filename in expected_files:
        output_path = output_dir / filename
        key = relative_or_absolute(output_path)
        require(key in outputs, f"{path}: output manifest omits {filename}")
        record = outputs[key]
        require(isinstance(record, dict), f"{path}: invalid output record for {filename}")
        require(record.get("size_bytes") == output_path.stat().st_size, f"{path}: size mismatch for {filename}")
        require(
            record.get("sha256") == cached_sha256(output_path, hash_cache),
            f"{path}: checksum mismatch for {filename}",
        )
    return {"status": "complete", "counts": expected_counts, "output_hashes_match": True}


def validate_old_reproduction_tables(
    new_metrics: dict[tuple[str, int, str, str, str], dict[str, str]],
    source_validation: dict[tuple[str, int, str], dict[str, str]],
    source_tests: dict[tuple[str, int, str, str], dict[str, str]],
    *,
    model_family: str,
    tolerance: float,
) -> float:
    maximum = 0.0
    for key in expected_config_keys():
        new_validation = new_metrics[(*key, "validation_selected_reproduction", key[0])]
        maximum = max(
            maximum,
            compare_old_reproduction(
                source_validation[key],
                new_validation,
                context=f"{model_family}:validation:{key}",
                tolerance=tolerance,
            ),
        )
        for test_ratio in TEST_RATIOS:
            maximum = max(
                maximum,
                compare_old_reproduction(
                    source_tests[(*key, test_ratio)],
                    new_metrics[(*key, "test", test_ratio)],
                    context=f"{model_family}:test:{(*key, test_ratio)}",
                    tolerance=tolerance,
                ),
            )
    return maximum


def validate_reproduction_checks(
    path: Path,
    *,
    model_family: str,
    tolerance: float,
) -> dict[str, Any]:
    fieldnames, rows = read_csv(path)
    required = {
        "model_family",
        "train_ratio",
        "seed",
        "condition",
        "stage",
        "eval_ratio",
        "max_abs_diff",
        "passed",
    }
    require(required.issubset(fieldnames), f"{path}: reproduction-check columns are incomplete")
    require(len(rows) == 1_225, f"{path}: expected 1,225 reproduction checks")
    keys: set[tuple[str, int, str, str, str]] = set()
    maximum = 0.0
    for row in rows:
        require(row["model_family"] == model_family, f"{path}: wrong model family")
        key = metric_key(row)
        require(key not in keys, f"{path}: duplicate reproduction check {key}")
        keys.add(key)
        require(selected(row["passed"]), f"{path}: failed reproduction check {key}")
        difference = as_float(row["max_abs_diff"], f"{path}:{key}.max_abs_diff")
        require(difference <= tolerance, f"{path}: reproduction difference exceeds tolerance for {key}")
        maximum = max(maximum, difference)
    expected_validation, expected_tests = expected_metric_keys()
    require(keys == set(expected_validation + expected_tests), f"{path}: reproduction-check domain is incomplete")
    return {"row_count": len(rows), "maximum_recorded_absolute_difference": maximum}


def benchmark_rows_for_group(
    *,
    stage: str,
    train_ratio: str,
    seed: int,
    condition: str,
    eval_ratio: str,
    noise_dir: Path,
    test_dir: Path,
    cache: dict[Path, list[dict[str, Any]]],
) -> tuple[Path, list[dict[str, Any]]]:
    if stage == "validation_selected_reproduction":
        path = noise_dir / ratio_dir_name(train_ratio) / f"seed_{seed}" / condition / "valid.jsonl"
    else:
        path = test_dir / ratio_dir_name(eval_ratio) / f"seed_{seed}" / "test.jsonl"
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = load_jsonl(resolved)
    return resolved, cache[resolved]


def validate_score_row(
    score_row: dict[str, str],
    benchmark_row: dict[str, Any],
    *,
    row_number: int,
    path: Path,
    model_family: str,
    run_name: str,
    model_name: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    score_type: str,
    decision_threshold: str,
    train_ratio: str,
    seed: int,
    condition: str,
    stage: str,
    eval_ratio: str,
) -> tuple[int, int, float]:
    context = f"{path}:score-row-{row_number}"
    expected_fixed = {
        "run_name": run_name,
        "model": model_name,
        "model_family": model_family,
        "train_ratio": train_ratio,
        "seed": str(seed),
        "condition": condition,
        "stage": stage,
        "eval_ratio": eval_ratio,
        "split": "valid" if stage == "validation_selected_reproduction" else "test",
        "hyperparameter_name": hyperparameter_name,
        "row_number": str(row_number),
        "score_type": score_type,
        "decision_threshold": decision_threshold,
    }
    for name, expected in expected_fixed.items():
        require(score_row[name] == expected, f"{context}: {name} is not canonical")
    recorded_hyperparameter = as_float(
        score_row["hyperparameter_value"],
        f"{context}.hyperparameter_value",
    )
    require(
        recorded_hyperparameter == hyperparameter_value,
        f"{context}: hyperparameter value differs from selected setting",
    )
    require(
        score_row["row_sha256"] == stable_row_sha256(benchmark_row),
        f"{context}: benchmark-row identity hash mismatch",
    )
    for name in IDENTITY_FIELDS:
        require(
            score_row[name] == canonical_csv_value(benchmark_row.get(name)),
            f"{context}: {name} differs from canonical JSONL row",
        )
    true_label = as_int(score_row["binary_label"], f"{context}.binary_label")
    expected_label = int(benchmark_row["binary_label"])
    require(true_label == expected_label, f"{context}: binary label differs from JSONL")
    require(true_label in {0, 1}, f"{context}: binary label is outside {{0,1}}")
    predicted_label = as_int(score_row["predicted_label"], f"{context}.predicted_label")
    require(predicted_label in {0, 1}, f"{context}: predicted label is outside {{0,1}}")
    score = as_float(score_row["positive_class_score"], f"{context}.positive_class_score")
    if score_type == "positive_class_probability":
        require(0.0 <= score <= 1.0, f"{context}: probability score is outside [0,1]")
        expected_prediction = int(score >= 0.5)
    else:
        expected_prediction = int(score > 0.0)
    require(
        predicted_label == expected_prediction,
        f"{context}: stored prediction does not follow the recorded threshold",
    )
    return true_label, predicted_label, score


def validate_prediction_scores(
    path: Path,
    *,
    model_family: str,
    run_name: str,
    hyperparameter_name: str,
    score_type: str,
    decision_threshold: str,
    selected_values: dict[tuple[str, int, str], float],
    new_metrics: dict[tuple[str, int, str, str, str], dict[str, str]],
    noise_dir: Path,
    test_dir: Path,
    metric_tolerance: float,
    benchmark_cache: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    require(path.is_file(), f"Missing prediction scores: {path}")
    validation_score_count = 0
    test_score_count = 0
    maximum_metric_difference = 0.0
    pr_auc_ap_difference_count = 0
    maximum_pr_auc_ap_difference = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Prediction score CSV has no header: {path}")
        missing = sorted(REQUIRED_SCORE_FIELDS - set(reader.fieldnames))
        require(not missing, f"{path} is missing score columns: {missing}")

        validation_keys, test_keys = expected_metric_keys()
        ordered_keys = []
        for config in expected_config_keys():
            ordered_keys.append((*config, "validation_selected_reproduction", config[0]))
            ordered_keys.extend((*config, "test", ratio) for ratio in TEST_RATIOS)
        require(set(ordered_keys) == set(validation_keys + test_keys), "Internal expected-order error")

        for key in ordered_keys:
            train_ratio, seed, condition, stage, eval_ratio = key
            benchmark_path, benchmark_rows = benchmark_rows_for_group(
                stage=stage,
                train_ratio=train_ratio,
                seed=seed,
                condition=condition,
                eval_ratio=eval_ratio,
                noise_dir=noise_dir,
                test_dir=test_dir,
                cache=benchmark_cache,
            )
            metric_row_value = new_metrics[key]
            model_name = metric_row_value["model"]
            hyperparameter_value = selected_values[(train_ratio, seed, condition)]
            y_true: list[int] = []
            y_pred: list[int] = []
            y_score: list[float] = []
            for row_number, benchmark_row in enumerate(benchmark_rows, start=1):
                try:
                    score_row = next(reader)
                except StopIteration as exc:
                    raise ValidationFailure(
                        f"{path}: ended before complete score group {key}"
                    ) from exc
                true_label, predicted_label, score = validate_score_row(
                    score_row,
                    benchmark_row,
                    row_number=row_number,
                    path=path,
                    model_family=model_family,
                    run_name=run_name,
                    model_name=model_name,
                    hyperparameter_name=hyperparameter_name,
                    hyperparameter_value=hyperparameter_value,
                    score_type=score_type,
                    decision_threshold=decision_threshold,
                    train_ratio=train_ratio,
                    seed=seed,
                    condition=condition,
                    stage=stage,
                    eval_ratio=eval_ratio,
                )
                y_true.append(true_label)
                y_pred.append(predicted_label)
                y_score.append(score)

            reconstructed = reconstruct_metrics(
                np.asarray(y_true, dtype=np.int64),
                np.asarray(y_pred, dtype=np.int64),
                np.asarray(y_score, dtype=np.float64),
            )
            maximum_metric_difference = max(
                maximum_metric_difference,
                compare_reconstructed_metrics(
                    metric_row_value,
                    reconstructed,
                    context=f"{model_family}:{key}",
                    tolerance=metric_tolerance,
                ),
            )
            ap_difference = abs(
                float(reconstructed["average_precision"]) - float(reconstructed["pr_auc"])
            )
            if ap_difference > metric_tolerance:
                pr_auc_ap_difference_count += 1
            maximum_pr_auc_ap_difference = max(maximum_pr_auc_ap_difference, ap_difference)
            if stage == "test":
                test_score_count += len(benchmark_rows)
            else:
                validation_score_count += len(benchmark_rows)

        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        require(extra is None, f"{path}: contains score rows beyond the complete design")

    require(
        validation_score_count == EXPECTED_VALIDATION_SCORES_PER_MODEL,
        f"{path}: expected 16,555 validation score rows, observed {validation_score_count}",
    )
    require(
        test_score_count == EXPECTED_TEST_SCORES_PER_MODEL,
        f"{path}: expected 98,700 test score rows, observed {test_score_count}",
    )
    return {
        "validation_score_rows": validation_score_count,
        "test_score_rows": test_score_count,
        "all_score_rows": validation_score_count + test_score_count,
        "maximum_reconstructed_metric_absolute_difference": maximum_metric_difference,
        "cells_where_ap_and_pr_auc_differ_beyond_tolerance": pr_auc_ap_difference_count,
        "maximum_ap_vs_pr_auc_absolute_difference": maximum_pr_auc_ap_difference,
        "canonical_jsonl_order_identity_and_labels_match": True,
        "score_group_keys_are_complete_and_unique": True,
    }


def validate_one_run(
    *,
    model_family: str,
    spec: dict[str, str],
    source_results_dir: Path,
    new_results_dir: Path,
    noise_dir: Path,
    test_dir: Path,
    metric_tolerance: float,
    old_tolerance: float,
    hash_cache: dict[Path, str],
    benchmark_cache: dict[Path, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    source_metrics_path = source_results_dir / spec["source"] / "metrics.csv"
    output_dir = new_results_dir / spec["new"]
    require(output_dir.is_dir(), f"Missing corrected rerun directory: {output_dir}")
    print(f"Validating {model_family}: manifests and checksums...", flush=True)
    manifest_report = validate_manifest(
        output_dir / "manifest.json",
        model_family=model_family,
        run_name=spec["new"],
        source_metrics_path=source_metrics_path,
        output_dir=output_dir,
        hash_cache=hash_cache,
    )
    checksum_report = validate_checksum_pair(
        output_dir / "input_checksums.csv",
        output_dir / "input_checksums_after.csv",
        hash_cache=hash_cache,
    )
    metric_fields, metric_rows, new_metrics = index_new_metrics(
        output_dir / "metrics.csv",
        model_family=model_family,
        run_name=spec["new"],
        hyperparameter_name=spec["hyperparameter"],
    )
    source_validation, source_tests = index_source_metrics(
        source_metrics_path,
        model_family=model_family,
    )
    selected_values = validate_selected_hyperparameters(
        output_dir / "selected_hyperparameters.csv",
        model_family=model_family,
        hyperparameter_name=spec["hyperparameter"],
        new_metrics=new_metrics,
        source_validation=source_validation,
    )
    old_maximum = validate_old_reproduction_tables(
        new_metrics,
        source_validation,
        source_tests,
        model_family=model_family,
        tolerance=old_tolerance,
    )
    reproduction_report = validate_reproduction_checks(
        output_dir / "reproduction_checks.csv",
        model_family=model_family,
        tolerance=old_tolerance,
    )
    print(f"Validating {model_family}: every retained score against benchmark JSONL...", flush=True)
    score_report = validate_prediction_scores(
        output_dir / "prediction_scores.csv",
        model_family=model_family,
        run_name=spec["new"],
        hyperparameter_name=spec["hyperparameter"],
        score_type=spec["score_type"],
        decision_threshold=spec["decision_threshold"],
        selected_values=selected_values,
        new_metrics=new_metrics,
        noise_dir=noise_dir,
        test_dir=test_dir,
        metric_tolerance=metric_tolerance,
        benchmark_cache=benchmark_cache,
    )
    report = {
        "source_run": spec["source"],
        "corrected_run": spec["new"],
        "selected_configurations": len(selected_values),
        "validation_metric_cells": EXPECTED_VALIDATION_CELLS_PER_MODEL,
        "test_metric_cells": EXPECTED_TEST_CELLS_PER_MODEL,
        "old_results_maximum_absolute_difference": old_maximum,
        "manifest": manifest_report,
        "input_checksums": checksum_report,
        "runner_reproduction_checks": reproduction_report,
        "independent_score_reconstruction": score_report,
    }
    return report, metric_fields, metric_rows


def write_report_outputs(
    report_dir: Path,
    *,
    report: dict[str, Any],
    metric_fieldnames: list[str],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> None:
    require(not report_dir.exists(), f"Refusing to overwrite report directory: {report_dir}")
    report_dir.mkdir(parents=True)
    validation_path = report_dir / "combined_validation_results_true_pr_auc_v1.csv"
    test_path = report_dir / "combined_test_results_true_pr_auc_v1.csv"
    write_csv_exact(validation_path, metric_fieldnames, validation_rows)
    write_csv_exact(test_path, metric_fieldnames, test_rows)
    report["generated_outputs"] = {
        validation_path.name: {
            "rows": len(validation_rows),
            "size_bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        },
        test_path.name: {
            "rows": len(test_rows),
            "size_bytes": test_path.stat().st_size,
            "sha256": sha256_file(test_path),
        },
    }
    write_json_new(report_dir / "validation_report.json", report)
    summary_lines = [
        "# CWE-119 True PR-AUC Rerun Validation",
        "",
        "- Status: **passed**",
        f"- Selected configurations: {EXPECTED_CONFIGS}",
        f"- Validation metric cells: {EXPECTED_VALIDATION_CELLS}",
        f"- Test metric cells: {EXPECTED_TEST_CELLS}",
        f"- Validation score rows: {EXPECTED_VALIDATION_SCORES}",
        f"- Test score rows: {EXPECTED_TEST_SCORES}",
        f"- All score rows: {EXPECTED_ALL_SCORES}",
        "- Every score row matched its canonical benchmark JSONL row and label.",
        "- Every stored metric was independently reconstructed from retained scores.",
        "- Average precision and trapezoidal PR-AUC passed a non-aliasing fixture.",
        "",
    ]
    summary_path = report_dir / "validation_summary.md"
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the frozen original MCC-selected runs.",
    )
    parser.add_argument(
        "--new-output-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the four corrected true-PR-AUC reruns.",
    )
    parser.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--report-dir",
        type=Path,
        help=f"New report directory (default: NEW_RESULTS/analysis/{REPORT_VERSION}).",
    )
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--old-result-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run the AP-versus-PR-AUC non-aliasing fixture without reading rerun files.",
    )
    args = parser.parse_args()
    require(args.metric_tolerance > 0, "Metric tolerance must be positive.")
    require(args.old_result_tolerance > 0, "Old-result tolerance must be positive.")
    if args.report_dir is None:
        args.report_dir = args.new_output_results_dir / "analysis" / REPORT_VERSION
    return args


def main() -> None:
    try:
        args = parse_args()
        fixture = metric_fixture()
        if args.self_test_only:
            print(json.dumps({"status": "passed", "metric_fixture": fixture}, indent=2))
            return
        require(
            not args.report_dir.exists(),
            f"Refusing to overwrite report directory: {args.report_dir}",
        )
        hash_cache: dict[Path, str] = {}
        benchmark_cache: dict[Path, list[dict[str, Any]]] = {}
        run_reports: dict[str, Any] = {}
        combined_fieldnames: list[str] | None = None
        validation_rows: list[dict[str, str]] = []
        test_rows: list[dict[str, str]] = []
        for model_family, spec in RUNS.items():
            run_report, fieldnames, rows = validate_one_run(
                model_family=model_family,
                spec=spec,
                source_results_dir=args.source_results_dir,
                new_results_dir=args.new_output_results_dir,
                noise_dir=args.noise_dir,
                test_dir=args.test_dir,
                metric_tolerance=args.metric_tolerance,
                old_tolerance=args.old_result_tolerance,
                hash_cache=hash_cache,
                benchmark_cache=benchmark_cache,
            )
            run_reports[model_family] = run_report
            if combined_fieldnames is None:
                combined_fieldnames = fieldnames
            require(fieldnames == combined_fieldnames, "Corrected metric CSV schemas differ by model")
            for row in rows:
                if row["stage"] == "test":
                    test_rows.append(row)
                else:
                    validation_rows.append(row)

        require(combined_fieldnames is not None, "No corrected metric rows were loaded")
        require(len(validation_rows) == EXPECTED_VALIDATION_CELLS, "Aggregate validation-cell count failed")
        require(len(test_rows) == EXPECTED_TEST_CELLS, "Aggregate test-cell count failed")
        require(
            sum(r["selected_configurations"] for r in run_reports.values()) == EXPECTED_CONFIGS,
            "Aggregate selected-configuration count failed",
        )
        aggregate_validation_scores = sum(
            r["independent_score_reconstruction"]["validation_score_rows"]
            for r in run_reports.values()
        )
        aggregate_test_scores = sum(
            r["independent_score_reconstruction"]["test_score_rows"]
            for r in run_reports.values()
        )
        require(aggregate_validation_scores == EXPECTED_VALIDATION_SCORES, "Aggregate validation-score count failed")
        require(aggregate_test_scores == EXPECTED_TEST_SCORES, "Aggregate test-score count failed")

        report: dict[str, Any] = {
            "status": "passed",
            "report_version": REPORT_VERSION,
            "validated_at_utc": utc_now(),
            "validator": relative_or_absolute(Path(__file__)),
            "validator_sha256": sha256_file(Path(__file__)),
            "metric_tolerance": args.metric_tolerance,
            "old_result_tolerance": args.old_result_tolerance,
            "metric_fixture": fixture,
            "aggregate_counts": {
                "model_families": len(run_reports),
                "selected_configurations": EXPECTED_CONFIGS,
                "validation_metric_cells": len(validation_rows),
                "test_metric_cells": len(test_rows),
                "validation_score_rows": aggregate_validation_scores,
                "test_score_rows": aggregate_test_scores,
                "all_score_rows": aggregate_validation_scores + aggregate_test_scores,
            },
            "runs": run_reports,
        }
        require(
            report["aggregate_counts"]["all_score_rows"] == EXPECTED_ALL_SCORES,
            "Aggregate all-score count failed",
        )
        write_report_outputs(
            args.report_dir,
            report=report,
            metric_fieldnames=combined_fieldnames,
            validation_rows=validation_rows,
            test_rows=test_rows,
        )
        print(f"Validation passed. Report written to {args.report_dir}", flush=True)
    except ValidationFailure as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

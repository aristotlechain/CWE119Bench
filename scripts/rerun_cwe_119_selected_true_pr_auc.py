#!/usr/bin/env python3
"""Reproduce the final MCC-selected models with auditable true PR-AUC.

This runner deliberately does not repeat hyperparameter selection. It reads the
selected validation row for each model/train-ratio/seed/condition group from a
previous final MCC-selected run, fits that setting once, and evaluates it on
the matching clean validation set and all requested clean test ratios.

The script writes to a new result directory and refuses to overwrite an
existing directory. It retains full per-example continuous scores so that AP,
trapezoidal PR-AUC, ROC-AUC, thresholded metrics, and PR curves can be audited
without another model-training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from run_cwe_119_model_experiments import (
    CONDITIONS,
    SEEDS,
    TEST_RATIOS,
    TRAIN_VALID_RATIOS,
    compute_metrics,
    evaluate_model,
    fit_model,
    load_jsonl,
    metric_row,
    ratio_dir_name,
    texts_and_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results/model_experiments"
DEFAULT_NOISE_DIR = PROJECT_ROOT / "data/processed/cwe_119_noise_ratio_preserved"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data/processed/cwe_119_imbalance"

FINAL_RUNS = {
    "logistic_regression": {
        "source_run": "logreg_balanced_ratio_preserved_noise_mcc_selected_v1",
        "output_run": "logreg_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
    },
    "linear_svm": {
        "source_run": "linear_svm_balanced_ratio_preserved_noise_mcc_selected_v1",
        "output_run": "linear_svm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
    },
    "random_forest": {
        "source_run": "random_forest_balanced_ratio_preserved_noise_mcc_selected_v1",
        "output_run": "random_forest_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
    },
    "lightgbm": {
        "source_run": "lightgbm_balanced_ratio_preserved_noise_mcc_selected_v1",
        "output_run": "lightgbm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
    },
}

FLOAT_REPRODUCTION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
]
INTEGER_REPRODUCTION_METRICS = [
    "n_rows",
    "positive_rows",
    "benign_rows",
    "tn",
    "fp",
    "fn",
    "tp",
]
PREDICTION_FIELDS = [
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
    "idx",
    "func_hash",
    "commit_id",
    "cve",
    "file_hash",
    "source_primevul_file",
    "benchmark_ratio",
    "benchmark_seed",
    "benchmark_row_role",
    "binary_label",
    "predicted_label",
    "positive_class_score",
    "score_type",
    "decision_threshold",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return f"external/{path.name}"


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


def is_selected(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def stable_row_sha256(row: dict[str, Any]) -> str:
    identity = [
        row.get("source_primevul_file"),
        row.get("idx"),
        row.get("func_hash"),
        row.get("commit_id"),
        row.get("cve"),
        row.get("file_hash"),
    ]
    material = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_csv_full_precision(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"No rows available for {path}.")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (bool, np.bool_)):
                    formatted[key] = str(bool(value))
                elif isinstance(value, (float, np.floating)):
                    formatted[key] = format(float(value), ".17g")
                elif isinstance(value, (int, np.integer)):
                    formatted[key] = int(value)
                elif value is None:
                    formatted[key] = ""
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ["scikit-learn", "lightgbm", "numpy", "pandas", "scipy"]:
        versions[package] = importlib.metadata.version(package)
    return versions


def git_status() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return "clean" if not result.stdout else "modified"


def validate_metric_implementation() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)
    metrics = compute_metrics(y_true, y_pred, y_score)
    if not np.isclose(metrics["average_precision"], 5.0 / 6.0):
        raise SystemExit("Average-precision self-test failed.")
    if not np.isclose(metrics["pr_auc"], 19.0 / 24.0):
        raise SystemExit("Trapezoidal PR-AUC self-test failed.")
    if np.isclose(metrics["average_precision"], metrics["pr_auc"]):
        raise SystemExit("Metric self-test failed: AP and PR-AUC are aliased.")


def load_source_results(
    metrics_path: Path,
    *,
    model_family: str,
    ratios: list[str],
    seeds: list[int],
    conditions: list[str],
    test_ratios: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], pd.Series], dict[tuple[str, int, str, str], pd.Series]]:
    if not metrics_path.exists():
        raise SystemExit(f"Missing source result file: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    required_columns = {
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
        *FLOAT_REPRODUCTION_METRICS,
        *INTEGER_REPRODUCTION_METRICS,
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise SystemExit(f"Source metrics are missing columns: {', '.join(missing)}")
    families = set(frame["model_family"].astype(str))
    if families != {model_family}:
        raise SystemExit(f"Unexpected model families in {metrics_path}: {families}")
    if not frame["model"].astype(str).str.contains("class_weight_balanced").all():
        raise SystemExit("Source results are not uniformly class-weight balanced.")

    validation = frame[
        (frame["stage"] == "validation_grid")
        & (frame["split"] == "valid")
        & is_selected(frame["selected_hyperparams"])
    ].copy()
    validation = validation[
        validation["train_ratio"].isin(ratios)
        & validation["seed"].isin(seeds)
        & validation["condition"].isin(conditions)
    ]
    expected_groups = len(ratios) * len(seeds) * len(conditions)
    group_columns = ["train_ratio", "seed", "condition"]
    if len(validation) != expected_groups or validation.duplicated(group_columns).any():
        raise SystemExit(
            "Selected source settings are incomplete or duplicated: "
            f"expected {expected_groups}, observed {len(validation)}."
        )

    tests = frame[
        (frame["stage"] == "test")
        & (frame["split"] == "test")
        & frame["train_ratio"].isin(ratios)
        & frame["seed"].isin(seeds)
        & frame["condition"].isin(conditions)
        & frame["eval_ratio"].isin(test_ratios)
        & is_selected(frame["selected_hyperparams"])
    ].copy()
    expected_tests = expected_groups * len(test_ratios)
    test_columns = [*group_columns, "eval_ratio"]
    if len(tests) != expected_tests or tests.duplicated(test_columns).any():
        raise SystemExit(
            "Source test rows are incomplete or duplicated: "
            f"expected {expected_tests}, observed {len(tests)}."
        )

    selected_lookup = {
        (str(row.train_ratio), int(row.seed), str(row.condition)): row
        for row in validation.itertuples(index=False)
    }
    selected_series_lookup = {
        key: validation[
            (validation["train_ratio"] == key[0])
            & (validation["seed"] == key[1])
            & (validation["condition"] == key[2])
        ].iloc[0]
        for key in selected_lookup
    }
    test_lookup = {
        (str(row.train_ratio), int(row.seed), str(row.condition), str(row.eval_ratio)): row
        for row in tests.itertuples(index=False)
    }
    test_series_lookup = {
        key: tests[
            (tests["train_ratio"] == key[0])
            & (tests["seed"] == key[1])
            & (tests["condition"] == key[2])
            & (tests["eval_ratio"] == key[3])
        ].iloc[0]
        for key in test_lookup
    }
    for key, test_row in test_series_lookup.items():
        selected_row = selected_series_lookup[key[:3]]
        if float(test_row["C"]) != float(selected_row["C"]):
            raise SystemExit(
                f"Source test hyperparameter differs from selected validation row: {key}."
            )
        if str(test_row["model"]) != str(selected_row["model"]):
            raise SystemExit(f"Source test model differs from selection: {key}.")
    return validation, selected_series_lookup, test_series_lookup


def input_paths(
    *,
    noise_dir: Path,
    test_dir: Path,
    ratios: list[str],
    seeds: list[int],
    conditions: list[str],
    test_ratios: list[str],
) -> list[Path]:
    paths: set[Path] = set()
    for ratio in ratios:
        for seed in seeds:
            for condition in conditions:
                condition_dir = noise_dir / ratio_dir_name(ratio) / f"seed_{seed}" / condition
                paths.add(condition_dir / "train.jsonl")
                paths.add(condition_dir / "valid.jsonl")
                paths.add(condition_dir / "metadata.json")
            for test_ratio in test_ratios:
                test_seed_dir = test_dir / ratio_dir_name(test_ratio) / f"seed_{seed}"
                paths.add(test_seed_dir / "test.jsonl")
                paths.add(test_seed_dir / "metadata.json")

    paths.update(
        [
            PROJECT_ROOT / "data/raw/primevul_main/primevul_train.jsonl",
            PROJECT_ROOT / "data/raw/primevul_main/primevul_valid.jsonl",
            PROJECT_ROOT / "data/raw/primevul_main/primevul_test.jsonl",
            PROJECT_ROOT / "data/manifests/cwe_119_imbalance_generation_summary.csv",
            PROJECT_ROOT / "data/manifests/noise_ratio_preserved_generation_summary.csv",
            PROJECT_ROOT / "scripts/build_cwe_119_imbalance_benchmarks.py",
            PROJECT_ROOT / "scripts/build_cwe_119_ratio_preserved_noise_benchmarks.py",
            PROJECT_ROOT / "scripts/run_cwe_119_model_experiments.py",
            Path(__file__).resolve(),
        ]
    )
    missing = [path for path in sorted(paths) if not path.exists()]
    if missing:
        raise SystemExit("Missing rerun inputs:\n" + "\n".join(map(str, missing)))
    return sorted(paths)


def checksum_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def comparison_row(
    *,
    model_family: str,
    train_ratio: str,
    seed: int,
    condition: str,
    stage: str,
    eval_ratio: str,
    old: pd.Series,
    new: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_family": model_family,
        "train_ratio": train_ratio,
        "seed": seed,
        "condition": condition,
        "stage": stage,
        "eval_ratio": eval_ratio,
    }
    passed = True
    maximum = 0.0
    for metric in INTEGER_REPRODUCTION_METRICS:
        old_value = int(old[metric])
        new_value = int(new[metric])
        difference = abs(new_value - old_value)
        out[f"old_{metric}"] = old_value
        out[f"new_{metric}"] = new_value
        out[f"abs_diff_{metric}"] = difference
        passed = passed and difference == 0
        maximum = max(maximum, float(difference))
    for metric in FLOAT_REPRODUCTION_METRICS:
        old_value = float(old[metric])
        new_value = float(new[metric])
        difference = abs(new_value - old_value)
        out[f"old_{metric}"] = old_value
        out[f"new_{metric}"] = new_value
        out[f"abs_diff_{metric}"] = difference
        passed = passed and difference <= tolerance
        maximum = max(maximum, difference)
    out["max_abs_diff"] = maximum
    out["passed"] = passed
    return out


def write_prediction_rows(
    writer: csv.DictWriter,
    *,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    scores: np.ndarray,
    run_name: str,
    model_name: str,
    model_family: str,
    train_ratio: str,
    seed: int,
    condition: str,
    stage: str,
    eval_ratio: str,
    split: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    score_type: str,
    decision_threshold: str,
) -> int:
    if len(rows) != len(predictions) or len(rows) != len(scores):
        raise SystemExit("Prediction row length mismatch.")
    for row_number, (row, prediction, score) in enumerate(
        zip(rows, predictions, scores),
        start=1,
    ):
        writer.writerow(
            {
                "run_name": run_name,
                "model": model_name,
                "model_family": model_family,
                "train_ratio": train_ratio,
                "seed": seed,
                "condition": condition,
                "stage": stage,
                "eval_ratio": eval_ratio,
                "split": split,
                "hyperparameter_name": hyperparameter_name,
                "hyperparameter_value": format(float(hyperparameter_value), ".17g"),
                "row_number": row_number,
                "row_sha256": stable_row_sha256(row),
                "idx": row.get("idx", ""),
                "func_hash": row.get("func_hash", ""),
                "commit_id": row.get("commit_id", ""),
                "cve": row.get("cve", ""),
                "file_hash": row.get("file_hash", ""),
                "source_primevul_file": row.get("source_primevul_file", ""),
                "benchmark_ratio": row.get("benchmark_ratio", ""),
                "benchmark_seed": row.get("benchmark_seed", ""),
                "benchmark_row_role": row.get("benchmark_row_role", ""),
                "binary_label": int(row["binary_label"]),
                "predicted_label": int(prediction),
                "positive_class_score": format(float(score), ".17g"),
                "score_type": score_type,
                "decision_threshold": decision_threshold,
            }
        )
    return len(rows)


def run(args: argparse.Namespace) -> None:
    validate_metric_implementation()
    config = FINAL_RUNS[args.model_family]
    source_run_name = args.source_run_name or config["source_run"]
    run_name = args.run_name or config["output_run"]
    source_metrics = args.source_results_dir / source_run_name / "metrics.csv"
    output_dir = args.output_results_dir / run_name
    if output_dir.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )

    selected, selected_lookup, old_test_lookup = load_source_results(
        source_metrics,
        model_family=args.model_family,
        ratios=args.ratios,
        seeds=args.seeds,
        conditions=args.conditions,
        test_ratios=args.test_ratios,
    )
    paths = input_paths(
        noise_dir=args.noise_dir,
        test_dir=args.test_dir,
        ratios=args.ratios,
        seeds=args.seeds,
        conditions=args.conditions,
        test_ratios=args.test_ratios,
    )
    print(
        f"Hashing {len(paths)} unique input/provenance files before training...",
        flush=True,
    )
    inputs = checksum_rows(paths)

    output_dir.mkdir(parents=True)
    started_at = utc_now()
    start = perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at,
        "run_name": run_name,
        "model_family": args.model_family,
        "source_run_name": source_run_name,
        "source_metrics": relative(source_metrics),
        "source_metrics_sha256": sha256_file(source_metrics),
        "project_git_commit_before_rerun": args.git_commit,
        "project_git_status_before_rerun": git_status(),
        "command": portable_command(),
        "working_directory": ".",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "factors": {
            "train_ratios": args.ratios,
            "test_ratios": args.test_ratios,
            "seeds": args.seeds,
            "conditions": args.conditions,
        },
        "fixed_settings": {
            "class_weight": "balanced",
            "tfidf_analyzer": "char_wb",
            "tfidf_ngram_range": [3, 5],
            "tfidf_max_features": 20000,
            "tfidf_min_df": 2,
            "tfidf_lowercase": False,
        },
        "metric_definitions": {
            "average_precision": "sklearn.metrics.average_precision_score(y_true, y_score)",
            "pr_auc": "sklearn.metrics.auc(recall, precision), where precision and recall come from precision_recall_curve(y_true, y_score, pos_label=1, drop_intermediate=False)",
            "score_policy": "positive-class probability for probability models, decision_function score for LinearSVC",
        },
        "reproduction_tolerance": args.reproduction_tolerance,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_csv_full_precision(output_dir / "input_checksums.csv", inputs)

    selected_rows = []
    for row in selected.to_dict(orient="records"):
        selected_rows.append(
            {
                "model": row["model"],
                "model_family": row["model_family"],
                "train_ratio": row["train_ratio"],
                "seed": int(row["seed"]),
                "condition": row["condition"],
                "hyperparameter_name": row["hyperparameter_name"],
                "hyperparameter_value": float(row["C"]),
                "source_validation_mcc": float(row["mcc"]),
                "source_metrics": relative(source_metrics),
            }
        )
    write_csv_full_precision(output_dir / "selected_hyperparameters.csv", selected_rows)

    metrics_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    prediction_count = 0
    validation_prediction_count = 0
    test_prediction_count = 0
    score_partial = output_dir / "prediction_scores.csv.partial"
    score_final = output_dir / "prediction_scores.csv"

    try:
        with score_partial.open("w", encoding="utf-8", newline="") as score_handle:
            score_writer = csv.DictWriter(score_handle, fieldnames=PREDICTION_FIELDS)
            score_writer.writeheader()
            group_total = len(args.ratios) * len(args.seeds) * len(args.conditions)
            group_index = 0

            for train_ratio in args.ratios:
                for seed in args.seeds:
                    for condition in args.conditions:
                        group_index += 1
                        group_start = perf_counter()
                        key = (train_ratio, seed, condition)
                        source_selected = selected_lookup[key]
                        hyperparameter_value = float(source_selected["C"])
                        hyperparameter_name = str(source_selected["hyperparameter_name"])
                        model_name = str(source_selected["model"])
                        condition_dir = (
                            args.noise_dir
                            / ratio_dir_name(train_ratio)
                            / f"seed_{seed}"
                            / condition
                        )
                        train_rows = load_jsonl(condition_dir / "train.jsonl")
                        valid_rows = load_jsonl(condition_dir / "valid.jsonl")
                        train_texts, y_train = texts_and_labels(
                            train_rows,
                            label_field="noisy_label",
                        )
                        valid_texts, y_valid = texts_and_labels(
                            valid_rows,
                            label_field="binary_label",
                        )
                        vectorizer = TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            max_features=20000,
                            min_df=2,
                            lowercase=False,
                        )
                        x_train = vectorizer.fit_transform(train_texts)
                        x_valid = vectorizer.transform(valid_texts)
                        model = fit_model(
                            args.model_family,
                            x_train,
                            y_train,
                            c_value=hyperparameter_value,
                            class_weight="balanced",
                        )
                        if not np.array_equal(
                            np.asarray(model.classes_, dtype=np.int64),
                            np.array([0, 1], dtype=np.int64),
                        ):
                            raise SystemExit(
                                f"Unexpected class order for {key}: {model.classes_}."
                            )
                        if args.model_family == "linear_svm":
                            score_type = "decision_function"
                            decision_threshold = ">0 (model.predict used, exact zero is class 0)"
                        else:
                            score_type = "positive_class_probability"
                            decision_threshold = ">=0.5"

                        valid_prediction, valid_score = evaluate_model(model, x_valid)
                        valid_metrics = compute_metrics(
                            y_valid,
                            valid_prediction,
                            valid_score,
                        )
                        validation_comparison = comparison_row(
                            model_family=args.model_family,
                            train_ratio=train_ratio,
                            seed=seed,
                            condition=condition,
                            stage="validation_selected_reproduction",
                            eval_ratio=train_ratio,
                            old=source_selected,
                            new=valid_metrics,
                            tolerance=args.reproduction_tolerance,
                        )
                        reproduction_rows.append(validation_comparison)
                        if not validation_comparison["passed"]:
                            raise SystemExit(
                                "Validation reproduction failed for "
                                f"{key}: max difference "
                                f"{validation_comparison['max_abs_diff']}."
                            )
                        metrics_rows.append(
                            metric_row(
                                run_name=run_name,
                                model_name=model_name,
                                model_family=args.model_family,
                                train_ratio=train_ratio,
                                seed=seed,
                                condition=condition,
                                stage="validation_selected_reproduction",
                                eval_ratio=train_ratio,
                                split="valid",
                                c_value=hyperparameter_value,
                                selected_hyperparams=True,
                                metrics=valid_metrics,
                                train_rows=len(train_rows),
                                valid_rows=len(valid_rows),
                                elapsed_seconds=perf_counter() - group_start,
                            )
                        )
                        written = write_prediction_rows(
                            score_writer,
                            rows=valid_rows,
                            predictions=valid_prediction,
                            scores=valid_score,
                            run_name=run_name,
                            model_name=model_name,
                            model_family=args.model_family,
                            train_ratio=train_ratio,
                            seed=seed,
                            condition=condition,
                            stage="validation_selected_reproduction",
                            eval_ratio=train_ratio,
                            split="valid",
                            hyperparameter_name=hyperparameter_name,
                            hyperparameter_value=hyperparameter_value,
                            score_type=score_type,
                            decision_threshold=decision_threshold,
                        )
                        prediction_count += written
                        validation_prediction_count += written

                        for test_ratio in args.test_ratios:
                            test_path = (
                                args.test_dir
                                / ratio_dir_name(test_ratio)
                                / f"seed_{seed}"
                                / "test.jsonl"
                            )
                            test_rows = load_jsonl(test_path)
                            test_texts, y_test = texts_and_labels(
                                test_rows,
                                label_field="binary_label",
                            )
                            x_test = vectorizer.transform(test_texts)
                            test_prediction, test_score = evaluate_model(model, x_test)
                            test_metrics = compute_metrics(
                                y_test,
                                test_prediction,
                                test_score,
                            )
                            old_test = old_test_lookup[
                                (train_ratio, seed, condition, test_ratio)
                            ]
                            test_comparison = comparison_row(
                                model_family=args.model_family,
                                train_ratio=train_ratio,
                                seed=seed,
                                condition=condition,
                                stage="test",
                                eval_ratio=test_ratio,
                                old=old_test,
                                new=test_metrics,
                                tolerance=args.reproduction_tolerance,
                            )
                            reproduction_rows.append(test_comparison)
                            if not test_comparison["passed"]:
                                raise SystemExit(
                                    "Test reproduction failed for "
                                    f"{(train_ratio, seed, condition, test_ratio)}: "
                                    f"max difference {test_comparison['max_abs_diff']}."
                                )
                            metrics_rows.append(
                                metric_row(
                                    run_name=run_name,
                                    model_name=model_name,
                                    model_family=args.model_family,
                                    train_ratio=train_ratio,
                                    seed=seed,
                                    condition=condition,
                                    stage="test",
                                    eval_ratio=test_ratio,
                                    split="test",
                                    c_value=hyperparameter_value,
                                    selected_hyperparams=True,
                                    metrics=test_metrics,
                                    train_rows=len(train_rows),
                                    valid_rows=len(valid_rows),
                                    elapsed_seconds=perf_counter() - group_start,
                                )
                            )
                            written = write_prediction_rows(
                                score_writer,
                                rows=test_rows,
                                predictions=test_prediction,
                                scores=test_score,
                                run_name=run_name,
                                model_name=model_name,
                                model_family=args.model_family,
                                train_ratio=train_ratio,
                                seed=seed,
                                condition=condition,
                                stage="test",
                                eval_ratio=test_ratio,
                                split="test",
                                hyperparameter_name=hyperparameter_name,
                                hyperparameter_value=hyperparameter_value,
                                score_type=score_type,
                                decision_threshold=decision_threshold,
                            )
                            prediction_count += written
                            test_prediction_count += written

                        print(
                            f"[{group_index:03d}/{group_total:03d}] "
                            f"{args.model_family} {train_ratio} seed={seed} "
                            f"condition={condition} hyperparameter="
                            f"{hyperparameter_name}={hyperparameter_value:g} "
                            f"elapsed={perf_counter() - group_start:.1f}s",
                            flush=True,
                        )

        score_partial.replace(score_final)
        write_csv_full_precision(output_dir / "metrics.csv", metrics_rows)
        write_csv_full_precision(
            output_dir / "reproduction_checks.csv",
            reproduction_rows,
        )
        if not all(bool(row["passed"]) for row in reproduction_rows):
            raise SystemExit("One or more reproduction checks failed.")

        expected_groups = len(args.ratios) * len(args.seeds) * len(args.conditions)
        expected_test_cells = expected_groups * len(args.test_ratios)
        observed_validation = sum(
            row["stage"] == "validation_selected_reproduction"
            for row in metrics_rows
        )
        observed_test = sum(row["stage"] == "test" for row in metrics_rows)
        if observed_validation != expected_groups or observed_test != expected_test_cells:
            raise SystemExit("Aggregate metric row-count validation failed.")

        inputs_after = checksum_rows(paths)
        if inputs_after != inputs:
            raise SystemExit("One or more input/provenance files changed during the rerun.")
        write_csv_full_precision(output_dir / "input_checksums_after.csv", inputs_after)

        elapsed = perf_counter() - start
        output_files = [
            output_dir / "input_checksums.csv",
            output_dir / "input_checksums_after.csv",
            output_dir / "selected_hyperparameters.csv",
            output_dir / "metrics.csv",
            output_dir / "prediction_scores.csv",
            output_dir / "reproduction_checks.csv",
        ]
        manifest.update(
            {
                "status": "complete",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": elapsed,
                "counts": {
                    "selected_model_fits": expected_groups,
                    "validation_metric_rows": observed_validation,
                    "test_metric_rows": observed_test,
                    "validation_prediction_rows": validation_prediction_count,
                    "test_prediction_rows": test_prediction_count,
                    "all_prediction_rows": prediction_count,
                    "reproduction_check_rows": len(reproduction_rows),
                },
                "outputs": {
                    relative(path): {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in output_files
                },
            }
        )
        write_json(output_dir / "manifest.json", manifest)
        summary = [
            "# Selected-Model True PR-AUC Rerun",
            "",
            f"- Status: complete",
            f"- Model family: `{args.model_family}`",
            f"- Source final run: `{source_run_name}`",
            f"- Selected model fits: {expected_groups}",
            f"- Test evaluation cells: {observed_test}",
            f"- Test prediction-score rows: {test_prediction_count}",
            f"- Validation prediction-score rows: {validation_prediction_count}",
            f"- Maximum allowed reproduction difference: {args.reproduction_tolerance:g}",
            f"- Runtime: {elapsed:.2f} seconds",
            "",
            "PR-AUC is calculated independently as trapezoidal area over the full "
            "precision-recall curve. Average precision remains a separate metric.",
            "",
        ]
        (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
        print(
            f"Completed {run_name}: {observed_test} test cells, "
            f"{test_prediction_count} test scores, {elapsed:.1f}s.",
            flush=True,
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": perf_counter() - start,
                "error": f"{type(exc).__name__}: {exc}",
                "partial_counts": {
                    "aggregate_metric_rows": len(metrics_rows),
                    "prediction_rows": prediction_count,
                    "reproduction_check_rows": len(reproduction_rows),
                },
            }
        )
        write_json(output_dir / "manifest.json", manifest)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce final selected models with true trapezoidal PR-AUC."
    )
    parser.add_argument("--model-family", choices=sorted(FINAL_RUNS), required=True)
    parser.add_argument("--source-run-name")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the frozen original final runs.",
    )
    parser.add_argument(
        "--output-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory in which the new versioned run is created.",
    )
    parser.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--ratios", nargs="+", default=TRAIN_VALID_RATIOS)
    parser.add_argument("--test-ratios", nargs="+", default=TEST_RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-6)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    unsupported_ratios = sorted(set(args.ratios) - set(TRAIN_VALID_RATIOS))
    unsupported_tests = sorted(set(args.test_ratios) - set(TEST_RATIOS))
    unsupported_seeds = sorted(set(args.seeds) - set(SEEDS))
    unsupported_conditions = sorted(set(args.conditions) - set(CONDITIONS))
    if unsupported_ratios or unsupported_tests or unsupported_seeds or unsupported_conditions:
        raise SystemExit(
            "Unsupported requested factors: "
            f"train={unsupported_ratios}, test={unsupported_tests}, "
            f"seeds={unsupported_seeds}, conditions={unsupported_conditions}"
        )
    if args.reproduction_tolerance <= 0:
        raise SystemExit("Reproduction tolerance must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())

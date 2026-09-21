#!/usr/bin/env python3
"""Run first-stage CWE-119 model experiments.

Protocol:
  - train on data/processed/cwe_119_noise_ratio_preserved/<ratio>/<seed>/<condition>/train.jsonl
  - tune hyperparameters on the matching clean validation file
  - evaluate the selected model on all clean test ratios for the same seed
  - use only the source-code text field `func` as input
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TRAIN_VALID_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
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
HYPERPARAMETER_GRIDS = {
    "logistic_regression": [0.1, 1.0, 10.0],
    "linear_svm": [0.1, 1.0, 10.0],
    "random_forest": [1.0, 2.0, 5.0],
    "lightgbm": [15.0, 31.0, 63.0],
}
MODEL_FAMILIES = list(HYPERPARAMETER_GRIDS)
SELECTION_METRICS = ["mcc", "f1", "balanced_accuracy", "roc_auc"]
DEFAULT_TIE_BREAK_METRICS = ["f1", "recall", "precision", "balanced_accuracy"]


def ratio_dir_name(ratio: str) -> str:
    return "ratio_" + ratio.replace("/", "_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def texts_and_labels(
    rows: list[dict[str, Any]],
    *,
    label_field: str,
) -> tuple[list[str], np.ndarray]:
    texts = [row.get("func") or "" for row in rows]
    labels = np.array([int(row[label_field]) for row in rows], dtype=np.int64)
    return texts, labels


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mcc_denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = 0.0 if mcc_denominator == 0 else (
        (tp * tn - fp * fn) / mcc_denominator
    )

    metrics: dict[str, Any] = {
        "n_rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "benign_rows": int((y_true == 0).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true,
            y_score,
            pos_label=1,
            drop_intermediate=False,
        )
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["pr_auc"] = None

    return metrics


def metric_display_name(metric_name: str) -> str:
    return {
        "mcc": "MCC",
        "f1": "F1",
        "balanced_accuracy": "balanced-accuracy",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "recall": "recall",
        "precision": "precision",
    }.get(metric_name, metric_name.replace("_", " "))


def metric_tiebreak_text(metric_names: list[str]) -> str:
    return "/".join(metric_display_name(metric_name) for metric_name in metric_names)


def metric_sort_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return -np.inf
    return float(value)


def selection_metric_key(
    metrics: dict[str, Any],
    selection_metric: str,
    tie_break_metrics: list[str],
) -> tuple[float, ...]:
    return tuple(
        metric_sort_value(metrics, metric_name)
        for metric_name in [selection_metric, *tie_break_metrics]
    )


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def metric_row(
    *,
    run_name: str,
    model_name: str,
    model_family: str,
    train_ratio: str,
    seed: int,
    condition: str,
    stage: str,
    eval_ratio: str,
    split: str,
    c_value: float,
    selected_hyperparams: bool,
    metrics: dict[str, Any],
    train_rows: int,
    valid_rows: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    hyperparameter_names = {
        "random_forest": "min_samples_leaf",
        "lightgbm": "num_leaves",
    }
    row = {
        "run_name": run_name,
        "model": model_name,
        "model_family": model_family,
        "hyperparameter_name": hyperparameter_names.get(model_family, "C"),
        "train_ratio": train_ratio,
        "seed": seed,
        "condition": condition,
        "stage": stage,
        "eval_ratio": eval_ratio,
        "split": split,
        "C": c_value,
        "selected_hyperparams": selected_hyperparams,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "elapsed_seconds": elapsed_seconds,
    }
    row.update(metrics)
    return row


def fit_logistic_regression(
    x_train,
    y_train: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=2000,
        random_state=0,
    )
    model.fit(x_train, y_train)
    return model


def fit_linear_svm(
    x_train,
    y_train: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> LinearSVC:
    model = LinearSVC(
        C=c_value,
        class_weight=class_weight,
        max_iter=10000,
        random_state=0,
    )
    model.fit(x_train, y_train)
    return model


def fit_random_forest(
    x_train,
    y_train: np.ndarray,
    *,
    min_samples_leaf: float,
    class_weight: str | None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=int(min_samples_leaf),
        max_features="sqrt",
        class_weight=class_weight,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def fit_lightgbm(
    x_train,
    y_train: np.ndarray,
    *,
    num_leaves: float,
    class_weight: str | None,
) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=25,
        learning_rate=0.05,
        num_leaves=int(num_leaves),
        min_child_samples=10,
        max_bin=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        class_weight=class_weight,
        random_state=0,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


def fit_model(
    model_family: str,
    x_train,
    y_train: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> (
    LogisticRegression
    | LinearSVC
    | RandomForestClassifier
    | LGBMClassifier
):
    if model_family == "logistic_regression":
        return fit_logistic_regression(
            x_train,
            y_train,
            c_value=c_value,
            class_weight=class_weight,
        )
    if model_family == "linear_svm":
        return fit_linear_svm(
            x_train,
            y_train,
            c_value=c_value,
            class_weight=class_weight,
        )
    if model_family == "random_forest":
        return fit_random_forest(
            x_train,
            y_train,
            min_samples_leaf=c_value,
            class_weight=class_weight,
        )
    if model_family == "lightgbm":
        return fit_lightgbm(
            x_train,
            y_train,
            num_leaves=c_value,
            class_weight=class_weight,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def evaluate_model(
    model: (
        LogisticRegression
        | LinearSVC
        | RandomForestClassifier
        | LGBMClassifier
    ),
    x_eval,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_eval)[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int64)
        return y_pred, y_score

    y_score = model.decision_function(x_eval)
    y_pred = model.predict(x_eval).astype(np.int64)
    return y_pred, y_score


def run_one_experiment(
    *,
    run_name: str,
    noise_dir: Path,
    test_dir: Path,
    train_ratio: str,
    seed: int,
    condition: str,
    class_weight: str | None,
    model_family: str,
    model_name: str,
    max_features: int,
    min_df: int,
    selection_metric: str,
    tie_break_metrics: list[str],
) -> list[dict[str, Any]]:
    start = perf_counter()
    condition_dir = noise_dir / ratio_dir_name(train_ratio) / f"seed_{seed}" / condition
    train_path = condition_dir / "train.jsonl"
    valid_path = condition_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise SystemExit(f"Missing train/valid condition files in {condition_dir}")

    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)

    train_texts, y_train = texts_and_labels(train_rows, label_field="noisy_label")
    valid_texts, y_valid = texts_and_labels(valid_rows, label_field="binary_label")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=min_df,
        lowercase=False,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(valid_texts)

    validation_rows: list[dict[str, Any]] = []
    best_model: (
        LogisticRegression
        | LinearSVC
        | RandomForestClassifier
        | LGBMClassifier
        | None
    ) = None
    best_c = None
    best_selection_key: tuple[float, ...] | None = None

    for c_value in HYPERPARAMETER_GRIDS[model_family]:
        model = fit_model(
            model_family,
            x_train,
            y_train,
            c_value=c_value,
            class_weight=class_weight,
        )
        y_valid_pred, y_valid_score = evaluate_model(model, x_valid)
        metrics = compute_metrics(y_valid, y_valid_pred, y_valid_score)
        validation_rows.append(
            metric_row(
                run_name=run_name,
                model_name=model_name,
                model_family=model_family,
                train_ratio=train_ratio,
                seed=seed,
                condition=condition,
                stage="validation_grid",
                eval_ratio=train_ratio,
                split="valid",
                c_value=c_value,
                selected_hyperparams=False,
                metrics=metrics,
                train_rows=len(train_rows),
                valid_rows=len(valid_rows),
                elapsed_seconds=perf_counter() - start,
            )
        )
        selection_key = selection_metric_key(
            metrics,
            selection_metric,
            tie_break_metrics,
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_c = c_value
            best_model = model

    if best_model is None or best_c is None:
        raise SystemExit(f"Could not select {model_family} hyperparameters.")

    output_rows: list[dict[str, Any]] = []
    for row in validation_rows:
        row = dict(row)
        row["selected_hyperparams"] = row["C"] == best_c
        output_rows.append(row)

    for test_ratio in TEST_RATIOS:
        test_path = test_dir / ratio_dir_name(test_ratio) / f"seed_{seed}" / "test.jsonl"
        if not test_path.exists():
            raise SystemExit(f"Missing clean test file: {test_path}")
        test_rows = load_jsonl(test_path)
        test_texts, y_test = texts_and_labels(test_rows, label_field="binary_label")
        x_test = vectorizer.transform(test_texts)
        y_test_pred, y_test_score = evaluate_model(best_model, x_test)
        metrics = compute_metrics(y_test, y_test_pred, y_test_score)
        output_rows.append(
            metric_row(
                run_name=run_name,
                model_name=model_name,
                model_family=model_family,
                train_ratio=train_ratio,
                seed=seed,
                condition=condition,
                stage="test",
                eval_ratio=test_ratio,
                split="test",
                c_value=best_c,
                selected_hyperparams=True,
                metrics=metrics,
                train_rows=len(train_rows),
                valid_rows=len(valid_rows),
                elapsed_seconds=perf_counter() - start,
            )
        )

    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = format(value, ".17g")
                elif value is None:
                    formatted[key] = ""
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def write_summary_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    run_dir: Path,
    *,
    selection_metric: str,
    tie_break_metrics: list[str],
) -> None:
    test_rows = [row for row in rows if row["stage"] == "test"]
    validation_rows = [row for row in rows if row["stage"] == "validation_grid"]
    lines: list[str] = []
    lines.append("# Model Experiment Run Summary")
    lines.append("")
    lines.append(f"Results directory: `{run_dir}`")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Features: `func` source-code text only")
    lines.append("- Vectorizer: TF-IDF `char_wb` n-grams 3-5, fit on training only")
    model_names = sorted({str(row["model"]) for row in rows})
    lines.append(f"- Model: `{', '.join(model_names)}`")
    lines.append("- Class weighting is recorded in the model name when supported")
    model_families = sorted({str(row["model_family"]) for row in rows})
    hyperparameter_names = sorted({str(row["hyperparameter_name"]) for row in rows})
    hyperparameter_values = sorted({float(row["C"]) for row in rows})
    lines.append(f"- Model family: `{', '.join(model_families)}`")
    lines.append(
        "- Hyperparameter grid: `"
        + ", ".join(hyperparameter_names)
        + " = "
        + ", ".join(f"{value:g}" for value in hyperparameter_values)
        + "`"
    )
    lines.append(
        "- Hyperparameter selection: highest validation "
        f"{metric_display_name(selection_metric)}, with "
        f"{metric_tiebreak_text(tie_break_metrics)} tie-breaks"
    )
    lines.append("- Test evaluation: selected model evaluated on all clean test ratios")
    lines.append(
        "- PR-AUC is the trapezoidal area of the precision-recall curve, "
        "computed from continuous scores with precision_recall_curve and "
        "auc(recall, precision). Average precision is retained separately."
    )
    lines.append("")

    lines.append("## Validation Grid")
    lines.append("")
    lines.append("| Train ratio | Seed | Condition | Hyperparameter | Selected | F1 | MCC | Recall | Precision | PR-AUC | Balanced accuracy |")
    lines.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in validation_rows:
        lines.append(
            "| "
            f"{row['train_ratio']} | "
            f"{row['seed']} | "
            f"{row['condition']} | "
            f"{row['hyperparameter_name']}={row['C']} | "
            f"{row['selected_hyperparams']} | "
            f"{fmt_metric(row['f1'])} | "
            f"{fmt_metric(row['mcc'])} | "
            f"{fmt_metric(row['recall'])} | "
            f"{fmt_metric(row['precision'])} | "
            f"{fmt_metric(row['pr_auc'])} | "
            f"{fmt_metric(row['balanced_accuracy'])} |"
        )
    lines.append("")

    lines.append("## Test Results")
    lines.append("")
    lines.append("| Train ratio | Seed | Condition | Test ratio | F1 | MCC | Recall | Precision | PR-AUC | Balanced accuracy | TP | FP | FN | TN |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in test_rows:
        lines.append(
            "| "
            f"{row['train_ratio']} | "
            f"{row['seed']} | "
            f"{row['condition']} | "
            f"{row['eval_ratio']} | "
            f"{fmt_metric(row['f1'])} | "
            f"{fmt_metric(row['mcc'])} | "
            f"{fmt_metric(row['recall'])} | "
            f"{fmt_metric(row['precision'])} | "
            f"{fmt_metric(row['pr_auc'])} | "
            f"{fmt_metric(row['balanced_accuracy'])} | "
            f"{row['tp']} | "
            f"{row['fp']} | "
            f"{row['fn']} | "
            f"{row['tn']} |"
        )
    lines.append("")
    lines.append("CSV metrics:")
    lines.append("")
    lines.append("- `metrics.csv`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run CWE-119 model experiments.")
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cwe_119_noise_ratio_preserved",
        help="Train/validation condition directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cwe_119_imbalance",
        help="Clean test-ratio directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results/model_experiments",
        help="Parent results directory.",
    )
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--ratios", nargs="+", default=TRAIN_VALID_RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default="logistic_regression",
        help="Model family to train.",
    )
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="mcc",
        help="Validation metric used for hyperparameter selection.",
    )
    parser.add_argument(
        "--tie-break-metrics",
        nargs="+",
        choices=["mcc", "f1", "recall", "precision", "balanced_accuracy", "roc_auc"],
        default=DEFAULT_TIE_BREAK_METRICS,
        help="Validation metrics used as tie-breaks, in order.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Estimator class_weight setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.results_dir / args.run_name
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing results directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    class_weight = None if args.class_weight == "none" else args.class_weight
    model_name = f"{args.model_family}_class_weight_{args.class_weight}"

    all_rows: list[dict[str, Any]] = []
    for ratio in args.ratios:
        if ratio not in TRAIN_VALID_RATIOS:
            raise SystemExit(f"Unsupported train/validation ratio: {ratio}")
        for seed in args.seeds:
            if seed not in SEEDS:
                raise SystemExit(f"Unsupported seed: {seed}")
            for condition in args.conditions:
                if condition not in CONDITIONS:
                    raise SystemExit(f"Unsupported condition: {condition}")
                rows = run_one_experiment(
                    run_name=args.run_name,
                    noise_dir=args.noise_dir,
                    test_dir=args.test_dir,
                    train_ratio=ratio,
                    seed=seed,
                    condition=condition,
                    class_weight=class_weight,
                    model_family=args.model_family,
                    model_name=model_name,
                    max_features=args.max_features,
                    min_df=args.min_df,
                    selection_metric=args.selection_metric,
                    tie_break_metrics=args.tie_break_metrics,
                )
                all_rows.extend(rows)

    write_csv(run_dir / "metrics.csv", all_rows)
    write_summary_markdown(
        run_dir / "summary.md",
        all_rows,
        run_dir,
        selection_metric=args.selection_metric,
        tie_break_metrics=args.tie_break_metrics,
    )


if __name__ == "__main__":
    main()

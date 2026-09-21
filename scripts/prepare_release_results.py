#!/usr/bin/env python3
"""Normalize a complete five-model analysis matrix for compact publication."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["logistic_regression", "linear_svm", "random_forest", "lightgbm", "codebert"]
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
KEYS = ["model_family", "train_ratio", "eval_ratio", "seed", "condition"]
METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
    "roc_auc", "average_precision", "pr_auc",
]
OUTPUT_COLUMNS = [
    "model", "model_family", "hyperparameter_name", "selected_hyperparameter_value",
    "learning_rate", "selected_epoch", "train_ratio", "eval_ratio", "seed", "condition",
    "train_rows", "valid_rows", "n_rows", "positive_rows", "benign_rows", *METRICS,
    "tn", "fp", "fn", "tp",
]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(OUTPUT_COLUMNS) - {"selected_hyperparameter_value"}
    required.update({"C", "hyperparameter_value"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input matrix is missing columns: {missing}")
    if len(frame) != 5250 or frame.duplicated(KEYS).any():
        raise ValueError("Expected 5,250 unique five-model test cells.")
    expected = set(itertools.product(MODELS, TRAIN_RATIOS, TEST_RATIOS, SEEDS, CONDITIONS))
    observed = set(frame[KEYS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"Five-model factor grid differs from the protocol. Missing={len(expected-observed)}.")
    if "stage" in frame and not frame["stage"].eq("test").all():
        raise ValueError("Input includes non-test rows.")
    result = frame.copy()
    result["selected_hyperparameter_value"] = result["hyperparameter_value"].combine_first(result["C"])
    for metric in METRICS:
        values = pd.to_numeric(result[metric], errors="coerce")
        lower = -1.0 if metric == "mcc" else 0.0
        if not np.isfinite(values).all() or not values.between(lower, 1.0).all():
            raise ValueError(f"Invalid {metric} values.")
        result[metric] = values
    ordered = result[OUTPUT_COLUMNS].copy()
    for column, categories in (
        ("model_family", MODELS),
        ("train_ratio", TRAIN_RATIOS),
        ("eval_ratio", TEST_RATIOS),
        ("condition", CONDITIONS),
    ):
        ordered[column] = pd.Categorical(ordered[column], categories=categories, ordered=True)
    ordered = ordered.sort_values(KEYS).reset_index(drop=True)
    for column in ("model_family", "train_ratio", "eval_ratio", "condition"):
        ordered[column] = ordered[column].astype(str)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
    result = normalize(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.17g")
    print(f"Wrote {len(result):,} normalized test cells to {args.output}")


if __name__ == "__main__":
    main()

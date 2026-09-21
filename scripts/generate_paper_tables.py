#!/usr/bin/env python3
"""Generate the paper's aggregate tables across five models.

This script preserves the paper's tabular formats. Figure generation uses
``generate_paper_figures.py``. Dataset and noise-construction tables stay
unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHED_RESULTS = (
    PROJECT_ROOT
    / "results/five_model_test_results.csv"
)
DEFAULT_PAPER_DIR = PROJECT_ROOT
CONDITIONS = [
    "random_flip_10",
    "random_flip_20",
    "random_fp_10",
    "random_fp_20",
    "heuristic_fp_10",
    "heuristic_fp_20",
]
MODELS = ["logistic_regression", "linear_svm", "random_forest", "lightgbm", "codebert"]
TRAIN_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
TEST_RATIOS = ["50/50", "60/40", "70/30", "75/25", "80/20", "90/10"]
METRICS = ["mcc", "f1", "precision", "recall", "pr_auc", "balanced_accuracy"]
DELTA_METRICS = [f"delta_{metric}" for metric in METRICS]
T_975_DF4 = 2.7764451051977987
TABLE_METRICS = ["mcc", "f1", "precision", "recall", "pr_auc"]
METRIC_LABELS = {
    "mcc": "MCC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall$^{\\dagger}$",
    "pr_auc": "PR-AUC",
}
CONDITION_LABELS = {
    "random_flip_10": "Paired flip 10\\%",
    "random_flip_20": "Paired flip 20\\%",
    "random_fp_10": "Random FP 10\\%",
    "random_fp_20": "Random FP 20\\%",
    "heuristic_fp_10": "Heuristic FP 10\\%",
    "heuristic_fp_20": "Heuristic FP 20\\%",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_seed_values(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
    cells_per_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group.sort_values("seed")[value_column].to_numpy(dtype=float)
        require(len(values) == 5, f"Expected five seed values for {dict(zip(group_columns, keys))}.")
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_975_DF4 * sd / np.sqrt(5)
        row: dict[str, float | int | str] = dict(zip(group_columns, keys))
        row.update(
            {
                "mean": mean,
                "seed_sd": sd,
                "ci95_half_width": half_width,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "minimum_seed_mean": float(values.min()),
                "maximum_seed_mean": float(values.max()),
                "n_seeds": 5,
                "cells_averaged_per_seed": cells_per_seed,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require(len(raw) == 5250, "Expected the complete five-model matrix of 5,250 test cells.")
    require(set(raw["model_family"]) == set(MODELS), "Unexpected model-family set.")
    require(set(raw["condition"]) == {"clean", *CONDITIONS}, "Unexpected condition set.")
    key = ["model_family", "train_ratio", "eval_ratio", "seed"]
    require(not raw.duplicated([*key, "condition"]).any(), "Duplicate matched test cell.")

    clean = raw[raw["condition"].eq("clean")][[*key, *METRICS]].copy()
    require(len(clean) == 750, "Expected 750 clean five-model cells.")
    noisy = raw[~raw["condition"].eq("clean")][[*key, "condition", *METRICS]].merge(
        clean,
        on=key,
        suffixes=("", "_clean"),
        validate="many_to_one",
    )
    require(len(noisy) == 4500, "Expected 4,500 matched noisy cells.")
    for metric in METRICS:
        noisy[f"delta_{metric}"] = noisy[metric] - noisy[f"{metric}_clean"]

    clean_seed = clean.groupby(["train_ratio", "eval_ratio", "seed"], as_index=False)[METRICS].mean()
    clean_long = clean_seed.melt(
        id_vars=["train_ratio", "eval_ratio", "seed"],
        value_vars=METRICS,
        var_name="metric",
        value_name="seed_mean",
    )
    clean_summary = summarize_seed_values(
        clean_long,
        group_columns=["train_ratio", "eval_ratio", "metric"],
        value_column="seed_mean",
        cells_per_seed=5,
    )

    noise_seed = noisy.groupby(
        ["condition", "train_ratio", "eval_ratio", "seed"], as_index=False
    )[DELTA_METRICS].mean()
    noise_long = noise_seed.melt(
        id_vars=["condition", "train_ratio", "eval_ratio", "seed"],
        value_vars=DELTA_METRICS,
        var_name="metric",
        value_name="seed_mean",
    )
    noise_summary = summarize_seed_values(
        noise_long,
        group_columns=["condition", "train_ratio", "eval_ratio", "metric"],
        value_column="seed_mean",
        cells_per_seed=5,
    )

    low = noise_seed[noise_seed["eval_ratio"].eq("50/50")].drop(columns="eval_ratio")
    high = noise_seed[noise_seed["eval_ratio"].eq("90/10")].drop(columns="eval_ratio")
    endpoint = low.merge(
        high,
        on=["condition", "train_ratio", "seed"],
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    for metric in DELTA_METRICS:
        endpoint[metric] = endpoint[f"{metric}_high"] - endpoint[f"{metric}_low"]
    endpoint_long = endpoint.melt(
        id_vars=["condition", "train_ratio", "seed"],
        value_vars=DELTA_METRICS,
        var_name="metric",
        value_name="seed_mean",
    )
    endpoint_summary = summarize_seed_values(
        endpoint_long,
        group_columns=["condition", "train_ratio", "metric"],
        value_column="seed_mean",
        cells_per_seed=5,
    )
    endpoint_summary["contrast_values_per_seed"] = 1
    endpoint_summary["contrast"] = "noise_delta_at_test_90/10_minus_noise_delta_at_test_50/50"
    endpoint_summary["low_level"] = "50/50"
    endpoint_summary["high_level"] = "90/10"
    endpoint_summary["cells_averaged_per_seed_at_each_endpoint"] = 5

    return clean_summary, noise_summary, endpoint_summary, noisy


def select_one(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    require(len(selected) == 1, f"Expected one summary row for {filters}.")
    return selected.iloc[0]


def clean_cell(row: pd.Series, *, discussed: bool, bold: bool) -> str:
    value = f"{float(row['mean']):.3f} \\pm {float(row['seed_sd']):.3f}"
    if bold:
        value = f"\\mathbf{{{value}}}"
    prefix = "\\cellcolor{black!10}" if discussed else ""
    return f"{prefix}${value}$"


def build_clean_table_preserving_format(clean: pd.DataFrame, path: Path) -> None:
    discussed = {
        (train, test, metric)
        for train in ("60/40", "90/10")
        for test in ("50/50", "90/10")
        for metric in ("mcc", "f1", "precision", "recall")
    } | {
        ("75/25", test, metric)
        for test in ("50/50", "90/10")
        for metric in ("mcc", "precision", "pr_auc")
    }
    reversal = {
        (train, test, metric)
        for train in ("60/40", "90/10")
        for test in ("50/50", "90/10")
        for metric in ("mcc", "f1")
    }
    lines = [
        "% Generated by scripts/generate_paper_tables.py.",
        "\\begin{table}[tbp]",
        "\\centering",
        "\\caption{Clean performance across train/validation and test class ratio configurations. Values show mean $\\pm$ SD across five benchmark datasets, averaged over the five models for each train--test ratio combination.}",
        "\\label{tab:clean-ratio-grid}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\renewcommand{\\arraystretch}{0.88}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Train/valid & Metric & Test 50:50 & Test 60:40 & Test 70:30 & Test 75:25 & Test 80:20 & Test 90:10 \\\\",
        "\\midrule",
    ]
    for train_index, train_ratio in enumerate(TRAIN_RATIOS):
        for metric_index, metric in enumerate(TABLE_METRICS):
            prefix = f"\\multirow{{5}}{{*}}{{{train_ratio.replace('/', ':')}}}" if metric_index == 0 else ""
            cells = [
                clean_cell(
                    select_one(clean, train_ratio=train_ratio, eval_ratio=test_ratio, metric=metric),
                    discussed=(train_ratio, test_ratio, metric) in discussed,
                    bold=(train_ratio, test_ratio, metric) in reversal,
                )
                for test_ratio in TEST_RATIOS
            ]
            lines.append(" & ".join([prefix, METRIC_LABELS[metric], *cells]) + " \\\\")
        if train_index != len(TRAIN_RATIOS) - 1:
            lines.extend(["%\\addlinespace[1.5pt]", "\\cmidrule{2-8}"])
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{1mm}",
            "\\parbox{0.98\\textwidth}{\\footnotesize \\textit{Emphasis:} Light gray identifies endpoint values discussed in the text. Boldface marks the F1/MCC four-corner comparison whose direction reverses between 50:50 and 90:10 testing. $^{\\dagger}$Recall is invariant since the vulnerable test instances are unchanged.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def signed_endpoint_cell(row: pd.Series) -> str:
    mean = float(row["mean"])
    mean_text = f"{mean:+.3f}"
    if mean > 0:
        mean_text = f"\\mathbf{{{mean_text}}}"
    return f"${mean_text} \\pm {float(row['seed_sd']):.3f}$"


def build_endpoint_table_preserving_format(endpoint: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by scripts/generate_paper_tables.py.",
        "\\begin{table}[tb]",
        "\\centering",
        "\\caption{Change in the matched noise effect between 50:50 and 90:10 test configurations. Each entry is noisy-minus-clean metric difference at 50:50 testing subtracted from the corresponding noisy-minus-clean difference at 90:10 testing per train ratio. Positive value means the difference is higher at 90:10 testing. (Mean $\\pm$ SD across five benchmark versions averaging five classifiers per test configuration). %Positive mean values are shown in bold.",
        "}",
        "\\label{tab:noise-test-interactions}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Metric & Noise condition & Train 60:40 & Train 70:30 & Train 75:25 & Train 80:20 & Train 90:10 \\\\",
        "\\midrule",
    ]
    for metric_index, metric in enumerate(["delta_mcc", "delta_pr_auc"]):
        for condition_index, condition in enumerate(CONDITIONS):
            prefix = (
                f"\\multirow{{6}}{{*}}{{{'MCC' if metric == 'delta_mcc' else 'PR-AUC'}}}"
                if condition_index == 0
                else ""
            )
            cells = [
                signed_endpoint_cell(
                    select_one(endpoint, condition=condition, train_ratio=train_ratio, metric=metric)
                )
                for train_ratio in TRAIN_RATIOS
            ]
            lines.append(" & ".join([prefix, CONDITION_LABELS[condition], *cells]) + " \\\\")
        if metric_index == 0:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_assets(paper_dir: Path, clean: pd.DataFrame, endpoint: pd.DataFrame) -> list[Path]:
    table_dir = paper_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    clean_path = table_dir / "tab_clean_train_test_joint_seed_aware.tex"
    endpoint_path = table_dir / "tab_noise_test_endpoint_interactions_seed_aware.tex"
    clean_csv = table_dir / "tab_clean_train_test_joint_seed_aware.csv"
    endpoint_csv = table_dir / "tab_noise_test_endpoint_interactions_seed_aware.csv"
    build_clean_table_preserving_format(clean, clean_path)
    build_endpoint_table_preserving_format(endpoint, endpoint_path)
    clean.to_csv(clean_csv, index=False, float_format="%.17g")
    endpoint.to_csv(endpoint_csv, index=False, float_format="%.17g")
    return [clean_path, endpoint_path, clean_csv, endpoint_csv]


def scalar(summary: pd.DataFrame, **filters: str) -> float:
    selected = summary
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    require(len(selected) == 1, f"Expected one summary row for {filters}.")
    return float(selected.iloc[0]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()

    raw = pd.read_csv(MATCHED_RESULTS)
    clean, noise, endpoint, paired = prepare_summaries(raw)
    outputs = write_assets(paper_dir, clean, endpoint)

    overall = noise.groupby(["condition", "metric"], as_index=False)["mean"].mean()
    worst_counts: dict[str, int] = {}
    for metric in ["delta_mcc", "delta_pr_auc"]:
        selected = noise[noise["metric"].eq(metric)]
        worst_counts[metric] = sum(
            group.loc[group["mean"].idxmin(), "condition"] == "random_flip_20"
            for _, group in selected.groupby(["train_ratio", "eval_ratio"])
        )
        require(worst_counts[metric] == 30, f"Paired flip 20% is not worst in every {metric} cell.")
    claims = {
        "status": "passed",
        "policy": "paper table formats, five-model aggregation",
        "source": str(MATCHED_RESULTS.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(MATCHED_RESULTS),
        "test_cells": len(raw),
        "model_families": MODELS,
        "generated_outputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        },
        "paper_values": {
            "paired_flip_20_overall_delta_mcc": scalar(
                overall, condition="random_flip_20", metric="delta_mcc"
            ),
            "paired_flip_20_overall_delta_pr_auc": scalar(
                overall, condition="random_flip_20", metric="delta_pr_auc"
            ),
            "paired_flip_20_worst_mcc_cells": worst_counts["delta_mcc"],
            "paired_flip_20_worst_pr_auc_cells": worst_counts["delta_pr_auc"],
            "clean_75_25_test_50_50_precision": scalar(
                clean, train_ratio="75/25", eval_ratio="50/50", metric="precision"
            ),
            "clean_75_25_test_90_10_precision": scalar(
                clean, train_ratio="75/25", eval_ratio="90/10", metric="precision"
            ),
            "clean_75_25_test_50_50_pr_auc": scalar(
                clean, train_ratio="75/25", eval_ratio="50/50", metric="pr_auc"
            ),
            "clean_75_25_test_90_10_pr_auc": scalar(
                clean, train_ratio="75/25", eval_ratio="90/10", metric="pr_auc"
            ),
            "clean_75_25_test_50_50_mcc": scalar(
                clean, train_ratio="75/25", eval_ratio="50/50", metric="mcc"
            ),
            "clean_75_25_test_90_10_mcc": scalar(
                clean, train_ratio="75/25", eval_ratio="90/10", metric="mcc"
            ),
            "paired_flip_20_train_60_40_test_90_10_delta_mcc": scalar(
                noise,
                condition="random_flip_20",
                train_ratio="60/40",
                eval_ratio="90/10",
                metric="delta_mcc",
            ),
            "paired_flip_20_train_90_10_test_90_10_delta_mcc": scalar(
                noise,
                condition="random_flip_20",
                train_ratio="90/10",
                eval_ratio="90/10",
                metric="delta_mcc",
            ),
            "paired_flip_20_excluding_train_90_10_delta_mcc": float(
                paired[
                    paired["condition"].eq("random_flip_20")
                    & ~paired["train_ratio"].eq("90/10")
                ]["delta_mcc"].mean()
            ),
        },
    }
    claim_path = PROJECT_ROOT / "tables/paper_asset_manifest.json"
    claim_path.write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()

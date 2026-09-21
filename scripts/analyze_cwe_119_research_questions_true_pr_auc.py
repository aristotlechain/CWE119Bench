#!/usr/bin/env python3
"""Build seed-aware research-question evidence from validated true PR-AUC results.

The script consumes only the independently validated combined test matrix.  It
does not alter manuscript files or any legacy result directory.  Uncertainty
is calculated across five seed-level factorial-grid means. Correlated model,
train-ratio, and test-ratio cells are not treated as independent replications.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = (
    PROJECT_ROOT
    / "results/model_experiments/analysis/cwe119_true_pr_auc_validation_v1"
)
DEFAULT_INPUT = VALIDATION_DIR / "combined_test_results_true_pr_auc_v1.csv"
DEFAULT_VALIDATION_REPORT = VALIDATION_DIR / "validation_report.json"
DEFAULT_DATASET_SUMMARY = PROJECT_ROOT / "data/manifests/cwe_119_imbalance_generation_summary.csv"
DEFAULT_NOISE_SUMMARY = PROJECT_ROOT / "data/manifests/noise_ratio_preserved_generation_summary.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/model_experiments/analysis/cwe119_true_pr_auc_rq_analysis_v5"
)

MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "lightgbm",
]
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
NOISY_CONDITIONS = CONDITIONS[1:]
METRICS = ["mcc", "f1", "precision", "recall", "pr_auc"]
RANK_METRICS = [*METRICS, "balanced_accuracy"]
PAIR_KEYS = ["model_family", "train_ratio", "eval_ratio", "seed"]
CELL_KEYS = [*PAIR_KEYS, "condition"]
T_CRITICAL_DF4 = float(stats.t.ppf(0.975, 4))

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}
CONDITION_DISPLAY = {
    "random_flip_10": "Paired flip 10%",
    "random_flip_20": "Paired flip 20%",
    "random_fp_10": "Random FP 10%",
    "random_fp_20": "Random FP 20%",
    "heuristic_fp_10": "Heuristic FP 10%",
    "heuristic_fp_20": "Heuristic FP 20%",
}


class AnalysisFailure(RuntimeError):
    """Raised when validated input invariants do not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisFailure(message)


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
        return str(path.resolve())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def validate_input(path: Path, report_path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing validated combined results: {path}")
    require(report_path.is_file(), f"Missing independent validation report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("status") == "passed", "Independent validation did not pass.")
    output_record = report.get("generated_outputs", {}).get(path.name)
    require(isinstance(output_record, dict), "Validation report does not record the input CSV.")
    require(output_record.get("rows") == 4200, "Validation report has the wrong test-row count.")
    require(output_record.get("sha256") == sha256_file(path), "Validated input checksum changed.")

    frame = pd.read_csv(path)
    required_columns = {
        "model_family",
        "train_ratio",
        "eval_ratio",
        "seed",
        "condition",
        "stage",
        "split",
        "average_precision",
        "balanced_accuracy",
        *METRICS,
    }
    missing = sorted(required_columns - set(frame.columns))
    require(not missing, f"Validated matrix is missing columns: {missing}")
    require(len(frame) == 4200, f"Expected 4,200 test cells. Observed {len(frame)}.")
    require(not frame.duplicated(CELL_KEYS).any(), "Validated matrix has duplicate test cells.")
    require(set(frame["model_family"]) == set(MODELS), "Model domain is incomplete.")
    require(set(frame["train_ratio"]) == set(TRAIN_RATIOS), "Train-ratio domain is incomplete.")
    require(set(frame["eval_ratio"]) == set(TEST_RATIOS), "Test-ratio domain is incomplete.")
    require(set(frame["seed"]) == set(SEEDS), "Seed domain is incomplete.")
    require(set(frame["condition"]) == set(CONDITIONS), "Condition domain is incomplete.")
    require((frame["stage"] == "test").all(), "Combined matrix contains non-test rows.")
    require((frame["split"] == "test").all(), "Combined matrix contains non-test splits.")
    return frame


def validate_generation_provenance(
    dataset_summary_path: Path,
    noise_summary_path: Path,
) -> dict[str, Any]:
    """Validate the generation facts cited by the findings narrative."""
    require(dataset_summary_path.is_file(), f"Missing dataset summary: {dataset_summary_path}")
    require(noise_summary_path.is_file(), f"Missing noise summary: {noise_summary_path}")

    dataset = pd.read_csv(dataset_summary_path)
    dataset_keys = ["ratio", "seed", "split"]
    require(len(dataset) == 80, f"Expected 80 dataset-summary rows. Observed {len(dataset)}.")
    require(not dataset.duplicated(dataset_keys).any(), "Dataset summary has duplicate cells.")
    require((dataset["target_cwe"] == "CWE-119").all(), "Dataset summary contains another CWE.")
    require(set(dataset["seed"]) == set(SEEDS), "Dataset-summary seed domain is incomplete.")
    require(
        set(dataset["ratio"]) == set(TEST_RATIOS),
        "Dataset-summary ratio domain is incomplete.",
    )
    test = dataset[dataset["split"] == "test"]
    train = dataset[dataset["split"] == "train"]
    valid = dataset[dataset["split"] == "valid"]
    require(len(test) == 30, "Expected 30 test dataset-summary rows.")
    require(len(train) == 25 and len(valid) == 25, "Expected 25 train and validation rows.")
    require((test["positive_rows"] == 21).all(), "A test benchmark does not retain 21 positives.")
    require((train["positive_rows"] == 679).all(), "A training benchmark does not retain 679 positives.")
    require((valid["positive_rows"] == 19).all(), "A validation benchmark does not retain 19 positives.")

    noise = pd.read_csv(noise_summary_path)
    noise_keys = ["ratio", "seed", "condition"]
    require(len(noise) == 175, f"Expected 175 noise-summary rows. Observed {len(noise)}.")
    require(not noise.duplicated(noise_keys).any(), "Noise summary has duplicate cells.")
    require((noise["target_cwe"] == "CWE-119").all(), "Noise summary contains another CWE.")
    require(set(noise["ratio"]) == set(TRAIN_RATIOS), "Noise-summary ratio domain is incomplete.")
    require(set(noise["seed"]) == set(SEEDS), "Noise-summary seed domain is incomplete.")
    require(set(noise["condition"]) == set(CONDITIONS), "Noise-summary condition domain is incomplete.")
    require(noise["validation_clean"].astype(bool).all(), "A validation label set is not clean.")
    require(noise["ratio_preserved"].astype(bool).all(), "A generated training-label ratio is not preserved.")

    boundary = noise[
        (noise["ratio"] == "90/10")
        & (noise["condition"] == "random_flip_20")
    ]
    require(len(boundary) == 5, "Expected five 90:10 paired-flip-20 boundary rows.")
    require((boundary["clean_train_positive_rows"] == 679).all(), "Boundary clean-positive count changed.")
    require((boundary["flips_1_to_0"] == 679).all(), "Boundary does not flip all 679 positives.")

    false_positive = noise[noise["noise_type"].isin(["random_fp", "heuristic_fp"])]
    require(
        (false_positive["compensation_benign_rows"] > 0).all(),
        "A false-positive condition lacks compensation rows.",
    )
    largest_fp = false_positive[
        (false_positive["ratio"] == "90/10")
        & (false_positive["noise_level"] == 0.2)
    ]
    require(len(largest_fp) == 10, "Expected ten largest false-positive condition rows.")
    require((largest_fp["compensation_benign_rows"] == 13580).all(), "Largest compensation count changed.")
    require((largest_fp["final_train_rows"] == 20370).all(), "Largest final training size changed.")

    return {
        "dataset_generation_summary": {
            "path": relative(dataset_summary_path),
            "sha256": sha256_file(dataset_summary_path),
            "rows": len(dataset),
            "verified_positive_counts": {"train": 679, "validation": 19, "test": 21},
        },
        "noise_generation_summary": {
            "path": relative(noise_summary_path),
            "sha256": sha256_file(noise_summary_path),
            "rows": len(noise),
            "verified_boundary_positive_flips": 679,
            "verified_largest_compensation_rows": 13580,
            "verified_largest_final_training_rows": 20370,
        },
    }


def build_paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[frame["condition"] == "clean"][PAIR_KEYS + METRICS].copy()
    require(len(clean) == 600, f"Expected 600 clean cells. Observed {len(clean)}.")
    require(not clean.duplicated(PAIR_KEYS).any(), "Clean pairing keys are duplicated.")
    clean = clean.rename(columns={metric: f"{metric}_clean" for metric in METRICS})
    noisy = frame[frame["condition"] != "clean"].copy()
    paired = noisy.merge(clean, on=PAIR_KEYS, how="left", validate="many_to_one")
    require(len(paired) == 3600, f"Expected 3,600 paired noisy cells. Observed {len(paired)}.")
    require(not paired[[f"{metric}_clean" for metric in METRICS]].isna().any().any(), "A noisy cell lacks its matched clean cell.")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_clean"]
    paired["noise_family"] = paired["condition"].map(
        {
            "random_flip_10": "paired_flip",
            "random_flip_20": "paired_flip",
            "random_fp_10": "random_false_positive",
            "random_fp_20": "random_false_positive",
            "heuristic_fp_10": "heuristic_false_positive",
            "heuristic_fp_20": "heuristic_false_positive",
        }
    )
    paired["requested_rate_percent"] = paired["condition"].str.extract(
        r"_(10|20)$",
        expand=False,
    ).astype(int)
    require(not paired[["noise_family", "requested_rate_percent"]].isna().any().any(), "Could not decode a noise condition.")
    return paired


def summarize_seed_means(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    count_groups = [*group_columns, "seed"]
    counts = frame.groupby(count_groups, dropna=False).size()
    require(counts.nunique() == 1, f"Unbalanced seed cells for groups {group_columns}.")
    cells_per_seed = int(counts.iloc[0])
    seed_means = (
        frame.groupby(count_groups, dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = seed_means.groupby(group_columns, dropna=False, sort=False)
    else:
        grouped = [((), seed_means)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        require(set(group["seed"]) == set(SEEDS), f"Missing seeds for group {key}.")
        for value_column in value_columns:
            values = group[value_column].to_numpy(dtype=float)
            mean = float(values.mean())
            seed_sd = float(values.std(ddof=1))
            half_width = T_CRITICAL_DF4 * seed_sd / np.sqrt(len(values))
            row = dict(zip(group_columns, key))
            row.update(
                {
                    "metric": value_column,
                    "mean": mean,
                    "seed_sd": seed_sd,
                    "ci95_half_width": half_width,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "minimum_seed_mean": float(values.min()),
                    "maximum_seed_mean": float(values.max()),
                    "n_seeds": len(values),
                    "cells_averaged_per_seed": cells_per_seed,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_contrasts(
    frame: pd.DataFrame,
    *,
    dimension: str,
    low: Any,
    high: Any,
    group_columns: list[str],
    value_columns: list[str],
    contrast_name: str,
) -> pd.DataFrame:
    endpoint_counts = frame.groupby(
        [*group_columns, "seed", dimension],
        dropna=False,
    ).size()
    require(
        endpoint_counts.nunique() == 1,
        f"Unbalanced endpoint cells for contrast {contrast_name}.",
    )
    cells_per_seed_at_endpoint = int(endpoint_counts.iloc[0])
    seed_dimensions = (
        frame.groupby([*group_columns, "seed", dimension], dropna=False, sort=False)[value_columns]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for value_column in value_columns:
        pivot = seed_dimensions.pivot(
            index=[*group_columns, "seed"],
            columns=dimension,
            values=value_column,
        )
        require(low in pivot and high in pivot, f"Missing endpoints for {contrast_name}.")
        contrast = (pivot[high] - pivot[low]).rename("contrast").reset_index()
        summary = summarize_seed_means(
            contrast,
            group_columns=group_columns,
            value_columns=["contrast"],
        )
        summary["metric"] = value_column
        summary["contrast"] = contrast_name
        summary["low_level"] = low
        summary["high_level"] = high
        summary = summary.rename(
            columns={"cells_averaged_per_seed": "contrast_values_per_seed"}
        )
        summary["cells_averaged_per_seed_at_each_endpoint"] = (
            cells_per_seed_at_endpoint
        )
        rows.extend(summary.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_boundary_sensitivity(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_columns = [f"delta_{metric}" for metric in METRICS]
    for scope, subset in [
        ("all_train_ratios", paired),
        ("excluding_90/10", paired[paired["train_ratio"] != "90/10"]),
    ]:
        summary = summarize_seed_means(
            subset,
            group_columns=["condition"],
            value_columns=value_columns,
        )
        summary.insert(1, "scope", scope)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def build_metric_rank_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rank_frames = []
    group_columns = ["train_ratio", "eval_ratio", "seed", "condition"]
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        ranked = group[["model_family", *group_columns]].copy()
        for metric in RANK_METRICS:
            ranked[f"{metric}_rank"] = group[metric].rank(
                ascending=False,
                method="average",
            )
        rank_frames.append(ranked)
    ranks = pd.concat(rank_frames, ignore_index=True)
    rows = []
    for left, right in itertools.combinations(RANK_METRICS, 2):
        rows.append(
            {
                "metric_left": left,
                "metric_right": right,
                "spearman_rank_correlation": ranks[f"{left}_rank"].corr(
                    ranks[f"{right}_rank"],
                    method="spearman",
                ),
                "ranked_model_rows": len(ranks),
                "factor_cells": 1050,
                "interpretation": "descriptive. Factor cells are repeated, not independent replications",
            }
        )
    return pd.DataFrame(rows)


def build_winner_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["train_ratio", "eval_ratio", "seed", "condition"]
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=False):
        winner_sets = {
            metric: set(
                group.loc[
                    np.isclose(
                        group[metric],
                        group[metric].max(),
                        rtol=0.0,
                        atol=1e-15,
                    ),
                    "model_family",
                ]
            )
            for metric in RANK_METRICS
        }
        row = dict(zip(group_columns, key))
        row.update(
            {
                "mcc_winner_count": len(winner_sets["mcc"]),
                "pr_auc_winner_count": len(winner_sets["pr_auc"]),
                "mcc_pr_auc_winner_sets_equal": winner_sets["mcc"] == winner_sets["pr_auc"],
                "mcc_pr_auc_share_any_winner": bool(
                    winner_sets["mcc"] & winner_sets["pr_auc"]
                ),
                "mcc_pr_auc_same_unique_winner": (
                    len(winner_sets["mcc"]) == 1
                    and len(winner_sets["pr_auc"]) == 1
                    and winner_sets["mcc"] == winner_sets["pr_auc"]
                ),
                "mcc_winners": ",".join(sorted(winner_sets["mcc"])),
                "pr_auc_winners": ",".join(sorted(winner_sets["pr_auc"])),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def get_value(
    frame: pd.DataFrame,
    filters: dict[str, Any],
    *,
    column: str = "mean",
) -> float:
    selected_rows = frame
    for name, value in filters.items():
        selected_rows = selected_rows[selected_rows[name] == value]
    require(len(selected_rows) == 1, f"Expected one analysis row for {filters}. Got {len(selected_rows)}.")
    return float(selected_rows.iloc[0][column])


def format_effect(mean: float, seed_sd: float) -> str:
    return f"{mean:+.3f} (seed SD {seed_sd:.3f})"


def build_findings(
    *,
    rq1: pd.DataFrame,
    severity_contrasts: pd.DataFrame,
    rq2_clean_models: pd.DataFrame,
    rq2_train: pd.DataFrame,
    rq2_train_model: pd.DataFrame,
    rq2_test: pd.DataFrame,
    train_contrasts: pd.DataFrame,
    test_contrasts: pd.DataFrame,
    rq3: pd.DataFrame,
    interaction: pd.DataFrame,
    sensitivity: pd.DataFrame,
    winner_agreement: pd.DataFrame,
) -> str:
    def effect(table: pd.DataFrame, filters: dict[str, Any]) -> tuple[float, float]:
        return (
            get_value(table, filters),
            get_value(table, filters, column="seed_sd"),
        )

    flip20_mcc = effect(rq1, {"condition": "random_flip_20", "metric": "delta_mcc"})
    flip20_pr = effect(rq1, {"condition": "random_flip_20", "metric": "delta_pr_auc"})
    h10_mcc = effect(rq1, {"condition": "heuristic_fp_10", "metric": "delta_mcc"})
    h10_precision = effect(
        rq1,
        {"condition": "heuristic_fp_10", "metric": "delta_precision"},
    )
    h10_recall = effect(rq1, {"condition": "heuristic_fp_10", "metric": "delta_recall"})
    paired_severity_mcc = effect(
        severity_contrasts,
        {"noise_family": "paired_flip", "metric": "delta_mcc"},
    )
    paired_severity_pr = effect(
        severity_contrasts,
        {"noise_family": "paired_flip", "metric": "delta_pr_auc"},
    )

    train_precision = effect(train_contrasts, {"metric": "precision"})
    train_recall = effect(train_contrasts, {"metric": "recall"})
    train_pr = effect(train_contrasts, {"metric": "pr_auc"})
    test_precision = effect(test_contrasts, {"metric": "precision"})
    test_recall = effect(test_contrasts, {"metric": "recall"})
    test_pr = effect(test_contrasts, {"metric": "pr_auc"})

    rf60_precision = effect(
        rq2_train_model,
        {"model_family": "random_forest", "train_ratio": "60/40", "metric": "precision"},
    )
    rf90_precision = effect(
        rq2_train_model,
        {"model_family": "random_forest", "train_ratio": "90/10", "metric": "precision"},
    )
    rf60_recall = effect(
        rq2_train_model,
        {"model_family": "random_forest", "train_ratio": "60/40", "metric": "recall"},
    )
    rf90_recall = effect(
        rq2_train_model,
        {"model_family": "random_forest", "train_ratio": "90/10", "metric": "recall"},
    )

    i_flip20_mcc = effect(
        interaction,
        {"condition": "random_flip_20", "metric": "delta_mcc"},
    )
    i_flip20_pr = effect(
        interaction,
        {"condition": "random_flip_20", "metric": "delta_pr_auc"},
    )
    flip20_ex90_mcc = effect(
        sensitivity,
        {
            "condition": "random_flip_20",
            "scope": "excluding_90/10",
            "metric": "delta_mcc",
        },
    )
    flip20_60_mcc = effect(
        rq3,
        {"condition": "random_flip_20", "train_ratio": "60/40", "metric": "delta_mcc"},
    )
    flip20_90_mcc = effect(
        rq3,
        {"condition": "random_flip_20", "train_ratio": "90/10", "metric": "delta_mcc"},
    )

    clean_pr_rows = rq2_clean_models[rq2_clean_models["metric"] == "pr_auc"]
    clean_pr_leader = clean_pr_rows.sort_values("mean", ascending=False).iloc[0]
    same_unique = int(winner_agreement["mcc_pr_auc_same_unique_winner"].sum())
    any_shared = int(winner_agreement["mcc_pr_auc_share_any_winner"].sum())

    return f"""# Seed-aware Research-Question Evidence (Validated True PR-AUC)

## Recommended research questions

1. **RQ1 — Noise mechanism and severity:** How do the type and requested rate of training-label noise change vulnerability-detection performance relative to clean training?
2. **RQ2 — Class balance:** How do the benign:vulnerable ratios in training and testing affect clean-model ranking and default-threshold behavior?
3. **RQ3 — Interaction:** How does the training class ratio modify the effect of each label-noise mechanism?

Model-family differences and metric disagreement are reported within these questions. They do not need a fourth research question.

## Statistical unit and uncertainty

Every noisy cell is paired with the clean cell having the same model, train ratio, test ratio, and seed. For each reported factorial-grid summary, the fixed model/ratio cells are first averaged within each seed. The primary summary is the mean and standard deviation (SD) across the **five seed-level means**. The CSV files also provide the seed range and a descriptive Student-$t$ interval ($df=4$), but this interval is not used as evidence of statistical significance. We do not obtain a narrow interval by pretending that 150 or 600 correlated grid cells are independent runs, and we do not report $p$-values with only five benchmark versions.

## RQ1: noise mechanism and severity

- Requested 20% paired flip is the largest overall degradation: $\\Delta$MCC {format_effect(*flip20_mcc)} and $\\Delta$PR-AUC {format_effect(*flip20_pr)}.
- The requested rate matters: relative to 10%, the 20% paired-flip condition changes the already-paired effect by {format_effect(*paired_severity_mcc)} for MCC and {format_effect(*paired_severity_pr)} for PR-AUC.
- Heuristic FP 10% is an operating-point shift, not evidence that wrong labels are beneficial: $\\Delta$precision {format_effect(*h10_precision)}, $\\Delta$recall {format_effect(*h10_recall)}, and $\\Delta$MCC {format_effect(*h10_mcc)}. This condition also adds benign compensation rows, so its effect combines relabeling with a larger training set.
- Random false-positive injection mainly reduces precision, while paired flip reduces both ranking and thresholded performance.

## RQ2: class balance

- Moving clean training from 60:40 to 90:10 changes the average operating point: precision {format_effect(*train_precision)} and recall {format_effect(*train_recall)}. The corresponding PR-AUC contrast is only {format_effect(*train_pr)}, so the large threshold shift is not a comparable ranking-quality shift.
- Random Forest is the clearest model-specific case: mean clean precision changes from {rf60_precision[0]:.3f} to {rf90_precision[0]:.3f}, while recall changes from {rf60_recall[0]:.3f} to {rf90_recall[0]:.3f}.
- Moving the clean test ratio from 50:50 to 90:10 changes precision by {format_effect(*test_precision)} and PR-AUC by {format_effect(*test_pr)}. Recall changes by {format_effect(*test_recall)} because all test-ratio versions retain the same 21 positive examples. The test-ratio manipulation changes the benign examples and positive prevalence.
- With corrected trapezoidal PR-AUC, the clean factorial-grid leader remains {MODEL_DISPLAY[str(clean_pr_leader['model_family'])]} at {float(clean_pr_leader['mean']):.3f}.

## RQ3: noise-by-imbalance interaction

- For requested 20% paired flip, mean $\\Delta$MCC changes from {format_effect(*flip20_60_mcc)} at 60:40 to {format_effect(*flip20_90_mcc)} at 90:10. The paired 90:10-minus-60:40 interaction contrast is {format_effect(*i_flip20_mcc)} for MCC and {format_effect(*i_flip20_pr)} for PR-AUC.
- The 90:10, 20% paired-flip cell is an intentional boundary case in which all 679 clean vulnerable training labels are flipped. The conclusion does not depend only on that cell: after excluding 90:10, paired flip 20% still has mean $\\Delta$MCC {format_effect(*flip20_ex90_mcc)}.
- False-positive conditions must be interpreted with their compensation-row mechanism. The growing training-size multiplier is part of the condition and prevents a pure label-noise causal interpretation.

## Secondary metric observation

MCC and PR-AUC have the same unique winning model in {same_unique}/1050 factor cells ({same_unique / 1050:.1%}). Their tied winner sets share at least one model in {any_shared}/1050 cells ({any_shared / 1050:.1%}). This is useful supporting evidence that ranking and default-threshold metrics answer different questions, but it is descriptive and need not become a separate RQ.

## Interpretation rule

All means are equal-weighted summaries over the stated factorial grid, not estimates of performance under an unspecified deployment prevalence. PR-AUC comparisons across test ratios must always name the test prevalence.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=DEFAULT_DATASET_SUMMARY,
    )
    parser.add_argument(
        "--noise-summary",
        type=Path,
        default=DEFAULT_NOISE_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    partial_dir = args.output_dir.with_name(args.output_dir.name + ".partial")
    require(not args.output_dir.exists(), f"Refusing to overwrite: {args.output_dir}")
    require(not partial_dir.exists(), f"Partial output already exists: {partial_dir}")

    frame = validate_input(args.input, args.validation_report)
    generation_provenance = validate_generation_provenance(
        args.dataset_summary,
        args.noise_summary,
    )
    paired = build_paired_deltas(frame)
    clean = frame[frame["condition"] == "clean"].copy()
    noisy = frame[frame["condition"] != "clean"].copy()
    delta_columns = [f"delta_{metric}" for metric in METRICS]

    rq1 = summarize_seed_means(
        paired,
        group_columns=["condition"],
        value_columns=delta_columns,
    )
    rq1_by_model = summarize_seed_means(
        paired,
        group_columns=["model_family", "condition"],
        value_columns=delta_columns,
    )
    rq1_by_test_ratio = summarize_seed_means(
        paired,
        group_columns=["condition", "eval_ratio"],
        value_columns=delta_columns,
    )
    severity_contrasts = summarize_seed_contrasts(
        paired,
        dimension="requested_rate_percent",
        low=10,
        high=20,
        group_columns=["noise_family"],
        value_columns=delta_columns,
        contrast_name="noise_effect_at_20_percent_minus_noise_effect_at_10_percent",
    )
    rq2_clean_models = summarize_seed_means(
        clean,
        group_columns=["model_family"],
        value_columns=METRICS,
    )
    rq2_train = summarize_seed_means(
        clean,
        group_columns=["train_ratio"],
        value_columns=METRICS,
    )
    rq2_train_model = summarize_seed_means(
        clean,
        group_columns=["model_family", "train_ratio"],
        value_columns=METRICS,
    )
    rq2_test = summarize_seed_means(
        clean,
        group_columns=["eval_ratio"],
        value_columns=METRICS,
    )
    rq2_test_model = summarize_seed_means(
        clean,
        group_columns=["model_family", "eval_ratio"],
        value_columns=METRICS,
    )
    rq2_train_test_joint = summarize_seed_means(
        clean,
        group_columns=["train_ratio", "eval_ratio"],
        value_columns=[*METRICS, "balanced_accuracy"],
    )
    train_contrasts = summarize_seed_contrasts(
        clean,
        dimension="train_ratio",
        low="60/40",
        high="90/10",
        group_columns=[],
        value_columns=METRICS,
        contrast_name="clean_train_90/10_minus_60/40",
    )
    train_model_contrasts = summarize_seed_contrasts(
        clean,
        dimension="train_ratio",
        low="60/40",
        high="90/10",
        group_columns=["model_family"],
        value_columns=METRICS,
        contrast_name="clean_train_90/10_minus_60/40",
    )
    test_contrasts = summarize_seed_contrasts(
        clean,
        dimension="eval_ratio",
        low="50/50",
        high="90/10",
        group_columns=[],
        value_columns=METRICS,
        contrast_name="clean_test_90/10_minus_50/50",
    )
    rq3 = summarize_seed_means(
        paired,
        group_columns=["condition", "train_ratio"],
        value_columns=delta_columns,
    )
    rq3_by_model = summarize_seed_means(
        paired,
        group_columns=["model_family", "condition", "train_ratio"],
        value_columns=delta_columns,
    )
    rq3_train_test_joint = summarize_seed_means(
        paired,
        group_columns=["condition", "train_ratio", "eval_ratio"],
        value_columns=delta_columns,
    )
    rq3_noisy_absolute_train_test_joint = summarize_seed_means(
        noisy,
        group_columns=["condition", "train_ratio", "eval_ratio"],
        value_columns=[*METRICS, "balanced_accuracy"],
    )
    rq3_test_endpoint_interactions = summarize_seed_contrasts(
        paired,
        dimension="eval_ratio",
        low="50/50",
        high="90/10",
        group_columns=["condition", "train_ratio"],
        value_columns=delta_columns,
        contrast_name="noise_delta_at_test_90/10_minus_noise_delta_at_test_50/50",
    )
    interaction = summarize_seed_contrasts(
        paired,
        dimension="train_ratio",
        low="60/40",
        high="90/10",
        group_columns=["condition"],
        value_columns=delta_columns,
        contrast_name="noise_delta_at_90/10_minus_noise_delta_at_60/40",
    )
    sensitivity = build_boundary_sensitivity(paired)
    rank_correlations = build_metric_rank_correlations(frame)
    winner_agreement = build_winner_agreement(frame)

    require(
        np.allclose(
            rq2_test[rq2_test["metric"] == "recall"]["mean"],
            rq2_test[rq2_test["metric"] == "recall"]["mean"].iloc[0],
            rtol=0.0,
            atol=1e-15,
        ),
        "Clean recall unexpectedly changes across test-ratio summaries.",
    )

    partial_dir.mkdir(parents=True)
    outputs = {
        "paired_noisy_minus_clean_true_pr_auc.csv": paired,
        "rq1_noise_condition_seed_aware.csv": rq1,
        "rq1_noise_by_model_seed_aware.csv": rq1_by_model,
        "rq1_noise_by_test_ratio_seed_aware.csv": rq1_by_test_ratio,
        "rq1_noise_severity_contrasts.csv": severity_contrasts,
        "rq2_clean_model_seed_aware.csv": rq2_clean_models,
        "rq2_clean_train_ratio_seed_aware.csv": rq2_train,
        "rq2_clean_train_ratio_by_model_seed_aware.csv": rq2_train_model,
        "rq2_clean_test_ratio_seed_aware.csv": rq2_test,
        "rq2_clean_test_ratio_by_model_seed_aware.csv": rq2_test_model,
        "rq2_clean_train_test_joint_seed_aware.csv": rq2_train_test_joint,
        "rq2_clean_train_endpoint_contrasts.csv": train_contrasts,
        "rq2_clean_train_endpoint_contrasts_by_model.csv": train_model_contrasts,
        "rq2_clean_test_endpoint_contrasts.csv": test_contrasts,
        "rq3_noise_by_train_ratio_seed_aware.csv": rq3,
        "rq3_noise_by_train_ratio_and_model_seed_aware.csv": rq3_by_model,
        "rq3_noise_by_train_test_ratio_seed_aware.csv": rq3_train_test_joint,
        "rq3_noisy_absolute_by_train_test_ratio_seed_aware.csv": rq3_noisy_absolute_train_test_joint,
        "rq3_noise_test_endpoint_interactions.csv": rq3_test_endpoint_interactions,
        "rq3_noise_imbalance_interaction_contrasts.csv": interaction,
        "rq3_boundary_sensitivity.csv": sensitivity,
        "metric_rank_correlations_true_pr_auc.csv": rank_correlations,
        "mcc_pr_auc_winner_agreement.csv": winner_agreement,
    }
    for filename, output in outputs.items():
        write_csv(partial_dir / filename, output)

    findings = build_findings(
        rq1=rq1,
        severity_contrasts=severity_contrasts,
        rq2_clean_models=rq2_clean_models,
        rq2_train=rq2_train,
        rq2_train_model=rq2_train_model,
        rq2_test=rq2_test,
        train_contrasts=train_contrasts,
        test_contrasts=test_contrasts,
        rq3=rq3,
        interaction=interaction,
        sensitivity=sensitivity,
        winner_agreement=winner_agreement,
    )
    (partial_dir / "findings.md").write_text(findings, encoding="utf-8")

    generated = {}
    for path in sorted(partial_dir.iterdir()):
        if path.is_file():
            generated[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "status": "complete",
        "analysis": "seed-aware research-question evidence",
        "input": relative(args.input),
        "input_sha256": sha256_file(args.input),
        "independent_validation_report": relative(args.validation_report),
        "independent_validation_report_sha256": sha256_file(args.validation_report),
        "generation_provenance": generation_provenance,
        "analysis_script": relative(Path(__file__)),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "uncertainty": {
            "unit": "seed-level factorial-grid mean",
            "n_seeds": 5,
            "primary_summary": "mean, sample SD, and range across seed-level means",
            "descriptive_interval": "two-sided 95% Student-t interval",
            "degrees_of_freedom": 4,
            "t_critical": T_CRITICAL_DF4,
            "significance_claimed": False,
            "p_values_reported": False,
        },
        "counts": {
            "validated_test_cells": len(frame),
            "clean_cells": len(clean),
            "noisy_cells": len(noisy),
            "paired_noisy_cells": len(paired),
        },
        "generated_outputs": generated,
    }
    (partial_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial_dir.rename(args.output_dir)
    print(f"Research-question analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()

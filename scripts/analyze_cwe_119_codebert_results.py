#!/usr/bin/env python3
"""Compare independently validated CodeBERT with the frozen four-model study.

Inputs are read-only. Summaries retain every train/test prevalence setting.
Standard deviations describe five correlated sampling versions, not CIs.
Partial validation produces diagnostic CSVs only, never publication assets.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "results/model_experiments/analysis/cwe119_true_pr_auc_validation_v1"
BASELINE_CSV = BASELINE_DIR / "combined_test_results_true_pr_auc_v1.csv"
MODELS = ["logistic_regression", "linear_svm", "random_forest", "lightgbm"]
ALL_MODELS = [*MODELS, "codebert"]
TRAIN_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
TEST_RATIOS = ["50/50", "60/40", "70/30", "75/25", "80/20", "90/10"]
SEEDS = [1, 2, 3, 4, 5]
CONDITIONS = ["clean", "random_flip_10", "random_flip_20", "random_fp_10",
              "random_fp_20", "heuristic_fp_10", "heuristic_fp_20"]
METRICS = ["pr_auc", "average_precision", "mcc", "f1", "precision", "recall", "balanced_accuracy"]
MATCH_KEYS = ["train_ratio", "eval_ratio", "seed", "condition"]
CELL_KEYS = ["model_family", *MATCH_KEYS]
PAIR_KEYS = ["model_family", "train_ratio", "eval_ratio", "seed"]
SUMMARY_KEYS = ["model_family", "train_ratio", "eval_ratio", "condition"]
DISPLAY = {"logistic_regression": "Logistic Regression", "linear_svm": "Linear SVM",
           "random_forest": "Random Forest", "lightgbm": "LightGBM", "codebert": "CodeBERT"}
NOISE_DISPLAY = {"random_flip_10": "Paired flip 10%", "random_flip_20": "Paired flip 20%",
                 "random_fp_10": "Random FP 10%", "random_fp_20": "Random FP 20%",
                 "heuristic_fp_10": "Heuristic FP 10%", "heuristic_fp_20": "Heuristic FP 20%"}


class AnalysisFailure(RuntimeError):
    """An input or exact-match invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_keys(models: list[str]) -> set[tuple[Any, ...]]:
    return set(itertools.product(models, TRAIN_RATIOS, TEST_RATIOS, SEEDS, CONDITIONS))


def validate_test_frame(frame: pd.DataFrame, models: list[str], *, complete: bool) -> pd.DataFrame:
    """Reject duplicates, unknown keys, wrong counts and invalid metric ranges."""
    required = set(CELL_KEYS + METRICS + ["stage", "split", "n_rows", "positive_rows", "benign_rows"])
    require(not (required - set(frame.columns)), f"Missing metric columns: {sorted(required - set(frame.columns))}")
    require(len(frame) > 0, "No evaluated CodeBERT test cells are available.")
    result = frame.copy()
    seeds = pd.to_numeric(result["seed"], errors="coerce")
    require(seeds.notna().all() and (seeds % 1 == 0).all(), "Dataset seeds must be integers.")
    result["seed"] = seeds.astype(int)
    require(not result[CELL_KEYS].isna().any().any(), "Null evaluation keys.")
    require(not result.duplicated(CELL_KEYS).any(), "Duplicate evaluation keys.")
    require((result["stage"] == "test").all() and (result["split"] == "test").all(), "Non-test rows in test matrix.")
    observed = set(result[CELL_KEYS].itertuples(index=False, name=None))
    expected = expected_keys(models)
    require(observed <= expected, f"Unexpected evaluation keys: {sorted(observed - expected)[:3]}")
    if complete:
        require(observed == expected, f"Incomplete evaluation matrix: missing {len(expected - observed)} cells.")
    for metric in METRICS:
        values = pd.to_numeric(result[metric], errors="coerce")
        require(np.isfinite(values).all(), f"Nonfinite {metric} values.")
        lower = -1 if metric == "mcc" else 0
        require(values.between(lower - 1e-12, 1 + 1e-12).all(), f"Out-of-range {metric} values.")
        result[metric] = values
    counts = {"50/50": 42, "60/40": 53, "70/30": 70, "75/25": 84, "80/20": 105, "90/10": 210}
    require((result["positive_rows"] == 21).all(), "Every canonical test cohort must retain 21 reference positives.")
    require((result["n_rows"] == result["eval_ratio"].map(counts)).all(), "Unexpected canonical test-cohort sizes.")
    require((result["benign_rows"] == result["n_rows"] - 21).all(), "Inconsistent benign/test row counts.")
    return result.sort_values(CELL_KEYS).reset_index(drop=True)


def validation_hashes(report: dict[str, Any]) -> dict[str, str]:
    """Accept the validator's checked-file map or its generated-output records."""
    records = report.get("checked_file_sha256", report.get("generated_outputs", {}))
    require(isinstance(records, dict), "Independent validation lacks a file-hash mapping.")
    return {name: record.get("sha256", "") if isinstance(record, dict) else record
            for name, record in records.items()}


def load_codebert(results_dir: Path, *, allow_partial: bool) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    report_path = results_dir / "validation_report.json"
    require(report_path.is_file(), "Run the independent CodeBERT validator before analysis.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    complete = report.get("status") == "full_passed"
    require(complete or (allow_partial and report.get("status") == "partial_passed"),
            "Independent CodeBERT validation must be full_passed (or partial_passed with --allow-partial).")
    hashes = validation_hashes(report)
    require("metrics.csv" in hashes and "manifest.json" in hashes, "Validation does not bind metrics.csv and manifest.json hashes.")
    for filename, expected in hashes.items():
        path = results_dir / filename
        require(path.resolve().is_relative_to(results_dir.resolve()), "Unsafe path in validation hash records.")
        require(path.is_file() and sha256_file(path) == expected, f"Validated file changed or is missing: {filename}")
    metrics = pd.read_csv(results_dir / "metrics.csv")
    require("stage" in metrics, "CodeBERT metrics lack stage.")
    require(set(metrics["stage"]) <= {"validation", "test"}, "Unexpected metric stage.")
    tests = validate_test_frame(metrics[metrics["stage"] == "test"], ["codebert"], complete=complete)
    validation = metrics[metrics["stage"] == "validation"]
    group_keys = ["train_ratio", "seed", "condition"]
    require(not validation.duplicated(group_keys).any(), "Duplicate selected validation groups.")
    require(set(validation["model_family"]) == {"codebert"}, "Non-CodeBERT validation groups.")
    test_groups = set(tests[group_keys].itertuples(index=False, name=None))
    valid_groups = set(validation[group_keys].itertuples(index=False, name=None))
    require(test_groups <= valid_groups, "A test group has no selected validation row.")
    if complete:
        require(len(valid_groups) == 175 and len(validation) == 175, "Full validation requires 175 selected groups.")
        require(test_groups == valid_groups, "Selected validation and test groups differ.")
        require(report.get("completed_groups") == 175 and report.get("test_cells") == 1050,
                "Full validation report does not certify 175 groups / 1,050 test cells.")
    return tests, report, complete


def load_baselines(
    baseline_csv: Path = BASELINE_CSV,
    validation_report: Path = BASELINE_DIR / "validation_report.json",
) -> pd.DataFrame:
    require(validation_report.is_file() and baseline_csv.is_file(), "Baseline validation evidence is missing.")
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    require(report.get("status") == "passed", "Frozen classical validation did not pass.")
    record = report.get("generated_outputs", {}).get(baseline_csv.name, {})
    require(record.get("rows") == 4200 and record.get("sha256") == sha256_file(baseline_csv),
            "Baseline matrix no longer matches its independent validation hash.")
    return validate_test_frame(pd.read_csv(baseline_csv), MODELS, complete=True)


def match_baselines(codebert: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Select precisely the frozen baseline cells evaluated by CodeBERT."""
    require(not codebert.duplicated(MATCH_KEYS).any(), "Duplicate CodeBERT comparison keys.")
    require(not baselines.duplicated(CELL_KEYS).any(), "Duplicate baseline comparison keys.")
    requested = codebert[MATCH_KEYS]
    matched = baselines.merge(requested, on=MATCH_KEYS, how="inner", validate="many_to_one")
    require(len(matched) == len(codebert) * 4, "Missing matched baseline comparisons.")
    require((matched.groupby(MATCH_KEYS)["model_family"].nunique() == 4).all(), "A context lacks one of four baseline families.")
    return matched.sort_values(CELL_KEYS).reset_index(drop=True)


def paired_noise_deltas(frame: pd.DataFrame, *, allow_partial: bool = False) -> tuple[pd.DataFrame, int]:
    require(not frame.duplicated(CELL_KEYS).any(), "Duplicate keys prevent matched noise deltas.")
    clean = frame[frame["condition"] == "clean"][PAIR_KEYS + METRICS]
    clean = clean.rename(columns={metric: f"clean_{metric}" for metric in METRICS})
    noisy = frame[frame["condition"] != "clean"][CELL_KEYS + METRICS]
    joined = noisy.merge(clean, on=PAIR_KEYS, how="left", validate="many_to_one", indicator=True)
    missing = int((joined["_merge"] != "both").sum())
    require(allow_partial or not missing, f"Missing matched clean cells for {missing} noisy cells.")
    joined = joined[joined["_merge"] == "both"].drop(columns="_merge")
    for metric in METRICS:
        joined[f"delta_{metric}"] = joined[metric] - joined[f"clean_{metric}"]
    return joined.sort_values(CELL_KEYS).reset_index(drop=True), missing


def paired_model_differences(codebert: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    cb = codebert[MATCH_KEYS + METRICS].rename(columns={metric: f"codebert_{metric}" for metric in METRICS})
    result = matched[CELL_KEYS + METRICS].merge(cb, on=MATCH_KEYS, validate="many_to_one")
    for metric in METRICS:
        result[f"codebert_minus_baseline_{metric}"] = result[f"codebert_{metric}"] - result[metric]
    return result.sort_values(CELL_KEYS).reset_index(drop=True)


def summarize_seeds(frame: pd.DataFrame, metrics: list[str], *, complete: bool) -> pd.DataFrame:
    require(not frame.duplicated(CELL_KEYS).any(), "Duplicate seed cells in summary.")
    groups = frame.groupby(SUMMARY_KEYS, sort=False, dropna=False)
    counts = groups["seed"].agg(n_seeds="nunique", seed_rows="size")
    require((counts["n_seeds"] == counts["seed_rows"]).all(), "A seed is repeated within a summary cell.")
    if complete:
        require((counts["n_seeds"] == 5).all(), "A complete summary cell lacks five seeds.")
    result = groups[metrics].agg(["mean", "std"])
    result.columns = [f"{metric}_{'sd' if statistic == 'std' else statistic}"
                      for metric, statistic in result.columns]
    return result.join(counts["n_seeds"]).reset_index().sort_values(SUMMARY_KEYS).reset_index(drop=True)


def winner_contexts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the four-model analysis's 1e-15 tie tolerance. Contexts remain 1,050."""
    rows = []
    for key, group in frame.groupby(MATCH_KEYS, sort=True):
        require(len(group) == 5 and set(group["model_family"]) == set(ALL_MODELS), "Incomplete five-model winner context.")
        row = dict(zip(MATCH_KEYS, key))
        for metric in ["mcc", "pr_auc"]:
            winners = sorted(group.loc[np.isclose(group[metric], group[metric].max(), rtol=0.0, atol=1e-15), "model_family"])
            row[f"{metric}_winners"] = "|".join(winners)
            row[f"{metric}_unique"] = len(winners) == 1
        row["same_unique_winner"] = row["mcc_unique"] and row["pr_auc_unique"] and row["mcc_winners"] == row["pr_auc_winners"]
        rows.append(row)
    result = pd.DataFrame(rows)
    require(len(result) == 1050, "Expected 1,050 five-model comparison contexts.")
    both_unique = int((result["mcc_unique"] & result["pr_auc_unique"]).sum())
    agreement = int(result["same_unique_winner"].sum())
    return result, {"comparison_contexts": 1050, "both_metrics_unique_contexts": both_unique,
                    "same_unique_winner_contexts": agreement,
                    "agreement_fraction_all_contexts": agreement / 1050,
                    "agreement_fraction_both_unique": agreement / both_unique if both_unique else None,
                    "tie_policy": "Absolute metric tolerance 1e-15, relative tolerance 0, matching the four-model analysis. Tied contexts are not same-unique-winner agreements."}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def write_compact_clean_table(output: Path, summary: pd.DataFrame) -> None:
    table = summary[(summary["train_ratio"] == "75/25") & summary["eval_ratio"].isin(["50/50", "90/10"])]
    require(len(table) == 10, "Compact clean table lacks its ten family/test cells.")
    write_csv(output / "clean_75_25_test_endpoints.csv", table)
    lines = ["# Clean 75:25 training at two test endpoints", "",
             "Means ± sample SD across five dataset seeds. Fixed model optimization seed. Sampling versions reuse the same 21 test positives.", "",
             "| Test ratio | Family | PR-AUC | MCC | F1 | Precision | Recall |", "|---|---|---:|---:|---:|---:|---:|"]
    latex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Test ratio & Family & PR-AUC & MCC & F1 & Precision & Recall \\", r"\midrule"]
    for ratio in ["50/50", "90/10"]:
        for family in ALL_MODELS:
            row = table[(table["eval_ratio"] == ratio) & (table["model_family"] == family)].iloc[0]
            numbers = [f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_sd']:.3f}" for m in ["pr_auc", "mcc", "f1", "precision", "recall"]]
            lines.append("| " + " | ".join([ratio.replace("/", ":"), DISPLAY[family], *numbers]) + " |")
            tex_numbers = [value.replace("±", r"$\pm$") for value in numbers]
            latex.append(" & ".join([ratio.replace("/", ":"), DISPLAY[family], *tex_numbers]) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "clean_75_25_test_endpoints.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "clean_75_25_test_endpoints.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def publication_figures(output: Path, clean: pd.DataFrame, noise: pd.DataFrame, raw: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]

    # Test prevalence is held at 90:10. Each point has its own training ratio.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for family, color in zip(ALL_MODELS, colors):
        cell = clean[(clean["model_family"] == family) & (clean["eval_ratio"] == "90/10")].set_index("train_ratio").loc[TRAIN_RATIOS]
        for ax, metric in zip(axes, ["pr_auc", "mcc"]):
            ax.errorbar(range(5), cell[f"{metric}_mean"], yerr=cell[f"{metric}_sd"],
                        label=DISPLAY[family], color=color, marker="o", markersize=3, capsize=2)
            ax.set_xticks(range(5), [ratio.replace("/", ":") for ratio in TRAIN_RATIOS])
            ax.set_xlabel("Train/validation benign:positive ratio")
            ax.set_ylabel("PR-AUC" if metric == "pr_auc" else "MCC")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Clean training · fixed 90:10 test ratio · mean ± sample SD over five sampling seeds")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"clean_five_models_fixed_90_10.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    cb = noise[(noise["model_family"] == "codebert") & (noise["eval_ratio"] == "90/10")]
    arrays = [cb.pivot(index="condition", columns="train_ratio", values=f"delta_{metric}_mean")
              .reindex(index=CONDITIONS[1:], columns=TRAIN_RATIOS).to_numpy() for metric in ["mcc", "pr_auc"]]
    magnitude = max(0.01, float(np.max(np.abs(np.concatenate([a.ravel() for a in arrays])))))
    for ax, array, metric in zip(axes, arrays, ["MCC", "PR-AUC"]):
        mesh = ax.imshow(array, cmap="RdBu", vmin=-magnitude, vmax=magnitude, aspect="auto")
        ax.set_xticks(range(5), [ratio.replace("/", ":") for ratio in TRAIN_RATIOS])
        ax.set_yticks(range(6), [NOISE_DISPLAY[condition] for condition in CONDITIONS[1:]])
        ax.set_xlabel("Train/validation benign:positive ratio")
        ax.set_title(f"CodeBERT Δ{metric}")
        for i in range(6):
            for j in range(5):
                ax.text(j, i, f"{array[i, j]:+.3f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(array[i, j]) > magnitude * 0.6 else "black")
    fig.colorbar(mesh, ax=axes, label="Matched noisy − clean mean (five seeds)", shrink=0.8)
    fig.suptitle("CodeBERT noise effects · fixed 90:10 test ratio")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"codebert_noise_deltas_fixed_90_10.{extension}", dpi=300)
    plt.close(fig)

    # No averaging over prevalence or training ratios in the operating-point panel.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True, sharex=True, sharey=True)
    conditions = ["clean", "random_flip_20", "heuristic_fp_20"]
    for ax, condition in zip(axes, conditions):
        for family, color in zip(ALL_MODELS, colors):
            cell = raw[(raw["model_family"] == family) & (raw["train_ratio"] == "75/25") &
                       (raw["eval_ratio"] == "90/10") & (raw["condition"] == condition)]
            require(len(cell) == 5, "Operating-point figure lacks five matched sampling seeds.")
            ax.scatter(cell["recall"], cell["precision"], color=color, alpha=0.3, s=15)
            ax.errorbar(cell["recall"].mean(), cell["precision"].mean(),
                        xerr=cell["recall"].std(ddof=1), yerr=cell["precision"].std(ddof=1),
                        color=color, marker="o", markersize=4, capsize=2, label=DISPLAY[family])
        ax.set_title("Clean" if condition == "clean" else NOISE_DISPLAY[condition])
        ax.set_xlabel("Recall")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("75:25 training · 90:10 testing · default thresholds · mean ± sample SD")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"operating_points_five_models_fixed_75_25_90_10.{extension}", dpi=300)
    plt.close(fig)


def protect_output(output: Path, inputs: Path, baseline_csv: Path) -> None:
    resolved = output.resolve()
    protected = [PROJECT_ROOT / "data", PROJECT_ROOT / "scripts", baseline_csv.parent]
    protected.extend(PROJECT_ROOT / "results/model_experiments" / name for name in [
        "logreg_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "linear_svm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "random_forest_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1",
        "lightgbm_balanced_ratio_preserved_noise_mcc_selected_true_pr_auc_v1"])
    require(not any(resolved.is_relative_to(p.resolve()) or p.resolve().is_relative_to(resolved) for p in protected),
            "Output directory overlaps preserved sources/data/classical results.")
    require(resolved != inputs.resolve() and not inputs.resolve().is_relative_to(resolved), "Output would overwrite CodeBERT input directory.")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Output directory must be new or empty.")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return path.name


def analyze(
    results_dir: Path,
    output: Path,
    *,
    allow_partial: bool = False,
    baseline_csv: Path = BASELINE_CSV,
    baseline_validation_report: Path = BASELINE_DIR / "validation_report.json",
) -> dict[str, Any]:
    protect_output(output, results_dir, baseline_csv)
    codebert, report, complete = load_codebert(results_dir, allow_partial=allow_partial)
    baselines = load_baselines(baseline_csv, baseline_validation_report)
    matched = match_baselines(codebert, baselines)
    combined = pd.concat([matched, codebert], ignore_index=True, sort=False)
    noise, missing_clean = paired_noise_deltas(combined, allow_partial=not complete)
    differences = paired_model_differences(codebert, matched)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "" if complete else "diagnostic_"
    write_csv(output / f"{prefix}matched_test_cells.csv", combined.sort_values(CELL_KEYS))
    write_csv(output / f"{prefix}paired_noise_deltas_by_seed.csv", noise)
    write_csv(output / f"{prefix}codebert_minus_baseline_by_seed.csv", differences)
    metadata: dict[str, Any] = {"status": "complete_validated_analysis" if complete else "partial_diagnostics_only",
        "codebert_validation_status": report["status"], "publication_assets_emitted": complete,
        "input_codebert_metrics": {"path": portable_path(results_dir / "metrics.csv"), "sha256": sha256_file(results_dir / "metrics.csv")},
        "input_codebert_validation": {"path": portable_path(results_dir / "validation_report.json"), "sha256": sha256_file(results_dir / "validation_report.json")},
        "input_frozen_classical": {"path": portable_path(baseline_csv), "sha256": sha256_file(baseline_csv)},
        "codebert_test_cells": len(codebert), "matched_classical_test_cells": len(matched),
        "combined_test_cells": len(combined), "paired_noisy_test_cells": len(noise),
        "noisy_cells_without_clean_diagnostic_reference": missing_clean,
        "summary_policy": "Means and sample SD (ddof=1) across five seeds at each exact model/train/test/condition cell. No prevalence pooling.",
        "scope": "CWE-119 only, unchanged reference evaluation labels, same positives reused across dataset seeds."}
    lines = ["# CodeBERT comparison with the frozen four-model study", "",
             "Status: " + metadata["status"], "",
             f"Evaluated CodeBERT cells: {len(codebert):,}. Exactly matched classical cells: {len(matched):,}.", "",
             "All comparisons preserve train/validation ratio, test ratio, dataset seed, and condition. Baseline files and their four-model aggregates were not modified.", ""]
    if complete:
        clean = summarize_seeds(combined[combined["condition"] == "clean"], METRICS, complete=True)
        all_summary = summarize_seeds(combined, METRICS, complete=True)
        noise_summary = summarize_seeds(noise, [f"delta_{metric}" for metric in METRICS], complete=True)
        diff_summary = summarize_seeds(differences, [f"codebert_minus_baseline_{metric}" for metric in METRICS], complete=True)
        write_csv(output / "clean_five_models_by_train_test.csv", clean)
        write_csv(output / "five_models_by_train_test_condition.csv", all_summary)
        write_csv(output / "paired_noise_deltas_summary.csv", noise_summary)
        write_csv(output / "codebert_minus_baseline_summary.csv", diff_summary)
        winners, winner_summary = winner_contexts(combined)
        write_csv(output / "five_model_winners_by_context.csv", winners)
        metadata["five_model_winners"] = winner_summary
        write_compact_clean_table(output, clean)
        publication_figures(output, clean, noise_summary, combined)
        lines.extend(["The five-family comparison has 875 selected groups and 5,250 test cells. These are repeated evaluations, not 5,250 independent datasets.", "",
                      "See [the compact clean endpoint table](clean_75_25_test_endpoints.md) for 75:25 training evaluated separately at 50:50 and 90:10 testing.", "",
                      "The matched noise summaries use noisy minus clean within each family, train/test configuration, and seed. The CodeBERT-minus-baseline summaries compare five matched seeds separately for every baseline and setting. Positive differences do not establish a universal winner.", "",
                      f"MCC and PR-AUC share a unique winner in {winner_summary['same_unique_winner_contexts']} of 1,050 five-model contexts ({winner_summary['agreement_fraction_all_contexts']:.1%}). Both metrics have unique winners in {winner_summary['both_metrics_unique_contexts']} contexts. Ties use the four-model analysis's absolute tolerance of 1e-15 (relative tolerance 0) and are listed in the CSV.", ""])
        # Give descriptive CodeBERT effects at declared fixed settings, without causal claims.
        lines.extend(["CodeBERT matched noise effects at 75:25 training / 90:10 testing:", "",
                      "| Condition | ΔPR-AUC mean ± SD | ΔMCC mean ± SD |", "|---|---:|---:|"])
        for condition in CONDITIONS[1:]:
            row = noise_summary[(noise_summary["model_family"] == "codebert") &
                                (noise_summary["train_ratio"] == "75/25") & (noise_summary["eval_ratio"] == "90/10") &
                                (noise_summary["condition"] == condition)].iloc[0]
            lines.append(f"| {NOISE_DISPLAY[condition]} | {row.delta_pr_auc_mean:+.4f} ± {row.delta_pr_auc_sd:.4f} | {row.delta_mcc_mean:+.4f} ± {row.delta_mcc_sd:.4f} |")
        lines.append("")
    else:
        lines.extend(["PARTIAL DIAGNOSTICS: no summary tables, publication figures, or winner claims are emitted. Missing evaluation cells and unmatched noisy/clean references cannot support a full-study paper update.", "",
                      f"Noisy diagnostic cells omitted for lack of a matched clean reference: {missing_clean}.", ""])
    lines.extend(["The five sampling seeds reuse the same 21 vulnerable test functions and 19 validation positives. Sample SD describes this construction's sampling variation and is not a confidence interval or independent-cohort evidence. A single test positive changes a per-model recall by about 4.76 percentage points.", "",
                  "The protocol fixes the optimization seed. These sampling versions do not measure variability across independent neural optimization seeds. CodeBERT fine-tuning, truncation, and pretrained representation differ from TF-IDF baselines. Synthetic corruption does not measure naturally occurring PrimeVul error rates. False-positive compensation also changes training size/composition. There are no additional-CWE experiments in this analysis.", "",
                  "PR-AUC is trapezoidal integration of the precision–recall curve from continuous scores. Average precision is a separate stored metric. Interpret both together with the declared test prevalence.", "",
                  "These are descriptive comparison artifacts. The validated input result files are read-only."])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata["generated_outputs"] = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                                     for path in sorted(output.iterdir()) if path.is_file()}
    (output / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebert-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument(
        "--baseline-validation-report",
        type=Path,
        default=BASELINE_DIR / "validation_report.json",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Allow independently validated partial diagnostics. Emit no publication assets.")
    args = parser.parse_args()
    try:
        result = analyze(
            args.codebert_results_dir,
            args.output_dir,
            allow_partial=args.allow_partial,
            baseline_csv=args.baseline_csv,
            baseline_validation_report=args.baseline_validation_report,
        )
    except (AnalysisFailure, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Analysis refused: {exc}\n")
    print(json.dumps({key: result[key] for key in ["status", "codebert_test_cells", "combined_test_cells", "publication_assets_emitted"]}, indent=2))


if __name__ == "__main__":
    main()

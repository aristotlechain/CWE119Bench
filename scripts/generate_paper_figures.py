#!/usr/bin/env python3
"""Generate the paper's aggregate Results figures across five models.

The paper contains three Results figures:

* the overall noise-effect point/error-bar figure.
* the paired-flip train/test interaction heatmap.
* a deliberately Random-Forest-specific precision/recall case study.

The first two figures average model families, so this script computes their
values over all five families, including CodeBERT, while retaining the paper's
layout, colours, axes, filenames, and aggregation order. The Random Forest
case study remains model-specific and is verified as a required input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHED_INPUT = PROJECT_ROOT / "results/five_model_test_results.csv"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT
    / "results/figure_evidence"
)

MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "lightgbm",
    "codebert",
]
TRAIN_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
TEST_RATIOS = ["50/50", "60/40", "70/30", "75/25", "80/20", "90/10"]
SEEDS = [1, 2, 3, 4, 5]
CONDITIONS = [
    "random_flip_10",
    "random_flip_20",
    "random_fp_10",
    "random_fp_20",
    "heuristic_fp_10",
    "heuristic_fp_20",
]
CONDITION_LABELS = {
    "clean": "Clean",
    "random_flip_10": "Paired flip 10%",
    "random_flip_20": "Paired flip 20%",
    "random_fp_10": "Random FP 10%",
    "random_fp_20": "Random FP 20%",
    "heuristic_fp_10": "Heuristic FP 10%",
    "heuristic_fp_20": "Heuristic FP 20%",
}
DELTA_METRICS = [
    "delta_mcc",
    "delta_f1",
    "delta_precision",
    "delta_recall",
    "delta_pr_auc",
]

COLORS = {
    "clean": "#2468A2",
    "random_flip": "#C43C39",
    "random_fp": "#E68613",
    "heuristic_fp": "#2E8B57",
}


class AssetFailure(RuntimeError):
    """Raised when validated inputs or output invariants do not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssetFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_color(condition: str) -> str:
    if condition == "clean":
        return COLORS["clean"]
    if condition.startswith("random_flip"):
        return COLORS["random_flip"]
    if condition.startswith("random_fp"):
        return COLORS["random_fp"]
    if condition.startswith("heuristic_fp"):
        return COLORS["heuristic_fp"]
    raise AssetFailure(f"Unexpected noise condition: {condition}")


def expected_paired_keys(models: Iterable[str]) -> set[tuple[object, ...]]:
    return set(
        itertools.product(
            models,
            TRAIN_RATIOS,
            TEST_RATIOS,
            SEEDS,
            CONDITIONS,
        )
    )


def load_validated_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(MATCHED_INPUT.is_file(), f"Missing five-model result matrix: {MATCHED_INPUT}")
    matched = pd.read_csv(MATCHED_INPUT)
    paired_keys = ["model_family", "train_ratio", "eval_ratio", "seed", "condition"]
    raw_keys = paired_keys
    require(len(matched) == 5250, f"Expected 5,250 test cells. Observed {len(matched)}.")
    require(not matched.duplicated(raw_keys).any(), "Duplicate matched-result keys.")
    require(set(matched["model_family"]) == set(MODELS), "Unexpected model families.")
    require(set(matched["condition"]) == {"clean", *CONDITIONS}, "Unexpected conditions.")
    require(set(matched["train_ratio"]) == set(TRAIN_RATIOS), "Unexpected train ratios.")
    require(set(matched["eval_ratio"]) == set(TEST_RATIOS), "Unexpected test ratios.")
    require(set(matched["seed"]) == set(SEEDS), "Unexpected dataset seeds.")
    require(
        set(matched[raw_keys].itertuples(index=False, name=None))
        == set(itertools.product(MODELS, TRAIN_RATIOS, TEST_RATIOS, SEEDS, ["clean", *CONDITIONS])),
        "The five-model factorial grid is incomplete.",
    )

    metric_names = [metric.removeprefix("delta_") for metric in DELTA_METRICS]
    clean = matched[matched["condition"].eq("clean")][
        ["model_family", "train_ratio", "eval_ratio", "seed", *metric_names]
    ].rename(columns={metric: f"clean_{metric}" for metric in metric_names})
    paired = matched[~matched["condition"].eq("clean")][
        [*paired_keys, *metric_names]
    ].merge(
        clean,
        on=["model_family", "train_ratio", "eval_ratio", "seed"],
        how="left",
        validate="many_to_one",
    )
    for delta_metric in DELTA_METRICS:
        metric = delta_metric.removeprefix("delta_")
        paired[delta_metric] = paired[metric] - paired[f"clean_{metric}"]
    require(len(paired) == 4500, f"Expected 4,500 paired cells. Observed {len(paired)}.")
    require(not paired.duplicated(paired_keys).any(), "Duplicate paired-result keys.")
    require(set(paired[paired_keys].itertuples(index=False, name=None)) == expected_paired_keys(MODELS),
            "The paired five-model factorial grid is incomplete.")
    return paired, matched


def seed_aggregates(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return overall/grid seed means and their five-seed summaries.

    The order matches the paper: first average model families (and,
    for the overall figure, all 30 ratio cells) within each dataset seed. Then
    take the mean and sample SD across the five seeds.
    """
    overall_seed = (
        paired.groupby(["condition", "seed"], as_index=False)[DELTA_METRICS]
        .mean()
        .sort_values(["condition", "seed"])
        .reset_index(drop=True)
    )
    require(len(overall_seed) == 30, "Expected 30 condition/seed overall means.")
    require(
        (paired.groupby(["condition", "seed"]).size() == 150).all(),
        "Each overall five-model seed mean must average 150 cells.",
    )
    overall_summary = (
        overall_seed.groupby("condition")[DELTA_METRICS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    overall_summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in overall_summary.columns
    ]

    grid_seed = (
        paired.groupby(
            ["condition", "train_ratio", "eval_ratio", "seed"], as_index=False
        )[DELTA_METRICS]
        .mean()
        .sort_values(["condition", "train_ratio", "eval_ratio", "seed"])
        .reset_index(drop=True)
    )
    require(len(grid_seed) == 900, "Expected 900 condition/ratio/seed means.")
    require(
        (paired.groupby(["condition", "train_ratio", "eval_ratio", "seed"]).size() == 5).all(),
        "Each grid seed mean must average exactly five model families.",
    )
    grid_summary = (
        grid_seed.groupby(["condition", "train_ratio", "eval_ratio"])[DELTA_METRICS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    grid_summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in grid_summary.columns
    ]
    return overall_seed, overall_summary, grid_seed, grid_summary


def configure_overall_style() -> None:
    # These are copied from plot_cwe_119_results_section.py.
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, title: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": title,
        "Subject": "Five-model seed-aware CWE-119 results",
        "Creator": Path(__file__).name,
        "CreationDate": datetime(2026, 9, 20, tzinfo=timezone.utc),
    }
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(pdf, bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(figure)
    return [pdf]


def plot_overall_noise_effects(overall_seed: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_overall_style()
    panels = [
        ("delta_mcc", "(a) MCC"),
        ("delta_pr_auc", "(b) PR-AUC"),
        ("delta_precision", "(c) Precision"),
        ("delta_recall", "(d) Recall"),
    ]
    figure, axes = plt.subplots(
        2, 2, figsize=(7.35, 5.15), sharey=True, constrained_layout=True
    )
    figure.patch.set_facecolor("white")
    y_positions = np.arange(len(CONDITIONS))
    seed_offsets = np.linspace(-0.13, 0.13, 5)

    for panel_index, (axis, (metric, title)) in enumerate(zip(axes.flat, panels)):
        axis.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
        for row, condition in enumerate(CONDITIONS):
            values = (
                overall_seed[overall_seed["condition"].eq(condition)]
                .sort_values("seed")[metric]
                .to_numpy(dtype=float)
            )
            require(len(values) == 5, f"Expected five seed means for {condition}, {metric}.")
            color = condition_color(condition)
            axis.scatter(
                values,
                row + seed_offsets,
                s=12,
                color=color,
                alpha=0.38,
                linewidths=0,
                zorder=2,
            )
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            axis.plot([0.0, mean], [row, row], color=color, alpha=0.55, linewidth=1.1)
            axis.errorbar(
                mean,
                row,
                xerr=sd,
                fmt="o",
                markersize=5.2,
                color=color,
                ecolor=color,
                elinewidth=1.3,
                capsize=2.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
        axis.set_title(title, fontweight="bold", pad=5)
        if metric in {"delta_precision", "delta_recall"}:
            axis.set_xlim(-0.29, 0.14)
            axis.set_xticks([-0.25, -0.15, -0.05, 0.05, 0.13])
        else:
            axis.set_xlim(-0.29, 0.085)
            axis.set_xticks([-0.25, -0.15, -0.05, 0.05])
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        axis.set_ylim(len(CONDITIONS) - 0.55, -0.55)
        axis.set_yticks(y_positions)
        if panel_index in {0, 2}:
            axis.set_yticklabels([CONDITION_LABELS[item] for item in CONDITIONS])
        axis.tick_params(axis="y", length=0)

    return save_figure(
        figure,
        output_dir,
        "fig_noise_effects_seed_aware_true_pr_auc",
        "Noise effects across five model families",
    )


def configure_heatmap_style() -> None:
    # These are copied from build_section5_ratio_conditioned_assets.py.
    # Reset first because this script builds the point plot in the same process.
    # the two source calculations were originally run independently.
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_paired_flip_grid(grid_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_heatmap_style()
    panels = [
        ("random_flip_10", "delta_mcc", "(a) Paired flip 10%: $\\Delta$MCC"),
        ("random_flip_10", "delta_pr_auc", "(b) Paired flip 10%: $\\Delta$PR-AUC"),
        ("random_flip_20", "delta_mcc", "(c) Paired flip 20%: $\\Delta$MCC"),
        ("random_flip_20", "delta_pr_auc", "(d) Paired flip 20%: $\\Delta$PR-AUC"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 5.15), constrained_layout=True)
    figure.patch.set_facecolor("white")
    norm = TwoSlopeNorm(vmin=-0.70, vcenter=0.0, vmax=0.08)
    image = None
    for panel_index, (axis, (condition, metric, title)) in enumerate(
        zip(axes.flat, panels)
    ):
        subset = grid_summary[grid_summary["condition"].eq(condition)]
        matrix = (
            subset.pivot(
                index="train_ratio", columns="eval_ratio", values=f"{metric}_mean"
            )
            .reindex(index=TRAIN_RATIOS, columns=TEST_RATIOS)
            .to_numpy(dtype=float)
        )
        require(matrix.shape == (5, 6), f"Incomplete figure matrix: {condition}, {metric}.")
        require(np.isfinite(matrix).all(), f"Non-finite figure matrix: {condition}, {metric}.")
        image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color="white" if value < -0.36 else "#111111",
                    fontweight="bold" if value < -0.36 else "normal",
                )
        axis.set_title(title, fontweight="bold", pad=5)
        axis.set_xticks(np.arange(len(TEST_RATIOS)))
        axis.set_xticklabels([ratio.replace("/", ":") for ratio in TEST_RATIOS])
        axis.set_yticks(np.arange(len(TRAIN_RATIOS)))
        if panel_index % 2 == 0:
            axis.set_yticklabels([ratio.replace("/", ":") for ratio in TRAIN_RATIOS])
            axis.set_ylabel("Train/validation ratio")
        else:
            axis.set_yticklabels([])
        if panel_index >= 2:
            axis.set_xlabel("Test ratio")
        axis.set_xticks(np.arange(-0.5, len(TEST_RATIOS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(TRAIN_RATIOS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
    require(image is not None, "Paired-flip figure was not created.")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_label("Mean change from matched clean training")
    return save_figure(
        figure,
        output_dir,
        "fig_paired_flip_train_test_interaction_true_pr_auc",
        "Paired-flip effects across five model families",
    )


def plot_random_forest_case(matched: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_overall_style()
    rf = matched[
        matched["model_family"].eq("random_forest")
        & matched["eval_ratio"].eq("90/10")
    ]
    grouped = rf.groupby(["train_ratio", "condition", "seed"], as_index=False)[
        ["precision", "recall"]
    ].mean()
    require(len(grouped) == len(TRAIN_RATIOS) * 7 * 5,
            "The Random Forest precision-recall grid is incomplete.")
    display_conditions = [
        "clean",
        "heuristic_fp_10",
        "heuristic_fp_20",
        "random_fp_10",
        "random_fp_20",
        "random_flip_10",
        "random_flip_20",
    ]
    markers = {
        "clean": "o",
        "heuristic_fp_10": "^",
        "heuristic_fp_20": "v",
        "random_fp_10": "D",
        "random_fp_20": "P",
        "random_flip_10": "X",
        "random_flip_20": "X",
    }
    figure, axes = plt.subplots(
        2, 3, figsize=(7.35, 4.85), sharex=True, sharey=True, constrained_layout=True
    )
    figure.patch.set_facecolor("white")
    for panel, ratio in enumerate(TRAIN_RATIOS):
        axis = axes.flat[panel]
        for condition in display_conditions:
            values = grouped[
                grouped["train_ratio"].eq(ratio)
                & grouped["condition"].eq(condition)
            ].sort_values("seed")
            require(len(values) == 5,
                    f"Incomplete Random Forest summary for {ratio}, {condition}.")
            color = "#9B59B6" if condition == "random_flip_10" else condition_color(condition)
            axis.scatter(
                values["recall"], values["precision"], marker=markers[condition],
                s=17, color=color, alpha=0.22, linewidths=0, zorder=1,
            )
            mean_recall = float(values["recall"].mean())
            mean_precision = float(values["precision"].mean())
            recall_values = values["recall"].to_numpy(dtype=float)
            precision_values = values["precision"].to_numpy(dtype=float)
            axis.errorbar(
                mean_recall,
                mean_precision,
                xerr=np.array([[mean_recall - recall_values.min()],
                               [recall_values.max() - mean_recall]]),
                yerr=np.array([[mean_precision - precision_values.min()],
                               [precision_values.max() - mean_precision]]),
                fmt=markers[condition], color=color, ecolor=color, markersize=5.3,
                markeredgecolor="white", markeredgewidth=0.45, elinewidth=0.75,
                capsize=1.8, zorder=3,
            )
        axis.set_title(f"Train/valid {ratio.replace('/', ':')}", fontweight="bold", pad=4)
        axis.set_xlim(-0.035, 1.02)
        axis.set_ylim(-0.035, 1.02)
        axis.set_xticks(np.arange(0.0, 1.1, 0.2))
        axis.set_yticks(np.arange(0.0, 1.1, 0.2))
        axis.grid(color="#D8D8D8", linewidth=0.55, alpha=0.8)
    axes.flat[5].axis("off")
    handles = []
    for condition in display_conditions:
        color = "#9B59B6" if condition == "random_flip_10" else condition_color(condition)
        handles.append(Line2D(
            [0], [0], marker=markers[condition], linestyle="none",
            markerfacecolor=color, markeredgecolor="white", markersize=6,
            label=CONDITION_LABELS[condition],
        ))
    axes.flat[5].legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.02, 0.52),
        frameon=False, handletextpad=0.45, labelspacing=0.55,
    )
    figure.supxlabel("Recall")
    figure.supylabel("Precision")
    return save_figure(
        figure,
        output_dir,
        "fig_rf_pr_by_train_ratio_all_conditions_seed_aware",
        "Random Forest precision-recall results at the fixed 90:10 test ratio",
    )


def write_evidence(
    evidence_dir: Path,
    overall_seed: pd.DataFrame,
    overall_summary: pd.DataFrame,
    grid_seed: pd.DataFrame,
    grid_summary: pd.DataFrame,
    generated: list[Path],
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall_five_model_seed_means.csv": overall_seed,
        "overall_five_model_summary.csv": overall_summary,
        "paired_flip_grid_five_model_seed_means.csv": grid_seed[
            grid_seed["condition"].isin(["random_flip_10", "random_flip_20"])
        ],
        "paired_flip_grid_five_model_summary.csv": grid_summary[
            grid_summary["condition"].isin(["random_flip_10", "random_flip_20"])
        ],
    }
    for filename, frame in paths.items():
        frame.to_csv(evidence_dir / filename, index=False, float_format="%.17g")

    record = {
        "status": "complete",
        "purpose": "Generate the five-model aggregate figures in the paper format",
        "input_test_results": {
            "path": str(MATCHED_INPUT.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(MATCHED_INPUT),
            "rows": 5250,
        },
        "aggregation": {
            "overall_figure": (
                "Within each dataset seed and noise condition, average 5 model families x "
                "5 train ratios x 6 test ratios = 150 matched noisy-minus-clean cells. "
                "then report mean and sample SD across five dataset seeds."
            ),
            "paired_flip_figure": (
                "Within each dataset seed, paired-flip condition, train ratio, and test ratio, "
                "average exactly five model families. Then report the mean across five seeds."
            ),
        },
        "generated_figure_files": {
            str(path.relative_to(PROJECT_ROOT)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        },
        "generated_evidence_files": {},
    }
    for path in sorted(evidence_dir.glob("*.csv")):
        record["generated_evidence_files"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Paper figure evidence",
        "",
        "The paper's figure designs are retained.",
        "",
        "- `fig_noise_effects_seed_aware_true_pr_auc`: same four-panel point/error-bar design. It aggregates five model families.",
        "- `fig_paired_flip_train_test_interaction_true_pr_auc`: same four-panel train/test heatmap. It aggregates five model families.",
        "- `fig_rf_pr_by_train_ratio_all_conditions_seed_aware`: Random Forest case study generated from its rows in the same matrix.",
        "",
        "All deltas are noisy minus the matched clean result for the same model family, train ratio, test ratio, and dataset seed. Sample SD is calculated after within-seed averaging and describes the five benchmark versions. It is not a confidence interval.",
        "",
    ]
    (evidence_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(output_dir: Path, evidence_dir: Path) -> None:
    output_dir = output_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    require(
        output_dir == (PROJECT_ROOT / "figures").resolve(),
        "Figure output must be the repository figures directory.",
    )
    paired, matched = load_validated_inputs()
    overall_seed, overall_summary, grid_seed, grid_summary = seed_aggregates(paired)
    generated = [
        *plot_overall_noise_effects(overall_seed, output_dir),
        *plot_paired_flip_grid(grid_summary, output_dir),
        *plot_random_forest_case(matched, output_dir),
    ]
    write_evidence(
        evidence_dir,
        overall_seed,
        overall_summary,
        grid_seed,
        grid_summary,
        generated,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "generated_figures": 3,
                "output_dir": str(output_dir),
                "evidence_dir": str(evidence_dir),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args()
    try:
        build(args.output_dir, args.evidence_dir)
    except (AssetFailure, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Figure update refused: {exc}\n")


if __name__ == "__main__":
    main()

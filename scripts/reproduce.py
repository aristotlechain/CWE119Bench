#!/usr/bin/env python3
"""Run the reproducibility stages for CWE119Bench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PUBLISHED_RESULTS = ROOT / "results/model_experiments"
REPRODUCTION_ROOT = ROOT / "results/reproduction"
CLASSICAL_FAMILIES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "lightgbm",
)


def run(*arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS / arguments[0]), *arguments[1:]]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def fetch_data() -> None:
    run("fetch_primevul.py")


def build_data() -> None:
    run("build_cwe_119_imbalance_benchmarks.py")
    run("build_cwe_119_ratio_preserved_noise_benchmarks.py")
    run("data_integrity.py", "verify", "--scope", "all")


def run_classical() -> None:
    output = REPRODUCTION_ROOT / "classical"
    for family in CLASSICAL_FAMILIES:
        run(
            "rerun_cwe_119_selected_true_pr_auc.py",
            "--model-family",
            family,
            "--source-results-dir",
            str(PUBLISHED_RESULTS),
            "--output-results-dir",
            str(output),
        )
    run(
        "validate_cwe_119_true_pr_auc_rerun.py",
        "--source-results-dir",
        str(PUBLISHED_RESULTS),
        "--new-output-results-dir",
        str(output),
        "--report-dir",
        str(REPRODUCTION_ROOT / "classical_validation"),
    )
    run(
        "analyze_cwe_119_research_questions_true_pr_auc.py",
        "--input",
        str(REPRODUCTION_ROOT / "classical_validation/combined_test_results_true_pr_auc_v1.csv"),
        "--validation-report",
        str(REPRODUCTION_ROOT / "classical_validation/validation_report.json"),
        "--output-dir",
        str(REPRODUCTION_ROOT / "classical_analysis"),
    )


def run_codebert(*, resume: bool) -> None:
    output = REPRODUCTION_ROOT / "codebert"
    analysis = REPRODUCTION_ROOT / "five_model_analysis"
    run("fetch_codebert.py")
    arguments = [
        "run_cwe_119_codebert_experiments.py",
        "--config",
        str(ROOT / "configs/codebert.json"),
        "--output-dir",
        str(output),
        "--evaluate-tests",
    ]
    if resume:
        arguments.append("--resume")
    run(*arguments)
    run("validate_cwe_119_codebert_results.py", "--results-dir", str(output))
    run(
        "analyze_cwe_119_codebert_results.py",
        "--codebert-results-dir",
        str(output),
        "--output-dir",
        str(analysis),
    )
    run(
        "prepare_release_results.py",
        "--input",
        str(analysis / "matched_test_cells.csv"),
        "--output",
        str(REPRODUCTION_ROOT / "five_model_test_results.csv"),
    )


def run_smoke() -> None:
    run(
        "run_cwe_119_model_experiments.py",
        "--ratios",
        "75/25",
        "--seeds",
        "1",
        "--conditions",
        "clean",
        "--model-family",
        "logistic_regression",
        "--class-weight",
        "balanced",
        "--selection-metric",
        "mcc",
        "--run-name",
        "classical_smoke",
        "--results-dir",
        str(REPRODUCTION_ROOT),
    )


def build_paper_assets() -> None:
    run("generate_paper_figures.py")
    run("generate_paper_tables.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("fetch-data", "build-data", "verify-data", "classical", "smoke", "paper-assets"):
        subparsers.add_parser(stage)
    codebert = subparsers.add_parser("codebert")
    codebert.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "fetch-data":
        fetch_data()
    elif args.stage == "build-data":
        build_data()
    elif args.stage == "verify-data":
        run("data_integrity.py", "verify", "--scope", "all")
    elif args.stage == "classical":
        run_classical()
    elif args.stage == "codebert":
        run_codebert(resume=args.resume)
    elif args.stage == "smoke":
        run_smoke()
    elif args.stage == "paper-assets":
        build_paper_assets()


if __name__ == "__main__":
    main()

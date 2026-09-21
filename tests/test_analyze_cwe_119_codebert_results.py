"""Exact-join and independent-validation guard tests. No model is trained."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_cwe_119_codebert_results as analyzer


def fixture(models: list[str]) -> pd.DataFrame:
    rows = []
    sizes = dict(zip(analyzer.TEST_RATIOS, [42, 53, 70, 84, 105, 210]))
    for model, train, test, seed, condition in itertools.product(
            models, analyzer.TRAIN_RATIOS, analyzer.TEST_RATIOS, analyzer.SEEDS, analyzer.CONDITIONS):
        # Deliberately vary prevalence, training, model and seed to detect bad joins.
        value = (0.50 + 0.015 * analyzer.ALL_MODELS.index(model) + 0.002 * seed
                 + 0.003 * analyzer.TRAIN_RATIOS.index(train) - 0.008 * analyzer.TEST_RATIOS.index(test))
        if condition != "clean":
            value -= 0.01 * analyzer.CONDITIONS.index(condition)
        row = dict(zip(analyzer.CELL_KEYS, [model, train, test, seed, condition]))
        row.update(stage="test", split="test", n_rows=sizes[test], positive_rows=21,
                   benign_rows=sizes[test] - 21, selected_epoch=2, learning_rate=2e-5)
        row.update({metric: value for metric in analyzer.METRICS})
        row["average_precision"] = value + 0.025
        rows.append(row)
    return pd.DataFrame(rows)


def write_input(directory: Path, codebert: pd.DataFrame, *, full: bool) -> None:
    directory.mkdir()
    selected = codebert.drop_duplicates(["train_ratio", "seed", "condition"]).copy()
    selected["stage"] = "validation"
    selected["split"] = "valid"
    selected["eval_ratio"] = selected["train_ratio"]
    pd.concat([codebert, selected]).to_csv(directory / "metrics.csv", index=False, float_format="%.17g")
    (directory / "manifest.json").write_text(json.dumps({"fixture": "Synthetic test only. Not experiment evidence"}))
    report = {"status": "full_passed" if full else "partial_passed", "completed_groups": len(selected),
              "test_cells": len(codebert), "checked_file_sha256": {
                  name: analyzer.sha256_file(directory / name) for name in ["metrics.csv", "manifest.json"]}}
    (directory / "validation_report.json").write_text(json.dumps(report))


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.codebert = fixture(["codebert"])
        cls.baselines = fixture(analyzer.MODELS)

    def test_complete_grid_rejects_duplicate_missing_and_unknown_keys(self) -> None:
        analyzer.validate_test_frame(self.codebert, ["codebert"], complete=True)
        duplicate = pd.concat([self.codebert, self.codebert.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "Duplicate"):
            analyzer.validate_test_frame(duplicate, ["codebert"], complete=True)
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "missing 1"):
            analyzer.validate_test_frame(self.codebert.iloc[1:], ["codebert"], complete=True)
        unknown = self.codebert.copy()
        unknown.loc[0, "seed"] = 6
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "Unexpected"):
            analyzer.validate_test_frame(unknown, ["codebert"], complete=True)

    def test_comparison_matches_every_condition_ratio_and_seed(self) -> None:
        subset = self.codebert[(self.codebert.train_ratio == "75/25") &
                               (self.codebert.eval_ratio == "90/10") & (self.codebert.seed == 2)]
        matched = analyzer.match_baselines(subset, self.baselines)
        self.assertEqual(len(matched), 28)
        self.assertEqual(set(matched.eval_ratio), {"90/10"})
        self.assertEqual(set(matched.seed), {2})
        self.assertEqual(set(matched.condition), set(analyzer.CONDITIONS))
        # Remove the actual context/family, not its post-merge positional index.
        key = matched.iloc[0][analyzer.CELL_KEYS]
        missing = self.baselines[~(self.baselines[analyzer.CELL_KEYS] == key).all(axis=1)]
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "Missing matched"):
            analyzer.match_baselines(subset, missing)
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "Duplicate"):
            analyzer.match_baselines(pd.concat([subset, subset.iloc[[0]]]), self.baselines)

    def test_noise_deltas_match_clean_within_model_train_test_and_seed(self) -> None:
        combined = pd.concat([self.baselines, self.codebert])
        deltas, missing = analyzer.paired_noise_deltas(combined)
        self.assertEqual(missing, 0)
        self.assertEqual(len(deltas), 4500)
        fp20 = deltas[deltas.condition == "random_fp_20"]
        self.assertTrue((abs(fp20.delta_mcc + 0.04) < 1e-12).all())
        cb = deltas[(deltas.model_family == "codebert") & (deltas.train_ratio == "75/25") &
                    (deltas.eval_ratio == "90/10") & (deltas.seed == 2) & (deltas.condition == "random_fp_20")].iloc[0]
        self.assertAlmostEqual(cb.clean_pr_auc, 0.50 + 0.015 * 4 + 0.002 * 2 + 0.003 * 2 - 0.008 * 5)
        without_clean = combined[~((combined.model_family == "codebert") & (combined.condition == "clean"))]
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "Missing matched clean"):
            analyzer.paired_noise_deltas(without_clean)
        partial, omitted = analyzer.paired_noise_deltas(without_clean, allow_partial=True)
        self.assertEqual(omitted, 900)
        self.assertEqual(len(partial), 3600)

    def test_five_seed_summary_uses_sample_sd_without_pooling_prevalence(self) -> None:
        summary = analyzer.summarize_seeds(self.codebert, analyzer.METRICS, complete=True)
        self.assertEqual(len(summary), 210)
        row = summary[(summary.train_ratio == "75/25") & (summary.eval_ratio == "90/10") &
                      (summary.condition == "clean")].iloc[0]
        self.assertEqual(row.n_seeds, 5)
        self.assertAlmostEqual(row.mcc_sd, pd.Series([0.002 * seed for seed in range(1, 6)]).std(ddof=1))
        self.assertAlmostEqual(row.average_precision_mean - row.pr_auc_mean, 0.025)
        with self.assertRaisesRegex(analyzer.AnalysisFailure, "five seeds"):
            analyzer.summarize_seeds(self.codebert[self.codebert.seed != 5], analyzer.METRICS, complete=True)

    def test_validation_is_required_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            write_input(directory, self.codebert.iloc[[0]], full=False)
            with self.assertRaisesRegex(analyzer.AnalysisFailure, "full_passed"):
                analyzer.load_codebert(directory, allow_partial=False)
            analyzer.load_codebert(directory, allow_partial=True)
            with (directory / "metrics.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(analyzer.AnalysisFailure, "changed"):
                analyzer.load_codebert(directory, allow_partial=True)

    def test_five_model_winner_denominator_and_ties(self) -> None:
        combined = pd.concat([self.baselines, self.codebert])
        _, counts = analyzer.winner_contexts(combined)
        self.assertEqual(counts["comparison_contexts"], 1050)
        self.assertEqual(counts["same_unique_winner_contexts"], 1050)
        tied = combined.copy()
        first = self.codebert.iloc[0]
        matching = (tied[analyzer.MATCH_KEYS] == first[analyzer.MATCH_KEYS]).all(axis=1)
        cbvalue = first.mcc
        tied.loc[matching & (tied.model_family == "lightgbm"), "mcc"] = cbvalue - 5e-16
        _, counts = analyzer.winner_contexts(tied)
        self.assertEqual(counts["same_unique_winner_contexts"], 1049)
        self.assertEqual(counts["both_metrics_unique_contexts"], 1049)

    def test_partial_analysis_never_emits_publication_assets(self) -> None:
        subset = self.codebert[(self.codebert.train_ratio == "75/25") &
                               (self.codebert.eval_ratio == "90/10") & (self.codebert.seed == 1)]
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "run", Path(temporary) / "analysis"
            write_input(run, subset, full=False)
            with patch.object(analyzer, "load_baselines", return_value=self.baselines.copy()):
                manifest = analyzer.analyze(run, output, allow_partial=True)
            self.assertFalse(manifest["publication_assets_emitted"])
            self.assertEqual(manifest["combined_test_cells"], 35)
            self.assertFalse(list(output.glob("*.pdf")) + list(output.glob("*.png")) + list(output.glob("*.tex")))
            self.assertIn("PARTIAL DIAGNOSTICS", (output / "README.md").read_text())

    def test_full_synthetic_analysis_writes_matched_not_pooled_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "run", Path(temporary) / "analysis"
            write_input(run, self.codebert, full=True)
            with patch.object(analyzer, "load_baselines", return_value=self.baselines.copy()):
                manifest = analyzer.analyze(run, output)
            self.assertEqual(manifest["combined_test_cells"], 5250)
            self.assertEqual(manifest["paired_noisy_test_cells"], 4500)
            self.assertTrue(manifest["publication_assets_emitted"])
            self.assertEqual(len(pd.read_csv(output / "clean_five_models_by_train_test.csv")), 150)
            self.assertEqual(len(pd.read_csv(output / "clean_75_25_test_endpoints.csv")), 10)
            self.assertEqual(len(list(output.glob("*.pdf"))), 3)
            self.assertEqual(len(list(output.glob("*.png"))), 3)
            self.assertEqual(manifest["five_model_winners"]["comparison_contexts"], 1050)
            self.assertIn("ddof=1", manifest["summary_policy"])


if __name__ == "__main__":
    unittest.main()

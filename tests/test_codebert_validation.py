"""Meaningful artifact corruption and threshold tests. No neural dependencies."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_cwe_119_codebert_results as audit


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ArtifactFixture:
    def __init__(self, root: Path, *, with_tests: bool = False):
        self.root = root
        self.noise = root / "noise"
        self.tests = root / "tests"
        self.result = root / "results"
        self.group = self.result / "groups/ratio_75_25/seed_1/clean"
        self.group.mkdir(parents=True)
        self.protocol = {"max_epochs": 5, "learning_rate": 2e-5,
                         "noise_dir": str(self.noise), "test_dir": str(self.tests),
                         "test_ratios": ["75/25"]}
        self.protocol_hash = audit.payload_sha256(self.protocol)
        self.inputs = {}
        train = self.records("train", [1, 0, 0, 0])
        for row, label in zip(train, [0, 1, 1, 0]):
            row["noisy_label"] = label
        self.inputs["train"] = self.write_input("train", train)
        valid = self.records("valid", [1, 0, 1, 0])
        self.inputs["valid"] = self.write_input("valid", valid)
        self.metrics, self.predictions = [], []
        self.add_cell("validation", valid)
        if with_tests:
            records = self.records("test", [1, 0, 1, 0])
            self.inputs["test:75/25"] = self.write_input("test", records)
            self.add_cell("test", records)
        self.history = [{"train_ratio": "75/25", "seed": 1, "condition": "clean", "epoch": epoch,
                         **audit.reconstruct_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])}
                        for epoch in range(1, 6)]
        self.selected = [{"model_family": "codebert", "train_ratio": "75/25", "seed": 1,
                          "condition": "clean", "selected_epoch": 1, "learning_rate": 2e-5}]
        model_config = {"codebert_selected_epoch": 1, "codebert_learning_rate": 2e-5,
                        "codebert_protocol_sha256": self.protocol_hash, "id2label": {"0": "benign", "1": "vulnerable"}}
        write_json(self.group / "selected_model/config.json", model_config)
        (self.group / "selected_model/model.safetensors").write_bytes(b"test checkpoint. Not loaded")
        self.manifest = {"status": "complete", "evaluation_status": "tests_complete" if with_tests else "validation_only",
                         "group": {"train_ratio": "75/25", "seed": 1, "condition": "clean"},
                         "protocol": self.protocol, "protocol_sha256": self.protocol_hash,
                         "config_sha256": self.protocol_hash, "training_seed": 0, "selected_epoch": 1,
                         "target_epochs": 5, "training_epochs_completed": 5,
                         "noisy_label_counts": {"0": 2, "1": 2}, "class_weights": [1.0, 1.0],
                         "input_paths": {name: str(path) for name, path in self.inputs.items()},
                         "input_sha256": {name: audit.sha256(path) for name, path in self.inputs.items()},
                         "code_sha256": {"validate_cwe_119_codebert_results.py": audit.sha256(Path(audit.__file__))},
                         "checkpoint_reload_tolerance": 1e-6, "checkpoint_reload_max_abs_score_difference": 0.0}
        self.top = {"status": "complete", "protocol": self.protocol, "protocol_sha256": self.protocol_hash,
                    "config_sha256": self.protocol_hash,
                    "matrix": {"train_ratios": ["75/25"], "seeds": [1], "conditions": ["clean"]}}
        self.save()

    def records(self, split: str, labels: list[int]) -> list[dict]:
        return [{"idx": index, "func_hash": f"{split}-{index}", "func": f"void f{index}() {{}}",
                 "commit_id": f"commit-{split}", "cve": "CVE-fixture", "file_hash": f"file-{split}",
                 "source_primevul_file": f"primevul_{split}.jsonl", "benchmark_ratio": "75/25",
                 "benchmark_seed": 1, "benchmark_row_role": "target_cwe_positive" if label else "base_benign",
                 "binary_label": label} for index, label in enumerate(labels, 1)]

    def write_input(self, split: str, rows: list[dict]) -> Path:
        path = self.tests / "ratio_75_25/seed_1/test.jsonl" if split == "test" else self.noise / f"ratio_75_25/seed_1/clean/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def add_cell(self, stage: str, records: list[dict]) -> None:
        scores = [0.9, 0.8, 0.7, 0.1]
        common = {"run_name": "fixture", "model": "codebert", "model_family": "codebert",
                  "train_ratio": "75/25", "seed": 1, "condition": "clean", "stage": stage,
                  "eval_ratio": "75/25", "split": "valid" if stage == "validation" else "test",
                  "selected_epoch": 1, "learning_rate": 2e-5}
        self.metrics.append({**common, **audit.reconstruct_metrics([row["binary_label"] for row in records], scores)})
        for ordinal, (source, probability) in enumerate(zip(records, scores), 1):
            self.predictions.append({**common, "hyperparameter_name": "learning_rate", "hyperparameter_value": 2e-5,
                                     "row_number": ordinal, "row_sha256": audit.stable_row_sha256(source),
                                     **{field: source[field] for field in audit.IDENTITY_FIELDS},
                                     "binary_label": source["binary_label"], "predicted_label": int(probability >= 0.5),
                                     "positive_class_score": probability, "score_type": "positive_class_probability",
                                     "decision_threshold": "0.5"})

    def save(self) -> None:
        for name, rows in (("metrics.csv", self.metrics), ("prediction_scores.csv", self.predictions),
                           ("validation_history.csv", self.history)):
            write_csv(self.group / name, rows)
            write_csv(self.result / name, rows)
        write_csv(self.result / "selected_hyperparameters.csv", self.selected)
        self.manifest["output_sha256"] = {name: audit.sha256(self.group / name) for name in
                                          ("metrics.csv", "prediction_scores.csv", "validation_history.csv",
                                           "selected_model/config.json", "selected_model/model.safetensors")}
        write_json(self.group / "manifest.json", self.manifest)
        self.top["output_sha256"] = {name: audit.sha256(self.result / name) for name in audit.TOP_OUTPUTS if name != "manifest.json"}
        write_json(self.result / "manifest.json", self.top)


class IndependentMetricsTests(unittest.TestCase):
    def test_pr_auc_is_not_average_precision(self):
        fixture = audit.metric_fixture()
        self.assertAlmostEqual(fixture["average_precision"], 5 / 6)
        self.assertAlmostEqual(fixture["pr_auc"], 19 / 24)

    def test_half_probability_is_positive_and_zero_mcc_denominator_is_valid(self):
        metrics = audit.reconstruct_metrics([0, 1], [0.5, 0.5])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (0, 1, 0, 1))
        self.assertEqual(metrics["mcc"], 0.0)

    def test_nonfinite_and_out_of_range_probabilities_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(audit.ValidationError):
                audit.reconstruct_metrics([0, 1], [value, 0.8])


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = ArtifactFixture(Path(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def test_valid_pilot_is_partial_and_cannot_certify_full_experiment(self):
        report = audit.validate_results(self.fixture.result, allow_partial=True)
        self.assertEqual(report["status"], "partial_passed")
        self.assertEqual(report["validation_score_rows"], 4)
        self.assertEqual(report["test_cells"], 0)
        self.assertEqual(report["checked_file_sha256"]["metrics.csv"], audit.sha256(self.fixture.result / "metrics.csv"))
        with self.assertRaises(audit.ValidationError):
            audit.validate_results(self.fixture.result)

    def test_subset_with_tests_still_is_not_full(self):
        fixture = ArtifactFixture(Path(self.directory.name) / "subset", with_tests=True)
        report = audit.validate_results(fixture.result, allow_partial=True)
        self.assertEqual(report["status"], "partial_passed")
        self.assertEqual(report["test_score_rows"], 4)

    def test_rejects_ap_aliased_into_pr_auc(self):
        self.fixture.metrics[0]["pr_auc"] = self.fixture.metrics[0]["average_precision"]
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "pr_auc"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_missing_or_duplicate_predictions_are_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                fixture = ArtifactFixture(Path(self.directory.name) / str(duplicate))
                fixture.predictions = fixture.predictions + [fixture.predictions[0]] if duplicate else fixture.predictions[:-1]
                fixture.save()
                with self.assertRaisesRegex(audit.ValidationError, "row count"):
                    audit.validate_results(fixture.result, allow_partial=True)

    def test_mismatched_reference_label_is_rejected(self):
        self.fixture.predictions[0]["binary_label"] = 0
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "reference label"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_noncanonical_row_order_is_rejected(self):
        self.fixture.predictions[0], self.fixture.predictions[1] = self.fixture.predictions[1], self.fixture.predictions[0]
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "order"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_learner_weights_cannot_use_hidden_clean_labels(self):
        self.fixture.manifest["class_weights"] = [4 / (2 * 3), 4 / (2 * 1)]
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "class_weights"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_exact_tie_must_select_earliest_epoch(self):
        self.fixture.manifest["selected_epoch"] = 2
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "tie-breaks"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_missing_validation_epoch_is_rejected(self):
        self.fixture.history.pop()
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "history"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_changed_output_hash_is_rejected(self):
        with (self.fixture.group / "metrics.csv").open("a") as handle:
            handle.write("corrupted\n")
        with self.assertRaisesRegex(audit.ValidationError, "Output checksum"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_duplicate_aggregate_rows_are_rejected(self):
        write_csv(self.fixture.result / "metrics.csv", self.fixture.metrics * 2)
        with self.assertRaisesRegex(audit.ValidationError, "duplicate"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_threshold_tie_mislabel_is_rejected(self):
        self.fixture.predictions[0]["positive_class_score"] = 0.5
        self.fixture.predictions[0]["predicted_label"] = 0
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, ">=0.5"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_selection_maximum_uses_metrics_before_epoch_tie_break(self):
        history = [{"epoch": 1, "mcc": 0.2, "f1": 0.8, "recall": 0.9, "precision": 0.7, "balanced_accuracy": 0.8},
                   {"epoch": 2, "mcc": 0.3, "f1": 0.7, "recall": 0.8, "precision": 0.6, "balanced_accuracy": 0.7}]
        self.assertEqual(audit.selection_epoch(history, 2), 2)

    def test_local_model_artifact_change_is_rejected(self):
        local_model = Path(self.directory.name) / "pinned-model"
        local_model.mkdir()
        weights = local_model / "pytorch_model.bin"
        weights.write_bytes(b"original pinned model")
        binding = {"pytorch_model.bin": audit.sha256(weights)}
        self.fixture.protocol.update(local_model_path=str(local_model), model_artifact_sha256=binding)
        protocol_hash = audit.payload_sha256(self.fixture.protocol)
        self.fixture.manifest.update(protocol_sha256=protocol_hash, config_sha256=protocol_hash,
                                     model_artifact_sha256=binding)
        self.fixture.top.update(protocol_sha256=protocol_hash, config_sha256=protocol_hash,
                                model_artifact_sha256=binding)
        model_config = audit.read_json(self.fixture.group / "selected_model/config.json")
        model_config["codebert_protocol_sha256"] = protocol_hash
        write_json(self.fixture.group / "selected_model/config.json", model_config)
        self.fixture.save()
        audit.validate_results(self.fixture.result, allow_partial=True)
        weights.write_bytes(b"changed local pretrained weights")
        with self.assertRaisesRegex(audit.ValidationError, "Pinned model artifact"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def bind_huggingface_snapshot(self):
        hub = Path(self.directory.name) / "model_cache/hub"
        repository = hub / "models--microsoft--codebert-base"
        snapshot = repository / "snapshots" / audit.MODEL_REVISION
        blobs = repository / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        binding, receipt_files = {}, []
        for name in sorted(audit.MODEL_FILES):
            blob = blobs / f"blob-{name}"
            blob.write_bytes(f"verified fixture {name}".encode())
            (snapshot / name).symlink_to(Path("../../blobs") / blob.name)
            binding[name] = audit.sha256(blob)
            receipt_files.append({"name": name, "sha256": binding[name], "bytes": blob.stat().st_size})
        self.fixture.protocol.update(model_name="microsoft/codebert-base", model_revision=audit.MODEL_REVISION,
                                     local_model_path=str(snapshot), model_cache_dir=str(hub), model_artifact_sha256=binding)
        protocol_hash = audit.payload_sha256(self.fixture.protocol)
        self.fixture.manifest.update(protocol_sha256=protocol_hash, config_sha256=protocol_hash,
                                     model_artifact_sha256=binding)
        self.fixture.top.update(protocol_sha256=protocol_hash, config_sha256=protocol_hash,
                                model_artifact_sha256=binding)
        model_config = audit.read_json(self.fixture.group / "selected_model/config.json")
        model_config["codebert_protocol_sha256"] = protocol_hash
        write_json(self.fixture.group / "selected_model/config.json", model_config)
        write_json(hub.parent / "pinned_model_receipt.json", {
            "model_name": "microsoft/codebert-base", "model_revision": audit.MODEL_REVISION,
            "remote_revision_match": True, "pytorch_weights_verified_against_remote_lfs_sha256": True,
            "snapshot_path": str(snapshot), "files": receipt_files})
        self.fixture.save()
        return snapshot, blobs

    def test_huggingface_snapshot_blob_symlinks_pass_with_receipt_and_hashes(self):
        self.bind_huggingface_snapshot()
        report = audit.validate_results(self.fixture.result, allow_partial=True)
        self.assertEqual(report["status"], "partial_passed")

    def test_snapshot_symlink_outside_declared_hub_is_rejected(self):
        snapshot, blobs = self.bind_huggingface_snapshot()
        outside = Path(self.directory.name) / "outside-model.bin"
        outside.write_bytes((blobs / "blob-pytorch_model.bin").read_bytes())
        (snapshot / "pytorch_model.bin").unlink()
        (snapshot / "pytorch_model.bin").symlink_to(outside)
        with self.assertRaisesRegex(audit.ValidationError, "symlink target"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_cached_blob_mutation_is_rejected(self):
        _, blobs = self.bind_huggingface_snapshot()
        (blobs / "blob-pytorch_model.bin").write_bytes(b"mutated pretrained blob")
        with self.assertRaisesRegex(audit.ValidationError, "checksum mismatch"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_conflicting_checkpoint_selection_metadata_is_rejected(self):
        self.fixture.manifest["selected_checkpoint"] = {
            "epoch": 1, "learning_rate": 2e-5, "protocol_sha256": self.fixture.protocol_hash}
        model_config = audit.read_json(self.fixture.group / "selected_model/config.json")
        model_config["codebert_selected_epoch"] = 2
        write_json(self.fixture.group / "selected_model/config.json", model_config)
        self.fixture.save()
        with self.assertRaisesRegex(audit.ValidationError, "configuration selection metadata"):
            audit.validate_results(self.fixture.result, allow_partial=True)

    def test_actual_runner_export_and_aggregation_contract(self):
        # Exercise the real exporters without Torch, model loading, or training.
        import numpy as np
        import run_cwe_119_codebert_experiments as runner
        records = audit.read_jsonl(self.fixture.inputs["valid"])
        config = {"run_name": "fixture", "learning_rate": 2e-5}
        group = self.fixture.manifest["group"]
        scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        self.fixture.metrics = [{**runner.meta_row(config, group, 1, "validation", "75/25"),
                                 **runner.probability_metrics(np.array([1, 0, 1, 0]), scores)}]
        self.fixture.predictions = runner.prediction_rows(config, group, 1, "validation", "75/25", records, scores)
        self.fixture.save()
        write_csv(self.fixture.group / "selected_hyperparameters.csv", self.fixture.selected)
        runner.aggregate(self.fixture.result, self.fixture.top)
        report = audit.validate_results(self.fixture.result, allow_partial=True)
        self.assertEqual(report["status"], "partial_passed")
        self.assertEqual(report["completed_groups"], 1)


if __name__ == "__main__":
    unittest.main()

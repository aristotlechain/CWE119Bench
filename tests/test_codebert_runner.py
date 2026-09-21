#!/usr/bin/env python3
"""CPU-only contract tests. No pretrained weights, GPU execution, or training."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codebert-tests-mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch

import run_cwe_119_codebert_experiments as runner
from validate_cwe_119_codebert_results import validate_group, validate_results


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, *, truncation, padding, add_special_tokens, max_length=None):
        self.calls += 1
        tokens = [[0, *range(3, len(text.split()) + 3), 2] for text in texts]
        if truncation:
            tokens = [ids if len(ids) <= max_length else ids[:max_length - 1] + [2] for ids in tokens]
        return {"input_ids": tokens}


class FakeCache:
    def encode(self, rows):
        return ([{"input_ids": [0, 3, 2], "attention_mask": [1, 1, 1]} for _ in rows],
                {"n_rows": len(rows), "truncated_rows": 0, "truncation_rate": 0,
                 "max_original_tokens": 3, "mean_original_tokens": 3})


def config_for(directory):
    return {
        "schema_version": 1, "model_name": "microsoft/codebert-base",
        "model_revision": runner.MODEL_REVISION,
        "local_model_path": str(directory / runner.MODEL_REVISION),
        "model_cache_dir": str(directory / "models"), "token_cache_dir": str(directory / "tokens"),
        "noise_dir": str(directory / "noise"), "test_dir": str(directory / "DO_NOT_READ_TESTS"),
        "epochs": 5, "train_batch_size": 8, "eval_batch_size": 8, "learning_rate": 2e-5,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "max_grad_norm": 1,
        "max_length": 512, "model_seed": 0, "dataset_seeds": [1],
        "train_ratios": ["75/25"], "test_ratios": list(runner.TEST_RATIOS), "conditions": ["clean"],
        "mixed_precision": "bf16", "num_workers": 0, "torch_threads": 6,
        "deterministic": True, "decision_threshold": 0.5, "run_name": "fixture",
    }


def create_pinned_receipt(config):
    snapshot = Path(config["local_model_path"])
    snapshot.mkdir(parents=True)
    files = []
    for name in runner.MODEL_FILES:
        path = snapshot / name
        path.write_bytes(("fixture " + name).encode())
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": runner.sha256_file(path)})
    receipt = {"model_name": config["model_name"], "model_revision": runner.MODEL_REVISION,
               "remote_revision_match": True, "pytorch_weights_verified_against_remote_lfs_sha256": True,
               "snapshot_path": str(snapshot), "files": files}
    receipt_path = Path(config["model_cache_dir"]).parent / "pinned_model_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return runner.verify_pinned_artifacts(config)


class RunnerTests(unittest.TestCase):
    def test_training_labels_do_not_fall_back_to_reference(self):
        rows = [{"binary_label": 0, "noisy_label": 1}, {"binary_label": 1, "noisy_label": 0}]
        np.testing.assert_array_equal(runner.labels_from_rows(rows, "noisy_label"), [1, 0])
        with self.assertRaises(KeyError):
            runner.labels_from_rows([{"binary_label": 1}], "noisy_label")
        with self.assertRaises(ValueError):
            runner.labels_from_rows([{"noisy_label": 0.7}], "noisy_label")

    def test_class_weights_use_observed_learner_labels(self):
        rows = [{"binary_label": label, "noisy_label": noisy}
                for label, noisy in zip([0, 0, 0, 1], [0, 1, 1, 1])]
        counts, weights = runner.balanced_weights(runner.labels_from_rows(rows, "noisy_label"))
        self.assertEqual(counts, [1, 3])
        np.testing.assert_allclose(weights, [2, 2 / 3])
        logits = torch.tensor([[0., 1.], [1., 0.], [0., 1.], [0., 1.]])
        labels = torch.tensor([0, 1, 1, 1])
        loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))(logits, labels)
        individual = -torch.log_softmax(logits, dim=1)[range(4), labels]
        applied_weights = torch.tensor(weights)[labels]
        self.assertAlmostEqual(float(loss), float((individual * applied_weights).sum() / applied_weights.sum()), places=6)
        with self.assertRaises(ValueError):
            runner.balanced_weights(np.array([1, 1]))

    def test_threshold_and_pr_auc_remain_distinct_from_ap(self):
        metrics = runner.probability_metrics(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8]))
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["pr_auc"], 19 / 24)
        boundary = runner.probability_metrics(np.array([0, 1]), np.array([.499999, .5]))
        self.assertEqual((boundary["tn"], boundary["tp"]), (1, 1))
        with self.assertRaises(ValueError):
            runner.probability_metrics(np.array([0, 1]), np.array([np.nan, .5]))

    def test_exact_epoch_tie_keeps_earliest(self):
        metrics = {"mcc": .1, "f1": .2, "recall": .3, "precision": .4, "balanced_accuracy": .5}
        history = [metrics.copy() for _ in range(5)]
        best_epoch, best_key = None, None
        for epoch, item in enumerate(history, 1):
            key = runner.selection_key(item)
            if best_key is None or key > best_key:
                best_epoch, best_key = epoch, key
        self.assertEqual(best_epoch, 1)
        changed = dict(metrics, recall=.31)
        self.assertGreater(runner.selection_key(changed), runner.selection_key(metrics))

    def test_token_cache_has_no_labels_and_preserves_long_functions(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FakeTokenizer()
            config = config_for(Path(temp))
            cache = runner.TokenCache(Path(temp) / "tokens", tokenizer, config)
            rows = [{"func": "x " * 700, "noisy_label": 0}, {"func": "", "noisy_label": 1}]
            encoded, stats = cache.encode(rows)
            self.assertEqual(len(encoded), 2)
            self.assertEqual(len(encoded[0]["input_ids"]), 512)
            self.assertEqual(encoded[0]["input_ids"][-1], 2)
            self.assertEqual(stats["truncated_rows"], 1)
            self.assertEqual(set(encoded[0]), {"input_ids", "attention_mask"})
            call_count = tokenizer.calls
            mutated = copy.deepcopy(rows)
            mutated[0]["noisy_label"] = 1
            mutated[0]["binary_label"] = 0
            self.assertEqual(cache.encode(mutated), (encoded, stats))
            self.assertEqual(tokenizer.calls, call_count)
            cache.close()
            reopened = runner.TokenCache(Path(temp) / "tokens", tokenizer, config)
            self.assertEqual(reopened.encode(mutated), (encoded, stats))
            self.assertEqual(tokenizer.calls, call_count)
            reopened.close()

    def test_pinned_artifact_mutation_and_unsigned_preferred_weights_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            digest = create_pinned_receipt(config)
            self.assertEqual(set(digest), runner.MODEL_FILES)
            vocab = Path(config["local_model_path"]) / "vocab.json"
            original = vocab.read_bytes()
            vocab.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runner.verify_pinned_artifacts(config)
            vocab.write_bytes(original)
            (Path(config["local_model_path"]) / "model.safetensors").write_bytes(b"unsigned override")
            with self.assertRaisesRegex(ValueError, "unverified loader artifacts"):
                runner.verify_pinned_artifacts(config)

    def test_tokenizer_artifact_change_cannot_hit_previous_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FakeTokenizer()
            config = config_for(Path(temp))
            config["model_artifact_sha256"] = {"vocab.json": "old digest"}
            cache = runner.TokenCache(Path(temp) / "tokens", tokenizer, config)
            cache.encode([{"func": "one two"}])
            calls = tokenizer.calls
            cache.close()
            config["model_artifact_sha256"] = {"vocab.json": "new digest"}
            cache = runner.TokenCache(Path(temp) / "tokens", tokenizer, config)
            cache.encode([{"func": "one two"}])
            self.assertGreater(tokenizer.calls, calls)
            cache.close()

    def test_config_rejects_seed_zero_and_wrong_model_revision(self):
        config = config_for(Path("/tmp/unused"))
        runner.validate_config(config)
        for field, value in (("dataset_seeds", [0]), ("model_revision", "main"),
                             ("max_length", 400), ("decision_threshold", .4), ("num_workers", 1)):
            changed = dict(config, **{field: value})
            with self.assertRaises(ValueError):
                runner.validate_config(changed)

    def test_default_cli_never_enables_test_access(self):
        with patch("sys.argv", ["runner", "--config", "a.json", "--output-dir", "out"]):
            self.assertFalse(runner.parse_args().evaluate_tests)

    def test_loading_requires_intact_encoder_and_fresh_seeded_head(self):
        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(4, 2)
                for parameter in self.parameters():
                    parameter.requires_grad_(False)

            def to(self, device):
                return self  # Keep this contract test entirely on CPU.

        info = {"missing_keys": ["classifier.dense.bias", "classifier.dense.weight",
                                 "classifier.out_proj.bias", "classifier.out_proj.weight"],
                "unexpected_keys": ["pooler.dense.bias", "pooler.dense.weight"],
                "mismatched_keys": [], "error_msgs": []}
        def factory(*args, **kwargs):
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            return StubModel(), copy.deepcopy(info)
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained", side_effect=factory):
            first, first_info = runner.load_model(torch, config_for(Path("/tmp/unused")), Path("/tmp/unused"), pretrained=True)
            _, second_info = runner.load_model(torch, config_for(Path("/tmp/unused")), Path("/tmp/unused"), pretrained=True)
            self.assertEqual(first_info["classifier_initialization_sha256"], second_info["classifier_initialization_sha256"])
            self.assertTrue(all(parameter.requires_grad for parameter in first.parameters()))
            info["missing_keys"].append("roberta.encoder.layer.0.attention.self.query.weight")
            with self.assertRaisesRegex(ValueError, "encoder did not load intact"):
                runner.load_model(torch, config_for(Path("/tmp/unused")), Path("/tmp/unused"), pretrained=True)

    def test_rng_roundtrip_preserves_cpu_and_loader_sequences(self):
        # CUDA state calls are replaced with empty CPU-only states.
        generator = torch.Generator().manual_seed(0)
        runner.seed_everything(torch, 0)
        with patch.object(torch.cuda, "get_rng_state_all", return_value=[]), patch.object(torch.cuda, "set_rng_state_all"):
            state = runner.rng_state(torch, generator)
            expected = (np.random.rand(), torch.rand(4), torch.randperm(8, generator=generator))
            runner.restore_rng(torch, generator, state)
            self.assertEqual(np.random.rand(), expected[0])
            self.assertTrue(torch.equal(torch.rand(4), expected[1]))
            self.assertTrue(torch.equal(torch.randperm(8, generator=generator), expected[2]))

    def create_completed_fixture(self, directory):
        config = config_for(directory)
        config["model_artifact_sha256"] = create_pinned_receipt(config)
        output = directory / "output"
        group = {"train_ratio": "75/25", "seed": 1, "condition": "clean"}
        condition = Path(config["noise_dir"]) / "ratio_75_25/seed_1/clean"
        condition.mkdir(parents=True)
        rows = [{"func": f"function {idx}", "idx": idx, "func_hash": str(idx),
                 "commit_id": "commit", "cve": None, "file_hash": None,
                 "source_primevul_file": "fixture", "binary_label": label,
                 "noisy_label": noisy, "benchmark_ratio": "75/25", "benchmark_seed": 1,
                 "benchmark_row_role": "original"}
                for idx, (label, noisy) in enumerate(zip([0, 0, 1, 1], [0, 1, 1, 1]))]
        for name in ("train", "valid"):
            (condition / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        group_dir = output / "groups/ratio_75_25/seed_1/clean"
        model_dir = group_dir / "selected_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"fixture-only: no model tensors")
        scores = np.array([.1, .4, .35, .8])
        metrics = runner.probability_metrics(np.array([0, 0, 1, 1]), scores)
        counts, weights = runner.balanced_weights(np.array([0, 1, 1, 1]))
        history = [{**group, "epoch": epoch, **metrics} for epoch in range(1, 6)]
        hashes = {"protocol_sha256": runner.content_hash(config), "config_sha256": runner.content_hash(config),
                  "code_sha256": runner.code_hashes(), "model_artifact_sha256": config["model_artifact_sha256"]}
        manifest = {"schema_version": 1, "status": "running", "group": group, **hashes,
                    "protocol": config, "effective_config": config, "training_seed": 0,
                    "input_paths": {name: str(condition / f"{name}.jsonl") for name in ("train", "valid")},
                    "input_sha256": {name: runner.sha256_file(condition / f"{name}.jsonl") for name in ("train", "valid")},
                    "selected_epoch": 1, "target_epochs": 5, "training_epochs_completed": 5,
                    "train_rows": 4, "valid_rows": 4, "class_weights": weights,
                    "noisy_label_counts": {"0": counts[0], "1": counts[1]},
                    "best_metrics": metrics, "selected_validation_scores": scores.tolist(),
                    "validation_history": history, "elapsed_seconds": .2, "truncation_stats": {},
                    "selected_checkpoint": {"epoch": 1, "learning_rate": 2e-5,
                                            "protocol_sha256": hashes["protocol_sha256"]}}
        manifest["provenance_sha256"] = runner.content_hash({**hashes, "group": group,
                                                             "training_input_sha256": manifest["input_sha256"]})
        with patch.object(runner, "load_model", return_value=(object(), {})), \
             patch.object(runner, "make_loader", return_value=None), \
             patch.object(runner, "score_model", return_value=scores), \
             patch.object(torch.cuda, "empty_cache"):
            completed = runner.evaluate_and_finish(torch, config, group_dir, manifest,
                                                   FakeTokenizer(), FakeCache(), False)
        return output, group_dir, config, completed

    def test_validation_only_export_passes_independent_audit_without_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            output, group_dir, config, manifest = self.create_completed_fixture(Path(temp))
            # This test directory never exists: touching any test input would fail.
            self.assertFalse(Path(config["test_dir"]).exists())
            self.assertEqual(manifest["evaluation_status"], "validation_only")
            report = validate_group(group_dir, allow_partial=True)
            self.assertEqual(report["status"], "partial_passed")
            self.assertEqual(report["counts"]["test_score_rows"], 0)
            top = {"status": "complete", "protocol": config, "protocol_sha256": runner.content_hash(config),
                   "config_sha256": runner.content_hash(config), "code_sha256": runner.code_hashes(),
                   "model_artifact_sha256": config["model_artifact_sha256"],
                   "matrix": {"train_ratios": ["75/25"], "seeds": [1], "conditions": ["clean"]}}
            runner.aggregate(output, top)
            report = validate_results(output, allow_partial=True)
            self.assertEqual(report["status"], "partial_passed")

    def test_completed_output_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            _, group_dir, _, manifest = self.create_completed_fixture(Path(temp))
            runner.checked_outputs(group_dir, manifest)
            with (group_dir / "prediction_scores.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runner.checked_outputs(group_dir, manifest)

    def test_resume_completed_validation_group_does_not_initialize_or_train(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _, config, manifest = self.create_completed_fixture(Path(temp))
            hashes = {name: manifest[name] for name in ("protocol_sha256", "config_sha256", "code_sha256", "model_artifact_sha256")}
            with patch.object(runner, "load_model", side_effect=AssertionError("Unexpected model initialization")):
                completed = runner.run_group(torch, config, output, manifest["group"], FakeTokenizer(), FakeCache(), hashes,
                                              resume=True, evaluate_tests=False)
            self.assertEqual(completed["evaluation_status"], "validation_only")

    def test_explicit_test_addition_reuses_selected_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output, group_dir, config, manifest = self.create_completed_fixture(Path(temp))
            rows = runner.load_jsonl(Path(manifest["input_paths"]["valid"]))
            for ratio in config["test_ratios"]:
                path = Path(config["test_dir"]) / runner.ratio_dir_name(ratio) / "seed_1/test.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("\n".join(json.dumps(dict(row, benchmark_ratio=ratio)) for row in rows) + "\n")
            hashes = {name: manifest[name] for name in ("protocol_sha256", "config_sha256", "code_sha256", "model_artifact_sha256")}
            scores = np.asarray(manifest["selected_validation_scores"])
            with patch.object(runner, "load_model", return_value=(object(), {})) as loading, \
                 patch.object(runner, "make_loader", return_value=None), \
                 patch.object(runner, "score_model", return_value=scores), \
                 patch.object(torch.cuda, "empty_cache"):
                completed = runner.run_group(torch, config, output, manifest["group"], FakeTokenizer(), FakeCache(), hashes,
                                              resume=True, evaluate_tests=True)
            self.assertEqual(loading.call_count, 1)
            self.assertFalse(loading.call_args.kwargs["pretrained"])
            self.assertEqual(completed["evaluation_status"], "tests_complete")
            report = validate_group(group_dir, allow_partial=True)
            self.assertEqual(report["counts"]["test_cells"], 6)
            self.assertEqual(report["counts"]["test_score_rows"], 24)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Generate ratio-preserved noisy train-only variants for CWE-119.

Policy:
  - start from the clean CWE-119 imbalance benchmark
  - apply label noise only to training rows
  - preserve clean truth in binary_label
  - store the training label in noisy_label
  - keep validation rows clean
  - do not duplicate clean test files here. Model runs reuse clean tests from
    data/processed/cwe_119_imbalance
  - for false-positive noise, add real unused benign rows from the original
    PrimeVul train split so the final noisy_label ratio is preserved
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TARGET_CWE = "CWE-119"
TRAIN_SOURCE_FILE = "primevul_train.jsonl"
VALID_SOURCE_FILE = "primevul_valid.jsonl"

RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
SEEDS = [1, 2, 3, 4, 5]

CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "clean": {"type": "clean", "level": 0.0},
    "random_flip_10": {"type": "random_flip", "level": 0.10},
    "random_flip_20": {"type": "random_flip", "level": 0.20},
    "random_fp_10": {"type": "random_fp", "level": 0.10},
    "random_fp_20": {"type": "random_fp", "level": 0.20},
    "heuristic_fp_10": {"type": "heuristic_fp", "level": 0.10},
    "heuristic_fp_20": {"type": "heuristic_fp", "level": 0.20},
}
CONDITIONS = list(CONDITION_SPECS)

HIGH_RISK_TOKENS = [
    "strcpy(",
    "strncpy(",
    "strcat(",
    "strncat(",
    "sprintf(",
    "snprintf(",
    "vsprintf(",
    "vsnprintf(",
    "gets(",
    "memcpy(",
    "memmove(",
]

MEDIUM_RISK_TOKENS = [
    "malloc(",
    "calloc(",
    "realloc(",
    "free(",
    "alloca(",
    "scanf(",
    "sscanf(",
    "recv(",
    "read(",
    "write(",
    "send(",
    "strlen(",
    "sizeof(",
]

WEAK_CONTEXT_TOKENS = [
    "buffer",
    "buf",
    "pointer",
    "ptr",
    "length",
    "len",
    "size",
    "copy",
    "parse",
]


def ratio_dir_name(ratio: str) -> str:
    return "ratio_" + ratio.replace("/", "_")


def parse_ratio(ratio: str) -> tuple[int, int]:
    try:
        benign_text, vulnerable_text = ratio.split("/", maxsplit=1)
        benign = int(benign_text)
        vulnerable = int(vulnerable_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ratio must look like '75/25', got {ratio!r}"
        ) from exc

    if benign <= 0 or vulnerable <= 0 or benign + vulnerable != 100:
        raise argparse.ArgumentTypeError(
            f"Ratio values must be positive and add to 100, got {ratio!r}"
        )
    return benign, vulnerable


def level_text(noise_level: float) -> str:
    return f"{noise_level:.2f}"


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def stable_sample_indices(
    indices: list[int],
    sample_size: int,
    *,
    seed_material: str,
) -> set[int]:
    if sample_size > len(indices):
        raise ValueError(
            f"Cannot sample {sample_size} rows from {len(indices)} candidates."
        )
    rng = random.Random(stable_hash_int(seed_material))
    return set(rng.sample(indices, sample_size))


def suspicious_score(func: str) -> int:
    text = func.lower()
    score = 0
    score += 2 * sum(1 for token in HIGH_RISK_TOKENS if token in text)
    score += sum(1 for token in MEDIUM_RISK_TOKENS if token in text)
    weak_hits = sum(1 for token in WEAK_CONTEXT_TOKENS if token in text)
    score += min(2, weak_hits)
    return score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_identity(row: dict[str, Any], *, source_file: str | None = None) -> str:
    source = source_file or row.get("source_primevul_file") or ""
    parts = [
        source,
        str(row.get("idx", "")),
        str(row.get("func_hash", "")),
        str(row.get("commit_id", "")),
        str(row.get("cve", "")),
        str(row.get("file_hash", "")),
    ]
    return "\x1f".join(parts)


def row_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    source = str(row.get("source_primevul_file") or "")
    try:
        idx = int(row.get("idx"))
    except (TypeError, ValueError):
        idx = 0
    return source, idx, row_identity(row)


def load_raw_train_benign_pool(raw_dir: Path) -> list[dict[str, Any]]:
    paired_files = sorted(raw_dir.glob("*paired*.jsonl"))
    if paired_files:
        names = ", ".join(path.name for path in paired_files)
        raise SystemExit(
            "Paired files are present in the raw input directory. "
            f"Remove them before generating final noisy data: {names}"
        )

    train_path = raw_dir / TRAIN_SOURCE_FILE
    if not train_path.exists():
        raise SystemExit(f"Missing required raw train file: {train_path}")

    rows: list[dict[str, Any]] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target") != 0:
                continue
            out = dict(row)
            out["_raw_line_number"] = line_number
            rows.append(out)
    return rows


def annotate_base_row(
    row: dict[str, Any],
    *,
    condition_name: str,
    noise_type: str,
    noise_level: float,
    split: str,
    selected: bool,
    flip_direction: str,
    noisy_label: int,
    heuristic_score: int | None = None,
) -> dict[str, Any]:
    out = dict(row)
    clean_label = int(out["binary_label"])
    if clean_label != int(row["binary_label"]):
        raise ValueError("Internal label mismatch.")

    out["noisy_label"] = int(noisy_label)
    out["noise_condition"] = condition_name
    out["noise_type"] = noise_type
    out["noise_level"] = noise_level
    out["noise_applied"] = selected
    out["noise_flip_direction"] = flip_direction if selected else "none"
    out["noise_split_policy"] = "train_only" if split == "train" else "clean_validation"
    out["is_compensation_benign"] = False

    if selected and flip_direction == "0_to_1":
        out["noise_row_role"] = "flipped_base_benign"
    elif selected and flip_direction == "1_to_0":
        out["noise_row_role"] = "flipped_base_vulnerable"
    else:
        out["noise_row_role"] = "base_clean_label"

    if heuristic_score is not None:
        out["heuristic_fp_score"] = heuristic_score
    return out


def annotate_compensation_row(
    row: dict[str, Any],
    *,
    ratio: str,
    dataset_seed: int,
    condition_name: str,
    noise_type: str,
    noise_level: float,
    target_cwe: str,
) -> dict[str, Any]:
    out = {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }
    out["split"] = "train"
    out["binary_label"] = 0
    out["noisy_label"] = 0
    out["benchmark_target_cwe"] = target_cwe
    out["benchmark_ratio"] = ratio
    out["benchmark_seed"] = dataset_seed
    out["benchmark_row_role"] = "compensation_benign"
    out["source_primevul_file"] = TRAIN_SOURCE_FILE
    out["noise_condition"] = condition_name
    out["noise_type"] = noise_type
    out["noise_level"] = noise_level
    out["noise_applied"] = False
    out["noise_flip_direction"] = "none"
    out["noise_split_policy"] = "train_only"
    out["noise_row_role"] = "compensation_benign"
    out["is_compensation_benign"] = True
    return out


def count_labels(rows: list[dict[str, Any]], label_field: str) -> Counter[int]:
    return Counter(int(row[label_field]) for row in rows)


def choose_compensation_rows(
    *,
    raw_train_benign_pool: list[dict[str, Any]],
    clean_train_ids: set[str],
    sample_size: int,
    ratio: str,
    dataset_seed: int,
    noise_type: str,
    noise_level: float,
) -> tuple[list[dict[str, Any]], str | None]:
    if sample_size == 0:
        return [], None

    candidates = [
        row
        for row in raw_train_benign_pool
        if row_identity(row, source_file=TRAIN_SOURCE_FILE) not in clean_train_ids
    ]
    if sample_size > len(candidates):
        raise SystemExit(
            f"Not enough unused train benign rows for compensation in {ratio} "
            f"seed {dataset_seed} {noise_type} {level_text(noise_level)}: "
            f"needed {sample_size}, available {len(candidates)}"
        )

    seed_material = (
        f"{TARGET_CWE}|{ratio}|{dataset_seed}|{noise_type}|"
        f"{level_text(noise_level)}|compensation_benign|train"
    )
    selected_positions = stable_sample_indices(
        list(range(len(candidates))),
        sample_size,
        seed_material=seed_material,
    )
    selected = [candidates[index] for index in selected_positions]
    return selected, seed_material


def expected_false_positive_final_counts(
    *,
    clean_positive: int,
    clean_benign: int,
    flipped_0_to_1: int,
) -> tuple[int, int, int]:
    expected_noisy_positive = clean_positive + flipped_0_to_1
    expected_noisy_benign = math.ceil(
        clean_benign * expected_noisy_positive / clean_positive
    )
    compensation_needed = expected_noisy_benign - (clean_benign - flipped_0_to_1)
    return expected_noisy_positive, expected_noisy_benign, compensation_needed


def select_noise_indices(
    train_rows: list[dict[str, Any]],
    *,
    ratio: str,
    dataset_seed: int,
    condition_name: str,
    noise_type: str,
    noise_level: float,
) -> tuple[set[int], set[int], dict[int, int], dict[str, Any]]:
    positive_indices = [
        index
        for index, row in enumerate(train_rows)
        if int(row["binary_label"]) == 1
    ]
    benign_indices = [
        index
        for index, row in enumerate(train_rows)
        if int(row["binary_label"]) == 0
    ]

    seed_materials: dict[str, str] = {}
    heuristic_scores: dict[int, int] = {}
    heuristic_candidate_count = 0

    if noise_type == "clean":
        return set(), set(), heuristic_scores, {
            "rows_selected_for_noise": 0,
            "requested_noise_rows": 0,
            "heuristic_candidate_rows": 0,
            "seed_materials": seed_materials,
        }

    total_train = len(train_rows)

    if noise_type == "random_flip":
        pair_count = math.ceil(noise_level * total_train / 2)
        if pair_count > len(positive_indices) or pair_count > len(benign_indices):
            raise SystemExit(
                f"Not enough rows for paired random flip in {ratio} seed "
                f"{dataset_seed} {condition_name}: pair_count={pair_count}, "
                f"positives={len(positive_indices)}, benign={len(benign_indices)}"
            )

        seed_0_to_1 = (
            f"{TARGET_CWE}|{ratio}|{dataset_seed}|random_flip|"
            f"{level_text(noise_level)}|0_to_1|train"
        )
        seed_1_to_0 = (
            f"{TARGET_CWE}|{ratio}|{dataset_seed}|random_flip|"
            f"{level_text(noise_level)}|1_to_0|train"
        )
        selected_0_to_1 = stable_sample_indices(
            benign_indices,
            pair_count,
            seed_material=seed_0_to_1,
        )
        selected_1_to_0 = stable_sample_indices(
            positive_indices,
            pair_count,
            seed_material=seed_1_to_0,
        )
        seed_materials["random_flip_0_to_1"] = seed_0_to_1
        seed_materials["random_flip_1_to_0"] = seed_1_to_0
        return selected_0_to_1, selected_1_to_0, heuristic_scores, {
            "rows_selected_for_noise": 2 * pair_count,
            "requested_noise_rows": math.ceil(noise_level * total_train),
            "pair_count": pair_count,
            "heuristic_candidate_rows": 0,
            "seed_materials": seed_materials,
        }

    if noise_type == "random_fp":
        rows_to_flip = math.ceil(noise_level * total_train)
        seed_material = (
            f"{TARGET_CWE}|{ratio}|{dataset_seed}|random_fp|"
            f"{level_text(noise_level)}|train"
        )
        selected_0_to_1 = stable_sample_indices(
            benign_indices,
            rows_to_flip,
            seed_material=seed_material,
        )
        seed_materials["random_fp_0_to_1"] = seed_material
        return selected_0_to_1, set(), heuristic_scores, {
            "rows_selected_for_noise": rows_to_flip,
            "requested_noise_rows": rows_to_flip,
            "heuristic_candidate_rows": 0,
            "seed_materials": seed_materials,
        }

    if noise_type == "heuristic_fp":
        rows_to_flip = math.ceil(noise_level * total_train)
        candidates: list[tuple[int, int, str]] = []
        for index in benign_indices:
            row = train_rows[index]
            score = suspicious_score(row.get("func") or "")
            if score < 2:
                continue
            heuristic_scores[index] = score
            tie_material = (
                f"{TARGET_CWE}|{ratio}|{dataset_seed}|heuristic_fp|"
                f"{level_text(noise_level)}|{row.get('idx')}|"
                f"{row.get('func_hash')}|train"
            )
            candidates.append((index, score, hashlib.sha256(tie_material.encode("utf-8")).hexdigest()))

        heuristic_candidate_count = len(candidates)
        if rows_to_flip > heuristic_candidate_count:
            raise SystemExit(
                f"Not enough heuristic false-positive candidates for {ratio} "
                f"seed {dataset_seed} {condition_name}: needed {rows_to_flip}, "
                f"available {heuristic_candidate_count}"
            )

        ranked = sorted(candidates, key=lambda item: (-item[1], item[2], item[0]))
        selected_0_to_1 = {index for index, _score, _tie in ranked[:rows_to_flip]}
        return selected_0_to_1, set(), heuristic_scores, {
            "rows_selected_for_noise": rows_to_flip,
            "requested_noise_rows": rows_to_flip,
            "heuristic_candidate_rows": heuristic_candidate_count,
            "seed_materials": {
                "heuristic_fp_tie_break": (
                    f"{TARGET_CWE}|{ratio}|{dataset_seed}|heuristic_fp|"
                    f"{level_text(noise_level)}|{{idx}}|{{func_hash}}|train"
                )
            },
        }

    raise ValueError(f"Unexpected noise type: {noise_type}")


def create_condition_rows(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    raw_train_benign_pool: list[dict[str, Any]],
    *,
    ratio: str,
    dataset_seed: int,
    condition_name: str,
    condition: dict[str, Any],
    target_cwe: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    noise_type = condition["type"]
    noise_level = float(condition["level"])
    clean_counts = count_labels(train_rows, "binary_label")
    clean_positive = clean_counts[1]
    clean_benign = clean_counts[0]
    clean_train_ids = {
        row_identity(row, source_file=row.get("source_primevul_file"))
        for row in train_rows
    }

    selected_0_to_1, selected_1_to_0, heuristic_scores, selection_meta = (
        select_noise_indices(
            train_rows,
            ratio=ratio,
            dataset_seed=dataset_seed,
            condition_name=condition_name,
            noise_type=noise_type,
            noise_level=noise_level,
        )
    )

    output_train: list[dict[str, Any]] = []
    flips_0_to_1 = 0
    flips_1_to_0 = 0

    for index, row in enumerate(train_rows):
        clean_label = int(row["binary_label"])
        if index in selected_0_to_1:
            flip_direction = "0_to_1"
            noisy_label = 1
            selected = True
            flips_0_to_1 += 1
        elif index in selected_1_to_0:
            flip_direction = "1_to_0"
            noisy_label = 0
            selected = True
            flips_1_to_0 += 1
        else:
            flip_direction = "none"
            noisy_label = clean_label
            selected = False

        output_train.append(
            annotate_base_row(
                row,
                condition_name=condition_name,
                noise_type=noise_type,
                noise_level=noise_level,
                split="train",
                selected=selected,
                flip_direction=flip_direction,
                noisy_label=noisy_label,
                heuristic_score=heuristic_scores.get(index),
            )
        )

    compensation_seed_material = None
    compensation_rows: list[dict[str, Any]] = []
    expected_noisy_positive: int
    expected_noisy_benign: int
    compensation_needed = 0

    if noise_type in {"random_fp", "heuristic_fp"}:
        (
            expected_noisy_positive,
            expected_noisy_benign,
            compensation_needed,
        ) = expected_false_positive_final_counts(
            clean_positive=clean_positive,
            clean_benign=clean_benign,
            flipped_0_to_1=flips_0_to_1,
        )
        selected_compensation, compensation_seed_material = choose_compensation_rows(
            raw_train_benign_pool=raw_train_benign_pool,
            clean_train_ids=clean_train_ids,
            sample_size=compensation_needed,
            ratio=ratio,
            dataset_seed=dataset_seed,
            noise_type=noise_type,
            noise_level=noise_level,
        )
        compensation_rows = [
            annotate_compensation_row(
                row,
                ratio=ratio,
                dataset_seed=dataset_seed,
                condition_name=condition_name,
                noise_type=noise_type,
                noise_level=noise_level,
                target_cwe=target_cwe,
            )
            for row in selected_compensation
        ]
        output_train.extend(compensation_rows)
    else:
        expected_noisy_positive = clean_positive
        expected_noisy_benign = clean_benign

    output_train.sort(key=row_sort_key)

    output_valid = [
        annotate_base_row(
            row,
            condition_name=condition_name,
            noise_type=noise_type,
            noise_level=0.0,
            split="valid",
            selected=False,
            flip_direction="none",
            noisy_label=int(row["binary_label"]),
        )
        for row in valid_rows
    ]

    noisy_counts = count_labels(output_train, "noisy_label")
    clean_output_counts = count_labels(output_train, "binary_label")
    valid_clean_counts = count_labels(output_valid, "binary_label")

    final_total = noisy_counts[0] + noisy_counts[1]
    final_vulnerable_pct = noisy_counts[1] / final_total * 100
    final_benign_pct = noisy_counts[0] / final_total * 100

    if noisy_counts[1] != expected_noisy_positive:
        raise SystemExit(
            f"Noisy positive mismatch for {ratio} seed {dataset_seed} "
            f"{condition_name}: expected {expected_noisy_positive}, "
            f"got {noisy_counts[1]}"
        )
    if noisy_counts[0] != expected_noisy_benign:
        raise SystemExit(
            f"Noisy benign mismatch for {ratio} seed {dataset_seed} "
            f"{condition_name}: expected {expected_noisy_benign}, "
            f"got {noisy_counts[0]}"
        )

    metadata = {
        "ratio": ratio,
        "dataset_seed": dataset_seed,
        "condition": condition_name,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "target_cwe": target_cwe,
        "source_policy": {
            "train_source": TRAIN_SOURCE_FILE,
            "valid_source": VALID_SOURCE_FILE,
            "paired_files_used": False,
            "test_files_copied": False,
            "test_policy": (
                "Clean tests are reused from data/processed/cwe_119_imbalance."
            ),
        },
        "label_policy": {
            "clean_truth_label": "binary_label",
            "training_label": "noisy_label",
            "validation_label": "binary_label",
            "test_label": "binary_label",
        },
        "train": {
            "base_total_rows": len(train_rows),
            "clean_positive_rows": clean_positive,
            "clean_benign_rows": clean_benign,
            "base_binary_positive_rows_after_compensation": clean_output_counts[1],
            "base_binary_benign_rows_after_compensation": clean_output_counts[0],
            "requested_noise_rows": selection_meta["requested_noise_rows"],
            "rows_selected_for_noise": selection_meta["rows_selected_for_noise"],
            "pair_count": selection_meta.get("pair_count", 0),
            "flips_0_to_1": flips_0_to_1,
            "flips_1_to_0": flips_1_to_0,
            "compensation_benign_rows": len(compensation_rows),
            "expected_compensation_benign_rows": compensation_needed,
            "final_total_rows": final_total,
            "noisy_positive_rows": noisy_counts[1],
            "noisy_benign_rows": noisy_counts[0],
            "final_vulnerable_pct": final_vulnerable_pct,
            "final_benign_pct": final_benign_pct,
            "ratio_preserved": True,
            "heuristic_candidate_rows": selection_meta["heuristic_candidate_rows"],
        },
        "valid": {
            "total_rows": len(output_valid),
            "clean_positive_rows": valid_clean_counts[1],
            "clean_benign_rows": valid_clean_counts[0],
            "rows_selected_for_noise": 0,
            "clean_validation": True,
        },
        "determinism": {
            "seed_materials": selection_meta["seed_materials"],
            "compensation_seed_material": compensation_seed_material,
        },
    }
    return output_train, output_valid, metadata


def verify_condition_output(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    clean_train_ids: set[str],
) -> None:
    ratio = metadata["ratio"]
    seed = metadata["dataset_seed"]
    condition = metadata["condition"]

    train_source_files = {row.get("source_primevul_file") for row in train_rows}
    if train_source_files != {TRAIN_SOURCE_FILE}:
        raise SystemExit(
            f"Train source boundary failure for {ratio} seed {seed} {condition}: "
            f"{train_source_files}"
        )

    valid_source_files = {row.get("source_primevul_file") for row in valid_rows}
    if valid_source_files != {VALID_SOURCE_FILE}:
        raise SystemExit(
            f"Valid source boundary failure for {ratio} seed {seed} {condition}: "
            f"{valid_source_files}"
        )

    seen_ids: set[str] = set()
    duplicate_count = 0
    compensation_ids: set[str] = set()
    for row in train_rows:
        identity = row_identity(row)
        if identity in seen_ids:
            duplicate_count += 1
        seen_ids.add(identity)
        if row.get("is_compensation_benign"):
            compensation_ids.add(identity)
            if int(row.get("target")) != 0 or int(row["binary_label"]) != 0:
                raise SystemExit(
                    f"Invalid compensation row label in {ratio} seed {seed} {condition}"
                )
            if int(row["noisy_label"]) != 0:
                raise SystemExit(
                    f"Compensation row has noisy_label != 0 in {ratio} seed {seed} {condition}"
                )

    if duplicate_count:
        raise SystemExit(
            f"Duplicate train row identities in {ratio} seed {seed} {condition}: "
            f"{duplicate_count}"
        )

    reused_compensation = compensation_ids & clean_train_ids
    if reused_compensation:
        raise SystemExit(
            f"Compensation reused clean train rows in {ratio} seed {seed} {condition}: "
            f"{len(reused_compensation)}"
        )

    dirty_valid_rows = [
        row
        for row in valid_rows
        if row.get("noise_applied") or int(row["noisy_label"]) != int(row["binary_label"])
    ]
    if dirty_valid_rows:
        raise SystemExit(
            f"Validation noise detected in {ratio} seed {seed} {condition}: "
            f"{len(dirty_valid_rows)} rows"
        )

    noisy_counts = count_labels(train_rows, "noisy_label")
    expected_positive = metadata["train"]["noisy_positive_rows"]
    expected_benign = metadata["train"]["noisy_benign_rows"]
    if noisy_counts[1] != expected_positive or noisy_counts[0] != expected_benign:
        raise SystemExit(
            f"Metadata/count mismatch in {ratio} seed {seed} {condition}: "
            f"metadata=({expected_positive}, {expected_benign}), "
            f"actual=({noisy_counts[1]}, {noisy_counts[0]})"
        )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    out = dict(metadata)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def summary_row(metadata: dict[str, Any]) -> dict[str, Any]:
    train = metadata["train"]
    valid = metadata["valid"]
    return {
        "target_cwe": metadata["target_cwe"],
        "ratio": metadata["ratio"],
        "seed": metadata["dataset_seed"],
        "condition": metadata["condition"],
        "noise_type": metadata["noise_type"],
        "noise_level": metadata["noise_level"],
        "base_train_rows": train["base_total_rows"],
        "final_train_rows": train["final_total_rows"],
        "valid_rows": valid["total_rows"],
        "requested_noise_rows": train["requested_noise_rows"],
        "rows_selected_for_noise": train["rows_selected_for_noise"],
        "pair_count": train["pair_count"],
        "flips_0_to_1": train["flips_0_to_1"],
        "flips_1_to_0": train["flips_1_to_0"],
        "compensation_benign_rows": train["compensation_benign_rows"],
        "clean_train_positive_rows": train["clean_positive_rows"],
        "clean_train_benign_rows": train["clean_benign_rows"],
        "noisy_train_positive_rows": train["noisy_positive_rows"],
        "noisy_train_benign_rows": train["noisy_benign_rows"],
        "final_vulnerable_pct": train["final_vulnerable_pct"],
        "final_benign_pct": train["final_benign_pct"],
        "heuristic_candidate_rows": train["heuristic_candidate_rows"],
        "validation_clean": valid["clean_validation"],
        "ratio_preserved": train["ratio_preserved"],
        "tests_copied": metadata["source_policy"]["test_files_copied"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_cwe",
        "ratio",
        "seed",
        "condition",
        "noise_type",
        "noise_level",
        "base_train_rows",
        "final_train_rows",
        "valid_rows",
        "requested_noise_rows",
        "rows_selected_for_noise",
        "pair_count",
        "flips_0_to_1",
        "flips_1_to_0",
        "compensation_benign_rows",
        "clean_train_positive_rows",
        "clean_train_benign_rows",
        "noisy_train_positive_rows",
        "noisy_train_benign_rows",
        "final_vulnerable_pct",
        "final_benign_pct",
        "heuristic_candidate_rows",
        "validation_clean",
        "ratio_preserved",
        "tests_copied",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["final_vulnerable_pct"] = f"{row['final_vulnerable_pct']:.4f}"
            formatted["final_benign_pct"] = f"{row['final_benign_pct']:.4f}"
            writer.writerow(formatted)


def write_summary_markdown(
    path: Path,
    *,
    input_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["ratio"], row["condition"])].append(row)

    lines: list[str] = []
    lines.append("# Ratio-Preserved Noise Benchmark Generation Summary")
    lines.append("")
    lines.append("This report records the corrected train-only noisy datasets.")
    lines.append("")
    lines.append("## Input and Output")
    lines.append("")
    lines.append(f"- clean input benchmarks: `{input_dir}`")
    lines.append(f"- raw PrimeVul main files: `{raw_dir}`")
    lines.append(f"- noisy train/validation output: `{output_dir}`")
    lines.append("- clean tests are not duplicated. Model runs reuse `data/processed/cwe_119_imbalance`")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append("- Noise is applied only to `train.jsonl`.")
    lines.append("- `valid.jsonl` is copied into each condition and remains clean.")
    lines.append("- `binary_label` remains the clean truth label.")
    lines.append("- `noisy_label` is the training label.")
    lines.append("- False-positive conditions add real unused benign rows from `primevul_train.jsonl`.")
    lines.append("- Paired files are not used.")
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    lines.append("| Condition | Noise type | Level | Ratio preservation |")
    lines.append("|---|---|---:|---|")
    for condition_name in CONDITIONS:
        spec = CONDITION_SPECS[condition_name]
        if spec["type"] == "clean":
            preservation = "clean labels"
        elif spec["type"] == "random_flip":
            preservation = "equal 0->1 and 1->0 flips"
        else:
            preservation = "compensation benign rows added"
        lines.append(
            f"| `{condition_name}` | {spec['type']} | "
            f"{int(spec['level'] * 100)}% | {preservation} |"
        )
    lines.append("")
    lines.append("## Output Count")
    lines.append("")
    lines.append(f"Generated {len(rows)} ratio/seed/condition combinations.")
    lines.append("")
    lines.append("## Count Summary By Ratio And Condition")
    lines.append("")
    lines.append("Ranges are across the generated dataset seeds.")
    lines.append("")
    lines.append("| Ratio | Condition | Final train rows | Noisy positives | Noisy benign | Flips 0->1 | Flips 1->0 | Compensation benign | Final vulnerable % | Final benign % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(grouped):
        ratio, condition = key
        group = grouped[key]

        def range_text(column: str) -> str:
            values = [row[column] for row in group]
            low = min(values)
            high = max(values)
            if low == high:
                return f"{low:,}" if isinstance(low, int) else f"{low:.4f}%"
            if isinstance(low, int):
                return f"{low:,}-{high:,}"
            return f"{low:.4f}-{high:.4f}%"

        lines.append(
            "| "
            f"{ratio} | "
            f"`{condition}` | "
            f"{range_text('final_train_rows')} | "
            f"{range_text('noisy_train_positive_rows')} | "
            f"{range_text('noisy_train_benign_rows')} | "
            f"{range_text('flips_0_to_1')} | "
            f"{range_text('flips_1_to_0')} | "
            f"{range_text('compensation_benign_rows')} | "
            f"{range_text('final_vulnerable_pct')} | "
            f"{range_text('final_benign_pct')} |"
        )
    lines.append("")
    lines.append("## Generated CSV")
    lines.append("")
    lines.append("- `data/manifests/noise_ratio_preserved_generation_summary.csv`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_noise_benchmarks(
    input_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    *,
    ratios: list[str],
    seeds: list[int],
    conditions: list[str],
    overwrite: bool,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_train_benign_pool = load_raw_train_benign_pool(raw_dir)
    summary_rows: list[dict[str, Any]] = []

    for ratio in ratios:
        parse_ratio(ratio)
        ratio_dir = input_dir / ratio_dir_name(ratio)
        if not ratio_dir.exists():
            raise SystemExit(f"Missing input ratio directory: {ratio_dir}")

        for dataset_seed in seeds:
            source_dir = ratio_dir / f"seed_{dataset_seed}"
            train_path = source_dir / "train.jsonl"
            valid_path = source_dir / "valid.jsonl"
            if not train_path.exists() or not valid_path.exists():
                raise SystemExit(f"Missing train/valid input files in {source_dir}")

            train_rows = load_jsonl(train_path)
            valid_rows = load_jsonl(valid_path)
            clean_train_ids = {
                row_identity(row, source_file=row.get("source_primevul_file"))
                for row in train_rows
            }

            for condition_name in conditions:
                if condition_name not in CONDITION_SPECS:
                    raise SystemExit(f"Unsupported condition: {condition_name}")
                condition = CONDITION_SPECS[condition_name]
                condition_dir = (
                    output_dir
                    / ratio_dir_name(ratio)
                    / f"seed_{dataset_seed}"
                    / condition_name
                )
                condition_dir.mkdir(parents=True, exist_ok=True)

                output_train, output_valid, metadata = create_condition_rows(
                    train_rows,
                    valid_rows,
                    raw_train_benign_pool,
                    ratio=ratio,
                    dataset_seed=dataset_seed,
                    condition_name=condition_name,
                    condition=condition,
                    target_cwe=TARGET_CWE,
                )
                verify_condition_output(
                    output_train,
                    output_valid,
                    metadata,
                    clean_train_ids=clean_train_ids,
                )

                write_jsonl(condition_dir / "train.jsonl", output_train)
                write_jsonl(condition_dir / "valid.jsonl", output_valid)
                write_metadata(condition_dir / "metadata.json", metadata)
                summary_rows.append(summary_row(metadata))

    write_summary_csv(
        reports_dir / "noise_ratio_preserved_generation_summary.csv",
        summary_rows,
    )
    write_summary_markdown(
        reports_dir / "noise_ratio_preserved_generation_summary.md",
        input_dir=input_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        rows=summary_rows,
    )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ratio-preserved noisy CWE-119 benchmarks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cwe_119_imbalance",
        help="Clean CWE-119 imbalance benchmark directory.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/primevul_main",
        help="Raw PrimeVul main split directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cwe_119_noise_ratio_preserved",
        help="Output directory for corrected noisy train/validation variants.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "data/manifests",
        help="Directory for generation reports.",
    )
    parser.add_argument("--ratios", nargs="+", default=RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    unsupported_ratios = [ratio for ratio in args.ratios if ratio not in RATIOS]
    if unsupported_ratios:
        raise SystemExit(f"Unsupported ratio(s): {', '.join(unsupported_ratios)}")
    unsupported_seeds = [seed for seed in args.seeds if seed not in SEEDS]
    if unsupported_seeds:
        raise SystemExit(
            "Unsupported dataset seed(s): "
            + ", ".join(str(seed) for seed in unsupported_seeds)
        )
    unsupported_conditions = [
        condition for condition in args.conditions if condition not in CONDITION_SPECS
    ]
    if unsupported_conditions:
        raise SystemExit(
            f"Unsupported condition(s): {', '.join(unsupported_conditions)}"
        )

    rows = generate_noise_benchmarks(
        args.input_dir,
        args.raw_dir,
        args.output_dir,
        args.reports_dir,
        ratios=args.ratios,
        seeds=args.seeds,
        conditions=args.conditions,
        overwrite=args.overwrite,
    )
    print(
        "Generated "
        f"{len(rows)} ratio/seed/condition combinations in {args.output_dir}"
    )


if __name__ == "__main__":
    main()

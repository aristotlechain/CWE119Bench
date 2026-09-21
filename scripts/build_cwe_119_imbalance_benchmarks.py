#!/usr/bin/env python3
"""Build controlled CWE-119 class-imbalance benchmarks from PrimeVul.

The benchmark policy is intentionally narrow and explicit:

- read only PrimeVul's main temporal train/valid/test files
- fail if paired JSONL files are present in the raw input folder
- positive class: target=1 and cwe contains CWE-119
- negative class: target=0 benign rows
- exclude target=1 rows from other CWE types
- preserve the original train/valid/test split boundaries
- sample benign rows inside each split to create requested benign/vulnerable ratios
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SPLIT_FILES = {
    "train": "primevul_train.jsonl",
    "valid": "primevul_valid.jsonl",
    "test": "primevul_test.jsonl",
}

DEFAULT_RATIOS = ["60/40", "70/30", "75/25", "80/20", "90/10"]
DEFAULT_TEST_ONLY_RATIOS = ["50/50"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
CWE_PATTERN = re.compile(r"(?:CWE-\d+|NVD-CWE-[A-Za-z0-9_-]+)")


def normalize_cwes(value: Any) -> list[str]:
    """Return normalized CWE identifiers from a PrimeVul cwe field."""
    if value is None:
        return []

    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(normalize_cwes(item))
        return sorted(set(found))

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan", "[]"}:
            return []

        matches = CWE_PATTERN.findall(text)
        if matches:
            return sorted(set(matches))

        return [text]

    return [str(value)]


def parse_ratio(ratio: str) -> tuple[int, int]:
    try:
        benign_text, vulnerable_text = ratio.split("/", maxsplit=1)
        benign = int(benign_text)
        vulnerable = int(vulnerable_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ratio must look like '75/25', got {ratio!r}"
        ) from exc

    if benign <= 0 or vulnerable <= 0:
        raise argparse.ArgumentTypeError(
            f"Ratio values must be positive, got {ratio!r}"
        )

    if benign + vulnerable != 100:
        raise argparse.ArgumentTypeError(
            f"Ratio should add to 100 for this project, got {ratio!r}"
        )

    return benign, vulnerable


def ratio_name(ratio: str) -> str:
    return "ratio_" + ratio.replace("/", "_")


def split_scope(output_splits: list[str]) -> str:
    if set(output_splits) == set(SPLIT_FILES):
        return "train_valid_test"
    if output_splits == ["test"]:
        return "test_only"
    return "custom"


def build_output_splits_by_ratio(
    ratios: list[str],
    test_only_ratios: list[str],
) -> dict[str, list[str]]:
    output_splits_by_ratio = {
        ratio: list(SPLIT_FILES.keys())
        for ratio in ratios
    }
    for ratio in test_only_ratios:
        output_splits_by_ratio[ratio] = ["test"]
    return output_splits_by_ratio


def deterministic_sample(
    items: list[int],
    sample_size: int,
    *,
    seed: int,
    split: str,
    ratio: str,
    target_cwe: str,
) -> set[int]:
    seed_material = f"{target_cwe}|{ratio}|{seed}|{split}".encode("utf-8")
    seed_int = int(hashlib.sha256(seed_material).hexdigest(), 16)
    rng = random.Random(seed_int)
    return set(rng.sample(items, sample_size))


def scan_eligible_lines(
    raw_dir: Path,
    target_cwe: str,
) -> dict[str, dict[str, Any]]:
    paired_files = sorted(raw_dir.glob("*paired*.jsonl"))
    if paired_files:
        names = ", ".join(path.name for path in paired_files)
        raise SystemExit(
            "Paired files are present in the clean raw input directory. "
            f"Remove them before building the main benchmark: {names}"
        )

    missing = [name for name in SPLIT_FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required PrimeVul main files: {', '.join(missing)}")

    split_info: dict[str, dict[str, Any]] = {}

    for split, filename in SPLIT_FILES.items():
        path = raw_dir / filename
        positive_lines: list[int] = []
        benign_lines: list[int] = []
        summary = {
            "total_rows": 0,
            "target_cwe_positive_rows": 0,
            "benign_pool_rows": 0,
            "excluded_other_vulnerable_rows": 0,
            "other_target_rows": 0,
        }

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)
                target = row.get("target")
                cwes = normalize_cwes(row.get("cwe"))

                summary["total_rows"] += 1

                if target == 1 and target_cwe in cwes:
                    positive_lines.append(line_number)
                    summary["target_cwe_positive_rows"] += 1
                elif target == 0:
                    benign_lines.append(line_number)
                    summary["benign_pool_rows"] += 1
                elif target == 1:
                    summary["excluded_other_vulnerable_rows"] += 1
                else:
                    summary["other_target_rows"] += 1

        if not positive_lines:
            raise SystemExit(f"No {target_cwe} positive rows found in {split}.")

        split_info[split] = {
            "path": path,
            "filename": filename,
            "positive_lines": positive_lines,
            "benign_lines": benign_lines,
            "summary": summary,
        }

    return split_info


def build_plan(
    split_info: dict[str, dict[str, Any]],
    output_splits_by_ratio: dict[str, list[str]],
    seeds: list[int],
    target_cwe: str,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], list[dict[str, Any]]]:
    plan: dict[tuple[str, int, str], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for ratio, output_splits in output_splits_by_ratio.items():
        benign_ratio, vulnerable_ratio = parse_ratio(ratio)
        for seed in seeds:
            for split in output_splits:
                info = split_info[split]
                positive_count = len(info["positive_lines"])
                benign_needed = math.ceil(
                    positive_count * benign_ratio / vulnerable_ratio
                )

                if benign_needed > len(info["benign_lines"]):
                    raise SystemExit(
                        f"Not enough benign rows in {split} for {ratio}: "
                        f"needed {benign_needed}, available {len(info['benign_lines'])}"
                    )

                benign_sample = deterministic_sample(
                    info["benign_lines"],
                    benign_needed,
                    seed=seed,
                    split=split,
                    ratio=ratio,
                    target_cwe=target_cwe,
                )

                key = (ratio, seed, split)
                plan[key] = {
                    "positive_lines": set(info["positive_lines"]),
                    "benign_lines": benign_sample,
                    "positive_count": positive_count,
                    "benign_count": benign_needed,
                    "total_count": positive_count + benign_needed,
                }

                total_count = positive_count + benign_needed
                summary_rows.append(
                    {
                        "target_cwe": target_cwe,
                        "ratio": ratio,
                        "seed": seed,
                        "split": split,
                        "positive_rows": positive_count,
                        "benign_rows": benign_needed,
                        "total_rows": total_count,
                        "actual_vulnerable_pct": positive_count / total_count * 100,
                        "actual_benign_pct": benign_needed / total_count * 100,
                        "requested_vulnerable_pct": vulnerable_ratio,
                        "requested_benign_pct": benign_ratio,
                        "source_file": split_info[split]["filename"],
                        "benign_pool_rows": len(info["benign_lines"]),
                        "excluded_other_vulnerable_rows": info["summary"][
                            "excluded_other_vulnerable_rows"
                        ],
                        "ratio_scope": split_scope(output_splits),
                    }
                )

    return plan, summary_rows


def write_benchmarks(
    split_info: dict[str, dict[str, Any]],
    plan: dict[tuple[str, int, str], dict[str, Any]],
    output_dir: Path,
    output_splits_by_ratio: dict[str, list[str]],
    seeds: list[int],
    target_cwe: str,
) -> dict[tuple[str, int, str], dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_counts: dict[tuple[str, int, str], dict[str, int]] = {}

    handles: dict[tuple[str, int, str], Any] = {}
    try:
        for ratio, output_splits in output_splits_by_ratio.items():
            for seed in seeds:
                run_dir = output_dir / ratio_name(ratio) / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                for split in output_splits:
                    key = (ratio, seed, split)
                    handles[key] = (run_dir / f"{split}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                    written_counts[key] = {
                        "positive_rows": 0,
                        "benign_rows": 0,
                        "total_rows": 0,
                    }

        for split in SPLIT_FILES:
            line_to_outputs: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
            for ratio, output_splits in output_splits_by_ratio.items():
                if split not in output_splits:
                    continue
                for seed in seeds:
                    key = (ratio, seed, split)
                    for line_number in plan[key]["positive_lines"]:
                        line_to_outputs[line_number].append((ratio, seed, 1))
                    for line_number in plan[key]["benign_lines"]:
                        line_to_outputs[line_number].append((ratio, seed, 0))

            with split_info[split]["path"].open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    outputs = line_to_outputs.get(line_number)
                    if not outputs:
                        continue

                    row = json.loads(line)
                    for ratio, seed, binary_label in outputs:
                        key = (ratio, seed, split)
                        out_row = dict(row)
                        out_row["split"] = split
                        out_row["binary_label"] = binary_label
                        out_row["benchmark_target_cwe"] = target_cwe
                        out_row["benchmark_ratio"] = ratio
                        out_row["benchmark_seed"] = seed
                        out_row["benchmark_row_role"] = (
                            "target_cwe_positive"
                            if binary_label == 1
                            else "sampled_benign"
                        )
                        out_row["source_primevul_file"] = split_info[split]["filename"]

                        handles[key].write(
                            json.dumps(out_row, ensure_ascii=False) + "\n"
                        )
                        if binary_label == 1:
                            written_counts[key]["positive_rows"] += 1
                        else:
                            written_counts[key]["benign_rows"] += 1
                        written_counts[key]["total_rows"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return written_counts


def verify_written_counts(
    plan: dict[tuple[str, int, str], dict[str, Any]],
    written_counts: dict[tuple[str, int, str], dict[str, int]],
) -> None:
    for key, expected in plan.items():
        actual = written_counts[key]
        if actual["positive_rows"] != expected["positive_count"]:
            raise SystemExit(
                f"Positive count mismatch for {key}: "
                f"expected {expected['positive_count']}, got {actual['positive_rows']}"
            )
        if actual["benign_rows"] != expected["benign_count"]:
            raise SystemExit(
                f"Benign count mismatch for {key}: "
                f"expected {expected['benign_count']}, got {actual['benign_rows']}"
            )
        if actual["total_rows"] != expected["total_count"]:
            raise SystemExit(
                f"Total count mismatch for {key}: "
                f"expected {expected['total_count']}, got {actual['total_rows']}"
            )


def write_run_metadata(
    split_info: dict[str, dict[str, Any]],
    plan: dict[tuple[str, int, str], dict[str, Any]],
    output_dir: Path,
    output_splits_by_ratio: dict[str, list[str]],
    seeds: list[int],
    target_cwe: str,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for ratio, output_splits in output_splits_by_ratio.items():
        benign_ratio, vulnerable_ratio = parse_ratio(ratio)
        for seed in seeds:
            run_dir = output_dir / ratio_name(ratio) / f"seed_{seed}"
            metadata = {
                "generated_at_utc": generated_at,
                "target_cwe": target_cwe,
                "ratio": ratio,
                "ratio_meaning": (
                    f"{benign_ratio}% benign / {vulnerable_ratio}% vulnerable"
                ),
                "ratio_scope": split_scope(output_splits),
                "generated_splits": output_splits,
                "seed": seed,
                "input_policy": {
                    "included_files": [SPLIT_FILES[split] for split in SPLIT_FILES],
                    "excluded_files": [
                        "primevul_train_paired.jsonl",
                        "primevul_valid_paired.jsonl",
                        "primevul_test_paired.jsonl",
                    ],
                    "paired_files_used": False,
                    "train_valid_test_policy": (
                        "Original PrimeVul main temporal split boundaries preserved."
                    ),
                },
                "label_policy": {
                    "positive": f"target=1 and cwe contains {target_cwe}",
                    "negative": "target=0 benign rows",
                    "excluded": f"target=1 rows whose cwe does not contain {target_cwe}",
                    "output_label_field": "binary_label",
                },
                "sampling_policy": {
                    "formula": (
                        "benign_needed = ceil(target_cwe_positive_count "
                        "x benign_ratio / vulnerable_ratio)"
                    ),
                    "benign_sampling": (
                        "Benign rows are sampled only from the same original split."
                    ),
                    "row_order": (
                        "Generated JSONL files are written in original PrimeVul file order."
                    ),
                },
                "splits": {},
            }

            for split in output_splits:
                key = (ratio, seed, split)
                metadata["splits"][split] = {
                    "source_file": split_info[split]["filename"],
                    "target_cwe_positive_rows": plan[key]["positive_count"],
                    "sampled_benign_rows": plan[key]["benign_count"],
                    "total_rows": plan[key]["total_count"],
                    "available_benign_pool_rows": len(split_info[split]["benign_lines"]),
                    "excluded_other_vulnerable_rows": split_info[split]["summary"][
                        "excluded_other_vulnerable_rows"
                    ],
                }

            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_cwe",
        "ratio",
        "seed",
        "split",
        "positive_rows",
        "benign_rows",
        "total_rows",
        "actual_vulnerable_pct",
        "actual_benign_pct",
        "requested_vulnerable_pct",
        "requested_benign_pct",
        "source_file",
        "benign_pool_rows",
        "excluded_other_vulnerable_rows",
        "ratio_scope",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            formatted = dict(row)
            formatted["actual_vulnerable_pct"] = (
                f"{row['actual_vulnerable_pct']:.4f}"
            )
            formatted["actual_benign_pct"] = f"{row['actual_benign_pct']:.4f}"
            writer.writerow(formatted)


def write_summary_markdown(
    path: Path,
    output_dir: Path,
    split_info: dict[str, dict[str, Any]],
    ratios: list[str],
    test_only_ratios: list[str],
    seeds: list[int],
    target_cwe: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# CWE-119 Class-Imbalance Benchmark Generation Summary")
    lines.append("")
    lines.append("This report records the generated controlled class-imbalance benchmarks.")
    lines.append("")
    lines.append("## Input Policy")
    lines.append("")
    lines.append("Included PrimeVul files:")
    lines.append("")
    for split, filename in SPLIT_FILES.items():
        lines.append(f"- `{split}`: `data/raw/primevul_main/{filename}`")
    lines.append("")
    lines.append("Excluded PrimeVul files:")
    lines.append("")
    lines.append("- `primevul_train_paired.jsonl`")
    lines.append("- `primevul_valid_paired.jsonl`")
    lines.append("- `primevul_test_paired.jsonl`")
    lines.append("")
    lines.append("We preserved PrimeVul's original main temporal train/validation/test split boundaries.")
    lines.append("")

    lines.append("## Label Policy")
    lines.append("")
    lines.append(f"- Positive class: `target=1` and `cwe` contains `{target_cwe}`")
    lines.append("- Negative class: `target=0` benign rows")
    lines.append(f"- Excluded: `target=1` rows whose `cwe` does not contain `{target_cwe}`")
    lines.append("- Output label field: `binary_label`")
    lines.append("")

    lines.append("## Sampling Policy")
    lines.append("")
    lines.append("For each generated split and ratio:")
    lines.append("")
    lines.append("```text")
    lines.append("benign_needed = ceil(target_cwe_positive_count x benign_ratio / vulnerable_ratio)")
    lines.append("```")
    lines.append("")
    lines.append("Benign rows are sampled only from the same original split.")
    lines.append("")
    lines.append("Ratios mean benign/vulnerable class balance, not train/test split.")
    lines.append("")
    lines.append(
        "`50/50` is generated as an additional balanced test-only ratio. "
        "It is not used as a train/validation condition."
    )
    lines.append("")

    lines.append("## Source Counts Used")
    lines.append("")
    lines.append("| Split | Total rows | CWE-119 positives kept | Benign pool | Other vulnerable rows excluded |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in SPLIT_FILES:
        summary = split_info[split]["summary"]
        lines.append(
            "| "
            f"{split} | "
            f"{summary['total_rows']:,} | "
            f"{summary['target_cwe_positive_rows']:,} | "
            f"{summary['benign_pool_rows']:,} | "
            f"{summary['excluded_other_vulnerable_rows']:,} |"
        )
    lines.append("")

    lines.append("## Generated Outputs")
    lines.append("")
    lines.append(f"Output directory: `{output_dir}`")
    lines.append("")
    lines.append(f"Generated {len(ratios) * len(seeds)} train/validation benchmark versions.")
    lines.append(f"Generated {len(test_only_ratios) * len(seeds)} additional test-only benchmark versions.")
    lines.append("")
    lines.append(f"- train/validation ratios: {', '.join(ratios)}")
    lines.append(f"- test-only ratios: {', '.join(test_only_ratios)}")
    lines.append(
        "- test ratios available for evaluation: "
        + ", ".join(test_only_ratios + ratios)
    )
    lines.append(f"- seeds: {', '.join(str(seed) for seed in seeds)}")
    lines.append(
        "- each train/validation ratio folder contains `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `metadata.json`"
    )
    lines.append(
        "- each 50/50 test-only folder contains `test.jsonl` and `metadata.json`"
    )
    lines.append("")

    lines.append("## Count Summary by Ratio")
    lines.append("")
    lines.append("| Ratio | Scope | Split | Positives | Benign | Total | Actual vulnerable % | Actual benign % |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    seen: set[tuple[str, str]] = set()
    for row in summary_rows:
        key = (row["ratio"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            "| "
            f"{row['ratio']} | "
            f"{row['ratio_scope']} | "
            f"{row['split']} | "
            f"{row['positive_rows']:,} | "
            f"{row['benign_rows']:,} | "
            f"{row['total_rows']:,} | "
            f"{row['actual_vulnerable_pct']:.4f}% | "
            f"{row['actual_benign_pct']:.4f}% |"
        )
    lines.append("")

    lines.append("## Plain-English Explanation")
    lines.append("")
    lines.append(
        "We kept all CWE-119 vulnerable samples in each split. Then we randomly sampled enough benign rows "
        "from that same split to reach the requested benign/vulnerable ratio. We did not create fake rows, "
        "duplicate vulnerable rows, append paired files, or move rows across train/validation/test."
    )
    lines.append("")
    lines.append(
        "Under the evaluation design, models are trained and tuned on matching "
        "train/validation ratios, then tested on every test ratio, including the new 50/50 test-only set."
    )
    lines.append("")

    lines.append("## Generated CSV")
    lines.append("")
    lines.append("- `data/manifests/cwe_119_imbalance_generation_summary.csv`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled CWE-119 class-imbalance benchmarks."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/primevul_main",
        help="Directory containing the three main PrimeVul JSONL split files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cwe_119_imbalance",
        help="Directory where generated benchmark files will be written.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "data/manifests",
        help="Directory where generation reports will be written.",
    )
    parser.add_argument(
        "--target-cwe",
        default="CWE-119",
        help="Target CWE identifier for the positive class.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=DEFAULT_RATIOS,
        help="Benign/vulnerable ratios to generate for train, validation, and test.",
    )
    parser.add_argument(
        "--test-only-ratios",
        nargs="+",
        default=DEFAULT_TEST_ONLY_RATIOS,
        help="Additional benign/vulnerable ratios to generate only for the test split.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds used for benign sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for ratio in args.ratios + args.test_only_ratios:
        parse_ratio(ratio)

    overlap = set(args.ratios).intersection(args.test_only_ratios)
    if overlap:
        raise SystemExit(
            "A ratio cannot be both train/validation and test-only: "
            + ", ".join(sorted(overlap))
        )

    output_splits_by_ratio = build_output_splits_by_ratio(
        args.ratios,
        args.test_only_ratios,
    )

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output directory already exists: {args.output_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(args.output_dir)

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    split_info = scan_eligible_lines(args.raw_dir, args.target_cwe)
    plan, summary_rows = build_plan(
        split_info,
        output_splits_by_ratio,
        args.seeds,
        args.target_cwe,
    )
    written_counts = write_benchmarks(
        split_info,
        plan,
        args.output_dir,
        output_splits_by_ratio,
        args.seeds,
        args.target_cwe,
    )
    verify_written_counts(plan, written_counts)
    write_run_metadata(
        split_info,
        plan,
        args.output_dir,
        output_splits_by_ratio,
        args.seeds,
        args.target_cwe,
    )
    write_summary_csv(
        args.reports_dir / "cwe_119_imbalance_generation_summary.csv",
        summary_rows,
    )
    write_summary_markdown(
        args.reports_dir / "cwe_119_imbalance_generation_summary.md",
        args.output_dir,
        split_info,
        args.ratios,
        args.test_only_ratios,
        args.seeds,
        args.target_cwe,
        summary_rows,
    )


if __name__ == "__main__":
    main()

# CWE119Bench

CWE119Bench is the implementation and result artifact for **“CWE119Bench: Evaluating Label Noise and Class Imbalance in Vulnerability Detection.”**[1] It constructs controlled CWE-119 benchmarks from PrimeVul, perturbs training labels under six noise conditions, and evaluates four TF-IDF classifiers and a fully fine-tuned CodeBERT model.

The experiment preserves PrimeVul's temporal train, validation, and test boundaries. Training and validation ratios are 60:40, 70:30, 75:25, 80:20, and 90:10 benign:vulnerable. Testing additionally includes 50:50. Validation and test labels remain clean in every condition.

## Repository contents

| Path | Contents |
|---|---|
| `scripts/` | Data construction, model training, independent validation, analysis, and paper-asset generation |
| `configs/` | The fixed CodeBERT protocol |
| `data/manifests/` | Source hashes, generation summaries, and deterministic data-integrity records |
| `results/` | Compact model-selection evidence and the final 5,250-cell test matrix |
| `figures/` | The three Results figures generated from the five-model matrix |
| `tables/` | The two generated Results tables and their supporting CSV files |
| `tests/` | CPU-only contract, corruption, and analysis tests |
| `docs/` | Protocol, result schema, and full reproduction instructions |

Upstream data and model provenance are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

Raw PrimeVul files, generated benchmark JSONL files, pretrained weights, fine-tuned checkpoints, and per-example prediction files are not committed. The repository downloads or reconstructs those artifacts and verifies their hashes.

## Quick start

Create the classical-model environment:

```bash
conda env create -f environment.yml
conda activate cwe119bench
```

Fetch PrimeVul and build the deterministic benchmark:

```bash
python scripts/reproduce.py fetch-data
python scripts/reproduce.py build-data
```

Run a small classical-model check:

```bash
python scripts/reproduce.py smoke
```

Regenerate Figures 3–5 and Tables 6–7 from the bundled five-model matrix:

```bash
python scripts/reproduce.py paper-assets
```

## Full experiments

The four selected classical-model runs can be reproduced with:

```bash
python scripts/reproduce.py classical
```

For CodeBERT, first install a CUDA-enabled PyTorch build suitable for the GPU, followed by the recorded neural dependencies:

```bash
python -m pip install -r requirements-codebert.txt
python scripts/reproduce.py codebert
```

The complete CodeBERT matrix contains 175 fits. On the study workstation it took about 22 hours, used approximately 5.3 GiB peak GPU memory, and occupied about 82 GiB while retaining selected checkpoints. Allow at least 90 GiB of free disk space. Runtime and memory use depend on the hardware and software build. Use `python scripts/reproduce.py codebert --resume` to resume an interrupted run.

The exact order of operations, expected counts, and validation commands are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Published result evidence

[`results/five_model_test_results.csv`](results/five_model_test_results.csv) is the normalized matrix used to regenerate the revised Results figures and tables:

```text
5 models × 5 train ratios × 6 test ratios × 5 dataset seeds × 7 conditions
= 5,250 test cells
```

PR-AUC denotes trapezoidal area under the full precision-recall curve. Average precision is retained as a separate metric. See [`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md) for the complete schema and aggregation rules.

## Scope

The benchmark studies CWE-119 within PrimeVul under controlled synthetic label perturbations. The five dataset seeds reuse the same vulnerable validation and test functions, and CodeBERT uses one fixed optimization seed. The results therefore describe this controlled design.

## External resources

- [PrimeVul dataset](https://huggingface.co/datasets/colin/PrimeVul)
- [PrimeVul source repository](https://github.com/DLVulDet/PrimeVul)
- [Microsoft CodeBERT model](https://huggingface.co/microsoft/codebert-base)



## References
[1] 

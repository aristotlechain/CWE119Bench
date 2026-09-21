# Reproducibility guide

Run every command from the repository root. Commands refuse to overwrite completed result directories.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate cwe119bench
```

The classical environment pins Python, NumPy, pandas, SciPy, scikit-learn, LightGBM, Matplotlib, and Seaborn. For CodeBERT, install a CUDA-enabled PyTorch build compatible with the local GPU, then run:

```bash
python -m pip install -r requirements-codebert.txt
```

The completed study used Python 3.12.7, CUDA 12.8, PyTorch `2.11.0.dev20260111+cu128`, Transformers 4.38.2, Tokenizers 0.15.2, Hugging Face Hub 0.34.1, and Safetensors 0.7.0. The exact nightly PyTorch wheel may no longer be distributed. The release manifest records it as execution provenance.

## 2. Fetch and build data

```bash
python scripts/reproduce.py fetch-data
python scripts/reproduce.py build-data
```

The second command builds clean imbalance datasets, constructs the six ratio-preserving training-noise variants, and verifies all 638 entries against `data/manifests/data_integrity_manifest.csv`.

Expected storage is approximately 573 MiB raw plus 3.4 GiB generated.

## 3. Classical smoke check

```bash
python scripts/reproduce.py smoke
```

This trains balanced logistic regression on the clean 75:25, seed-1 configuration, selects `C` by validation MCC, and evaluates all six clean test ratios. It does not modify the bundled published results.

## 4. Reproduce the four classical families

```bash
python scripts/reproduce.py classical
```

The exact final rerun consumes the committed first-stage MCC-selection grids, refits 700 selected configurations, emits per-example validation and test scores, validates 4,200 test cells independently, and creates the seed-aware four-model analysis under `results/reproduction/`.

The four final reruns took about 2.3 hours in total on the study workstation. The selection-grid runs that produced the committed hyperparameter evidence are also reproducible directly with `run_cwe_119_model_experiments.py`.

## 5. Reproduce CodeBERT

```bash
python scripts/reproduce.py codebert
```

The driver fetches and verifies the pinned CodeBERT snapshot, trains all 175 groups, evaluates tests only after validation checkpoint selection, runs the independent validator, and joins the result to the four classical families.

Resume an interrupted run with:

```bash
python scripts/reproduce.py codebert --resume
```

The study run took approximately 22 hours on an NVIDIA GeForce RTX 5070 Ti. Peak allocated GPU memory was about 5.3 GiB. Retaining all selected checkpoints required roughly 82 GiB. Allow at least 90 GiB free.

## 6. Regenerate paper assets

```bash
python scripts/reproduce.py paper-assets
```

This validates the 5,250-cell matrix and deterministically regenerates Figures 3–5 and Tables 6–7.

## 7. Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_cwe_119_true_pr_auc_rerun.py --self-test-only
```

The unit tests use synthetic records and do not download data, load pretrained weights, or train a neural model.

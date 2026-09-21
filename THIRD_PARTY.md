# Third-party resources

This repository contains original experiment code, compact numerical results, and paper assets. It does not redistribute the PrimeVul JSONL files, Microsoft CodeBERT weights, or fine-tuned checkpoints.

## PrimeVul

- Dataset page: https://huggingface.co/datasets/colin/PrimeVul
- Source repository: https://github.com/DLVulDet/PrimeVul
- Pinned dataset revision: `4fd7158322872d711e90f091dbd8673ef32cb1be`

`scripts/fetch_primevul.py` retrieves the source files directly from the pinned upstream revision. PrimeVul records include functions from multiple software projects. Users are responsible for following the applicable upstream dataset and source-project terms.

## CodeBERT

- Model page: https://huggingface.co/microsoft/codebert-base
- Pinned model revision: `3b0952feddeffad0063f274080e3c23d75e7eb39`

`scripts/fetch_codebert.py` retrieves the model directly and records SHA-256 hashes for the seven files used by the experiment. No pretrained or fine-tuned weight file is committed here.

## Python packages

The environment files name third-party Python packages required to run the experiments. Those packages remain subject to their respective licenses.

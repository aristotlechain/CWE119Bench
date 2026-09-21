# Results

`five_model_test_results.csv` is the compact, normalized test matrix used for the paper's revised Results section. It contains 5,250 unique cells and no source-code text or local filesystem paths.

The complete factorial count is:

| Dimension | Levels |
|---|---:|
| Model family | 5 |
| Train/validation ratio | 5 |
| Test ratio | 6 |
| Dataset seed | 5 |
| Training condition | 7 |
| **Total** | **5,250** |

`model_experiments/` contains compact evidence for the model-selection and validation process:

- four original MCC-selection grids used by the exact classical rerun.
- four final classical metric files and selected settings.
- CodeBERT metrics, selected epochs, validation histories, a sanitized release manifest, and a validation summary.
- independently validated combined classical validation and test matrices.

Directories ending in `_mcc_selected_v1` are frozen first-stage selection inputs. They are included only so the exact selected hyperparameters can be reproduced. Their legacy `pr_auc` field is not used as paper evidence. The `_true_pr_auc_v1` runs and combined validation matrices contain the independently checked trapezoidal PR-AUC values.

`figure_evidence/` contains the seed-level and summary values used by Figures 3 and 4.

The Git repository omits per-example prediction scores and 175 fine-tuned CodeBERT checkpoints. Their hashes and omission reasons are recorded in the CodeBERT release manifest. A full rerun recreates them under `results/reproduction/`, which is intentionally ignored by Git.

# Experiment protocol

## Benchmark target

- Vulnerability family: CWE-119
- Source: PrimeVul main temporal splits
- Positive rows: `target=1` and a CWE list containing `CWE-119`
- Negative rows: `target=0`
- Other vulnerable rows: excluded from this binary benchmark
- Dataset seeds: 1, 2, 3, 4, 5

All CWE-119 positives are retained within each original split. Benign rows are sampled from that same split to create each requested benign:vulnerable ratio. No row crosses a temporal split.

## Class ratios

- Train and validation: 60:40, 70:30, 75:25, 80:20, 90:10
- Test: 50:50, 60:40, 70:30, 75:25, 80:20, 90:10

The 50:50 configuration is test-only. Each test construction retains 21 CWE-119 positives. Validation retains 19 and training retains 679.

## Training-label conditions

- clean
- paired random flip at 10% and 20%
- random false-positive injection at 10% and 20%
- heuristic false-positive injection at 10% and 20%

Only training labels are perturbed. `binary_label` preserves the reference label and `noisy_label` is the learner-facing label. Validation and test labels are unchanged. False-positive conditions add unused benign training rows as compensation so the requested learner-label ratio remains fixed.

## Classical models

- Logistic regression
- Linear SVM
- Random Forest
- LightGBM

The input is the `func` field only. TF-IDF uses `char_wb` character n-grams of length 3–5, `max_features=20000`, `min_df=2`, and `lowercase=False`. All four models use inverse-frequency class weighting. Hyperparameters are selected by validation MCC, followed by F1, recall, precision, and balanced accuracy as deterministic tie breakers.

## CodeBERT

- Model: `microsoft/codebert-base`
- Revision: `3b0952feddeffad0063f274080e3c23d75e7eb39`
- Maximum sequence length: 512 tokens, right truncation
- Epoch budget: 5
- Batch size: 8 for training and evaluation
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup ratio: 0.1
- Mixed precision: BF16
- Optimization seed: 0
- Decision threshold: 0.5

Every group is trained for all five epochs. The checkpoint is selected on clean validation MCC with the same tie-break sequence as the classical models. Test access occurs only after checkpoint selection and requires the explicit `--evaluate-tests` flag.

## Metrics

Reported metrics include MCC, F1, precision, recall, balanced accuracy, ROC-AUC, average precision, and trapezoidal PR-AUC. Average precision and PR-AUC are stored separately.

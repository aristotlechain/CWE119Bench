# Result schema

`results/five_model_test_results.csv` contains one row per model, train ratio, test ratio, dataset seed, and training condition.

## Identity and protocol columns

| Column | Meaning |
|---|---|
| `model`, `model_family` | Trained estimator and normalized family name |
| `hyperparameter_name` | Selected parameter (`C`, `min_samples_leaf`, `num_leaves`, or `learning_rate`) |
| `selected_hyperparameter_value` | Selected value, normalized across all five families |
| `learning_rate`, `selected_epoch` | CodeBERT settings. Empty for classical rows |
| `train_ratio`, `eval_ratio` | Benign:vulnerable training/validation and test ratios |
| `seed` | Dataset sampling seed |
| `condition` | Clean or one of six label-noise conditions |
| `train_rows`, `valid_rows`, `n_rows` | Training, validation, and evaluated test sizes |
| `positive_rows`, `benign_rows` | Clean test-label counts |

## Metric columns

`accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `roc_auc`, `average_precision`, and `pr_auc` are computed per test cell. `tn`, `fp`, `fn`, and `tp` record the corresponding confusion matrix at the fixed model threshold.

`pr_auc` is trapezoidal integration of the complete precision-recall curve. It is distinct from `average_precision`.

## Aggregation used by the paper

Noise deltas pair every noisy row with the clean row having the same model family, train ratio, test ratio, and dataset seed. Figures 3 and 4 first average the declared cells within each seed and then report the mean, sample standard deviation, or heatmap mean across the five dataset seeds. Table 6 averages the five model families within each seed at each exact train-test ratio pair. Table 7 computes the 90:10-test minus 50:50-test contrast within each seed before summarizing.

The five seeds reuse the same positive validation and test functions. Their sample standard deviations describe the benchmark's benign-sampling variation and are not independent-cohort confidence intervals.

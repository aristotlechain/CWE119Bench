# Scripts

`reproduce.py` is the main entry point. Its stages run these scripts in dependency order.

## Data construction

1. `fetch_primevul.py`
2. `build_cwe_119_imbalance_benchmarks.py`
3. `build_cwe_119_ratio_preserved_noise_benchmarks.py`
4. `data_integrity.py`

## Classical models

1. `run_cwe_119_model_experiments.py` performs the three-value validation grid for one model family.
2. `rerun_cwe_119_selected_true_pr_auc.py` refits the selected configuration and retains full-precision metrics and scores.
3. `validate_cwe_119_true_pr_auc_rerun.py` reconstructs metrics independently from the per-example scores.
4. `analyze_cwe_119_research_questions_true_pr_auc.py` creates seed-aware summaries.

## CodeBERT

1. `fetch_codebert.py` fetches and hashes the pinned public model snapshot.
2. `run_cwe_119_codebert_experiments.py` fine-tunes 175 configurations and selects epochs using clean validation MCC with declared tie breakers.
3. `validate_cwe_119_codebert_results.py` independently audits inputs, metrics, scores, checkpoints, and the full factor grid.
4. `analyze_cwe_119_codebert_results.py` joins validated CodeBERT cells to the four classical families by model-independent experiment keys.
5. `prepare_release_results.py` validates and normalizes the complete 5,250-cell matrix for compact publication.

## Paper assets

- `generate_paper_figures.py`
- `generate_paper_tables.py`

Both asset generators read `results/five_model_test_results.csv`. Neither reads manuscript source files.

# Figures

| Paper figure | File | Provenance |
|---|---|---|
| Figure 3 | `fig_noise_effects_seed_aware_true_pr_auc.pdf` | Generated from the five-model test matrix |
| Figure 4 | `fig_paired_flip_train_test_interaction_true_pr_auc.pdf` | Generated from the five-model test matrix |
| Figure 5 | `fig_rf_pr_by_train_ratio_all_conditions_seed_aware.pdf` | Random Forest rows from the five-model test matrix |

Regenerate these Results figures with:

```bash
python scripts/generate_paper_figures.py
```

The generator validates all 5,250 factorial cells before writing the files. Its PDF metadata and creation date are fixed so repeated runs with the pinned plotting environment are byte-reproducible. The paper's conceptual pipeline diagrams are documented by the benchmark-construction scripts and experiment protocol rather than duplicated in this code-and-results repository.

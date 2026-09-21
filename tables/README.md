# Tables

This directory contains the two Results tables used by the paper:

- `tab_clean_train_test_joint_seed_aware.tex` (Table 6)
- `tab_noise_test_endpoint_interactions_seed_aware.tex` (Table 7)

The matching CSV files expose the full-precision values underlying the formatted LaTeX cells. Regenerate and validate both tables with:

```bash
python scripts/generate_paper_tables.py
```

`paper_asset_manifest.json` records the source-matrix hash, output hashes, and numerical claims checked during generation.

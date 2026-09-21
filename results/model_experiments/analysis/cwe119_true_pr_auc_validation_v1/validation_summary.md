# CWE-119 True PR-AUC Rerun Validation

- Status: **passed**
- Selected configurations: 700
- Validation metric cells: 700
- Test metric cells: 4200
- Validation score rows: 66220
- Test score rows: 394800
- All score rows: 461020
- Every score row matched its canonical benchmark JSONL row and label.
- Every stored metric was independently reconstructed from retained scores.
- Average precision and trapezoidal PR-AUC passed a non-aliasing fixture.

# Portable execution readiness and staged migration

## Verified repository capabilities

The code in `scripts/summarize_archived_results.py` independently recalculates paper-summary numbers and budget-stratified metrics **from archived E3.36 CSV records** with Python standard library. This is a post-hoc summary reconstruction, not model prediction or experiment rerun. The original `scripts/verify_reported_results.py` performs consistency assertions. `scripts/audit_inputs.py` compares frozen checkpoint references with the partially archived checkpoint filenames; it does not validate checkpoint identity or model weights.

## Historical run chain (do not execute blindly)

- `research_steps/01_*`: scenario 104 generation; relies on Santander source dataset and original Kaggle context.
- `02_*`: prequential interaction history features; depends on output of scenario generation.
- `03_*`: tensor/scaler preparation; depends on scenario feature outputs.
- `04_*`, `05_*`, `07_*`: training/validation; depend on prior definitions, GPU and original file paths.
- `06_*`: recovery of model definitions/preprocessing from original context.
- `08_*`: **historical frozen held-out test**; do not rerun or use for tuning.

## Python dependencies observed in the final-stage source extracts

`numpy`, `pandas`, `torch`, `scikit-learn`, `joblib`. This inventory is extracted from Python import statements and **is not a verified pinned environment**. Exact versions, CUDA runtime and compute/device configuration are unavailable in the published source archive. Do not fabricate pins or advertise a fully reproducible training installation.

## Distinct checkpoints and dataset dependencies

The frozen protocol lists 18 checkpoint paths on the original Kaggle instance. The present package includes 8 checkpoint artifacts, and does not include the other checkpoint files, complete scenario snapshots or all training-fitted preprocessing files. `audit_inputs.py` provides a mechanical inventory. Model weights and checkpoints were **not loaded or executed** during packaging.

## Safe progression to a fully portable implementation

1. Obtain original data sources or their verified authorized retrieval instructions and record data origin/license/checksum.
2. Export the complete simulation-104 snapshots and train-fitted scalers from the original experiment without touching held-out labels or retraining.
3. Recover every required epoch-60 checkpoint; calculate content hashes and map them to protocol paths.
4. Freeze runtime versions from original Kaggle logs/environment if available; test environment installation.
5. Refactor modular preprocessing and training with unit tests for temporal leakage, coverage constraint, edge-budget rounding, and deterministic masks; retain original archival code unchanged.
6. In an independent clean environment, reproduce training/validation and compare to frozen checkpoints and summaries; treat the recorded E3.36 held-out metrics as archival records, not a source for model development.

## Commands safe to run now

```bash
python scripts/verify_stage_sources.py
python scripts/verify_reported_results.py
python scripts/summarize_archived_results.py
python scripts/audit_inputs.py
```

These commands read repository metadata and archived files only. They do not access Santander data, train models, regenerate labels or evaluate test snapshots.

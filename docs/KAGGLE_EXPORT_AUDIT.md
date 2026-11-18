# Kaggle frozen-artifact export audit

Export date: 2026-11-18. The user's read-only Kaggle inventory confirmed that all 18 frozen checkpoint paths exist. Two ZIP exports were subsequently provided and inspected independently.

- `E3_36_FROZEN_18_CHECKPOINTS_v2.zip`: 18 checkpoint files plus manifest; all SHA-256 and sizes matched the export manifest and the frozen protocol lists exactly the same paths.
- `E3_36_SCENARIO_104_SUPPORT_v2.zip`: 22 source files plus manifest; all SHA-256 and sizes matched the export manifest.
- All 18 frozen checkpoint files are staged under `artifacts/frozen_checkpoints/`, preserving provenance paths. The two training-fitted scaler files are staged under `artifacts/generated_support/`.
- The 12 source historical feature archives t00–t11 are **not** all included in the support export: it contains t00–t09; test-time features are reconstructed causally in original Kaggle code and are not pre-exported t10/t11 feature snapshots.
- `e3_29_causal_feature_audit.json` in the support export was **empty (0 bytes)**, so it is not used or presented as a valid audit report.
- Observed train/validation/test files and SIoT derived topology are retained only in the separately supplied support ZIP, **not** in this GitHub candidate pending review of the original Santander data's redistribution terms.
- No model weights have been modified. No training, inference or held-out test run was executed while preparing this repository candidate.
- The manifest is an original export record; local checkpoint paths are `artifacts/` + the recorded archive path.

## Verification commands

```bash
python scripts/audit_inputs.py
python scripts/verify_support_scalers.py
python scripts/verify_reported_results.py
python scripts/summarize_archived_results.py
python scripts/verify_stage_sources.py
```

All commands examine archived files only. A fresh-clone end-to-end training and test rerun has **not** been established. Do not load `.pt` / `.joblib` from untrusted sources; these files originate from the author's own Kaggle export.

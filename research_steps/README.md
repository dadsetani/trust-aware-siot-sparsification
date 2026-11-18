# Historical study-cell map (not a runnable standalone application)

These files are **unchanged source extracts** of selected research cells from
`notebooks/TAGS_v3_E3_36_research_archive.ipynb`. They are organized by scientific
stage for inspection, review, and subsequent portable refactoring. Their source cell
numbers and SHA-256 checksums are recorded in `configs/SOURCE_CELL_MANIFEST.json`.

| Original notebook cell (zero-based) | Purpose | File |
|---:|---|---|
| 145 | E3.28: owner-correlated trust scenario, seed 104 | `research_steps/01_simulated_scenario_seed104.py` |
| 146 | E3.29: prequential trust features | `research_steps/02_historical_trust_features.py` |
| 147 | E3.30: model input preparation | `research_steps/03_model_input_preparation.py` |
| 148 | E3.31: model training across seeds | `research_steps/04_training_seed_setup.py` |
| 151 | E3.34: budget sensitivity retraining | `research_steps/05_budget_sensitivity.py` |
| 159 | E3.RECOVER: recovery and model definitions | `research_steps/06_recover_model_and_preprocessing.py` |
| 167 | E3.35: validation replication | `research_steps/07_validation_three_seeds.py` |
| 168 | E3.36: frozen one-time held-out test | `research_steps/08_heldout_test_original.py` |

**Important:** Extracts rely on prior notebook variables, Kaggle file paths,
original data and checkpoints, and some exploratory cells outside this table.
They are not independently runnable entrypoints. In particular, **do not run**
`08_heldout_test_original.py` as if it were a fresh held-out evaluation; the
reported final test is frozen and should not be repeated for publication claims.
The canonical fixed archival version is separately retained in
`scripts/E3_36_final_test_cell_FIXED.py` and may differ from the notebook cell.

Validate source identity and Python syntax without executing code:

```bash
python scripts/verify_stage_sources.py
python scripts/verify_reported_results.py
```

This stage map is an audit and migration aid, **not** a finished `src/` package.
Future refactoring should reproduce the frozen results before claiming parity.

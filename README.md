# Trust-aware SIoT graph sparsification

Research archive for **GNN-Based versus Trust-Only Edge Scoring in Trust-Aware Social IoT Graph Sparsification: A Controlled Simulation Study**.

## Research question

This controlled simulation compares two **edge scoring** strategies at equal edge budgets (25%, 50%, 75%) on a fixed Santander SIoT topology (16,216 nodes; 146,117 undirected edges). The GNN-based edge scorer uses graph representations plus historical trust features; the trust-only edge scorer uses historical trust features. **Both configurations still use graph-based downstream prediction architectures**, trained separately. The downstream task is interaction-failure prediction, not friendship/service ranking.

## What is included

- `notebooks/TAGS_v3_E3_36_research_archive.ipynb`: sanitized **historical Kaggle notebook** with execution outputs cleared. It contains numerous exploratory and recovery cells and **is not** a clean run-all pipeline.
- `scripts/E3_36_final_test_cell_FIXED.py`: the original final held-out test cell, **dependent on the state restored by prior Kaggle notebook cells**. This is not an independently runnable CLI.
- `configs/FROZEN_PROTOCOL_BEFORE_TEST.json`: exact protocol frozen before held-out testing; duplicated verbatim under `results/heldout_test/`.
- `results/heldout_test/`: 18 held-out test records, 9 paired comparisons, and frozen protocol.
- `results/validation/`: archived validation summaries and histories (earlier partial checkpoint duplicates removed).
- `artifacts/frozen_checkpoints/`: all 18 epoch-60 frozen checkpoint files, retaining original relative Kaggle paths with SHA-256 manifest.
- `artifacts/generated_support/`: the two training-fitted scaler artifacts; source manifest under `docs/SUPPORT_EXPORT_MANIFEST.json`.
- `scripts/verify_reported_results.py`: standard-library checks for held-out summary numbers, pair count, and retention budgets.
- `figures/`: descriptive plots used in the manuscript.
- `docs/REPRODUCIBILITY.md`: provenance, limitations, and outstanding inputs for full reruns.

## Source map for the final research stages

The `research_steps/` directory contains **verbatim, non-standalone** extracts of eight
key source cells from the historical notebook: scenario generation, historical
features, input preparation, training setup, budget sensitivity, recovery,
validation, and frozen held-out evaluation. See `research_steps/README.md` and
`configs/SOURCE_CELL_MANIFEST.json` for provenance. Run
`python scripts/verify_stage_sources.py` to validate source identity **without
executing the experiments**. These extracts are not yet a portable training package.

## Verify published summary metrics

Run from the repository root:

```bash
python scripts/verify_reported_results.py
```

This checks the immutable CSV-based results without training or reading held-out inputs anew. It **does not** independently reproduce the original training and prediction procedure.

## Read-only numerical reproduction and input audit

The following commands work from a fresh checkout with **Python 3 standard library only**; they do not generate new test predictions:

```bash
python scripts/verify_reported_results.py
python scripts/summarize_archived_results.py
python scripts/audit_inputs.py
python scripts/verify_stage_sources.py
```

The numerical summary reconstructs paper means and budget effects from archived result CSVs. The inventory script cryptographically checks all 18 frozen checkpoint files against the Kaggle export manifest. See `docs/PORTABILITY_PLAN.md` for exact boundaries and steps to a fully portable pipeline. The `requirements-analysis.txt` file is an **unpinned import inventory**, not a validated training environment.

## Experimental protocol

- Observed topology: Santander SIoT social relationships. The trust outcomes are **synthetically generated**, not measured trust labels from Santander.
- One owner-correlated simulation (`B_owner_correlated`, scenario seed 104), with 12 chronological snapshots (0–11).
- Historical warmup: t0–4; train: t5–7; validation: t8–9; held-out test: t10–11.
- Model initialization seeds: 44, 45, 46; 60 fixed training epochs; checkpoint selection at epoch 60.
- Both scorers use coverage-constrained masks at equal edge budgets; one-time held-out evaluation is reported without test-time model updates.

## Findings and interpretation

The observed average AP difference (GNN minus trust-only) is -0.000316 over 9 paired seed/budget configurations. Average mask churn is lower for GNN-based scoring (0.058610 vs 0.077920), but this does not establish better edge quality, adaptation, or network performance. These are descriptive results for **one** simulated trust realization and are not evidence of statistical equivalence.

## Status and license

**Initial research archive, not yet a verified end-to-end reproduction package.** All 18 original epoch-60 checkpoints and two training-fitted scalers are now included. The Santander-derived topology, generated observations, full historical feature files, and a portable runnable training/inference pipeline are **not** included pending licensing review and runtime validation. Original Kaggle paths are documented in the frozen protocol and historical code; do not represent them as available locally.

No code license is granted in this preliminary archive pending confirmation of authorship and third-party data/code terms. Please do not interpret the absence of a license as permission to redistribute third-party material. The exact manuscript citation and Zenodo DOI will be added after release.

## Newly recovered frozen artifacts (2026-11-18)

Independent ZIP integrity, SHA-256, and filename checks matched all **18/18** archived checkpoint files against the export manifest and frozen protocol. The two scaler files are also preserved with verified hashes. See `docs/KAGGLE_EXPORT_AUDIT.md`. **Do not claim complete end-to-end reproduction:** the released repository candidate intentionally omits the Santander-derived generated data until redistribution rights are established, and the Kaggle notebook is still not a clean standalone pipeline.

Verify after extracting this package with `python scripts/audit_inputs.py` and `python scripts/verify_support_scalers.py`.

# Reproducibility and source provenance

## Verified inputs used to assemble this package

- `TAGS_v3_with_E3_36_FIXED.ipynb` (local historical Kaggle notebook; 169 cells): preserved without execution outputs.
- `E3_36_final_test_cell_FIXED.py` (the final E3.36 test script; designed to append after recovery cells).
- Original `TAGS_PAPER_FINAL_RESULTS.zip` plus newly verified separate Kaggle artifact exports for all 18 checkpoints and supporting files. See `KAGGLE_EXPORT_AUDIT.md`.
- Figures derived from previously reported data.

## Re-running

The `verify_reported_results.py` script only validates consistency of archived CSV outputs, not the full experimental pipeline. To reproduce training and held-out prediction, one must additionally provide:

1. The correctly licensed original Santander SIoT device metadata and relationship graph.
2. The generated scenario-104 trust observations and reconstruction code/state, with the exact historical prequential feature transformation and training-fitted scalers.
3. All 18 frozen epoch-60 checkpoints (now included and checksum-verified); independent loading/inference still requires validating the original Kaggle execution state.
4. A clean, fully validated execution entrypoint, environment lockfile, and hardware/software versions.

The notebook includes earlier explorations and recovery operations. **Do not press “Run all” and claim this executes the manuscript experiment.** The final test code asserts prior restored notebook symbols and original Kaggle paths; edits are needed for portable execution and should be validated against original outputs.

## Frozen protocol details

Consult `configs/FROZEN_PROTOCOL_BEFORE_TEST.json` for the exact 18 checkpoint paths, seed/budget combinations, and chronology. These paths describe the **original Kaggle machine**, not GitHub files. All values in `results/heldout_test` are archived observations of the previously completed test; no test evaluation was repeated in building this package.

## Data provenance and permissions

The graph substrate originates from an external Santander SIoT dataset; the binary interaction labels and trust states originate from simulation. The underlying third-party data license has not yet been verified for redistribution. Consequently, **the original dataset is omitted** from this package. Do not upload third-party inputs or other people's code until terms are checked.

## Correct interpretation

The trust-only baseline removes GNN embeddings from the **selection scorer only**; both branches have graph-based downstream encoders. Lower churn is not proof of higher graph selection quality. There is only one scenario realization and three model seeds. Avoid claims about runtime, memory, energy, or deployed network QoS, which were not compared here.

## To do before Zenodo release

- [ ] Confirm authors' permission and third-party licenses; add LICENSE where appropriate.
- [ ] Add full portable training/scenario-generation/evaluation commands with dependency lockfile.
- [x] Include all 18 frozen checkpoint/model files and both scalers with checksums.
- [ ] Provide permitted original inputs or exact download instructions; source-derived data are still excluded pending review.
- [ ] Run full reproduction in a clean environment and compare every metric with frozen CSVs.
- [ ] Add paper citation metadata and reviewed authorship via `CITATION.cff`.
- [ ] Publish a GitHub release, then connect/archive using Zenodo, and insert stable DOI into manuscript.

## Extracted research steps (stage map)

The `research_steps/` files are copies of original notebook cells, checked against
`configs/SOURCE_CELL_MANIFEST.json`. Extracting them does not establish that they
are independently runnable. No training, predictions, test outcomes or checkpoints
were regenerated in this packaging step. Use `verify_stage_sources.py` to confirm
source fidelity before any subsequent refactoring.

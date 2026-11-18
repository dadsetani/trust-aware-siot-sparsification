
# ============================================================
# E3.29 — CAUSAL TRUST FEATURES / SCENARIO B SEED 104
# ============================================================
# Reproduce E3.22 exactly with simulation seed 104.
#
# 12 causal features, snapshots t=0,...,9.
# Snapshot at t uses observations strictly before t.
#
# Read only:
#   observed/siot_social_topology.csv.gz
#   observed/train.csv.gz
#   observed/validation.csv.gz
#
# Do NOT access test or hidden simulator state.
# No model training, no overwriting previous artifacts.
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE329 = Path(
    "/kaggle/working/phase2_adaptive_siot/"
    "E3_tags_v3/scenarios/B_owner_correlated"
)

SOURCE329 = BASE329 / "seed_104"
REFERENCE329 = BASE329 / "seed_103"

OBS329 = SOURCE329 / "observed"
SAVE329 = SOURCE329 / "causal_features"

if SAVE329.exists():
    raise FileExistsError(
        f"Output already exists: {SAVE329}"
    )

N329 = 16216
M329 = 146117
DECAY329 = 0.80
CONF_K329 = 5.0
HORIZON329 = 12

FEATURE_NAMES329 = [
    "edge_history_count",
    "edge_success_rate",
    "edge_decayed_count",
    "edge_decayed_success_rate",
    "edge_evidence_confidence",
    "edge_age_normalized",
    "edge_never_observed",
    "node_u_success_rate",
    "node_v_success_rate",
    "node_u_evidence_confidence",
    "node_v_evidence_confidence",
    "edge_recent_minus_longterm"
]

# ------------------------------------------------------------
# 1. Verify reference specification and scenario report
# ------------------------------------------------------------

with open(
    REFERENCE329 /
    "causal_features/e3_22_causal_feature_audit.json",
    encoding="utf-8"
) as f:
    ref329 = json.load(f)

assert ref329["feature_count"] == 12
assert ref329["feature_names"] == FEATURE_NAMES329
assert ref329["temporal_decay"] == DECAY329
assert ref329["evidence_confidence_k"] == CONF_K329
assert ref329["age_normalization_horizon"] == HORIZON329

with open(
    SOURCE329 / "e3_28_report.json",
    encoding="utf-8"
) as f:
    sim_report329 = json.load(f)

assert sim_report329["simulation_seed"] == 104
assert sim_report329["undirected_edges"] == M329
assert sim_report329["nodes"] == N329

# ------------------------------------------------------------
# 2. Fixed canonical topology
# ------------------------------------------------------------

topology329 = pd.read_csv(
    OBS329 / "siot_social_topology.csv.gz"
).sort_values(
    ["u", "v"]
).reset_index(drop=True)

reference_mapping329 = pd.read_csv(
    REFERENCE329 /
    "causal_features/edge_index_mapping.csv.gz"
).sort_values(
    ["u", "v"]
).reset_index(drop=True)

pd.testing.assert_frame_equal(
    topology329,
    reference_mapping329,
    check_dtype=True
)

assert len(topology329) == M329

u329 = topology329["u"].to_numpy(
    dtype=np.int64
) - 1

v329 = topology329["v"].to_numpy(
    dtype=np.int64
) - 1

assert np.all(u329 < v329)
assert u329.min() >= 0
assert v329.max() < N329

edge_keys329 = u329 * N329 + v329

assert np.all(np.diff(edge_keys329) > 0)

# ------------------------------------------------------------
# 3. Read permitted observations only
# ------------------------------------------------------------

train329 = pd.read_csv(
    OBS329 / "train.csv.gz"
)

validation329 = pd.read_csv(
    OBS329 / "validation.csv.gz"
)

assert train329["time"].between(0, 7).all()
assert validation329["time"].between(8, 9).all()

assert len(train329) == sim_report329["train_events"]
assert (
    len(validation329)
    == sim_report329["validation_events"]
)

events329 = pd.concat(
    [train329, validation329],
    ignore_index=True
)

assert events329["time"].between(0, 9).all()
assert events329["success"].isin([0, 1]).all()

event_u329 = (
    events329["source"]
    .to_numpy(np.int64) - 1
)

event_v329 = (
    events329["target"]
    .to_numpy(np.int64) - 1
)

event_t329 = events329[
    "time"
].to_numpy(np.int16)

event_success329 = events329[
    "success"
].to_numpy(np.float32)

assert np.all(event_u329 < event_v329)

event_keys329 = (
    event_u329 * N329 + event_v329
)

event_ids329 = np.searchsorted(
    edge_keys329,
    event_keys329
)

assert np.all(event_ids329 < M329)
assert np.array_equal(
    edge_keys329[event_ids329],
    event_keys329
)

# At most one interaction per edge per time step.
unique_keys329 = (
    event_t329.astype(np.int64) * M329
    + event_ids329
)

assert (
    np.unique(unique_keys329).size
    == len(events329)
)

# ------------------------------------------------------------
# 4. Initialize historical accumulators
# ------------------------------------------------------------

edge_success329 = np.zeros(
    M329, dtype=np.float32
)

edge_failure329 = np.zeros(
    M329, dtype=np.float32
)

decay_success329 = np.zeros(
    M329, dtype=np.float32
)

decay_failure329 = np.zeros(
    M329, dtype=np.float32
)

node_success329 = np.zeros(
    N329, dtype=np.float32
)

node_failure329 = np.zeros(
    N329, dtype=np.float32
)

last_seen329 = np.full(
    M329, -1, dtype=np.int16
)

def smoothed_rate329(s, f):
    return (s + 1.0) / (s + f + 2.0)

def evidence_conf329(s, f):
    count = s + f
    return count / (count + CONF_K329)

audit329 = []
cumulative329 = 0

print("=" * 76)
print("E3.29 — CAUSAL FEATURES / SEED 104")
print("=" * 76)

print("Nodes:", N329)
print("Edges:", M329)
print("Features:", len(FEATURE_NAMES329))
print("Temporal decay:", DECAY329)
print("Reference E3.22 specification: VERIFIED")
print("Train observations:", len(train329))
print("Validation observations:", len(validation329))
print("Test accessed: NO")

# ------------------------------------------------------------
# 5. Construct snapshots using strictly prior history
# ------------------------------------------------------------

# All checks above pass before creating the output folder.
SAVE329.mkdir(parents=True)

for t in range(10):

    # Same update order as the verified E3.22 code.
    if t > 0:
        decay_success329 *= DECAY329
        decay_failure329 *= DECAY329

    history_count329 = (
        edge_success329 + edge_failure329
    )

    history_rate329 = smoothed_rate329(
        edge_success329,
        edge_failure329
    )

    decayed_count329 = (
        decay_success329 + decay_failure329
    )

    decayed_rate329 = smoothed_rate329(
        decay_success329,
        decay_failure329
    )

    edge_conf329 = evidence_conf329(
        edge_success329,
        edge_failure329
    )

    node_rate329 = smoothed_rate329(
        node_success329,
        node_failure329
    )

    node_conf329 = evidence_conf329(
        node_success329,
        node_failure329
    )

    never_seen329 = (
        last_seen329 == -1
    )

    age329 = np.where(
        never_seen329,
        HORIZON329,
        t - last_seen329
    ).astype(np.float32)

    features329 = np.column_stack([
        history_count329,
        history_rate329,
        decayed_count329,
        decayed_rate329,
        edge_conf329,
        age329 / HORIZON329,
        never_seen329.astype(np.float32),
        node_rate329[u329],
        node_rate329[v329],
        node_conf329[u329],
        node_conf329[v329],
        decayed_rate329 - history_rate329
    ]).astype(np.float32)

    assert features329.shape == (M329, 12)
    assert np.isfinite(features329).all()

    # Strict prequential history checks.
    expected_history329 = int(
        np.count_nonzero(event_t329 < t)
    )

    assert cumulative329 == expected_history329

    assert int(
        history_count329.sum()
    ) == expected_history329

    if t == 0:
        assert np.all(features329[:, 0] == 0)
        assert np.all(features329[:, 1] == 0.5)
        assert np.all(features329[:, 6] == 1)

    filename329 = (
        f"snapshot_t{t:02d}.npz"
    )

    np.savez_compressed(
        SAVE329 / filename329,
        features=features329,
        time=np.array(
            t, dtype=np.int16
        ),
        feature_names=np.array(
            FEATURE_NAMES329
        )
    )

    current329 = (
        event_t329 == t
    )

    ids329 = event_ids329[
        current329
    ]

    successes329 = event_success329[
        current329
    ]

    failures329 = (
        1.0 - successes329
    )

    audit329.append({
        "time": int(t),
        "history_before_t": int(cumulative329),
        "edges_with_history": int(
            (history_count329 > 0).sum()
        ),
        "observations_at_t": int(
            len(ids329)
        ),
        "snapshot": filename329
    })

    # Update accumulators AFTER saving snapshot t.
    np.add.at(
        edge_success329,
        ids329,
        successes329
    )

    np.add.at(
        edge_failure329,
        ids329,
        failures329
    )

    np.add.at(
        decay_success329,
        ids329,
        successes329
    )

    np.add.at(
        decay_failure329,
        ids329,
        failures329
    )

    np.add.at(
        node_success329,
        u329[ids329],
        successes329
    )

    np.add.at(
        node_success329,
        v329[ids329],
        successes329
    )

    np.add.at(
        node_failure329,
        u329[ids329],
        failures329
    )

    np.add.at(
        node_failure329,
        v329[ids329],
        failures329
    )

    last_seen329[ids329] = t

    cumulative329 += len(ids329)

    print(
        f"t={t:02d} | "
        f"past={expected_history329:7,d} | "
        f"observed_edges="
        f"{audit329[-1]['edges_with_history']:7,d} | "
        f"current={len(ids329):6,d}",
        flush=True
    )

# ------------------------------------------------------------
# 6. Final integrity checks
# ------------------------------------------------------------

assert cumulative329 == len(events329)
assert len(events329) == 218911

with np.load(
    SAVE329 / "snapshot_t09.npz"
) as archive329:

    final_features329 = archive329[
        "features"
    ]

    assert int(
        archive329["time"]
    ) == 9

    assert (
        archive329["feature_names"].tolist()
        == FEATURE_NAMES329
    )

assert final_features329.shape == (M329, 12)

expected_t9_history329 = int(
    np.count_nonzero(event_t329 < 9)
)

assert int(
    final_features329[:, 0].sum()
) == expected_t9_history329

topology329.to_csv(
    SAVE329 / "edge_index_mapping.csv.gz",
    index=False,
    compression="gzip"
)

# ------------------------------------------------------------
# 7. Save audit report
# ------------------------------------------------------------

report329 = {
    "experiment": "E3.29",
    "scenario": "B_owner_correlated",
    "simulation_seed": 104,
    "reference_feature_pipeline": "E3.22",
    "feature_count": 12,
    "feature_names": FEATURE_NAMES329,
    "temporal_decay": DECAY329,
    "evidence_confidence_k": CONF_K329,
    "age_normalization_horizon": HORIZON329,
    "snapshots": audit329,
    "train_events": int(len(train329)),
    "validation_events": int(len(validation329)),
    "permitted_events": int(len(events329)),
    "historical_events_at_t9":
        expected_t9_history329,
    "prequential": True,
    "test_accessed": False,
    "hidden_state_accessed": False,
    "model_training_performed": False,
    "previous_artifacts_modified": False
}

REPORT329 = (
    SAVE329 / "e3_29_causal_feature_audit.json"
)

with open(
    REPORT329,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report329, f, indent=2
    )

print("\n" + "=" * 76)
print("E3.29 COMPLETED — SEED 104 FEATURES")
print("=" * 76)

print("Snapshots saved:", len(audit329))
print("Feature shape:", final_features329.shape)
print("Historical observations at t=0: 0")
print(
    "Historical observations at t=9:",
    expected_t9_history329
)
print(
    "Train + validation observations:",
    len(events329)
)
print("Feature specification unchanged: VERIFIED")
print("Test accessed: NO")
print("Hidden state accessed: NO")
print("Model training performed: NO")
print("Previous artifacts modified: NO")
print("Saved:", REPORT329)

print(
    "NEXT: E3.30 — Prepare seed-104 model inputs "
    "with train-only StandardScalers."
)

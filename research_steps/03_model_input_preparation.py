
# ============================================================
# E3.30 — PREPARE SEED 104 MODEL INPUTS
# ============================================================
# Reuse verified preprocessing from E3.16h.
# Train-only scaling; CUDA tensors.
# No training, no test access, no overwriting.
# ============================================================

from pathlib import Path
import json
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

required = [
    "build_raw_snapshots_h",
    "type_onehot",
    "data103",
    "edge_scaler103",
    "node_scaler103",
    "U", "V", "K"
]

missing = [x for x in required if x not in globals()]
assert not missing, f"Missing objects: {missing}"

BASE = Path(
    "/kaggle/working/phase2_adaptive_siot/"
    "E3_tags_v3/scenarios/B_owner_correlated"
)

SOURCE = BASE / "seed_104"
OUT = SOURCE / "e3_30_model_inputs"

if OUT.exists():
    raise FileExistsError(OUT)

TRAIN = [5, 6, 7]
VAL = [8, 9]
TIMES = TRAIN + VAL
N, M = 16216, 146117

assert U.is_cuda and V.is_cuda
assert K == 73058

assert (
    SOURCE /
    "causal_features/e3_29_causal_feature_audit.json"
).is_file()

print("=" * 70)
print("E3.30 — SEED 104 MODEL INPUT PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Verify preprocessing using existing Seed 103
# ------------------------------------------------------------

print("\n[1] VERIFY SEED 103 RECONSTRUCTION")

raw103_check = build_raw_snapshots_h(BASE / "seed_103")

edge_error = 0.0
node_error = 0.0

for t in TIMES:

    xe = edge_scaler103.transform(
        raw103_check[t]["edge"]
    ).astype(np.float32)

    xc = node_scaler103.transform(
        raw103_check[t]["node"]
    ).astype(np.float32)

    old_xe = data103[t]["xe"].cpu().numpy()
    old_xc = data103[t]["xc"].cpu().numpy()

    edge_error = max(
        edge_error,
        float(np.max(np.abs(xe - old_xe)))
    )

    node_error = max(
        node_error,
        float(np.max(np.abs(xc - old_xc)))
    )

    assert np.allclose(xe, old_xe, atol=1e-5)
    assert np.allclose(xc, old_xc, atol=1e-5)

print("Max edge error:", edge_error)
print("Max node error:", node_error)
print("Seed 103 reconstruction: PASSED")

del raw103_check

# ------------------------------------------------------------
# 2. Build Seed 104 raw features
# ------------------------------------------------------------

print("\n[2] BUILD SEED 104 RAW FEATURES")

raw104 = build_raw_snapshots_h(SOURCE)

assert set(raw104) == set(TIMES)

expected_counts = {
    t: int(raw104[t]["count"])
    for t in TIMES
}

assert sum(
    expected_counts[t] for t in VAL
) == 43832

for t in TIMES:
    assert raw104[t]["edge"].shape == (M, 17)
    assert raw104[t]["node"].shape == (N, 4)
    assert len(raw104[t]["target"]) == raw104[t]["count"]
    assert len(raw104[t]["y"]) == raw104[t]["count"]

print("Raw input dimensions: VERIFIED")

# ------------------------------------------------------------
# 3. Train-only scalers
# ------------------------------------------------------------

print("\n[3] FIT TRAIN-ONLY SCALERS")

edge_scaler104 = StandardScaler()
node_scaler104 = StandardScaler()

for t in TRAIN:
    edge_scaler104.partial_fit(raw104[t]["edge"])
    node_scaler104.partial_fit(raw104[t]["node"])

assert int(edge_scaler104.n_samples_seen_) == M * 3
assert int(node_scaler104.n_samples_seen_) == N * 3

print(
    "Edge scaler samples:",
    int(edge_scaler104.n_samples_seen_)
)
print(
    "Node scaler samples:",
    int(node_scaler104.n_samples_seen_)
)
print("Validation used for fitting: NO")

# ------------------------------------------------------------
# 4. Construct CUDA tensors
# ------------------------------------------------------------

print("\n[4] CONSTRUCT data104")

data104 = {}

for t in TIMES:

    r = raw104[t]

    xe = edge_scaler104.transform(
        r["edge"]
    ).astype(np.float32)

    xc = node_scaler104.transform(
        r["node"]
    ).astype(np.float32)

    xn = np.concatenate([
        type_onehot.astype(np.float32),
        xc
    ], axis=1)

    data104[t] = {
        "time": t,
        "xn": torch.as_tensor(
            xn, dtype=torch.float32,
            device=U.device
        ),
        "xe": torch.as_tensor(
            xe, dtype=torch.float32,
            device=U.device
        ),
        "xc": torch.as_tensor(
            xc, dtype=torch.float32,
            device=U.device
        ),
        "target": torch.as_tensor(
            r["target"], dtype=torch.long,
            device=U.device
        ),
        "y": torch.as_tensor(
            r["y"], dtype=torch.float32,
            device=U.device
        ),
        "count": int(r["count"])
    }

    assert data104[t]["xn"].shape == (N, 20)
    assert data104[t]["xe"].shape == (M, 17)
    assert data104[t]["xc"].shape == (N, 4)

    print(
        f"t={t} | "
        f"events={r['count']:,} | "
        f"failure={data104[t]['y'].mean().item():.4f}",
        flush=True
    )

# ------------------------------------------------------------
# 5. Save report and scalers
# ------------------------------------------------------------

OUT.mkdir(parents=True)

joblib.dump(
    edge_scaler104,
    OUT / "edge_scaler104.joblib"
)
joblib.dump(
    node_scaler104,
    OUT / "node_scaler104.joblib"
)

report = {
    "experiment": "E3.30",
    "simulation_seed": 104,
    "train_times": TRAIN,
    "validation_times": VAL,
    "edge_dim": 17,
    "node_dim": 20,
    "edge_budget": K,
    "reference_reconstruction_edge_error": edge_error,
    "reference_reconstruction_node_error": node_error,
    "test_accessed": False,
    "hidden_state_accessed": False,
    "training_performed": False
}

with open(OUT / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("E3.30 COMPLETED")
print("=" * 70)
print("data104 READY ON", U.device)
print("Validation events:", sum(data104[t]["count"] for t in VAL))
print("Train-only normalization: VERIFIED")
print("Test accessed: NO")
print("Saved:", OUT / "report.json")

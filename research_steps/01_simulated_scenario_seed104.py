
# ============================================================
# E3.28 — SCENARIO B / SIMULATION SEED 104
# ============================================================
# Same simulation procedure as E3.21.
# Change only the independent simulation seed.
#
# Train: t=0..7
# Validation: t=8..9
# Test: t=10..11, GENERATED AND SEALED
#
# No model training.
# No prior artifacts overwritten.
# ============================================================

from pathlib import Path
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd

ROOT328 = Path(
    "/kaggle/working/phase2_adaptive_siot"
)
BASE328 = (
    ROOT328 /
    "E3_tags_v3/scenarios/B_owner_correlated"
)

REF328 = BASE328 / "seed_103"
OUT328 = BASE328 / "seed_104"

if OUT328.exists():
    raise FileExistsError(
        f"Existing output: {OUT328}"
    )

SIM_SEED328 = 104
N328 = 16216
M328 = 146117
T328 = 12

# Frozen Scenario B parameters
P328 = {
    "base_reliability": "Beta(5,2)",
    "owner_effect_std": 0.65,
    "edge_effect_std": 0.45,
    "temporal_rho": 0.85,
    "temporal_std": 0.12,
    "interaction_rate": 0.15,
    "observation_flip": 0.03,
    "shock_start": 6,
    "shock_fraction": 0.08,
    "shock_logit": -1.6,
    "shock_decay": 0.90
}

# ------------------------------------------------------------
# 1. Verify parameters against E3.21
# ------------------------------------------------------------

with open(
    REF328 / "e3_21_report.json",
    encoding="utf-8"
) as f:
    ref_report328 = json.load(f)

assert ref_report328["simulation_seed"] == 103
assert ref_report328["parameters"] == P328

topology328 = pd.read_csv(
    REF328 /
    "observed/siot_social_topology.csv.gz"
).sort_values(
    ["u", "v"]
).reset_index(drop=True)

assert len(topology328) == M328

u328 = (
    topology328["u"].to_numpy(np.int32) - 1
)
v328 = (
    topology328["v"].to_numpy(np.int32) - 1
)

assert np.all(u328 < v328)
assert u328.min() >= 0
assert v328.max() < N328

archive328 = (
    ROOT328 /
    "E1_trust_scoring/santander/raw/"
    "objects_description.zip"
)

with zipfile.ZipFile(archive328) as z:
    with z.open("objects_description.csv") as f:
        devices328 = pd.read_csv(f)

devices328 = devices328.sort_values(
    "id_device"
).reset_index(drop=True)

assert np.array_equal(
    devices328["id_device"].to_numpy(),
    np.arange(1, N328 + 1)
)

owner_codes328, owner_ids328 = pd.factorize(
    devices328["id_user"],
    sort=True
)

O328 = len(owner_ids328)
assert O328 == 4001

assert int(
    (owner_codes328[u328] ==
     owner_codes328[v328]).sum()
) == 65667

# ------------------------------------------------------------
# 2. Latent simulator initialization
# ------------------------------------------------------------

rng328 = np.random.default_rng(SIM_SEED328)

def sigmoid328(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -30, 30))
    )

def logit328(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

base_reliability328 = rng328.beta(
    5.0, 2.0, size=N328
)

owner_effect328 = rng328.normal(
    0.0, P328["owner_effect_std"], size=O328
)

device_owner_effect328 = (
    owner_effect328[owner_codes328]
)

edge_effect328 = rng328.normal(
    0.0, P328["edge_effect_std"], size=M328
)

base_logit328 = logit328(
    base_reliability328
)

shock_count328 = int(round(
    P328["shock_fraction"] * N328
))

shocked_nodes328 = rng328.choice(
    N328,
    size=shock_count328,
    replace=False
)

shock_indicator328 = np.zeros(
    N328, dtype=np.float32
)

shock_indicator328[shocked_nodes328] = 1.0

temporal_state328 = np.zeros(
    N328, dtype=np.float32
)

# ------------------------------------------------------------
# 3. Generate events in frozen RNG order
# ------------------------------------------------------------

frames328 = []
audit328 = []

print("=" * 72)
print("E3.28 — SCENARIO B / SIMULATION SEED 104")
print("=" * 72)

print("Nodes:", N328)
print("Owners:", O328)
print("Undirected edges:", M328)
print("Parameters match Seed 103:", True)

for t in range(T328):

    temporal_state328 = (
        P328["temporal_rho"] * temporal_state328
        + rng328.normal(
            0.0,
            P328["temporal_std"],
            size=N328
        )
    ).astype(np.float32)

    if t >= P328["shock_start"]:
        shock_strength328 = (
            P328["shock_logit"]
            * P328["shock_decay"] **
            (t - P328["shock_start"])
        )
    else:
        shock_strength328 = 0.0

    device_logits328 = (
        base_logit328
        + device_owner_effect328
        + temporal_state328
        + shock_strength328 * shock_indicator328
    )

    edge_logits328 = (
        0.5 * (
            device_logits328[u328]
            + device_logits328[v328]
        )
        + edge_effect328
    )

    probs328 = sigmoid328(edge_logits328)

    selected328 = np.flatnonzero(
        rng328.random(M328)
        < P328["interaction_rate"]
    )

    success328 = (
        rng328.random(len(selected328))
        < probs328[selected328]
    ).astype(np.int8)

    flip328 = (
        rng328.random(len(selected328))
        < P328["observation_flip"]
    )

    success328[flip328] = (
        1 - success328[flip328]
    )

    frames328.append(
        pd.DataFrame({
            "time": np.full(
                len(selected328), t,
                dtype=np.int16
            ),
            "source": (
                u328[selected328] + 1
            ).astype(np.int32),
            "target": (
                v328[selected328] + 1
            ).astype(np.int32),
            "success": success328
        })
    )

    if t <= 9:
        audit328.append({
            "time": int(t),
            "observations": int(len(selected328)),
            "success_rate": float(
                success328.mean()
            )
        })

        print(
            f"t={t:02d} | "
            f"observations={len(selected328):,} | "
            f"success={success328.mean():.4f}"
        )
    else:
        print(
            f"t={t:02d} | "
            "TEST GENERATED — METRICS SEALED"
        )

# ------------------------------------------------------------
# 4. Separate temporal partitions
# ------------------------------------------------------------

events328 = pd.concat(
    frames328, ignore_index=True
).sort_values(
    ["time", "source", "target"]
).reset_index(drop=True)

train328 = events328[
    events328["time"] <= 7
].copy()

validation328 = events328[
    events328["time"].between(8, 9)
].copy()

# Do not calculate test metrics.
test328 = events328[
    events328["time"] >= 10
].copy()

assert len(events328) == (
    len(train328)
    + len(validation328)
    + len(test328)
)

assert not events328[
    ["time", "source", "target"]
].duplicated().any()

assert train328["time"].between(0, 7).all()
assert validation328["time"].between(8, 9).all()

# ------------------------------------------------------------
# 5. Persist separate data partitions
# ------------------------------------------------------------

OBS328 = OUT328 / "observed"
HIDDEN328 = OUT328 / "simulation_ground_truth"

OBS328.mkdir(parents=True)
HIDDEN328.mkdir()

topology328.to_csv(
    OBS328 / "siot_social_topology.csv.gz",
    compression="gzip",
    index=False
)

train328.to_csv(
    OBS328 / "train.csv.gz",
    compression="gzip",
    index=False
)

validation328.to_csv(
    OBS328 / "validation.csv.gz",
    compression="gzip",
    index=False
)

# Sealed partition: write only.
test328.to_csv(
    OBS328 / "test.csv.gz",
    compression="gzip",
    index=False
)

# Simulation-only ground truth, excluded from learning.
np.savez_compressed(
    HIDDEN328 / "hidden_simulator_state.npz",
    base_reliability=base_reliability328,
    owner_effect=owner_effect328,
    owner_codes=owner_codes328,
    edge_effect=edge_effect328,
    shocked_device_ids=shocked_nodes328 + 1
)

def sha256_328(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

report328 = {
    "experiment": "E3.28",
    "scenario": "B_owner_correlated",
    "simulation_seed": SIM_SEED328,
    "reference_experiment": "E3.21",
    "parameters": P328,
    "nodes": N328,
    "owners": O328,
    "undirected_edges": M328,
    "train_events": int(len(train328)),
    "validation_events": int(len(validation328)),
    "permitted_temporal_audit": audit328,
    "train_sha256": sha256_328(
        OBS328 / "train.csv.gz"
    ),
    "validation_sha256": sha256_328(
        OBS328 / "validation.csv.gz"
    ),
    "test_generated_but_not_evaluated": True,
    "hidden_state_not_used_for_learning": True,
    "model_training_performed": False,
    "prior_artifacts_overwritten": False
}

REPORT328 = OUT328 / "e3_28_report.json"

with open(
    REPORT328, "w", encoding="utf-8"
) as f:
    json.dump(report328, f, indent=2)

# Remove sensitive simulation and sealed-test
# variables from notebook memory.
del test328, events328, frames328
del probs328, edge_logits328, device_logits328
del success328, flip328, selected328
del base_reliability328, owner_effect328
del device_owner_effect328, edge_effect328
del shocked_nodes328, shock_indicator328
del temporal_state328, base_logit328

print("\n" + "=" * 72)
print("E3.28 COMPLETED — SEED 104")
print("=" * 72)
print("Train interactions:", len(train328))
print("Validation interactions:", len(validation328))
print("Test: SEALED")
print("Parameters changed: SEED ONLY")
print("Training performed: NO")
print("Previous artifacts overwritten: NO")
print("Saved:", REPORT328)
print("NEXT: E3.29 — causal feature snapshots")

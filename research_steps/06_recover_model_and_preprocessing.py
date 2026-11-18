# TAGS-v3 / E3.RECOVER — faithful source extraction from uploaded notebook
# READ ONLY: no optimizer steps; never loads test.csv.gz or ground truth.
# Extracted from original notebook cells; do not run old training cells.
from pathlib import Path
import copy, json, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

ROOT = Path('/kaggle/working/phase2_adaptive_siot')
SOURCE = ROOT / 'E3_tags_v3/scenarios/B_owner_correlated/seed_104'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type == 'cuda', 'Original validation uses CUDA; enable Kaggle GPU.'
N = N_H = N_GPU = 16216
M = M_H = M_GPU = 146117
K = 73058
REL_H = ['OOR', 'POR', 'C_LOR', 'SOR', 'SOR2']
TIMES_H = [5,6,7,8,9]
VAL324 = [8,9]
M324 = M

# Canonical edge order: do NOT sort independently, must equal saved mapping.
topology = pd.read_csv(SOURCE / 'causal_features/edge_index_mapping.csv.gz')
assert len(topology) == M
U = torch.as_tensor(topology['u'].to_numpy(np.int64)-1, dtype=torch.long, device=DEVICE)
V = torch.as_tensor(topology['v'].to_numpy(np.int64)-1, dtype=torch.long, device=DEVICE)
assert bool((U < V).all())
assert bool((U[1:] * N + V[1:] > U[:-1] * N + V[:-1]).all())

# The original 16-way device type one-hot from the source archive.
archive = ROOT / 'E1_trust_scoring/santander/raw/objects_description.zip'
assert archive.is_file(), f'Missing original device metadata: {archive}'
with zipfile.ZipFile(archive) as z:
    with z.open('objects_description.csv') as f:
        devices = pd.read_csv(f)
devices = devices.sort_values('id_device')
assert np.array_equal(devices['id_device'].to_numpy(), np.arange(1,N+1))
type_codes, type_names = pd.factorize(devices['device_type'], sort=True)
type_onehot = np.eye(len(type_names), dtype=np.float32)[type_codes]
assert type_onehot.shape == (N,16)

edge_scaler104 = joblib.load(SOURCE / 'e3_30_model_inputs/edge_scaler104.joblib')
node_scaler104 = joblib.load(SOURCE / 'e3_30_model_inputs/node_scaler104.joblib')
assert edge_scaler104.n_features_in_ == 17
assert node_scaler104.n_features_in_ == 4


# ========== original SAGE layer (notebook cell 101) ==========
class TAGSWeightedSAGE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_map = nn.Linear(in_dim, out_dim)
        self.neighbor_map = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, u, v, weight):
        w = weight[:, None]

        msg = torch.zeros_like(x)
        msg = msg.index_add(0, u, x[v] * w)
        msg = msg.index_add(0, v, x[u] * w)

        deg = x.new_zeros((x.shape[0], 1))
        deg = deg.index_add(0, u, w)
        deg = deg.index_add(0, v, w)

        agg = msg / deg.clamp_min(1e-6)

        return (
            self.self_map(x)
            + self.neighbor_map(agg)
        )


# ========== original TAGS encoder (notebook cell 101) ==========
class TAGSEncoder(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.first = TAGSWeightedSAGE(
            input_dim, hidden
        )
        self.second = TAGSWeightedSAGE(
            hidden, hidden
        )

    def forward(self, x, u, v, weights):
        h = F.relu(
            self.first(x, u, v, weights)
        )
        return self.second(h, u, v, weights)


# ========== original TAGS v3 (notebook cell 101) ==========
class TAGSv3(nn.Module):
    def __init__(
        self, node_dim, edge_dim, controller_dim,
        hidden=32
    ):
        super().__init__()

        self.encoder = TAGSEncoder(
            node_dim, hidden
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.controller = nn.Sequential(
            nn.Linear(controller_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Direct trust pathway.
        self.trust_head = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # GNN correction pathway.
        self.graph_head = nn.Sequential(
            nn.Linear(hidden * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self, xn, xe, xc, u, v, target,
        keep_count, tau=1.0
    ):
        full = xn.new_ones(len(u))

        # Full-graph pass for edge selection.
        h = self.encoder(
            xn, u, v, full
        )

        edge_rep = torch.cat([
            h[u] + h[v],
            torch.abs(h[u] - h[v]),
            xe
        ], dim=1)

        edge_score = self.scorer(
            edge_rep
        ).squeeze(-1)

        priority = self.controller(
            xc
        ).squeeze(-1)

        score = edge_score + 0.5 * (
            priority[u] + priority[v]
        )

        # Shared symmetric decision for each
        # undirected social relationship.
        mask, hard = straight_through_topk(
            score, keep_count, tau
        )

        # Sparse-graph pass.
        z = self.encoder(
            xn, u, v, mask
        )

        a = u[target]
        b = v[target]

        pair = torch.cat([
            z[a] + z[b],
            torch.abs(z[a] - z[b]),
            z[a] * z[b]
        ], dim=1)

        trust_logit = self.trust_head(
            xe[target]
        ).squeeze(-1)

        graph_logit = self.graph_head(
            pair
        ).squeeze(-1)

        return {
            "logits": trust_logit + graph_logit,
            "scores": score,
            "mask": mask,
            "hard_mask": hard,
            "trust_logits": trust_logit,
            "graph_logits": graph_logit
        }


# ========== stabilized SAGE layer (notebook cell 102) ==========
class StableTAGSWeightedSAGE(TAGSWeightedSAGE):

    def forward(self, x, u, v, weight):
        w = weight.reshape(-1, 1)

        msg = torch.zeros_like(x)
        msg = msg.index_add(0, u, x[v] * w)
        msg = msg.index_add(0, v, x[u] * w)

        deg = x.new_zeros((x.size(0), 1))
        deg = deg.index_add(0, u, w)
        deg = deg.index_add(0, v, w)

        # Stabilized weighted aggregation:
        # avoids near-zero denominator gradients.
        agg = msg / (1.0 + deg)

        return (
            self.self_map(x)
            + self.neighbor_map(agg)
        )


# ========== stabilized encoder (notebook cell 102) ==========
class StableTAGSEncoder(nn.Module):

    def __init__(self, input_dim, hidden=32):
        super().__init__()

        self.first = StableTAGSWeightedSAGE(
            input_dim, hidden
        )
        self.second = StableTAGSWeightedSAGE(
            hidden, hidden
        )

    def forward(self, x, u, v, weights):
        h = F.relu(
            self.first(x, u, v, weights)
        )
        return self.second(h, u, v, weights)


# ========== stabilized TAGS (notebook cell 102) ==========
class TAGSv3Stable(TAGSv3):

    def __init__(
        self, node_dim, edge_dim,
        controller_dim, hidden=32
    ):
        super().__init__(
            node_dim,
            edge_dim,
            controller_dim,
            hidden
        )

        self.encoder = StableTAGSEncoder(
            node_dim, hidden
        )


# ========== original raw input builder (notebook cell 132) ==========
def build_raw_snapshots_h(folder):

    topology = pd.read_csv(
        folder / "causal_features/"
        "edge_index_mapping.csv.gz"
    )

    assert len(topology) == M_H

    u = topology["u"].to_numpy(np.int64) - 1
    v = topology["v"].to_numpy(np.int64) - 1

    reference_u = U.detach().cpu().numpy()
    reference_v = V.detach().cpu().numpy()

    assert np.array_equal(u, reference_u)
    assert np.array_equal(v, reference_v)

    relations = topology[REL_H].to_numpy(
        dtype=np.float32
    )

    static_degree = np.bincount(
        np.concatenate([u, v]),
        minlength=N_H
    )

    degree_feature = np.log1p(
        static_degree
    ).astype(np.float32)

    # Only allowed observed outcomes.
    train = pd.read_csv(
        folder / "observed/train.csv.gz"
    )

    val = pd.read_csv(
        folder / "observed/validation.csv.gz"
    )

    assert train["time"].between(0, 7).all()
    assert val["time"].between(8, 9).all()

    assert train["success"].isin([0, 1]).all()
    assert val["success"].isin([0, 1]).all()

    events = pd.concat(
        [train, val], ignore_index=True
    )

    # Canonical edge mapping.
    keys = u * N_H + v

    assert np.all(np.diff(keys) > 0)

    source = events["source"].to_numpy(
        np.int64
    ) - 1

    target = events["target"].to_numpy(
        np.int64
    ) - 1

    assert np.all(source < target)

    event_keys = source * N_H + target

    event_edge_ids = np.searchsorted(
        keys, event_keys
    )

    assert np.all(event_edge_ids < M_H)
    assert np.array_equal(
        keys[event_edge_ids], event_keys
    )

    event_times = events["time"].to_numpy(
        np.int16
    )

    event_success = events["success"].to_numpy(
        np.float64
    )

    raw = {}

    for t in TIMES_H:

        # Strictly historical observations.
        # At t=9, t=8 validation observations may
        # contribute to causal history, but never
        # to model/scaler fitting.
        past = event_times < t

        past_u = source[past]
        past_v = target[past]
        past_success = event_success[past]

        node_count = np.bincount(
            np.concatenate([past_u, past_v]),
            minlength=N_H
        )

        node_success = (
            np.bincount(
                past_u,
                weights=past_success,
                minlength=N_H
            )
            + np.bincount(
                past_v,
                weights=past_success,
                minlength=N_H
            )
        )

        node_raw = np.column_stack([
            degree_feature,
            np.log1p(node_count),
            (node_success + 1.0) /
            (node_count + 2.0),
            node_count / (node_count + 5.0)
        ]).astype(np.float32)

        assert node_raw.shape == (N_H, 4)

        with np.load(
            folder / "causal_features" /
            f"snapshot_t{t:02d}.npz"
        ) as archive:

            history = archive["features"].astype(
                np.float32
            )

            assert int(archive["time"]) == t

        assert history.shape == (M_H, 12)

        # Independent temporal-integrity check.
        assert abs(
            float(history[:, 0].sum())
            - int(past.sum())
        ) < 1.0

        ru = node_raw[u, 2]
        rv = node_raw[v, 2]

        cu = node_raw[u, 3]
        cv = node_raw[v, 3]

        endpoints = np.column_stack([
            (ru + rv) / 2.0,
            np.abs(ru - rv),
            (cu + cv) / 2.0,
            np.abs(cu - cv)
        ]).astype(np.float32)

        edge_raw = np.column_stack([
            np.log1p(history[:, 0]),
            history[:, 1],
            np.log1p(history[:, 2]),
            history[:, 3:7],
            endpoints,
            history[:, 11],
            relations
        ]).astype(np.float32)

        assert edge_raw.shape == (M_H, 17)

        current = event_times == t

        current_ids = event_edge_ids[current]

        # Failure = 1 - success.
        failures = (
            1.0 - event_success[current]
        ).astype(np.float32)

        assert len(current_ids) == len(failures)

        raw[t] = {
            "node": node_raw,
            "edge": edge_raw,
            "target": current_ids.astype(np.int64),
            "y": failures,
            "count": int(len(current_ids))
        }

    return raw


# ========== original GPU selector (notebook cell 115) ==========
@torch.no_grad()
def coverage_topk_gpu(scores, k):
    """
    Exact equivalent of the existing CPU greedy
    coverage selection, including stable tie order.

    Phase 1:
      Select the highest-priority incident edge
      for every node, then take their union.

    Phase 2:
      Fill the remaining budget with the
      highest-ranked unselected edges.
    """

    assert scores.is_cuda
    assert scores.ndim == 1
    assert scores.numel() == M_GPU
    assert bool(torch.isfinite(scores).all())
    assert 0 < k <= M_GPU

    device = scores.device

    # Same ranking semantics as:
    # np.argsort(-scores, kind="stable")
    order = torch.argsort(
        scores.detach(),
        descending=True,
        stable=True
    )

    rank = torch.empty_like(order)

    rank[order] = torch.arange(
        M_GPU,
        device=device,
        dtype=torch.long
    )

    # For each node, find its earliest incident
    # edge in the globally ranked ordering.
    first_rank = torch.full(
        (N_GPU,),
        M_GPU,
        dtype=torch.long,
        device=device
    )

    first_rank.scatter_reduce_(
        0, U, rank,
        reduce="amin",
        include_self=True
    )

    first_rank.scatter_reduce_(
        0, V, rank,
        reduce="amin",
        include_self=True
    )

    if bool((first_rank == M_GPU).any()):
        raise RuntimeError(
            "Original topology contains isolated nodes."
        )

    # One highest-ranked incident edge per node.
    # Multiple nodes can select the same edge.
    cover_ids = order[first_rank]

    selected = torch.zeros(
        M_GPU,
        dtype=torch.bool,
        device=device
    )

    selected[cover_ids] = True

    coverage_count = int(
        selected.sum().item()
    )

    if coverage_count > k:
        raise RuntimeError(
            "Node coverage exceeds edge budget."
        )

    # Complete the exact global budget, using
    # the original ranking order.
    remaining = k - coverage_count

    if remaining > 0:
        available = order[
            ~selected[order]
        ]

        selected[
            available[:remaining]
        ] = True

    hard = selected.to(
        dtype=scores.dtype
    )

    return hard, coverage_count


# ========== original scorer forward (notebook cell 140) ==========
def forward324(net, batch, use_gnn, training):

    xn = batch["xn"]
    xe = batch["xe"]
    ids = batch["target"]

    if use_gnn:
        full = xn.new_ones(M324)

        h = net.encoder(
            xn, U, V, full
        )

        scoring_input = torch.cat([
            h[U] + h[V],
            torch.abs(h[U] - h[V]),
            xe
        ], dim=1)

        scores = net.scorer(
            scoring_input
        ).squeeze(-1)
    else:
        scores = net.scorer(
            xe
        ).squeeze(-1)

    hard, coverage_count = coverage_topk_gpu(
        scores, K
    )

    if training:
        threshold = torch.topk(
            scores.detach(), K
        ).values[-1]

        soft = torch.sigmoid(
            scores - threshold
        )

        mask = hard + (soft - soft.detach())
    else:
        mask = hard

    z = net.encoder(
        xn, U, V, mask
    )

    a = U[ids]
    b = V[ids]

    pair_features = torch.cat([
        z[a] + z[b],
        torch.abs(z[a] - z[b]),
        z[a] * z[b]
    ], dim=1)

    logits = (
        net.trust_head(
            xe[ids]
        ).squeeze(-1)
        +
        net.graph_head(
            pair_features
        ).squeeze(-1)
    )

    return logits, hard, int(coverage_count)


# ========== original validation (notebook cell 140) ==========
@torch.no_grad()
def validate324(net, use_gnn, dataset):

    net.eval()

    ys, ps, masks = [], [], []
    cover_counts = []

    for t in VAL324:
        batch = dataset[t]

        logits, hard, coverage = forward324(
            net, batch, use_gnn, False
        )

        ys.append(
            batch["y"].detach().cpu().numpy()
        )

        ps.append(
            torch.sigmoid(logits)
            .detach().cpu().numpy()
        )

        masks.append(
            hard.detach().cpu().numpy().astype(bool)
        )

        cover_counts.append(coverage)

    y = np.concatenate(ys)
    p = np.clip(
        np.concatenate(ps),
        1e-7,
        1 - 1e-7
    )

    assert all(
        int(mask.sum()) == K
        for mask in masks
    )

    return {
        "AP": float(
            average_precision_score(y, p)
        ),
        "AUC": float(
            roc_auc_score(y, p)
        ),
        "BCE": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "churn": float(
            np.count_nonzero(
                masks[0] != masks[1]
            ) / (2 * K)
        ),
        "coverage_t8": cover_counts[0],
        "coverage_t9": cover_counts[1]
    }


# Reconstruct original E3.30 inputs using saved TRAIN-ONLY scalers.
raw104 = build_raw_snapshots_h(SOURCE)
assert set(raw104) == set(TIMES_H)
data104 = {}
for t in TIMES_H:
    r = raw104[t]
    xe = edge_scaler104.transform(r['edge']).astype(np.float32)
    xc = node_scaler104.transform(r['node']).astype(np.float32)
    xn = np.concatenate([type_onehot.astype(np.float32), xc], axis=1)
    data104[t] = {
        'time': t,
        'xn': torch.as_tensor(xn, dtype=torch.float32, device=DEVICE),
        'xe': torch.as_tensor(xe, dtype=torch.float32, device=DEVICE),
        'xc': torch.as_tensor(xc, dtype=torch.float32, device=DEVICE),
        'target': torch.as_tensor(r['target'], dtype=torch.long, device=DEVICE),
        'y': torch.as_tensor(r['y'], dtype=torch.float32, device=DEVICE),
        'count': int(r['count'])
    }
    assert data104[t]['xn'].shape == (N,20)
    assert data104[t]['xe'].shape == (M,17)
    assert data104[t]['xc'].shape == (N,4)
    print(f'RESTORED t={t}, observations={r["count"]:,}')
assert sum(data104[t]['count'] for t in VAL324) == 43832

stable = TAGSv3Stable(20,17,4,hidden=32).to(DEVICE)

# Strict, paired metric reproduction from the REAL saved epoch60 checkpoints.
# Scientific check: require differences <=2e-5 for AP, AUC, BCE and churn.
# No file writes.
for seed in [44,45,46]:
    report_file = SOURCE / 'tags_v3'/f'e3_31_model_seed_{seed}'/'report.json'
    report = json.loads(report_file.read_text(encoding='utf-8'))
    assert report['test_accessed'] is False
    for item in report['results']:
        method = item['model']
        ckpt = SOURCE / 'tags_v3'/f'e3_31_model_seed_{seed}'/f'{method}_checkpoint.pt'
        saved = torch.load(ckpt, map_location='cpu',weights_only=True)
        net = TAGSv3Stable(20,17,4,hidden=32).to(DEVICE)
        if method == 'Trust_only_scorer':
            net.scorer = nn.Sequential(nn.Linear(17,64), nn.ReLU(), nn.Linear(64,1)).to(DEVICE)
        net.load_state_dict(saved['epoch60_model_state_dict'], strict=True)
        actual = validate324(net, method=='GNN_scorer', data104)
        print(f'SEED {seed} {method}: AP={actual["AP"]:.6f}, AUC={actual["AUC"]:.6f}, BCE={actual["BCE"]:.6f}, churn={actual["churn"]:.6f}')
        for key in ['AP','AUC','BCE','churn']:
            expected = float(item['epoch60_'+key])
            error = abs(actual[key] - expected)
            if error > 2e-5:
                raise AssertionError(f'VALIDATION REPRODUCTION FAILED: {seed}, {method}, {key}, expected={expected}, actual={actual[key]}, error={error}. DO NOT TRAIN.')
        del net

print('\nE3.RECOVERY PASSED: all six epoch60 models reproduce original validation metrics.')
print('data104, stable, U, V, K, coverage_topk_gpu, forward324 and validate324 are ready.')
print('No training; no test; no files overwritten.')

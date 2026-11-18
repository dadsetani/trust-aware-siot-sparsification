
# ============================================================
# E3.34 — BUDGET SENSITIVITY, RETRAINED SCORERS
# ============================================================
# Scenario seed: 104 | Model seed: 44
# Budgets: 10%, 25%, 75%
# Compare GNN scorer vs Trust-only scorer.
#
# 50% results: existing E3.31
# Full graph: existing E3.33
#
# All new models trained independently for 60 epochs.
# Exact coverage-aware GPU selection.
# No test or hidden-state access.
# No existing outputs overwritten.
# ============================================================

from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss
)

required = [
    "stable", "data104", "U", "V",
    "coverage_topk_gpu"
]
missing = [x for x in required if x not in globals()]
assert not missing, f"Missing: {missing}"

ROOT334 = Path(
    "/kaggle/working/phase2_adaptive_siot/"
    "E3_tags_v3/scenarios/B_owner_correlated/"
    "seed_104"
)
OUT334 = ROOT334 / "budget_study/e3_34a_seed44"

if OUT334.exists():
    raise FileExistsError(OUT334)

MODEL_SEED = 44
BUDGETS = [0, 25, 0.75]
EPOCHS = 60
TRAIN = [5, 6, 7]
VALID = [8, 9]

M = U.numel()
N = data104[5]["xn"].shape[0]
assert M == 146117 and N == 16216
assert U.is_cuda and V.is_cuda

def reset334(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_mask334(mask, k):
    if int(mask.sum().item()) != k:
        raise RuntimeError(
            f"Budget violation: {int(mask.sum())} != {k}"
        )
    idx = torch.where(mask > 0)[0]
    coverage = torch.zeros(
        N, device=U.device, dtype=torch.bool
    )
    coverage[U[idx]] = True
    coverage[V[idx]] = True
    if not bool(coverage.all()):
        raise RuntimeError("Node coverage violated")

def forward334(model, batch, use_gnn, budget_k, training):
    xn, xe, ids = (
        batch["xn"], batch["xe"], batch["target"]
    )

    if use_gnn:
        full = xn.new_ones(M)
        h = model.encoder(xn, U, V, full)
        scorer_input = torch.cat([
            h[U] + h[V],
            (h[U] - h[V]).abs(),
            xe
        ], dim=1)
        scores = model.scorer(scorer_input).squeeze(-1)
    else:
        scores = model.scorer(xe).squeeze(-1)

    hard, _ = coverage_topk_gpu(scores, budget_k)
    check_mask334(hard, budget_k)

    if training:
        threshold = torch.topk(
            scores.detach(), budget_k
        ).values[-1]
        soft = torch.sigmoid(scores - threshold)
        mask = hard + soft - soft.detach()
    else:
        mask = hard

    z = model.encoder(xn, U, V, mask)
    a, b = U[ids], V[ids]

    pair = torch.cat([
        z[a] + z[b],
        (z[a] - z[b]).abs(),
        z[a] * z[b]
    ], dim=1)

    logits = (
        model.trust_head(xe[ids]).squeeze(-1)
        + model.graph_head(pair).squeeze(-1)
    )
    return logits, hard

@torch.no_grad()
def evaluate334(model, use_gnn, budget_k):
    model.eval()
    ys, ps, masks = [], [], []

    for t in VALID:
        batch = data104[t]
        logits, mask = forward334(
            model, batch, use_gnn, budget_k, False
        )
        ys.append(batch["y"].cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
        masks.append(mask.cpu().numpy().astype(bool))

    y = np.concatenate(ys)
    p = np.clip(np.concatenate(ps), 1e-7, 1 - 1e-7)

    return {
        "AP": float(average_precision_score(y, p)),
        "AUC": float(roc_auc_score(y, p)),
        "BCE": float(log_loss(y, p, labels=[0, 1])),
        "churn": float(
            np.count_nonzero(masks[0] != masks[1])
            / (2 * budget_k)
        )
    }

def build_pair334():
    reset334(MODEL_SEED)

    base = type(stable)(
        node_dim=20,
        edge_dim=17,
        controller_dim=4,
        hidden=32
    ).to(U.device)

    gnn = copy.deepcopy(base)
    trust = copy.deepcopy(base)
    trust.scorer = nn.Sequential(
        nn.Linear(17, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(U.device)

    for key in base.state_dict():
        if key.startswith(
            ("encoder.", "trust_head.", "graph_head.")
        ):
            assert torch.equal(
                gnn.state_dict()[key],
                trust.state_dict()[key]
            )
    return gnn, trust

# ------------------------------------------------------------
# Feasibility check BEFORE training or writing
# ------------------------------------------------------------

print("=" * 75)
print("E3.34 — RETRAINED BUDGET SENSITIVITY")
print("=" * 75)

print("Scenario seed: 104")
print("Model seed: 44")
print("Budget fractions:", BUDGETS)

with torch.no_grad():
    probe_gnn, probe_trust = build_pair334()
    for frac in BUDGETS:
        k = int(round(M * frac))
        for name, net, use_gnn in [
            ("GNN", probe_gnn, True),
            ("Trust", probe_trust, False)
        ]:
            for t in TRAIN + VALID:
                try:
                    _, mask = forward334(
                        net, data104[t], use_gnn, k, False
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Budget {frac:.0%} is infeasible "
                        f"or selector failed for {name} "
                        f"at t={t}: {exc}"
                    ) from exc

        print(f"Budget {frac:.0%}: k={k:,} PASSED")

    del probe_gnn, probe_trust
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Train and persist each budget independently
# ------------------------------------------------------------

OUT334.mkdir(parents=True, exist_ok=False)

records = []

for frac in BUDGETS:
    k = int(round(M * frac))
    gnn, trust = build_pair334()

    for name, model, use_gnn in [
        ("GNN_scorer", gnn, True),
        ("Trust_only_scorer", trust, False)
    ]:
        reset334(MODEL_SEED)

        params = [
            p for param_name, p in model.named_parameters()
            if not param_name.startswith("controller.")
        ]
        optimizer = torch.optim.AdamW(
            params, lr=1e-3, weight_decay=1e-4
        )

        history = []

        print(
            f"\nBudget={frac:.0%} | "
            f"edges={k:,} | model={name}",
            flush=True
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            total_n = 0

            for t in TRAIN:
                batch = data104[t]
                optimizer.zero_grad(set_to_none=True)

                logits, _ = forward334(
                    model, batch, use_gnn, k, True
                )

                loss = F.binary_cross_entropy_with_logits(
                    logits, batch["y"]
                )

                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite loss")

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    params, 1.0
                )
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite gradient")
                optimizer.step()

                n = batch["count"]
                total_loss += float(loss.item()) * n
                total_n += n

            metrics = evaluate334(model, use_gnn, k)
            history.append({
                "epoch": epoch,
                "train_BCE": total_loss / total_n,
                **metrics
            })

            if epoch in [1, 20, 40, 60]:
                print(
                    f"epoch={epoch:02d} | "
                    f"AP={metrics['AP']:.6f} | "
                    f"AUC={metrics['AUC']:.6f} | "
                    f"BCE={metrics['BCE']:.6f}",
                    flush=True
                )

        final = history[-1]

        record = {
            "scenario_seed": 104,
            "model_seed": MODEL_SEED,
            "budget_fraction": frac,
            "retained_edges": k,
            "model": name,
            "epoch60_AP": final["AP"],
            "epoch60_AUC": final["AUC"],
            "epoch60_BCE": final["BCE"],
            "epoch60_churn": final["churn"]
        }
        records.append(record)

        tag = f"{name}_budget_{int(frac*100)}"

        pd.DataFrame(history).to_csv(
            OUT334 / f"{tag}_history.csv", index=False
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "budget": frac,
                "retained_edges": k,
                "model_seed": MODEL_SEED,
                "simulation_seed": 104,
                "epoch": 60
            },
            OUT334 / f"{tag}_checkpoint.pt"
        )

    del gnn, trust
    torch.cuda.empty_cache()

result = pd.DataFrame(records)
result.to_csv(
    OUT334 / "results.csv", index=False
)

# Existing 50% results, no retraining.
for seed in [44]:
    path = (
        ROOT334 / "tags_v3"
        / f"e3_31_model_seed_{seed}/report.json"
    )
    with open(path, encoding="utf-8") as f:
        old = json.load(f)

    for row in old["results"]:
        records.append({
            "scenario_seed": 104,
            "model_seed": seed,
            "budget_fraction": 0.50,
            "retained_edges": 73058,
            "model": row["model"],
            "epoch60_AP": row["epoch60_AP"],
            "epoch60_AUC": row["epoch60_AUC"],
            "epoch60_BCE": row["epoch60_BCE"],
            "epoch60_churn": row["epoch60_churn"]
        })

combined = pd.DataFrame(records).sort_values(
    ["budget_fraction", "model"]
)
combined.to_csv(
    OUT334 / "budget_curve_with_existing_50.csv",
    index=False
)

pivot = combined.pivot(
    index="budget_fraction",
    columns="model",
    values="epoch60_AP"
)
pivot["delta_AP"] = (
    pivot["GNN_scorer"] - pivot["Trust_only_scorer"]
)

with open(
    OUT334 / "report.json", "w", encoding="utf-8"
) as f:
    json.dump({
        "experiment": "E3.34",
        "simulation_seed": 104,
        "model_seed": 44,
        "new_budgets": BUDGETS,
        "existing_budget": 0.50,
        "epochs": EPOCHS,
        "checkpoint": "epoch60",
        "design": "independent retraining per budget",
        "all_node_coverage_checked": True,
        "test_accessed": False,
        "hidden_state_accessed": False,
        "limitations": [
            "One scenario seed and one model seed.",
            "Exploratory budget sensitivity only.",
            "No statistical inference from this pilot."
        ]
    }, f, indent=2)

print("\n" + "=" * 75)
print("E3.34 COMPLETED — BUDGET SENSITIVITY")
print("=" * 75)
print("\nAP BY BUDGET")
print(pivot.round(6).to_string())

print("\nALL METRICS")
print(combined.round(6).to_string(index=False))
print("\nSaved:", OUT334 / "report.json")
print("Test accessed: NO")

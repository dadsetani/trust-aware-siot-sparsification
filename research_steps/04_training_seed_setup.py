
# ============================================================
# E3.31 — SCENARIO B / SEED 104 / MODEL SEEDS 44,45,46
# ============================================================
# GNN scorer vs Trust-only scorer
# Fixed protocol: 60 epochs, AdamW, 50% edge budget
# No controller; GPU coverage selector
# Validation only t=8,9; TEST remains sealed.
#
# Reuses verified E3.24 forward and validation.
# Saves each model-seed experiment separately.
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

required331 = [
    "stable", "data104", "U", "V", "K",
    "forward324", "validate324",
    "coverage_topk_gpu"
]

missing331 = [
    x for x in required331 if x not in globals()
]
if missing331:
    raise RuntimeError(f"Missing objects: {missing331}")

ROOT331 = Path(
    "/kaggle/working/phase2_adaptive_siot/"
    "E3_tags_v3/scenarios/B_owner_correlated/"
    "seed_104/tags_v3"
)

SEEDS331 = [44, 45, 46]
EPOCHS331 = 60
TRAIN331 = [5, 6, 7]
VALID331 = [8, 9]

assert U.is_cuda and V.is_cuda
assert K == 73058
assert sorted(data104.keys()) == TRAIN331 + VALID331

assert Path(
    "/kaggle/working/phase2_adaptive_siot/"
    "E3_tags_v3/scenarios/B_owner_correlated/"
    "seed_104/e3_30_model_inputs/report.json"
).is_file()

# Stop before training if any destination already exists.
destinations331 = {
    seed: ROOT331 / f"e3_31_model_seed_{seed}"
    for seed in SEEDS331
}

existing331 = [
    str(path)
    for path in destinations331.values()
    if path.exists()
]

if existing331:
    raise FileExistsError(
        "Refusing to overwrite prior experiment(s): "
        + str(existing331)
    )

def reset_rng331(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def active_parameters331(model):
    return [
        p for name, p in model.named_parameters()
        if not name.startswith("controller.")
    ]

def train_one331(net, use_gnn, seed, model_name):

    reset_rng331(seed)

    params = active_parameters331(net)

    optimizer = torch.optim.AdamW(
        params, lr=1e-3, weight_decay=1e-4
    )

    history = []
    best_ap = -float("inf")
    best_epoch = None
    best_state = None

    print(
        f"\nMODEL SEED {seed} | {model_name}",
        flush=True
    )

    for epoch in range(1, EPOCHS331 + 1):

        net.train()
        losses = []

        for t in TRAIN331:
            batch = data104[t]

            optimizer.zero_grad(set_to_none=True)

            logits, hard, coverage = forward324(
                net, batch, use_gnn, True
            )

            assert int(hard.sum().item()) == K

            loss = F.binary_cross_entropy_with_logits(
                logits, batch["y"]
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Nonfinite loss: {model_name}, "
                    f"seed={seed}, epoch={epoch}, t={t}"
                )

            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(
                params, max_norm=1.0
            )

            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite gradient")

            optimizer.step()

            losses.append((
                float(loss.item()),
                int(batch["count"])
            ))

        train_bce = (
            sum(loss * count for loss, count in losses)
            / sum(count for _, count in losses)
        )

        # Explicitly use Scenario Seed 104.
        val = validate324(net, use_gnn, data104)

        record = {
            "epoch": epoch,
            "train_BCE": train_bce,
            **val
        }
        history.append(record)

        if val["AP"] > best_ap + 1e-5:
            best_ap = val["AP"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                net.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:02d} | "
                f"AP={val['AP']:.6f} | "
                f"AUC={val['AUC']:.6f} | "
                f"BCE={val['BCE']:.6f} | "
                f"churn={val['churn']:.4f}",
                flush=True
            )

    final = history[-1]

    result = {
        "model": model_name,
        "simulation_seed": 104,
        "model_seed": seed,
        "best_epoch": best_epoch,
        "best_AP": float(best_ap),
        "epoch60_AP": final["AP"],
        "epoch60_AUC": final["AUC"],
        "epoch60_BCE": final["BCE"],
        "epoch60_churn": final["churn"],
        "coverage_t8": final["coverage_t8"],
        "coverage_t9": final["coverage_t9"]
    }

    checkpoint = {
        "best_model_state_dict": best_state,
        "epoch60_model_state_dict":
            copy.deepcopy(net.state_dict()),
        "simulation_seed": 104,
        "model_seed": seed,
        "method": model_name,
        "epochs": EPOCHS331
    }

    return result, history, checkpoint

print("=" * 76)
print("E3.31 — THREE MATCHED SCORER COMPARISONS")
print("=" * 76)
print("Scenario seed: 104")
print("Model seeds:", SEEDS331)
print("Device:", torch.cuda.get_device_name(0))
print("Training:", TRAIN331)
print("Validation:", VALID331)
print("Epochs:", EPOCHS331)
print("Exact edge budget:", K)
print("Controller: DISABLED")
print("Test accessed: NO")

all_results331 = []
all_deltas331 = []

for seed in SEEDS331:

    # Fresh and matched shared initialization.
    reset_rng331(seed)

    b = data104[5]

    base = type(stable)(
        node_dim=b["xn"].shape[1],
        edge_dim=b["xe"].shape[1],
        controller_dim=b["xc"].shape[1],
        hidden=32
    ).to(U.device)

    gnn = copy.deepcopy(base)
    trust = copy.deepcopy(base)

    trust.scorer = nn.Sequential(
        nn.Linear(17, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(U.device)

    keys = [
        name for name in base.state_dict()
        if name.startswith((
            "encoder.",
            "trust_head.",
            "graph_head."
        ))
    ]

    assert keys
    assert all(
        torch.equal(
            gnn.state_dict()[name],
            trust.state_dict()[name]
        )
        for name in keys
    )

    experiment_results = []
    experiment_history = {}
    experiment_checkpoints = {}

    for model_name, model, use_gnn in [
        ("GNN_scorer", gnn, True),
        ("Trust_only_scorer", trust, False)
    ]:

        result, history, checkpoint = train_one331(
            model, use_gnn, seed, model_name
        )

        experiment_results.append(result)
        experiment_history[model_name] = history
        experiment_checkpoints[model_name] = checkpoint

        del result, history, checkpoint

    # Fixed-epoch paired comparison.
    gnn_result = experiment_results[0]
    trust_result = experiment_results[1]

    delta = {
        metric: (
            gnn_result[f"epoch60_{metric}"]
            - trust_result[f"epoch60_{metric}"]
        )
        for metric in ["AP", "AUC", "BCE", "churn"]
    }

    # Save this seed immediately, before starting the next.
    directory = destinations331[seed]
    directory.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(experiment_results).to_csv(
        directory / "comparison.csv",
        index=False
    )

    for name, history in experiment_history.items():
        pd.DataFrame(history).to_csv(
            directory / f"{name}_history.csv",
            index=False
        )

        torch.save(
            experiment_checkpoints[name],
            directory / f"{name}_checkpoint.pt"
        )

    report = {
        "experiment": "E3.31",
        "scenario": "B_owner_correlated",
        "simulation_seed": 104,
        "model_seed": seed,
        "training_epochs": EPOCHS331,
        "checkpoint_for_main_comparison": "epoch60",
        "train_times": TRAIN331,
        "validation_times": VALID331,
        "matched_shared_initialization": True,
        "fresh_initialization": True,
        "controller_enabled": False,
        "exact_edge_budget": K,
        "results": experiment_results,
        "gnn_minus_trust_only": delta,
        "test_accessed": False,
        "hidden_state_accessed": False
    }

    with open(
        directory / "report.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    all_results331.extend(experiment_results)
    all_deltas331.append({
        "scenario_seed": 104,
        "model_seed": seed,
        **{f"delta_{k}": v for k, v in delta.items()}
    })

    print("\n" + "=" * 60)
    print(f"SEED {seed} — COMPLETED AND SAVED")
    print("=" * 60)

    for metric, value in delta.items():
        print(f"delta_{metric}: {value:+.6f}")

    print("Saved:", directory / "report.json")

    # Release GPU memory from completed models.
    del base, gnn, trust
    del experiment_checkpoints
    torch.cuda.empty_cache()

print("\n" + "=" * 76)
print("E3.31 COMPLETED — SCENARIO SEED 104")
print("=" * 76)

comparison331 = pd.DataFrame(all_results331)
deltas331 = pd.DataFrame(all_deltas331)

print("\nSIX TRAINED MODELS")
print(
    comparison331[[
        "model_seed",
        "model",
        "epoch60_AP",
        "epoch60_AUC",
        "epoch60_BCE",
        "epoch60_churn"
    ]].round(6).to_string(index=False)
)

print("\nTHREE PAIRED DIFFERENCES")
print(
    deltas331.round(6).to_string(index=False)
)

print("\nMEAN DIFFERENCES")
for metric in ["AP", "AUC", "BCE", "churn"]:
    print(
        f"mean_delta_{metric}: "
        f"{deltas331[f'delta_{metric}'].mean():+.6f}"
    )

print("\nTest accessed: NO")
print("Hidden simulator state accessed: NO")
print("Previous artifacts overwritten: NO")
print("All three model seeds saved independently.")

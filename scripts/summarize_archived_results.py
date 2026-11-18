#!/usr/bin/env python3
"""Rebuild descriptive E3.36 summary using CSV only; never invoke models or test inference."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "heldout_test" / "test_metrics_all_18.csv"
METRICS = ("test_AP", "test_AUC", "test_BCE", "test_churn")
METHODS = ("GNN_scorer", "Trust_only_scorer")


def summarize():
    with DATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 18:
        raise ValueError(f"Expected 18 rows; found {len(rows)}")
    pairs = {}
    for row in rows:
        key = (int(row["model_seed"]), float(row["budget_fraction"]))
        method = row["model"]
        if method not in METHODS or method in pairs.setdefault(key, {}):
            raise ValueError(f"Unexpected or duplicate model: {key} {method}")
        pairs[key][method] = row
    if len(pairs) != 9 or any(set(p) != set(METHODS) for p in pairs.values()):
        raise ValueError("Expected 9 complete paired comparisons")
    result = {"provenance": "archived E3.36 CSV, descriptive only; no retraining or test inference",
              "evaluation_rows": len(rows), "paired_comparisons": len(pairs),
              "means": {}, "by_budget": {}, "pairwise_counts": {}}
    for metric in METRICS:
        gn = statistics.mean(float(p[METHODS[0]][metric]) for p in pairs.values())
        tr = statistics.mean(float(p[METHODS[1]][metric]) for p in pairs.values())
        result["means"][metric] = {"GNN_scorer": gn, "Trust_only_scorer": tr,
                                    "delta_GNN_minus_Trust": gn-tr}
    for budget in sorted({budget for _,budget in pairs}):
        matching = [pair for (seed,b), pair in sorted(pairs.items()) if b == budget]
        result["by_budget"][str(budget)] = {
            metric: {
                "GNN_scorer": statistics.mean(float(p[METHODS[0]][metric]) for p in matching),
                "Trust_only_scorer": statistics.mean(float(p[METHODS[1]][metric]) for p in matching),
                "delta_GNN_minus_Trust": statistics.mean(float(p[METHODS[0]][metric])-float(p[METHODS[1]][metric]) for p in matching)
            } for metric in METRICS
        }
    result["pairwise_counts"]["Trust_higher_AP"] = sum(
        float(p[METHODS[1]]["test_AP"]) > float(p[METHODS[0]]["test_AP"]) for p in pairs.values())
    result["pairwise_counts"]["GNN_lower_churn"] = sum(
        float(p[METHODS[0]]["test_churn"]) < float(p[METHODS[1]]["test_churn"]) for p in pairs.values())
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, sort_keys=True))

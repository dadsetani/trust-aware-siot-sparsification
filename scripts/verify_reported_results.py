#!/usr/bin/env python3
"""Verify archived E3.36 summary statistics; no model or test data access."""
from pathlib import Path
import csv, math, statistics, json
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "heldout_test" / "test_metrics_all_18.csv"
PROTOCOL = ROOT / "configs" / "FROZEN_PROTOCOL_BEFORE_TEST.json"
with PROTOCOL.open(encoding="utf-8") as f: p=json.load(f)
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
assert len(rows) == 18, f"Expected 18 model evaluations, got {len(rows)}"
assert p['simulation_seed'] == 104 and p['epochs'] == 60
assert p['test_times'] == [10,11] and p['optimizer_steps'] == 0
assert p['model_seeds'] == [44,45,46] and p['budgets'] == [.25,.5,.75]
assert all(int(r['simulation_seed'])==104 for r in rows)
by_key={(int(r['model_seed']), float(r['budget_fraction']),r['model']):r for r in rows}
assert len(by_key)==18
names=['GNN_scorer','Trust_only_scorer']
for s in p['model_seeds']:
 for b in p['budgets']:
  assert all((s,b,n) in by_key for n in names)
  for n in names:
   r=by_key[(s,b,n)]
   assert int(r['retained_edges']) == round(146117*b)
metrics=['test_AP','test_AUC','test_BCE','test_churn']
for met in metrics:
 gm=statistics.mean(float(r[met]) for r in rows if r['model']==names[0])
 tm=statistics.mean(float(r[met]) for r in rows if r['model']==names[1])
 print(f'{met:12} GNN={gm:.6f}  Trust={tm:.6f}  Delta={gm-tm:+.6f}')
expected={'test_AP':(.463257,.463573),'test_AUC':(.648486,.648642), 'test_BCE':(.602769,.602544),'test_churn':(.058610,.077920)}
for met,(eg,et) in expected.items():
 for name,target in zip(names,[eg,et]):
  actual=statistics.mean(float(r[met]) for r in rows if r['model']==name)
  assert abs(actual-target)<1e-6,(met,name,actual,target)
ap_favors_trust=0; churn_favors_gnn=0
for s in p['model_seeds']:
 for b in p['budgets']:
  gn=by_key[(s,b,names[0])];tr=by_key[(s,b,names[1])]
  ap_favors_trust+=float(tr['test_AP'])>float(gn['test_AP'])
  churn_favors_gnn+=float(gn['test_churn'])<float(tr['test_churn'])
assert ap_favors_trust==8 and churn_favors_gnn==7
print(f'PASS: 18 evaluations, 9 paired comparisons; trust AP wins {ap_favors_trust}/9, GNN lower churn {churn_favors_gnn}/9')

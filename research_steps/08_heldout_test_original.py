# E3.36 — ONE-TIME FINAL HELD-OUT TEST (NO TRAINING)
# Append AFTER executing the existing E3.RECOVER cell (159).
# Fixed protocol: scenario 104; seeds 44/45/46; budgets .25/.50/.75;
# methods GNN_scorer and Trust_only_scorer; EPOCH 60 weights only.
# Test times 10,11; chronological history updated before each time step.
# Never chooses a winning configuration from test results.
from pathlib import Path
from datetime import datetime, timezone
import json, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

required_336 = ('SOURCE','ROOT','U','V','N','M','K','stable','data104',
                'edge_scaler104','node_scaler104','type_onehot','coverage_topk_gpu','validate324')
missing_336 = [name for name in required_336 if name not in globals()]
assert not missing_336, f'Run recovery and E3.34b setup first: {missing_336}'
assert SOURCE == Path('/kaggle/working/phase2_adaptive_siot/E3_tags_v3/scenarios/B_owner_correlated/seed_104')
assert U.is_cuda and V.is_cuda and (N,M,K)==(16216,146117,73058)
assert sorted(data104)==[5,6,7,8,9], 'Unexpected training/validation restoration'

METHODS_336 = [('GNN_scorer',True), ('Trust_only_scorer',False)]
SEEDS_336 = (44,45,46)
BUDGETS_336 = (.25,.50,.75)
TIMES_336 = (10,11)

def checkpoint_336(seed, frac, method):
    if frac == .50:
        return SOURCE/'tags_v3'/f'e3_31_model_seed_{seed}'/f'{method}_checkpoint.pt'
    if seed == 44:
        return SOURCE/'budget_study'/'e3_34b_recovered_seed44_25_75'/f'{method}_{int(frac*100)}pct_seed{seed}_checkpoint.pt'
    return SOURCE/'budget_study'/'e3_35_replicate_seed45_46'/f'{method}_{int(frac*100)}pct_seed{seed}_checkpoint.pt'

# Preflight only: verify existing inputs/checkpoints. t10/t11 causal features will be computed in memory.
cp_paths_336 = [checkpoint_336(seed,frac,method)
                for seed in SEEDS_336 for frac in BUDGETS_336 for method,_ in METHODS_336]
required_files_336 = [SOURCE/'observed/train.csv.gz', SOURCE/'observed/validation.csv.gz',
                      SOURCE/'observed/test.csv.gz',
                      SOURCE/'causal_features/edge_index_mapping.csv.gz',
                      SOURCE/'causal_features/snapshot_t09.npz', *cp_paths_336]
missing_files_336 = [str(p) for p in required_files_336 if not p.is_file()]
assert not missing_files_336, 'Missing files; TEST has not been read:\n'+'\n'.join(missing_files_336)
for cp_336 in cp_paths_336:
    obj_336 = torch.load(cp_336, map_location='cpu',weights_only=True)
    assert 'epoch60_model_state_dict' in obj_336, f'Wrong checkpoint format: {cp_336}'
    assert isinstance(obj_336['epoch60_model_state_dict'],dict)
    del obj_336

# Existing validation parity, before reading test labels.
ref_336 = json.loads((SOURCE/'tags_v3/e3_31_model_seed_44/report.json').read_text())
model_336 = type(stable)(20,17,4,hidden=32).to(U.device)
model_336.load_state_dict(torch.load(checkpoint_336(44,.50,'GNN_scorer'),
                                    map_location='cpu',weights_only=True)['epoch60_model_state_dict'], strict=True)
parity_336 = validate324(model_336,True,data104)
reference_336 = next(x for x in ref_336['results'] if x['model']=='GNN_scorer')
assert abs(parity_336['AP']-float(reference_336['epoch60_AP']))<2e-6, 'Validation parity failed; DO NOT access test'
del model_336

OUT_336 = SOURCE/'final_test'/'e3_36_frozen_epoch60_3seeds_3budgets'
assert not OUT_336.exists(), f'Final test already exists. Do not rerun or overwrite: {OUT_336}'
# Frozen choices declared and persisted BEFORE reading Test data.
OUT_336.mkdir(parents=True,exist_ok=False)
manifest_336 = dict(experiment='E3.36', scenario='B_owner_correlated', simulation_seed=104,
    model_seeds=list(SEEDS_336), budgets=list(BUDGETS_336),
    methods=[x[0] for x in METHODS_336], epochs=60, checkpoint='epoch60_model_state_dict',
    train_times=[5,6,7], validation_times=[8,9], test_times=list(TIMES_336),
    metrics=['AP','AUC','BCE','churn'],
    aggregation='pool predictions across t10 and t11; paired GNN-minus-Trust by seed and budget',
    historical_policy='Strict E3.29 accumulator reconstruction in memory; at t11, t10 outcomes are history; no current/future outcomes used in features',
    model_selection='NONE on test', optimizer_steps=0,
    checkpoint_paths=[str(x) for x in cp_paths_336],
    frozen_at_utc=datetime.now(timezone.utc).isoformat())
(OUT_336/'FROZEN_PROTOCOL_BEFORE_TEST.json').write_text(json.dumps(manifest_336,indent=2),encoding='utf-8')
print('Protocol frozen, validation parity passed. Reading held-out test exactly once.',flush=True)

# The same strictly historical feature recipe as the original E3.RECOVER cell.
train_336 = pd.read_csv(SOURCE/'observed/train.csv.gz')
val_336 = pd.read_csv(SOURCE/'observed/validation.csv.gz')
test_336 = pd.read_csv(SOURCE/'observed/test.csv.gz')
assert train_336.time.between(0,7).all()
assert val_336.time.between(8,9).all()
assert test_336.time.isin(TIMES_336).all() and set(test_336.time)==set(TIMES_336)
assert all(x.success.isin([0,1]).all() for x in (train_336,val_336,test_336))
events_336 = pd.concat([train_336,val_336,test_336],ignore_index=True)
top_336 = pd.read_csv(SOURCE/'causal_features/edge_index_mapping.csv.gz')
assert len(top_336)==M
u_336=U.detach().cpu().numpy(); v_336=V.detach().cpu().numpy()
assert np.array_equal(top_336.u.to_numpy(np.int64)-1,u_336)
assert np.array_equal(top_336.v.to_numpy(np.int64)-1,v_336)
keys_336=u_336*N+v_336
src_336=events_336.source.to_numpy(np.int64)-1
dst_336=events_336.target.to_numpy(np.int64)-1
assert np.all(src_336<dst_336)
event_keys_336=src_336*N+dst_336
edge_ids_336=np.searchsorted(keys_336,event_keys_336)
assert (edge_ids_336<M).all() and np.array_equal(keys_336[edge_ids_336],event_keys_336)
t_336=events_336.time.to_numpy(np.int16)
success_336=events_336.success.to_numpy(np.float64)
relations_336=top_336[['OOR','POR','C_LOR','SOR','SOR2']].to_numpy(np.float32)
static_degree_336=np.bincount(np.concatenate([u_336,v_336]),minlength=N)
degree_336=np.log1p(static_degree_336).astype(np.float32)

# Rebuild causal history in memory using the E3.29 accumulator recipe.
# No missing t10/t11 snapshots are needed or written to existing folders.
# Crucially: history at time t includes only observations with time < t.
def historical_features_336(current_time):
    edge_s = np.zeros(M, dtype=np.float32)
    edge_f = np.zeros(M, dtype=np.float32)
    decay_s = np.zeros(M, dtype=np.float32)
    decay_f = np.zeros(M, dtype=np.float32)
    node_s = np.zeros(N, dtype=np.float32)
    node_f = np.zeros(N, dtype=np.float32)
    last_seen = np.full(M, -1, dtype=np.int16)
    for tt in range(current_time + 1):
        if tt > 0:
            decay_s *= np.float32(0.80)
            decay_f *= np.float32(0.80)
        if tt == current_time:
            break  # do not update with current-time labels
        mask = t_336 == tt
        ids = edge_ids_336[mask]
        successes = success_336[mask].astype(np.float32)
        failures = np.float32(1.0) - successes
        np.add.at(edge_s, ids, successes)
        np.add.at(edge_f, ids, failures)
        np.add.at(decay_s, ids, successes)
        np.add.at(decay_f, ids, failures)
        np.add.at(node_s, u_336[ids], successes)
        np.add.at(node_s, v_336[ids], successes)
        np.add.at(node_f, u_336[ids], failures)
        np.add.at(node_f, v_336[ids], failures)
        last_seen[ids] = tt
    hist_count = edge_s + edge_f
    decay_count = decay_s + decay_f
    hist_rate = (edge_s + np.float32(1.0)) / (hist_count + np.float32(2.0))
    decay_rate = (decay_s + np.float32(1.0)) / (decay_count + np.float32(2.0))
    node_rate = (node_s + np.float32(1.0)) / (node_s + node_f + np.float32(2.0))
    node_conf = (node_s + node_f) / (node_s + node_f + np.float32(5.0))
    never_seen = last_seen == -1
    age = np.where(never_seen, 12, current_time - last_seen).astype(np.float32)
    out = np.column_stack((hist_count, hist_rate, decay_count, decay_rate,
        hist_count / (hist_count + np.float32(5.0)), age / np.float32(12.0),
        never_seen.astype(np.float32), node_rate[u_336], node_rate[v_336],
        node_conf[u_336], node_conf[v_336], decay_rate - hist_rate)).astype(np.float32)
    assert out.shape == (M, 12) and np.isfinite(out).all()
    assert int(out[:,0].sum()) == int(np.count_nonzero(t_336 < current_time))
    return out

# Strict parity check against the genuine previously-saved E3.29 t=9 snapshot.
with np.load(SOURCE/'causal_features/snapshot_t09.npz') as old_336:
    expected_336 = old_336['features']
    assert int(old_336['time']) == 9
trainval_336 = t_336 < 10
# At t=9, test outcomes must have zero effect. Compare all features.
reconstructed9_336 = historical_features_336(9)
assert np.allclose(reconstructed9_336, expected_336, rtol=1e-6, atol=1e-6), \
    'E3.29 historical-feature parity failed; STOP'
del reconstructed9_336, expected_336
print('E3.29 feature parity t=9: PASSED', flush=True)

data_test_336={}
for current_336 in TIMES_336:
    past_336=t_336<current_336
    pu_336=src_336[past_336]; pv_336=dst_336[past_336]; ps_336=success_336[past_336]
    node_count_336=np.bincount(np.concatenate([pu_336,pv_336]),minlength=N)
    node_success_336=(np.bincount(pu_336,weights=ps_336,minlength=N)
                      +np.bincount(pv_336,weights=ps_336,minlength=N))
    node_raw_336=np.column_stack((degree_336,np.log1p(node_count_336),
        (node_success_336+1)/(node_count_336+2),node_count_336/(node_count_336+5))).astype(np.float32)
    hist_336=historical_features_336(current_336)
    assert hist_336.shape==(M,12)
    assert abs(float(hist_336[:,0].sum())-int(past_336.sum()))<1.0, f'Historical snapshot mismatch at t={current_336}'
    ru_336=node_raw_336[u_336,2]; rv_336=node_raw_336[v_336,2]
    cu_336=node_raw_336[u_336,3]; cv_336=node_raw_336[v_336,3]
    ends_336=np.column_stack(((ru_336+rv_336)/2,abs(ru_336-rv_336),
                              (cu_336+cv_336)/2,abs(cu_336-cv_336))).astype(np.float32)
    edge_raw_336=np.column_stack((np.log1p(hist_336[:,0]),hist_336[:,1],
        np.log1p(hist_336[:,2]),hist_336[:,3:7],ends_336,hist_336[:,11],relations_336)).astype(np.float32)
    assert edge_raw_336.shape==(M,17)
    curr_336=t_336==current_336
    targets_336=edge_ids_336[curr_336]
    y_336=(1-success_336[curr_336]).astype(np.float32)
    assert len(targets_336)==len(y_336)>0
    xe_336=edge_scaler104.transform(edge_raw_336).astype(np.float32)
    xc_336=node_scaler104.transform(node_raw_336).astype(np.float32)
    xn_336=np.concatenate((type_onehot.astype(np.float32),xc_336),axis=1)
    data_test_336[current_336]=dict(time=current_336,
        xn=torch.as_tensor(xn_336,device=U.device),
        xe=torch.as_tensor(xe_336,device=U.device),
        xc=torch.as_tensor(xc_336,device=U.device),
        target=torch.as_tensor(targets_336,device=U.device),
        y=torch.as_tensor(y_336,device=U.device),count=int(len(y_336)))
    print(f't={current_336} historical features checked; n={len(y_336)}',flush=True)

@torch.no_grad()
def forward_test_336(net, batch, use_gnn, k):
    # Exact selector and inference equations from E3.34b (no training).
    xn,xe,ids=batch['xn'],batch['xe'],batch['target']
    if use_gnn:
        h=net.encoder(xn,U,V,xn.new_ones(M))
        inp=torch.cat((h[U]+h[V],(h[U]-h[V]).abs(),xe),dim=1)
        scores=net.scorer(inp).squeeze(-1)
    else:
        scores=net.scorer(xe).squeeze(-1)
    hard,coverage=coverage_topk_gpu(scores,k)
    z=net.encoder(xn,U,V,hard)
    a,b=U[ids],V[ids]
    pairs=torch.cat((z[a]+z[b],(z[a]-z[b]).abs(),z[a]*z[b]),dim=1)
    logits=net.trust_head(xe[ids]).squeeze(-1)+net.graph_head(pairs).squeeze(-1)
    return logits,hard,int(coverage)

records_336=[]
for seed_336 in SEEDS_336:
    for frac_336 in BUDGETS_336:
        k_336=round(M*frac_336)
        for method_336,use_gnn_336 in METHODS_336:
            net_336=type(stable)(20,17,4,hidden=32).to(U.device)
            if not use_gnn_336:
                net_336.scorer=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,1)).to(U.device)
            ckpt_336=torch.load(checkpoint_336(seed_336,frac_336,method_336),map_location='cpu',weights_only=True)
            net_336.load_state_dict(ckpt_336['epoch60_model_state_dict'],strict=True)
            net_336.eval()
            ys_336=[]; probs_336=[]; masks_336=[]; covers_336=[]
            with torch.inference_mode():
                for tstep_336 in TIMES_336:
                    logit_336,mask_336,coverage_336=forward_test_336(net_336,data_test_336[tstep_336],use_gnn_336,k_336)
                    ys_336.append(data_test_336[tstep_336]['y'].cpu().numpy())
                    probs_336.append(torch.sigmoid(logit_336).cpu().numpy())
                    masks_336.append(mask_336.cpu().numpy().astype(bool))
                    covers_336.append(int(coverage_336))
            ys_336=np.concatenate(ys_336)
            probs_336=np.clip(np.concatenate(probs_336),1e-7,1-1e-7)
            assert all(int(x.sum())==k_336 for x in masks_336)
            assert len(np.unique(ys_336))==2, 'AUC undefined for a single-class test set'
            result_336=dict(simulation_seed=104,model_seed=seed_336,
                budget_fraction=frac_336,retained_edges=k_336,model=method_336,
                test_AP=float(average_precision_score(ys_336,probs_336)),
                test_AUC=float(roc_auc_score(ys_336,probs_336)),
                test_BCE=float(log_loss(ys_336,probs_336,labels=[0,1])),
                test_churn=float(np.count_nonzero(masks_336[0]!=masks_336[1])/(2*k_336)),
                coverage_t10=covers_336[0],coverage_t11=covers_336[1],
                test_n=int(len(ys_336)),test_failure_rate=float(ys_336.mean()))
            records_336.append(result_336)
            print(f'SEED {seed_336} | {frac_336:.0%} | {method_336}: AP={result_336["test_AP"]:.6f} AUC={result_336["test_AUC"]:.6f} BCE={result_336["test_BCE"]:.6f} churn={result_336["test_churn"]:.6f}',flush=True)
            del net_336,ckpt_336
        torch.cuda.empty_cache()

results_336=pd.DataFrame(records_336)
assert len(results_336)==18
results_336.to_csv(OUT_336/'test_metrics_all_18.csv',index=False)
paired_336=results_336.pivot(index=['model_seed','budget_fraction'],columns='model',values=['test_AP','test_AUC','test_BCE','test_churn'])
for metric_336 in ('test_AP','test_AUC','test_BCE','test_churn'):
    paired_336[(metric_336,'GNN_minus_Trust')]=paired_336[(metric_336,'GNN_scorer')]-paired_336[(metric_336,'Trust_only_scorer')]
paired_336.to_csv(OUT_336/'test_paired_differences.csv')
summary_336=results_336.groupby(['budget_fraction','model'])[['test_AP','test_AUC','test_BCE','test_churn']].agg(['mean','std'])
summary_336.to_csv(OUT_336/'test_summary_mean_std.csv')
(OUT_336/'SUCCESS.json').write_text(json.dumps(dict(finished_at_utc=datetime.now(timezone.utc).isoformat(),
    test_rows=18,train_steps=0,original_checkpoints_unchanged=True),indent=2),encoding='utf-8')
print('\nE3.36 COMPLETED — frozen held-out test, 18 evaluations, 0 training steps.')
print(summary_336.round(6).to_string())
print('Saved:',OUT_336)

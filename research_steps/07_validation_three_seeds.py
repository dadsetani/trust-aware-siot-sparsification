# E3.35 — replicate budget sensitivity across model seeds 45 and 46
# Run in Kaggle AFTER E3.RECOVERY and E3.34b.
# Uses ORIGINAL restored forward_b, validate_b, reset_rng_b functions.
# Train/Validation only; no test access and no previous artifacts overwritten.
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

required_335 = ['data104','stable','U','V','K','SOURCE','forward_b','validate_b','reset_rng_b']
missing_335 = [v for v in required_335 if v not in globals()]
assert not missing_335, f'Missing restored objects: {missing_335}'
assert sorted(data104) == [5,6,7,8,9]
assert U.is_cuda and V.is_cuda and U.numel() == 146117 and K == 73058
assert data104[5]['xn'].shape == (16216,20)
OUT335 = SOURCE/'budget_study'/'e3_35_replicate_seed45_46'
OUT335.mkdir(parents=True,exist_ok=True)
SEEDS335 = (45,46)
BUDGETS335 = (0.25,0.75)
METHODS335 = (('GNN_scorer',True),('Trust_only_scorer',False))

# Confirm the restored scorer/validation implementation against existing
# epoch-60 checkpoints before any new training.
print('E3.35 — REPLICATION PRECHECK',flush=True)
for seed335 in SEEDS335:
    report335 = json.loads((SOURCE/'tags_v3'/f'e3_31_model_seed_{seed335}'/'report.json').read_text())
    for method335,use_gnn335 in METHODS335:
        obj335=torch.load(SOURCE/'tags_v3'/f'e3_31_model_seed_{seed335}'/f'{method335}_checkpoint.pt',
                          map_location='cpu', weights_only=True)
        net335=type(stable)(node_dim=20,edge_dim=17,controller_dim=4,hidden=32).to(U.device)
        if not use_gnn335:
            net335.scorer=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,1)).to(U.device)
        net335.load_state_dict(obj335['epoch60_model_state_dict'],strict=True)
        m335=validate_b(net335,use_gnn335,K)
        ref335=next(x for x in report335['results'] if x['model']==method335)
        assert abs(m335['AP']-float(ref335['epoch60_AP']))<2e-6,(seed335,method335,m335)
        print(f'PARITY seed={seed335} {method335}: AP={m335["AP"]:.6f}',flush=True)
        del net335
    torch.cuda.empty_cache()

# Each completed model saved immediately. A resumed run skips complete
# checkpoint+history pairs; incomplete pairs stop rather than overwrite.
for seed335 in SEEDS335:
    for frac335 in BUDGETS335:
        k335=round(U.numel()*frac335)
        assert k335>=16216, 'Insufficient budget for guaranteed greedy coverage'
        reset_rng_b(seed335)
        base335=type(stable)(node_dim=20,edge_dim=17,controller_dim=4,hidden=32).to(U.device)
        gnn335=copy.deepcopy(base335)
        trust335=copy.deepcopy(base335)
        trust335.scorer=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,1)).to(U.device)
        for pname335 in base335.state_dict():
            if pname335.startswith(('encoder.','trust_head.','graph_head.')):
                assert torch.equal(gnn335.state_dict()[pname335],trust335.state_dict()[pname335])
        del base335
        for method335,use_gnn335,net335 in (
            ('GNN_scorer',True,gnn335),('Trust_only_scorer',False,trust335)
        ):
            tag335=f'{method335}_{int(frac335*100)}pct_seed{seed335}'
            statepath335=OUT335/f'{tag335}_checkpoint.pt'
            histpath335=OUT335/f'{tag335}_history.csv'
            rowpath335=OUT335/f'{tag335}_result.json'
            trio335=[statepath335,histpath335,rowpath335]
            if all(p.exists() for p in trio335):
                print('SKIPPED completed:',tag335,flush=True)
                continue
            if any(p.exists() for p in trio335):
                raise RuntimeError(f'Partial output for {tag335}; inspect before continuing')
            reset_rng_b(seed335)
            params335=[p for n,p in net335.named_parameters() if not n.startswith('controller.')]
            opt335=torch.optim.AdamW(params335,lr=1e-3,weight_decay=1e-4)
            hist335=[]; best335=-float('inf'); bestepoch335=None; beststate335=None
            print(f'\nSTART seed={seed335} budget={frac335:.0%} model={method335}',flush=True)
            for ep335 in range(1,61):
                net335.train(); weighted335=0.0; total335=0
                for t335 in (5,6,7):
                    batch335=data104[t335]
                    opt335.zero_grad(set_to_none=True)
                    logits335,_,_=forward_b(net335,batch335,use_gnn335,k335,True)
                    loss335=F.binary_cross_entropy_with_logits(logits335,batch335['y'])
                    assert bool(torch.isfinite(loss335)),(seed335,frac335,method335,ep335,'loss')
                    loss335.backward()
                    norm335=torch.nn.utils.clip_grad_norm_(params335,1.0)
                    assert bool(torch.isfinite(norm335)),(seed335,frac335,method335,ep335,'grad')
                    opt335.step()
                    n335=int(batch335['count']);total335+=n335;weighted335+=float(loss335)*n335
                metrics335=validate_b(net335,use_gnn335,k335)
                hist335.append(dict(epoch=ep335,train_BCE=weighted335/total335,**metrics335))
                if metrics335['AP']>best335+1e-5:
                    best335=metrics335['AP'];bestepoch335=ep335
                    beststate335=copy.deepcopy(net335.state_dict())
                if ep335==1 or ep335%10==0:
                    print(f' epoch={ep335:02d} AP={metrics335["AP"]:.6f} '
                          f'AUC={metrics335["AUC"]:.6f} BCE={metrics335["BCE"]:.6f}',flush=True)
            result335=dict(simulation_seed=104,model_seed=seed335,budget_fraction=frac335,
                           retained_edges=k335,model=method335,best_epoch=bestepoch335,
                           best_AP=best335,epoch60_AP=metrics335['AP'],
                           epoch60_AUC=metrics335['AUC'],epoch60_BCE=metrics335['BCE'],
                           epoch60_churn=metrics335['churn'])
            # Save the model first, then history/metadata. A partial save is
            # deliberately detected on the next execution rather than overwritten.
            torch.save(dict(best_model_state_dict=beststate335,
                            epoch60_model_state_dict=copy.deepcopy(net335.state_dict()),
                            simulation_seed=104,model_seed=seed335,method=method335,
                            budget_fraction=frac335,epochs=60),statepath335)
            pd.DataFrame(hist335).to_csv(histpath335,index=False)
            rowpath335.write_text(json.dumps(result335,indent=2),encoding='utf-8')
            print('SAVED:',tag335,flush=True)
        del gnn335,trust335
        torch.cuda.empty_cache()

# Aggregate the three initializations without touching old files.
first335=SOURCE/'budget_study'/'e3_34b_recovered_seed44_25_75'/'completed_results.csv'
assert first335.is_file(), 'Missing seed44 budget results from E3.34b'
rows335=pd.read_csv(first335).to_dict('records')
for seed335 in SEEDS335:
    for frac335 in BUDGETS335:
        for method335,_ in METHODS335:
            path335=OUT335/f'{method335}_{int(frac335*100)}pct_seed{seed335}_result.json'
            rows335.append(json.loads(path335.read_text(encoding='utf-8')))
for seed335 in (44,45,46):
    report335=json.loads((SOURCE/'tags_v3'/f'e3_31_model_seed_{seed335}'/'report.json').read_text())
    for r335 in report335['results']:
        rows335.append(dict(simulation_seed=104,model_seed=seed335,budget_fraction=0.50,
                            retained_edges=K,model=r335['model'],
                            epoch60_AP=float(r335['epoch60_AP']),
                            epoch60_AUC=float(r335['epoch60_AUC']),
                            epoch60_BCE=float(r335['epoch60_BCE']),
                            epoch60_churn=float(r335['epoch60_churn'])))
all335=pd.DataFrame(rows335).drop_duplicates(subset=['simulation_seed','model_seed','budget_fraction','model'])
assert len(all335)==18,len(all335)
pairs335=all335.pivot(index=['model_seed','budget_fraction'],columns='model',values='epoch60_AP')
pairs335['GNN_minus_Trust_AP']=pairs335['GNN_scorer']-pairs335['Trust_only_scorer']
summary335=pairs335.groupby(level='budget_fraction')['GNN_minus_Trust_AP'].agg(['mean','std','min','max'])
all335.to_csv(OUT335/'all_three_seeds_metrics.csv',index=False)
pairs335.to_csv(OUT335/'paired_delta_ap.csv')
summary335.to_csv(OUT335/'budget_replication_summary.csv')
print('\nE3.35 COMPLETED — ALL 3 SEEDS, BUDGETS 25% / 50% / 75%')
print('\nPAIRED AP RESULTS')
print(pairs335.round(6).to_string())
print('\nMEAN DELTA AP BY BUDGET')
print(summary335.round(6).to_string())
print('TEST accessed: NO. Existing experiments overwritten: NO.')
print('Saved:',OUT335)
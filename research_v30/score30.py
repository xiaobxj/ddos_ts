from common30 import *
from contract30 import audit_policy

def main():
    check_frozen();check_phase('extension');assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');unique=csv('universal_model_predictions');feedback=feedback_from(unique);events,ledger,members=simulate(feedback)
    assert len(ledger)==272 and ledger.date.is_unique and ledger.cutoff.lt(ledger.date).all();audit=audit_policy(feedback,events,ledger,members)
    files=[]
    for name,frame in [('feedback_pool',feedback),('events',events),('published_ledger',ledger),('monitor_membership',members),('independent_trigger_checks',audit)]:p=OUT/f'{name}.csv';frame.to_csv(p,index=False);files.append(p)
    # The adaptive route is saved before aggregate metrics or counterfactual outcomes.
    save(OUT/'policy_manifest.json',dict(status='FROZEN',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},new_neural_fits=0,new_head_fits=0));files.append(OUT/'policy_manifest.json')
    candidate=route_predictions(unique,ledger);models=pd.concat([csv('baseline_model_predictions'),candidate],ignore_index=True);ensemble=ensemble_from(models);assert len(models)==21760 and len(ensemble)==8160
    pd.testing.assert_frame_equal(ensemble[ensemble.history.ne('error16')].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    budget=pd.read_csv(V29/'results/budgets.csv');budget=pd.concat([budget,pd.DataFrame([runtime_budget(events)])],ignore_index=True)
    for name,frame in [('model_predictions',models),('ensemble_predictions',ensemble),('budgets',budget)]:p=OUT/f'{name}.csv';frame.to_csv(p,index=False);files.append(p)
    optional=events[~events.mandatory];check_frozen();finish(run,files,optional_checks=17,triggered_refits=int(optional.refit.sum()),warmup_checks=int((~optional.ready).sum()),refit_dates=int(events.refit.sum()),new_neural_fits=0,new_head_fits=0,model_rows=len(models),ensemble_rows=len(ensemble),weeks=272)
    print(events[['cutoff','previous_model_cutoff','available_mature_weeks','older8_excess','recent8_excess','reason','selected_model_cutoff']].to_string(index=False),flush=True);print('Scoring PASS: causal feedback policy and all frozen baselines.',flush=True)
if __name__=='__main__':main()

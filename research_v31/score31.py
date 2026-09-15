from common31 import *
from contract31 import audit_policy
def main():
    check_frozen();assert not (OUT/'scoring_manifest.json').exists();run=manifest('scoring');unique=csv('universal_model_predictions');feedback=feedback_from(unique);events,ledger,members=simulate(feedback);audit=audit_policy(feedback,events,ledger,members);route=shadow_route(ledger)
    base=csv('baseline_model_predictions');half,blendroute=half_blend(base);files=[]
    for name,g in [('feedback_pool',feedback),('events',events),('paired_ledger',ledger),('trial_membership',members),('independent_trial_checks',audit),('shadow_routing',route),('blend_routing',blendroute)]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    save(OUT/'policy_manifest.json',dict(status='FROZEN',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));files.append(OUT/'policy_manifest.json')
    models=pd.concat([base,route_predictions(unique,route),half],ignore_index=True);ensemble=ensemble_from(models);assert len(models)==30464 and len(ensemble)==11424;pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),csv('baseline_ensemble_predictions'),check_dtype=False,atol=1e-14,rtol=0)
    for name,g in [('model_predictions',models),('ensemble_predictions',ensemble),('budgets',runtime_budgets(events))]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,promotions=int(events.promote.sum()),selected_model_cutoffs=int(events.selected_model_cutoff.nunique()),optional_checks=17,distinct_model_checks=int(events.ready.sum()),new_neural_fits=0,new_head_fits=0,new_inference_passes=0)
    print(events[['cutoff','incumbent_cutoff','evaluated_challenger_cutoff','paired_mature_weeks','challenger_minus_incumbent_brier','challenger_correct_gain','reason','selected_model_cutoff']].to_string(index=False),flush=True);print('Scoring PASS: paired trial and fixed half blend.',flush=True)
if __name__=='__main__':main()

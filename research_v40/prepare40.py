from common40 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);run=manifest('preparation');run['old_evidence']=old_evidence();save(OUT/'initial_freeze.json',run);files=[OUT/'initial_freeze.json']
    for src,dst in [('model_predictions.csv','baseline_model_predictions.csv'),('ensemble_predictions.csv','baseline_ensemble_predictions.csv'),('ensemble_metrics.csv','baseline_metrics.csv'),('routing.csv','quarterly_routing.csv'),('weekly_context.csv','weekly_context.csv'),('source_seed_effects.csv','source_seed_effects.csv'),('correction_heads.json','quarterly_heads.json'),('gate_decisions.csv','quarterly_gates.csv'),('split_membership.csv','quarterly_members.csv'),('schedule.csv','quarterly_schedule.csv'),('seed_routing.csv','quarterly_seed_routing.csv'),('weekly_policy_effects.csv','quarterly_weekly_effects.csv')]:
        p=OUT/dst;p.write_bytes((V39/'results'/src).read_bytes());files.append(p)
    route=csv('quarterly_routing').rename(columns={'head_cutoff':'quarter_cutoff'});route['head_cutoff']=route.date.map(monthly_for);p=OUT/'routing.csv';route.to_csv(p,index=False);files.append(p)
    assert set(route.head_cutoff).issubset(cfg()['decision_dates']) and route.head_cutoff.lt(route.date).all()
    assert read(V39/'results/verification.json')['status']=='PASS';finish(run,files,old_files=5554,new_fits=0,new_predictions=0);print('R40 source, protocol, inputs and5554old files frozen.',flush=True)
if __name__=='__main__':main()

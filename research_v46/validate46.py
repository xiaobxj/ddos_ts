from common46 import *
def main():
    check_frozen();check_phase('fitting');assert not (OUT/'validation_manifest.json').exists();run=manifest('validation');g,v=validate_records(csv('weekly_signal_bank'),csv('split_membership'),csv('schedule'),read(OUT/'correction_heads.json'));assert len(g)==736;files=[]
    for n,t in [('gate_decisions',g),('validation_predictions',v)]:p=OUT/f'{n}.csv';t.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,gate_cells=736,accepted_cells=int(g.accepted.sum()),refits_after_validation=0,future_scores_used=False);print(f'Validation gates frozen: {g.accepted.sum()}/736 accepted. No future scoring yet.',flush=True)
if __name__=='__main__':main()

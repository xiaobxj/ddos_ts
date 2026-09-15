from common45 import *
def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');heads=fit_records(csv('weekly_signal_bank'),csv('split_membership'),csv('schedule'));assert len(heads)==1104
    p=OUT/'correction_heads.json';save(p,heads);q=OUT/'correction_parameters.csv';pd.DataFrame(heads).to_csv(q,index=False);check_frozen();finish(run,[p,q],scalar_fits=sum(h['fit_eligible'] for h in heads),head_records=1104,validation_labels_used=False,new_neural_fits=0,new_feature_inference=0)
    print(f"Training-only offsets frozen: {sum(h['fit_eligible'] for h in heads)} scalar fits. Validation labels not used.",flush=True)
if __name__=='__main__':main()

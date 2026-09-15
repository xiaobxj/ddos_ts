"""Compare existing model states under the same evaluation-mode training loss.

This is a training-only interpretation of archived outputs; no new fit or
validation selection. It is not the dropout-averaged regularized objective.
"""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent/'results'


def main():
    audit=pd.read_csv(OUT/'training_loss_audit.csv')
    audit=audit[audit.state.eq('frozen_mse_model')][['cutoff','seed','scaled_huber_loss']]
    old=pd.read_csv(OUT/'mse_training_curves.csv');old=old[old.epoch.eq(20)][['cutoff','seed','final_model_train_auxiliary_mse']]
    new=pd.read_csv(OUT/'huber_training_curves.csv');new=new[new.epoch.eq(20)][['cutoff','seed','final_model_train_scaled_huber','final_model_train_auxiliary_mse']]
    table=audit.merge(old,on=['cutoff','seed'],validate='one_to_one').merge(new,on=['cutoff','seed'],validate='one_to_one',suffixes=('_mse','_huber'))
    table['mse_model_under_huber_joint']=table.scaled_huber_loss+.1*table.final_model_train_auxiliary_mse_mse
    table['huber_model_under_huber_joint']=table.final_model_train_scaled_huber+.1*table.final_model_train_auxiliary_mse_huber
    table['huber_model_improved_joint']=table.huber_model_under_huber_joint<table.mse_model_under_huber_joint
    summary=dict(comparisons=len(table),huber_models_with_lower_own_training_objective=int(table.huber_model_improved_joint.sum()),
        mean_mse_model_huber_joint=float(table.mse_model_under_huber_joint.mean()),mean_huber_model_huber_joint=float(table.huber_model_under_huber_joint.mean()),
        additional_fitting=False,training_only=True,evaluation_mode=True,
        interpretation='Same evaluation-mode data-fit objective, not proof of convergence or the dropout-averaged regularized training objective.')
    table.to_csv(OUT/'matched_training_objectives.csv',index=False)
    (OUT/'matched_training_objective_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

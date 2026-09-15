from common42 import *

def main():
    check_frozen();assert not (OUT/'routing_manifest.json').exists();run=manifest('routing');decisions=retention_decisions(csv('gate_decisions'),csv('schedule').cutoff.tolist());assert len(decisions)==1088
    p=OUT/'retention_decisions.csv';decisions.to_csv(p,index=False);check_frozen();finish(run,[p],decision_cells=len(decisions),fresh_cells=int(decisions.action.eq('fresh').sum()),carried_cells=int(decisions.action.eq('carry').sum()),fallback_cells=int(decisions.action.eq('fallback').sum()),new_fits=0,future_scores_used=False);print(f"Retention decisions frozen: {sum(decisions.action=='fresh')}fresh, {sum(decisions.action=='carry')}carry, {sum(decisions.action=='fallback')}fallback cells. No future scoring yet.",flush=True)
if __name__=='__main__':main()

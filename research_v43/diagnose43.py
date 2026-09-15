from common43 import *

def diagnose(subjects,bank,members,heads):
    hi={(h['cutoff'],h['method'],h['component'],h['seed']):h for h in heads}
    bi=bank.set_index(['method','seed','row_index']); state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state
    valids={d:g.row_index.tolist() for d,g in members[members.role.eq('validation')].groupby('cutoff')}
    seedrows=[];weekrows=[];cells=[]
    for r in subjects.itertuples(index=False):
        info=r._asdict(); local=[]
        if r.availability in ['in_life','expired']:
            ids=[i for i in valids.get(r.cutoff,[]) if state[i]==r.state]
            source_val=set(valids.get(r.source_cutoff,[]))
            for idx in ids:
                temp=[]
                for seed in cfg()['seeds']:
                    b=bi.loc[(r.method,seed,idx)];old=hi[(r.source_cutoff,r.method,r.state,seed)]['offset'];new=hi[(r.cutoff,r.method,r.state,seed)]['offset']
                    pold=b.probability if old==0 else float(1/(1+np.exp(-b.annual_logit-old)))
                    pnew=(b.probability if new==0 else float(1/(1+np.exp(-b.annual_logit-new)))) if r.current_fit_eligible else None
                    row=dict(subject_id=r.subject_id,cutoff=r.cutoff,method=r.method,state=r.state,source_cutoff=r.source_cutoff,seed=seed,row_index=int(idx),date=b.date,joint_completed=b.joint_completed,annual_encoder_cutoff=b.encoder_cutoff,actual_up=int(b.actual_up),annual_probability=float(b.probability),old_probability=pold,current_probability=pnew,old_offset=old,current_offset=new if r.current_fit_eligible else None,matured_after_source=b.joint_completed>r.source_cutoff,matured_since_previous_decision=b.joint_completed>r.previous_decision_cutoff,in_source_validation=idx in source_val)
                    temp.append(row);seedrows.append(row)
                row={k:v for k,v in temp[0].items() if k not in ['seed','old_offset','current_offset']}
                for name in ['annual_probability','old_probability','current_probability']:
                    row[name]=float(np.mean([a[name] for a in temp])) if temp[0][name] is not None else None
                y=row['actual_up'];p0=row['annual_probability'];p=row['old_probability'];q=row['current_probability']
                row.update(old_brier_difference=(p-y)**2-(p0-y)**2,old_correct_difference=int((p>.5)==y)-int((p0>.5)==y),current_brier_difference=(q-y)**2-(p0-y)**2 if q is not None else None,current_correct_difference=int((q>.5)==y)-int((p0>.5)==y) if q is not None else None)
                weekrows.append(row);local.append(row)
        for view in VIEWS:
            selected=[x for x in local if view=='full_validation' or (x['matured_after_source'] if view=='post_source_validation' else x['matured_since_previous_decision'])];n=len(selected)
            row=dict(**info,view=view,n=n,known_at_source_n=sum(not x['matured_after_source'] for x in selected),source_validation_overlap_n=sum(x['in_source_validation'] for x in selected),new_since_previous_n=sum(x['matured_since_previous_decision'] for x in selected))
            for role in ['old','current']:
                available=role=='old' or r.current_fit_eligible
                bd=float(np.mean([x[f'{role}_brier_difference'] for x in selected])) if n and available else None
                cd=int(sum(x[f'{role}_correct_difference'] for x in selected)) if available else None
                status='no_saved_record' if r.availability=='absent' else 'annual_mismatch' if r.availability=='annual_mismatch' else evidence(n,bd,cd,available)
                row.update({f'{role}_brier_difference':bd,f'{role}_correct_difference':cd,f'{role}_evidence':status})
            row['comparison']=comparison(row['old_evidence'],row['current_evidence']);cells.append(row)
    return pd.DataFrame(seedrows),pd.DataFrame(weekrows),pd.DataFrame(cells)

def main():
    check_frozen();assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis')
    seed,weeks,cells=diagnose(csv('subjects'),csv('weekly_signal_bank'),csv('split_membership'),read(OUT/'correction_heads.json'))
    cohorts=summarize(cells,weeks)
    conflicts=cells[cells.availability.eq('in_life')&cells.original_reason.isin(['brier_not_better','direction_worse'])].copy()
    files=[]
    for name,g in [('validation_seed_diagnostics',seed),('validation_week_diagnostics',weeks),('diagnostic_cells',cells),('diagnostic_summary',cohorts),('quality_clear_diagnostics',conflicts)]:
        p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_fits=0,new_policies=0,new_forecast_histories=0,new_pvalues=0,diagnostic_seed_rows=len(seed),diagnostic_week_rows=len(weeks),diagnostic_cells=len(cells),summary_cells=len(cohorts),quality_clear_view_cells=len(conflicts))
    print(f'Revalidation diagnosis complete: {len(cells)} cells across three fixed evidence views; original 23 forecast histories unchanged.',flush=True)
if __name__=='__main__':main()

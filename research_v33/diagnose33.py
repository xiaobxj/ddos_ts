from common33 import *
def main():
    check_frozen();assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis');archive=csv('model_predictions');lookup={(r.history,r.method,r.seed,r.date):r for r in archive.itertuples()};components=[];parameters=[];transforms=[];obs,_,_=data();maximum=0.
    for job in jobs():
        p,t=parameter_changes(job);parameters.extend(p);transforms.extend(t)
        for method,ha,hn,xa,xn in zip(METHODS,job['old_heads'],job['new_heads'],job['old_x'],job['new_x']):
            a,b,c,t=product_components(xa,np.asarray(ha['coefficients']),xn,np.asarray(hn['coefficients']),method);za=a.sum(1);zn=b.sum(1);pa=expit(za);pn=expit(zn);w=secant(za,zn)
            for j,i in enumerate(job['rows']):
                date=obs.date.iloc[i];ra=lookup[(cfg()['reference'],method,job['seed'],date)];rn=lookup[(cfg()['candidate'],method,job['seed'],date)];gap=max(abs(pa[j]-ra.probability),abs(pn[j]-rn.probability));maximum=max(maximum,gap);assert gap<1e-13 and int(pa[j]>.5)==ra.direction_up and int(pn[j]>.5)==rn.direction_up
                for k,group in enumerate(GROUPS):components.append(dict(method=method,seed=job['seed'],row_index=int(i),date=date,head_cutoff=job['head_cutoff'],encoder_cutoff=job['encoder_cutoff'],group=group,old_term=float(a[j,k]),new_term=float(b[j,k]),delta_logit=float(b[j,k]-a[j,k]),coefficient_delta_logit=float(c[j,k]),transform_delta_logit=float(t[j,k]),old_logit=float(za[j]),new_logit=float(zn[j]),old_probability=float(pa[j]),new_probability=float(pn[j]),secant_weight=float(w[j]),probability_contribution=float(w[j]*(b[j,k]-a[j,k])),coefficient_probability_contribution=float(w[j]*c[j,k]),transform_probability_contribution=float(w[j]*t[j,k])))
    seed=pd.DataFrame(components);assert len(seed)==16320;files=[]
    for name,g in [('seed_components',seed),('parameter_changes',pd.DataFrame(parameters)),('transform_changes',pd.DataFrame(transforms))]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    save(OUT/'component_manifest.json',dict(status='FROZEN',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},outcome_based_summary_started=False,maximum_archived_probability_gap=maximum));files.append(OUT/'component_manifest.json')
    weekly=weekly_from(seed);summary=summaries(weekly);shifts,members=window_changes();summary.update(weekly_attribution=weekly,training_window_changes=shifts,training_membership_changes=members,all_2026_cases=weekly[weekly.year.eq(2026)].reset_index(drop=True),all_direction_flips=weekly[weekly.case.isin(['regression','recovery'])].reset_index(drop=True),flips_2026=weekly[weekly.year.eq(2026)&weekly.case.isin(['regression','recovery'])].reset_index(drop=True))
    for name,g in summary.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    assert len(weekly)==1088;check_frozen();finish(run,files,new_neural_fits=0,new_head_fits=0,new_feature_inference=0,new_policies=0,new_inferential_tests=0,seed_component_rows=16320,weekly_pairs=1088,maximum_archived_probability_gap=maximum)
    print(summary['performance_summary'][summary['performance_summary'].period.eq('year_2026')].to_string(index=False),flush=True);print('Diagnostic accounting complete; independent verification pending.',flush=True)
if __name__=='__main__':main()

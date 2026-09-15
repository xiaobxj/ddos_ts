from common13 import *

def main():
    check_frozen();rolling=read(OUT/'rolling_manifest.json');assert rolling.get('finished_utc')
    for n,d in rolling['artifacts'].items():assert sha(OUT/n)==d,n
    path=OUT/'calibration_manifest.json';assert not path.exists(),'Preserve frozen coefficients'
    run=manifest('calibration');save(path,run)
    pool=pd.read_csv(OUT/'rolling_ensemble_predictions.csv');member=pd.read_csv(OUT/'calibration_membership.csv')
    coefficients=[];contributions=[];jackknife=[];files=[]
    for cutoff,members in member.groupby('outer_cutoff'):
        g=pool[pool.row_index.isin(members.row_index)].sort_values('date').copy()
        assert g.joint_completed.le(cutoff).all() and g.date.le(cutoff).all()
        assert g.inner_cutoff.lt(g.date).all()
        p=OUT/f'calibration_input_{cutoff}.csv';g.to_csv(p,index=False);files.append(p)
        fit=coefficient(g);fit.update(outer_cutoff=cutoff,first_date=g.date.min(),last_date=g.date.max(),
            latest_label_completion=g.joint_completed.max(),input_file=str(p.relative_to(ROOT)),input_sha256=sha(p))
        coefficients.append(fit)
        for year,h in g.groupby(g.date.str[:4]):
            x=h.predicted_return.to_numpy(float)-h.training_mean.to_numpy(float);z=h.actual.to_numpy(float)-h.training_mean.to_numpy(float)
            contributions.append(dict(outer_cutoff=cutoff,year=int(year),n=len(h),numerator=float(x@z),denominator=float(x@x)))
            keep=g[g.date.str[:4].ne(year)]
            jackknife.append(dict(outer_cutoff=cutoff,omitted_year=int(year),diagnostic_only=True,**coefficient(keep,minimum_years=2)))
    save(OUT/'coefficients.json',coefficients)
    pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    pd.DataFrame(contributions).to_csv(OUT/'calibration_year_contributions.csv',index=False)
    pd.DataFrame(jackknife).to_csv(OUT/'calibration_leave_year_out.csv',index=False)
    files += [OUT/n for n in ['coefficients.json','coefficients.csv','calibration_year_contributions.csv','calibration_leave_year_out.csv']]
    run.update(finished_utc=now(),coefficients=3,outer_evaluation_during_calibration=False,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(path,run);print(pd.DataFrame(coefficients)[['outer_cutoff','n','raw_alpha','alpha','fallback']].to_string(index=False),flush=True)

if __name__=='__main__':main()

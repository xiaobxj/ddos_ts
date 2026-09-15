from common35 import *

def unlabelled_effects(models):
    key=['method','seed','row_index','date'];parts=[]
    for arm,history in ARMS.items():
        g=models[models.history.eq(history)&models.method.isin(LEARNED)].sort_values(key).reset_index(drop=True);assert len(g)==3264
        if not parts:r=g[key].copy()
        else:pd.testing.assert_frame_equal(r[key],g[key],check_exact=True)
        r['p_'+arm]=g.probability.to_numpy();parts.append(arm)
    for n,v in effects(r.p_annual,r.p_add,r.p_remove,r.p_both).items():r[n]=v
    np.testing.assert_allclose(r.add_effect+r.remove_effect,r.total_delta,rtol=0,atol=1e-15);return r

def coefficient_effects():
    annuals=read(OUT/'source_heads.json');heads=read(OUT/'heads.json');both=read(OUT/'both_heads.json');records=[];summary=[]
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff)
        if cutoff==annual:continue
        for seed in cfg()['seeds']:
            for method in LEARNED:
                ts={}
                ts['annual']=np.asarray(next(h for h in annuals if h['cutoff']==annual and h['seed']==seed and h['method']==method)['coefficients'])
                ts['both']=np.asarray(next(h for h in both if h['cutoff']==cutoff and h['seed']==seed and h['method']==method)['coefficients'])
                for arm in ['add','remove']:ts[arm]=np.asarray(next(h for h in heads if h['arm']==arm and h['cutoff']==cutoff and h['seed']==seed and h['method']==method)['coefficients'])
                eff=effects(ts['annual'],ts['add'],ts['remove'],ts['both']);key=dict(cutoff=cutoff,encoder_cutoff=annual,seed=seed,method=method)
                for i in range(len(ts['annual'])):records.append(dict(**key,coordinate=i,is_intercept=i==len(ts['annual'])-1,**{'theta_'+a:float(v[i]) for a,v in ts.items()},**{n:float(v[i]) for n,v in eff.items()}))
                summary.append(dict(**key,dimensions=len(ts['annual']),**{n+'_norm':float(np.linalg.norm(v)) for n,v in eff.items()},**{n+'_intercept':float(v[-1]) for n,v in eff.items()}))
    assert len(records)==6375 and len(summary)==204;return pd.DataFrame(records),pd.DataFrame(summary)

def weekly_from(seed,ensemble,route):
    cols=['p_'+a for a in ARMS]+EFFECTS
    weekly=seed.groupby(['method','row_index','date'],sort=True)[cols].mean().reset_index();assert len(weekly)==1088
    base=ensemble[ensemble.history.eq(ARMS['annual'])&ensemble.method.isin(LEARNED)][['method','row_index','date','actual_up']]
    weekly=weekly.merge(base,on=['method','row_index','date'],validate='one_to_one');weekly=weekly.merge(route[['row_index','head_cutoff','encoder_cutoff']],on='row_index',validate='many_to_one');weekly['year']=weekly.date.str[:4].astype(int)
    a=weekly.p_annual.gt(.5).eq(weekly.actual_up);b=weekly.p_both.gt(.5).eq(weekly.actual_up)
    weekly['case']=np.where(a&~b,'regression',np.where(~a&b,'recovery',np.where(a,'stable_correct','stable_wrong')))
    for n in EFFECTS:weekly['signed_'+n]=(2*weekly.actual_up-1)*weekly[n]
    sa=weekly.signed_add_effect.to_numpy();sr=weekly.signed_remove_effect.to_numpy();adverse=np.minimum(sa,sr)<-1e-12;tie=abs(sa-sr)<=1e-12
    weekly['dominant_adverse']=np.where(~adverse,'none',np.where(tie,'tie',np.where(sa<sr,'add','remove')))
    return weekly

def summaries(weekly):
    periods=cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'])]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)];rows=[]
    for w in periods:
        g=weekly[weekly.date.between(w['start'],w['end'])]
        for method,s in g.groupby('method'):
            for case in ['all','regression','recovery','stable_correct','stable_wrong']:
                v=s if case=='all' else s[s['case'].eq(case)]
                if len(v)==0:continue
                rows.append(dict(period=w['name'],method=method,case=case,n=len(v),annual_correct=int(v.p_annual.gt(.5).eq(v.actual_up).sum()),both_correct=int(v.p_both.gt(.5).eq(v.actual_up).sum()),**{'mean_'+n:float(v[n].mean()) for n in EFFECTS},**{'mean_signed_'+n:float(v['signed_'+n].mean()) for n in EFFECTS},**{'dominant_'+n:int(v.dominant_adverse.eq(n).sum()) for n in ['add','remove','tie','none']}))
    return pd.DataFrame(rows)

def all_flips(weekly):
    parts=[]
    for arm in ['add','remove','both']:
        g=weekly.copy();a=g.p_annual.gt(.5).eq(g.actual_up);b=g['p_'+arm].gt(.5).eq(g.actual_up);g['arm']=arm;g['history']=ARMS[arm];g['arm_case']=np.where(a&~b,'regression',np.where(~a&b,'recovery','unchanged'));parts.append(g[g.arm_case.ne('unchanged')])
    return pd.concat(parts,ignore_index=True)

def main():
    check_frozen();check_phase('evaluation');assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis');seed=unlabelled_effects(csv('model_predictions'));coef,cs=coefficient_effects();files=[]
    for n,g in [('seed_effects',seed),('coefficient_effects',coef),('coefficient_effect_summary',cs)]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    save(OUT/'component_manifest.json',dict(status='FROZEN',completed_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),outcome_summary_started=False,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));files.append(OUT/'component_manifest.json')
    weekly=weekly_from(seed,csv('ensemble_predictions'),csv('routing'));summary=summaries(weekly);flips=all_flips(weekly)
    for n,g in [('weekly_effects',weekly),('effect_summary',summary),('direction_flips',flips),('all_2026_cases',weekly[weekly.year.eq(2026)]),('flips_2026',flips[flips.year.eq(2026)])]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,seed_effect_rows=3264,weekly_effect_rows=1088,coefficient_coordinates=6375,new_fits=0)
    print('Four-arm probability and coefficient effects calculated on both operation orders; independent verification pending.',flush=True)
if __name__=='__main__':main()

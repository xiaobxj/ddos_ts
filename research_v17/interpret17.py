"""Post-evaluation tail accounting only; no model/forecast changes."""
from common17 import *

def main():
    check_frozen();v=read(OUT/'verification.json');assert v['status']=='PASS'
    assert not (OUT/'posthoc_tail_provenance.json').exists(),'Preserve descriptive analysis'
    sources={h['job']:h for h in read(OUT/'source_heads.json')};refs={r['job']:r for r in read(V16/'results/validation_features.json')};rows=[]
    for head in read(OUT/'heads.json'):
        source=sources[head['source_job']];tr=load_npz(source);te=load_npz(refs[source['job']]);d=load_npz(head)
        names=previous.NAMES if head['kind']=='raw' else [f'decoded_feature_{j:02d}' for j in range(25)]
        for j,name in enumerate(names):
            a=tr['features'][:,j].astype(float);b=te['features'][:,j].astype(float);low=d['lower'][j];high=d['upper'][j]
            rows.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],feature=j,name=name,
                train_n=len(a),validation_n=len(b),lower=low,upper=high,training_lower_fraction=float((a<low).mean()),training_upper_fraction=float((a>high).mean()),
                validation_lower_fraction=float((b<low).mean()),validation_upper_fraction=float((b>high).mean()),
                validation_raw_mean=float(b.mean()),training_clipped_mean=float(d['mean'][j]),training_clipped_sd=float(d['sd'][j]),
                standardized_validation_mean_before=float(((b-d['mean'][j])/d['sd'][j]).mean()),
                standardized_validation_mean_after=float(((np.clip(b,low,high)-d['mean'][j])/d['sd'][j]).mean())))
    pd.DataFrame(rows).to_csv(OUT/'posthoc_tail_details.csv',index=False)
    save(OUT/'posthoc_tail_provenance.json',dict(completed_utc=now(),after_verification_utc=v['completed_utc'],new_fits=0,new_predictions=0,
        purpose='Descriptive lower/upper clipping accounting chosen after evaluation. Does not rank causal importance or select states, features or models.',
        artifacts={'results\\posthoc_tail_details.csv':sha(OUT/'posthoc_tail_details.csv')}))
    print(pd.DataFrame(rows).query("method=='raw25_clip'").groupby('cutoff')[['validation_lower_fraction','validation_upper_fraction']].mean().to_string(),flush=True)

if __name__=='__main__':main()

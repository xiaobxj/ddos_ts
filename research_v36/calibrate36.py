from common36 import *
def main():
    check_frozen();assert not (OUT/'calibration_manifest.json').exists();run=manifest('calibration');obs,price=raw_data();raw=descriptor_matrix(obs,price);required=csv('required_rows');heads=read(OUT/'annual_heads.json');banks=read(OUT/'feature_banks.json');thresholds=[];states=[];decoded=[]
    for annual,g in required.groupby('encoder_cutoff',sort=True):
        rows=g.row_index.to_numpy(int);tr=training_rows(obs,annual);t=calibrate(raw,tr);thresholds.append(dict(encoder_cutoff=annual,train_n=len(tr),maximum_maturity=obs.joint_completed.iloc[tr].max(),**t));r=g.copy();r['state']=assign(raw[rows],t)
        for j,n in enumerate(NAMES):r[n]=raw[rows,j]
        cov,out=coverage(raw[rows],np.array(t['lower']),np.array(t['upper']),np.array(t['mean']),np.array(t['sd']))
        for n,v in cov.items():r['market5_'+n]=v
        for j,n in enumerate(NAMES):r['outside_'+n]=out[:,j].astype(int)
        states.append(r)
        for seed in cfg()['seeds']:
            h=next(h for h in heads if h['cutoff']==annual and h['seed']==seed and h['method']==LEARNED[0]);d=arrays(h);bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));f=bank_slice(bank,rows);cov,_=coverage(f,d['rep__lower'],d['rep__upper'],d['rep__mean'],d['rep__sd']);r=g.copy();r['seed']=seed
            for n,v in cov.items():r['decoded25_'+n]=v
            decoded.append(r)
    states=pd.concat(states,ignore_index=True);decoded=pd.concat(decoded,ignore_index=True);dc=[n for n in decoded if n.startswith('decoded25_')];avg=decoded.groupby(['encoder_cutoff','row_index'],sort=False)[dc].mean().reset_index();states=states.merge(avg,on=['encoder_cutoff','row_index'],validate='one_to_one');assert len(thresholds)==6 and len(decoded)==3*len(states)
    files=[]
    p=OUT/'state_thresholds.json';save(p,thresholds);files.append(p)
    for n,g in [('state_observations',states),('decoded_coverage_by_seed',decoded)]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,annual_calibrations=6,observation_rows=len(states),neural_coverage_rows=len(decoded),outcome_conditioned_summaries=False,new_neural_fits=0,new_head_fits=0,new_feature_inference=0);print('Six annual state boundaries and label-free feature coverage frozen.',flush=True)
if __name__=='__main__':main()

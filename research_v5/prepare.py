"""Construct shared mature labels and verified prequantization representations."""
from common5 import *
import importlib.util
spec=importlib.util.spec_from_file_location('v4_data_for_v5',V4/'data.py')
v4_data=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4_data)
pack=v4_data.pack


def auxiliary_labels(frame,anchor):
    future=frame.iloc[anchor+1:anchor+6][['open','high','low','close','volume']].to_numpy(float)
    if len(future)!=5 or not np.isfinite(future).all():return None
    o,h,l,c,v=future.T
    if not ((future[:,:4]>0).all() and (v>=0).all() and (h>=np.maximum(o,c)).all()
            and (l<=np.minimum(o,c)).all() and (h>=l).all()):return None
    prices=np.log(future[:,:4]/float(frame.close.iloc[anchor]))
    observed_volume=np.log1p(frame.volume.iloc[anchor-19:anchor+1].to_numpy(float)).mean()
    volume=np.log1p(future[:,4])-observed_volume
    return np.c_[prices,volume].reshape(-1)


def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    torch=initialize();start=time.time();manifest=start_manifest('preparation')
    save(OUT/'preparation_manifest.json',manifest)
    obs=pd.read_csv(V4/'results/observation_table.csv')
    frame=pd.read_csv(V1/'data/1_000300.csv')
    rows=[];aux=[]
    for i,row in obs.iterrows():
        target=auxiliary_labels(frame,int(row.anchor))
        if target is None:continue
        r=row.to_dict();r['v4_row']=int(i);r['auxiliary_completed']=frame.date.iloc[int(row.anchor)+5]
        r['joint_completed']=max(r['completed'],r['auxiliary_completed'])
        rows.append(r);aux.append(target)
    selected=pd.DataFrame(rows)
    selected.to_csv(OUT/'observation_table.csv',index=False)
    ids=selected.v4_row.to_numpy(int)
    np.savez_compressed(CACHE/'targets.npz',returns=selected.exec_return.to_numpy(float),auxiliary=np.array(aux))
    prior=json.loads((V4/'results/preparation_manifest.json').read_text(encoding='utf-8'))
    for item in prior['tokenizer_weight']['files']:assert sha(V3/item['file'])==item['sha256']
    sys.path.insert(0,str(V3/'vendor/Kronos'))
    from model import KronosTokenizer
    tokenizer=KronosTokenizer.from_pretrained(str(V3/'models/Kronos-Tokenizer-base'),local_files_only=True).eval().cuda()
    normalized=np.load(V4/'cache/tokenizer_inputs.npy')
    old=np.load(V4/'cache/window_representations.npz');continuous=[];sign_errors=0;max_norm_error=0.
    for lower in range(0,len(normalized),64):
        with torch.inference_mode():
            z=tokenizer.embed(torch.from_numpy(normalized[lower:lower+64]).cuda())
            for layer in tokenizer.encoder:z=layer(z)
            z=torch.nn.functional.normalize(tokenizer.quant_embed(z),dim=-1)
        values=z.cpu().numpy();continuous.append(values)
        sign_errors+=int(np.count_nonzero((values>0)!=(old['kronos'][lower:lower+64]>0)))
        max_norm_error=max(max_norm_error,float(np.max(np.abs(np.linalg.norm(values,axis=-1)-1))))
    assert sign_errors==0 and max_norm_error<1e-6
    continuous=np.concatenate(continuous)
    representations=dict(raw=old['raw'][ids],bits=old['kronos'][ids],continuous=continuous[ids])
    np.savez_compressed(CACHE/'representations.npz',**representations)
    for name,values in representations.items():
        packed=pack(values,np.full((len(selected),25),5,dtype=np.int16))
        np.savez_compressed(CACHE/f'packed_{name}.npz',**packed)
    eligible=np.flatnonzero((selected.joint_completed<=cfg()['capacity']['sample_cutoff']).to_numpy())
    sample_ids=eligible[np.rint(np.linspace(0,len(eligible)-1,cfg()['capacity']['sample_count'])).astype(int)]
    assert len(np.unique(sample_ids))==32
    selected.iloc[sample_ids].to_csv(OUT/'capacity_samples.csv',index=False)
    np.save(CACHE/'capacity_indices.npy',sample_ids)
    save(OUT/'representation_verification.json',dict(status='PASS',all_prior_windows=len(normalized),
         compared_prequant_sign_coordinates=int(old['kronos'].size),sign_mismatches=sign_errors,
         maximum_continuous_unit_norm_error=max_norm_error,selected_observations=len(selected),
         old_observations=len(obs),capacity_samples=32,capacity_last_completed=selected.joint_completed.iloc[sample_ids].max(),
         pre2018_pretrained_inputs_role='IN_SAMPLE_CAPACITY_DIAGNOSTIC_ONLY'))
    assert old_evidence()==manifest['old_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start,
                    tokenizer_weight=prior['tokenizer_weight'],cache_sha256={str(p.relative_to(ROOT)):sha(p) for p in CACHE.iterdir()},
                    observation_table_sha256=sha(OUT/'observation_table.csv'))
    save(OUT/'preparation_manifest.json',manifest)
    print(json.dumps(dict(observations=len(selected),sign_errors=sign_errors,norm_error=max_norm_error,
                          seconds=time.time()-start),indent=2),flush=True)


if __name__=='__main__':main()

"""Freeze causal per-window representations and P0-matched segment boundaries."""
from util import *
from data import observed_window,raw_transform,tokenizer_transform,build_packed
import time
import subprocess


def main():
    CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    config=cfg();start=time.time()
    manifest=dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
                  source_sha256={n:sha(ROOT/n) for n in ['util.py','data.py','prepare.py']},
                  previous_evidence=previous_evidence(),python=sys.version,executable=sys.executable)
    save_json(OUT/'preparation_manifest.json',manifest)
    frame=pd.read_csv(V1/'data/1_000300.csv')
    obs=observation_table(frame)
    previous=pd.read_csv(V3/'results/observation_table.csv').drop(columns=['panel_valid'])
    pd.testing.assert_frame_equal(obs,previous,check_dtype=False)
    obs.to_csv(OUT/'observation_table.csv',index=False)
    windows=np.array([observed_window(frame,t) for t in obs.anchor.to_numpy(int)])
    raw=np.array([raw_transform(x) for x in windows])
    normalized=np.array([tokenizer_transform(x) for x in windows])
    np.save(CACHE/'tokenizer_inputs.npy',normalized)
    print(f'Observed-prefix input windows: {len(obs)}',flush=True)
    torch=initialize_torch()
    sys.path.insert(0,str(V3/'vendor/Kronos'))
    from model import KronosTokenizer
    model_path=V3/'models/Kronos-Tokenizer-base'
    weights=json.loads((V3/'sources/weights_manifest.json').read_text(encoding='utf-8'))
    record=next(x for x in weights if x['repo']=='NeoQuasar/Kronos-Tokenizer-base')
    for file in record['files']:
        assert sha(V3/file['file'])==file['sha256']
    pretrained=KronosTokenizer.from_pretrained(str(model_path),local_files_only=True).eval().cuda()
    torch.manual_seed(20260913)
    random=KronosTokenizer(**json.loads((model_path/'config.json').read_text(encoding='utf-8'))).eval().cuda()
    torch.save(random.state_dict(),CACHE/'random_tokenizer_state.pt')
    representations={'raw':raw};token_stats=[]
    for name,model in [('kronos',pretrained),('random_tokenizer',random)]:
        outputs=[]
        for lower in range(0,len(obs),64):
            batch=torch.from_numpy(normalized[lower:lower+64]).cuda()
            with torch.inference_mode():
                ids=model.encode(batch,half=True)
                bits=model.indices_to_bits(ids,half=True)
            assert bits.shape[1:]==(125,20)
            outputs.append(bits.cpu().numpy())
        result=np.concatenate(outputs)
        assert np.isfinite(result).all()
        representations[name]=result
        token_stats.append(dict(name=name,positive_bit_fraction=float((result>0).mean()),
                                dimensions=20,window_count=len(result),
                                per_dimension_positive_fraction=(result>0).mean(axis=(0,1)).tolist()))
        print(f'{name} tokenizer complete: {time.time()-start:.1f}s',flush=True)
    np.savez_compressed(CACHE/'window_representations.npz',**representations)
    save_json(OUT/'tokenizer_diagnostics.json',token_stats)
    del pretrained,random;torch.cuda.empty_cache()
    sys.path.insert(0,str(V1))
    from segmentation import ending_costs,reverse_lengths
    prototype_records=[]
    for year in [2024,2025,2026]:
        source=V1/'results'/f'prototypes_{year}.npz'
        prototypes=np.load(source)['prototypes']
        _,norm=ending_costs(frame.close.to_numpy(float),prototypes)
        lengths=np.zeros((len(obs),25),dtype=np.int16);permuted=np.zeros_like(lengths)
        for i,anchor in enumerate(obs.anchor.to_numpy(int)):
            sizes=reverse_lengths(norm,int(anchor)+1)
            lengths[i,-len(sizes):]=sizes
            permuted[i,-len(sizes):]=np.random.default_rng(20260909+int(anchor)).permutation(sizes)
        np.savez_compressed(CACHE/f'lengths_{year}.npz',adaptive=lengths,permuted=permuted)
        prototype_records.append(dict(year=year,source=str(source.relative_to(PROJECT)),sha256=sha(source),
                                      cutoff=f'{year-1}-12-31',mean_patch_count=float((lengths>0).sum(axis=1).mean())))
        print(f'P0 boundaries {year} complete: {time.time()-start:.1f}s',flush=True)
    save_json(OUT/'prototype_provenance.json',prototype_records)
    for arm in config['arms']:
        for year in ([2024] if arm['segmentation']=='fixed' else [2024,2025,2026]):
            build_packed(arm,year)
        print(f"Packed {arm['name']}: {time.time()-start:.1f}s",flush=True)
    assert previous_evidence()==manifest['previous_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start,
                    cache_sha256={str(p.relative_to(ROOT)):sha(p) for p in CACHE.iterdir() if p.is_file()},
                    tokenizer_weight=record,random_tokenizer_seed=20260913,
                    tokenizer_source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (V3/'vendor/Kronos/model').glob('*.py')},
                    input_windows=len(obs),previous_preserved=True)
    save_json(OUT/'preparation_manifest.json',manifest)
    print('Frozen representations and patch inputs ready.',flush=True)


if __name__=='__main__':
    main()

"""One-time mechanical port of audited workflow to a new research directory."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'research_v22'
def write(name,s):
    path=ROOT/name;assert not path.exists();path.write_text(s,encoding='utf-8')
def template(name):return (OLD/name.replace('23','22')).read_text(encoding='utf-8').replace('common22','common23').replace('evaluate22','evaluate23').replace('contract22','contract23')
def replace(s,a,b):
    assert a in s,a;return s.replace(a,b)

s=template('common23.py')
s=replace(s,"V21=PROJECT/'research_v21'","V22=PROJECT/'research_v22'").replace('str(V21)','str(V22)').replace('import common21 as previous','import common22 as previous')
s=replace(s,'base_inputs=previous.base_inputs;','')
s=replace(s,'from states22 import features,scalar_features,assign,run_diagnostics,GATES,NUMERIC,STATE_PARTITIONS','from states23 import features,scalar_features,run_diagnostics,GATES,NUMERIC,STATE_PARTITIONS\nimport common20 as parent_inputs\ndef base_inputs(source,split):return parent_inputs.source_inputs(source,19,split)\ndef source_ray(source):return load_npz(source)[\'ray\']')
a=s.index('CANDIDATES=');b=s.index("def cfg():",a)
s=s[:a]+"""CANDIDATES={'learned_order_extension':'learned_vol_interaction','raw_order_extension':'raw_vol_interaction'}
GATE_MAP={m:'excess_order' for m in CANDIDATES};CONTROLS={'excess_order':'order_only'}
OLD_INTERACTIONS={'learned_order_extension':'learned_market','raw_order_extension':'raw_trend'}
METHODS=previous.METHODS+list(CANDIDATES)+list(CONTROLS.values());SEED_METHODS=previous.SEED_METHODS+['learned_order_extension']
PARTITIONS=dict(previous.PARTITIONS,**STATE_PARTITIONS)
CORE=['states23.py','common23.py','prepare23.py','contract23.py','train23.py','score23.py','evaluate23.py','verify23.py']
"""+s[b:]
s=s.replace('V21/', 'V22/').replace('3166','3292')
s=replace(s,"'hmm_signal_states.csv','verification.json'","'validation_signal_states.csv','heads.json','verification.json'")
write('common23.py',s)

s=template('train23.py').replace('np.asarray(source[\'coefficients\'])[:25]','source_ray(source)')
for a,b in [('60heads','30heads'),('/60','/30'),('len(heads)==60','len(heads)==30'),('projections==48','projections==24'),('total==94650','total==47325'),('len(coefs)==1476','len(coefs)==762'),('new_primary_fits=60','new_primary_fits=30'),('new_projection_fits=48','new_projection_fits=24')]:s=replace(s,a,b)
s=replace(s,"else 'additive' if h['source_job'] else 'state'","else 'original_interaction' if h['source_job'] and i==len(theta)-3 else 'parent' if h['source_job'] else 'state'")
write('train23.py',s)

s=template('score23.py').replace('V21/', 'V22/').replace('refitted_additive','refitted_parent')
for a,b in [('new_primary_fits\']==60','new_primary_fits\']==30'),('len(new)==2610','len(new)==1305'),('len(models)==11223','len(models)==12528'),('len(ensemble)==6786','len(ensemble)==7569'),('new_probability_forecasts=2610','new_probability_forecasts=1305'),('reused_model_records=8613','reused_model_records=11223'),('model_records=11223','model_records=12528'),('ensemble_records=6786','ensemble_records=7569'),('interaction_components=2088','interaction_components=1044'),('Scored2610new probabilities;26methods','Scored1305new probabilities;29methods')]:s=replace(s,a,b)
# Restore accidental substring replacement in reused count.
s=s.replace('reused_model_records=12528','reused_model_records=11223')
s=replace(s,"correlation_abs_trend60=float(np.corrcoef(u,abs(old.trend60))[0,1])))","correlation_abs_trend60=float(np.corrcoef(u,abs(old.trend60))[0,1]),correlation_efficiency60=float(np.corrcoef(u,pd.read_csv(OUT/f'{split}_signal_states.csv',float_precision='round_trip').query('cutoff == @f[\"cutoff\"]')['efficiency60'])[0,1])))") if False else s
write('score23.py',s)

s=template('evaluate23.py')
s=replace(s,"for frame in [old,hmm,new]", "inherited=pd.read_csv(V22/'results/validation_signal_states.csv')[['cutoff','row_index','date']+list(previous.STATE_PARTITIONS)]\n    for frame in [old,hmm,inherited,new]")
s=replace(s,'len(pairs)==52','len(pairs)==26');s=replace(s,"len(tables['state_metrics'])==1196","len(tables['state_metrics'])==1508");s=replace(s,"len(tables['reliability_bins'])==250","len(tables['reliability_bins'])==280")
for a,b in [('primary_contrasts=52','primary_contrasts=26'),('methods=26','methods=29'),('metric_groups=1520','metric_groups=1868'),('state_metric_rows=1196','state_metric_rows=1508'),('reliability_bins=250','reliability_bins=280')]:s=replace(s,a,b)
write('evaluate23.py',s)

s=template('verify23.py').replace('refitted_additive','refitted_parent').replace('V21/','V22/')
s=replace(s,"col='relative_volatility_state' if gate=='relative_volatility' else 'persistence_state';high='elevated' if gate=='relative_volatility' else 'persistent'","col='order_state';high='positive_order'")
a=s.index('            valid=np.isfinite(scalar).all(axis=1);');b=s.index('            replay=features',a)
s=s[:a]+"""            valid=np.isfinite(scalar).all(axis=1);labels=np.where(valid,np.where(scalar[:,-1]<-.05,'negative_order',np.where(scalar[:,-1]>.05,'positive_order','near_zero_order')),'missing')
            np.testing.assert_array_equal(d.order_state,labels)
"""+s[b:]
s=replace(s,"original={s['job']:s for s in read(V18/'results/heads.json')}","original={s['job']:s for s in read(V19/'results/heads.json')}")
s=replace(s,"oldpred=pd.read_csv(V18/'results/model_predictions.csv'","oldpred=pd.read_csv(V19/'results/model_predictions.csv'")
s=replace(s,"ray=np.asarray(source['coefficients'])[:25]","ray=source_ray(source)")
for a,b in [('new_primary_fits\']==60','new_primary_fits\']==30'),('new_projection_fits\']==48','new_projection_fits\']==24'),('len(heads)==len(jobs)==60','len(heads)==len(jobs)==30'),('total==94650','total==47325'),('testtotal==2610','testtotal==1305'),('len(coefs)==1476','len(coefs)==762'),('len(projections)==48','len(projections)==24'),('len(parts)==2088','len(parts)==1044'),('len(summary)==96','len(summary)==48'),('len(preds)==11223','len(preds)==12528'),('len(ensemble)==6786','len(ensemble)==7569'),('len(new)==2610','len(new)==1305'),('groups==1520','groups==1868'),("len(tables['reliability_bins'])==250","len(tables['reliability_bins'])==280"),('len(pairs)==52','len(pairs)==26'),('range(52)','range(26)'),('(52-rank)','(26-rank)'),('previous_files_preserved=3166','previous_files_preserved=3292'),('new_primary_fits=60','new_primary_fits=30'),('independent_solutions=60','independent_solutions=30'),('independent_qr_projections=48','independent_qr_projections=24'),('training_feature_rows=94650','training_feature_rows=47325'),('new_probability_forecasts=2610','new_probability_forecasts=1305'),('reused_model_records=8613','reused_model_records=11223'),('model_records=11223','model_records=12528'),('ensemble_records=6786','ensemble_records=7569'),('metric_groups=1520','metric_groups=1868'),('state_metric_rows=1196','state_metric_rows=1508'),('reliability_bins=250','reliability_bins=280'),('primary_contrasts=52','primary_contrasts=26')]:s=replace(s,a,b)
s=s.replace('reused_model_records=12528','reused_model_records=11223')
write('verify23.py',s)
print('Created5workflow files; core not frozen or fitted.')

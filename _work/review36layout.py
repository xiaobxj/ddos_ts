from pathlib import Path
import sys,json,hashlib,textwrap
P=Path(__file__).resolve().parents[1];R=P/'research_v36';O=R/'results';sys.path.insert(0,str(R));import report36 as r
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
path=O/'report_manifest.json';initial=path.read_bytes();manifest=json.loads(initial);assert not (O/'layout_adjustment.json').exists()
archived={}
for ext in ['png','svg']:
    original=O/f'state_forecast_2026.{ext}';archive=O/f'state_forecast_2026_initial.{ext}';assert not archive.exists();assert sha(original)==manifest['artifacts'][str(original.relative_to(R))];archive.write_bytes(original.read_bytes());archived[str(original.relative_to(R))]=dict(archive=str(archive.relative_to(R)),sha256=sha(archive))
(O/'report_manifest_initial.json').write_bytes(initial)
# Re-render the exact frozen plot code with one presentation-only margin change.
# No model, state, summary, figure value, legend or text changes.
source=(R/'report36.py').read_text(encoding='utf-8');start=source.index('    fig,axs=plt.subplots(1,3');end=source.index("    plt.close(fig);(OUT/'report_manifest.json')",start);code=textwrap.dedent(source[start:end]);assert code.count('right=.99')==1;code=code.replace('right=.99','right=.95')
font=r.FontProperties(fname='C:/Windows/Fonts/msyh.ttc');r.plt.rcParams['font.family']=font.get_name();r.plt.rcParams['axes.unicode_minus']=False
ns=dict(vars(r));ns.update(pm=r.csv('state_prediction_metrics'),files=[]);exec(compile(code,str(Path(__file__)), 'exec'),ns);r.plt.close(ns['fig'])
adjustment=dict(status='LAYOUT_ONLY',completed_utc=r.now(),reason='Rightmost long x-axis label touched the image boundary in the initial forecast chart. Increase right margin from1% to5%.',source_script_sha256=sha(R/'report36.py'),render_helper_sha256=sha(Path(__file__)),source_metrics_sha256=sha(O/'state_prediction_metrics.csv'),initial_report_manifest_sha256=sha(O/'report_manifest_initial.json'),initial_artifacts=archived,new_artifacts={str(p.relative_to(R)):sha(p) for p in ns['files']},predictions_and_numbers_unchanged=True)
(O/'layout_adjustment.json').write_text(json.dumps(adjustment,ensure_ascii=False,indent=2),encoding='utf-8')
manifest['artifacts'].update(adjustment['new_artifacts'])
for p in [O/'layout_adjustment.json',O/'report_manifest_initial.json']+[R/v['archive'] for v in archived.values()]:manifest['artifacts'][str(p.relative_to(R))]=sha(p)
manifest['finished_utc']=r.now();manifest['presentation_amendment']='results/layout_adjustment.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Forecast chart margin corrected; initial chart and manifest retained, all data unchanged.')

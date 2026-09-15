from pathlib import Path
import sys,json,hashlib,textwrap
ROOT=Path('D:/ddos_v3/research_v47');OUT=ROOT/'results';sys.path.insert(0,str(ROOT))
import report47 as report
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=OUT/'visual_attempt1';assert not archive.exists();archive.mkdir()
original={}
for name in ['training_weights.png','training_weights.svg','report_manifest.json']:
    p=OUT/name;(archive/name).write_bytes(p.read_bytes());original[str((archive/name).relative_to(ROOT))]=sha(archive/name)
source=(ROOT/'report47.py').read_text(encoding='utf-8');start=source.index('    fig,axes=plt.subplots(1,3,figsize=(15,5.5))');end=source.index('    plt.close(fig)',start)+len('    plt.close(fig)')
snippet=textwrap.dedent(source[start:end]).replace('bbox_to_anchor=(.5,-.13)','bbox_to_anchor=(.5,-.24)').replace('bottom=.29','bottom=.34')
code=OUT/'visual_revision_source.py.txt';code.write_text(snippet+'\n',encoding='utf-8')
report.plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
namespace=vars(report).copy();namespace.update(summary=report.csv('weight_summary'),weights=report.csv('training_weights'));exec(compile(snippet,str(code),'exec'),namespace)
revision=dict(kind='layout_only',completed_utc=report.pd.Timestamp.now(tz='UTC').isoformat(),reason='Original training-weight legends overlapped x-axis labels. Move legends lower and increase bottom margin. Data,limits,methods and scientific sources unchanged.',frozen_report_source_sha256=sha(ROOT/'report47.py'),original_artifacts=original,revision_source_sha256=sha(code),updated_artifacts={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['training_weights.png','training_weights.svg']})
p=OUT/'visual_revision.json';p.write_text(json.dumps(revision,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=json.loads((OUT/'report_manifest.json').read_text(encoding='utf-8'));manifest['artifacts'].update(revision['updated_artifacts']);manifest['artifacts'].update(original);manifest['artifacts'][str(p.relative_to(ROOT))]=sha(p);manifest['artifacts'][str(code.relative_to(ROOT))]=sha(code);manifest['finished_utc']=revision['completed_utc'];manifest['visual_revision']='results/visual_revision.json'
(OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('R47 weight-figure layout revised;original artifacts archived;visual inspection pending.')

"""Layout-only revision after visual review; retain original outputs and manifest."""
from pathlib import Path
import sys,json,hashlib,shutil,datetime
P=Path(__file__).resolve().parents[1];root=P/'research_v31';out=root/'results'
sys.path.insert(0,str(root));import report31 as r
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
original=out/'visual_attempt1';assert not original.exists();original.mkdir()
for name in ['shadow_and_blend_yearly.png','shadow_and_blend_yearly.svg','report_manifest.json']:shutil.copy2(out/name,original/name)
manifest=json.loads((out/'report_manifest.json').read_text(encoding='utf-8'));yr=r.csv('yearly_metrics');font=r.FontProperties(fname='C:/Windows/Fonts/msyh.ttc');r.plt.rcParams['font.family']=font.get_name();r.plt.rcParams['axes.unicode_minus']=False
fig,axs=r.plt.subplots(1,3,figsize=(17,6),sharey=True);fig.subplots_adjust(left=.055,right=.99,top=.82,bottom=.23,wspace=.12)
policies=['rolling5_annual20','rolling5_quarterly20','shadow_trial','annual_quarter_half'];colors=['#4c78a8','#999999','#c44e52','#2b927b'];styles=['-','--','-','-'];markers=['o','s','^','D'];years=sorted(yr.year.unique())
for ax,method in zip(axs,r.PRIMARY):
    for h,color,style,marker in zip(policies,colors,styles,markers):
        g=yr[yr.history.eq(h)&yr.method.eq(method)].set_index('year').loc[years];ax.plot(r.np.arange(6),g.accuracy,color=color,ls=style,marker=marker,lw=2,label=r.HL[h])
    ax.set_title(r.ML[method],fontsize=13);ax.set_xticks(r.np.arange(6),['2021','2022','2023','2024','2025','2026*'],rotation=35);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(r.matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=4,frameon=False,fontsize=11);fig.suptitle('5 年窗口、自然 20 遍：延迟切换与固定融合的逐年表现',fontsize=17,y=.965);fig.text(.5,.035,'同一批历史周；三种子等权。*2026 年截至 8 月（30 周）；其余年份 47–50 周。',ha='center',fontsize=10)
for ext in ['png','svg']:fig.savefig(out/f'shadow_and_blend_yearly.{ext}',dpi=150)
r.plt.close(fig);shutil.copy2(Path(__file__),out/'visual_revision_source.py.txt')
revision=dict(status='LAYOUT_REVISED_PENDING_VISUAL_REVIEW',reason='Initial bottom legend overlapped the sample-period footnote; use explicit non-overlapping rows for axes, legend and footnote.',research_sources_changed=False,protocol_changed=False,metrics_changed=False,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_source_sha256=sha(Path(__file__)),original_artifacts={str(p.relative_to(root)):sha(p) for p in original.iterdir()})
(out/'visual_revision.json').write_text(json.dumps(revision,indent=2,ensure_ascii=False),encoding='utf-8')
manifest['finished_utc']=revision['completed_utc'];manifest['layout_revision']='results/visual_revision.json';manifest['original_report_manifest_sha256']=sha(original/'report_manifest.json')
for p in [out/'shadow_and_blend_yearly.png',out/'shadow_and_blend_yearly.svg',out/'visual_revision.json',out/'visual_revision_source.py.txt']+list(original.iterdir()):manifest['artifacts'][str(p.relative_to(root))]=sha(p)
(out/'report_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8');print('Figure layout revised; initial artifacts preserved. Research results unchanged.')

"""Chinese report for a fixed candidate stability study, with complete evidence links."""
from pathlib import Path
import json
import hashlib
import re
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
HISTORIES=['full','drop_early','drop_middle','drop_recent']
HN={'full':'完整历史（复用）','drop_early':'删除最早约 20%','drop_middle':'删除中间约 20%','drop_recent':'删除最近约 20%'}
RN={'candidate20':'固定组合候选 20 轮','baseline10':'原基准 10 轮','baseline20':'原模型 20 轮','combined10':'组合模型 10 轮（诊断）'}


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(x):return f'{x*100:.2f}%'
def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)


def main():
    audit=read('verification.json');assert audit['status']=='PASS'
    findings=read('findings.json');assessment=read('stability_assessment.json')
    comparison=pd.read_csv(OUT/'candidate_comparisons.csv')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    years=pd.read_csv(OUT/'yearly_metrics.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    shifts=pd.read_csv(OUT/'prediction_sensitivity.csv')
    for frame in [metrics,years,seeds,shifts]:
        frame['history']=pd.Categorical(frame.history,categories=HISTORIES,ordered=True)
        frame['role']=pd.Categorical(frame.role,categories=['candidate20','baseline10','baseline20','combined10'],ordered=True)
    metrics=metrics.sort_values(['history','role'])
    years=years.sort_values(['history','role','year']);seeds=seeds.sort_values(['history','role','seed'])
    shifts=shifts.sort_values(['history','role','level','seed'])
    tables={}
    tables['comparison']=md(['训练历史','候选准确率','候选 RMSE','原基准 10 轮 RMSE','训练均值 RMSE','候选对均值 MSE 改善','优于均值的候选种子'],[
        [HN[r.history],pct(r.candidate_accuracy),f'{r.candidate_rmse:.6f}',f'{r.baseline10_rmse:.6f}',
         f'{r.training_mean_rmse:.6f}',pct(r.mse_skill_vs_mean),f'{r.candidate_seeds_better_than_mean}/3'] for r in comparison.itertuples()])
    descriptions={'absolute_advantage':'三个删除历史的集成 MSE 都低于训练均值',
                  'relative_advantage':'三个删除历史的集成 MSE 都低于原基准 10 轮',
                  'seed_consistency':'每个删除历史至少两个种子的 MSE 低于训练均值'}
    tables['flags']=md(['预先固定的一致性条件','结果'],[[descriptions[k],'满足' if v else '不满足'] for k,v in assessment['flags'].items()])
    tables['primary']=md(['训练历史','主要对照','候选 − 对照 ΔRMSE','95% 区间','MSE 原始 p','MSE Holm p'],[
        [HN[r['history']],RN.get(r['reference'],'训练均值'),f"{r['rmse']['difference']:+.6f}",
         f"[{r['rmse']['ci95_low']:+.6f}, {r['rmse']['ci95_high']:+.6f}]",f"{r['mse']['p']:.4f}",
         f"{r['mse']['holm_adjusted_p']:.4f}"] for r in read('primary_comparisons.json')])
    tables['seeds']=md(['训练历史','种子','准确率','RMSE','对训练均值 MSE 改善'],[
        [HN[r.history],r.seed,pct(r.accuracy),f'{r.rmse:.6f}',pct(r.mse_skill_vs_training_mean)]
        for r in seeds[seeds.role=='candidate20'].itertuples()])
    tables['years']=md(['训练历史','验证年','周数','准确率','RMSE','对训练均值 MSE 改善'],[
        [HN[r.history],r.year,r.n,pct(r.accuracy),f'{r.rmse:.6f}',pct(r.mse_skill_vs_training_mean)]
        for r in years[years.role=='candidate20'].itertuples()])
    tables['all']=md(['训练历史','角色','准确率','RMSE','对训练均值 MSE 改善','折内预测 SD／bp','输入重配后 MSE 变化'],[
        [HN[r.history],RN[r.role],pct(r.accuracy),f'{r.rmse:.6f}',pct(r.mse_skill_vs_training_mean),
         f'{r.within_fold_forecast_std*10000:.2f}',pct(r.reassignment_mse_increase/r.mse)]
        for r in metrics.itertuples()])
    tables['shifts']=md(['训练历史','角色','预测 RMS 改变／bp','训练均值 RMS 改变／bp','去除训练均值后 RMS 改变／bp','方向改变比例'],[
        [HN[r.history],RN[r.role],f'{r.prediction_rms_shift*10000:.2f}',f'{r.training_mean_rms_shift*10000:.2f}',
         f'{r.mean_removed_prediction_rms_shift*10000:.2f}',pct(r.direction_disagreement)]
        for r in shifts[(shifts.level=='ensemble')&shifts.role.isin(['candidate20','baseline10'])].itertuples()])
    from report_text7 import report_text
    content=report_text(audit,findings,assessment,tables)
    destination=OUT/'第七轮测试报告.md';destination.write_text(content,encoding='utf-8')
    links=re.findall(r'\]\(([^)]+)\)',content)
    for target in links:
        if not target.startswith(('https://','http://')):assert (OUT/target).resolve().exists(),target
    files={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json':
            with p.open('rb') as stream:files[str(p.relative_to(ROOT))]=hashlib.file_digest(stream,'sha256').hexdigest()
    save=dict(report=str(destination.relative_to(ROOT)),report_links_checked=len(links),files=files)
    (OUT/'delivery_manifest.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(destination),characters=len(content),hashed_files=len(files),links=len(links))))


if __name__=='__main__':main()

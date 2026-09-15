"""Create immutable data/protocol, verify labels and freeze before fitting."""
from core import *

def main():
    OUT.mkdir(parents=True,exist_ok=False);(ROOT/'data').mkdir()
    protocol=dict(stock='603986',data_end='2026-09-15',input_bars=125,horizons=[1,5],
        target='gross close(t) to close(t+H), corporate-action entitlement adjusted; no trading execution claim',
        annual_cutoffs=[f'{y}-12-31' for y in range(2022,2026)],quarterly_full_cutoffs=['2026-03-31','2026-06-30'],
        seeds=SEEDS,epochs=[10,20],neural_fits=84,primary='mse_5y_e20.vol.U',threshold=.5,
        backbone='fresh 38551 parameter raw fixed-patch regularized Crossformer; no pretrained weights',
        training='AdamW lr .001 wd .1, dropout .3, batch 128 natural epochs; MSE standardized returns or BCE raw return>0; .1 auxiliary MSE on next H bars only',
        probability_heads=dict(methods=METHODS,schemes=['U','W','I','S'],ridge=.01,clip=[.01,.99],half_life_days=730.5),
        auxiliary_ablation='10 vs 20 checkpoints; no new Huber/balanced-batch/no-aux search; nominal batch imbalance diagnosed',
        controls=['training_frequency_same_members','market4_same_members','constant_0.5','always_up','always_down','BCE_native'],
        calibration=dict(source='causal annual OOS primary ensemble only',train_daily=252,validation_daily=63,
            purge='all train label endpoints strictly before first validation anchor',min_train_nonoverlap=10,min_validation_nonoverlap=5,
            ridge_sum_loss=20,offset_cap=.5,accept='Brier improves >1e-12 and correct count does not fall; no refit after validation',
            policies=['quarter_four_state','month_four_state','quarter_pooled','quarter_ungated','quarter_platt'],
            year_reset='Q1 returns to annual baseline; no carry/renewal of old state parameters',
            diagnostics=['leave-one-disjoint-block validation sensitivity','adjacent-window sample reuse','actual coverage vs accepted cells','state-wise next-period performance']),
        evaluation=dict(start='2023-01-01',end='2026-09-15',strict_common_dates_per_horizon=True,
            metrics=['Brier','logloss','accuracy','balanced_accuracy','AUC','probability_sd','calibration_bins','per_year','per_state','seed_variability'],
            bootstrap_blocks=[20,40],bootstrap_draws=10000,bootstrap_seed=603986,
            primary_comparisons='primary vs training_frequency and market4 in H1/H5, Brier differences; Holm 4 primary comparisons',
            secondary='all other comparisons descriptive exploratory; no post-result promotion; historical dates previously researched',
            weekly_overlap='daily H5 labels remain overlapping; greedy disjoint counts are not effective N'),
        integrity=['no future data in inputs/scales/states/labels','independent target math','H1 no H5 maturity delay',
            'future suffix and prefix replay','checkpoint and feature replay','head optimality and residual rank',
            'pending targets excluded from scores','no historical prediction inserted into prospective ledger','old files preserved'],
        limitations=['single stock selected by user, not survivorship-free universe','retrospective vendor history not PIT revision archive',
            'no intraday/news/fundamentals data','not full paper reproduction','no profitability test or same-close executable entry assumption',
            'quarterly full-refit test limited to 2026; no expanding neural refit, HMM, trigger search or old-state memory policy in this round'])
    save(ROOT/'protocol.json',protocol)
    src=PROJECT/'stock_weekly_reviews/603986/2026-09-14_asof_2026-09-15'
    m=read(src/'manifest.json')
    for p,s in m['files'].items():assert sha(src/p)==s,p
    raw=load_csv(PROJECT/'stock_model_603986_v1/data/raw.csv').drop(columns='valid_ohlc')
    recent=load_csv(src/'daily.csv').rename(columns={'volume_lots':'volume'})[raw.columns]
    overlap=raw[raw.date.ge(recent.date.min())].reset_index(drop=True)
    pd.testing.assert_frame_equal(overlap,recent[recent.date.le(raw.date.max())].reset_index(drop=True),check_exact=True)
    raw=pd.concat([raw,recent[recent.date.gt(raw.date.max())]],ignore_index=True)
    cal=load_csv(old.CALENDAR).date;cal=cal[cal.ge(raw.date.min())].tolist()
    # The new Sep15 session has an observed final stock daily bar, not an imputed day.
    assert recent.date.max()==protocol['data_end'];cal.extend(d for d in recent.date if d>cal[-1])
    f=old.build_frame(raw,cal);csv(ROOT/'data/raw.csv',raw);csv(ROOT/'data/model_frame.csv',f)
    # Exact sealed-frame prefix, including NaN suspension sessions.
    pd.testing.assert_frame_equal(f.iloc[:-1][old.data().columns].reset_index(drop=True),old.data(),check_exact=True)
    contract=[]
    for h in [1,5]:
        obs,targets=observations(f,h);csv(ROOT/f'data/observations_h{h}.csv',obs)
        assert np.isfinite(targets['auxiliary'][obs.matured]).all()
        for row in obs[obs.matured].itertuples():
            # Independent action-based total return, on raw share ownership.
            shares=1.;cash=0.
            for date,split,div in old.ACTIONS:
                if row.date<date<=row.joint_completed:cash+=shares*div;shares*=split
            direct=(shares*f.raw_close.iloc[row.exit]+cash)/f.raw_close.iloc[row.anchor]-1
            assert abs(direct-row.actual_return)<2e-14
            assert row.exit-row.anchor==h and row.joint_completed==f.date.iloc[row.anchor+h]
        for t in tasks(h):
            ids,lower=membership(obs,t['cutoff'],t['years'])
            prefix=f[f.date.le(t['cutoff'])].reset_index(drop=True);po,pt=observations(prefix,h)
            pids,_=membership(po,t['cutoff'],t['years'])
            pd.testing.assert_frame_equal(obs.iloc[ids].reset_index(drop=True),po.iloc[pids].reset_index(drop=True),check_exact=True)
            np.testing.assert_array_equal(targets['auxiliary'][ids],pt['auxiliary'][pids])
            contract.append(dict(**t,train_n=len(ids),lower=lower,last_label=obs.joint_completed.iloc[ids].max(),last_anchor=obs.date.iloc[ids].max()))
        assert obs.iloc[-1].date==protocol['data_end'] and not obs.iloc[-1].matured
    csv(OUT/'training_membership_plan.csv',pd.DataFrame(contract))
    # Preserve all files in the existing stock delivery and the sealed index inventory.
    preserved={}
    v1=PROJECT/'stock_model_603986_v1';dm=read(v1/'results/delivery_manifest.json')
    for p,s in dm['files'].items():assert sha(v1/p)==s,p;preserved[rel(v1/p)]=s
    preserved[rel(v1/'results/delivery_manifest.json')]=sha(v1/'results/delivery_manifest.json')
    p=PROJECT/'research_v51/results/freeze.json'
    prior=read(p)
    preserved.update(prior['old_evidence'])
    for path in (PROJECT/'research_v51').rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):preserved[rel(path)]=sha(path)
    save(OUT/'preserved_files.json',preserved)
    reports=[]
    for n in range(1,52):
        folder=PROJECT/('research' if n==1 else f'research_v{n}')/'results'
        ps=list(folder.glob('*.md'));p=min(ps,key=lambda p:p.stat().st_size)
        reports.append(dict(round=n,file=rel(p),sha256=sha(p)))
    save(OUT/'prior_report_sources.json',reports)
    r,_,_=modules()
    files=set(ROOT.glob('*.py'))|{ROOT/'protocol.json',src/'manifest.json',src/'daily.csv',old.CALENDAR}
    files.update(p for p in (ROOT/'data').iterdir() if p.is_file())
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.is_relative_to(PROJECT) and '.venv' not in str(p) and p.suffix=='.py':files.add(p)
    files.update(PROJECT.glob('research*/protocol.json'))
    save(OUT/'contract.json',dict(status='PASS',checked_utc=now(),target_math='all mature rows independently recomputed',
        raw_rows=len(raw),calendar_rows=len(f),missing_sessions=int((~f.valid_ohlc).sum()),source_prefix='exact',
        horizons=[1,5],membership_prefix_checks=len(contract),prior_rounds=51))
    save(OUT/'freeze.json',dict(created_utc=now(),files={rel(p):sha(p) for p in sorted(files)}))
    print(json.dumps(contract,indent=2))

if __name__=='__main__':main()

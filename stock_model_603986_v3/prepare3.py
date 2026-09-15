from experiment3 import *

def main():
    OUT.mkdir(exist_ok=False)
    protocol=dict(stock='603986',data_end='2026-09-15',purpose='extend H5 quarterly WFO, constrain probabilities, H1 external context, genuine daily prospective records',
        new_cutoffs=[f'{y}-{md}' for y in range(2023,2026) for md in ['03-31','06-30','09-30']],
        neural_fits=27,neural_recipe='v2 H5 MSE 5-year, same 3 seeds and natural 20 epochs; all old fits reused with hashes',
        slow=dict(cadences=['annual','quarterly'],methods=METHODS,schemes=['raw','platt','shrink'],primary='quarterly.vol.shrink',
            train_daily=252,validation_daily=63,purge='train target endpoints strictly before first validation anchor',
            reset='Q1 raw base as v2',gate='validation Brier improves >1e-12, number correct does not fall, no refit after validation',
            min_train_disjoint=10,min_validation_disjoint=5,
            platt='same v2 unconstrained logit slope/intercept ridge .01',
            shrink='q=training-bank up frequency; p_new=q+alpha*(p-q), alpha clipped to [0,1], train Brier minimizer; no ranking inversion',
            calibration_bank='each cadence and method uses its own causal OOS probabilities; never borrow annual bank for quarterly model',
            controls=['same-window base_frequency','same-window market4','quarter calibration-train frequency','constant .5'],
            primary_comparisons=[['quarterly.vol.raw','annual.vol.raw'],['quarterly.vol.shrink','quarterly.vol.raw'],
                ['quarterly.vol.shrink','quarterly.frequency.raw'],['quarterly.vol.shrink','quarterly.market4.raw']],
            primary_metrics=['Brier','logloss','accuracy','balanced_accuracy','AUC','calibration_bins','year','state','seed'],
            bootstrap=dict(blocks=[20,40],draws=10000,seed=603987,holm='four declared H5 Brier contrasts'),
            old_interactions='keep additive, vol, joint-order and fixed-parent-order; no post-result head selection'),
        fast=dict(model='simple train-only clipped/scaled ridge logistic .01',h=1,cadence='annual',lookback_years=5,
            features=['own short OHLCV','own + CSI300 context','own + verified semiconductor index context','own + both'],
            matching='identical eligible training members and test dates for context comparisons; full-own-history diagnostic retained',
            inference='same-day close or older only; no present constituent backcast',minimum_train_rows=400,
            secondary='descriptive; no neural architecture search or selection from test results'),
        live=dict(directory='stock_daily_603986',mode='manual daily after close, genuine same-date records only',
            record_window_cst=['18:00','23:59:59'],labels='H1 and H5 close to close, calendar positions, no fills claimed',
            displayed_models=['v2 H1 annual vol','v2 H1 current quarterly vol','H5 quarterly raw/platt/shrink','H5 annual raw'],
            fixed_evaluation='no retrospective inserts, append-only hash chained events, overlapping H5 samples flagged',
            parameter_expiry='2026-09-30; block October predictions until a separately verified new parameter package exists',
            split_action_guard='compare reconstructed forward-adjusted closes to vendor hfq; unknown corporate actions block',
            no_promotion='prospective candidate family fixed before viewing new results'),
        history='all historical dates have already been examined: exploratory, not independent confirmation',
        exclusions=['no orders, account access, or modification of index app/ledger','no HMM/DTW/pretrained or unbounded parameter search'])
    save(ROOT/'protocol.json',protocol)
    preserved=read(V2/'results/preserved_files.json').copy()
    for p,s in read(V2/'results/delivery_manifest.json')['files'].items():
        assert sha(V2/p)==s,p;preserved[rel(V2/p)]=s
    preserved[rel(V2/'results/delivery_manifest.json')]=sha(V2/'results/delivery_manifest.json')
    save(OUT/'preserved_files.json',preserved)
    plan=[];f=data();obs,_=observations(f,5)
    for task in tasks():
        ids,lower=membership(obs,task['cutoff'],5)
        po,_=observations(f[f.date.le(task['cutoff'])].reset_index(drop=True),5);pi,_=membership(po,task['cutoff'],5)
        pd.testing.assert_frame_equal(obs.iloc[ids].reset_index(drop=True),po.iloc[pi].reset_index(drop=True),check_exact=True)
        plan.append(dict(**task,n=len(ids),lower=lower,last_maturity=obs.joint_completed.iloc[ids].max()))
    csv(OUT/'training_plan.csv',pd.DataFrame(plan))
    files=dict(read(V2/'results/freeze.json')['files'])
    for p in [ROOT/'experiment3.py',ROOT/'prepare3.py',ROOT/'train3.py',ROOT/'protocol.json',V2/'results/delivery_manifest.json']:
        files[rel(p)]=sha(p)
    save(OUT/'freeze.json',dict(created_utc=now(),files=files))
    save(OUT/'contract.json',dict(status='PASS',cutoffs=len(plan),neural_fits=27,prefix_memberships='exact',data_end=protocol['data_end']))
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()

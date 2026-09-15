"""Quarterly state offsets from a purged mature weekly bank."""
from common50 import *
from dataset50 import observations
from forecast49 import Engine as PriorEngine,inputs,validate_forecast

class Engine(PriorEngine):
    def __init__(self,package):
        self.p=package;self.legacy=modules(True)[0];self.models={};self.pipelines={}
        for m in package['models']:
            self.models[m['seed']]=self.legacy.load_model(m)[0]
            h=next(h for h in package['uniform'] if h['seed']==m['seed']);self.pipelines[m['seed']]=self.legacy.arrays(h)
    def annual_signal(self,frame,date):
        guard(self.p['annual_cutoff']==annual_for(date),'Wrong annual encoder for calibration signal')
        f,v=inputs(frame[frame.date.le(date)].reset_index(drop=True),date);r=self.legacy;obs=pd.DataFrame(dict(anchor=[len(f)-1]));mf=r.market(f,obs,[0]);g=r.gates(f,obs,[0]);values={k:r.torch.from_numpy(x).cuda() for k,x in v.items()}
        state=('negative' if mf[0,0]<0 else 'nonnegative')+('_low' if mf[0,1]<=self.p['volatility_median'] else '_high');rows=[]
        for seed in SEEDS:
            features,_=r.neural.extract_features(self.models[seed],values);xx=r.apply_pipeline(features,mf,g,self.pipelines[seed])
            for method,x in zip(METHODS,xx):
                h=next(h for h in self.p['uniform'] if h['seed']==seed and h['method']==method);z=float(np.r_[x[0],1.]@np.asarray(h['coefficients']));p=float(r.probability(np.array([z]))[0])
                rows.append(dict(date=date,method=method,seed=seed,annual_logit=z,probability=p,state=state,encoder_cutoff=self.p['annual_cutoff']))
        return rows
def extend_bank(frame,cutoff,parent,packages,events):
    original=load_csv(checked_path(parent['calibration_bank'])) if 'calibration_bank' in parent else load_csv(PROJECT/'research_v39/results/weekly_signal_bank.csv')
    cols=['date','joint_completed','actual_up','method','seed','annual_logit','probability','state','encoder_cutoff']
    bank=original[cols].copy();bank=bank[bank.joint_completed.le(cutoff)].reset_index(drop=True)
    f,obs,_=observations(frame,cutoff);weekly=obs[obs.date.ge('2021-01-01')&obs.weekday.eq(4)];known=set(bank.date);added=[];engines={}
    predictions={e['key']:e for e in events if e['kind']=='prediction'}
    for row in weekly[~weekly.date.isin(known)].itertuples():
        if row.date in predictions:
            event=predictions[row.date];forecast=event['payload']['forecast'];validate_forecast(forecast)
            guard(event['recorded_utc'][:10]<=row.date and row.date< f.date.iloc[row.entry],'Late forecast in calibration bank')
            rows=[dict(date=row.date,method=r['method'],seed=r['seed'],annual_logit=r['logit'],probability=r['probability'],state=forecast['state'],encoder_cutoff=annual_for(row.date)) for r in forecast['seed_predictions'] if r['history']==U]
            source='committed_prospective_prediction'
        else:
            annual=annual_for(row.date);guard(annual in packages,'Missing historical annual package for calibration replay')
            if annual not in engines:engines[annual]=Engine(packages[annual])
            rows=engines[annual].annual_signal(f,row.date);source='calibration_only_replay_not_prospective'
        for r in rows:r.update(joint_completed=row.joint_completed,actual_up=int(row.exec_return>0),calibration_source=source)
        added.extend(rows)
    if added:bank=pd.concat([bank,pd.DataFrame(added)],ignore_index=True)
    bank=bank.sort_values(['date','method','seed']).reset_index(drop=True)
    guard(not bank.duplicated(['date','method','seed']).any(),'Duplicate calibration member')
    for date,g in bank.groupby('date'):
        guard(set(zip(g.method,g.seed))=={(m,s) for m in METHODS for s in SEEDS} and len(g)==12,'Incomplete calibration signal')
    return bank
def fit_quarter(bank,cutoff):
    _,_,q=modules();columns=['date','joint_completed'];pool=bank[columns].drop_duplicates().sort_values('date').reset_index(drop=True);guard(pool.date.is_unique,'Inconsistent bank maturity')
    train,val,schedule=q.split(pool,cutoff);heads=[];decisions=[];offsets=[]
    for method in METHODS:
        for state in STATES:
            probs=[];bases=[];labels=None;state_heads=[]
            for seed in SEEDS:
                g=bank[bank.method.eq(method)&bank.seed.eq(seed)]
                tr=g[g.date.isin(train.date)];va=g[g.date.isin(val.date)&g.state.eq(state)].sort_values('date')
                group=tr[tr.state.eq(state)];settings=q.cfg()['settings'];eligible=schedule['mode']=='ready' and len(group)>=settings['minimum_state_train']
                solved=q.v37.solve(group.annual_logit.to_numpy(),group.actual_up.to_numpy(float),settings['ridge_sum_lambda'],settings['absolute_logit_cap'],settings['bisection_iterations']) if eligible else dict(offset=0.,status=schedule['mode'] if schedule['mode']!='ready' else 'insufficient_state_train',n=len(group),up_n=int(group.actual_up.sum()))
                h=dict(family='state',component=state,fit_eligible=eligible,**solved)
                h=dict(h,method=method,seed=seed,cutoff=cutoff);heads.append(h);state_heads.append(h)
                p0=va.probability.to_numpy();probs.append(q.corrected_probability(va.annual_logit.to_numpy(),p0,h['offset']));bases.append(p0);labels=va.actual_up.to_numpy()
            decision=q.gate_decision(schedule['mode'],'state',state_heads[0]['n'],len(labels),np.mean(bases,axis=0),np.mean(probs,axis=0),labels)
            decisions.append(dict(method=method,state=state,cutoff=cutoff,training_n=state_heads[0]['n'],**decision))
            for h in state_heads:offsets.append(dict(method=method,seed=h['seed'],state=state,raw_offset=h['offset'],accepted=decision['accepted'],applied_offset=h['offset'] if decision['accepted'] else 0.,gate_reason=decision['reason']))
    return dict(schedule=schedule,train_dates=train.date.tolist(),validation_dates=val.date.tolist(),heads=heads,decisions=decisions,quarter_offsets=offsets)

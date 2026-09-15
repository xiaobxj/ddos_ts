"""Two-state, zero-mean Gaussian variance HMM; missing bars advance transitions."""
import numpy as np

def daily_returns(price):
    close=price.close.to_numpy(float);valid=price.valid_ohlc.to_numpy(bool)&np.isfinite(close)&(close>0)
    ok=valid[1:]&valid[:-1];r=np.full(len(close)-1,np.nan);r[ok]=np.log(close[1:][ok])-np.log(close[:-1][ok]);return r

def forward(y,variances,transition,initial=None,posterior=False):
    y=np.asarray(y,float);v=np.asarray(variances,float);p=np.asarray(transition,float);n=len(y);assert n>0 and np.all(v>0)
    loge=np.zeros((n,2));finite=np.isfinite(y);loge[finite]=-.5*(np.log(2*np.pi*v)[None,:]+np.square(y[finite,None])/v[None,:])
    offset=loge.max(axis=1);e=np.exp(loge-offset[:,None]);alpha=np.empty((n,2));pred=np.empty((n,2));scale=np.empty(n)
    prior=np.asarray([.5,.5] if initial is None else initial,float);assert np.all(prior>=0) and abs(prior.sum()-1)<1e-10
    p00,p01,p10,p11=map(float,p.ravel());a0,a1=map(float,prior)
    for t in range(n):
        if t:pr0=a0*p00+a1*p10;pr1=a0*p01+a1*p11
        else:pr0,pr1=a0,a1
        pred[t]=[pr0,pr1];w0=pr0*e[t,0];w1=pr1*e[t,1];den=w0+w1
        a0=w0/den;a1=w1/den;alpha[t]=[a0,a1];scale[t]=den
    out=dict(filtered=alpha,predicted=pred,log_likelihood=float(np.log(scale).sum()+offset.sum()))
    if posterior:
        beta=np.ones((n,2))
        for t in range(n-2,-1,-1):
            q0=e[t+1,0]*beta[t+1,0]/scale[t+1];q1=e[t+1,1]*beta[t+1,1]/scale[t+1]
            beta[t]=[p00*q0+p01*q1,p10*q0+p11*q1]
        gamma=alpha*beta;gamma/=gamma.sum(axis=1,keepdims=True)
        xi=np.einsum('ti,ij,tj->ij',alpha[:-1],p,e[1:]*beta[1:]/scale[1:,None])
        out.update(smoothed_training=gamma,transition_counts=xi)
    return out

def likelihood_gradient(y,variances,transition,posterior):
    y=np.asarray(y,float);finite=np.isfinite(y);g=posterior['smoothed_training'];xi=posterior['transition_counts'];p=np.asarray(transition)
    gv=.5*np.sum(g[finite]*(np.square(y[finite,None])/variances[None,:]-1),axis=0)
    gt=np.array([xi[0,0]*(1-p[0,0])-xi[0,1]*p[0,0],xi[1,1]*(1-p[1,1])-xi[1,0]*p[1,1]])
    return -np.r_[gv,gt]/len(y)

def fit_hmm(y,settings):
    y=np.asarray(y,float);finite=np.isfinite(y);s2=float(np.square(y[finite]).mean());lo,hi=settings['variance_bounds'];eps=settings['transition_floor']
    variances=np.clip(np.array(settings['initial_variance_multipliers'])*s2,lo,hi);stay=settings['initial_stay_probability'];p=np.array([[stay,1-stay],[1-stay,stay]])
    trace=[];last=-np.inf
    for iteration in range(settings['max_iterations']+1):
        z=forward(y,variances,p,posterior=True);ll=z['log_likelihood'];gradient=likelihood_gradient(y,variances,p,z);norm=float(np.max(np.abs(gradient)))
        assert ll>=last-settings['likelihood_roundoff_tolerance'],'EM likelihood decreased'
        trace.append(dict(iteration=iteration,log_likelihood=ll,mean_negative_log_likelihood=-ll/len(y),gradient_inf=norm,
            variance0=float(variances[0]),variance1=float(variances[1]),stay0=float(p[0,0]),stay1=float(p[1,1])))
        if norm<=settings['gradient_infinity_tolerance']:break
        assert iteration<settings['max_iterations'],'HMM EM iteration budget exhausted'
        gamma=z['smoothed_training'];xi=z['transition_counts'];variances=np.clip(np.sum(gamma[finite]*np.square(y[finite,None]),axis=0)/gamma[finite].sum(axis=0),lo,hi)
        proposed=xi/xi.sum(axis=1,keepdims=True);a=float(np.clip(proposed[0,0],eps,1-eps));b=float(np.clip(proposed[1,1],eps,1-eps));p=np.array([[a,1-a],[1-b,b]]);last=ll
    order=np.argsort(variances);variances=variances[order];p=p[np.ix_(order,order)];z=forward(y,variances,p,posterior=True)
    assert variances[1]/variances[0]>=settings['minimum_variance_ratio'] and z['smoothed_training'].sum(axis=0).min()>=settings['minimum_soft_occupancy']
    assert np.all(variances>lo) and np.all(variances<hi) and np.all(p>eps) and np.all(p<1-eps),'Boundary HMM requires separate review'
    return dict(variances=variances,transition=p,initial=np.array([.5,.5]),**z),trace

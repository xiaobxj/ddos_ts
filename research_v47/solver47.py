"""Fixed-design weighted convex logistic and conditional offset solvers."""
import numpy as np
import pandas as pd
def probability(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))
def design(x):return np.column_stack([np.asarray(x,float),np.ones(len(x))])
def weights_for(dates,cutoff):
    age=(pd.Timestamp(cutoff)-pd.to_datetime(dates)).dt.total_seconds().to_numpy(float)/86400
    assert len(age)>0 and (age>=0).all()
    raw=np.exp2(-age/730.5);w=raw/raw.sum();assert (w>0).all() and np.isfinite(w).all()
    return raw,w,age
def objective(theta,a,y,w,penalty=.01,hessian=True):
    z=a@theta;p=probability(z);value=float(w@(np.logaddexp(0.,z)-y*z)+.5*penalty*(theta[:-1]@theta[:-1]))
    g=a.T@(w*(p-y));g[:-1]+=penalty*theta[:-1]
    if not hessian:return value,g
    h=a.T@((w*p*(1-p))[:,None]*a);h[:-1,:-1]+=penalty*np.eye(a.shape[1]-1)
    return value,g,h
def fit_newton(x,y,w,settings):
    a=design(x);y=np.asarray(y,float);w=np.asarray(w,float);assert (w>0).all() and abs(w.sum()-1)<1e-12
    rate=float(w@y);assert 0<rate<1;theta=np.zeros(a.shape[1]);theta[-1]=np.log(rate/(1-rate));trace=[];step=0.;backtracks=0
    for iteration in range(settings['max_iterations']+1):
        val,g,h=objective(theta,a,y,w,settings['l2_lambda']);norm=float(abs(g).max());trace.append(dict(iteration=iteration,objective=val,gradient_inf=norm,last_step=step,last_backtracks=backtracks))
        if norm<=settings['gradient_infinity_tolerance']:return theta,trace
        assert iteration<settings['max_iterations'],'Weighted Newton budget exhausted';direction=np.linalg.solve(h,g);descent=float(g@direction);assert descent>0
        for backtracks in range(settings['line_search_max_steps']):
            step=2.**(-backtracks);candidate=theta-step*direction;value,_=objective(candidate,a,y,w,settings['l2_lambda'],False)
            if value<=val-settings['armijo']*step*descent+settings['objective_roundoff_allowance']:theta=candidate;break
        else:raise AssertionError('Weighted Newton line search failed')
    raise AssertionError('Unreachable')
def offset_objective(gamma,offset,x,y,w,penalty=.01):
    z=offset+gamma*x;p=probability(z)
    return float(w@(np.logaddexp(0.,z)-y*z)+.5*penalty*gamma*gamma),float(w@(x*(p-y))+penalty*gamma),float(w@(x*x*p*(1-p))+penalty)
def fit_offset(offset,x,y,w,settings):
    gamma=0.;trace=[]
    for iteration in range(settings['max_iterations']+1):
        val,g,h=offset_objective(gamma,offset,x,y,w,settings['l2_lambda']);assert h>=settings['l2_lambda'];trace.append(dict(iteration=iteration,gamma=gamma,objective=val,gradient= g,hessian=h))
        if abs(g)<=settings['gradient_absolute_tolerance']:return gamma,trace
        step=g/h
        for backtrack in range(settings['line_search_max_steps']):
            scale=2.**(-backtrack);candidate=gamma-scale*step;trial=offset_objective(candidate,offset,x,y,w,settings['l2_lambda'])[0]
            if trial<=val-settings['armijo']*scale*g*step+settings['objective_roundoff_allowance']:gamma=float(candidate);break
        else:raise AssertionError('Weighted offset line search failed')
    raise AssertionError('Weighted offset Newton budget exhausted')

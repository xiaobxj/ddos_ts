"""Deterministic convex ridge-logistic probe, with analytic gradient and Hessian."""
import numpy as np

def probability(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))

def normalize_train(features,floor=1e-6):
    f=np.asarray(features,float);mean=f.mean(axis=0);sd=np.maximum(f.std(axis=0),floor)
    return (f-mean)/sd,mean,sd

def design(features):return np.column_stack([np.asarray(features,float),np.ones(len(features))])

def objective(theta,a,y,penalty=.01,hessian=True):
    z=a@theta;p=probability(z);n=len(y)
    nll=float(np.mean(np.logaddexp(0.,z)-y*z));ridge=.5*penalty*float(theta[:-1]@theta[:-1])
    gradient=a.T@(p-y)/n;gradient[:-1]+=penalty*theta[:-1]
    if not hessian:return nll+ridge,gradient
    h=a.T@((p*(1-p))[:,None]*a)/n
    h[:-1,:-1]+=penalty*np.eye(a.shape[1]-1)
    return nll+ridge,gradient,h

def fit_newton(features,y,settings):
    a=design(features);y=np.asarray(y,float);rate=float(y.mean());assert 0<rate<1
    theta=np.zeros(a.shape[1]);theta[-1]=np.log(rate/(1-rate));trace=[];step=0.;backtracks=0
    for iteration in range(settings['max_iterations']+1):
        value,gradient,hessian=objective(theta,a,y,settings['l2_lambda']);norm=float(np.max(np.abs(gradient)))
        trace.append(dict(iteration=iteration,objective=value,gradient_inf=norm,last_step=step,last_backtracks=backtracks))
        if norm<=settings['gradient_infinity_tolerance']:return theta,trace
        assert iteration<settings['max_iterations'],'Newton iteration budget exhausted'
        direction=np.linalg.solve(hessian,gradient);descent=float(gradient@direction);assert descent>0
        for backtracks in range(settings['line_search_max_steps']):
            step=2.**(-backtracks);candidate=theta-step*direction
            candidate_value,_=objective(candidate,a,y,settings['l2_lambda'],False)
            if candidate_value<=value-settings['armijo']*step*descent+settings['objective_roundoff_allowance']:
                theta=candidate;break
        else:raise RuntimeError('Newton line search failed')
    raise RuntimeError('Unreachable solver termination')

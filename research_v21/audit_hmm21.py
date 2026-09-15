"""Independent log-domain likelihood, posterior and derivative implementation."""
import numpy as np
from scipy.special import expit,logsumexp

def encode(v,p):return np.r_[np.log(v),np.log(np.array([p[0,0],p[1,1]])/np.array([p[0,1],p[1,0]]))]
def decode(theta):
    v=np.exp(theta[:2]);a,b=expit(theta[2:]);return v,np.array([[a,1-a],[1-b,b]])
def log_reference(y,v,p,initial=None,posterior=True):
    y=np.asarray(y,float);n=len(y);ok=np.isfinite(y);emission=np.zeros((n,2));emission[ok]=-.5*np.log(2*np.pi*v)-np.square(y[ok,None])/(2*v)
    logp=np.log(p);a=np.empty((n,2));a[0]=np.log([.5,.5] if initial is None else initial)+emission[0]
    for t in range(1,n):
        a[t,0]=emission[t,0]+np.logaddexp(a[t-1,0]+logp[0,0],a[t-1,1]+logp[1,0])
        a[t,1]=emission[t,1]+np.logaddexp(a[t-1,0]+logp[0,1],a[t-1,1]+logp[1,1])
    ll=float(np.logaddexp(a[-1,0],a[-1,1]));filtered=np.exp(a-np.logaddexp(a[:,0],a[:,1])[:,None]);out=dict(log_likelihood=ll,filtered=filtered)
    if posterior:
        b=np.zeros((n,2))
        for t in range(n-2,-1,-1):
            b[t,0]=np.logaddexp(logp[0,0]+emission[t+1,0]+b[t+1,0],logp[0,1]+emission[t+1,1]+b[t+1,1])
            b[t,1]=np.logaddexp(logp[1,0]+emission[t+1,0]+b[t+1,0],logp[1,1]+emission[t+1,1]+b[t+1,1])
        gamma=np.exp(a+b-ll);gamma/=gamma.sum(axis=1,keepdims=True)
        xi=np.exp(a[:-1,:,None]+logp[None,:,:]+emission[1:,None,:]+b[1:,None,:]-ll).sum(axis=0)
        out.update(smoothed_training=gamma,transition_counts=xi)
    return out
def independent_objective(theta,y):
    v,p=decode(theta);z=log_reference(y,v,p);ok=np.isfinite(y);g=z['smoothed_training'];xi=z['transition_counts']
    gv=.5*np.sum(g[ok]*(np.square(y[ok,None])/v-1),axis=0)
    gt=np.array([xi[0,0]-p[0,0]*xi[0].sum(),xi[1,1]-p[1,1]*xi[1].sum()])
    return -z['log_likelihood']/len(y),-np.r_[gv,gt]/len(y)

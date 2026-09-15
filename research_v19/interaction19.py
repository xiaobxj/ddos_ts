"""One train-residualized volatility by inherited representation-score product."""
import numpy as np

def fit_interaction(base,volatility,ray,sd_floor=1e-6,rcond=1e-12):
    x=np.asarray(base,float);v=np.asarray(volatility,float);ray=np.asarray(ray,float)
    assert len(v)==len(x) and len(ray)==25 and x.shape[1]>=25
    signal=x[:,:25]@ray;product=v*signal;a=np.column_stack([x,np.ones(len(x))])
    projection,_,rank,singular=np.linalg.lstsq(a,product,rcond=rcond)
    residual=product-a@projection;mean=float(residual.mean());raw_sd=float(residual.std());sd=max(raw_sd,sd_floor)
    interaction=(residual-mean)/sd
    augmented=np.column_stack([x,interaction]);rank_before=int(np.linalg.matrix_rank(a,tol=1e-8));rank_after=int(np.linalg.matrix_rank(np.column_stack([a,interaction]),tol=1e-8))
    return dict(standardized=augmented,volatility=v,signal=signal,product=product,ray=ray,projection=projection,residual=residual,
        residual_mean=np.asarray(mean),residual_raw_sd=np.asarray(raw_sd),residual_sd=np.asarray(sd),interaction=interaction,
        projection_rank=np.asarray(rank),singular_values=singular,base_design_rank=np.asarray(rank_before),augmented_design_rank=np.asarray(rank_after),
        training_orthogonality_inf=np.asarray(float(np.max(np.abs(a.T@interaction)/len(x)))))

def apply_interaction(base,volatility,transform):
    x=np.asarray(base,float);v=np.asarray(volatility,float);a=np.column_stack([x,np.ones(len(x))])
    signal=x[:,:25]@transform['ray'];product=v*signal;residual=product-a@transform['projection']
    h=(residual-float(transform['residual_mean']))/float(transform['residual_sd'])
    return np.column_stack([x,h]),dict(signal=signal,product=product,residual=residual,interaction=h)

from common5 import *
from architecture import SegmentedCrossformer
import torch
from torch import nn


class DiagnosticCrossformer(SegmentedCrossformer):
    def __init__(self,dims,variant,seed,dropout):
        params={k:v for k,v in cfg()['crossformer'].items() if k!='dropout_validation'}
        super().__init__(dims,dropout=dropout,**params)
        self.variant=variant
        # Added branch and nonzero head use private RNG scopes. Backbone and
        # subsequent training dropout retain the frozen baseline RNG stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed+100)
            self.auxiliary_head=nn.Linear(dims,5)
        if variant=='return_nonzero':
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed+101)
                nn.init.xavier_uniform_(self.return_head.weight,gain=.1)
                nn.init.zeros_(self.return_head.bias)

    def outputs(self,patches,geometry,valid,auxiliary=False):
        features=self.decoded_features(patches,geometry,valid)
        returns=self.return_head(features.flatten(1)).squeeze(-1)
        return returns,self.auxiliary_head(features).flatten(1) if auxiliary else None


class CapacityMLP(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(125*dims,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,patches,geometry,valid):
        return self.net(patches).squeeze(-1)


def make_model(representation,variant,seed,capacity=False):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dims=5 if representation=='raw' else 20
    if variant=='mlp':return CapacityMLP(dims).cuda()
    return DiagnosticCrossformer(dims,variant,seed,0. if capacity else .1).cuda()


def batch_tensors(representation,indices):
    arrays=np.load(CACHE/f'packed_{representation}.npz')
    return {key:torch.from_numpy(arrays[key][indices]).cuda() for key in ['patches','geometry','valid']}


def standardized_targets(indices):
    data=np.load(CACHE/'targets.npz');result={};scales={}
    for key in ['returns','auxiliary']:
        y=data[key][indices].astype(float);mean=y.mean(axis=0);sd=np.maximum(y.std(axis=0),1e-6)
        result[key]=torch.tensor((y-mean)/sd,dtype=torch.float32,device='cuda')
        scales[key+'_mean']=np.asarray(mean).tolist();scales[key+'_sd']=np.asarray(sd).tolist()
    return result,scales


def predict(network,values,auxiliary=False):
    if isinstance(network,CapacityMLP):
        return network(values['patches'],values['geometry'],values['valid']),None
    return network.outputs(values['patches'],values['geometry'],values['valid'],auxiliary)


def grad_norm(parameters):
    norms=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.

from common6 import *
from architecture import SegmentedCrossformer
import torch
from torch import nn


class RegularizedCrossformer(SegmentedCrossformer):
    def __init__(self, arm, seed):
        settings = cfg()
        super().__init__(**settings['architecture'],
                         **{k: arm[k] for k in ['d_model', 'd_ff', 'dropout']})
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed + 100)
            self.auxiliary_head = nn.Linear(5, 5)

    def outputs(self, patches, geometry, valid, auxiliary=False):
        features = self.decoded_features(patches, geometry, valid)
        returns = self.return_head(features.flatten(1)).squeeze(-1)
        return returns, self.auxiliary_head(features).flatten(1) if auxiliary else None


def make_model(arm_name, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    arm = next(x for x in cfg()['arms'] if x['name'] == arm_name)
    return RegularizedCrossformer(arm, seed).cuda()


def batch_tensors(indices):
    with np.load(V5 / 'cache/packed_raw.npz') as arrays:
        return {key: torch.from_numpy(arrays[key][indices]).cuda() for key in ['patches', 'geometry', 'valid']}


def standardized_targets(indices):
    result = {}
    scales = {}
    with np.load(V5 / 'cache/targets.npz') as data:
        for key in ['returns', 'auxiliary']:
            y = data[key][indices].astype(float)
            mean = y.mean(axis=0)
            sd = np.maximum(y.std(axis=0), 1e-6)
            result[key] = torch.tensor((y - mean) / sd, dtype=torch.float32, device='cuda')
            scales[key + '_mean'] = np.asarray(mean).tolist()
            scales[key + '_sd'] = np.asarray(sd).tolist()
    return result, scales


def predict(network, values, auxiliary=False):
    return network.outputs(values['patches'], values['geometry'], values['valid'], auxiliary)


def evaluate_training(network, values, labels, batch_size=128):
    network.eval()
    return_sum = 0.
    aux_sum = 0.
    count = len(labels['returns'])
    with torch.inference_mode():
        for lo in range(0, count, batch_size):
            v = {k: t[lo:lo + batch_size] for k, t in values.items()}
            out, aux = predict(network, v, True)
            return_sum += float(((out - labels['returns'][lo:lo + batch_size]) ** 2).sum())
            aux_sum += float(((aux - labels['auxiliary'][lo:lo + batch_size]) ** 2).mean(dim=1).sum())
    return return_sum / count, aux_sum / count

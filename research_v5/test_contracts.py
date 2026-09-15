"""Necessary contract checks for the new objective and representation interface."""
import unittest
from models5 import *


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):initialize()

    def test_unused_auxiliary_preserves_frozen_model_and_rng(self):
        seed=20260910;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        baseline=SegmentedCrossformer(5).cuda().eval()
        cpu_before=torch.get_rng_state().clone();gpu_before=torch.cuda.get_rng_state().clone()
        new=make_model('raw','return_zero',seed).eval()
        torch.testing.assert_close(torch.get_rng_state(),cpu_before,rtol=0,atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),gpu_before,rtol=0,atol=0)
        for name,tensor in baseline.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name],tensor,rtol=0,atol=0)
        values=batch_tensors('raw',np.array([0,500,1000]))
        with torch.no_grad():
            a=baseline.decoded_features(**values);b=new.decoded_features(**values)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_auxiliary_objective_connects_backbone_at_zero_readout(self):
        indices=np.load(CACHE/'capacity_indices.npy');values=batch_tensors('raw',indices)
        labels,_=standardized_targets(indices)
        pure=make_model('raw','return_zero',20260910,capacity=True)
        out,_=predict(pure,values,False);((out-labels['returns'])**2).mean().backward()
        self.assertEqual(grad_norm(pure.backbone.parameters()),0.)
        joint=make_model('raw','joint_ohlcv',20260910,capacity=True)
        out,aux=predict(joint,values,True)
        (((out-labels['returns'])**2).mean()+.1*((aux-labels['auxiliary'])**2).mean()).backward()
        self.assertGreater(grad_norm(joint.backbone.parameters()),1e-6)

    def test_same_dimension_and_prequant_signs(self):
        arrays=np.load(CACHE/'representations.npz')
        self.assertEqual(arrays['bits'].shape,arrays['continuous'].shape)
        np.testing.assert_array_equal(arrays['bits']>0,arrays['continuous']>0)
        np.testing.assert_allclose(np.linalg.norm(arrays['continuous'],axis=-1),1,rtol=0,atol=1e-6)

    def test_shared_joint_maturity_and_future_input_boundary(self):
        obs=pd.read_csv(OUT/'observation_table.csv')
        self.assertTrue((obs.joint_completed>=obs.completed).all())
        self.assertTrue((obs.joint_completed>=obs.auxiliary_completed).all())
        import importlib.util
        spec=importlib.util.spec_from_file_location('v5_preparation_for_test',ROOT/'prepare.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        auxiliary_labels=module.auxiliary_labels
        frame=pd.read_csv(V1/'data/1_000300.csv');anchor=1000
        before=auxiliary_labels(frame,anchor)
        frame.loc[anchor+6:,['open','high','low','close','volume']]*=1000
        np.testing.assert_array_equal(before,auxiliary_labels(frame,anchor))
        prior=np.load(V4/'cache/window_representations.npz')['raw'][obs.v4_row.to_numpy(int)]
        np.testing.assert_array_equal(np.load(CACHE/'representations.npz')['raw'],prior)


if __name__=='__main__':unittest.main(verbosity=2)

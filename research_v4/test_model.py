import unittest
from architecture import *


class MaskContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def model_input(self):
        torch.manual_seed(19423)
        model=SegmentedCrossformer(5,dropout=0.).eval()
        nn.init.normal_(model.return_head.weight,std=.1)
        return model,torch.randn(3,25,5,5),torch.rand(3,25,2)

    def test_all_valid_matches_official_crossformer(self):
        model,values,geometry=self.model_input()
        nn.init.zeros_(model.geometry.weight);nn.init.zeros_(model.geometry.bias)
        valid=torch.ones((3,25),dtype=torch.bool)
        with torch.no_grad():
            ours=model.decoded_features(values,geometry,valid)
            official=model.backbone(values.reshape(3,125,5))
        torch.testing.assert_close(ours,official,rtol=2e-6,atol=2e-6)

    def test_masked_value_and_geometry_perturbations_do_not_change_output(self):
        model,values,geometry=self.model_input()
        valid=torch.ones((3,25),dtype=torch.bool);valid[:,:13]=False
        with torch.no_grad():
            original=model(values,geometry,valid)
            values[~valid]=1e6;geometry[~valid]=-1e6
            changed=model(values,geometry,valid)
        torch.testing.assert_close(original,changed,rtol=0,atol=0)

    def test_invalid_inputs_have_zero_gradient(self):
        model,values,geometry=self.model_input()
        values.requires_grad_();geometry.requires_grad_()
        valid=torch.ones((3,25),dtype=torch.bool);valid[:,:17]=False
        model(values,geometry,valid).sum().backward()
        self.assertEqual(float(values.grad[~valid].abs().max()),0.)
        self.assertEqual(float(geometry.grad[~valid].abs().max()),0.)
        self.assertGreater(float(values.grad[valid].abs().max()),0.)

    def test_one_valid_patch_survives_all_scales(self):
        model,values,geometry=self.model_input()
        valid=torch.zeros((3,25),dtype=torch.bool);valid[:,-1]=True
        output=model(values,geometry,valid)
        self.assertTrue(bool(torch.isfinite(output).all()))
        with self.assertRaises(ValueError):
            model(values,geometry,torch.zeros_like(valid))

    def test_training_can_fit_small_synthetic_batch(self):
        model,values,geometry=self.model_input()
        model.train()
        valid=torch.ones((3,25),dtype=torch.bool)
        target=torch.tensor([-.8,.2,1.1])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
        start=float(((model(values,geometry,valid)-target)**2).mean().detach())
        for i in range(35):
            optimizer.zero_grad(set_to_none=True)
            loss=((model(values,geometry,valid)-target)**2).mean()
            loss.backward();optimizer.step()
        finish=float(((model(values,geometry,valid)-target)**2).mean().detach())
        self.assertLess(finish,start*.3)


if __name__=='__main__':
    unittest.main(verbosity=2)

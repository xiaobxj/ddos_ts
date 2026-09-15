import unittest
import numpy as np
import pandas as pd
from core import V1,observation_table,economic_window,multi_features,adaptive_features,fold_masks,ridge_path,logistic_path,prototype_features


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=pd.read_csv(V1/'data/1_000300.csv')
        cls.obs=observation_table(cls.frame)

    def test_friday_execution_matches_actual_next_rebalance(self):
        f=self.frame
        ids=np.flatnonzero(pd.to_datetime(f.date).dt.dayofweek.to_numpy()==4)
        next_fri=dict(zip(ids[:-1],ids[1:]))
        for row in self.obs[self.obs.weekday==4].itertuples():
            self.assertEqual(row.exit,next_fri[row.anchor]+1)
            self.assertAlmostEqual(row.exec_return,f.open.iloc[row.exit]/f.open.iloc[row.anchor+1]-1)

    def test_holiday_label_must_be_completed_before_refit(self):
        row=self.obs[self.obs.date=='2020-12-25'].iloc[0]
        self.assertEqual(row.exit_date,'2021-01-11')
        tr,_=fold_masks(self.obs,'2020-12-31','2021-03-31',0)
        self.assertFalse(tr[self.obs.index[self.obs.date=='2020-12-25'][0]])

    def test_memory_boundaries(self):
        tr,_=fold_masks(self.obs,'2023-03-31','2023-06-30',3)
        self.assertGreaterEqual(self.obs.date[tr].min(),'2020-04-01')
        self.assertLessEqual(self.obs.completed[tr].max(),'2023-03-31')

    def test_no_future_features(self):
        f=self.frame.copy()
        t=3000
        original=economic_window(f.iloc[t-124:t+1])
        f.loc[t+1:,['open','high','low','close','volume']]*=100.
        revised=economic_window(f.iloc[t-124:t+1])
        np.testing.assert_array_equal(original[0],revised[0])
        np.testing.assert_array_equal(multi_features(*original),multi_features(*revised))

    def test_volume_features_are_prefix_causal(self):
        window=self.frame.iloc[2000:2125].copy()
        before=economic_window(window)[0]
        window.iloc[90:,window.columns.get_loc('volume')]*=100
        after=economic_window(window)[0]
        np.testing.assert_array_equal(before[:90],after[:90])

    def test_feature_dimensions_and_adaptive_masks(self):
        channels,base=economic_window(self.frame.iloc[2000:2125])
        self.assertEqual(len(base),75)
        self.assertEqual(len(multi_features(channels,base)),255)
        out=adaptive_features(channels,base,np.array([5,20,25,25,25,25]))
        self.assertEqual(len(out),250)
        patch=out[75:].reshape(25,7)
        self.assertEqual(patch[:,-1].sum(),6)
        self.assertAlmostEqual(patch[:,4].sum(),1.)
        self.assertEqual(patch[-1,5],1.)
        self.assertTrue(np.isfinite(out).all())

    def test_ridge_matches_direct_normal_equations(self):
        rng=np.random.default_rng(51)
        x=rng.normal(size=(50,7)); y=rng.normal(size=50); xt=rng.normal(size=(9,7))
        out=ridge_path(x,y,xt,[.1])['0.1']
        z=(x-x.mean(0))/x.std(0); zt=(xt-x.mean(0))/x.std(0)
        expected=zt@np.linalg.solve(z.T@z+len(x)*.1*np.eye(7),z.T@(y-y.mean()))+y.mean()
        np.testing.assert_allclose(out,expected,atol=1e-12)

    def test_logistic_learns_known_signal(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(250,3)); y=np.where(x[:,0]>0,.02,-.02)
        p,audit=logistic_path(x,y,x,[.01])
        self.assertGreater(((p['0.01']>.5)==(y>0)).mean(),.95)
        self.assertTrue(all(a['success'] for a in audit))


if __name__=='__main__':
    unittest.main(verbosity=2)

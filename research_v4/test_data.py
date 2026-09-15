import unittest
from data import *


class DataContracts(unittest.TestCase):
    def test_future_changes_cannot_change_observed_transforms(self):
        frame=pd.read_csv(V1/'data/1_000300.csv')
        t=3400
        before=observed_window(frame,t)
        frame.loc[t+1:,['open','high','low','close','volume']]*=1000
        after=observed_window(frame,t)
        np.testing.assert_array_equal(raw_transform(before),raw_transform(after))
        np.testing.assert_array_equal(tokenizer_transform(before),tokenizer_transform(after))
        self.assertTrue((tokenizer_transform(after)[:,-1]==0).all())

    def test_fixed_packing_preserves_every_input_value(self):
        x=np.random.default_rng(7).normal(size=(3,125,5)).astype(np.float32)
        packed=pack(x,np.full((3,25),5,dtype=np.int16))
        np.testing.assert_array_equal(packed['patches'].reshape(x.shape),x)
        self.assertTrue(packed['valid'].all())
        np.testing.assert_allclose(packed['geometry'][0,:,1],np.arange(1,26)/25,atol=1e-7)

    def test_permutation_preserves_length_multiset_and_masks(self):
        for year in [2024,2025,2026]:
            lengths=np.load(CACHE/f'lengths_{year}.npz')
            np.testing.assert_array_equal(np.sort(lengths['adaptive'],axis=1),np.sort(lengths['permuted'],axis=1))
            self.assertTrue((lengths['adaptive'].sum(axis=1)==125).all())
        arms={a['name']:a for a in cfg()['arms']}
        for prefix in ['raw','kronos']:
            a=np.load(packed_path(arms[prefix+'_adaptive'],2024));b=np.load(packed_path(arms[prefix+'_permuted'],2024))
            np.testing.assert_array_equal(a['valid'],b['valid'])
            self.assertTrue((a['patches'][~a['valid']]==0).all())
            self.assertTrue((a['geometry'][~a['valid']]==0).all())
            np.testing.assert_array_equal(a['geometry'][:,-1,1],np.ones(len(a['valid']),dtype=np.float32))

    def test_tokenizer_coordinates_are_scaled_bits_not_numeric_ids(self):
        windows=np.load(CACHE/'window_representations.npz')
        scale=np.float32(1/np.sqrt(20))
        for name in ['kronos','random_tokenizer']:
            a=windows[name]
            self.assertEqual(a.shape[1:],(125,20))
            np.testing.assert_array_equal(np.abs(a),np.full_like(a,scale))
        self.assertFalse(np.array_equal(windows['kronos'],windows['random_tokenizer']))


if __name__=='__main__':
    unittest.main(verbosity=2)

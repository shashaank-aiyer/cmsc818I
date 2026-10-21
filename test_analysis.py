import unittest
import warnings
import numpy as np
from analyze_fixed import summarize, record
from experiment import METHODS


class PairedArithmeticTests(unittest.TestCase):
    def fixture(self, good_counts):
        n = len(good_counts)
        novelty = np.zeros((n, len(METHODS), 300), dtype=np.uint8)
        novelty[:, :, :2] = 1
        good = METHODS.index('good_0.5')
        novelty[:, good] = 0
        for i, count in enumerate(good_counts):
            novelty[i, good, :count] = 1
        data = {'novelty': novelty, 'actions': np.zeros_like(novelty),
                'outcomes': np.full_like(novelty, -1, dtype=np.int16),
                'exact_uniform': np.full((n, 300), 2.)}
        meta = {'family': 'canonical', 'overlap': 0,
                'runs': [{'environment': 0, 'replicate': i, 'order': [0, 1, 2]} for i in range(n)]}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result, _ = summarize(data, meta)
        return record(result)

    def test_negative_difference_stays_negative(self):
        r = self.fixture([1, 1, 1, 1])
        self.assertEqual(r['mean'], 1.)
        self.assertEqual(r['delta_uniform'], -1.)
        self.assertEqual(r['delta_ci'], [-1., -1.])
        self.assertEqual(r['p_unadjusted'], 0.)

    def test_mixed_sign_differences_cancel(self):
        r = self.fixture([1, 3, 1, 3])
        self.assertEqual(r['delta_uniform'], 0.)
        self.assertTrue(r['delta_ci'][0] < 0 < r['delta_ci'][1])
        self.assertEqual(r['p_unadjusted'], 1.)


if __name__ == '__main__':
    unittest.main()

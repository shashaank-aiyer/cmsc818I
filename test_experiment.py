import math
import unittest
import numpy as np
from experiment import Selector, distribution, uniform_exact, sample_streams, environment_specs, GOOD_C


class DiscoveryTests(unittest.TestCase):
    def test_equal_success_probabilities_and_identical_limit(self):
        for overlap in [0, .25, .5, .75, 1]:
            p = distribution([2, 20, 100], .6, 20, overlap, [2, 0, 1])
            np.testing.assert_allclose(p.sum(axis=1), [.6] * 3)
            self.assertTrue((p >= 0).all())
            if overlap == 1:
                np.testing.assert_array_equal(p[0], p[1])
                np.testing.assert_array_equal(p[0], p[2])

    def test_analytic_baselines(self):
        p = distribution([2, 20, 100], .6, 20, 0, [0, 1, 2])
        self.assertAlmostEqual(uniform_exact(p, 150)[-1], 43.51139279488123)
        self.assertAlmostEqual(sum(1 - (1 - p[2]) ** 150), 59.45309966170977)
        self.assertAlmostEqual(GOOD_C['good_theory'], (1 + math.sqrt(2)) * math.sqrt(3))

    def test_singletons_removed_across_arms(self):
        s = Selector('good_0.5', 3, 4, 1)
        self.assertEqual(s.update(0, 2), 1)
        self.assertEqual(s.singleton, [1, 0, 0])
        self.assertEqual(s.update(1, 2), 0)
        self.assertEqual(s.singleton, [0, 0, 0])
        s.update(2, 2)
        s.update(1, 3)
        s.update(1, -1)
        self.assertEqual(s.singleton, [0, 1, 0])
        self.assertEqual(s.n, [1, 3, 1])
        self.assertEqual(s.successes, [1, 2, 1])

    def test_equal_success_histories_different_diversity(self):
        s = Selector('good_0.5', 3, 30, 0)
        for i in range(10):
            s.update(0, 0 if i < 6 else -1)
            s.update(1, 1 + i % 2 if i < 6 else -1)
            s.update(2, 10 + i if i < 6 else -1)
        self.assertEqual(s.successes, [6, 6, 6])
        self.assertEqual(s.choose(31), 2)

    def test_singleton_expectation_is_next_missing_mass_shift(self):
        # For a single arm, E[F1/n] = sum p_j*(1-p_j)^(n-1).
        # Enumerate all length-3 histories on two outcomes plus no outcome.
        import itertools
        weights = [.2, .3, .5]
        estimate = 0
        for history in itertools.product(range(3), repeat=3):
            probability = math.prod(weights[x] for x in history)
            singles = sum(history.count(c) == 1 for c in [0, 1])
            estimate += probability * singles / 3
        self.assertAlmostEqual(estimate, .2 * .8 ** 2 + .3 * .7 ** 2)

    def test_sampling_support_and_success_rate(self):
        p = distribution([2, 20, 100], .6, 20, .5, [0, 1, 2])
        draws = sample_streams(p, 100000, 1234)
        for a in range(3):
            self.assertLess(abs(np.mean(draws[a] >= 0) - .6), .006)
            self.assertTrue(all(p[a, x] > 0 for x in np.unique(draws[a]) if x >= 0))
        np.testing.assert_array_equal(draws, sample_streams(p, 100000, 1234))

    def test_oracle_requires_information_and_uses_remaining_mass(self):
        s = Selector('oracle_greedy', 3, 5, 0)
        with self.assertRaises(ValueError):
            s.choose(1)
        self.assertEqual(s.choose(1, [.1, .5, .2]), 1)

    def test_all_generated_environments_valid_and_reproducible(self):
        envs = environment_specs('generated')
        self.assertEqual(len(envs), 40)
        self.assertEqual(envs, environment_specs('generated'))
        for e in envs:
            self.assertTrue(.45 <= e['q'] <= .75)
            self.assertTrue(2 <= e['sizes'][0] <= 5)
            self.assertTrue(15 <= e['sizes'][1] <= 30)
            self.assertTrue(80 <= e['sizes'][2] <= 140)


if __name__ == '__main__':
    unittest.main()

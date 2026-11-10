"""The 16-input criteria reuse the frozen v1 budgets unchanged."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import parity_criteria as v1
import parity_criteria_v2 as v2


class CriteriaV2Tests(unittest.TestCase):
    def setUp(self):
        self.v1_features = np.array([0., 5., -10., 10., 20., 5., 0., 0., 12500., 1., 1.171875, 20., 10., 2.])
        self.features = np.concatenate([self.v1_features, [1.05, 0.03]])

    def test_contract_is_the_frozen_v1_file(self):
        self.assertEqual(v2.CONTRACT_SHA256, v1.CONTRACT_SHA256)
        self.assertEqual(v2.CRITERIA_DESCRIPTION["budgets_from_v1_contract"],
                         {"feature_scale_factor": 1e-4, "output_spacings": 8, "probability_absolute": 1e-4,
                          "attribution_absolute": 1e-5, "ranking_gap": 2e-5, "threshold": .5, "debounce": 2})
        self.assertEqual(v2.LOCAL_SCALES[:14], v1.CONTRACT["local_scales"])
        self.assertEqual(v2.NAMES[14:], ["ratio_of_ratios", "ir_perfusion_index"])

    def test_first_14_budgets_match_v1_exactly(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            expected = self.v1_features * rng.uniform(.1, 10, 14)
            expected[5] = abs(expected[5])
            actual = expected + rng.normal(0, 1e-3, 14)
            old = v1.feature_check(expected, actual)
            new = v2.feature_check(np.concatenate([expected, [1.05, .03]]), np.concatenate([actual, [1.05, .03]]))
            self.assertEqual(new["budgets"][:14], old["budgets"])
            self.assertEqual(new["absolute_errors"][:14], old["absolute_errors"])

    def test_new_feature_scales(self):
        result = v2.feature_check(self.features, self.features)
        spacing = lambda x: float(abs(np.spacing(np.float32(x))))
        self.assertAlmostEqual(result["budgets"][14], 1e-4 * 1.05 + 8 * spacing(1.05), places=15)
        self.assertAlmostEqual(result["budgets"][15], 1e-4 * .03 + 8 * spacing(.03), places=15)
        small = self.features.copy()
        small[14] = .5
        self.assertAlmostEqual(v2.feature_check(small, small)["budgets"][14], 1e-4 + 8 * spacing(.5), places=15)
        actual = self.features.copy()
        actual[15] += 4e-6
        self.assertFalse(v2.feature_check(self.features, actual)["pass"])

    def test_all_16_deltas_and_ranking(self):
        reference = np.array([.3, .2, .1] + [0.] * 13)
        actual = reference.copy()
        actual[15] = 1.1e-5
        self.assertFalse(v2.attribution_check(reference, actual)["pass"])
        reference = np.array([.3, .2] + [0.] * 12 + [.10003, .1])
        actual = reference.copy()
        actual[14:] = actual[14:][::-1]
        self.assertIn([14, 15], v2.attribution_check(reference, actual)["rank_violations"])

    def test_bad_input_cannot_pass(self):
        for bad in (np.full(16, np.nan), np.zeros(14), np.zeros((1, 16))):
            with self.assertRaises(ValueError):
                v2.feature_check(self.features, bad)

    def test_candidate_must_agree_exactly(self):
        deltas = np.zeros(16)
        self.assertFalse(v2.check_window(self.features, self.features, .5, .49999, deltas, deltas)["pass"])
        self.assertTrue(v2.check_window(self.features, self.features, .5, .50001, deltas, deltas)["pass"])


if __name__ == "__main__":
    unittest.main()

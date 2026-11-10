"""Contract boundaries, all-delta checks, and ranking across selection boundaries."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parity_criteria import CONTRACT, feature_check, attribution_check, check_window


class CriteriaTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([0., 5., -10., 10., 20., 5., 0., 0., 12500., 1., 1.171875, 20., 10., 2.])

    def test_frozen_budgets(self):
        self.assertEqual([CONTRACT[k] for k in ("feature_scale_factor", "output_spacings", "probability_absolute",
                                               "attribution_absolute", "ranking_gap", "threshold", "debounce")],
                         [1e-4, 8, 1e-4, 1e-5, 2e-5, .5, 2])

    def test_mean_uses_signal_scale_and_retains_original_error(self):
        actual = self.features.copy()
        actual[0] = .0004
        result = feature_check(self.features, actual)
        self.assertTrue(result["pass"])
        self.assertFalse(result["original_absolute_1e_4_pass"])
        self.assertEqual(result["absolute_errors"][0], .0004)
        actual[0] = .0006
        self.assertFalse(feature_check(self.features, actual)["pass"])

    def test_frequency_requires_exact_bin(self):
        actual = self.features.copy()
        actual[10] += 1e-8
        self.assertFalse(feature_check(self.features, actual)["pass"])

    def test_bad_input_cannot_pass(self):
        for bad in (np.full(14, np.nan), np.full(14, np.inf), np.zeros(13), np.zeros((1, 14))):
            with self.assertRaises(ValueError):
                feature_check(self.features, bad)

    def test_untransmitted_delta_is_still_checked(self):
        reference = np.array([.3, .2, .1] + [0.] * 11)
        actual = reference.copy()
        actual[13] = 1.1e-5
        result = attribution_check(reference, actual)
        self.assertFalse(result["pass"])
        self.assertFalse(result["top3_set_changed"])

    def test_close_order_swap_is_reported(self):
        reference = np.array([.11, .110005, .05] + [0.] * 11)
        actual = reference.copy()
        actual[:2] = actual[:2][::-1]
        result = attribution_check(reference, actual)
        self.assertTrue(result["pass"])
        self.assertTrue(result["top3_order_changed"])
        self.assertFalse(result["top3_set_changed"])

    def test_rank_check_includes_selection_boundary(self):
        reference = np.array([.3, .2, .10003, .1] + [0.] * 10)
        actual = reference.copy()
        actual[2:4] = actual[2:4][::-1]
        self.assertIn([2, 3], attribution_check(reference, actual)["rank_violations"])

    def test_small_probability_error_does_not_excuse_crossing(self):
        result = check_window(self.features, self.features, .499999, .500001, np.zeros(14), np.zeros(14))
        self.assertTrue(result["probability_pass"])
        self.assertFalse(result["candidate_agrees"])
        self.assertFalse(result["pass"])

    def test_amplitude_unit_rescaling(self):
        scale = np.array([2., 2., 2., 2., 2., 2., 1., 1., 4., 2., 1., 4., 4., 1.])
        actual = self.features.copy()
        actual[[0, 8, 11, 13]] += [.0006, 4., .01, .001]
        a = feature_check(self.features, actual)
        b = feature_check(self.features * scale, actual * scale)
        self.assertEqual(a["failed_features"], b["failed_features"])


if __name__ == "__main__":
    unittest.main()

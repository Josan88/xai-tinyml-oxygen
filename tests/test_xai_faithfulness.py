"""Analytic checks of masking, ranking, metric definitions, and frozen inference."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_xai_faithfulness import (FrozenKAN, NAMES, curve_metrics, load_features,
                                       mean_perturbation, perturbation_curves, ranked_orders)
from verify_features import numpy_kan_predict


class FaithfulnessTests(unittest.TestCase):
    def test_signed_and_firmware_rankings_are_distinct_and_stable(self):
        attribution = np.array([[.1, -.8, .1, 0.]])
        order = ranked_orders(attribution, random_repeats=2)
        np.testing.assert_array_equal(order["signed_risk"], [[0, 2, 3, 1]])
        np.testing.assert_array_equal(order["firmware_absolute"], [[1, 0, 2, 3]])
        np.testing.assert_array_equal(order["reverse_signed"], [[1, 3, 2, 0]])

    def test_random_controls_repeat_and_cover_all_features(self):
        a = np.zeros((30, 14))
        first = ranked_orders(a, seed=19, random_repeats=2)
        again = ranked_orders(a, seed=19, random_repeats=2)
        for name in ("random_00", "random_01"):
            np.testing.assert_array_equal(first[name], again[name])
            np.testing.assert_array_equal(np.sort(first[name], axis=1), np.tile(np.arange(14), (30, 1)))
        self.assertFalse(np.array_equal(first["random_00"], first["random_01"]))

    def test_known_linear_probability_curve_and_signed_negative_contribution(self):
        predict = lambda x: .5 + x @ np.array([.2, -.1, .05])
        scaled = np.ones((1, 3))
        before = scaled.copy()
        probability, attribution = mean_perturbation(predict, scaled)
        np.testing.assert_allclose(probability, [.65])
        np.testing.assert_allclose(attribution, [[.2, -.1, .05]])
        orders = ranked_orders(attribution, random_repeats=1)
        deletion, insertion = perturbation_curves(predict, scaled, orders["signed_risk"])
        np.testing.assert_allclose(deletion, [[.65, .45, .40, .50]])
        np.testing.assert_allclose(insertion, [[.50, .70, .75, .65]])
        metrics = curve_metrics(deletion, insertion)
        self.assertAlmostEqual(metrics["deletion_aopc"][0], .20)
        self.assertAlmostEqual(metrics["insertion_aopc"][0], .20)
        self.assertAlmostEqual(metrics["deletion_auc"][0], .475)
        self.assertAlmostEqual(metrics["insertion_auc"][0], .675)
        # Discrete AOPC differs from baseline minus trapezoidal probability AUC.
        self.assertNotAlmostEqual(metrics["deletion_aopc"][0], .65 - metrics["deletion_auc"][0])
        reversed_deletion, _ = perturbation_curves(predict, scaled, orders["reverse_signed"])
        self.assertGreater(metrics["deletion_aopc"][0], curve_metrics(reversed_deletion, insertion)["deletion_aopc"][0])
        np.testing.assert_array_equal(scaled, before)

    def test_nonlinear_interaction_is_not_assumed_additive_or_re_ranked(self):
        predict = lambda x: .2 + .1 * x[:, 0] + .05 * x[:, 1] + .2 * x[:, 0] * x[:, 1]
        x = np.ones((1, 2))
        _, attribution = mean_perturbation(predict, x)
        deletion, insertion = perturbation_curves(predict, x, ranked_orders(attribution, random_repeats=1)["signed_risk"])
        np.testing.assert_allclose(attribution, [[.3, .25]])
        np.testing.assert_allclose(deletion, [[.55, .25, .2]])
        np.testing.assert_allclose(insertion, [[.2, .3, .55]])
        self.assertNotAlmostEqual(attribution.sum(), deletion[0, 0] - deletion[0, -1])

    def test_masking_endpoints_and_zero_input(self):
        predict = lambda x: 1 / (1 + np.exp(-x.sum(axis=1)))
        x = np.array([[1., -2.], [0., 0.]])
        deletion, insertion = perturbation_curves(predict, x, np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(deletion[:, 0], predict(x))
        np.testing.assert_allclose(insertion[:, -1], predict(x))
        np.testing.assert_allclose(deletion[:, -1], .5)
        np.testing.assert_allclose(insertion[:, 0], .5)
        self.assertEqual(curve_metrics(deletion, insertion)["deletion_aopc"][1], 0.)
        with self.assertRaises(ValueError):
            perturbation_curves(predict, x, np.array([[0, 0], [0, 1]]))

    def test_vectorized_frozen_model_matches_existing_independent_reference(self):
        model = FrozenKAN()
        p = model.parameters
        scaled = np.random.default_rng(82).uniform(-4, 4, (20, 14))
        raw = (scaled * p["scaler_std"] + p["scaler_mean"]).astype(np.float32)
        actual = model.predict(model.standardize(raw))
        expected = [numpy_kan_predict(row.astype(np.float64), p)[1] for row in raw]
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)

    def test_loader_rejects_wrong_count_and_invalid_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            fields = ["recording_id", "start_time_seconds", "end_time_seconds", "label", *NAMES]
            row = dict.fromkeys(NAMES, 0.)
            row.update(recording_id="recording_a", start_time_seconds=0, end_time_seconds=10, label=1)
            def write():
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
            write()
            raw, metadata = load_features(path, 1)
            self.assertEqual(raw.shape, (1, 14))
            self.assertEqual(metadata[0]["source_row"], 0)
            with self.assertRaises(ValueError):
                load_features(path, 2)
            row["mean"] = float("nan")
            write()
            with self.assertRaises(ValueError):
                load_features(path, 1)


if __name__ == "__main__":
    unittest.main()

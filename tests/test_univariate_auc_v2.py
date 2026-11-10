"""Host tests for scripts/univariate_auc_v2.py. No device, no port, no training."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import univariate_auc_v2 as u


class UnivariateAucTest(unittest.TestCase):
    def test_matches_pair_count(self):
        x = np.array([0.1, 0.4, 0.35, 0.8, 0.8, 0.2])
        y = np.array([0, 0, 1, 1, 0, 1])
        pos, neg = x[y == 1], x[y == 0]
        pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
        self.assertAlmostEqual(u.univariate_auc(x, y)["auc_raw"], float(np.mean(pairs)))

    def test_orientation(self):
        y = np.array([0, 0, 1, 1])
        lower = u.univariate_auc(np.array([4.0, 3.0, 2.0, 1.0]), y)
        self.assertEqual(lower["auc_raw"], 0.0)
        self.assertEqual(lower["auc_oriented"], 1.0)
        self.assertEqual(lower["direction"], "lower value -> label 1")
        higher = u.univariate_auc(np.array([1.0, 2.0, 3.0, 4.0]), y)
        self.assertEqual(higher["direction"], "higher value -> label 1")

    def test_nonfinite_dropped_and_counted(self):
        r = u.univariate_auc(np.array([1.0, np.nan, 3.0, np.inf, 2.0]), np.array([0, 1, 1, 0, 0]))
        self.assertEqual((r["rows_used"], r["rows_dropped_nonfinite"]), (3, 2))

    def test_one_class_rejected(self):
        with self.assertRaises(ValueError):
            u.univariate_auc(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(u.main(["--output", tmp]), 2)


if __name__ == "__main__":
    unittest.main()

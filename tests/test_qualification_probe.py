"""Independent synthetic checks of batch replay, reset, metadata and all deltas."""
import argparse
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qualify_numerical_parity import run_probe
from parity_criteria import check_window, NAMES
from check_probability_parity import canonical_feature_function, reference, HEADER
from verify_features import parse_c_header
from audit_feature_errors import fp32


class QualificationProbeTests(unittest.TestCase):
    def test_reset_and_context_decisions(self):
        raw = []
        counts = np.array([1000, 750, 1000])
        t = np.arange(1000) / 50.
        for frequency in (2.2, 1.2, 2.2):
            raw.append(np.asarray(2000 + 100 * np.sin(2 * np.pi * frequency * t) + 20 * np.sin(.6 * t), dtype=np.float32))
        actual = run_probe(PROBE, np.array(raw), counts)
        sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
        compute = canonical_feature_function()
        parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
        for i, n in enumerate(counts):
            offset, emissions = (0, 1) if n == 750 else (250, 2)
            filtered = signal.sosfiltfilt(sos, fp32(raw[i][:n]))[offset:offset + 500]
            f = compute(filtered, 50.)
            features = np.array([f[name] for name in NAMES])
            p, deltas = reference(features[None, :], parameters)
            result = check_window(features, actual[i, 9:23], p[0], actual[i, 8], deltas[0], actual[i, 32:46])
            self.assertTrue(result["pass"], result)
            np.testing.assert_array_equal(actual[i, 3:8], [emissions - 1, n, offset, offset + 500, 250])
            self.assertEqual(actual[i, 1], emissions)
            top = actual[i, 26:32].reshape(3, 2)
            indices = np.argsort(-np.abs(actual[i, 32:46]), kind="stable")[:3]
            np.testing.assert_array_equal(top[:, 0], indices)
            np.testing.assert_array_equal(top[:, 1], actual[i, 32:46][indices])
        np.testing.assert_array_equal(actual[0, 23:26], [1, 1, 2])
        np.testing.assert_array_equal(actual[1, 23:26], [0, 0, 0])
        np.testing.assert_array_equal(actual[2, 23:26], [1, 1, 2])

    def test_invalid_and_incomplete_input_rejected(self):
        for payload in ("749\n", "1000 1\n"):
            result = subprocess.run([str(PROBE)], input=payload, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    PROBE = args.probe.resolve()
    unittest.main(argv=[sys.argv[0]])

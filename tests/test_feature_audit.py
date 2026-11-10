"""Check audit alignment, unchanged numerical serialization, and rounding bounds."""
import argparse
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_feature_errors import feature_summary, fp32, run_audit_probe
from check_training_context import run_context_probe


class FeatureAuditTests(unittest.TestCase):
    def test_recording_offsets_match_streaming_ring_wrap(self):
        t = np.arange(1750) / 50.
        raw = np.asarray(100000 + 100 * np.sin(2 * np.pi * 1.2 * t) + 20 * np.cos(.6 * t), dtype=np.float32)
        stream = run_context_probe(CONTEXT_PROBE, raw)
        contexts = []
        for row in stream:
            start, end = int(row[2]), int(row[1])
            left = max(0, start - 250)
            contexts.append((raw[left:end], start - left))
        actual = run_audit_probe(AUDIT_PROBE, "raw", contexts)
        np.testing.assert_array_equal(actual[:, 500:], fp32(np.asarray(stream)[:, 6:]))
        repeated = run_audit_probe(AUDIT_PROBE, "filtered", actual[:, :500])
        np.testing.assert_array_equal(repeated, actual)

    def test_known_filtered_constant_statistics(self):
        actual = run_audit_probe(AUDIT_PROBE, "filtered", [np.full(500, 2.)])[0, 500:]
        # No filtering: mean 2, std 0, min=max=2, range 0, RMS 2, energy 2000.
        np.testing.assert_array_equal(actual[[0, 1, 2, 3, 4, 5, 8, 9]], [2, 0, 2, 2, 0, 2, 2000, 0])

    def test_fp32_lower_bound_exposes_impossible_absolute_gate(self):
        expected = np.full((1, 14), 2. ** 26 + .125)
        summary = feature_summary(expected, fp32(expected), np.ones(14))
        self.assertEqual(summary[8]["nearest_fp32_already_exceeds_1e_4"], 1)
        self.assertEqual(summary[8]["max_nearest_fp32_rounding_error"], .125)
        self.assertEqual(summary[8]["max_fp32_spacing"], 8.)

    def test_invalid_count_is_rejected_before_reading(self):
        result = subprocess.run([str(AUDIT_PROBE), "raw"], input="1001 250\n", text=True, capture_output=True)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-probe", required=True, type=Path)
    parser.add_argument("--context-probe", required=True, type=Path)
    args = parser.parse_args()
    AUDIT_PROBE, CONTEXT_PROBE = args.audit_probe.resolve(), args.context_probe.resolve()
    unittest.main(argv=[sys.argv[0]])

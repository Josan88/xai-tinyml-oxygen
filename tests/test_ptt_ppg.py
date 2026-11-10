"""Host-only PTT-PPG helpers: resample, windows, injected records, no training."""
from pathlib import Path
import hashlib
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_ptt_ppg import (ACTIVITIES, FEATURE_NAMES, HOP, THRESHOLD, WINDOW,
                              evaluate, iter_windows, load_record, parse_record_name,
                              resample_ppg_50hz, score_windows, summarize)


def _record(name, activity, fs=500.0, seconds=20.0, channel="pleth_2"):
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    ppg = 800.0 + 40.0 * np.sin(2 * np.pi * 1.2 * t)
    acc = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 1.0)])
    values = np.column_stack([ppg, acc])
    return types.SimpleNamespace(
        comments=[f"<filename>: {name} <activity>: {activity}"],
        sig_name=[channel, "a_x", "a_y", "a_z"],
        units=["NU", "g", "g", "g"],
        adc_gain=[1.0, 1.0, 1.0, 1.0],
        baseline=[0, 0, 0, 0],
        fs=fs,
        sig_len=n,
        p_signal=values,
    )


class PttPpgTests(unittest.TestCase):
    def test_record_names_match_published_1_1_0(self):
        self.assertEqual(parse_record_name("s1_sit"), ("s1", "sit"))
        self.assertEqual(parse_record_name("s22_run"), ("s22", "run"))
        with self.assertRaises(ValueError):
            parse_record_name("s0_sit")
        with self.assertRaises(ValueError):
            parse_record_name("s1_jog")

    def test_resample_500_to_50_length_and_rejects_gaps(self):
        x = np.sin(2 * np.pi * 1.2 * np.arange(5000) / 500.0)
        y = resample_ppg_50hz(x, 500.0)
        self.assertEqual(y.shape, (500,))
        self.assertTrue(np.isfinite(y).all())
        np.testing.assert_allclose(resample_ppg_50hz(y, 50.0), y)
        broken = x.copy()
        broken[10] = np.nan
        with self.assertRaises(ValueError):
            resample_ppg_50hz(broken, 500.0)

    def test_windows_are_500_samples_with_250_hop(self):
        samples = np.arange(1000, dtype=float)
        windows = list(iter_windows(samples))
        self.assertEqual([start for start, _ in windows], [0, 250, 500])
        self.assertTrue(all(len(window) == WINDOW for _, window in windows))
        np.testing.assert_array_equal(windows[1][1], samples[250:750])
        self.assertEqual(HOP, 250)

    def test_load_record_uses_named_ppg_and_keeps_imu_as_context(self):
        fake = _record("s1_walk", "walk")

        def reader(path, physical=True):
            self.assertTrue(physical)
            self.assertTrue(str(path).endswith("s1_walk"))
            return fake

        ppg, acceleration, audit = load_record(Path("s1_walk"), "pleth_2", reader=reader)
        self.assertEqual(audit["activity"], "walk")
        self.assertEqual(audit["ppg_channel"], "pleth_2")
        self.assertEqual(audit["ppg_missing_samples"], 0)
        self.assertEqual(audit["acceleration_channels"], ["a_x", "a_y", "a_z"])
        self.assertEqual(ppg.shape, (10000,))
        self.assertEqual(acceleration.shape, (10000, 3))
        fake.p_signal[3, 0] = np.nan
        with self.assertRaises(ValueError):
            load_record(Path("s1_walk"), "pleth_2", reader=reader)

    def test_score_windows_does_not_feed_imu_to_tinykan(self):
        from score_tinykan_explanation_rubric import parse_c_header
        header = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h"
        weights = parse_c_header(header)
        window = np.sin(2 * np.pi * 1.2 * np.arange(WINDOW) / 50.0).astype(np.float32)
        p, features, top3, reference = next(score_windows([window], weights))
        self.assertEqual(len(features), 14)
        self.assertEqual([name for name, _ in top3], [item[0] for item in top3])
        self.assertTrue(all(name in FEATURE_NAMES for name, _ in top3))
        self.assertEqual(p, reference)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertNotIn("imu_rms_header_units", FEATURE_NAMES)
        self.assertNotIn("a_x", FEATURE_NAMES)

    def test_summarize_candidate_fraction_uses_frozen_threshold(self):
        rows = [
            {"activity": "sit", "subject_id": "s1", "recording_id": "s1_sit",
             "p": 0.49, "candidate": 0, "top1_feature": "kurtosis"},
            {"activity": "sit", "subject_id": "s1", "recording_id": "s1_sit",
             "p": 0.50, "candidate": 1, "top1_feature": "std"},
            {"activity": "walk", "subject_id": "s1", "recording_id": "s1_walk",
             "p": 0.80, "candidate": 1, "top1_feature": "std"},
        ]
        summary = {row["activity"]: row for row in summarize(rows)}
        self.assertEqual(THRESHOLD, 0.50)
        self.assertEqual(summary["sit"]["windows"], 2)
        self.assertEqual(summary["sit"]["candidate_windows"], 1)
        self.assertEqual(summary["sit"]["candidate_fraction"], 0.5)
        self.assertEqual(summary["walk"]["candidate_fraction"], 1.0)
        self.assertEqual(set(ACTIVITIES), {"sit", "walk", "run"})

    def test_evaluate_injected_records_write_activity_rows_without_training(self):
        records = {f"s1_{activity}": _record(f"s1_{activity}", activity) for activity in ACTIVITIES}

        def reader(path, physical=True):
            return records[Path(path).name]

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            output = Path(directory) / "out"
            data_root.mkdir()
            checksum_lines = []
            for name in records:
                for suffix, payload in ((".hea", b"header"), (".dat", b"waveform")):
                    path = data_root / f"{name}{suffix}"
                    path.write_bytes(payload)
                    checksum_lines.append(f"{hashlib.sha256(payload).hexdigest()} {path.name}")
            (data_root / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
            report = evaluate(data_root, list(records), "pleth_2", output, reader=reader)
            self.assertEqual(report["threshold"], 0.50)
            self.assertIn("not a validation cohort", report["scope"])
            self.assertTrue(report["accuracy_metrics"].startswith("not computed"))
            self.assertTrue((output / "windows.csv").is_file())
            self.assertTrue((output / "activity_summary.csv").is_file())
            activities = {row["activity"] for row in report["summary"]}
            self.assertEqual(activities, set(ACTIVITIES))
            self.assertGreater(sum(row["windows"] for row in report["summary"]), 0)
            with self.assertRaises(FileExistsError):
                evaluate(data_root, list(records), "pleth_2", output, reader=reader)


if __name__ == "__main__":
    unittest.main()

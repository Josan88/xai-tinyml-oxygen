"""Independent SciPy check of the 5-second future-context pipeline.

Run: python tests/test_context_pipeline.py --probe build/parity/cmake/context_probe.exe
Uses synthetic PPG only; recorded-dataset comparison lives in check_training_context.py.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_probability_parity import canonical_feature_function
from verify_features import numpy_kan_predict, parse_c_header


def synthetic_ppg(n):
    t = np.arange(n, dtype=np.float64) / 50.0
    return (100.0 * np.sin(2.0 * np.pi * 1.2 * t)
            + 20.0 * np.sin(2.0 * np.pi * 0.1 * t)
            + 5.0 * (np.sin(13.0 * t + 0.5) + np.cos(29.0 * t + 1.2))).astype(np.float32)


def run_context_probe(executable, samples):
    payload = " ".join(format(float(x), ".9g") for x in np.asarray(samples, dtype=np.float32)) + "\n"
    result = subprocess.run([str(executable)], input=payload, text=True, capture_output=True, check=True)
    rows = np.asarray([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
    if rows.ndim != 2 or rows.shape[1] != 20 or not np.isfinite(rows).all():
        raise RuntimeError(f"Invalid context probe output: {rows.shape}")
    return rows


def python_context_probability(samples, start, features, sos, parameters):
    left = max(0, start - 250)
    right = min(len(samples), start + 750)
    filtered = signal.sosfiltfilt(sos, samples[left:right].astype(np.float64))
    window = filtered[start - left:start - left + 500]
    values = np.array(list(features(window, 50.).values()))
    return float(numpy_kan_predict(values, parameters)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    samples = synthetic_ppg(1250)
    actual = run_context_probe(args.probe.resolve(), samples)
    if actual.shape[0] != 3:
        raise RuntimeError(f"Expected 3 decisions, got {actual.shape[0]}")
    parameters = {k: np.asarray(v, dtype=np.float32).astype(np.float64)
                  for k, v in parse_c_header(ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h").items()}
    features = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    expected_meta = ((0, 750, 0, 500, 250), (1, 1000, 250, 750, 250), (2, 1250, 500, 1000, 250))
    passed = True
    for row, meta in zip(actual, expected_meta):
        ok_meta = tuple(int(x) for x in row[:5]) == meta
        expected_p = python_context_probability(samples, meta[2], features, sos, parameters)
        error = abs(row[5] - expected_p)
        ok = ok_meta and error < 1e-4 and ((row[5] >= .5) == (expected_p >= .5))
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'} window {meta[0]}: meta={ok_meta} probability error={error:.3g}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

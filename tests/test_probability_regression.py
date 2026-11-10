"""Independent saved-window regression; needs NumPy/SciPy, no local dataset.

Run: python tests/test_probability_regression.py --probe build/parity/probability_probe.exe
The broader absolute raw-feature gate remains in check_probability_parity.py.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_probability_parity import canonical_feature_function, reference, run_probe
from verify_features import parse_c_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--windows", type=Path,
                        help="Optional local replay CSV; recorded inputs remain under ignored outputs/")
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/parity_windows.csv"
    if args.windows:
        fixture = args.windows
    windows = np.loadtxt(fixture, delimiter=",", dtype=np.float32)
    cases = ([f"replay_{i}" for i in range(len(windows))] if args.windows else
             json.loads(fixture.with_suffix(".json").read_text())["cases"])
    if windows.shape != (len(cases), 500) or (not args.windows and len(cases) != 9) or len(cases) == 0:
        raise ValueError("Incomplete regression fixture")
    parameters = parse_c_header(ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h")
    parameters = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in parameters.items()}
    compute_features = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    features = np.array([list(compute_features(signal.sosfiltfilt(sos, row.astype(np.float64)), 50.).values())
                         for row in windows])
    expected_p, expected_xai = reference(features, parameters)
    actual = run_probe(args.probe.resolve(), "raw", windows)
    probability_error = np.abs(actual[:, 0] - expected_p)
    attribution_error = np.max(np.abs(actual[:, 29:] - expected_xai), axis=1)
    # Probability threshold is the existing deployment threshold. Numerical
    # probability tolerance is tighter than the project's <0.01 requirement.
    passed = ((probability_error < 1e-4) & (attribution_error < 1e-5)
              & ((actual[:, 0] >= .50) == (expected_p >= .50)))
    for name, ok, prob_error, xai_error in zip(cases, passed, probability_error, attribution_error):
        print(f"{'PASS' if ok else 'FAIL'} {name}: probability error={prob_error:.3g}, attribution error={xai_error:.3g}")
    return 0 if passed.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())

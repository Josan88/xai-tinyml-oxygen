"""Frozen numerical contract adopted on 2026-09-10; no CLI budget overrides."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests/fixtures/numerical_parity_contract.json"
CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
CONTRACT = json.loads(CONTRACT_BYTES)
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_BYTES).hexdigest()
NAMES = CONTRACT["feature_names"]


def vector(values, label):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain 14 finite values")
    return result


# Two feature criteria are reported; only one is the gate.
#   "pass": error <= budget, the scale-aware FP32 budget from the frozen contract.
#     This is the ACTIVE gate and the only feature criterion that feeds
#     check_window()["pass"] and every report's "failed_windows".
#   "original_absolute_1e_4_pass": error <= 1e-4 absolute. LEGACY reference from
#     before the scale-aware budget was adopted, kept so the old criterion stays
#     visible. It does not gate anything. It is expected to fail on raw-scale
#     features: std reaches order 1e5 and energy order 1e13, where one FP32 ulp
#     alone exceeds 1e-4. Reports surface it as
#     "original_absolute_1e_4_failed_windows"; a non-zero count there next to
#     "failed_windows": 0 is the expected outcome, not an unexplained failure.
def feature_check(expected, actual):
    expected, actual = vector(expected, "Reference features"), vector(actual, "Actual features")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = expected.astype(np.float32)
        spacing = np.abs(np.spacing(rounded).astype(np.float64))
    if not np.isfinite(rounded).all() or not np.isfinite(spacing).all() or expected[5] < 0:
        raise ValueError("Reference features cannot define a finite FP32 budget")
    scale = np.abs(expected)
    for i, kind in enumerate(CONTRACT["local_scales"]):
        if kind == "rms":
            scale[i] = expected[5]
        elif kind == "absolute_or_rms":
            scale[i] = max(scale[i], expected[5])
        elif kind == "absolute_or_one":
            scale[i] = max(1., scale[i])
    budget = CONTRACT["feature_scale_factor"] * scale + CONTRACT["output_spacings"] * spacing
    budget[10] = 0.
    error = np.abs(actual - expected)
    passed = error <= budget
    return {"pass": bool(passed.all()), "absolute_errors": error.tolist(),
            "budgets": budget.tolist(), "failed_features": [NAMES[i] for i in np.flatnonzero(~passed)],
            "original_absolute_1e_4_pass": bool(np.all(error <= 1e-4))}


def attribution_check(expected, actual):
    expected, actual = vector(expected, "Reference deltas"), vector(actual, "Actual deltas")
    if np.any(np.abs(expected) > 1) or np.any(np.abs(actual) > 1):
        raise ValueError("Probability deltas must be in [-1, 1]")
    error = np.abs(actual - expected)
    magnitudes, observed = np.abs(expected), np.abs(actual)
    violations, ambiguous = [], []
    for i in range(14):
        for j in range(i + 1, 14):
            gap = magnitudes[i] - magnitudes[j]
            if abs(gap) <= CONTRACT["ranking_gap"]:
                ambiguous.append([i, j])
            elif (gap > 0 and observed[i] <= observed[j]) or (gap < 0 and observed[i] >= observed[j]):
                violations.append([i, j])
    expected_top = np.argsort(-magnitudes, kind="stable")[:3].tolist()
    actual_top = np.argsort(-observed, kind="stable")[:3].tolist()
    delta_pass = bool(np.all(error <= CONTRACT["attribution_absolute"]))
    return {"pass": delta_pass and not violations, "delta_pass": delta_pass,
            "absolute_errors": error.tolist(), "max_absolute_error": float(error.max()),
            "failed_deltas": [NAMES[i] for i in np.flatnonzero(error > CONTRACT["attribution_absolute"])],
            "rank_violations": violations, "ambiguous_reference_pairs": ambiguous,
            "reference_top3": expected_top, "actual_top3": actual_top,
            "top3_order_changed": expected_top != actual_top,
            "top3_set_changed": set(expected_top) != set(actual_top)}


def check_window(expected_features, actual_features, expected_probability, actual_probability,
                 expected_deltas, actual_deltas):
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (expected_probability, actual_probability)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    features = feature_check(expected_features, actual_features)
    attributions = attribution_check(expected_deltas, actual_deltas)
    probability_error = abs(float(actual_probability) - float(expected_probability))
    candidate_agrees = (expected_probability >= CONTRACT["threshold"]) == (actual_probability >= CONTRACT["threshold"])
    probability_pass = probability_error <= CONTRACT["probability_absolute"]
    return {"pass": bool(features["pass"] and attributions["pass"] and probability_pass and candidate_agrees),
            "features": features, "attributions": attributions,
            "probability_absolute_error": probability_error, "probability_pass": probability_pass,
            "reference_probability": float(expected_probability), "actual_probability": float(actual_probability),
            "candidate_agrees": bool(candidate_agrees)}

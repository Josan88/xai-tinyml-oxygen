"""Frozen numerical contract applied to the 16-input v2 model (ADR-0006).

Every budget is read from the frozen v1 contract (tests/fixtures/numerical_parity_contract.json,
adopted 2026-09-10) through scripts/parity_criteria.py; none is restated or changed here:
local-scale 1e-4 plus 8 FP32 output spacings for features, probability 1e-4, each signed
attribution delta 1e-5, ranking gap 2e-5, exact 0.50 candidate, exact dominant-frequency bin.

The frozen contract names 14 features. The v2 model adds two, and the only new decision in
this file is which existing local-scale class each one uses. It was fixed before any v2
parity run (this file was committed first):
  ratio_of_ratios     "absolute_or_one"  dimensionless ratio, the class of bandpower_ratio
  ir_perfusion_index  "absolute"         positive amplitude ratio, the class of std
For features 0-13 the budgets are bit-identical to parity_criteria.feature_check.
"""
import numpy as np

from parity_criteria import CONTRACT, CONTRACT_SHA256, NAMES as V1_NAMES

EXTRA_NAMES = ["ratio_of_ratios", "ir_perfusion_index"]
EXTRA_LOCAL_SCALES = ["absolute_or_one", "absolute"]
NAMES = list(V1_NAMES) + EXTRA_NAMES
LOCAL_SCALES = list(CONTRACT["local_scales"]) + EXTRA_LOCAL_SCALES
N = len(NAMES)
RMS_INDEX = V1_NAMES.index("rms")
FREQUENCY_INDEX = V1_NAMES.index("dominant_frequency_hz")


def vector(values, label):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (N,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {N} finite values")
    return result


# Same two criteria as parity_criteria.feature_check: "pass" is the adopted
# local-scale gate; "original_absolute_1e_4_pass" is the legacy reference that
# gates nothing and is expected to fail on raw-scale features.
def feature_check(expected, actual):
    expected, actual = vector(expected, "Reference features"), vector(actual, "Actual features")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = expected.astype(np.float32)
        spacing = np.abs(np.spacing(rounded).astype(np.float64))
    if not np.isfinite(rounded).all() or not np.isfinite(spacing).all() or expected[RMS_INDEX] < 0:
        raise ValueError("Reference features cannot define a finite FP32 budget")
    scale = np.abs(expected)
    for i, kind in enumerate(LOCAL_SCALES):
        if kind == "rms":
            scale[i] = expected[RMS_INDEX]
        elif kind == "absolute_or_rms":
            scale[i] = max(scale[i], expected[RMS_INDEX])
        elif kind == "absolute_or_one":
            scale[i] = max(1., scale[i])
    budget = CONTRACT["feature_scale_factor"] * scale + CONTRACT["output_spacings"] * spacing
    budget[FREQUENCY_INDEX] = 0.
    error = np.abs(actual - expected)
    passed = error <= budget
    relative = np.divide(error, np.abs(expected), out=np.full(N, np.nan), where=expected != 0)
    return {"pass": bool(passed.all()), "absolute_errors": error.tolist(), "relative_errors": relative.tolist(),
            "budgets": budget.tolist(), "failed_features": [NAMES[i] for i in np.flatnonzero(~passed)],
            "original_absolute_1e_4_pass": bool(np.all(error <= 1e-4)),
            "original_absolute_1e_4_failed_features": [NAMES[i] for i in np.flatnonzero(error > 1e-4)]}


def attribution_check(expected, actual):
    expected, actual = vector(expected, "Reference deltas"), vector(actual, "Actual deltas")
    if np.any(np.abs(expected) > 1) or np.any(np.abs(actual) > 1):
        raise ValueError("Probability deltas must be in [-1, 1]")
    error = np.abs(actual - expected)
    magnitudes, observed = np.abs(expected), np.abs(actual)
    violations, ambiguous = [], []
    for i in range(N):
        for j in range(i + 1, N):
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


CRITERIA_DESCRIPTION = {
    "v1_contract_sha256": CONTRACT_SHA256,
    "feature_names": NAMES,
    "local_scales": LOCAL_SCALES,
    "new_in_v2": dict(zip(EXTRA_NAMES, EXTRA_LOCAL_SCALES)),
    "budgets_from_v1_contract": {k: CONTRACT[k] for k in ("feature_scale_factor", "output_spacings",
                                                          "probability_absolute", "attribution_absolute",
                                                          "ranking_gap", "threshold", "debounce")},
}

"""P5 parity criteria (job F2), exactly as pre-registered in docs/p5-parity-criteria.md (commit f0f655a).

Budgets come from the frozen v1 contract (tests/fixtures/numerical_parity_contract.json, through
scripts/parity_criteria.py); none is restated or changed here:
  (b) features   |f_cpp - f_py| <= 1e-4 x local scale + 8 FP32 spacings of f_py, on decisions whose
                 peak lists agree and that are beat-valid on both sides; local-scale classes fixed
                 in the pre-registration: r_beat absolute_or_one, pi_beat absolute, hr_bpm absolute,
                 sqi_template absolute_or_one, sqi_ibi_ratio absolute_or_one;
  (c)            probability 1e-4, each of the 5 signed deltas 1e-5, ranking gap 2e-5, exact 0.50;
  (d) s_hat      |ds_hat| <= 4e-4 x KANSAT_TARGET_STD + 8 FP32 spacings of s_hat_py.
(a) discrete agreement and (e) exact fields are checked by the callers (exact equality).
"""
import numpy as np

from parity_criteria import CONTRACT, CONTRACT_SHA256

NAMES = ["r_beat", "pi_beat", "hr_bpm", "sqi_template", "sqi_ibi_ratio"]
LOCAL_SCALES = ["absolute_or_one", "absolute", "absolute", "absolute_or_one", "absolute_or_one"]
N = len(NAMES)
S_HAT_OUTPUT_ERROR = 4e-4          # regressor-output budget (pre-registered (d)); times KANSAT_TARGET_STD


def _vector(values, label):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (N,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {N} finite values")
    return result


def _spacing(values):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.abs(np.spacing(np.asarray(values, dtype=np.float64).astype(np.float32)).astype(np.float64))


def feature_check(expected, actual):
    expected, actual = _vector(expected, "Reference features"), _vector(actual, "Actual features")
    scale = np.abs(expected)
    for i, kind in enumerate(LOCAL_SCALES):
        if kind == "absolute_or_one":
            scale[i] = max(1.0, scale[i])
    budget = CONTRACT["feature_scale_factor"] * scale + CONTRACT["output_spacings"] * _spacing(expected)
    error = np.abs(actual - expected)
    passed = error <= budget
    return {"pass": bool(passed.all()), "absolute_errors": error.tolist(), "budgets": budget.tolist(),
            "relative_errors": np.divide(error, np.abs(expected), out=np.full(N, np.nan), where=expected != 0).tolist(),
            "failed_features": [NAMES[i] for i in np.flatnonzero(~passed)]}


def attribution_check(expected, actual):
    expected, actual = _vector(expected, "Reference deltas"), _vector(actual, "Actual deltas")
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
    return {"pass": delta_pass and not violations, "delta_pass": delta_pass, "max_absolute_error": float(error.max()),
            "failed_deltas": [NAMES[i] for i in np.flatnonzero(error > CONTRACT["attribution_absolute"])],
            "rank_violations": violations, "ambiguous_reference_pairs": ambiguous,
            "top3_order_changed": expected_top != actual_top, "top3_set_changed": set(expected_top) != set(actual_top)}


def probability_check(expected, actual):
    error = abs(float(actual) - float(expected))
    agrees = (expected >= CONTRACT["threshold"]) == (actual >= CONTRACT["threshold"])
    return {"pass": bool(error <= CONTRACT["probability_absolute"] and agrees), "absolute_error": error,
            "probability_pass": bool(error <= CONTRACT["probability_absolute"]), "candidate_agrees": bool(agrees)}


def s_hat_budget(expected, target_std):
    return S_HAT_OUTPUT_ERROR * float(target_std) + CONTRACT["output_spacings"] * float(_spacing([expected])[0])


def s_hat_check(expected, actual, target_std):
    error = abs(float(actual) - float(expected))
    budget = s_hat_budget(expected, target_std)
    return {"pass": bool(error <= budget), "absolute_error": error, "budget": budget,
            "display_text_differs": f"{float(expected):.1f}" != f"{np.float32(actual):.1f}"}


CRITERIA_DESCRIPTION = {
    "preregistration": "docs/p5-parity-criteria.md (commit f0f655a)",
    "v1_contract_sha256": CONTRACT_SHA256, "feature_names": NAMES, "local_scales": dict(zip(NAMES, LOCAL_SCALES)),
    "budgets_from_v1_contract": {k: CONTRACT[k] for k in ("feature_scale_factor", "output_spacings", "probability_absolute",
                                                          "attribution_absolute", "ranking_gap", "threshold", "debounce")},
    "s_hat": "4e-4 x KANSAT_TARGET_STD + 8 FP32 spacings of the Python s_hat",
    "classes": {"a": "polarity, full integer peak list, n_beats, beat_valid identical",
                "b": "feature budgets on (a)-pass decisions beat-valid on both sides; both invalid: inputs bit-equal to the imputation mean",
                "c": "probability 1e-4, 5 deltas 1e-5, ranking gap 2e-5, exact 0.50 candidate; every decision",
                "d": "s_hat budget; every decision", "e": "metadata, Candidate/Confirmed/streak, top-3 payload exact"},
}

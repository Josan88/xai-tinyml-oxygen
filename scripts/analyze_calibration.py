"""Probability calibration of the TinyKAN risk score on OpenOximetry (host only).

Question: is the TinyKAN P(Low-Oxygen Risk) calibrated, i.e. does P = 0.3 mean that about 30%
of such windows are low-oxygen? Measurement only. Nothing is retrained or recalibrated; the
0.50 operating point and ADR-0005 N = 2 are unchanged.

Inputs are existing out-of-fold files, loaded and fold-checked as in
scripts/analyze_lamaakal_style.py (load_inputs). They come from the cross-validation fold
models, not from the shipped all-rows headers. The stored y_pred / predicted_label columns are
never read (the v1 y_pred is the 0.30 cut); every flag is recomputed from the probability at 0.50.

Rules kept here: no board I/O, no firmware or header writes, no existing outputs/ file is
written. The output directory is write-once: the script stops if it already exists, and it
writes nothing when a sanity check fails.

The metric functions at the top are pure (NumPy only) and are covered by
tests/test_calibration_metrics.py. Every one takes optional window weights, so a recording
bootstrap (multiplicity as weight) is identical to replicating the drawn recordings.

Run (from the repository root):
  C:\\Python314\\python.exe scripts/analyze_calibration.py
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "calibration-v1"

THRESHOLD = 0.50
N_BINS = 10
CLIP = 1e-6
BOOTSTRAP_SEED = 20261103
N_BOOT = 2000
NEAR_CUT = ((0.45, 0.50), (0.50, 0.55))


# ----------------------------------------------------------------------------- pure metrics

def _weights(n, w):
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def logit(p, clip=CLIP):
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)


def _expit(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def base_rate(y, w=None):
    y = np.asarray(y, dtype=float)
    w = _weights(len(y), w)
    return float(np.sum(w * y) / np.sum(w))


def brier(y, p, w=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = _weights(len(y), w)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def brier_skill(y, p, w=None):
    """(Brier, Brier of the constant base-rate predictor, BSS = 1 - Brier / Brier_ref)."""
    score = brier(y, p, w)
    rate = base_rate(y, w)
    reference = rate * (1.0 - rate)
    return score, reference, (1.0 - score / reference) if reference > 0 else float("nan")


def equal_width_bins(p, n_bins=N_BINS):
    """Bin index for [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]; p = 1 falls in the last bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.searchsorted(edges[1:-1], np.asarray(p, dtype=float), side="right")


def equal_mass_bins(p, n_bins=N_BINS, w=None, order=None):
    """Bin index for n_bins contiguous runs of the sorted p holding equal total weight.

    Windows are sorted by p (stable, so ties keep row order) and the bin of each window is
    floor(n_bins * weight-before-it / total weight). With unit weights the bins are the
    N/n_bins smallest, the next N/n_bins, and so on. A tied value can straddle a bin edge.
    """
    p = np.asarray(p, dtype=float)
    w = _weights(len(p), w)
    order = np.argsort(p, kind="stable") if order is None else order
    sorted_w = w[order]
    before = np.cumsum(sorted_w) - sorted_w
    index = np.empty(len(p), dtype=np.int64)
    index[order] = np.minimum((n_bins * before / sorted_w.sum()).astype(np.int64), n_bins - 1)
    return index


def bin_table(y, p, index, n_bins=N_BINS, w=None):
    """Per-bin total weight, mean p and observed rate (NaN in empty bins)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = _weights(len(y), w)
    weight = np.bincount(index, weights=w, minlength=n_bins)
    sum_p = np.bincount(index, weights=w * p, minlength=n_bins)
    sum_y = np.bincount(index, weights=w * y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weight, sum_p / weight, sum_y / weight


def ece_mce(y, p, index, n_bins=N_BINS, w=None):
    """ECE = sum_k (W_k / W) |mean p_k - observed_k|; MCE = max over non-empty bins."""
    weight, mean_p, observed = bin_table(y, p, index, n_bins, w)
    filled = weight > 0
    gap = np.abs(mean_p[filled] - observed[filled])
    return float(np.sum(weight[filled] * gap) / weight.sum()), float(gap.max())


def murphy_decomposition(y, p, index, n_bins=N_BINS, w=None):
    """Binned Murphy decomposition: Brier = REL - RES + UNC + residual.

    The residual is the within-bin variance/covariance term; it is 0 when p is constant
    within every bin.
    """
    weight, mean_p, observed = bin_table(y, p, index, n_bins, w)
    filled = weight > 0
    total = weight.sum()
    rate = base_rate(y, w)
    reliability = float(np.sum(weight[filled] * (mean_p[filled] - observed[filled]) ** 2) / total)
    resolution = float(np.sum(weight[filled] * (observed[filled] - rate) ** 2) / total)
    uncertainty = rate * (1.0 - rate)
    residual = brier(y, p, w) - (reliability - resolution + uncertainty)
    return {"reliability": reliability, "resolution": resolution, "uncertainty": uncertainty,
            "residual": residual}


def _newton_logistic(x_columns, y, w, offset, start, max_iter=100, tol=1e-10):
    """Weighted unpenalised logistic regression by Newton steps with step halving."""
    design = np.column_stack(x_columns)
    beta = np.array(start, dtype=float)

    def loglik(b):
        eta = offset + design @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))

    current = loglik(beta)
    for _ in range(max_iter):
        mu = _expit(offset + design @ beta)
        gradient = design.T @ (w * (y - mu))
        hessian = (design * (w * mu * (1.0 - mu))[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value = loglik(candidate)
            if value >= current - 1e-12 or scale < 1e-8:
                break
            scale *= 0.5
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            return beta
    raise RuntimeError("Logistic calibration fit did not converge")


def calibration_slope_intercept(y, p, w=None):
    """(intercept, slope) of logit P(y=1) = a + b logit(p), p clipped to [1e-6, 1 - 1e-6].

    slope < 1: overconfident (scores too spread); slope > 1: underconfident (compressed).
    """
    y = np.asarray(y, dtype=float)
    x = logit(p)
    w = _weights(len(y), w)
    a, b = _newton_logistic((np.ones(len(y)), x), y, w, 0.0, (0.0, 1.0))
    return float(a), float(b)


def calibration_offset_intercept(y, p, w=None):
    """Intercept a of logit P(y=1) = a + 1 * logit(p) (slope fixed at 1, logit(p) as offset)."""
    y = np.asarray(y, dtype=float)
    w = _weights(len(y), w)
    (a,) = _newton_logistic((np.ones(len(y)),), y, w, logit(p), (0.0,))
    return float(a)


def operating_point_rates(y, p, threshold=THRESHOLD, w=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = _weights(len(y), w)
    flag = p >= threshold
    out = {}
    for low, high in NEAR_CUT:
        band = (p >= low) & (p < high)
        key = f"{low:.2f}_{high:.2f}".replace(".", "_")
        out[f"n_{key}"] = float(np.sum(w[band]))
        out[f"observed_rate_{key}"] = float(np.sum(w[band] * y[band]) / np.sum(w[band])) if band.any() else float("nan")
    positives, negatives = np.sum(w * y), np.sum(w * (1 - y))
    out["ppv_at_0_50"] = float(np.sum(w[flag] * y[flag]) / np.sum(w[flag]))
    out["npv_at_0_50"] = float(np.sum(w[~flag] * (1 - y[~flag])) / np.sum(w[~flag]))
    out["recall_at_0_50"] = float(np.sum(w[flag] * y[flag]) / positives)
    out["specificity_at_0_50"] = float(np.sum(w[~flag] * (1 - y[~flag])) / negatives)
    out["candidate_rate_at_0_50"] = float(np.sum(w[flag]) / np.sum(w))
    return out


def calibration_metrics(y, p, w=None, order=None, n_bins=N_BINS):
    """Every per-series number in one dict (used for the point estimate, folds and resamples)."""
    score, reference, skill = brier_skill(y, p, w)
    width = equal_width_bins(p, n_bins)
    mass = equal_mass_bins(p, n_bins, w, order)
    ece_w, mce_w = ece_mce(y, p, width, n_bins, w)
    ece_m, _ = ece_mce(y, p, mass, n_bins, w)
    intercept, slope = calibration_slope_intercept(y, p, w)
    rate = base_rate(y, w)
    mean_p = float(np.sum(_weights(len(y), w) * np.asarray(p, float)) / np.sum(_weights(len(y), w)))
    return {"base_rate": rate, "mean_p": mean_p, "calibration_in_the_large": mean_p - rate,
            "brier": score, "brier_reference": reference, "bss": skill,
            "ece_equal_width": ece_w, "ece_equal_mass": ece_m, "mce_equal_width": mce_w,
            "calibration_intercept": intercept, "calibration_slope": slope,
            "offset_intercept": calibration_offset_intercept(y, p, w)}


# ----------------------------------------------------------------------------- inputs

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


P = {
    "v1_features": ROOT / "outputs/openox_canonical_window_features.csv",
    "v1_tinykan_oof": ROOT / "outputs/openox_tinykan_predictions.csv",
    "v1_logreg_oof": ROOT / "outputs/openox_logreg_local_explanations.csv",
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v2_tinykan16_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "v2_tinykan16_patient_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
    "v2_logreg16_oof": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
}

SERIES = {
    "v1_tinykan": {"version": "v1", "model": "TinyKAN", "grouping": "recording",
                   "label": "v1 TinyKAN (14 IR, as-coded labels, recording-grouped OOF)"},
    "v1_logreg": {"version": "v1", "model": "LogReg", "grouping": "recording",
                  "label": "v1 LogReg (notebook 11, same folds)"},
    "v2_tinykan": {"version": "v2", "model": "TinyKAN", "grouping": "recording",
                   "label": "v2 TinyKAN (16 inputs, header-aligned labels, recording-grouped OOF)"},
    "v2_tinykan_patient": {"version": "v2", "model": "TinyKAN", "grouping": "patient",
                           "label": "v2 TinyKAN (patient-grouped OOF)"},
    "v2_logreg": {"version": "v2", "model": "LogReg", "grouping": "recording",
                  "label": "v2 LogReg (same recording-grouped folds)"},
}
PAIRS = (("v1", "v1_tinykan", "v1_logreg"), ("v2_recording", "v2_tinykan", "v2_logreg"))
EXPECTED = {
    "v1": {"windows": 52090, "recordings": 162},
    "v2": {"windows": 48662, "recordings": 154},
    "v1_tinykan": {"recall": 0.767, "specificity": 0.394},
    "v2_tinykan": {"recall": 0.893, "specificity": 0.868},
    "v2_tinykan_patient": {"recall": 0.899, "specificity": 0.864},
}


def load_inputs():
    """Same sort order, columns and fold checks as analyze_lamaakal_style.load_inputs."""
    import pandas as pd

    v1 = pd.read_csv(P["v1_features"], usecols=["recording_id", "label"])
    v1["recording_id"] = v1["recording_id"].astype(str)
    v1_kan = pd.read_csv(P["v1_tinykan_oof"], usecols=["row_index", "recording_id", "split_index", "y_true", "y_prob"])
    v1_kan = v1_kan.sort_values("row_index").reset_index(drop=True)
    v1_lr = pd.read_csv(P["v1_logreg_oof"], usecols=["fold", "source_index", "recording_id", "true_label",
                                                     "probability_low_oxygen_risk"])
    v1_lr = v1_lr.sort_values("source_index").reset_index(drop=True)
    checks = {}
    if not np.array_equal(v1_kan["row_index"], np.arange(len(v1))):
        raise RuntimeError("v1 TinyKAN predictions do not cover every canonical row once")
    if not (v1_kan["recording_id"].astype(str).to_numpy() == v1["recording_id"].to_numpy()).all():
        raise RuntimeError("v1 TinyKAN recording IDs differ from the canonical table")
    if not (v1_kan["y_true"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("v1 TinyKAN labels differ from the canonical table")
    if not np.array_equal(v1_lr["source_index"], np.arange(len(v1))):
        raise RuntimeError("v1 LogReg predictions do not cover every canonical row once")
    if not (v1_lr["true_label"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("v1 LogReg labels differ")
    if not (v1_lr["recording_id"].astype(str).to_numpy() == v1["recording_id"].to_numpy()).all():
        raise RuntimeError("v1 LogReg recording IDs differ from the canonical table")
    if not (v1_lr["fold"].to_numpy() == v1_kan["split_index"].to_numpy()).all():
        raise RuntimeError("v1 LogReg folds differ from TinyKAN split_index")
    if v1_kan.groupby("recording_id")["split_index"].nunique().max() != 1:
        raise RuntimeError("A v1 recording spans several folds")
    checks["v1_folds"] = ("split_index of openox_tinykan_predictions.csv; identical to fold of "
                          "openox_logreg_local_explanations.csv (aligned on source_index = row_index); each recording in one fold")

    v2 = pd.read_csv(P["v2_features"], usecols=["recording_id", "patient_id", "label"])
    v2["recording_id"] = v2["recording_id"].astype(str)
    v2_oof = {}
    for key in ("v2_tinykan16_oof", "v2_tinykan16_patient_oof", "v2_logreg16_oof"):
        frame = pd.read_csv(P[key], usecols=["row_index", "recording_id", "patient_id", "fold", "y_true", "y_prob"])
        frame = frame.sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(frame["row_index"], np.arange(len(v2))):
            raise RuntimeError(f"{key} does not cover every v2 row once")
        if not (frame["recording_id"].astype(str).to_numpy() == v2["recording_id"].to_numpy()).all():
            raise RuntimeError(f"{key} recording IDs differ")
        if not (frame["patient_id"].astype(str).to_numpy() == v2["patient_id"].astype(str).to_numpy()).all():
            raise RuntimeError(f"{key} patient IDs differ")
        if not (frame["y_true"].to_numpy() == v2["label"].to_numpy()).all():
            raise RuntimeError(f"{key} labels differ")
        v2_oof[key] = frame
    folds_rec = v2_oof["v2_tinykan16_oof"]["fold"].to_numpy()
    if not np.array_equal(v2_oof["v2_logreg16_oof"]["fold"].to_numpy(), folds_rec):
        raise RuntimeError("v2 LogReg folds differ from predictions_b_v2_tinykan16.csv")
    if v2_oof["v2_tinykan16_oof"].groupby("recording_id")["fold"].nunique().max() != 1:
        raise RuntimeError("A v2 recording spans several recording folds")
    if v2_oof["v2_tinykan16_patient_oof"].groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("A v2 patient spans several patient folds")
    checks["v2_folds"] = ("fold of predictions_b_v2_tinykan16.csv (recording-grouped), identical in predictions_c_v2_logreg16.csv; "
                          "fold of predictions_b_v2_tinykan16_patient.csv is patient-grouped (each patient in one fold)")

    y1, y2 = v1["label"].to_numpy(int), v2["label"].to_numpy(int)
    rec1, rec2 = v1["recording_id"].to_numpy(), v2["recording_id"].to_numpy()
    pat2 = v2["patient_id"].astype(str).to_numpy()
    data = {
        "v1_tinykan": (y1, v1_kan["y_prob"].to_numpy(float), v1_kan["split_index"].to_numpy(int), rec1, None),
        "v1_logreg": (y1, v1_lr["probability_low_oxygen_risk"].to_numpy(float), v1_lr["fold"].to_numpy(int), rec1, None),
        "v2_tinykan": (y2, v2_oof["v2_tinykan16_oof"]["y_prob"].to_numpy(float), folds_rec, rec2, pat2),
        "v2_tinykan_patient": (y2, v2_oof["v2_tinykan16_patient_oof"]["y_prob"].to_numpy(float),
                               v2_oof["v2_tinykan16_patient_oof"]["fold"].to_numpy(int), rec2, pat2),
        "v2_logreg": (y2, v2_oof["v2_logreg16_oof"]["y_prob"].to_numpy(float), folds_rec, rec2, pat2),
    }
    for name, (_, p, _, _, _) in data.items():
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise RuntimeError(f"{name}: probabilities outside [0, 1] or non-finite")
    return data, checks


def sanity_checks(data, checks):
    """The brief's reproduction checks; any failure raises before anything is written."""
    failures = []
    for version, key in (("v1", "v1_tinykan"), ("v2", "v2_tinykan")):
        y, _, _, rec, _ = data[key]
        got = {"windows": int(len(y)), "recordings": int(len(np.unique(rec)))}
        checks[f"{version}_counts"] = {"observed": got, "expected": EXPECTED[version], "pass": got == EXPECTED[version]}
        if got != EXPECTED[version]:
            failures.append(f"{version} counts {got} != {EXPECTED[version]}")
    for key in ("v1_tinykan", "v2_tinykan", "v2_tinykan_patient"):
        y, p, _, _, _ = data[key]
        rates = operating_point_rates(y, p)
        got = {"recall": round(rates["recall_at_0_50"], 3), "specificity": round(rates["specificity_at_0_50"], 3)}
        ok = got == EXPECTED[key]
        checks[f"{key}_at_0_50"] = {"observed_3dp": got, "expected": EXPECTED[key], "pass": ok,
                                    "recall": rates["recall_at_0_50"], "specificity": rates["specificity_at_0_50"]}
        if not ok:
            failures.append(f"{key} recall/specificity {got} != {EXPECTED[key]}")
    y, p, _, _, _ = data["v1_logreg"]
    rates = operating_point_rates(y, p)
    checks["v1_logreg_at_0_50_same_fold_row"] = {"recall": rates["recall_at_0_50"], "specificity": rates["specificity_at_0_50"],
                                                 "note": "notebook-11 same-fold row (0.866 / 0.261), not the README 0.859 / 0.264 reconstruction"}
    checks["stored_y_pred_columns_read"] = False
    if failures:
        raise SystemExit("SANITY CHECK FAILED, nothing written:\n  " + "\n  ".join(failures))
    return checks


def sklearn_cross_check(data):
    """Newton fit vs sklearn LogisticRegression without a penalty, on the full series."""
    import warnings
    from sklearn.linear_model import LogisticRegression

    out = {}
    for name, (y, p, _, _, _) in data.items():
        x = logit(p)[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = LogisticRegression(penalty=None, tol=1e-12, max_iter=10000)
                model.fit(x, y)
            except (TypeError, ValueError):
                model = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000)
                model.fit(x, y)
        a, b = calibration_slope_intercept(y, p)
        out[name] = {"abs_diff_intercept": abs(a - float(model.intercept_[0])), "abs_diff_slope": abs(b - float(model.coef_[0, 0]))}
    return out


# ----------------------------------------------------------------------------- analysis

def group_index_of(groups):
    _, index = np.unique(np.asarray(groups), return_inverse=True)
    return index, int(index.max()) + 1


def bootstrap_counts(n_groups, n_boot, seed):
    """B x n_groups multiplicities (same generator as lamaakal_metrics.group_bootstrap_counts)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.zeros((n_boot, n_groups), dtype=np.int32)
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=n_groups)
    return counts


BOOT_METRICS = ("brier", "bss", "ece_equal_width", "ece_equal_mass", "calibration_slope",
                "calibration_intercept", "offset_intercept", "calibration_in_the_large")


def bootstrap_series(y, p, group_index, counts, band_index):
    """Resample draws of every metric plus the observed rate in fixed equal-mass bins."""
    order = np.argsort(p, kind="stable")
    draws = {metric: np.empty(len(counts)) for metric in BOOT_METRICS}
    band = np.empty((len(counts), N_BINS))
    for b, row in enumerate(counts):
        w = row[group_index].astype(float)
        m = calibration_metrics(y, p, w, order)
        for metric in BOOT_METRICS:
            draws[metric][b] = m[metric]
        band[b] = bin_table(y, p, band_index, N_BINS, w)[2]
    return draws, band


def percentile_ci(values, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def analyse(data):
    import pandas as pd

    metric_rows, bin_rows, fold_rows = [], [], []
    point = {}
    for name, (y, p, folds, rec, pat) in data.items():
        info = SERIES[name]
        m = calibration_metrics(y, p)
        width = equal_width_bins(p)
        mass = equal_mass_bins(p)
        decomposition = murphy_decomposition(y, p, width)
        rates = operating_point_rates(y, p)
        # pos_weight BCE / class_weight='balanced' trains as if prevalence were 0.5; a pure prior
        # shift would put the slope-1 intercept at logit(base rate).
        point[name] = {**m, "logit_base_rate": float(logit(m["base_rate"])),
                       **{f"murphy_{k}": v for k, v in decomposition.items()}, **rates}
        metric_rows.append({"series": name, **info, "n_windows": int(len(y)), "n_recordings": int(len(np.unique(rec))),
                            "n_patients": int(len(np.unique(pat))) if pat is not None else "",
                            **point[name]})
        for binning, index in (("equal_width", width), ("equal_mass", mass)):
            weight, mean_p, observed = bin_table(y, p, index)
            for k in range(N_BINS):
                members = p[index == k]
                bin_rows.append({
                    "series": name, "binning": binning, "bin": k,
                    "edge_low": k / N_BINS if binning == "equal_width" else (float(members.min()) if len(members) else np.nan),
                    "edge_high": (k + 1) / N_BINS if binning == "equal_width" else (float(members.max()) if len(members) else np.nan),
                    "n_windows": int(weight[k]), "n_recordings": int(len(np.unique(rec[index == k]))),
                    "mean_p": mean_p[k], "observed_rate": observed[k], "gap_observed_minus_mean_p": observed[k] - mean_p[k],
                    "p_min": float(members.min()) if len(members) else np.nan,
                    "p_max": float(members.max()) if len(members) else np.nan,
                })
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fm = calibration_metrics(y[mask], p[mask])
            fold_rows.append({"series": name, **info, "fold": int(fold), "n_windows": int(mask.sum()),
                              "n_recordings": int(len(np.unique(rec[mask]))), "base_rate": fm["base_rate"],
                              "mean_p": fm["mean_p"], "brier": fm["brier"], "bss": fm["bss"],
                              "ece_equal_width": fm["ece_equal_width"], "calibration_slope": fm["calibration_slope"],
                              "calibration_intercept": fm["calibration_intercept"]})

    # Bootstraps: one recording resample matrix per dataset, shared by every series on it (paired);
    # v2 patient-grouped series also gets a patient-level resample.
    plans = [
        ("v1", "recording", BOOTSTRAP_SEED, ("v1_tinykan", "v1_logreg"), 3),
        ("v2", "recording", BOOTSTRAP_SEED + 1, ("v2_tinykan", "v2_tinykan_patient", "v2_logreg"), 3),
        ("v2", "patient", BOOTSTRAP_SEED + 2, ("v2_tinykan_patient",), 4),
    ]
    draws, bands, ci_rows, boot_info = {}, {}, [], []
    for version, unit, seed, names, column in plans:
        group_index, n_groups = group_index_of(data[names[0]][column])
        counts = bootstrap_counts(n_groups, N_BOOT, seed)
        boot_info.append({"version": version, "resampled_unit": unit, "seed": seed, "n_boot": N_BOOT,
                          "n_units": n_groups, "series": list(names)})
        for name in names:
            y, p = data[name][0], data[name][1]
            print(f"  bootstrap {name} over {unit}s ({n_groups} units)", flush=True)
            d, band = bootstrap_series(y, p, group_index, counts, equal_mass_bins(p))
            draws[(name, unit)] = d
            bands[(name, unit)] = band
            for metric in BOOT_METRICS:
                low, high = percentile_ci(d[metric])
                ci_rows.append({"series": name, **SERIES[name], "resampled_unit": unit, "metric": metric,
                                "point": point[name][metric], "ci_low": low, "ci_high": high,
                                "boot_sd": float(np.std(d[metric], ddof=1)), "n_boot": N_BOOT, "seed": seed,
                                "n_units": n_groups})
    for row in bin_rows:
        if row["binning"] == "equal_mass":
            band = bands[(row["series"], "recording")][:, row["bin"]]
            row["observed_ci_low"], row["observed_ci_high"] = percentile_ci(band)
        else:
            row["observed_ci_low"] = row["observed_ci_high"] = np.nan

    paired_rows = []
    for pair, tiny, logreg in PAIRS:
        seed = BOOTSTRAP_SEED if pair == "v1" else BOOTSTRAP_SEED + 1
        for metric in ("brier", "ece_equal_width", "ece_equal_mass", "calibration_slope"):
            delta = draws[(tiny, "recording")][metric] - draws[(logreg, "recording")][metric]
            low, high = percentile_ci(delta)
            paired_rows.append({"pair": pair, "comparison": "TinyKAN - LogReg", "metric": metric,
                                "tinykan": point[tiny][metric], "logreg": point[logreg][metric],
                                "difference": point[tiny][metric] - point[logreg][metric],
                                "ci_low": low, "ci_high": high, "se": float(np.std(delta, ddof=1)),
                                "fraction_resamples_le_0": float(np.mean(delta <= 0)),
                                "method": f"paired recording bootstrap ({N_BOOT} resamples, 95% percentile CI)",
                                "seed": seed})
    tables = {"metrics": pd.DataFrame(metric_rows), "bins": pd.DataFrame(bin_rows), "folds": pd.DataFrame(fold_rows),
              "ci": pd.DataFrame(ci_rows), "paired": pd.DataFrame(paired_rows)}
    return tables, point, boot_info


# ----------------------------------------------------------------------------- figure

COL2 = 7.16
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e2e1dc"


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
    })
    return plt


def plot_reliability(plt, data, bins, path):
    panels = (("v1 (as-coded labels, 14 IR)\nrecording-grouped OOF", ("v1_tinykan", "v1_logreg")),
              ("v2 (header-aligned, 16 inputs)\nrecording-grouped OOF", ("v2_tinykan", "v2_logreg")),
              ("v2 patient-grouped OOF\n(TinyKAN only)", ("v2_tinykan_patient",)))
    style = {"TinyKAN": (BLUE, "-", "o"), "LogReg": (ORANGE, "--", "s")}
    fig = plt.figure(figsize=(COL2, 3.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(3.0, 1.0))
    for c, (title, names) in enumerate(panels):
        ax = fig.add_subplot(grid[0, c])
        hist = fig.add_subplot(grid[1, c], sharex=ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.7, linestyle=":", label="perfect calibration")
        for name in names:
            model = SERIES[name]["model"]
            color, line, marker = style[model]
            block = bins[(bins.series == name) & (bins.binning == "equal_mass")].sort_values("bin")
            ax.fill_between(block.mean_p, block.observed_ci_low, block.observed_ci_high, color=color, alpha=0.15, linewidth=0)
            ax.plot(block.mean_p, block.observed_rate, color=color, linestyle=line, marker=marker, markersize=3.2,
                    markeredgecolor="white", markeredgewidth=0.4, label=model)
            hist.hist(data[name][1], bins=np.linspace(0, 1, 41), weights=np.full(len(data[name][1]), 1.0 / len(data[name][1])),
                      histtype="step", color=color, linestyle=line, linewidth=0.9)
        for axis in (ax, hist):
            axis.axvline(THRESHOLD, color=INK2, linewidth=0.7, linestyle="-.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(labelbottom=False)
        hist.set_xlabel("Predicted P(Low-Oxygen Risk)")
        if c == 0:
            ax.set_ylabel("Observed low-oxygen rate\n(10 equal-mass bins)")
            hist.set_ylabel("Share of windows")
            ax.text(THRESHOLD + 0.015, 0.04, "operating point 0.50", fontsize=6, color=INK2)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(path, dpi=300, facecolor="white", metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- text

def _ci(table, name, metric, unit="recording"):
    row = table[(table.series == name) & (table.metric == metric) & (table.resampled_unit == unit)].iloc[0]
    return row.point, row.ci_low, row.ci_high


def _fmt(value, low, high, digits=3):
    return f"{value:.{digits}f} (95% CI {low:.{digits}f} to {high:.{digits}f})"


def paper_sentences(tables, point):
    ci, paired = tables["ci"], tables["paired"]
    s = []
    for name, labels in (("v1_tinykan", "as-coded notebook-10 labels"), ("v2_tinykan", "header-aligned ScalcO2 labels estimated from ETO2")):
        version = SERIES[name]["version"]
        logreg = f"{version}_logreg"
        s.append(
            f"On the {version} OpenOximetry out-of-fold windows ({labels}; recording-grouped cross-validation fold models, "
            f"not the shipped all-rows header), the TinyKAN risk probability had Brier {_fmt(*_ci(ci, name, 'brier'))}, "
            f"Brier skill {_fmt(*_ci(ci, name, 'bss'))}, ECE {_fmt(*_ci(ci, name, 'ece_equal_width'))} with 10 equal-width bins "
            f"and calibration slope {_fmt(*_ci(ci, name, 'calibration_slope'), digits=2)}; mean P was "
            f"{-point[name]['calibration_in_the_large']:.3f} below the observed low-oxygen rate of {point[name]['base_rate']:.3f}; "
            f"logistic regression on the same folds had "
            f"Brier {point[logreg]['brier']:.3f}, ECE {point[logreg]['ece_equal_width']:.3f} and slope {point[logreg]['calibration_slope']:.2f} "
            f"(CIs: 2,000-resample recording bootstrap; windows overlap, so recordings, not windows, are the effective sample).")
    for pair, tiny, logreg in PAIRS:
        rows = paired[(paired.pair == pair)].set_index("metric")
        b, e = rows.loc["brier"], rows.loc["ece_equal_width"]
        label = "v1" if pair == "v1" else "v2 recording-grouped"
        s.append(f"Paired over the same recording resamples ({label}), TinyKAN minus logistic regression was "
                 f"{b.difference:+.3f} in Brier (95% CI {b.ci_low:+.3f} to {b.ci_high:+.3f}) and {e.difference:+.3f} in ECE "
                 f"(95% CI {e.ci_low:+.3f} to {e.ci_high:+.3f}); window-level, OpenOximetry out-of-fold windows from the "
                 f"CV fold models only.")
    name = "v2_tinykan_patient"
    s.append(f"With patient-grouped folds (v2 only; v1 folds let a patient appear in training and held-out folds), TinyKAN had Brier "
             f"{_fmt(*_ci(ci, name, 'brier', 'patient'))}, ECE {_fmt(*_ci(ci, name, 'ece_equal_width', 'patient'))} and slope "
             f"{_fmt(*_ci(ci, name, 'calibration_slope', 'patient'), digits=2)} (patient bootstrap). These are window-level calibration "
             f"figures on OpenOximetry only: a risk probability, not an SpO2 value or a clinical probability for a patient, and they do "
             f"not transfer to the MAX30102 or to live captures.")
    header = ("# Paper-ready sentences: probability calibration (calibration-v1)\n\n"
              "Generated by `scripts/analyze_calibration.py`. Host only; out-of-fold probabilities from the CV fold models, "
              "not the shipped header; OpenOximetry only, not the sensor. Nothing was recalibrated.\n\n")
    return header + "\n".join(f"- {line}" for line in s) + "\n"


# ----------------------------------------------------------------------------- main

def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.8g", lineterminator="\n")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not serialisable: {type(value)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    input_hashes = {P[k].relative_to(ROOT).as_posix(): sha256(P[k]) for k in P}

    data, checks = load_inputs()
    checks = sanity_checks(data, checks)
    checks["newton_vs_sklearn_unpenalised_logistic"] = sklearn_cross_check(data)
    print("Sanity checks passed; computing metrics and bootstraps ...", flush=True)
    tables, point, boot_info = analyse(data)
    m = tables["metrics"].set_index("series")
    checks["murphy_residual_max_abs"] = float(np.max(np.abs(m["murphy_residual"])))
    if any(v["abs_diff_slope"] > 1e-4 or v["abs_diff_intercept"] > 1e-4 for v in checks["newton_vs_sklearn_unpenalised_logistic"].values()):
        raise SystemExit("SANITY CHECK FAILED, nothing written: Newton calibration fit disagrees with sklearn")
    plt = setup_style()

    out.mkdir(parents=True)
    write_csv(tables["metrics"], out / "calibration_metrics.csv")
    write_csv(tables["bins"], out / "reliability_bins.csv")
    write_csv(tables["folds"], out / "per_fold.csv")
    write_csv(tables["ci"], out / "bootstrap_ci.csv")
    write_csv(tables["paired"], out / "paired_differences.csv")
    plot_reliability(plt, data, tables["bins"], out / "fig_reliability.png")
    (out / "paper_sentences.md").write_text(paper_sentences(tables, point), encoding="utf-8", newline="\n")

    ci = tables["ci"]
    headlines = {}
    for name in SERIES:
        headlines[name] = {"label": SERIES[name]["label"], **{k: point[name][k] for k in point[name]}}
        units = ("recording", "patient") if name == "v2_tinykan_patient" else ("recording",)
        for unit in units:
            block = ci[(ci.series == name) & (ci.resampled_unit == unit)]
            headlines[name][f"ci_{unit}_bootstrap"] = {r.metric: [r.ci_low, r.ci_high] for r in block.itertuples()}
    headlines["paired_tinykan_minus_logreg"] = {
        f"{r.pair}_{r.metric}": {"difference": r.difference, "ci": [r.ci_low, r.ci_high]} for r in tables["paired"].itertuples()}

    import matplotlib
    import pandas as pd
    import scipy
    import sklearn
    summary = {
        "created_utc": created,
        "duration_seconds": round(time.perf_counter() - started, 1),
        "script": "scripts/analyze_calibration.py",
        "scope": ("host only; measurement of out-of-fold probabilities from the CV fold models (not the shipped all-rows "
                  "headers, not the MAX30102); nothing retrained or recalibrated; 0.50 and N = 2 unchanged; no board I/O; "
                  "no firmware, header or existing outputs/ writes"),
        "threshold": THRESHOLD, "debounce_N": 2, "n_bins": N_BINS, "logit_clip": CLIP,
        "bootstrap": {"n_boot": N_BOOT, "base_seed": BOOTSTRAP_SEED, "generator": "numpy default_rng integers",
                      "ci": "95% percentile", "plans": boot_info,
                      "reliability_bands": "observed rate per equal-mass bin, bins fixed at the full-data membership, recording bootstrap"},
        "binning": {"equal_width": "10 bins [0, 0.1), ..., [0.9, 1.0]",
                    "equal_mass": "10 contiguous runs of the stable sort of p with equal total weight; tied p values keep row order and can straddle a bin edge"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "executable": sys.executable,
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "inputs_sha256": input_hashes,
        "checks": checks,
        "headlines": headlines,
        "outputs_sha256": {f.name: sha256(f) for f in sorted(out.iterdir()) if f.is_file()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if {P[k].relative_to(ROOT).as_posix(): sha256(P[k]) for k in P} != input_hashes:
        raise RuntimeError("An input file changed during the run")
    print(f"Done in {summary['duration_seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

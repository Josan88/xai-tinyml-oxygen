"""Metric helpers for the Lamaakal-style results analysis (host only, NumPy/SciPy).

Every function here is deterministic and is covered by tests/test_lamaakal_metrics.py.
Nothing in this module trains a model, reads a file, or touches the board.

Conventions:
  * y is 0/1 with 1 = Low-Oxygen Risk (median ScalcO2 < 90); p is P(Low-Oxygen Risk).
  * A window is a Candidate when p >= threshold (the firmware compares with >=).
  * Recording/patient bootstrap resamples whole groups with replacement and carries the
    multiplicity as window weights, which is identical to replicating the drawn groups.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

REPORTED_THRESHOLD = 0.50


def confusion_at(y, p, threshold=REPORTED_THRESHOLD, weights=None):
    """Return (tn, fp, fn, tp) at p >= threshold; weights default to 1 per window."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, dtype=float) >= threshold
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    tp = float(np.sum(w * ((y == 1) & pred)))
    fn = float(np.sum(w * ((y == 1) & ~pred)))
    fp = float(np.sum(w * ((y == 0) & pred)))
    tn = float(np.sum(w * ((y == 0) & ~pred)))
    return tn, fp, fn, tp


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def rates_from_counts(tn, fp, fn, tp):
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    n = tn + fp + fn + tp
    return {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": _ratio(tp + tn, n),
        "candidate_rate": _ratio(tp + fp, n),
    }


def binary_metrics(y, p, threshold=REPORTED_THRESHOLD):
    """Threshold-free (AUC, PR-AUC) and threshold metrics for one prediction vector."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite probabilities")
    tn, fp, fn, tp = confusion_at(y, p, threshold)
    result = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auc": weighted_auc(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    result.update(rates_from_counts(tn, fp, fn, tp))
    return result


class RankedScores:
    """Pre-sorted scores so that a weighted Mann-Whitney AUC costs O(n) per call.

    AUC = sum_{i pos, j neg} w_i w_j (1[s_j < s_i] + 0.5 * 1[s_j == s_i]) / (W_pos W_neg),
    which equals sklearn roc_auc_score(y, s, sample_weight=w) and, for integer w, the AUC
    of the data set in which each window is replicated w times.
    """

    def __init__(self, y, scores):
        self.y = np.asarray(y).astype(float)
        if not set(np.unique(self.y)).issubset({0.0, 1.0}):
            raise ValueError("Labels must be 0/1")
        _, self.inverse = np.unique(np.asarray(scores, dtype=float), return_inverse=True)
        self.levels = int(self.inverse.max()) + 1

    def auc(self, weights=None):
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, dtype=float)
        pos = np.bincount(self.inverse, weights=w * self.y, minlength=self.levels)
        neg = np.bincount(self.inverse, weights=w * (1.0 - self.y), minlength=self.levels)
        total_pos, total_neg = pos.sum(), neg.sum()
        if total_pos <= 0 or total_neg <= 0:
            return float("nan")
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + 0.5 * neg)) / (total_pos * total_neg))


def weighted_auc(y, p, weights=None):
    return RankedScores(y, p).auc(weights)


def group_bootstrap_counts(group_index, n_groups, n_boot, seed):
    """B x n_groups multiplicities from resampling n_groups groups with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.zeros((n_boot, n_groups), dtype=np.int32)
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=n_groups)
    if np.any(np.asarray(group_index) >= n_groups):
        raise ValueError("group_index out of range")
    return counts


def group_bootstrap(y, probabilities, groups, n_boot, seed, threshold=REPORTED_THRESHOLD):
    """Paired group bootstrap for several models scored on the same windows.

    probabilities: dict name -> p vector (same windows, same order as y and groups).
    Returns (point, draws): point[name][metric] and draws[name][metric] -> length-B array
    for metric in auc, recall, specificity. Every model sees the same resampled groups,
    so differences between models are paired.
    """
    y = np.asarray(y).astype(int)
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(group_index.max()) + 1
    counts = group_bootstrap_counts(group_index, n_groups, n_boot, seed)
    point, draws = {}, {}
    for name, p in probabilities.items():
        p = np.asarray(p, dtype=float)
        ranked = RankedScores(y, p)
        pred = p >= threshold
        pos_hit = (y == 1) & pred
        neg_hit = (y == 0) & ~pred
        # Per-group sums, so each resample is a dot product with the multiplicities.
        g_pos = np.bincount(group_index, weights=(y == 1), minlength=n_groups)
        g_neg = np.bincount(group_index, weights=(y == 0), minlength=n_groups)
        g_tp = np.bincount(group_index, weights=pos_hit, minlength=n_groups)
        g_tn = np.bincount(group_index, weights=neg_hit, minlength=n_groups)
        aucs = np.empty(n_boot)
        for b in range(n_boot):
            aucs[b] = ranked.auc(counts[b][group_index])
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = (counts @ g_tp) / (counts @ g_pos)
            specificity = (counts @ g_tn) / (counts @ g_neg)
        tn, fp, fn, tp = confusion_at(y, p, threshold)
        point[name] = {"auc": ranked.auc(), "recall": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp)}
        draws[name] = {"auc": aucs, "recall": recall, "specificity": specificity}
    return point, draws


def percentile_ci(values, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def delong_paired(y, p1, p2):
    """Paired DeLong test for two correlated AUCs (Sun & Xu 2014 midrank form).

    Treats windows as independent. Windows from one recording are correlated, so on this
    data the DeLong variance is anti-conservative; the recording bootstrap is the primary
    paired comparison.
    """
    y = np.asarray(y).astype(int)
    scores = np.vstack([np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)])
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("Need at least two windows of each class")
    tx = np.vstack([stats.rankdata(row) for row in pos])
    ty = np.vstack([stats.rankdata(row) for row in neg])
    tz = np.vstack([stats.rankdata(row) for row in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v_pos = (tz[:, :m] - tx) / n          # structural components over positives
    v_neg = 1.0 - (tz[:, m:] - ty) / m    # structural components over negatives
    covariance = np.cov(v_pos) / m + np.cov(v_neg) / n
    variance_diff = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    diff = aucs[0] - aucs[1]
    if variance_diff <= 0:
        z, p_value = float("nan"), float("nan")
    else:
        z = float(diff / np.sqrt(variance_diff))
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "difference": float(diff),
        "se_difference": float(np.sqrt(max(variance_diff, 0.0))),
        "z": z, "p_value": p_value,
        "covariance": covariance.tolist(),
    }


def threshold_sweep(y, p, thresholds):
    rows = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_at(y, p, threshold)
        row = {"threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        row.update(rates_from_counts(tn, fp, fn, tp))
        rows.append(row)
    return rows


def groups_holding_share(errors_per_group, share=0.5):
    """Smallest number of groups whose error counts add up to at least `share` of all errors."""
    errors = np.sort(np.asarray(errors_per_group, dtype=float))[::-1]
    total = errors.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(errors), share * total - 1e-9) + 1)


def concentration_curve(errors_per_group):
    """Lorenz-style curve: x = share of groups (most errors first), y = cumulative error share."""
    errors = np.sort(np.asarray(errors_per_group, dtype=float))[::-1]
    total = errors.sum()
    x = np.arange(0, len(errors) + 1) / max(len(errors), 1)
    y = np.concatenate([[0.0], np.cumsum(errors) / total]) if total > 0 else np.zeros(len(errors) + 1)
    return x, y


def firmware_abs_order(attribution):
    """Descending |delta|, original feature index first on ties (kan_sort_attributions bubble sort)."""
    attribution = np.asarray(attribution, dtype=float)
    return np.argsort(-np.abs(attribution), axis=-1, kind="stable")


def median_member(values, mask):
    """Index of the median-valued member of mask: sort by (value, index), take position (k-1)//2."""
    candidates = np.flatnonzero(np.asarray(mask))
    if len(candidates) == 0:
        return None
    values = np.asarray(values, dtype=float)[candidates]
    order = np.lexsort((candidates, values))
    return int(candidates[order[(len(candidates) - 1) // 2]])


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")

"""Tests for scripts/lamaakal_metrics.py (metric helpers of the Lamaakal-style analysis)."""
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lamaakal_metrics as lm  # noqa: E402


def make_data(n=400, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    if ties:
        p = np.round(p, 1)
    return y, p


def test_binary_metrics_match_sklearn_at_050():
    y, p = make_data()
    m = lm.binary_metrics(y, p, 0.50)
    pred = (p >= 0.50).astype(int)
    assert m["auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["specificity"] == pytest.approx(recall_score(1 - y, 1 - pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["candidate_rate"] == pytest.approx(pred.mean())
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)


def test_threshold_is_inclusive_like_firmware():
    tn, fp, fn, tp = lm.confusion_at([1, 0], [0.50, 0.50], 0.50)
    assert (tn, fp, fn, tp) == (0, 1, 0, 1)


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_auc_matches_sklearn_and_replication(ties):
    y, p = make_data(300, seed=3, ties=ties)
    w = np.random.default_rng(9).integers(0, 4, len(y)).astype(float)
    assert lm.weighted_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert lm.weighted_auc(y, p, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    replicated = np.repeat(np.arange(len(y)), w.astype(int))
    assert lm.weighted_auc(y, p, w) == pytest.approx(roc_auc_score(y[replicated], p[replicated]), abs=1e-12)


def test_group_bootstrap_paired_and_deterministic():
    y, p1 = make_data(600, seed=1)
    _, p2 = make_data(600, seed=2)
    groups = np.repeat(np.arange(60), 10)
    point_a, draws_a = lm.group_bootstrap(y, {"a": p1, "b": p2}, groups, 50, seed=7)
    point_b, draws_b = lm.group_bootstrap(y, {"a": p1, "b": p2}, groups, 50, seed=7)
    assert np.array_equal(draws_a["a"]["auc"], draws_b["a"]["auc"])
    assert point_a["a"]["auc"] == pytest.approx(roc_auc_score(y, p1))
    # Same model under two names: identical draws, so the paired difference is exactly zero.
    _, draws_same = lm.group_bootstrap(y, {"a": p1, "a_copy": p1.copy()}, groups, 50, seed=7)
    assert np.all(draws_same["a"]["auc"] - draws_same["a_copy"]["auc"] == 0)


def test_group_bootstrap_resample_equals_replicated_groups():
    y, p = make_data(120, seed=4)
    groups = np.repeat(np.arange(12), 10)
    counts = lm.group_bootstrap_counts(groups, 12, 1, seed=11)[0]
    _, draws = lm.group_bootstrap(y, {"m": p}, groups, 1, seed=11)
    keep = np.concatenate([np.repeat(np.flatnonzero(groups == g), c) for g, c in enumerate(counts)])
    pred = p[keep] >= 0.5
    assert draws["m"]["auc"][0] == pytest.approx(roc_auc_score(y[keep], p[keep]), abs=1e-12)
    assert draws["m"]["recall"][0] == pytest.approx(np.mean(pred[y[keep] == 1]))
    assert draws["m"]["specificity"][0] == pytest.approx(np.mean(~pred[y[keep] == 0]))


def brute_force_delong(y, p1, p2):
    def psi(a, b):
        return (a > b) + 0.5 * (a == b)
    pos = [np.asarray(p)[y == 1] for p in (p1, p2)]
    neg = [np.asarray(p)[y == 0] for p in (p1, p2)]
    m, n = len(pos[0]), len(neg[0])
    v10 = np.array([[psi(pos[k][i], neg[k]).mean() for i in range(m)] for k in range(2)])
    v01 = np.array([[psi(pos[k], neg[k][j]).mean() for j in range(n)] for k in range(2)])
    aucs = v10.mean(axis=1)
    cov = np.cov(v10) / m + np.cov(v01) / n
    return aucs, cov


@pytest.mark.parametrize("ties", [False, True])
def test_delong_matches_structural_components(ties):
    y, p1 = make_data(80, seed=5, ties=ties)
    _, p2 = make_data(80, seed=6, ties=ties)
    result = lm.delong_paired(y, p1, p2)
    aucs, cov = brute_force_delong(y, p1, p2)
    assert result["auc_1"] == pytest.approx(roc_auc_score(y, p1), abs=1e-12)
    assert result["auc_2"] == pytest.approx(roc_auc_score(y, p2), abs=1e-12)
    assert np.allclose([result["auc_1"], result["auc_2"]], aucs, atol=1e-12)
    assert np.allclose(np.asarray(result["covariance"]), cov, atol=1e-12)
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert result["se_difference"] == pytest.approx(se, abs=1e-12)


def test_delong_identical_models_has_no_test():
    y, p = make_data(60, seed=8)
    result = lm.delong_paired(y, p, p)
    assert result["difference"] == 0
    assert np.isnan(result["p_value"])


def test_groups_holding_share():
    assert lm.groups_holding_share([10, 0, 5, 5]) == 1          # 10 of 20 is exactly half
    assert lm.groups_holding_share([3, 3, 3, 3]) == 2
    assert lm.groups_holding_share([1, 1, 1, 1, 1], 0.8) == 4
    assert lm.groups_holding_share([0, 0]) == 0
    x, y = lm.concentration_curve([4, 0, 1])
    assert np.allclose(y, [0, 0.8, 1.0, 1.0]) and np.allclose(x, [0, 1 / 3, 2 / 3, 1])


def firmware_bubble_sort(values):
    """Python transcription of kan_sort_attributions in kan_inference.h."""
    items = list(enumerate(values))
    for i in range(len(items) - 1):
        for j in range(len(items) - i - 1):
            if abs(items[j][1]) < abs(items[j + 1][1]):
                items[j], items[j + 1] = items[j + 1], items[j]
    return [index for index, _ in items]


def test_firmware_abs_order_matches_bubble_sort_with_ties():
    rng = np.random.default_rng(12)
    for _ in range(200):
        values = np.round(rng.normal(0, 0.05, 16), 2)   # rounding forces ties, including +/- pairs
        values[rng.integers(0, 16)] = -values[rng.integers(0, 16)]
        assert lm.firmware_abs_order(values).tolist() == firmware_bubble_sort(values.tolist())


def test_median_member_is_deterministic():
    values = np.array([0.9, 0.2, 0.5, 0.5, 0.7, 0.1])
    mask = np.array([True, False, True, True, True, False])
    # members 0, 2, 3, 4 sorted by (value, index): 2 (0.5), 3 (0.5), 4 (0.7), 0 (0.9) -> position (4-1)//2 = 1
    assert lm.median_member(values, mask) == 3
    assert lm.median_member(values, np.zeros(6, bool)) is None


def test_threshold_sweep_monotone():
    y, p = make_data(500, seed=10)
    rows = lm.threshold_sweep(y, p, [0.3, 0.5, 0.7])
    recalls = [r["recall"] for r in rows]
    specs = [r["specificity"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert specs == sorted(specs)

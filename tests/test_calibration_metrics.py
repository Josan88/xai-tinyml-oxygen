"""Tests for the pure calibration metrics in scripts/analyze_calibration.py (synthetic data only)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_calibration as cal  # noqa: E402


def calibrated_draws(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_perfectly_calibrated_draws_give_ece_near_0_and_slope_near_1():
    y, p = calibrated_draws()
    m = cal.calibration_metrics(y, p)
    assert m["ece_equal_width"] < 0.01
    assert m["ece_equal_mass"] < 0.01
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.03)
    assert m["calibration_intercept"] == pytest.approx(0.0, abs=0.03)
    assert m["offset_intercept"] == pytest.approx(0.0, abs=0.02)
    assert abs(m["calibration_in_the_large"]) < 0.01


def test_constant_base_rate_predictor_has_zero_skill():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    p = np.full(len(y), y.mean())
    score, reference, skill = cal.brier_skill(y, p)
    assert score == pytest.approx(reference, abs=1e-15)
    assert skill == pytest.approx(0.0, abs=1e-12)


def test_brier_hand_computed():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.6, 0.3])
    # (0.01 + 0.04 + 0.16 + 0.09) / 4
    assert cal.brier(y, p) == pytest.approx(0.075, abs=1e-15)
    score, reference, skill = cal.brier_skill(y, p)
    assert reference == pytest.approx(0.25)
    assert skill == pytest.approx(1 - 0.075 / 0.25)


def test_murphy_identity_holds_when_p_constant_within_bins():
    rng = np.random.default_rng(2)
    p = rng.choice([0.05, 0.35, 0.55, 0.75, 0.95], size=3000)
    y = (rng.uniform(size=len(p)) < np.clip(p + 0.1, 0, 1)).astype(int)
    d = cal.murphy_decomposition(y, p, cal.equal_width_bins(p))
    assert d["residual"] == pytest.approx(0.0, abs=1e-12)
    assert cal.brier(y, p) == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-12)


def test_overconfident_transform_gives_slope_below_1_and_compressed_above_1():
    y, q = calibrated_draws(seed=3)
    over = cal._expit(2.0 * cal.logit(q))
    compressed = cal._expit(0.5 * cal.logit(q))
    assert cal.calibration_slope_intercept(y, over)[1] == pytest.approx(0.5, abs=0.03)
    assert cal.calibration_slope_intercept(y, compressed)[1] == pytest.approx(2.0, abs=0.1)


def test_equal_width_bins_edges():
    index = cal.equal_width_bins(np.array([0.0, 0.0999, 0.1, 0.5, 0.95, 1.0]))
    assert index.tolist() == [0, 0, 1, 5, 9, 9]


def test_equal_mass_bins_hold_equal_counts():
    p = np.random.default_rng(4).uniform(size=1000)
    counts = np.bincount(cal.equal_mass_bins(p), minlength=10)
    assert counts.tolist() == [100] * 10


def test_integer_weights_equal_replicated_rows():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < p).astype(int)
    w = rng.integers(0, 4, 400)
    weighted = cal.calibration_metrics(y, p, w)
    replicated = cal.calibration_metrics(np.repeat(y, w), np.repeat(p, w))
    for key, value in weighted.items():
        if key == "ece_equal_mass":
            # Weighted: all copies of a window share one bin; replicated copies can straddle an edge.
            assert value == pytest.approx(replicated[key], abs=0.02)
        else:
            assert value == pytest.approx(replicated[key], abs=1e-9), key


def test_operating_point_rates_use_inclusive_0_50():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.50, 0.50, 0.49, 0.10])
    r = cal.operating_point_rates(y, p)
    assert r["ppv_at_0_50"] == pytest.approx(0.5)
    assert r["npv_at_0_50"] == pytest.approx(0.5)
    assert r["n_0_45_0_50"] == 1 and r["observed_rate_0_45_0_50"] == 1.0
    assert r["n_0_50_0_55"] == 2 and r["observed_rate_0_50_0_55"] == 0.5


def test_ece_and_mce_on_a_tiny_table():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.15, 0.15, 0.85, 0.85])
    ece, mce = cal.ece_mce(y, p, cal.equal_width_bins(p))
    # bin 1: |0.15 - 0.5| = 0.35; bin 8: |0.85 - 1.0| = 0.15; equal weights
    assert ece == pytest.approx(0.25)
    assert mce == pytest.approx(0.35)

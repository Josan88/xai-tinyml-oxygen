"""Tests for scripts/compare_gradient_vs_perturbation.py (host-only gradient x input check)."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_gradient_vs_perturbation as g  # noqa: E402
from lamaakal_models import HeaderKAN  # noqa: E402

HEADERS = (g.P["v1_header"], g.P["v2_header"])


def central_difference(model, x, step=1e-6):
    out = np.empty_like(x)
    for feature in range(x.shape[1]):
        up, down = x.copy(), x.copy()
        up[:, feature] += step
        down[:, feature] -= step
        out[:, feature] = (model.predict(up) - model.predict(down)) / (2 * step)
    return out


def test_analytic_gradient_matches_finite_difference_on_both_headers():
    rng = np.random.default_rng(3)
    for header in HEADERS:
        model = HeaderKAN(header)
        x = rng.normal(0, 1, (300, model.n_inputs))
        probability, gradient = g.probability_gradient(model, x)
        np.testing.assert_allclose(probability, model.predict(x), atol=1e-15)
        error = np.abs(gradient - central_difference(model, x))
        assert np.quantile(error, 0.99) < 1e-7


def test_gradient_is_zero_when_every_hidden_unit_is_clamped():
    model = HeaderKAN(g.P["v1_header"])
    # Huge inputs push every first-layer pre-activation outside [-3, 3] unless a row of W1 is zero.
    x = np.full((1, model.n_inputs), 1e6)
    u1 = model.hidden_preactivation(x)
    _, gradient = g.probability_gradient(model, x)
    if np.all(np.abs(u1) > 3):
        np.testing.assert_array_equal(gradient, 0.0)


def test_gradient_x_input_is_first_order_mean_perturbation_near_zero():
    model = HeaderKAN(g.P["v1_header"])
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1e-4, (50, model.n_inputs))
    _, delta = model.mean_perturbation(x)
    _, gradient = g.probability_gradient(model, x)
    np.testing.assert_allclose(x * gradient, delta, atol=1e-8)


def test_firmware_order_breaks_ties_by_feature_index():
    order = g.firmware_order(np.array([[0.1, -0.8, -0.1, 0.0]]))
    np.testing.assert_array_equal(order, [[1, 0, 2, 3]])


def test_spearman_rows_identity_and_reversal():
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.isclose(g.spearman_rows(a, a)[0], 1.0)
    assert np.isclose(g.spearman_rows(a, -a[:, ::-1])[0], -1.0)
    assert np.isnan(g.spearman_rows(np.zeros((1, 4)), a)[0])

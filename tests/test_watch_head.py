"""Tests for scripts/analyze_watch_head.py pure helpers (written before the script).

Pre-registration: docs/watch-head-preregistration.md. No training, no board. The band and
y95 counts use the real v2 table; everything else uses synthetic vectors.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_watch_head as wh  # noqa: E402

V2_CSV = ROOT / "outputs" / "openox-v2" / "window_features.csv"


# ----------------------------------------------------------------------------- real table

@pytest.fixture(scope="module")
def v2():
    frame = pd.read_csv(V2_CSV)
    frame["recording_id"] = frame["recording_id"].astype(str)
    return frame


def test_label_is_median_below_90_on_every_row(v2):
    assert len(v2) == 48662
    assert (v2["label"].to_numpy() == (v2["median_spo2"].to_numpy() < 90).astype(int)).all()


def test_band_counts_on_real_table(v2):
    band = wh.bands(v2["median_spo2"].to_numpy())
    s = v2["median_spo2"].to_numpy()
    assert int(((s >= 89) & (s < 91)).sum()) == 0
    expected = {"L": (28795, 149, 85), "M": (7785, 151, 86), "N": (12082, 154, 87)}
    for name, (windows, recordings, patients) in expected.items():
        rows = band == name
        assert int(rows.sum()) == windows, name
        assert v2.loc[rows, "recording_id"].nunique() == recordings, name
        assert v2.loc[rows, "patient_id"].nunique() == patients, name
    assert set(np.unique(band)) == {"L", "M", "N"}


def test_y95_counts_on_real_table(v2):
    y = wh.y95(v2["median_spo2"].to_numpy())
    assert y.dtype == np.int64
    assert int(y.sum()) == 36580
    band = wh.bands(v2["median_spo2"].to_numpy())
    s_mask = v2["median_spo2"].to_numpy() >= 91
    assert int(s_mask.sum()) == 19867
    assert np.array_equal(y[s_mask], (band[s_mask] == "M").astype(np.int64))


def test_94_96_sensitivity_counts_on_real_table(v2):
    s = v2["median_spo2"].to_numpy()
    band = wh.bands(s)
    drop = wh.in_94_96(s)
    assert int(drop.sum()) == 5358
    assert int(((band == "M") & ~drop).sum()) == 5550
    assert int(((band == "N") & ~drop).sum()) == 8959


# ----------------------------------------------------------------------------- state rule

@pytest.mark.parametrize("p_a, p_b, state", [
    (0.90, 0.10, "LOW"),
    (0.90, 0.90, "LOW"),       # P_A >= 0.5 wins even if P_B >= 0.5
    (0.50, 0.99, "LOW"),       # exactly 0.50 counts as >=
    (0.50, 0.00, "LOW"),
    (0.49, 0.50, "WATCH"),     # exactly 0.50 counts as >=
    (0.10, 0.80, "WATCH"),
    (0.4999999, 0.4999999, "NONE"),
    (0.10, 0.10, "NONE"),
    (0.00, 0.00, "NONE"),
])
def test_state_rule_truth_table(p_a, p_b, state):
    assert wh.state_rule(np.array([p_a]), np.array([p_b]))[0] == state


def test_state_rule_vectorised_and_alert_equivalence():
    rng = np.random.default_rng(0)
    p_a = rng.uniform(size=1000)
    p_b = rng.uniform(size=1000)
    p_a[:5] = 0.5
    p_b[5:10] = 0.5
    states = wh.state_rule(p_a, p_b)
    alerted = np.isin(states, ("LOW", "WATCH"))
    assert np.array_equal(alerted, np.maximum(p_a, p_b) >= 0.5)
    assert np.array_equal(states == "LOW", p_a >= 0.5)


# ----------------------------------------------------------------------------- gate

def synthetic(recall_rec=1.0, spec_rec=1.0, recall_pat=1.0, spec_pat=1.0, a_ranks=False):
    """40 recordings x 10 windows, 5 M and 5 N in each. P_A never flags (< 0.5).

    recall_* and spec_* set the share of M with P_B >= 0.5 and of N with P_B < 0.5. Scores:
    M hit 0.80-0.90, M miss 0.45-0.46, N true negative 0.10-0.20, N false flag 0.55-0.56, so
    P_B ranks M above N far better than P_A (AUC about 0.5). a_ranks=True makes P_A_rec rank
    M above N perfectly, so with P_B also at AUC 1 the paired difference is 0 in every
    resample (G2 fails).
    """
    n_rec, per = 40, 10
    groups = np.repeat(np.arange(n_rec), per).astype(str)
    m = np.tile(np.r_[np.ones(5, int), np.zeros(5, int)], n_rec)
    rng = np.random.default_rng(1)

    def p_b(recall, spec):
        p = np.empty(len(m))
        pos, neg = np.flatnonzero(m == 1), np.flatnonzero(m == 0)
        n_hit = int(round(recall * len(pos)))
        n_tn = int(round(spec * len(neg)))
        p[pos] = 0.45 + 0.01 * rng.uniform(size=len(pos))
        p[pos[:n_hit]] = 0.80 + 0.1 * rng.uniform(size=n_hit)
        p[neg] = 0.10 + 0.1 * rng.uniform(size=len(neg))
        p[neg[n_tn:]] = 0.55 + 0.01 * rng.uniform(size=len(neg) - n_tn)
        return p

    if a_ranks:
        p_a_rec = np.where(m == 1, 0.30, 0.10) + 0.01 * rng.uniform(size=len(m))
    else:
        p_a_rec = 0.2 + 0.1 * rng.uniform(size=len(m))
    p_a_pat = 0.2 + 0.1 * rng.uniform(size=len(m))
    return dict(m=m, p_a_rec=p_a_rec, p_b_rec=p_b(recall_rec, spec_rec), p_a_pat=p_a_pat,
                p_b_pat=p_b(recall_pat, spec_pat), rec_groups=groups)


def run_gate(**kw):
    return wh.gate(**synthetic(**kw), n_boot=200, seed=20261025)


def conditions(g):
    return (g["G1"]["recall_pass"], g["G1"]["specificity_pass"], g["G2"]["pass"],
            g["G3"]["recall_pass"], g["G3"]["specificity_pass"])


def test_gate_pass():
    g = run_gate()
    assert conditions(g) == (True, True, True, True, True)
    assert g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"]
    assert g["verdict"] == "pass"


def test_gate_recall_fail():
    g = run_gate(recall_rec=0.70)
    assert g["G1"]["recall"] == pytest.approx(0.70)
    assert conditions(g) == (False, True, True, True, True)
    assert not g["G1"]["pass"] and g["verdict"] == "fail"


def test_gate_spec_fail():
    g = run_gate(spec_rec=0.40)
    assert g["G1"]["specificity"] == pytest.approx(0.40)
    assert conditions(g) == (True, False, True, True, True)
    assert not g["G1"]["pass"] and g["verdict"] == "fail"


def test_gate_g2_fail():
    g = run_gate(a_ranks=True)
    assert g["G2"]["ci_low"] <= 0
    assert conditions(g) == (True, True, False, True, True)
    assert g["verdict"] == "fail"


def test_gate_g3_fail():
    g = run_gate(recall_pat=0.50)
    assert conditions(g) == (True, True, True, False, True)
    assert not g["G3"]["pass"] and g["verdict"] == "fail"


def test_gate_boundaries_are_inclusive():
    g = run_gate(recall_rec=0.80, spec_rec=0.50, recall_pat=0.80, spec_pat=0.50)
    assert g["G1"]["recall"] == 0.80 and g["G1"]["specificity"] == 0.50
    assert conditions(g) == (True, True, True, True, True)
    assert g["verdict"] == "pass"


def test_gate_recall_counts_low_and_watch():
    """An M window flagged LOW by head A counts as alerted even if P_B < 0.5."""
    d = synthetic(recall_rec=0.70)
    miss = np.flatnonzero((d["m"] == 1) & (d["p_b_rec"] < 0.5))
    d["p_a_rec"] = d["p_a_rec"].copy()
    d["p_a_rec"][miss[: len(miss) // 2 + 1]] = 0.9
    g = wh.gate(**d, n_boot=50, seed=20261025)
    assert g["G1"]["recall"] > 0.80


# ----------------------------------------------------------------------------- label swap

def small_frame():
    s = np.array([50.0, 88.0, 91.0, 94.9, 95.0, 99.0])
    return pd.DataFrame({"recording_id": list("aabbcc"), "patient_id": list("xxyyzz"), "median_spo2": s,
                         "label": (s < 90).astype(np.int64), "mean": np.arange(6, dtype=float)})


def test_label_swap_changes_only_label():
    frame = small_frame()
    before = frame.copy()
    copy = wh.label_swap_copy(frame, expected_positive=4)
    assert frame.equals(before)                                   # the original is untouched
    assert list(copy.columns) == list(frame.columns) and copy.index.equals(frame.index)
    assert copy.drop(columns="label").equals(frame.drop(columns="label"))
    assert copy["label"].dtype == np.int64
    assert copy["label"].tolist() == [1, 1, 1, 1, 0, 0]


def test_label_swap_check_fires_if_another_column_changes():
    frame = small_frame()
    copy = wh.label_swap_copy(frame, expected_positive=4)
    copy.loc[0, "mean"] = 99.0
    with pytest.raises(AssertionError):
        wh.check_label_swap(frame, copy, expected_positive=4)


def test_label_swap_check_fires_on_wrong_count_or_bad_original():
    frame = small_frame()
    with pytest.raises(AssertionError):
        wh.label_swap_copy(frame, expected_positive=3)
    bad = frame.copy()
    bad.loc[0, "label"] = 0                                       # original label no longer median < 90
    with pytest.raises(AssertionError):
        wh.label_swap_copy(bad, expected_positive=4)


def test_label_swap_on_real_table(v2):
    copy = wh.label_swap_copy(v2)                                 # default expected_positive = 36,580
    assert int(copy["label"].sum()) == 36580
    assert copy.drop(columns="label").equals(v2.drop(columns="label"))

"""Tests for scripts/physio_features.py on synthetic PPG (no repository data, no board)."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import physio_features as pf  # noqa: E402

FS = 50.0
SOS = signal.butter(4, [0.5, 8.0], btype="bandpass", fs=FS, output="sos")


def pulse_train(hr_bpm, seconds=30.0, tau=0.06, notch=0.2):
    """Asymmetric blood-volume pulse: fast systolic rise, slow decay, small dicrotic wave. Peak 1."""
    t = np.arange(int(seconds * FS)) / FS
    period = 60.0 / hr_bpm
    out = np.zeros_like(t)
    for onset in np.arange(-2 * period, seconds, period):
        u = t - onset
        rise = np.where(u > 0, (u / tau) ** 2 * np.exp(2.0 - u / tau) / 4.0, 0.0)
        out += rise + notch * np.exp(-(((u - 0.35) / 0.06) ** 2)) * (u > 0)
    return out / out.max()


def window(hr_bpm, ac_ir=0.02, ac_red=0.01, dc_ir=1.0e5, dc_red=6.0e4, intensity=True):
    """Filtered IR/red window (500 samples) and raw-window DC means, as in v2_ir_red_windows.npz.

    intensity=True mimics transmitted light (systole is a dip); False is volume orientation.
    """
    shape = pulse_train(hr_bpm)
    sign = -1.0 if intensity else 1.0
    raw_ir = dc_ir * (1.0 + sign * ac_ir * shape)
    raw_red = dc_red * (1.0 + sign * ac_red * shape)
    lo, hi = 750, 1250
    ir = signal.sosfiltfilt(SOS, raw_ir)[lo:hi]
    red = signal.sosfiltfilt(SOS, raw_red)[lo:hi]
    return ir, red, raw_ir[lo:hi].mean(), raw_red[lo:hi].mean()


@pytest.mark.parametrize("hr", [60.0, 90.0, 120.0])
@pytest.mark.parametrize("ac_ratio", [0.5, 1.0, 1.6])
def test_recovers_hr_and_r(hr, ac_ratio):
    ac_ir = 0.02
    features = pf.extract(*window(hr, ac_ir=ac_ir, ac_red=ac_ratio * ac_ir))
    assert features["beat_valid"]
    assert abs(features["hr_bpm"] - hr) <= 1.0
    # Known R = (AC_red / DC_red) / (AC_ir / DC_ir) = ac_ratio by construction.
    assert abs(features["r_beat"] / ac_ratio - 1.0) <= 0.02
    assert features["sqi_template"] > 0.95
    assert features["sqi_ibi_ratio"] < 1.1


def test_perfusion_index_tracks_ac_over_dc():
    low = pf.extract(*window(75.0, ac_ir=0.01, ac_red=0.01))
    high = pf.extract(*window(75.0, ac_ir=0.03, ac_red=0.03))
    assert high["pi_beat"] / low["pi_beat"] == pytest.approx(3.0, rel=0.02)


@pytest.mark.parametrize("hr", [60.0, 90.0, 120.0])
def test_polarity_inversion_gives_same_hr_and_r(hr):
    ir, red, dc_ir, dc_red = window(hr, ac_red=0.013)
    a = pf.extract(ir, red, dc_ir, dc_red)
    b = pf.extract(-ir, -red, dc_ir, dc_red)
    assert a["polarity"] == -b["polarity"]
    for name in pf.FEATURES + ("n_beats",):
        assert a[name] == pytest.approx(b[name], rel=1e-12, abs=1e-12)


def test_polarity_rule_points_systole_up():
    assert pf.extract(*window(80.0, intensity=True))["polarity"] == -1
    assert pf.extract(*window(80.0, intensity=False))["polarity"] == 1


@pytest.mark.parametrize("flat", [np.zeros(500), np.full(500, 3.0)])
def test_flat_line_is_beat_invalid(flat):
    features = pf.extract(flat, flat, 1.0e5, 6.0e4)
    assert not features["beat_valid"]
    assert all(np.isnan(features[name]) for name in pf.FEATURES)


def test_nonpositive_dc_is_beat_invalid():
    ir, red, _, _ = window(70.0)
    assert not pf.extract(ir, red, 0.0, 6.0e4)["beat_valid"]
    assert not pf.extract(ir, red, 1.0e5, -1.0)["beat_valid"]


def test_three_pulses_is_beat_invalid():
    ir, red, dc_ir, dc_red = window(60.0, intensity=False)
    keep = np.zeros(500)
    keep[60:200] = 1.0                       # about 3 beats at 60 bpm, then flat
    features = pf.extract(ir * keep, red * keep, dc_ir, dc_red)
    assert features["n_beats"] < pf.MIN_BEATS
    assert not features["beat_valid"]


def test_noise_is_separated_by_sqi_not_by_beat_count():
    """Pre-registered limitation: band-limited noise passes the < 4-beat rule; the SQIs separate it."""
    rng = np.random.default_rng(20261030)
    clean_sqi = min(pf.extract(*window(hr))["sqi_template"] for hr in (60.0, 90.0, 120.0))
    for _ in range(20):
        noise = signal.sosfiltfilt(SOS, rng.standard_normal(2000))[750:1250]
        features = pf.extract(noise, 0.5 * noise, 1.0e5, 6.0e4)
        assert features["beat_valid"]        # documents the limitation; not a desired property
        assert features["sqi_template"] < 0.8 < clean_sqi
        assert features["sqi_ibi_ratio"] > 1.5


def test_extract_many_matches_extract():
    windows = [window(hr) for hr in (60.0, 95.0)]
    ir = np.stack([w[0] for w in windows])
    red = np.stack([w[1] for w in windows])
    dc_ir = np.array([w[2] for w in windows])
    dc_red = np.array([w[3] for w in windows])
    many = pf.extract_many(ir, red, dc_ir, dc_red)
    for i in range(2):
        one = pf.extract(ir[i], red[i], dc_ir[i], dc_red[i])
        for name in pf.FEATURES:
            assert many[name][i] == one[name]

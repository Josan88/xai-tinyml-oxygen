"""Physiological beat features from one filtered PPG window (host only, NumPy/SciPy, deterministic).

Definitions are fixed in docs/physio-concept-preregistration.md (Stage 1). Nothing here is
learned, nothing reads a file, and nothing touches the board. Each step is a plain loop
that could be ported to C on the MCU. The op counts below are estimates for a
500-sample window with about 12 beats; board cost has not been measured.

Inputs per window: filtered IR x and red r (Butterworth 4, 0.5-8 Hz, 500 samples at
50 Hz) and the raw-window DC means dc_ir, dc_red.

  1. Unusable window: std(x) == 0, dc_ir <= 0 or dc_red <= 0 -> beat-invalid.
     (~1.5k ops: one mean/variance pass)
  2. Polarity: d = diff(x); s = +1 if P99(d) >= -P1(d) else -1. The steepest slope of a
     pulse is the systolic upstroke, so s*x has systolic peaks pointing up.
     (~0.5k subtractions + two order statistics, ~5k compares with a selection pass)
  3. Beats: scipy.signal.find_peaks(s*x, distance=17, prominence=1.0*std(x)).
     17 samples = 0.34 s at 50 Hz (firmware HR_MIN_PEAK_GAP), so <= ~176 bpm.
     (~0.5k compares for local maxima; prominence walks left/right from each candidate,
     ~1-2k ops; distance pruning sorts ~20 candidates)
  4. Beat-valid iff step 1 passes and n_beats >= 4 (>= 3 intervals, firmware
     HR_MIN_INTERVALS). Otherwise all five features are NaN.
  5. IBI from 3-point-parabola refined peak positions; hr_bpm = 60 / median(IBI);
     sqi_ibi_ratio = max(IBI) / min(IBI). (~10 ops/beat + a ~12-element insertion sort)
  6. Segments [p_k, p_{k+1}] (inclusive) between consecutive integer IR peaks:
     AC_ir = max - min of x, AC_red = max - min of r on the same samples.
     r_beat = median of (AC_red/dc_red)/(AC_ir/dc_ir); pi_beat = median of AC_ir/dc_ir.
     Segments with AC_ir <= 0 are skipped. (~2 compares per sample per channel, ~2k ops,
     plus ~4 divisions per beat and two small sorts)
  7. sqi_template: each segment linearly resampled to the median segment length L; mean
     beat = average of the resampled segments; SQI = mean Pearson r of each resampled
     segment with the mean beat. (~beats x L x 10 ops, ~6k ops; on the MCU the mean beat
     can be accumulated in a first pass and the correlations in a second, so only L
     floats of buffer are needed)

Total: roughly 15-20k scalar operations per window, of the order of the existing
14-feature extraction but without its FFT. This is a count from the algorithm, not a
measurement.

Difference from firmware estimateHeartRateBpm() (firmware/Arduino Nano 33/src/main.cpp,
display-only HR, not changed): the firmware runs on the raw MAX30102 trace over 8 s,
inverts it by subtracting a 180 ms smooth from a 1.5 s baseline (MAX30102 systole is an
IR dip), takes local maxima above 60% of the window range at >= 17 samples apart, needs
>= 3 intervals, rejects the window when max/min gap > 1.4, uses the mean gap, and limits
HR to 40-180 bpm. Here the detector runs on the bandpassed 10 s window, chooses polarity
from the data, thresholds by prominence (1.0 x std) instead of height, uses the median
interval, and reports max/min gap as an input instead of using it as a rejection rule.

Known limitation (pre-registered): band-limited Gaussian noise yields 17-19 peaks per
window under a std-relative prominence, so noise is not beat-invalid under the < 4-beat
rule. The SQIs separate it (noise sqi_template ~0.61-0.70, sqi_ibi_ratio ~2.3-2.7).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SAMPLE_RATE_HZ = 50.0
MIN_PEAK_DISTANCE = 17          # samples (0.34 s); firmware HR_MIN_PEAK_GAP
PROMINENCE_STD = 1.0            # prominence threshold in units of the window's std
MIN_BEATS = 4                   # >= 3 intervals; firmware HR_MIN_INTERVALS = 3
POLARITY_PERCENTILE = 99.0
FEATURES = ("r_beat", "pi_beat", "hr_bpm", "sqi_template", "sqi_ibi_ratio")


def polarity(x):
    """+1 if the steepest slope of x is upward (systolic upstroke), else -1."""
    d = np.diff(np.asarray(x, dtype=np.float64))
    return 1 if np.percentile(d, POLARITY_PERCENTILE) >= -np.percentile(d, 100.0 - POLARITY_PERCENTILE) else -1


def refine_peaks(y, peaks):
    """Sub-sample peak positions from a 3-point parabola through each integer peak."""
    y = np.asarray(y, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)
    out = peaks.astype(np.float64)
    inner = (peaks > 0) & (peaks < len(y) - 1)
    a, b, c = y[peaks[inner] - 1], y[peaks[inner]], y[peaks[inner] + 1]
    denom = a - 2.0 * b + c
    offset = np.zeros_like(denom)
    curved = denom < 0
    offset[curved] = 0.5 * (a[curved] - c[curved]) / denom[curved]
    out[inner] += np.clip(offset, -0.5, 0.5)
    return out


def detect_beats(x):
    """Return (polarity, integer peak indices) of systolic peaks in a filtered IR window."""
    x = np.asarray(x, dtype=np.float64)
    s = polarity(x)
    peaks, _ = signal.find_peaks(s * x, distance=MIN_PEAK_DISTANCE, prominence=PROMINENCE_STD * float(np.std(x)))
    return s, peaks


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def template_sqi(x, peaks):
    segments = [x[peaks[k]:peaks[k + 1] + 1] for k in range(len(peaks) - 1)]
    length = int(np.floor(np.median([len(seg) for seg in segments]) + 0.5))
    grid = np.linspace(0.0, 1.0, length)
    resampled = np.array([np.interp(grid * (len(seg) - 1), np.arange(len(seg)), seg) for seg in segments])
    mean_beat = resampled.mean(axis=0)
    return float(np.mean([_pearson(beat, mean_beat) for beat in resampled]))


def invalid(n_beats=0, s=0):
    return {**{name: float("nan") for name in FEATURES}, "n_beats": int(n_beats), "beat_valid": False, "polarity": int(s)}


def extract(ir, red, ir_dc, red_dc):
    """Five beat features of one window plus n_beats, beat_valid and the chosen polarity."""
    x = np.asarray(ir, dtype=np.float64)
    r = np.asarray(red, dtype=np.float64)
    ir_dc, red_dc = float(ir_dc), float(red_dc)
    if not (np.std(x) > 0 and np.isfinite(x).all() and np.isfinite(r).all() and ir_dc > 0 and red_dc > 0):
        return invalid()
    s, peaks = detect_beats(x)
    if len(peaks) < MIN_BEATS:
        return invalid(len(peaks), s)
    ibi = np.diff(refine_peaks(s * x, peaks)) / SAMPLE_RATE_HZ
    ratios, perfusion = [], []
    for k in range(len(peaks) - 1):
        lo, hi = peaks[k], peaks[k + 1] + 1
        ac_ir = float(x[lo:hi].max() - x[lo:hi].min())
        ac_red = float(r[lo:hi].max() - r[lo:hi].min())
        if ac_ir <= 0:
            continue
        ratios.append((ac_red / red_dc) / (ac_ir / ir_dc))
        perfusion.append(ac_ir / ir_dc)
    if not ratios:
        return invalid(len(peaks), s)
    return {
        "r_beat": float(np.median(ratios)),
        "pi_beat": float(np.median(perfusion)),
        "hr_bpm": float(60.0 / np.median(ibi)),
        "sqi_template": template_sqi(s * x, peaks),
        "sqi_ibi_ratio": float(ibi.max() / ibi.min()),
        "n_beats": int(len(peaks)),
        "beat_valid": True,
        "polarity": int(s),
    }


def extract_many(ir, red, ir_dc, red_dc):
    """Row-wise extract over N x 500 arrays. Returns a dict of length-N arrays."""
    rows = [extract(ir[i], red[i], ir_dc[i], red_dc[i]) for i in range(len(ir))]
    keys = FEATURES + ("n_beats", "beat_valid", "polarity")
    return {key: np.array([row[key] for row in rows]) for key in keys}

"""Python reference of the DEPLOYED v2 path (ADR-0006), shared by the v2 host and board tools.

Deployed path = OxygenRiskPipeline (BasicOxygenRiskPipeline<250>) built with OXY_MODEL_V2:
paired 50 Hz IR/red samples; a decision every 250 samples from sample 750; each decision
filters [max(0, start - 250), start + 750) of EACH channel with sosfiltfilt (4th-order
Butterworth 0.5-8 Hz, SciPy default 27-sample odd padding) and scores the 500-sample target
window [start, start + 500). Features are the canonical notebook-10 14 IR features plus the
export's two_wavelength_features() (imported, not copied). Mean-Perturbation: 1 baseline + 16
passes, each zeroing one standardized input. ADR-0005: candidate P >= 0.50, confirmed after
N=2 consecutive candidates.

Arithmetic follows the frozen contract: input samples and model constants are FP32-quantized
(what the firmware holds), SciPy/NumPy then work in FP64. Where the export would give NaN, the
firmware substitutes the training means for both two-wavelength inputs; so does this reference.

Explainer reference: by default each pass sets the input to 0 (the training mean), as the
shipped images do. load_reference("label0") returns KAN_XAI_REF_Z from kan_xai_reference_v2.h
(FP32, as the -DOXY_XAI_REF_LABEL0 images hold it); pass it as `reference` to
probabilities_and_deltas() / stream() for those images. Nothing else changes.
"""
from pathlib import Path
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_probability_parity import canonical_feature_function  # noqa: E402
from export_openox_windows_v2 import two_wavelength_features, IR_FEATURE_NAMES, EXTRA_FEATURE_NAMES  # noqa: E402
from verify_features import parse_c_header, numpy_kan_predict  # noqa: E402
from xai_reference import HEADER_V2 as REFERENCE_HEADER_V2, parse_reference_header  # noqa: E402

HEADER_V2 = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h"
XAI_REFERENCES = ("mean", "label0")
FEATURE_NAMES = list(IR_FEATURE_NAMES) + list(EXTRA_FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)
WINDOW, CONTEXT, STRIDE, FIRST = 500, 250, 250, 750
THRESHOLD, DEBOUNCE = 0.50, 2
SOS = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
_COMPUTE_IR = canonical_feature_function()


def load_parameters(header=HEADER_V2):
    """Header constants quantized exactly as stored in the FP32 firmware, then FP64."""
    parameters = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in parse_c_header(header).items()}
    if parameters["scaler_mean"].shape != (N_FEATURES,) or parameters["w1"].shape != (12, N_FEATURES):
        raise ValueError("Not the 16-input v2 header")
    return parameters


def _bspline(u, centers, step):
    dist = np.abs((np.clip(u, -3.0, 3.0)[..., None] - centers) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0, None)**3 / 6.0
    return np.where(dist < 1.0, inner, outer)


def logits_from_scaled(scaled, p):
    """Vectorized numpy_kan_predict (verify_features.py) on standardized rows."""
    u1 = scaled @ p["w1"].T + p["b1"]
    out1 = np.sum(_bspline(u1, p["centers1"], p["step1"]) * p["theta1"], axis=-1)
    u2 = out1 @ p["w2"].T + p["b2"]
    return np.sum(_bspline(u2[..., 0], p["centers2"], p["step2"]) * p["theta2"][0], axis=-1)


def load_reference(name="mean", header=REFERENCE_HEADER_V2):
    """None for the training mean (0); else KAN_XAI_REF_Z from the reference header, FP32 then FP64."""
    if name == "mean":
        return None
    if name not in XAI_REFERENCES:
        raise ValueError(f"Unknown explainer reference {name!r}")
    tag, values = parse_reference_header(header)
    if tag != name or values.shape != (N_FEATURES,):
        raise ValueError(f"{header} holds reference {tag!r} with {values.size} inputs")
    return values.astype(np.float64)


def probabilities_and_deltas(features, p, reference=None):
    """Baseline probability and all 16 signed Mean-Perturbation deltas per row.

    reference None: each pass sets the standardized input to 0 (the training mean, shipped
    images). Otherwise input i is set to reference[i] (load_reference)."""
    x = np.atleast_2d(np.asarray(features, dtype=np.float64))
    scaled = (x - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)
    probability = 1.0 / (1.0 + np.exp(-logits_from_scaled(scaled, p)))
    perturbed = np.repeat(scaled[:, None, :], N_FEATURES, axis=1)
    index = np.arange(N_FEATURES)
    perturbed[:, index, index] = 0.0 if reference is None else reference
    perturbed_probability = 1.0 / (1.0 + np.exp(-logits_from_scaled(perturbed, p)))
    return probability, probability[:, None] - perturbed_probability


def self_check_reference(p, reference, rows=64, seed=0):
    """Vectorized reference deltas must equal a per-row, per-input loop to 1e-12."""
    rng = np.random.default_rng(seed)
    x = p["scaler_mean"] + rng.normal(size=(rows, N_FEATURES)) * p["scaler_std"]
    probability, deltas = probabilities_and_deltas(x, p, reference)
    scaled = (x - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)
    loop = np.empty_like(deltas)
    for r in range(rows):
        for i in range(N_FEATURES):
            changed = scaled[r].copy()
            changed[i] = reference[i]
            loop[r, i] = probability[r] - 1.0 / (1.0 + np.exp(-logits_from_scaled(changed[None], p)[0]))
    error = float(np.max(np.abs(deltas - loop)))
    if error > 1e-12:
        raise AssertionError(f"Vectorized reference deltas differ from the loop by {error}")
    return error


def self_check(p, rows=64, seed=0):
    """Vectorized KAN must equal the v1 scalar reference (numpy_kan_predict) to 1e-12."""
    rng = np.random.default_rng(seed)
    x = p["scaler_mean"] + rng.normal(size=(rows, N_FEATURES)) * p["scaler_std"]
    vectorized, deltas = probabilities_and_deltas(x, p)
    scalar = np.array([numpy_kan_predict(row, p)[1] for row in x])
    scalar_deltas = []
    for row, base in zip(x, scalar):
        changed = np.repeat(row[None], N_FEATURES, axis=0)
        changed[np.arange(N_FEATURES), np.arange(N_FEATURES)] = p["scaler_mean"]
        scalar_deltas.append([base - numpy_kan_predict(c, p)[1] for c in changed])
    error = max(np.max(np.abs(vectorized - scalar)), np.max(np.abs(deltas - np.array(scalar_deltas))))
    if error > 1e-12:
        raise AssertionError(f"Vectorized KAN differs from numpy_kan_predict by {error}")
    return float(error)


def window_features(ir_raw, red_raw, ir_filtered, red_filtered, p):
    """16 inputs for one scored window; returns (features, two_wavelength_valid)."""
    f = _COMPUTE_IR(ir_filtered, 50.)
    ir_features = [f[name] for name in IR_FEATURE_NAMES]
    extra = two_wavelength_features(red_raw, red_filtered, ir_raw, ir_filtered)
    values = [extra[name] for name in EXTRA_FEATURE_NAMES]
    valid = bool(np.all(np.isfinite(values)))
    if not valid:
        values = list(p["scaler_mean"][len(IR_FEATURE_NAMES):])
    return np.array(ir_features + values, dtype=np.float64), valid


def decision_count(samples):
    return 0 if samples < FIRST else (samples - FIRST) // STRIDE + 1


def stream(ir, red, p, reference=None):
    """All decisions of a reset deployed pipeline fed paired FP32 samples ir[i], red[i]."""
    ir = np.asarray(ir, dtype=np.float32).astype(np.float64)
    red = np.asarray(red, dtype=np.float32).astype(np.float64)
    if ir.shape != red.shape or ir.ndim != 1 or not (np.isfinite(ir).all() and np.isfinite(red).all()):
        raise ValueError("Paired finite 1-D IR/red streams required")
    rows, valid = [], []
    meta = []
    for i in range(decision_count(len(ir))):
        start = i * STRIDE
        left, emitted = max(0, start - CONTEXT), start + FIRST
        offset = start - left
        ir_f = signal.sosfiltfilt(SOS, ir[left:emitted])[offset:offset + WINDOW]
        red_f = signal.sosfiltfilt(SOS, red[left:emitted])[offset:offset + WINDOW]
        features, ok = window_features(ir[start:start + WINDOW], red[start:start + WINDOW], ir_f, red_f, p)
        rows.append(features)
        valid.append(ok)
        meta.append([i, emitted, start, start + WINDOW, CONTEXT])
    if not rows:
        empty = np.zeros((0, N_FEATURES))
        return {"meta": np.zeros((0, 5), dtype=np.int64), "features": empty, "probability": np.zeros(0),
                "deltas": empty, "valid": np.zeros(0, dtype=bool), "candidate": np.zeros(0, dtype=bool),
                "confirmed": np.zeros(0, dtype=bool), "streak": np.zeros(0, dtype=np.int64),
                "top3": np.zeros((0, 3), dtype=np.int64)}
    features = np.array(rows)
    probability, deltas = probabilities_and_deltas(features, p, reference)
    candidate = probability >= THRESHOLD
    streak, run = [], 0
    for flag in candidate:
        run = min(255, run + 1) if flag else 0
        streak.append(run)
    streak = np.array(streak, dtype=np.int64)
    return {"meta": np.array(meta, dtype=np.int64), "features": features, "probability": probability,
            "deltas": deltas, "valid": np.array(valid), "candidate": candidate, "confirmed": streak >= DEBOUNCE,
            "streak": streak, "top3": np.argsort(-np.abs(deltas), axis=1, kind="stable")[:, :3]}

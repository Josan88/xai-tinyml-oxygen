"""Python reference of the DEPLOYED P5 path (job F2, ADR-0008), shared by the P5 host and board tools.

Deployed path = OxygenRiskPipeline (BasicOxygenRiskPipeline<250>) built with -DOXY_MODEL_P5
-DOXY_XAI_REF_LABEL0: paired 50 Hz IR/red samples; a decision every 250 samples from sample 750.
Each decision filters [max(0, start - 250), start + 750) of each channel with sosfiltfilt (the filter
path of scripts/v2_deployed_reference.py, imported) and scores the window [start, start + 500):
  DC   np.mean of the raw FP32 target window (float64), as v2's ratio inputs;
  beat scripts/physio_features.extract (the frozen extractor) on the float64 filtered windows;
       beat-invalid -> every input takes KAN_P5_IMPUTATION_MEAN (FP32), so z = 0;
  model the FP32-constant replica (lamaakal_models.HeaderKAN arithmetic: raw inputs rounded to FP32,
       header constants FP32, float64 arithmetic, u clamped to [-3, 3]) of kan_model_weights_p5.h;
  XAI  1 baseline + 5 passes, input i set to KAN_XAI_REF_Z[i] (kan_xai_reference_p5.h);
  s_hat the same arithmetic on kan_sat_regressor_p5.h: output * KANSAT_TARGET_STD + KANSAT_TARGET_MEAN;
  ADR-0005 candidate P >= 0.50, confirmed after N=2 consecutive candidates.
NumPy/SciPy only (no torch/sklearn), so the board-session interpreter can import it.
"""
from pathlib import Path
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import physio_features as pf  # noqa: E402
from train_tinykan_v2 import parse_header  # noqa: E402
from v2_deployed_reference import SOS, WINDOW, CONTEXT, STRIDE, FIRST, THRESHOLD, DEBOUNCE, decision_count  # noqa: E402,F401
from xai_reference import HEADER_P5 as REFERENCE_HEADER, parse_reference_header  # noqa: E402

INCLUDE = ROOT / "firmware/Arduino Nano 33/include"
HEADER = INCLUDE / "kan_model_weights_p5.h"
SAT_HEADER = INCLUDE / "kan_sat_regressor_p5.h"
FEATURE_NAMES = list(pf.FEATURES)
N_FEATURES = len(FEATURE_NAMES)


def _fp64(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def load_parameters():
    """Both networks, the scaler, the imputation mean, the target scaling and the label-0 reference."""
    w = {k: _fp64(v) for k, v in parse_header(HEADER).items()}
    s = {k: _fp64(v) for k, v in parse_header(SAT_HEADER).items()}
    tag, reference = parse_reference_header(REFERENCE_HEADER)
    if w["SCALER_MEAN"].shape != (N_FEATURES,) or w["KAN1_W"].shape != (12, N_FEATURES) or tag != "label0":
        raise ValueError("Not the 5-input P5 headers")
    net = lambda p, pre: {k: p[f"{pre}{n}"] for k, n in (("w1", "1_W"), ("b1", "1_B"), ("theta1", "1_THETA"),  # noqa: E731
                                                       ("centers1", "1_CENTERS"), ("step1", "1_STEP"), ("w2", "2_W"),
                                                       ("b2", "2_B"), ("theta2", "2_THETA"), ("centers2", "2_CENTERS"),
                                                       ("step2", "2_STEP"))}
    return {"scaler_mean": w["SCALER_MEAN"], "scaler_std": w["SCALER_STD"], "imputation": w["KAN_P5_IMPUTATION_MEAN"],
            "risk": net(w, "KAN"), "sat": net(s, "KANSAT"), "reference": reference.astype(np.float64),
            "target_mean": float(s["KANSAT_TARGET_MEAN"]), "target_std": float(s["KANSAT_TARGET_STD"])}


def _bspline(u, centers, step):
    dist = np.abs((np.clip(u, -3.0, 3.0)[..., None] - centers) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0, None)**3 / 6.0
    return np.where(dist < 1.0, inner, outer)


def network_output(scaled, net):
    u1 = scaled @ net["w1"].T + net["b1"]
    out1 = np.sum(_bspline(u1, net["centers1"], net["step1"]) * net["theta1"], axis=-1)
    u2 = out1 @ net["w2"].T + net["b2"]
    return np.sum(_bspline(u2[..., 0], net["centers2"], net["step2"]) * net["theta2"][0], axis=-1)


def standardize(features, p):
    return (_fp64(np.atleast_2d(features)) - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)


def evaluate(features, p):
    """Probability, the 5 label-0-reference deltas and s_hat for rows of model inputs."""
    scaled = standardize(features, p)
    probability = 1.0 / (1.0 + np.exp(-network_output(scaled, p["risk"])))
    perturbed = np.repeat(scaled[:, None, :], N_FEATURES, axis=1)
    index = np.arange(N_FEATURES)
    perturbed[:, index, index] = p["reference"]
    deltas = probability[:, None] - 1.0 / (1.0 + np.exp(-network_output(perturbed, p["risk"])))
    s_hat = network_output(scaled, p["sat"]) * p["target_std"] + p["target_mean"]
    return probability, deltas, s_hat


def model_inputs(beat, p):
    """The five inputs the device feeds the model: beat features, or the imputation mean when beat-invalid."""
    if beat["beat_valid"]:
        return np.array([beat[name] for name in FEATURE_NAMES], dtype=np.float64)
    return p["imputation"].copy()


def window_filter(ir, red, start):
    left, emitted = max(0, start - CONTEXT), start + FIRST
    offset = start - left
    ir_f = signal.sosfiltfilt(SOS, ir[left:emitted])[offset:offset + WINDOW]
    red_f = signal.sosfiltfilt(SOS, red[left:emitted])[offset:offset + WINDOW]
    return ir_f, red_f, float(np.mean(ir[start:start + WINDOW])), float(np.mean(red[start:start + WINDOW]))


def stream(ir, red, p):
    """All decisions of a reset deployed P5 pipeline fed paired FP32 samples ir[i], red[i]."""
    ir = _fp64(ir)
    red = _fp64(red)
    if ir.shape != red.shape or ir.ndim != 1 or not (np.isfinite(ir).all() and np.isfinite(red).all()):
        raise ValueError("Paired finite 1-D IR/red streams required")
    rows, meta, beats = [], [], []
    for i in range(decision_count(len(ir))):
        start = i * STRIDE
        ir_f, red_f, dc_ir, dc_red = window_filter(ir, red, start)
        beat = pf.extract(ir_f, red_f, dc_ir, dc_red)
        usable = bool(np.std(ir_f) > 0 and dc_ir > 0 and dc_red > 0)
        peaks = pf.detect_beats(ir_f)[1].tolist() if usable else []
        beats.append({"beat_valid": bool(beat["beat_valid"]), "n_beats": int(beat["n_beats"]),
                      "polarity": int(beat["polarity"]), "peaks": peaks,
                      "features": [float(beat[n]) for n in FEATURE_NAMES], "dc_ir": dc_ir, "dc_red": dc_red})
        rows.append(model_inputs(beat, p))
        meta.append([i, start + FIRST, start, start + WINDOW, CONTEXT])
    features = np.array(rows).reshape(-1, N_FEATURES)
    if len(features):
        probability, deltas, s_hat = evaluate(features, p)
    else:
        probability, deltas, s_hat = np.zeros(0), np.zeros((0, N_FEATURES)), np.zeros(0)
    candidate = probability >= THRESHOLD
    streak, run = [], 0
    for flag in candidate:
        run = min(255, run + 1) if flag else 0
        streak.append(run)
    streak = np.array(streak, dtype=np.int64)
    return {"meta": np.array(meta, dtype=np.int64).reshape(-1, 5), "features": features, "probability": probability,
            "deltas": deltas, "s_hat": s_hat, "beats": beats, "candidate": candidate, "confirmed": streak >= DEBOUNCE,
            "streak": streak, "valid": np.array([b["beat_valid"] for b in beats], dtype=bool),
            "n_beats": np.array([b["n_beats"] for b in beats], dtype=np.int64),
            "top3": np.argsort(-np.abs(deltas), axis=1, kind="stable")[:, :3]}


def self_check(p, rows=64, seed=0):
    """Vectorized deltas must equal a per-row, per-input loop to 1e-12."""
    rng = np.random.default_rng(seed)
    x = p["scaler_mean"] + rng.normal(size=(rows, N_FEATURES)) * p["scaler_std"]
    probability, deltas, _ = evaluate(x, p)
    scaled = standardize(x, p)
    worst = 0.0
    for r in range(rows):
        for i in range(N_FEATURES):
            changed = scaled[r].copy()
            changed[i] = p["reference"][i]
            loop = probability[r] - 1.0 / (1.0 + np.exp(-network_output(changed[None], p["risk"])[0]))
            worst = max(worst, abs(loop - deltas[r, i]))
    if worst > 1e-12:
        raise AssertionError(f"Vectorized P5 deltas differ from the loop by {worst}")
    return worst

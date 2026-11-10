"""Host-only int8 / LUT / pruning ablation rows for the v1 TinyKAN (14, 12, 1). Never deployed.

Jason, 24 Oct 2026: "Host-only ablation rows: quantise/prune copies of v1, report AUC/size
change, label them 'never deployed'. Shipped FP32 model untouched."

Every quantised or pruned model built here is an analysis model. None is written to a header,
none replaces the shipped FP32 v1 (ADR-0004), and none ran on the board, so latency, RAM and
energy are "not measured (never deployed)" for every variant.

Two evaluations per variant (variants are fixed in VARIANTS before any run):
  (a) out-of-fold: the 5 fold models are retrained with the exact v1 CV recipe (notebook 12:
      80 epochs, seed 42) on the split_index folds, in one freshly spawned process, as Section C
      of the Lamaakal-style analysis did. Their out-of-fold P is checked against the frozen
      openox_tinykan_predictions.csv first. Each variant is then applied to each fold model and
      scored on its held-out fold. The paired recording bootstrap of Section C
      (scripts/lamaakal_metrics.group_bootstrap) gives the 95% CI of AUC(variant) - AUC(FP32).
  (b) shipped header: the variant is applied to the shipped kan_model_weights.h constants and run
      on the header's own training table (in-sample, as the AOPC run): |dP| vs FP32, candidate
      flag (P >= 0.50) disagreements, and mean-perturbation top-1 / top-3 agreement.

Quantisation is simulated: integer values are dequantised to float64 for inference. The
explainer is mean-perturbation (standardized feature set to its training mean 0, N + 1 passes,
ranked by |delta| with ties broken by feature index). The split is recording-grouped.

Run (repository root, the interpreter with torch 2.12 / sklearn 1.9):
  C:\\Python314\\python.exe scripts/ablate_quant_prune_v1.py                 # outputs/analysis-quant-prune-v1/
  C:\\Python314\\python.exe scripts/ablate_quant_prune_v1.py --output build/quant-prune-rerun
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lamaakal_metrics as lm  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "analysis-quant-prune-v1"
P = {
    "v1_features": ROOT / "outputs/openox_canonical_window_features.csv",
    "v1_tinykan_oof": ROOT / "outputs/openox_tinykan_predictions.csv",
    "v1_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h",
    "kan_inference": ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
    "e_ablation": ROOT / "outputs/analysis-lamaakal-style-v1/e_ablation.csv",
    "f_deployment": ROOT / "outputs/analysis-lamaakal-style-v1/f_deployment.csv",
}
# Write-once directories of docs/analysis-lamaakal-style.md plus that analysis' own output.
WRITE_ONCE = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/ptt-ppg/subject-1-v1",
    ROOT / "outputs/ptt_ppg_motion_spike.csv",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
    ROOT / "outputs/analysis-lamaakal-style-v1",
)
UNTOUCHED_FILES = (
    ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h",
    ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
    ROOT / "outputs/openox_tinykan_predictions.csv",
)

IR_FEATURES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
THRESHOLD = 0.50
BOOTSTRAP_SEED = 20261024     # Section C base seed; one paired draw set shared by every variant
N_BOOT = 2000
REPRODUCTION_TOLERANCE = 1e-5
V1_RECIPE = {"hidden": 12, "knots": 8, "spline_range": (-3.0, 3.0), "epochs": 80, "seed": 42}
SPLINE_LO, SPLINE_HI = -3.0, 3.0
TENSORS = ("W1", "b1", "T1", "W2", "b2", "T2")   # the 297 learned parameters, in header order
N_PARAMS = 297
SCALER_FLOATS = 28            # SCALER_MEAN[14] + SCALER_STD[14]
CENTER_STEP_FLOATS = 18       # KAN1/KAN2 centres (8 + 8) + KAN1/KAN2 step (2)
THRESHOLD_FLOATS = 1
INPUT_CLIP_PERCENTILE = 99.9  # int8 input-activation calibration (per feature, training rows only)
FINE_TUNE = {"epochs": 20, "lr": 1e-3, "weight_decay": 1e-4, "smoothness": 1e-5, "clip": 10.0}
NEVER = "not measured (never deployed)"
INVALIDATES = ("board parity (2,464/2,464 locked windows), the 249.7 ms timing, the on-device "
               "mean-perturbation payload and every host replica check of kan_model_weights.h")

# Fixed before any run. weights: None | per_tensor | per_channel. lut: None | table entries.
VARIANTS = (
    {"key": "fp32", "label": "FP32 (reference)", "family": "FP32", "weights": None, "act": False, "lut": None, "prune": 0.0, "fine_tune": False},
    {"key": "int8_pt", "label": "int8 weights, per-tensor", "family": "int8", "weights": "per_tensor", "act": False, "lut": None, "prune": 0.0, "fine_tune": False},
    {"key": "int8_pc", "label": "int8 weights, per-channel", "family": "int8", "weights": "per_channel", "act": False, "lut": None, "prune": 0.0, "fine_tune": False},
    {"key": "int8_pc_act", "label": "int8 weights (per-channel) + int8 activations", "family": "int8", "weights": "per_channel", "act": True, "lut": None, "prune": 0.0, "fine_tune": False},
    {"key": "int8_lut64", "label": "int8 weights (per-channel) + int8 LUT-64 splines", "family": "int8 + LUT", "weights": "per_channel", "act": False, "lut": 64, "prune": 0.0, "fine_tune": False},
    {"key": "int8_lut256", "label": "int8 weights (per-channel) + int8 LUT-256 splines", "family": "int8 + LUT", "weights": "per_channel", "act": False, "lut": 256, "prune": 0.0, "fine_tune": False},
    {"key": "int8_act_lut256", "label": "int8 weights + int8 activations + int8 LUT-256", "family": "int8 + LUT", "weights": "per_channel", "act": True, "lut": 256, "prune": 0.0, "fine_tune": False},
    {"key": "prune10", "label": "magnitude prune 10%, no fine-tune", "family": "pruned", "weights": None, "act": False, "lut": None, "prune": 0.10, "fine_tune": False},
    {"key": "prune25", "label": "magnitude prune 25%, no fine-tune", "family": "pruned", "weights": None, "act": False, "lut": None, "prune": 0.25, "fine_tune": False},
    {"key": "prune50", "label": "magnitude prune 50%, no fine-tune", "family": "pruned", "weights": None, "act": False, "lut": None, "prune": 0.50, "fine_tune": False},
    {"key": "prune10_ft", "label": "magnitude prune 10% + fine-tune", "family": "pruned + fine-tune", "weights": None, "act": False, "lut": None, "prune": 0.10, "fine_tune": True},
    {"key": "prune25_ft", "label": "magnitude prune 25% + fine-tune", "family": "pruned + fine-tune", "weights": None, "act": False, "lut": None, "prune": 0.25, "fine_tune": True},
    {"key": "prune50_ft", "label": "magnitude prune 50% + fine-tune", "family": "pruned + fine-tune", "weights": None, "act": False, "lut": None, "prune": 0.50, "fine_tune": True},
)


# ----------------------------------------------------------------------------- quantise / LUT / prune

def quantize_symmetric(w, axis=None, bits=8):
    """Symmetric int-b quantisation, zero-point 0: q = clip(round(w / s), -(2^(b-1) - 1), 2^(b-1) - 1).

    s = max|w| / (2^(b-1) - 1) over the whole tensor (axis None) or per slice along `axis`
    (axis=0: one scale per row). An all-zero slice gets s = 1 so that q = 0 dequantises to 0.
    Returns (q as int array, s broadcastable to w). Lamaakal et al. eq. (42) clips to
    [-2^(b-1), 2^(b-1) - 1]; with s = max|w| / 127 the value -128 is never reached.
    """
    w = np.asarray(w, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    if axis is None:
        peak = np.max(np.abs(w)) if w.size else 0.0
        scale = np.asarray(peak / qmax if peak > 0 else 1.0)
    else:
        reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
        peak = np.max(np.abs(w), axis=reduce_axes, keepdims=True)
        scale = np.where(peak > 0, peak / qmax, 1.0)
    q = np.clip(np.rint(w / scale), -qmax, qmax).astype(np.int32)
    return q, scale


def fake_quantize(w, axis=None, bits=8):
    q, scale = quantize_symmetric(w, axis, bits)
    return q * scale


def fake_quantize_with_scale(x, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) / scale), -qmax, qmax) * scale


def bspline_basis(u, centers, step):
    """Firmware basis of kan_inference.h: u clamped to [-3, 3], cubic B-spline of |u - c| / step."""
    distance = np.abs((np.clip(u, SPLINE_LO, SPLINE_HI)[..., None] - centers) / (step + 1e-10))
    inner = 2.0 / 3.0 - distance ** 2 + 0.5 * distance ** 3
    outer = np.maximum(2.0 - distance, 0.0) ** 3 / 6.0
    return np.where(distance < 1.0, inner, outer)


def phi(u, theta, centers, step):
    """phi(u) = sum_k B_k(u) theta_k for one node."""
    return bspline_basis(np.asarray(u, dtype=np.float64), centers, step) @ np.asarray(theta, dtype=np.float64)


def lut_grid(entries):
    return np.linspace(SPLINE_LO, SPLINE_HI, entries)


def build_lut(theta, centers, step, entries, int8=True):
    """Tabulate phi on `entries` uniform points over the clamp range [-3, 3]; optionally int8 per table."""
    table = phi(lut_grid(entries), theta, centers, step)
    if not int8:
        return table, 1.0
    q, scale = quantize_symmetric(table)
    return q * scale, float(scale)


def lut_eval(u, table):
    """Linear interpolation in a uniform table over [-3, 3], u clamped as the firmware clamps it."""
    entries = len(table)
    h = (SPLINE_HI - SPLINE_LO) / (entries - 1)
    t = (np.clip(np.asarray(u, dtype=np.float64), SPLINE_LO, SPLINE_HI) - SPLINE_LO) / h
    index = np.minimum(np.floor(t).astype(np.int64), entries - 2)
    frac = t - index
    return (1.0 - frac) * table[index] + frac * table[index + 1]


def lut_error_bound(theta, centers, step, entries, table_scale=0.0):
    """Linear-interpolation bound h^2/8 * max|phi''| plus half an int8 step of the table.

    phi'' of a cubic B-spline sum is piecewise linear, so its maximum magnitude sits on a knot
    of the basis; evaluating it on a fine grid that contains every knot gives max|phi''|.
    """
    theta = np.asarray(theta, dtype=np.float64)
    u = np.unique(np.concatenate([np.linspace(SPLINE_LO, SPLINE_HI, 6001), np.asarray(centers, dtype=np.float64)]))
    d = (u[:, None] - centers) / step
    a = np.abs(d)
    second = np.where(a < 1.0, -2.0 + 3.0 * a, np.where(a < 2.0, 2.0 - a, 0.0)) / step ** 2
    peak = float(np.max(np.abs(second @ theta)))
    h = (SPLINE_HI - SPLINE_LO) / (entries - 1)
    return h * h / 8.0 * peak + 0.5 * table_scale


def global_prune_mask(params, rate):
    """Global magnitude pruning of floor(rate * total) learned parameters.

    Ranked by |w| over all tensors (TENSORS order, then flat index, so ties are deterministic);
    the smallest are set to zero. Returns {tensor: bool keep-mask}.
    """
    flat = np.concatenate([np.abs(np.asarray(params[name], dtype=np.float64)).ravel() for name in TENSORS])
    k = int(np.floor(rate * flat.size))
    keep = np.ones(flat.size, dtype=bool)
    if k > 0:
        keep[np.argsort(flat, kind="stable")[:k]] = False
    masks, offset = {}, 0
    for name in TENSORS:
        size = np.asarray(params[name]).size
        masks[name] = keep[offset:offset + size].reshape(np.asarray(params[name]).shape)
        offset += size
    return masks


def apply_mask(params, masks):
    return {name: (np.asarray(params[name], dtype=np.float64) * masks[name]) if name in masks
            else params[name] for name in params}


# ----------------------------------------------------------------------------- numpy model

class VariantKAN:
    """Host model of kan_inference.h for one variant, float64 arithmetic on (dequantised) constants.

    params: W1 (12, 14), b1 (12,), T1 (12, 8), W2 (1, 12), b2 (1,), T2 (1, 8); centres/steps are fixed
    buffers (not learned). With no variant options it is the same arithmetic as HeaderKAN.

    int8 activations (act=True), simulated: standardized input x per-feature int8 (scale from the
    INPUT_CLIP_PERCENTILE of |x| on the calibration rows, saturating), pre-activations u1 and u2
    int8 on the clamp range (scale 3/127, so the clamp is the int8 range), hidden outputs phi_j(u1)
    int8 per unit (scale max_u |phi_j(u)| / 127 over [-3, 3], data-independent). The output logit
    phi_2(u2) stays float before the sigmoid.

    LUT (lut=L): every learned univariate function, the 12 hidden phi_j and the output phi_2, is
    replaced by an L-entry int8 table over [-3, 3] with linear interpolation; theta and the B-spline
    basis are not used at inference.
    """

    def __init__(self, params, fixed, weights=None, act=False, lut=None, calibration_x=None):
        self.fixed = {k: np.asarray(v, dtype=np.float64) for k, v in fixed.items()}
        p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        self.scales = 0
        if weights == "per_tensor":
            p = {k: fake_quantize(v) for k, v in p.items()}
            self.scales = len(TENSORS)
        elif weights == "per_channel":
            # One scale per hidden unit for W1 rows and T1 rows; W2, T2 are single rows; biases per tensor.
            p = {"W1": fake_quantize(p["W1"], axis=0), "b1": fake_quantize(p["b1"]), "T1": fake_quantize(p["T1"], axis=0),
                 "W2": fake_quantize(p["W2"], axis=0), "b2": fake_quantize(p["b2"]), "T2": fake_quantize(p["T2"], axis=0)}
            self.scales = p["W1"].shape[0] + 1 + p["T1"].shape[0] + 1 + 1 + 1
        self.p = p
        self.act = act
        self.lut = lut
        hidden = p["W1"].shape[0]
        if lut:
            self.tables1, self.table_scales1 = zip(*(build_lut(p["T1"][j], self.fixed["C1"], float(self.fixed["S1"]), lut)
                                                    for j in range(hidden)))
            self.table2, self.table_scale2 = build_lut(p["T2"][0], self.fixed["C2"], float(self.fixed["S2"]), lut)
            self.tables1 = np.stack(self.tables1)
        if act:
            if calibration_x is None:
                raise ValueError("int8 activations need calibration rows")
            peak = np.percentile(np.abs(np.asarray(calibration_x, dtype=np.float64)), INPUT_CLIP_PERCENTILE, axis=0)
            self.input_scale = np.where(peak > 0, peak / 127.0, 1.0)
            self.u_scale = SPLINE_HI / 127.0
            grid = np.linspace(SPLINE_LO, SPLINE_HI, 6001)
            h_grid = self._hidden_phi(np.repeat(grid[:, None], hidden, axis=1))
            peak_h = np.max(np.abs(h_grid), axis=0)
            self.hidden_scale = np.where(peak_h > 0, peak_h / 127.0, 1.0)

    def _hidden_phi(self, u1):
        if self.lut:
            return np.stack([lut_eval(u1[:, j], self.tables1[j]) for j in range(u1.shape[1])], axis=1)
        return np.sum(bspline_basis(u1, self.fixed["C1"], float(self.fixed["S1"])) * self.p["T1"][None, :, :], axis=2)

    def _output_phi(self, u2):
        if self.lut:
            return lut_eval(u2, self.table2)
        return bspline_basis(u2, self.fixed["C2"], float(self.fixed["S2"])) @ self.p["T2"][0]

    def logit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.act:
            x = fake_quantize_with_scale(x, self.input_scale)
        u1 = x @ self.p["W1"].T + self.p["b1"]
        if self.act:
            u1 = fake_quantize_with_scale(np.clip(u1, SPLINE_LO, SPLINE_HI), self.u_scale)
        h = self._hidden_phi(u1)
        if self.act:
            h = fake_quantize_with_scale(h, self.hidden_scale)
        u2 = h @ self.p["W2"][0] + self.p["b2"][0]
        if self.act:
            u2 = fake_quantize_with_scale(np.clip(u2, SPLINE_LO, SPLINE_HI), self.u_scale)
        return self._output_phi(u2)

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-self.logit(x)))

    def mean_perturbation(self, x):
        x = np.asarray(x, dtype=np.float64)
        probability = self.predict(x)
        attribution = np.empty_like(x)
        for feature in range(x.shape[1]):
            changed = x.copy()
            changed[:, feature] = 0.0
            attribution[:, feature] = probability - self.predict(changed)
        return probability, attribution


def build_variant(spec, params, fixed, calibration_x):
    if spec["prune"] > 0 and not spec["fine_tune"]:
        params = apply_mask(params, global_prune_mask(params, spec["prune"]))
    return VariantKAN(params, fixed, weights=spec["weights"], act=spec["act"], lut=spec["lut"],
                      calibration_x=calibration_x if spec["act"] else None)


# ----------------------------------------------------------------------------- sizes

def variant_sizes(spec, params_for_count, hidden=12, n_inputs=14, knots=8):
    """Byte accounting. Scales are FP32 (4 B); symmetric, so no zero-points are stored."""
    nonzero = int(sum(np.count_nonzero(np.asarray(params_for_count[name])) for name in TENSORS))
    linear = hidden * n_inputs + hidden + hidden + 1          # W1, b1, W2, b2 = 193
    spline = hidden * knots + knots                            # T1, T2 = 104
    fixed_floats = SCALER_FLOATS + THRESHOLD_FLOATS
    row = {"params": N_PARAMS, "nonzero_params": nonzero, "fp32_bytes": 4 * N_PARAMS,
           "fp32_nonzero_value_bytes": 4 * nonzero, "int8_param_bytes": None, "scale_bytes": None, "lut_bytes": None,
           "int8_bytes": None, "header_constant_bytes": None, "lut_entries_per_function": spec["lut"],
           "lut_functions": None}
    if spec["weights"] is None:
        row["header_constant_bytes"] = 4 * (N_PARAMS + fixed_floats + CENTER_STEP_FLOATS)
        return row
    if spec["weights"] == "per_tensor":
        scales = len(TENSORS)
    else:
        scales = hidden + 1 + (0 if spec["lut"] else hidden) + 1 + 1 + (0 if spec["lut"] else 1)
    act_scales = (n_inputs + hidden + 1) if spec["act"] else 0          # input, hidden-output, shared u scale
    if spec["lut"]:
        functions = hidden + 1
        int8_params = linear
        lut_bytes = functions * spec["lut"]
        scales += functions                                              # one scale per table
        grid_floats = 2                                                  # table range [-3, 3]
        row.update({"lut_functions": functions, "lut_bytes": lut_bytes})
        int8_bytes = int8_params + lut_bytes + 4 * (scales + act_scales + grid_floats)
        header = int8_bytes + 4 * fixed_floats
    else:
        int8_params = linear + spline
        int8_bytes = int8_params + 4 * (scales + act_scales)
        header = int8_bytes + 4 * (fixed_floats + CENTER_STEP_FLOATS)
    row.update({"int8_param_bytes": int8_params, "scale_bytes": 4 * (scales + act_scales + (2 if spec["lut"] else 0)),
                "int8_bytes": int8_bytes, "header_constant_bytes": header})
    return row


# ----------------------------------------------------------------------------- training (worker side)

def torch_to_params(model):
    l1, l2 = model.layers
    params = {"W1": l1.W, "b1": l1.b, "T1": l1.theta, "W2": l2.W, "b2": l2.b, "T2": l2.theta}
    params = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in params.items()}
    fixed = {"C1": l1.spline.centers, "S1": l1.spline.step, "C2": l2.spline.centers, "S2": l2.spline.step}
    fixed = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in fixed.items()}
    return params, fixed


def train_folds_job(payload):
    """All 5 folds in one fresh process, in the order and arithmetic of lamaakal_models.run_cv_job."""
    import torch
    from sklearn.preprocessing import StandardScaler
    import lamaakal_models as models

    x, y, folds = payload
    started = time.perf_counter()
    spec = {"name": "v1_tinykan_seed42_qp", "learner": "tinykan", **V1_RECIPE}
    out = {}
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train]).astype(np.float32)
        x_test = scaler.transform(x[test]).astype(np.float32)
        model = models.train_tinykan(x_train, y[train], spec["hidden"], spec["knots"], tuple(spec["spline_range"]),
                                     spec["epochs"], spec["seed"])
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32))).numpy().ravel()
        params, fixed = torch_to_params(model)
        out[int(fold)] = {"params": params, "fixed": fixed, "p_torch": p.astype(np.float64),
                          "x_train": x_train, "x_test": x_test, "y_train": y[train]}
    return {"folds": out, "seconds": time.perf_counter() - started}


def fine_tune_job(payload):
    """Masked fine-tune of one pruned fold model on its training fold only (FINE_TUNE settings)."""
    import torch
    import torch.nn as nn
    from train_tinykan_v2 import KANBinaryClassifier

    fold, rate, params, x_train, y_train = payload
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(V1_RECIPE["seed"])
    model = KANBinaryClassifier((x_train.shape[1], V1_RECIPE["hidden"], 1), V1_RECIPE["knots"], 3, V1_RECIPE["spline_range"])
    masks = global_prune_mask(params, rate)
    pruned = apply_mask(params, masks)
    l1, l2 = model.layers
    tensors = {"W1": l1.W, "b1": l1.b, "T1": l1.theta, "W2": l2.W, "b2": l2.b, "T2": l2.theta}
    mask_t = {}
    with torch.no_grad():
        for name, tensor in tensors.items():
            tensor.copy_(torch.tensor(pruned[name], dtype=torch.float32))
            mask_t[name] = torch.tensor(masks[name], dtype=torch.float32)
    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE["lr"], weight_decay=FINE_TUNE["weight_decay"])
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    model.train()
    for _ in range(FINE_TUNE["epochs"]):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t) + FINE_TUNE["smoothness"] * model.regularization()
        loss.backward()
        for name, tensor in tensors.items():
            tensor.grad.mul_(mask_t[name])
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=FINE_TUNE["clip"])
        optimizer.step()
        with torch.no_grad():
            for name, tensor in tensors.items():
                tensor.mul_(mask_t[name])
    model.eval()
    tuned, _ = torch_to_params(model)
    return {"fold": fold, "rate": rate, "params": tuned, "seconds": time.perf_counter() - started}


# ----------------------------------------------------------------------------- utilities

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(path):
    path = Path(path)
    if path.is_file():
        return {path.relative_to(ROOT).as_posix(): sha256(path)}
    return {f.relative_to(ROOT).as_posix(): sha256(f) for f in sorted(path.rglob("*")) if f.is_file()}


def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.8g", lineterminator="\n")


def header_params(path):
    from train_tinykan_v2 import parse_header

    raw = parse_header(path)
    params = {"W1": raw["KAN1_W"], "b1": raw["KAN1_B"], "T1": raw["KAN1_THETA"],
              "W2": raw["KAN2_W"], "b2": raw["KAN2_B"], "T2": raw["KAN2_THETA"]}
    fixed = {"C1": raw["KAN1_CENTERS"], "S1": raw["KAN1_STEP"], "C2": raw["KAN2_CENTERS"], "S2": raw["KAN2_STEP"]}
    to64 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
    return ({k: to64(v) for k, v in params.items()}, {k: to64(v) for k, v in fixed.items()},
            to64(raw["SCALER_MEAN"]), to64(raw["SCALER_STD"]))


# ----------------------------------------------------------------------------- figure

def plot(table, path):
    from analyze_lamaakal_style import BLUE, GREEN, INK, INK2, MUTED, ORANGE, COL2, setup_style

    plt = setup_style()
    style = {"FP32": (INK, "o"), "int8": (BLUE, "s"), "int8 + LUT": (ORANGE, "D"),
             "pruned": (GREEN, "^"), "pruned + fine-tune": (GREEN, "v")}
    fig, (left, right) = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    for family, block in table.groupby("family", sort=False):
        colour, marker = style[family]
        stored = block["int8_bytes"].fillna(block["fp32_nonzero_value_bytes"])
        filled = family != "pruned"
        left.scatter(stored, block["oof_auc"], s=22, marker=marker, color=colour if filled else "white",
                     edgecolors=colour, linewidths=0.9, label=family, zorder=3)
    # (text, offset in points, horizontal alignment): placed so no label overlaps a marker or another label.
    labels = {"fp32": ("FP32", (0, 6), "center"), "int8_pt": ("int8", (0, 6), "center"),
              "int8_lut64": ("LUT-64", (5, -9), "left"), "int8_lut256": ("LUT-256", (0, 6), "center"),
              "prune50": ("50% pruned", (6, -3), "left"), "prune50_ft": ("50% + FT", (6, 1), "left")}
    for _, r in table.iterrows():
        if r.key in labels:
            stored = r.int8_bytes if pd.notna(r.int8_bytes) else r.fp32_nonzero_value_bytes
            text, offset, ha = labels[r.key]
            left.annotate(text, (stored, r.oof_auc), xytext=offset, textcoords="offset points", fontsize=6,
                          color=INK2, ha=ha)
    left.set_xscale("log")
    left.set_xlabel("Stored weight bytes (log)\nint8: incl. scales and LUTs; FP32: non-zero values")
    left.set_ylabel("Out-of-fold AUC")
    left.set_title("(a) AUC vs stored bytes, never deployed")
    left.legend(loc="center left", handletextpad=0.3)

    rows = table[table.key != "fp32"].reset_index(drop=True)
    ypos = np.arange(len(rows))[::-1]
    for y_at, (_, r) in zip(ypos, rows.iterrows()):
        colour, marker = style[r.family]
        right.plot([r.delta_auc_ci_low, r.delta_auc_ci_high], [y_at, y_at], color=colour, linewidth=1.2, zorder=2)
        right.scatter([r.delta_auc], [y_at], marker=marker, s=20, color=colour if r.family != "pruned" else "white",
                      edgecolors=colour, linewidths=0.9, zorder=3)
    right.axvline(0.0, color=MUTED, linewidth=0.6, linestyle="--")
    right.set_yticks(ypos)
    right.set_yticklabels(rows["label"], fontsize=5.6)
    right.set_xlabel("AUC(variant) - AUC(FP32)\npaired recording bootstrap, 95% CI")
    right.set_title("(b) Out-of-fold AUC change vs FP32")
    right.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, dpi=300, facecolor="white", metadata={"Software": None}, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        raise FileExistsError(f"Write-once output exists: {out}")
    started = time.perf_counter()
    hashes_before = {str(p.relative_to(ROOT).as_posix()): tree_hashes(p) for p in WRITE_ONCE}
    untouched_before = {p.relative_to(ROOT).as_posix(): sha256(p) for p in UNTOUCHED_FILES}

    v1 = pd.read_csv(P["v1_features"])
    v1["recording_id"] = v1["recording_id"].astype(str)
    frozen = pd.read_csv(P["v1_tinykan_oof"]).sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(frozen["row_index"], np.arange(len(v1))):
        raise RuntimeError("Frozen predictions do not cover every canonical row once")
    if not (frozen["y_true"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("Frozen labels differ from the canonical table")
    x_raw = v1.loc[:, IR_FEATURES].to_numpy(np.float32)
    y = v1["label"].to_numpy(np.int64)
    folds = frozen["split_index"].to_numpy(int)
    groups = v1["recording_id"].to_numpy()
    p_frozen = frozen["y_prob"].to_numpy(np.float64)
    if len(y) != 52090 or len(np.unique(folds)) != 5:
        raise RuntimeError("Unexpected v1 table")
    if pd.Series(folds).groupby(groups).nunique().max() != 1:
        raise RuntimeError("A recording spans several folds")

    # (a) retrain the 5 fold models in one fresh process (Section C conditions).
    print("training 5 fold models (v1 CV recipe, 80 epochs, seed 42) ...", flush=True)
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        trained = pool.submit(train_folds_job, (x_raw, y, folds)).result()
    fold_ids = sorted(trained["folds"])
    p_torch = np.full(len(y), np.nan)
    for fold in fold_ids:
        p_torch[folds == fold] = trained["folds"][fold]["p_torch"]
    per_fold_diff = {f: float(np.max(np.abs(p_torch[folds == f] - p_frozen[folds == f]))) for f in fold_ids}
    reproduction = {
        "reference": "outputs/openox_tinykan_predictions.csv y_prob (notebook 12, frozen OOF)",
        "max_abs_diff": float(np.max(np.abs(p_torch - p_frozen))),
        "max_abs_diff_per_fold": per_fold_diff,
        "tolerance": REPRODUCTION_TOLERANCE,
        "folds_within_tolerance": [f for f, d in per_fold_diff.items() if d <= REPRODUCTION_TOLERANCE],
        "candidate_flag_disagreements_at_0_50": int(np.sum((p_torch >= THRESHOLD) != (p_frozen >= THRESHOLD))),
        "auc_retrain_minus_reference": lm.weighted_auc(y, p_torch) - lm.weighted_auc(y, p_frozen),
        "train_seconds": trained["seconds"],
        "section_c_reference_max_abs_diff": 1.233816146850586e-05,
    }
    print(f"  reproduction max |dP| vs frozen OOF: {reproduction['max_abs_diff']:.3e}", flush=True)
    if reproduction["max_abs_diff"] > 1e-4 or reproduction["candidate_flag_disagreements_at_0_50"] > 0:
        raise RuntimeError(f"Fold retrain does not reproduce the frozen OOF: {reproduction}")

    # Pruned + fine-tune fold models, each in its own fresh process.
    rates = sorted({v["prune"] for v in VARIANTS if v["fine_tune"]})
    payloads = [(f, r, trained["folds"][f]["params"], trained["folds"][f]["x_train"], trained["folds"][f]["y_train"])
                for f in fold_ids for r in rates]
    print(f"fine-tuning {len(payloads)} pruned fold models ...", flush=True)
    tuned = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs), max_tasks_per_child=1) as pool:
        for result in pool.map(fine_tune_job, payloads):
            tuned[(result["fold"], result["rate"])] = result

    # Score every variant out-of-fold.
    oof = {v["key"]: np.full(len(y), np.nan) for v in VARIANTS}
    fold_nonzero = {v["key"]: [] for v in VARIANTS}
    numpy_vs_torch = 0.0
    for fold in fold_ids:
        entry = trained["folds"][fold]
        test = folds == fold
        params64 = {k: v.astype(np.float64) for k, v in entry["params"].items()}
        fixed64 = {k: v.astype(np.float64) for k, v in entry["fixed"].items()}
        for spec in VARIANTS:
            base = params64
            if spec["fine_tune"]:
                base = {k: v.astype(np.float64) for k, v in tuned[(fold, spec["prune"])]["params"].items()}
            model = build_variant(spec, base, fixed64, entry["x_train"])
            oof[spec["key"]][test] = model.predict(entry["x_test"])
            counted = apply_mask(base, global_prune_mask(base, spec["prune"])) if (spec["prune"] and not spec["fine_tune"]) else base
            fold_nonzero[spec["key"]].append(int(sum(np.count_nonzero(model.p[n]) for n in TENSORS)) if not spec["lut"]
                                             else int(sum(np.count_nonzero(counted[n]) for n in TENSORS)))
        numpy_vs_torch = max(numpy_vs_torch, float(np.max(np.abs(oof["fp32"][test] - entry["p_torch"]))))
    for key, p in oof.items():
        if not np.isfinite(p).all():
            raise RuntimeError(f"{key}: incomplete out-of-fold coverage")
    reproduction["numpy_fp32_vs_torch_fold_models_max_abs_p"] = numpy_vs_torch

    print("paired recording bootstrap ...", flush=True)
    point, draws = lm.group_bootstrap(y, oof, groups, N_BOOT, BOOTSTRAP_SEED, THRESHOLD)

    # (b) shipped header, in-sample.
    print("shipped header variants (in-sample) ...", flush=True)
    h_params, h_fixed, h_mean, h_std = header_params(P["v1_header"])
    scaled = (x_raw.astype(np.float64) - h_mean) / (h_std + 1e-10)
    from lamaakal_models import HeaderKAN

    replica = HeaderKAN(P["v1_header"])
    ref_model = VariantKAN(h_params, h_fixed)
    ref_p, ref_attr = ref_model.mean_perturbation(scaled)
    rep_p, rep_attr = replica.mean_perturbation(replica.standardize(x_raw))
    header_checks = {"variant_fp32_vs_HeaderKAN_max_abs_p": float(np.max(np.abs(ref_p - rep_p))),
                     "variant_fp32_vs_HeaderKAN_max_abs_delta": float(np.max(np.abs(ref_attr - rep_attr)))}
    ref_order = lm.firmware_abs_order(ref_attr)
    ref_flag = ref_p >= THRESHOLD

    rows = []
    for spec in VARIANTS:
        m = lm.binary_metrics(y, oof[spec["key"]], THRESHOLD)
        diff = {metric: draws[spec["key"]][metric] - draws["fp32"][metric] for metric in ("auc", "recall", "specificity")}
        ci = {metric: lm.percentile_ci(values) for metric, values in diff.items()}
        dp = oof[spec["key"]] - oof["fp32"]
        sizes = variant_sizes(spec, apply_mask(h_params, global_prune_mask(h_params, spec["prune"])) if spec["prune"] else h_params)
        row = {"key": spec["key"], "variant": f"v1 {spec['label']}", "label": spec["label"], "family": spec["family"],
               "learner": "TinyKAN", "source": "analysis variant of v1 fold retrains (never deployed)" if spec["key"] != "fp32"
               else "v1 fold retrain, FP32 (reproduces frozen OOF)",
               "labels": "as-coded", "inputs": "14 IR", "n_inputs": 14, "hidden": 12, "spline_range": "[-3, 3]", "knots": 8,
               "epochs": 80, "folds": "v1 recording-grouped split_index (5)", "rows": m["n"],
               "weight_quant": spec["weights"] or "none (FP32)", "activation_quant": "int8" if spec["act"] else "none",
               "spline_eval": f"int8 LUT-{spec['lut']} linear interp." if spec["lut"] else "cubic B-spline (firmware basis)",
               "prune_rate": spec["prune"], "fine_tune": f"{FINE_TUNE['epochs']} epochs masked, training fold only" if spec["fine_tune"] else "no",
               "oof_auc": m["auc"], "oof_pr_auc": m["pr_auc"], "oof_recall_050": m["recall"], "oof_specificity_050": m["specificity"],
               "oof_balanced_accuracy_050": m["balanced_accuracy"],
               "delta_auc": point[spec["key"]]["auc"] - point["fp32"]["auc"], "delta_auc_ci_low": ci["auc"][0], "delta_auc_ci_high": ci["auc"][1],
               "delta_recall_050": point[spec["key"]]["recall"] - point["fp32"]["recall"],
               "delta_recall_ci_low": ci["recall"][0], "delta_recall_ci_high": ci["recall"][1],
               "delta_specificity_050": point[spec["key"]]["specificity"] - point["fp32"]["specificity"],
               "delta_specificity_ci_low": ci["specificity"][0], "delta_specificity_ci_high": ci["specificity"][1],
               "oof_max_abs_dp_vs_fp32": float(np.max(np.abs(dp))),
               "oof_flag_disagreements_vs_fp32": int(np.sum((oof[spec["key"]] >= THRESHOLD) != (oof["fp32"] >= THRESHOLD))),
               "oof_nonzero_params_per_fold": " ".join(str(n) for n in fold_nonzero[spec["key"]])}
        if spec["fine_tune"]:
            row.update({"header_max_abs_dp": None, "header_median_abs_dp": None, "header_flag_disagreements": None,
                        "header_flag_disagreement_rate": None, "header_top1_agreement": None, "header_top3_ordered_agreement": None,
                        "header_top3_set_agreement": None, "header_note": "not run: fine-tuning the shipped header would need its training run (out of scope)"})
        else:
            hv = build_variant(spec, h_params, h_fixed, scaled)
            hp, hattr = hv.mean_perturbation(scaled)
            order = lm.firmware_abs_order(hattr)
            abs_dp = np.abs(hp - ref_p)
            flips = int(np.sum((hp >= THRESHOLD) != ref_flag))
            row.update({"header_max_abs_dp": float(abs_dp.max()), "header_median_abs_dp": float(np.median(abs_dp)),
                        "header_flag_disagreements": flips, "header_flag_disagreement_rate": flips / len(y),
                        "header_top1_agreement": float(np.mean(order[:, 0] == ref_order[:, 0])),
                        "header_top3_ordered_agreement": float(np.mean(np.all(order[:, :3] == ref_order[:, :3], axis=1))),
                        "header_top3_set_agreement": float(np.mean(np.all(np.sort(order[:, :3], axis=1) == np.sort(ref_order[:, :3], axis=1), axis=1))),
                        "header_note": "shipped kan_model_weights.h on its own 52,090 training windows (in-sample)"})
        row.update(sizes)
        row.update({"board_decision_block_median_ms": None, "board_latency": NEVER, "board_peak_ram": NEVER, "board_energy": NEVER,
                    "board_note": "shipped FP32 v1: 249.7 ms whole decision block (f_deployment.csv); this variant never ran on the board"
                    if spec["key"] == "fp32" else NEVER,
                    "deploying_would_invalidate": "nothing (reference arithmetic)" if spec["key"] == "fp32" else INVALIDATES})
        rows.append(row)
    table = pd.DataFrame(rows)

    out.mkdir(parents=True)
    write_csv(table, out / "ablation_quant_prune.csv")
    # Per-tensor pruned counts on the shipped header (where global magnitude pruning bites).
    prune_rows = []
    for rate in sorted({v["prune"] for v in VARIANTS if v["prune"]}):
        masks = global_prune_mask(h_params, rate)
        for name in TENSORS:
            prune_rows.append({"source": "shipped v1 header", "prune_rate": rate, "tensor": name, "size": int(masks[name].size),
                               "pruned": int((~masks[name]).sum()), "kept": int(masks[name].sum())})
    write_csv(pd.DataFrame(prune_rows), out / "prune_mask_by_tensor.csv")
    # LUT approximation error on the shipped header splines (before any int8 of the linear weights).
    lut_rows = []
    for entries in (64, 256):
        for layer, unit, theta, centers, step in ([(1, j, h_params["T1"][j], h_fixed["C1"], float(h_fixed["S1"])) for j in range(12)]
                                                  + [(2, 0, h_params["T2"][0], h_fixed["C2"], float(h_fixed["S2"]))]):
            fine = np.linspace(SPLINE_LO, SPLINE_HI, 60001)
            exact = phi(fine, theta, centers, step)
            table_f, _ = build_lut(theta, centers, step, entries, int8=False)
            table_q, scale = build_lut(theta, centers, step, entries, int8=True)
            lut_rows.append({"source": "shipped v1 header", "entries": entries, "layer": layer, "unit": unit,
                             "phi_range": float(exact.max() - exact.min()), "table_scale": scale,
                             "max_abs_err_float_table": float(np.max(np.abs(lut_eval(fine, table_f) - exact))),
                             "max_abs_err_int8_table": float(np.max(np.abs(lut_eval(fine, table_q) - exact))),
                             "bound_float_table": lut_error_bound(theta, centers, step, entries),
                             "bound_int8_table": lut_error_bound(theta, centers, step, entries, scale)})
    write_csv(pd.DataFrame(lut_rows), out / "lut_error_header.csv")
    plot(table, out / "fig_auc_vs_bytes.png")

    hashes_after = {str(p.relative_to(ROOT).as_posix()): tree_hashes(p) for p in WRITE_ONCE}
    untouched_after = {p.relative_to(ROOT).as_posix(): sha256(p) for p in UNTOUCHED_FILES}
    import sklearn
    import scipy
    import torch
    import matplotlib

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 1),
        "script": "scripts/ablate_quant_prune_v1.py",
        "scope": "host only; v1 only (14 IR features, as-coded labels, 52,090 windows, split_index folds); "
                 "every variant is an analysis model, never deployed, never written to a header; shipped FP32 v1 untouched (ADR-0004)",
        "threshold": THRESHOLD,
        "explainer": "mean-perturbation (standardized feature -> 0, N + 1 = 15 passes, |delta| ranking, ties by feature index)",
        "split": "recording-grouped (split_index of openox_tinykan_predictions.csv)",
        "recipe_v1_cv": {**V1_RECIPE, "spline_range": list(V1_RECIPE["spline_range"]), "lr": 0.01, "weight_decay": 1e-4,
                         "source": "notebook 12 / lamaakal_models.train_tinykan, all 5 folds in one freshly spawned process"},
        "variants_fixed_before_run": [dict(v) for v in VARIANTS],
        "quantisation": {
            "scheme": "symmetric, zero-point 0, q = clip(round(w / s), -127, 127), s = max|w| / 127; simulated (dequantised to float64)",
            "per_tensor": "one scale for each of W1, b1, T1 (theta), W2, b2, T2",
            "per_channel": "one scale per hidden unit for W1 rows and T1 rows; W2 and T2 are single rows; b1 and b2 per tensor (b2 is one value, so exact)",
            "activations": f"standardized input per feature, scale = {INPUT_CLIP_PERCENTILE}th percentile of |x| on the calibration rows / 127 (training fold in (a), the header's own table in (b)), saturating; u1 and u2 on the clamp range, scale 3/127; hidden phi_j outputs per unit, scale max_u |phi_j| / 127 over [-3, 3]; output logit float",
            "lut": "each of the 13 learned univariate functions (12 hidden phi_j, 1 output phi) tabulated at L uniform points on [-3, 3] (L = 64, 256), int8 per table, linear interpolation, u clamped; theta and the B-spline basis are not used at inference",
            "zero_points_stored": 0,
        },
        "pruning": {"rule": "global magnitude over all 297 learned parameters (W1, b1, T1, W2, b2, T2), smallest floor(rate * 297) set to 0, ties by tensor order then flat index",
                    "counts": {str(r): int(np.floor(r * N_PARAMS)) for r in (0.10, 0.25, 0.50)},
                    "fine_tune": {**FINE_TUNE, "optimizer": "Adam, fresh state", "loss": "pos_weight BCE + 1e-5 smoothness", "mask": "gradients and weights re-masked every step", "data": "training fold only (a); not run for (b)"},
                    "storage": "dense; unstructured zeros save no bytes without a sparse format, whose index overhead is not counted in fp32_nonzero_value_bytes"},
        "bootstrap": {"unit": "recording", "n_boot": N_BOOT, "seed": BOOTSTRAP_SEED, "paired": "every variant scored on the same resamples as FP32",
                      "code": "scripts/lamaakal_metrics.group_bootstrap (Section C)"},
        "checks": {"reproduction_of_frozen_oof": reproduction, "header": header_checks},
        "fine_tune_seconds": {f"fold{k[0]}_rate{k[1]}": round(v["seconds"], 1) for k, v in sorted(tuned.items())},
        "board": "latency, peak RAM and energy are 'not measured (never deployed)' for every variant",
        "deploying_any_variant_would_invalidate": INVALIDATES,
        "write_once_unchanged": hashes_before == hashes_after,
        "write_once_sha256": hashes_after,
        "untouched_files_unchanged": untouched_before == untouched_after,
        "untouched_files_sha256": untouched_after,
        "inputs_sha256": {k: sha256(v) for k, v in P.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                        "matplotlib": matplotlib.__version__, "executable": sys.executable,
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if not summary["write_once_unchanged"] or not summary["untouched_files_unchanged"]:
        raise RuntimeError("A write-once directory or untouched file changed during the run")
    print(table[["key", "oof_auc", "delta_auc", "delta_auc_ci_low", "delta_auc_ci_high", "header_flag_disagreements",
                 "header_top1_agreement", "int8_bytes", "nonzero_params"]].to_string(index=False))
    print(f"done in {summary['duration_seconds']} s -> {out}")
    return 0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    raise SystemExit(main())

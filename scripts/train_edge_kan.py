"""Edge KAN: a spline on every edge, no W, no bias. Host build, evaluation and export.

Pre-registration: docs/edge-kan-preregistration.md (its SHA-256 is recorded in every output).

Each layer computes out[j] = sum_i sum_k B_k(clamp(x[i], -3, 3)) * theta[j, i, k], with theta
stored as (d_out, d_in, n_knots). Layers (d_in, 12, 1). Basis and training recipe are TinyKAN
v2's, unchanged (scripts/train_tinykan_v2.py): 8 centred cubic B-splines on [-3, 3], Adam lr
0.01, weight decay 1e-4, 100 full-batch epochs, seed 42, pos_weight BCE, smoothness 1e-5, grad
clip 10. No hyperparameter search.

Stages (each write-once; outputs under outputs/edge-kan-v1/):
  cv      E0 same-split checks, E1 out-of-fold metrics at 0.50, E2 paired patient bootstrap vs TinyKAN
  final   E3 export: fit ekan16 and ekan14 on all rows -> include/edge_kan_weights_{v2,ir14}.h,
          torch vs parsed-header parity on every row
  parity  E3 host parity: zig C++ probe of edge_kan_inference.h vs the float64 header forward,
          incremental vs full-pass mean perturbation
  aopc    E4 deletion AOPC on the ekan16 export, after reproducing the shipped v2 TinyKAN run
  size    E5 Cortex-M4 object sizes (PlatformIO arm-none-eabi-g++), edge KAN vs shipped v2 engine

Never edits a shipped weight header, kan_inference.h, main.cpp or platformio.ini. Nothing is flashed.

Run from the repository root:  python scripts/train_edge_kan.py {cv,final,parity,aopc,size}
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_tinykan_v2 as tk  # noqa: E402
from evaluate_xai_faithfulness import curve_metrics, perturbation_curves, ranked_orders  # noqa: E402

OUT = ROOT / "outputs" / "edge-kan-v1"
PREREG = ROOT / "docs" / "edge-kan-preregistration.md"
FIRMWARE = ROOT / "firmware" / "Arduino Nano 33"
INCLUDE = FIRMWARE / "include"
ENGINE = INCLUDE / "edge_kan_inference.h"
PROBE_SRC = ROOT / "tests" / "edge_kan_probe.cpp"
AOPC_V2_CSV = ROOT / "outputs" / "analysis-lamaakal-style-v1" / "d_aopc_v2_new_run.csv"
SHIPPED_V2_HEADER = INCLUDE / "kan_model_weights_v2.h"
PROTECTED = (INCLUDE / "kan_model_weights.h", SHIPPED_V2_HEADER, INCLUDE / "kan_inference.h",
             FIRMWARE / "src" / "main.cpp", FIRMWARE / "platformio.ini")

MODELS = {
    "ekan16": {"features": tk.V2_FEATURES, "header": INCLUDE / "edge_kan_weights_v2.h", "define": None,
               "description": "16 v2 inputs (14 IR + ratio_of_ratios + ir_perfusion_index); primary"},
    "ekan14": {"features": tk.IR_FEATURES, "header": INCLUDE / "edge_kan_weights_ir14.h", "define": "EKAN_WEIGHTS_IR14",
               "description": "14 IR inputs; secondary"},
}
# (variant, model, grouping, TinyKAN comparator in outputs/openox-v2/predictions_<comparator>.csv)
CV_VARIANTS = (
    ("ekan16_recording", "ekan16", "recording_id", "b_v2_tinykan16"),
    ("ekan16_patient", "ekan16", "patient_id", "b_v2_tinykan16_patient"),
    ("ekan14_recording", "ekan14", "recording_id", "a_v2_tinykan14"),
)
PAIRED = (("ekan16_recording", "b_v2_tinykan16"), ("ekan14_recording", "a_v2_tinykan14"))
LOGREG = "c_v2_logreg16"
THRESHOLD = tk.DECISION_THRESHOLD
GATE = {"recall": 0.80, "specificity": 0.50}
BOOT_N = 2000
BOOT_SEED = 20261111
PARITY_TOL = 1e-5
INCREMENTAL_TOL = 1e-6
SMOOTHNESS = 1e-5
GRAD_CLIP = 10.0

ZIG = ["uvx", "--from", "ziglang==0.16.0", "python", "-m", "ziglang"]
ZIG_FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-mcpu=baseline", "-target", "x86_64-windows-gnu"]
ARM_BIN = Path.home() / ".platformio" / "packages" / "toolchain-gccarmnoneeabi" / "bin"
ARM_FLAGS = ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16", "-Os",
             "-ffunction-sections", "-fdata-sections", "-std=gnu++17", "-fno-exceptions", "-fno-rtti"]

sha256 = tk.sha256


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
class EdgeKANLayer(nn.Module):
    """out[b, j] = sum_i sum_k B_k(x[b, i]) * theta[j, i, k]. No W, no bias."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.spline = tk.BSpline(tk.N_SPLINE_KNOTS, tk.SPLINE_DEGREE, *tk.SPLINE_RANGE)
        scale = torch.empty(d_out, d_in)
        nn.init.xavier_uniform_(scale)
        grid = torch.linspace(*tk.SPLINE_RANGE, 100)
        identity = torch.linalg.lstsq(self.spline(grid), grid.unsqueeze(1)).solution.squeeze(-1)
        # Starts as a clamped linear map: edge (j, i) is w[j, i] * x.
        self.theta = nn.Parameter(scale.unsqueeze(-1) * identity)

    def forward(self, x):
        return torch.einsum("bik,jik->bj", self.spline(x), self.theta)

    def spline_smoothness_penalty(self):
        t = self.theta
        return (t[..., 2:] - 2 * t[..., 1:-1] + t[..., :-2]).pow(2).sum()


class EdgeKANClassifier(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(EdgeKANLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization(self):
        return sum(layer.spline_smoothness_penalty() for layer in self.layers)


def train_edge_kan(x_scaled: np.ndarray, y: np.ndarray):
    """TinyKAN v2 recipe (tk.train_kan) with the edge KAN in place of KANBinaryClassifier."""
    torch.manual_seed(tk.SEED)
    model = EdgeKANClassifier((x_scaled.shape[1], tk.HIDDEN_DIM, 1))
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=tk.LEARNING_RATE, weight_decay=tk.WEIGHT_DECAY)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    losses = []
    model.train()
    for _ in range(tk.N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t) + SMOOTHNESS * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    return model, losses


def torch_probabilities(model, x_scaled):
    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    return 1.0 / (1.0 + np.exp(-logits)), logits


def parameter_count(d_in: int) -> int:
    return tk.HIDDEN_DIM * d_in * tk.N_SPLINE_KNOTS + tk.HIDDEN_DIM * tk.N_SPLINE_KNOTS


def operation_counts(d_in: int, hidden: int = tk.HIDDEN_DIM, knots: int = tk.N_SPLINE_KNOTS) -> dict:
    """Analytic per-decision counts. A basis evaluation is one scalar into all `knots` values."""
    edge_forward_mac = hidden * d_in * knots + hidden * knots
    edge_forward_basis = d_in + hidden
    tiny_forward_mac = hidden * d_in + hidden * knots + hidden + knots
    tiny_forward_basis = hidden + 1
    return {
        "edge_kan": {
            "forward_multiply_adds": edge_forward_mac,
            "forward_basis_evaluations": edge_forward_basis,
            "xai_fullpass_multiply_adds": (d_in + 1) * edge_forward_mac,
            "xai_fullpass_basis_evaluations": (d_in + 1) * edge_forward_basis,
            "xai_incremental_multiply_adds": edge_forward_mac + d_in * (hidden * knots + hidden * knots),
            "xai_incremental_basis_evaluations": edge_forward_basis + d_in * (2 + hidden),
            "theta_floats": parameter_count(d_in), "theta_bytes_fp32": 4 * parameter_count(d_in),
        },
        "tinykan": {
            "forward_multiply_adds": tiny_forward_mac,
            "forward_basis_evaluations": tiny_forward_basis,
            "xai_fullpass_multiply_adds": (d_in + 1) * tiny_forward_mac,
            "xai_fullpass_basis_evaluations": (d_in + 1) * tiny_forward_basis,
            "parameter_floats": hidden * d_in + hidden + hidden * knots + hidden + 1 + knots,
        },
        "note": "One basis evaluation = 8 bump values of one scalar (divide, fabs, cubic each). Scaler, sigmoid and sort not counted.",
    }


# --------------------------------------------------------------------------------------------
# Header export and float64 reference forward
# --------------------------------------------------------------------------------------------
def c_float(value) -> str:
    """Nine significant digits: the exact FP32 value round-trips."""
    return np.format_float_scientific(np.float32(value), precision=8, unique=False, exp_digits=2) + "f"


def c_array(array: np.ndarray, indent: int = 4) -> str:
    if array.ndim == 1:
        return "{" + ", ".join(c_float(v) for v in array) + "}"
    pad = " " * indent
    inner = (",\n" + pad).join(c_array(sub, indent + 4) for sub in array)
    return "{\n" + pad + inner + "\n" + " " * (indent - 4) + "}"


def render_header(name: str, params: dict, provenance: dict) -> str:
    features = MODELS[name]["features"]
    n_in, hidden, knots = len(features), tk.HIDDEN_DIM, tk.N_SPLINE_KNOTS
    names = ", ".join(f'"{f}"' for f in features)
    return f"""// Auto-generated by scripts/train_edge_kan.py final. Do not edit.
// Edge KAN {name}: a spline on every edge, no W, no bias.
//   out[j] = sum_i sum_k B_k(clamp(x[i], EKAN_RANGE_MIN, EKAN_RANGE_MAX)) * THETA[j][i][k]
// Trained on all {provenance['rows']} rows of outputs/openox-v2/window_features.csv
//   (sha256 {provenance['table_sha256']}), header-aligned labels.
// Pre-registration docs/edge-kan-preregistration.md sha256 {provenance['prereg_sha256']}
// {n_in} inputs in model order: {', '.join(features)}
// Weights: {parameter_count(n_in)} floats ({4 * parameter_count(n_in)} B) in EKAN1_THETA + EKAN2_THETA, plus {2 * n_in} scaler floats.
// No decision threshold is stored: 0.50 is TinyKAN's frozen cut and is not calibrated for this model.
// Not included by any firmware translation unit; read by edge_kan_inference.h.

#pragma once

#define EKAN_INPUT_DIM {n_in}
#define EKAN_HIDDEN_DIM {hidden}
#define EKAN_KNOTS {knots}

const char* const EKAN_FEATURE_NAMES[EKAN_INPUT_DIM] = {{{names}}};

const float EKAN_SCALER_MEAN[{n_in}] = {c_array(params['scaler_mean'])};

const float EKAN_SCALER_STD[{n_in}] = {c_array(params['scaler_std'])};

const float EKAN_CENTERS[{knots}] = {c_array(params['centers'])};

const float EKAN_STEP = {c_float(params['step'])};
const float EKAN_RANGE_MIN = {c_float(tk.SPLINE_RANGE[0])};
const float EKAN_RANGE_MAX = {c_float(tk.SPLINE_RANGE[1])};

// Layer 1: [hidden j][input i][basis k]
const float EKAN1_THETA[{hidden}][{n_in}][{knots}] = {c_array(params['theta1'], 4)};

// Layer 2: [output][hidden i][basis k]
const float EKAN2_THETA[1][{hidden}][{knots}] = {c_array(params['theta2'], 4)};
"""


class EdgeHeaderKAN:
    """Float64 forward pass from a generated edge KAN header (FP32 constants, float64 arithmetic,
    raw inputs rounded to FP32 before standardisation), the edge-KAN analogue of
    lamaakal_models.HeaderKAN. Not bitwise MCU arithmetic."""

    def __init__(self, header_path: Path):
        raw = tk.parse_header(header_path)
        self.p = {key: np.asarray(value, dtype=np.float32).astype(np.float64) for key, value in raw.items()}
        self.lo, self.hi = float(self.p["EKAN_RANGE_MIN"]), float(self.p["EKAN_RANGE_MAX"])
        self.n_inputs = self.p["EKAN_SCALER_MEAN"].shape[0]

    def standardize(self, raw):
        raw = np.asarray(raw, dtype=np.float32).astype(np.float64)
        return (raw - self.p["EKAN_SCALER_MEAN"]) / (self.p["EKAN_SCALER_STD"] + 1e-10)

    def basis(self, x):
        distance = np.abs((np.clip(x, self.lo, self.hi)[..., None] - self.p["EKAN_CENTERS"])
                          / (float(self.p["EKAN_STEP"]) + 1e-10))
        inner = 2.0 / 3.0 - distance ** 2 + 0.5 * distance ** 3
        outer = np.maximum(2.0 - distance, 0.0) ** 3 / 6.0
        return np.where(distance < 1.0, inner, outer)

    def hidden(self, scaled):
        return np.tensordot(self.basis(np.asarray(scaled, dtype=np.float64)), self.p["EKAN1_THETA"], axes=([1, 2], [1, 2]))

    def logit(self, scaled):
        return np.tensordot(self.basis(self.hidden(scaled)), self.p["EKAN2_THETA"], axes=([1, 2], [1, 2]))[:, 0]

    def predict(self, scaled):
        scaled = np.asarray(scaled, dtype=np.float64)
        if scaled.ndim != 2 or scaled.shape[1] != self.n_inputs or not np.isfinite(scaled).all():
            raise ValueError(f"Expected finite N x {self.n_inputs} standardized features")
        return 1.0 / (1.0 + np.exp(-self.logit(scaled)))

    def mean_perturbation(self, scaled):
        """N+1 passes: delta_i = P(x) - P(x with standardized input i set to 0, the training mean)."""
        probability = self.predict(scaled)
        attribution = np.empty_like(scaled, dtype=np.float64)
        for feature in range(scaled.shape[1]):
            changed = np.array(scaled, dtype=np.float64, copy=True)
            changed[:, feature] = 0.0
            attribution[:, feature] = probability - self.predict(changed)
        return probability, attribution


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}


def provenance(extra: tuple[Path, ...] = ()) -> dict:
    files = (PREREG, Path(__file__).resolve(), SCRIPTS / "train_tinykan_v2.py", SCRIPTS / "evaluate_xai_faithfulness.py",
             tk.V2_CSV) + extra
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files}


def protected_hashes() -> dict:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in PROTECTED}


def assert_protected(before: dict) -> None:
    if protected_hashes() != before:
        raise RuntimeError("A protected firmware file changed during the run")


def refuse_overwrite(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Write-once output exists: {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_table() -> pd.DataFrame:
    frame = pd.read_csv(tk.V2_CSV)
    frame["recording_id"] = frame["recording_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def gate(result: dict) -> bool:
    return bool(result["recall"] >= GATE["recall"] and result["specificity"] >= GATE["specificity"])


def read_comparator(name: str) -> pd.DataFrame:
    frame = pd.read_csv(tk.V2_DIR / f"predictions_{name}.csv").sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(frame["row_index"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError(f"{name}: row_index is not 0..n-1")
    return frame


def reading(low: float, high: float) -> str:
    if low > 0:
        return "edge KAN higher"
    if high < 0:
        return "edge KAN lower"
    return "no difference detected at this resolution"


def paired_bootstrap(y, p_edge, p_tiny, patients, n=BOOT_N, seed=BOOT_SEED) -> dict:
    """Patient-cluster bootstrap of edge minus TinyKAN in AUC, recall@0.50 and specificity@0.50."""
    codes, uniques = pd.factorize(pd.Series(patients))
    n_patients = len(uniques)
    rng = np.random.default_rng(seed)

    def counts(p):
        predicted = p >= THRESHOLD
        out = {}
        for key, mask in (("tp", predicted & (y == 1)), ("fn", ~predicted & (y == 1)),
                          ("tn", ~predicted & (y == 0)), ("fp", predicted & (y == 0))):
            out[key] = np.bincount(codes, weights=mask.astype(float), minlength=n_patients)
        return out

    c_edge, c_tiny = counts(p_edge), counts(p_tiny)

    def rates(c, m):
        tp, fn, tn, fp = (float(m @ c[k]) for k in ("tp", "fn", "tn", "fp"))
        return tp / (tp + fn), tn / (tn + fp)

    ones = np.ones(n_patients)
    point_recall_e, point_spec_e = rates(c_edge, ones)
    point_recall_t, point_spec_t = rates(c_tiny, ones)
    point = {"auc": roc_auc_score(y, p_edge) - roc_auc_score(y, p_tiny),
             "recall": point_recall_e - point_recall_t, "specificity": point_spec_e - point_spec_t}
    draws = {"auc": [], "recall": [], "specificity": []}
    skipped = 0
    for _ in range(n):
        multiplicity = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients).astype(float)
        weight = multiplicity[codes]
        keep = weight > 0
        y_k = y[keep]
        if y_k.min() == y_k.max():
            skipped += 1
            continue
        w_k = weight[keep]
        draws["auc"].append(roc_auc_score(y_k, p_edge[keep], sample_weight=w_k)
                            - roc_auc_score(y_k, p_tiny[keep], sample_weight=w_k))
        r_e, s_e = rates(c_edge, multiplicity)
        r_t, s_t = rates(c_tiny, multiplicity)
        draws["recall"].append(r_e - r_t)
        draws["specificity"].append(s_e - s_t)
    result = {"patients": int(n_patients), "resamples": n, "skipped_single_class": skipped, "seed": seed}
    for key, values in draws.items():
        low, high = np.percentile(values, [2.5, 97.5])
        result[key] = {"difference_edge_minus_tinykan": float(point[key]), "ci95_low": float(low),
                       "ci95_high": float(high), "reading": reading(low, high)}
    return result


# --------------------------------------------------------------------------------------------
# Stage cv: E0, E1, E2
# --------------------------------------------------------------------------------------------
def run_cv() -> None:
    summary_path = OUT / "cv_summary.json"
    refuse_overwrite(summary_path, OUT / "cv_summary.csv",
                     *(OUT / f"predictions_{variant}.csv" for variant, *_ in CV_VARIANTS))
    before = protected_hashes()
    frame = load_table()
    y = frame["label"].to_numpy(dtype=np.int64)

    # E0: same folds and labels as the TinyKAN comparator files, checked before any training.
    e0 = {}
    split_cache = {}
    for variant, model_name, grouping, comparator in CV_VARIANTS:
        features = MODELS[model_name]["features"]
        x = frame.loc[:, features].to_numpy(dtype=np.float32)
        groups = frame[grouping].to_numpy()
        folds = np.zeros(len(frame), dtype=int)
        splits = list(GroupKFold(n_splits=tk.N_SPLITS).split(x, y, groups=groups))
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError(f"{variant}: group leakage between train and test")
            folds[test_idx] = fold
        comp = read_comparator(comparator)
        same_folds = bool(np.array_equal(comp["fold"].to_numpy(), folds))
        same_labels = bool(np.array_equal(comp["y_true"].to_numpy(), y))
        e0[variant] = {"comparator": comparator, "same_folds": same_folds, "same_labels": same_labels}
        if not (same_folds and same_labels):
            raise RuntimeError(f"E0 failed for {variant}: {e0[variant]}")
        split_cache[variant] = (x, groups, splits, folds)
    print("E0 passed:", json.dumps(e0), flush=True)

    rows, details, predictions = [], {}, {}
    for variant, model_name, grouping, comparator in CV_VARIANTS:
        print(f"Training {variant} ...", flush=True)
        x, groups, splits, folds = split_cache[variant]
        p = np.full(len(frame), np.nan)
        fold_info = []
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
            x_test = scaler.transform(x[test_idx]).astype(np.float32)
            model, losses = train_edge_kan(x_train, y[train_idx])
            p[test_idx] = torch_probabilities(model, x_test)[0]
            fold_info.append({"fold": fold, "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                              "loss_first": losses[0], "loss_last": losses[-1],
                              "losses_finite": bool(np.isfinite(losses).all())})
        if not np.isfinite(p).all():
            raise RuntimeError(f"{variant}: non-finite out-of-fold probabilities")
        result = {"variant": variant, "learner": "edge_kan", "model": model_name, "grouping": grouping,
                  "n_features": len(MODELS[model_name]["features"]), "groups": int(pd.Series(groups).nunique()),
                  **tk.metrics(y, p)}
        result["meets_gate_at_0_50"] = gate(result)
        result["fold_aucs"] = [float(roc_auc_score(y[folds == f], p[folds == f])) for f in range(1, tk.N_SPLITS + 1)
                               if len(np.unique(y[folds == f])) == 2]
        rows.append(result)
        details[variant] = {"folds": fold_info, "comparator": comparator}
        predictions[variant] = pd.DataFrame({
            "row_index": np.arange(len(frame)), "recording_id": frame["recording_id"], "patient_id": frame["patient_id"],
            "start_time_seconds": frame["start_time_seconds"], "fold": folds, "y_true": y, "y_prob": p})
        print(f"  AUC={result['auc']:.4f} recall={result['recall']:.4f} spec={result['specificity']:.4f} "
              f"gate={result['meets_gate_at_0_50']}", flush=True)

    # Comparators, recomputed from their write-once prediction files (descriptive rows).
    reference = {v["variant"]: v for v in json.loads((tk.V2_DIR / "summary.json").read_text(encoding="utf-8"))["variants"]}
    comparator_rows, comparator_checks = [], {}
    for name in sorted({c for *_, c in CV_VARIANTS} | {LOGREG}):
        comp = read_comparator(name)
        result = {"variant": name, "learner": reference[name]["learner"], "model": name,
                  "grouping": reference[name]["grouping"], "n_features": reference[name]["n_features"],
                  "groups": reference[name]["groups"], **tk.metrics(comp["y_true"].to_numpy(), comp["y_prob"].to_numpy())}
        result["meets_gate_at_0_50"] = gate(result)
        comparator_rows.append(result)
        comparator_checks[name] = max(abs(result[k] - reference[name][k]) for k in ("auc", "recall", "specificity"))
    if max(comparator_checks.values()) > 1e-12:
        raise RuntimeError(f"Comparator metrics do not reproduce outputs/openox-v2/summary.json: {comparator_checks}")

    # E2: paired patient-cluster bootstrap on recording-grouped out-of-fold predictions.
    e2 = {}
    for variant, comparator in PAIRED:
        print(f"Bootstrap {variant} vs {comparator} ...", flush=True)
        comp = read_comparator(comparator)
        e2[f"{variant}_vs_{comparator}"] = paired_bootstrap(
            y, predictions[variant]["y_prob"].to_numpy(), comp["y_prob"].to_numpy(), frame["patient_id"].to_numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    for variant, table in predictions.items():
        table.to_csv(OUT / f"predictions_{variant}.csv", index=False)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "fold_aucs"} for r in rows + comparator_rows])
    table.to_csv(OUT / "cv_summary.csv", index=False)
    assert_protected(before)
    write_json(summary_path, {
        "created_utc": now(), "stage": "cv", "threshold": THRESHOLD, "gate": GATE,
        "threshold_note": "0.50 is TinyKAN's frozen operating point, applied unchanged; not calibrated for the edge KAN",
        "recipe": {"layers": "edge KAN, theta (d_out, d_in, n_knots), no W, no bias", "hidden": tk.HIDDEN_DIM,
                   "knots": tk.N_SPLINE_KNOTS, "degree": tk.SPLINE_DEGREE, "spline_range": tk.SPLINE_RANGE,
                   "init": "theta[j,i,:] = xavier_uniform w[j,i] * identity coefficients", "lr": tk.LEARNING_RATE,
                   "weight_decay": tk.WEIGHT_DECAY, "epochs": tk.N_EPOCHS, "seed": tk.SEED, "smoothness": SMOOTHNESS,
                   "grad_clip": GRAD_CLIP, "scaler": "StandardScaler per training fold", "cv": f"GroupKFold({tk.N_SPLITS})"},
        "e0_same_split_checks": e0, "variants": rows, "variant_training": details,
        "comparators": comparator_rows, "comparator_reproduction_max_abs_difference": comparator_checks,
        "e2_paired_bootstrap": e2, "provenance": provenance(), "protected_files_unchanged": before,
        "environment": environment()})
    print(table[["variant", "groups", "auc", "recall", "specificity", "precision", "f1", "meets_gate_at_0_50"]]
          .round(4).to_string(index=False))
    print(json.dumps(e2, indent=2))


# --------------------------------------------------------------------------------------------
# Stage final: E3 export, torch vs header
# --------------------------------------------------------------------------------------------
def run_final() -> None:
    report_path = OUT / "final_model_report.json"
    refuse_overwrite(report_path, *(spec["header"] for spec in MODELS.values()))
    before = protected_hashes()
    frame = load_table()
    y = frame["label"].to_numpy(dtype=np.int64)
    meta = {"rows": len(frame), "table_sha256": sha256(tk.V2_CSV), "prereg_sha256": sha256(PREREG)}
    models = {}
    for name, spec in MODELS.items():
        print(f"Fitting {name} on all rows ...", flush=True)
        x_raw = frame.loc[:, spec["features"]].to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw).astype(np.float32)
        model, losses = train_edge_kan(x_scaled, y)
        p_torch, logit_torch = torch_probabilities(model, x_scaled)
        layer1, layer2 = model.layers
        params = {"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_,
                  "theta1": layer1.theta.detach().numpy(), "theta2": layer2.theta.detach().numpy(),
                  "centers": layer1.spline.centers.numpy(), "step": float(layer1.spline.step.item())}
        if not np.array_equal(layer1.spline.centers.numpy(), layer2.spline.centers.numpy()):
            raise RuntimeError("Layer grids differ; the header stores one grid")
        tmp = spec["header"].with_suffix(".h.tmp")
        tmp.write_bytes(render_header(name, params, meta).encode("utf-8"))
        reference = EdgeHeaderKAN(tmp)
        logit_header = reference.logit(reference.standardize(x_raw))
        p_header = 1.0 / (1.0 + np.exp(-logit_header))
        parity = {"rows": int(len(frame)), "tolerance": PARITY_TOL,
                  "max_abs_logit_diff": float(np.max(np.abs(logit_header - logit_torch))),
                  "max_abs_probability_diff": float(np.max(np.abs(p_header - p_torch))),
                  "decision_disagreements_at_0_50": int(np.sum((p_header >= THRESHOLD) != (p_torch >= THRESHOLD))),
                  "comparison": "parsed-header NumPy float64 forward on raw FP32 features vs torch float32 forward on scaler output"}
        print(f"  E3.1 {name}: {parity}", flush=True)
        if parity["max_abs_probability_diff"] > PARITY_TOL:
            tmp.unlink()
            raise RuntimeError(f"E3.1 failed for {name}: {parity}")
        tmp.replace(spec["header"])
        hidden = reference.hidden(reference.standardize(x_raw))
        models[name] = {
            "header": str(spec["header"].relative_to(ROOT)).replace("\\", "/"), "header_sha256": sha256(spec["header"]),
            "description": spec["description"], "features": list(spec["features"]),
            "dims": [len(spec["features"]), tk.HIDDEN_DIM, 1], "parameter_count": parameter_count(len(spec["features"])),
            "torch_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "loss_first": losses[0], "loss_last": losses[-1], "losses_finite": bool(np.isfinite(losses).all()),
            "e3_1_torch_vs_header": parity,
            "training_set_metrics_at_0_50": tk.metrics(y, p_torch),
            "training_set_metrics_note": "resubstitution on the training rows; not a generalisation estimate (see cv_summary.json)",
            "hidden_activation_range": {"min": float(hidden.min()), "max": float(hidden.max()),
                                        "fraction_outside_grid": float(np.mean(np.abs(hidden) > tk.SPLINE_RANGE[1]))},
            "input_fraction_outside_grid": float(np.mean(np.abs(reference.standardize(x_raw)) > tk.SPLINE_RANGE[1])),
            "operation_counts": operation_counts(len(spec["features"])),
        }
        if models[name]["torch_parameter_count"] != models[name]["parameter_count"]:
            raise RuntimeError("Parameter count mismatch")
    assert_protected(before)
    write_json(report_path, {"created_utc": now(), "stage": "final", "models": models,
                             "provenance": provenance((ENGINE,)), "protected_files_unchanged": before,
                             "environment": environment()})
    print(json.dumps({k: {kk: v[kk] for kk in ("header", "parameter_count", "e3_1_torch_vs_header", "hidden_activation_range")}
                      for k, v in models.items()}, indent=2))


# --------------------------------------------------------------------------------------------
# Stage parity: E3.2 - E3.4 (zig C++ probe)
# --------------------------------------------------------------------------------------------
def zig_version() -> str:
    return subprocess.run(ZIG + ["version"], capture_output=True, text=True, check=True, timeout=600).stdout.strip()


def top3_mismatch_gaps(reference_deltas, top_ref, top_c):
    """For rows whose top-3 differs, the largest |(|d_ref[c]| - |d_ref[r]|)| over differing positions."""
    rows = np.nonzero(np.any(top_ref != top_c, axis=1))[0]
    gaps = []
    for r in rows:
        diff = top_ref[r] != top_c[r]
        a = np.abs(reference_deltas[r, top_ref[r][diff]])
        b = np.abs(reference_deltas[r, top_c[r][diff]])
        gaps.append(float(np.max(np.abs(a - b))))
    return rows, np.array(gaps)


def run_parity() -> None:
    parity_dir = OUT / "parity"
    report_path = parity_dir / "report.json"
    refuse_overwrite(report_path)
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx not found; the host probe needs zig 0.16.0 via uv")
    before = protected_hashes()
    parity_dir.mkdir(parents=True, exist_ok=True)
    frame = load_table()
    version = zig_version()
    results = {}
    for name, spec in MODELS.items():
        features = spec["features"]
        d = len(features)
        x_raw = frame.loc[:, features].to_numpy(dtype=np.float32)
        in_path = parity_dir / f"{name}_inputs.bin"
        out_path = parity_dir / f"{name}_probe_output.bin"
        exe = parity_dir / f"edge_kan_probe_{name}.exe"
        with in_path.open("wb") as stream:
            stream.write(np.array([len(x_raw), d], dtype="<u4").tobytes())
            stream.write(np.ascontiguousarray(x_raw, dtype="<f4").tobytes())
        command = ZIG + ["c++", *ZIG_FLAGS, "-I", str(INCLUDE)]
        if spec["define"]:
            command.append(f"-D{spec['define']}")
        command += [str(PROBE_SRC), "-o", str(exe)]
        subprocess.run(command, check=True, timeout=1200, cwd=ROOT)
        run = subprocess.run([str(exe), str(in_path), str(out_path)], capture_output=True, text=True, check=True, timeout=1200)
        cols = 2 * (1 + d + 3)
        out = np.fromfile(out_path, dtype="<f4").astype(np.float64).reshape(len(x_raw), cols)
        if not np.isfinite(out).all():
            raise RuntimeError(f"{name}: non-finite probe output")
        half = 1 + d + 3
        p_inc, d_inc, top_inc = out[:, 0], out[:, 1:1 + d], out[:, 1 + d:half].astype(int)
        p_full, d_full, top_full = out[:, half], out[:, half + 1:half + 1 + d], out[:, half + 1 + d:].astype(int)

        reference = EdgeHeaderKAN(spec["header"])
        p_ref, d_ref = reference.mean_perturbation(reference.standardize(x_raw))
        top_ref = np.argsort(-np.abs(d_ref), axis=1, kind="stable")[:, :3]

        mismatch_rows, gaps = top3_mismatch_gaps(d_ref, top_ref, top_inc)
        full_mismatch_rows, full_gaps = top3_mismatch_gaps(d_ref, top_ref, top_full)
        e32 = {"max_abs_probability_diff": float(np.max(np.abs(p_inc - p_ref))),
               "max_abs_attribution_diff": float(np.max(np.abs(d_inc - d_ref))),
               "fullpass_max_abs_probability_diff": float(np.max(np.abs(p_full - p_ref))),
               "fullpass_max_abs_attribution_diff": float(np.max(np.abs(d_full - d_ref))),
               "decision_disagreements_at_0_50": int(np.sum((p_inc >= THRESHOLD) != (p_ref >= THRESHOLD))),
               "tolerance": PARITY_TOL}
        e32["pass"] = bool(e32["max_abs_probability_diff"] <= PARITY_TOL and e32["max_abs_attribution_diff"] <= PARITY_TOL)
        e33 = {"rows_top3_differs": int(len(mismatch_rows)),
               "max_gap_among_mismatches": float(gaps.max()) if len(gaps) else 0.0,
               "rows_not_explained_by_near_tie": int(np.sum(gaps > PARITY_TOL)) if len(gaps) else 0,
               "fullpass_rows_top3_differs": int(len(full_mismatch_rows)),
               "fullpass_max_gap_among_mismatches": float(full_gaps.max()) if len(full_gaps) else 0.0,
               "near_tie_tolerance": PARITY_TOL}
        e33["pass"] = e33["rows_not_explained_by_near_tie"] == 0
        inc_vs_full_rows = int(np.sum(np.any(top_inc != top_full, axis=1)))
        e34 = {"max_abs_probability_diff": float(np.max(np.abs(p_inc - p_full))),
               "max_abs_attribution_diff": float(np.max(np.abs(d_inc - d_full))),
               "rows_top3_differs": inc_vs_full_rows, "tolerance": INCREMENTAL_TOL}
        e34["pass"] = bool(e34["max_abs_probability_diff"] <= INCREMENTAL_TOL and e34["max_abs_attribution_diff"] <= INCREMENTAL_TOL)
        results[name] = {"rows": int(len(x_raw)), "dims": d, "header_sha256": sha256(spec["header"]),
                         "probe_exe_sha256": sha256(exe), "probe_stderr": run.stderr.strip(),
                         "e3_2_c_vs_header": e32, "e3_3_top3": e33, "e3_4_incremental_vs_fullpass": e34}
        print(f"{name}: E3.2 {e32}\n        E3.3 {e33}\n        E3.4 {e34}", flush=True)
        in_path.unlink()
        out_path.unlink()
    assert_protected(before)
    write_json(report_path, {
        "created_utc": now(), "stage": "parity", "results": results,
        "all_pass": bool(all(r[k]["pass"] for r in results.values()
                             for k in ("e3_2_c_vs_header", "e3_3_top3", "e3_4_incremental_vs_fullpass"))),
        "host_compiler": f"zig {version} c++ (clang) {' '.join(ZIG_FLAGS)}",
        "reference": "EdgeHeaderKAN: FP32 header constants, float64 arithmetic, raw inputs rounded to FP32",
        "scope": "host only; not board parity", "provenance": provenance((ENGINE, PROBE_SRC) + tuple(s["header"] for s in MODELS.values())),
        "protected_files_unchanged": before, "environment": environment()})


# --------------------------------------------------------------------------------------------
# Stage aopc: E4
# --------------------------------------------------------------------------------------------
def aopc_windows(model, scaled, attribution, seed, repeats):
    orders = ranked_orders(attribution, seed=seed, random_repeats=repeats)
    metrics = {}
    for name, order in orders.items():
        deletion, insertion = perturbation_curves(model.predict, scaled, order)
        metrics[name] = curve_metrics(deletion, insertion)
    random_names = [n for n in orders if n.startswith("random_")]
    metrics["random_mean"] = {key: np.mean([metrics[n][key] for n in random_names], axis=0) for key in metrics[random_names[0]]}
    return metrics


def candidate_gain_bootstrap(gain, mask, patients, n=BOOT_N, seed=BOOT_SEED) -> dict:
    codes, uniques = pd.factorize(pd.Series(patients))
    sums = np.bincount(codes, weights=np.where(mask, gain, 0.0), minlength=len(uniques))
    counts = np.bincount(codes, weights=mask.astype(float), minlength=len(uniques))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        m = np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength=len(uniques)).astype(float)
        denominator = m @ counts
        if denominator > 0:
            draws.append((m @ sums) / denominator)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean_gain": float(gain[mask].mean()), "ci95_low": float(low), "ci95_high": float(high),
            "patients_with_candidates": int(np.sum(counts > 0)), "resamples_used": len(draws), "seed": seed}


def run_aopc() -> None:
    import analyze_lamaakal_style as als
    import lamaakal_models as lmods

    aopc_dir = OUT / "aopc"
    refuse_overwrite(aopc_dir / "aopc_deletion.csv", aopc_dir / "summary.json")
    before = protected_hashes()
    frame = load_table()
    y = frame["label"].to_numpy(dtype=np.int64)
    patients = frame["patient_id"].to_numpy()
    raw16 = frame.loc[:, tk.V2_FEATURES].to_numpy(dtype=np.float32)

    # Same-harness check: the shipped v2 TinyKAN must reproduce d_aopc_v2_new_run.csv.
    print("Reproducing the shipped v2 TinyKAN AOPC run ...", flush=True)
    tiny = lmods.HeaderKAN(SHIPPED_V2_HEADER)
    tiny_scaled = tiny.standardize(raw16)
    tiny_p, tiny_a = tiny.mean_perturbation(tiny_scaled)
    tiny_summary = als.aopc_run(tiny, tiny_scaled, tiny_p, tiny_a, y)
    cited = pd.read_csv(AOPC_V2_CSV)
    column_map = {"deletion_aopc_window_mean": "deletion_aopc", "insertion_aopc_window_mean": "insertion_aopc",
                  "deletion_gain_vs_random_mean": "deletion_gain", "fraction_better_than_own_random_mean": "fraction_better"}
    worst = 0.0
    for row in cited.itertuples(index=False):
        block = tiny_summary[row.subset][row.ranking]
        if block["windows"] != row.windows:
            raise RuntimeError(f"Window count differs for {row.subset}/{row.ranking}")
        for column, key in column_map.items():
            cited_value = getattr(row, column)
            if pd.isna(cited_value):
                continue
            worst = max(worst, abs(block[key] - cited_value))
    if worst > 1e-8:
        raise RuntimeError(f"Harness does not reproduce d_aopc_v2_new_run.csv (max abs diff {worst})")
    print(f"  reproduced within {worst:.2e}", flush=True)
    tiny_windows = aopc_windows(tiny, tiny_scaled, tiny_a, als.AOPC_SEED, als.AOPC_RANDOM_REPEATS)

    print("Edge KAN (ekan16) AOPC ...", flush=True)
    edge = EdgeHeaderKAN(MODELS["ekan16"]["header"])
    edge_scaled = edge.standardize(raw16)
    edge_p, edge_a = edge.mean_perturbation(edge_scaled)
    edge_summary = als.aopc_run(edge, edge_scaled, edge_p, edge_a, y)
    edge_windows = aopc_windows(edge, edge_scaled, edge_a, als.AOPC_SEED, als.AOPC_RANDOM_REPEATS)

    # Per-window arrays must give the aopc_run means exactly (they feed the bootstrap).
    for summary, windows, probability in ((tiny_summary, tiny_windows, tiny_p), (edge_summary, edge_windows, edge_p)):
        mask = probability >= THRESHOLD
        for ranking in ("firmware_absolute", "random_mean"):
            if abs(windows[ranking]["deletion_aopc"][mask].mean() - summary["candidate_p_ge_0_50"][ranking]["deletion_aopc"]) > 1e-12:
                raise RuntimeError("Per-window AOPC does not match aopc_run")

    rows = []
    for version, summary in (("tinykan_v2_shipped", tiny_summary), ("edge_kan_ekan16", edge_summary)):
        for subset, block in summary.items():
            for ranking, values in block.items():
                rows.append({"version": version, "subset": subset, "windows": values["windows"], "ranking": ranking,
                             "deletion_aopc_window_mean": values["deletion_aopc"],
                             "insertion_aopc_window_mean": values["insertion_aopc"],
                             "deletion_gain_vs_random_mean": values.get("deletion_gain", float("nan")),
                             "fraction_better_than_own_random_mean": values.get("fraction_better", float("nan"))})
    table = pd.DataFrame(rows)

    readings = {}
    for version, summary, windows, probability in (("tinykan_v2_shipped", tiny_summary, tiny_windows, tiny_p),
                                                   ("edge_kan_ekan16", edge_summary, edge_windows, edge_p)):
        readings[version] = {}
        for subset, mask in (("candidate_p_ge_0_50", probability >= THRESHOLD), ("all", np.ones(len(y), bool))):
            fw = summary[subset]["firmware_absolute"]
            rnd = summary[subset]["random_mean"]
            gain = windows["firmware_absolute"]["deletion_aopc"] - windows["random_mean"]["deletion_aopc"]
            readings[version][subset] = {
                "windows": fw["windows"], "firmware_absolute": fw["deletion_aopc"], "random_mean": rnd["deletion_aopc"],
                "beats_random": bool(fw["deletion_aopc"] > rnd["deletion_aopc"]),
                "fraction_better_than_own_random_mean": fw["fraction_better"],
                "gain_bootstrap": candidate_gain_bootstrap(gain, mask, patients)}

    aopc_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(aopc_dir / "aopc_deletion.csv", index=False, float_format="%.8g")
    assert_protected(before)
    write_json(aopc_dir / "summary.json", {
        "created_utc": now(), "stage": "aopc", "seed": als.AOPC_SEED, "random_repeats": als.AOPC_RANDOM_REPEATS,
        "reference": "training mean (standardized 0), deletion to 0", "rows": int(len(y)), "in_sample": True,
        "harness_reproduction_max_abs_difference": worst, "harness_reproduced": str(AOPC_V2_CSV.relative_to(ROOT)).replace("\\", "/"),
        "readings": readings, "candidates": {"tinykan_v2_shipped": int(np.sum(tiny_p >= THRESHOLD)),
                                             "edge_kan_ekan16": int(np.sum(edge_p >= THRESHOLD))},
        "provenance": provenance((MODELS["ekan16"]["header"], SHIPPED_V2_HEADER, SCRIPTS / "analyze_lamaakal_style.py",
                                  SCRIPTS / "lamaakal_models.py", AOPC_V2_CSV)),
        "protected_files_unchanged": before, "environment": environment()})
    print(json.dumps(readings, indent=2))


# --------------------------------------------------------------------------------------------
# Stage size: E5
# --------------------------------------------------------------------------------------------
SIZE_UNITS = {
    "edge_kan_ekan16": ('#include "edge_kan_inference.h"\n'
                        'extern "C" float size_probe_entry(const float* raw, EkanXaiAttribution* top3) {\n'
                        '    return ekan_predict_with_xai(raw, top3, 0);\n}\n'),
    "tinykan_v2_shipped": ('#define OXY_MODEL_V2\n#include "kan_inference.h"\n'
                           'extern "C" float size_probe_entry(const float* raw, KanXaiAttribution* top3) {\n'
                           '    return kan_predict_with_xai(raw, top3);\n}\n'),
}


def run_size() -> None:
    size_dir = OUT / "size"
    refuse_overwrite(size_dir / "summary.json")
    before = protected_hashes()
    gxx, size_tool = ARM_BIN / "arm-none-eabi-g++.exe", ARM_BIN / "arm-none-eabi-size.exe"
    if not gxx.exists():
        raise RuntimeError(f"Missing {gxx}")
    size_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(gxx), "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    results = {}
    for name, source in SIZE_UNITS.items():
        src, obj = size_dir / f"{name}.cpp", size_dir / f"{name}.o"
        src.write_text(source, encoding="utf-8")
        subprocess.run([str(gxx), *ARM_FLAGS, "-I", str(INCLUDE), "-c", str(src), "-o", str(obj)], check=True, timeout=600)
        listing = subprocess.run([str(size_tool), "-A", str(obj)], capture_output=True, text=True, check=True).stdout
        sections = {}
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith((".text", ".rodata", ".data", ".bss")) and parts[1].isdigit():
                sections[parts[0]] = int(parts[1])
        text = {k: v for k, v in sections.items() if k.startswith(".text") and v}
        rodata = {k: v for k, v in sections.items() if k.startswith(".rodata") and v}
        results[name] = {"object_sha256": sha256(obj), "text_sections": text, "rodata_sections": rodata,
                         "data_bss_bytes": sum(v for k, v in sections.items() if k.startswith((".data", ".bss"))),
                         "text_bytes": sum(text.values()), "rodata_bytes": sum(rodata.values())}
        print(f"{name}: text {results[name]['text_bytes']} B, rodata {results[name]['rodata_bytes']} B", flush=True)
    assert_protected(before)
    write_json(size_dir / "summary.json", {
        "created_utc": now(), "stage": "size", "compiler": version, "flags": ARM_FLAGS, "results": results,
        "operation_counts_16_inputs": operation_counts(16),
        "note": ("Object-file sizes, not a linked image and not board flash. kan_inference.h functions have external "
                 "linkage, so every one is emitted (kan_predict_probability is off the XAI path); the edge engine's "
                 "functions are static inline, so only the XAI path is emitted. expf/fabsf come from libm at link time."),
        "provenance": provenance((ENGINE, MODELS["ekan16"]["header"], SHIPPED_V2_HEADER, INCLUDE / "kan_inference.h")),
        "protected_files_unchanged": before, "environment": environment()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("cv", "final", "parity", "aopc", "size"))
    args = parser.parse_args()
    {"cv": run_cv, "final": run_final, "parity": run_parity, "aopc": run_aopc, "size": run_size}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

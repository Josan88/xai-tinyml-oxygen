"""TinyKAN v2 evaluation and final-model export on the header-aligned OpenOximetry export.

Recipe is scripts/train_final_model.py unchanged except the input count: B-spline node KAN,
8 knots, cubic, range [-3, 3], hidden 12, Adam lr 0.01, weight decay 1e-4, 100 epochs, seed 42,
StandardScaler, pos_weight BCE, smoothness 1e-5, grad clip 10. Threshold 0.50.

Stages:
  cv     recording-grouped GroupKFold(5) out-of-fold evaluation (as scripts/sweep_kan.py) ->
         outputs/openox-v2/{predictions_*.csv, summary.csv, summary.json}
  final  fit 16-feature TinyKAN on all v2 rows -> firmware/Arduino Nano 33/include/kan_model_weights_v2.h
         (new file, included by nothing) + outputs/openox-v2/final_model_report.json

Never writes the shipped kan_model_weights.h. All outputs are write-once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CSV = ROOT / "outputs" / "openox_canonical_window_features.csv"
FROZEN_PREDICTIONS_CSV = ROOT / "outputs" / "openox_tinykan_predictions.csv"
V2_DIR = ROOT / "outputs" / "openox-v2"
V2_CSV = V2_DIR / "window_features.csv"
HEADER_V2 = ROOT / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights_v2.h"
SHIPPED_HEADER = ROOT / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights.h"

SEED = 42
N_SPLITS = 5
N_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
N_SPLINE_KNOTS = 8
SPLINE_DEGREE = 3
SPLINE_RANGE = (-3.0, 3.0)
HIDDEN_DIM = 12
DECISION_THRESHOLD = 0.50
PARITY_ROWS = 1000
PARITY_TOLERANCE = 1e-5

IR_FEATURES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
V2_FEATURES = IR_FEATURES + ("ratio_of_ratios", "ir_perfusion_index")

torch.set_num_threads(1)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Model classes copied from scripts/train_final_model.py.
class BSpline(nn.Module):
    def __init__(self, n_knots=8, degree=3, range_min=-3.0, range_max=3.0):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max
        self.register_buffer("centers", torch.linspace(range_min, range_max, n_knots))
        self.register_buffer("step", torch.tensor((range_max - range_min) / (n_knots - 1)))

    def forward(self, u):
        u_clamped = u.clamp(self.range_min, self.range_max)
        distance = torch.abs((u_clamped.unsqueeze(-1) - self.centers) / (self.step + 1e-10))
        inner = (2.0 / 3.0) - distance.pow(2) + 0.5 * distance.pow(3)
        outer = (2.0 - distance).clamp(min=0).pow(3) / 6.0
        return torch.where(distance < 1.0, inner, outer)


class KANLayer(nn.Module):
    def __init__(self, d_in, d_out, n_knots=8, degree=3, range_min=-3.0, range_max=3.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_knots = n_knots
        self.W = nn.Parameter(torch.empty(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.W)
        self.spline = BSpline(n_knots, degree, range_min, range_max)
        self.theta = nn.Parameter(torch.empty(d_out, n_knots))
        self._init_spline_to_identity()

    def _init_spline_to_identity(self):
        grid = torch.linspace(self.spline.range_min, self.spline.range_max, 100)
        basis_grid = self.spline(grid)
        target = grid.unsqueeze(1)
        theta_init = torch.linalg.lstsq(basis_grid, target).solution.squeeze(-1)
        self.theta.data = theta_init.unsqueeze(0).expand(self.d_out, -1).clone()

    def forward(self, x):
        u = F.linear(x, self.W, self.b)
        batch = u.shape[0]
        basis = self.spline(u.reshape(-1)).reshape(batch, self.d_out, self.n_knots)
        return (basis * self.theta.unsqueeze(0)).sum(dim=-1)

    def spline_smoothness_penalty(self):
        diff2 = self.theta[:, 2:] - 2 * self.theta[:, 1:-1] + self.theta[:, :-2]
        return diff2.pow(2).sum()


class KANBinaryClassifier(nn.Module):
    def __init__(self, dims=(14, 12, 1), n_knots=8, degree=3, spline_range=(-3.0, 3.0)):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLayer(dims[i], dims[i + 1], n_knots, degree, spline_range[0], spline_range[1])
            for i in range(len(dims) - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization(self):
        return sum(layer.spline_smoothness_penalty() for layer in self.layers)


def train_kan(x_scaled: np.ndarray, y: np.ndarray, epochs: int = N_EPOCHS) -> KANBinaryClassifier:
    torch.manual_seed(SEED)
    model = KANBinaryClassifier((x_scaled.shape[1], HIDDEN_DIM, 1), N_SPLINE_KNOTS, SPLINE_DEGREE, SPLINE_RANGE)
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t) + 1e-5 * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    model.eval()
    return model


def kan_probabilities(model: KANBinaryClassifier, x_scaled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x_scaled, dtype=torch.float32))).numpy().ravel()


def logreg_probabilities(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # Notebook 11 comparator configuration.
    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEED)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def out_of_fold(frame: pd.DataFrame, features: tuple[str, ...], group_column: str, learner: str, epochs: int = N_EPOCHS):
    x = frame.loc[:, features].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    groups = frame[group_column].to_numpy()
    probabilities = np.full(len(frame), np.nan)
    folds = np.zeros(len(frame), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=N_SPLITS).split(x, y, groups=groups), start=1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Group leakage between train and test")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = scaler.transform(x[test_idx]).astype(np.float32)
        if learner == "tinykan":
            probabilities[test_idx] = kan_probabilities(train_kan(x_train, y[train_idx], epochs), x_test)
        else:
            probabilities[test_idx] = logreg_probabilities(x_train, y[train_idx], x_test)
        folds[test_idx] = fold
    return probabilities, folds


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    predicted = (p >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    result = {
        "rows": int(len(y)), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": float(recall), "specificity": float(tn / (tn + fp)), "precision": float(precision),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "accuracy": float((tp + tn) / len(y)),
    }
    for label, name in ((1, "pos"), (0, "neg")):
        q25, q50, q75 = np.quantile(p[y == label], [0.25, 0.5, 0.75])
        result.update({f"p_{name}_median": float(q50), f"p_{name}_q25": float(q25), f"p_{name}_q75": float(q75)})
    return result


def run_cv() -> None:
    if (V2_DIR / "summary.json").exists():
        raise FileExistsError(f"Write-once output exists: {V2_DIR / 'summary.json'}")
    v2 = pd.read_csv(V2_CSV)
    frozen = pd.read_csv(FROZEN_CSV)
    for frame in (v2, frozen):
        frame["recording_id"] = frame["recording_id"].astype(str)

    variants = [
        ("a_prime_frozen_tinykan14", "frozen rows/labels, TinyKAN 14 IR features (regression check)", frozen, IR_FEATURES, "recording_id", "tinykan", N_EPOCHS),
        ("a_prime_frozen_tinykan14_e80", "as a_prime with notebook 12's 80 epochs (exact-reproduction check)", frozen, IR_FEATURES, "recording_id", "tinykan", 80),
        ("a_v2_tinykan14", "v2 rows/labels, TinyKAN 14 IR features (isolates label fix)", v2, IR_FEATURES, "recording_id", "tinykan", N_EPOCHS),
        ("b_v2_tinykan16", "v2 rows/labels, TinyKAN 14 + ratio_of_ratios + ir_perfusion_index", v2, V2_FEATURES, "recording_id", "tinykan", N_EPOCHS),
        ("c_v2_logreg16", "v2 rows/labels, LogReg on (b) inputs, same folds", v2, V2_FEATURES, "recording_id", "logreg", None),
        ("b_v2_tinykan16_patient", "as (b) with GroupKFold(5) on patient_id", v2, V2_FEATURES, "patient_id", "tinykan", N_EPOCHS),
    ]
    rows = []
    extras = {}
    for name, description, frame, features, group_column, learner, epochs in variants:
        print(f"Running {name} ...", flush=True)
        p, folds = out_of_fold(frame, features, group_column, learner, epochs or N_EPOCHS)
        y = frame["label"].to_numpy(dtype=np.int64)
        predictions = pd.DataFrame({"row_index": np.arange(len(frame)), "recording_id": frame["recording_id"],
                                    "start_time_seconds": frame["start_time_seconds"], "fold": folds,
                                    "y_true": y, "y_prob": p})
        if "patient_id" in frame:
            predictions.insert(2, "patient_id", frame["patient_id"])
        predictions.to_csv(V2_DIR / f"predictions_{name}.csv", index=False)
        result = {"variant": name, "description": description, "learner": learner, "n_features": len(features),
                  "grouping": group_column, "epochs": epochs, "recordings": int(frame["recording_id"].nunique()),
                  "groups": int(frame[group_column].nunique()), **metrics(y, p),
                  "fold_aucs": [float(roc_auc_score(y[folds == f], p[folds == f])) for f in range(1, N_SPLITS + 1)
                                if len(np.unique(y[folds == f])) == 2]}
        rows.append(result)
        print(f"  AUC={result['auc']:.4f} recall={result['recall']:.4f} spec={result['specificity']:.4f}", flush=True)
        if name == "a_prime_frozen_tinykan14_e80":
            published = pd.read_csv(FROZEN_PREDICTIONS_CSV).sort_values("row_index")
            extras["notebook12_reproduction"] = {
                "max_abs_probability_difference": float(np.max(np.abs(published["y_prob"].to_numpy() - p))),
                "same_folds": bool((published["split_index"].to_numpy() == folds).all()),
                "published_auc": float(roc_auc_score(published["y_true"], published["y_prob"])),
            }

    summary = pd.DataFrame(rows)
    summary.drop(columns=["fold_aucs"]).to_csv(V2_DIR / "summary.csv", index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": DECISION_THRESHOLD,
        "recipe": {"hidden": HIDDEN_DIM, "knots": N_SPLINE_KNOTS, "degree": SPLINE_DEGREE, "spline_range": SPLINE_RANGE,
                   "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs": N_EPOCHS, "seed": SEED,
                   "scaler": "StandardScaler per training fold", "cv": f"GroupKFold({N_SPLITS})"},
        "label_rule": "median ScalcO2 < 90; windows with median in [89, 91] dropped at export",
        "variants": rows,
        **extras,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (V2_CSV, FROZEN_CSV, FROZEN_PREDICTIONS_CSV)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "torch": torch.__version__, "sklearn": sklearn.__version__},
    }
    (V2_DIR / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = ["variant", "rows", "groups", "prevalence", "auc", "recall", "specificity", "precision", "f1",
               "p_pos_median", "p_pos_q25", "p_pos_q75", "p_neg_median", "p_neg_q25", "p_neg_q75"]
    print(summary[columns].round(4).to_string(index=False))
    print(json.dumps(extras, indent=2))


def arr_to_c(arr, precision=6):
    if len(arr.shape) == 1:
        return ", ".join([f"{val:.{precision}f}f" for val in arr])
    elif len(arr.shape) == 2:
        rows = []
        for row in arr:
            rows.append("{" + ", ".join([f"{val:.{precision}f}f" for val in row]) + "}")
        return ",\n    ".join(rows)
    return ""


def render_header(params: dict, feature_names: tuple[str, ...], precision: int) -> str:
    """Same layout and symbol names as scripts/train_final_model.py; only the input count differs."""
    n_in = len(feature_names)
    c = lambda key: arr_to_c(params[key], precision)  # noqa: E731
    order = ", ".join(feature_names)
    return f"""// Auto-generated by train_tinykan_v2.py
// Do not edit this file manually.
// v2: header-aligned OpenOximetry labels, {n_in} inputs in this order:
// {order}
// Not included by any firmware translation unit.

#pragma once

#define KAN_INPUT_DIM {n_in}
#define KAN_HIDDEN_DIM {HIDDEN_DIM}
#define KAN_KNOTS {N_SPLINE_KNOTS}

// StandardScaler Parameters
const float SCALER_MEAN[{n_in}] = {{
    {c('scaler_mean')}
}};

const float SCALER_STD[{n_in}] = {{
    {c('scaler_std')}
}};

// KAN Layer 1: {n_in} -> {HIDDEN_DIM}
const float KAN1_W[{HIDDEN_DIM}][{n_in}] = {{
    {c('w1')}
}};

const float KAN1_B[{HIDDEN_DIM}] = {{
    {c('b1')}
}};

const float KAN1_THETA[{HIDDEN_DIM}][{N_SPLINE_KNOTS}] = {{
    {c('theta1')}
}};

const float KAN1_CENTERS[{N_SPLINE_KNOTS}] = {{
    {c('centers1')}
}};

const float KAN1_STEP = {params['step1']:.6f}f;

// KAN Layer 2: {HIDDEN_DIM} -> 1
const float KAN2_W[1][{HIDDEN_DIM}] = {{
    {c('w2')}
}};

const float KAN2_B[1] = {{
    {c('b2')}
}};

const float KAN2_THETA[1][{N_SPLINE_KNOTS}] = {{
    {c('theta2')}
}};

const float KAN2_CENTERS[{N_SPLINE_KNOTS}] = {{
    {c('centers2')}
}};

const float KAN2_STEP = {params['step2']:.6f}f;

const float KAN_DECISION_THRESHOLD = {DECISION_THRESHOLD:.2f}f;
"""


def parse_header(path: Path) -> dict[str, np.ndarray | float]:
    text = Path(path).read_text(encoding="utf-8")
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    values: dict[str, np.ndarray | float] = {}
    for name, dims, body in re.findall(r"const float (\w+)((?:\[\d+\])+)\s*=\s*\{(.*?)\};", text, flags=re.S):
        shape = tuple(int(d) for d in re.findall(r"\[(\d+)\]", dims))
        flat = np.array([float(v) for v in re.findall(number + r"(?=f)", body)], dtype=np.float32)
        values[name] = flat.reshape(shape)
    for name, value in re.findall(r"const float (\w+)\s*=\s*(" + number + r")f;", text):
        values[name] = np.float32(value)
    return values


def numpy_bspline(u, centers, step, range_min, range_max):
    u_clamped = np.clip(u, range_min, range_max)
    dist = np.abs((u_clamped[:, np.newaxis] - centers) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0, None)**3 / 6.0
    return np.where(dist < 1.0, inner, outer)


def header_logits(weights: dict, x_raw: np.ndarray) -> np.ndarray:
    """Pure NumPy two-layer KAN forward pass from parsed header arrays (float64 arithmetic)."""
    lo, hi = SPLINE_RANGE
    x_scaled = (x_raw.astype(np.float64) - weights["SCALER_MEAN"]) / weights["SCALER_STD"]
    u1 = x_scaled @ weights["KAN1_W"].astype(np.float64).T + weights["KAN1_B"]
    out1 = np.stack([np.sum(numpy_bspline(u1[:, j], weights["KAN1_CENTERS"], float(weights["KAN1_STEP"]), lo, hi) * weights["KAN1_THETA"][j], axis=-1)
                     for j in range(u1.shape[1])], axis=1)
    u2 = out1 @ weights["KAN2_W"].astype(np.float64).T + weights["KAN2_B"]
    return np.sum(numpy_bspline(u2[:, 0], weights["KAN2_CENTERS"], float(weights["KAN2_STEP"]), lo, hi) * weights["KAN2_THETA"][0], axis=-1)


def header_probabilities(weights: dict, x_raw: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-header_logits(weights, x_raw)))


def run_final() -> None:
    if HEADER_V2.exists():
        raise FileExistsError(f"Write-once output exists: {HEADER_V2}")
    if (V2_DIR / "final_model_report.json").exists():
        raise FileExistsError(f"Write-once output exists: {V2_DIR / 'final_model_report.json'}")
    shipped_hash = sha256(SHIPPED_HEADER)
    v2 = pd.read_csv(V2_CSV)
    x_raw = v2.loc[:, V2_FEATURES].to_numpy(dtype=np.float32)
    y = v2["label"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)
    model = train_kan(x_scaled, y)
    with torch.no_grad():
        torch_logits = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    torch_probabilities = 1.0 / (1.0 + np.exp(-torch_logits))

    state = model.state_dict()
    params = {
        "scaler_mean": scaler.mean_, "scaler_std": scaler.scale_,
        "w1": state["layers.0.W"].numpy(), "b1": state["layers.0.b"].numpy(), "theta1": state["layers.0.theta"].numpy(),
        "centers1": model.layers[0].spline.centers.numpy(), "step1": float(model.layers[0].spline.step.item()),
        "w2": state["layers.1.W"].numpy(), "b2": state["layers.1.b"].numpy(), "theta2": state["layers.1.theta"].numpy(),
        "centers2": model.layers[1].spline.centers.numpy(), "step2": float(model.layers[1].spline.step.item()),
    }

    rng = np.random.default_rng(SEED)
    parity_idx = np.sort(rng.choice(len(v2), size=PARITY_ROWS, replace=False))
    attempts = []
    for precision in (6, 9):
        text = render_header(params, V2_FEATURES, precision)
        tmp = HEADER_V2.with_suffix(".h.tmp")
        tmp.write_text(text, encoding="utf-8")
        weights = parse_header(tmp)
        logits = header_logits(weights, x_raw[parity_idx])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        attempt = {"decimal_places": precision,
                   "max_abs_logit_diff": float(np.max(np.abs(logits - torch_logits[parity_idx]))),
                   "max_abs_probability_diff": float(np.max(np.abs(probabilities - torch_probabilities[parity_idx]))),
                   "decision_disagreements": int(np.sum((probabilities >= DECISION_THRESHOLD) != (torch_probabilities[parity_idx] >= DECISION_THRESHOLD)))}
        attempts.append(attempt)
        print(f"Header parity at {precision} decimals: {attempt}", flush=True)
        if attempt["max_abs_probability_diff"] < PARITY_TOLERANCE and attempt["max_abs_logit_diff"] < PARITY_TOLERANCE:
            tmp.replace(HEADER_V2)
            break
        tmp.unlink()
    else:
        raise RuntimeError(f"Header parity failed at every precision: {attempts}")
    if sha256(SHIPPED_HEADER) != shipped_hash:
        raise RuntimeError("Shipped kan_model_weights.h changed during the run")

    train_metrics = metrics(y, torch_probabilities)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "header": str(HEADER_V2.relative_to(ROOT)), "header_sha256": sha256(HEADER_V2),
        "shipped_header_sha256_unchanged": shipped_hash,
        "features": list(V2_FEATURES), "kan_dims": [len(V2_FEATURES), HIDDEN_DIM, 1],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_rows": int(len(v2)), "training_recordings": int(v2["recording_id"].nunique()),
        "training_prevalence": float(y.mean()),
        "training_set_metrics_at_0_50": train_metrics,
        "training_set_metrics_note": "resubstitution on the training rows; not a generalisation estimate (see summary.json)",
        "header_parity": {"rows": PARITY_ROWS, "row_selection": f"numpy default_rng({SEED}) sample without replacement",
                          "tolerance": PARITY_TOLERANCE, "attempts": attempts,
                          "comparison": "parsed-header NumPy float64 forward pass vs torch float32 forward pass on raw v2 features"},
        "scaler_mean": params["scaler_mean"].tolist(), "scaler_std": params["scaler_std"].tolist(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "sklearn": sklearn.__version__},
    }
    (V2_DIR / "final_model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("header", "parameter_count", "training_rows", "header_parity")}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("cv", "final", "all"))
    args = parser.parse_args()
    if args.stage in ("cv", "all"):
        run_cv()
    if args.stage in ("final", "all"):
        run_final()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

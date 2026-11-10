"""Lamaakal-style results analysis of the frozen TinyKAN v1 and v2 (host only).

Maps the results sections of Lamaakal et al. (2026), MAKE 8(3):55, onto this project:
  A  baselines on the same folds (their Table 3)
  B  operating-point sensitivity, descriptive only (Table 5 / Fig. 2); 0.50 stays the cut
  C  per-fold spread, recording bootstrap, paired TinyKAN-LogReg tests, seed spread (Table 6)
  D  mean-perturbation case studies, global ranking, spline inspection, AOPC (Figs. 3-4)
  E  ablation with TinyML columns (Table 7)
  F  deployment table from files on disk (Table 10)
  G  error analysis (Section 6.1)

Rules kept here: no board I/O, no firmware or header writes, no existing outputs/ directory
is written. Retrained TinyKANs are analysis models only. Every threshold metric uses 0.50
recomputed from y_prob; the stored y_pred in openox_tinykan_predictions.csv is the 0.30 cut
and is ignored. The explainer is mean-perturbation (training-mean replacement, N+1 passes,
ranked by |delta|). The top_attribution_feature column of openox_tinykan_predictions.csv is
gradient x input from notebook 12 and is not used.

Run (from the repository root, with the interpreter that has torch 2.12 / sklearn 1.9):
  C:\\Python314\\python.exe scripts/analyze_lamaakal_style.py
  C:\\Python314\\python.exe scripts/analyze_lamaakal_style.py --output build/rerun-check
"""
from __future__ import annotations

import os

# Single-threaded numerical libraries in this process and in every worker, so that
# floating-point reductions do not depend on thread scheduling.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "analysis-lamaakal-style-v1"

IR_FEATURES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
RED_IR_FEATURES = ("ratio_of_ratios", "ir_perfusion_index")
V2_FEATURES = IR_FEATURES + RED_IR_FEATURES
# Operator glosses, copied from dashboard/dashboard_core.js FEATURE_GLOSS.
GLOSS = {
    "mean": "filtered-window mean", "std": "amplitude variability", "minimum": "waveform extremes",
    "maximum": "waveform extremes", "value_range": "waveform extremes", "rms": "overall amplitude",
    "skewness": "waveform asymmetry", "kurtosis": "waveform peakedness", "energy": "signal energy",
    "derivative_std": "rate-of-change variability", "dominant_frequency_hz": "dominant frequency",
    "bandpower_0_5_2_hz": "low-frequency power", "bandpower_2_5_hz": "higher-frequency power",
    "bandpower_ratio_0_5_2_to_2_5": "low vs high frequency power",
    "ratio_of_ratios": "red/infrared pulse ratio", "ir_perfusion_index": "infrared pulse strength (perfusion index)",
}

P = {
    "v1_features": ROOT / "outputs/openox_canonical_window_features.csv",
    "v1_tinykan_oof": ROOT / "outputs/openox_tinykan_predictions.csv",
    "v1_logreg_oof": ROOT / "outputs/openox_logreg_local_explanations.csv",
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v2_tinykan16_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "v2_tinykan16_patient_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
    "v2_logreg16_oof": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
    "v2_tinykan14_oof": ROOT / "outputs/openox-v2/predictions_a_v2_tinykan14.csv",
    "v1labels_tinykan14_e100_oof": ROOT / "outputs/openox-v2/predictions_a_prime_frozen_tinykan14.csv",
    "v2_summary": ROOT / "outputs/openox-v2/summary.json",
    "v2_final_report": ROOT / "outputs/openox-v2/final_model_report.json",
    "v1_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h",
    "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "operating_point_sweep": ROOT / "outputs/openox_operating_point_sweep.csv",
    "tinykan_sweep": ROOT / "outputs/openox_tinykan_sweep_results.csv",
    "aopc_summary": ROOT / "outputs/xai-faithfulness/openox-aopc-v1/summary.json",
    "aopc_inputs": ROOT / "outputs/xai-faithfulness/openox-aopc-v1/inputs_attributions.npz",
    "timing_v1": ROOT / "outputs/parity/timing-board-v1/report.json",
    "timing_v2_r2": ROOT / "outputs/timing-v2-board-20261022-r2/report.json",
    "timing_v2_20261015": ROOT / "outputs/timing-v2-board-20261015/report.json",
    "parity_v1_locked": ROOT / "outputs/parity/qualification/locked-board-replay-v1/report.json",
    "parity_v2_r2": ROOT / "outputs/parity/v2-board-20261022-r2/report.json",
    "v1_live_receipt": ROOT / "outputs/firmware-v1-rollback/rollback-receipt.json",
    "v2_live_r2_build_log": ROOT / "outputs/firmware-v2-f1a-r2/build.log",
    "v2_timing_r2_build_log": ROOT / "outputs/timing-v2-prepared-r2/build.log",
    "on_device_timing_md": ROOT / "docs/on-device-timing.md",
    "spline_plot_script": ROOT / "scripts/plot_tinykan_architecture_splines.py",
    "train_tinykan_v2_script": ROOT / "scripts/train_tinykan_v2.py",
    "evaluate_xai_script": ROOT / "scripts/evaluate_xai_faithfulness.py",
}
WRITE_ONCE = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/ptt-ppg/subject-1-v1",
    ROOT / "outputs/ptt_ppg_motion_spike.csv",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
)

THRESHOLD = 0.50
SWEEP_THRESHOLDS = tuple(np.round(np.arange(0.30, 0.7001, 0.05), 2))
SEEDS = (42, 1, 2, 3, 4, 5)
BOOTSTRAP_SEED = 20261024
N_BOOT = 2000
AOPC_SEED = 20260911          # same seed and repeats as the write-once v1 AOPC run
AOPC_RANDOM_REPEATS = 5
REPRODUCTION_TOLERANCE = 1e-5
ZERO_DELTA = 1e-12            # |delta| at or below this is an exact zero (output spline clamped on both passes)
V1_RECIPE = {"hidden": 12, "knots": 8, "spline_range": (-3.0, 3.0), "epochs": 80}
V2_RECIPE = {"hidden": 12, "knots": 8, "spline_range": (-3.0, 3.0), "epochs": 100}

# Figure style: IEEE widths, reference palette slots 1-3, identity never by colour alone.
COL1, COL2 = 3.5, 7.16
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e2e1dc"


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


def write_gzip_csv(frame, path):
    text = frame.to_csv(index=False, float_format="%.7g", lineterminator="\n").encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(text)


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "legend.frameon": False, "hatch.linewidth": 0.6, "hatch.color": INK2,
        "savefig.dpi": 300, "figure.dpi": 100,
    })
    return plt


def save_figure(fig, path):
    fig.savefig(path, dpi=300, facecolor="white", metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


# ----------------------------------------------------------------------------- data

def load_inputs():
    v1 = pd.read_csv(P["v1_features"])
    v1["recording_id"] = v1["recording_id"].astype(str)
    v1_kan = pd.read_csv(P["v1_tinykan_oof"]).sort_values("row_index").reset_index(drop=True)
    v1_lr = pd.read_csv(P["v1_logreg_oof"]).sort_values("source_index").reset_index(drop=True)
    checks = {}
    if not np.array_equal(v1_kan["row_index"], np.arange(len(v1))):
        raise RuntimeError("v1 TinyKAN predictions do not cover every canonical row once")
    if not (v1_kan["recording_id"].astype(str).to_numpy() == v1["recording_id"].to_numpy()).all():
        raise RuntimeError("v1 TinyKAN recording IDs differ from the canonical table")
    if not (v1_kan["y_true"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("v1 TinyKAN labels differ from the canonical table")
    if not np.array_equal(v1_lr["source_index"], np.arange(len(v1))):
        raise RuntimeError("v1 LogReg predictions do not cover every canonical row once")
    if not (v1_lr["true_label"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("v1 LogReg labels differ")
    if not (v1_lr["fold"].to_numpy() == v1_kan["split_index"].to_numpy()).all():
        raise RuntimeError("v1 LogReg folds differ from TinyKAN split_index")
    if v1_kan.groupby("recording_id")["split_index"].nunique().max() != 1:
        raise RuntimeError("A v1 recording spans several folds")
    checks["v1_folds"] = "recovered from split_index in openox_tinykan_predictions.csv; identical to fold in openox_logreg_local_explanations.csv; each recording in one fold"
    checks["v1_stored_y_pred_is_0_30_cut"] = bool(((v1_kan["y_prob"] >= 0.30).astype(int) == v1_kan["y_pred"]).all())

    v2 = pd.read_csv(P["v2_features"])
    v2["recording_id"] = v2["recording_id"].astype(str)
    v2_oof = {}
    for key in ("v2_tinykan16_oof", "v2_tinykan16_patient_oof", "v2_logreg16_oof", "v2_tinykan14_oof"):
        frame = pd.read_csv(P[key]).sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(frame["row_index"], np.arange(len(v2))):
            raise RuntimeError(f"{key} does not cover every v2 row once")
        if not (frame["recording_id"].astype(str).to_numpy() == v2["recording_id"].to_numpy()).all():
            raise RuntimeError(f"{key} recording IDs differ")
        if not (frame["y_true"].to_numpy() == v2["label"].to_numpy()).all():
            raise RuntimeError(f"{key} labels differ")
        v2_oof[key] = frame
    folds_rec = v2_oof["v2_tinykan16_oof"]["fold"].to_numpy()
    for key in ("v2_logreg16_oof", "v2_tinykan14_oof"):
        if not np.array_equal(v2_oof[key]["fold"].to_numpy(), folds_rec):
            raise RuntimeError(f"{key} folds differ from predictions_b_v2_tinykan16.csv")
    if v2_oof["v2_tinykan16_patient_oof"].groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("A v2 patient spans several patient folds")
    if not np.isfinite(v2.loc[:, V2_FEATURES].to_numpy()).all():
        raise RuntimeError("Non-finite v2 features")
    checks["v2_folds"] = "fold column of predictions_b_v2_tinykan16.csv (recording-grouped GroupKFold(5), sklearn 1.9); identical in predictions_c and predictions_a"
    aprime = pd.read_csv(P["v1labels_tinykan14_e100_oof"]).sort_values("row_index").reset_index(drop=True)
    if not (aprime["y_true"].to_numpy() == v1["label"].to_numpy()).all():
        raise RuntimeError("a_prime labels differ from the canonical table")
    return v1, v1_kan, v1_lr, v2, v2_oof, aprime, checks


def build_jobs(v1, v1_kan, v2, v2_oof):
    data = {
        "v1": (v1.loc[:, IR_FEATURES].to_numpy(np.float32), v1["label"].to_numpy(np.int64),
               v1_kan["split_index"].to_numpy(int)),
        "v2_16": (v2.loc[:, V2_FEATURES].to_numpy(np.float32), v2["label"].to_numpy(np.int64),
                  v2_oof["v2_tinykan16_oof"]["fold"].to_numpy(int)),
        "v2_14": (v2.loc[:, IR_FEATURES].to_numpy(np.float32), v2["label"].to_numpy(np.int64),
                  v2_oof["v2_tinykan16_oof"]["fold"].to_numpy(int)),
        "v2_2": (v2.loc[:, RED_IR_FEATURES].to_numpy(np.float32), v2["label"].to_numpy(np.int64),
                 v2_oof["v2_tinykan16_oof"]["fold"].to_numpy(int)),
        "v2_16_patient": (v2.loc[:, V2_FEATURES].to_numpy(np.float32), v2["label"].to_numpy(np.int64),
                          v2_oof["v2_tinykan16_patient_oof"]["fold"].to_numpy(int)),
    }
    jobs = []
    for version, dataset in (("v1", "v1"), ("v2", "v2_16")):
        for learner in ("svm_rbf", "random_forest", "mlp_64_32", "mlp_16", "hist_gb", "knn", "logreg"):
            jobs.append({"name": f"{version}_{learner}", "dataset": dataset, "learner": learner})
    jobs.append({"name": "v2patient_logreg", "dataset": "v2_16_patient", "learner": "logreg"})
    for seed in SEEDS:
        jobs.append({"name": f"v1_tinykan_seed{seed}", "dataset": "v1", "learner": "tinykan", **V1_RECIPE, "seed": seed})
        jobs.append({"name": f"v2_tinykan_seed{seed}", "dataset": "v2_16", "learner": "tinykan", **V2_RECIPE, "seed": seed})
    ablations = [
        ("v1_tinykan_h4", "v1", {**V1_RECIPE, "hidden": 4}),
        ("v1_tinykan_h32", "v1", {**V1_RECIPE, "hidden": 32}),
        ("v1_tinykan_range2", "v1", {**V1_RECIPE, "spline_range": (-2.0, 2.0)}),
        ("v1_tinykan_k6", "v1", {**V1_RECIPE, "knots": 6}),
        ("v1_tinykan_k12", "v1", {**V1_RECIPE, "knots": 12}),
        ("v2_tinykan_h4", "v2_16", {**V2_RECIPE, "hidden": 4}),
        ("v2_tinykan_h32", "v2_16", {**V2_RECIPE, "hidden": 32}),
        ("v2_tinykan_redir2", "v2_2", dict(V2_RECIPE)),
    ]
    for name, dataset, recipe in ablations:
        jobs.append({"name": name, "dataset": dataset, "learner": "tinykan", **recipe, "seed": 42})
    jobs.append({"name": "v2_logreg_ir14", "dataset": "v2_14", "learner": "logreg"})
    jobs.append({"name": "v2_logreg_redir2", "dataset": "v2_2", "learner": "logreg"})
    # Longest first so the pool packs well.
    weight = {"tinykan": 3, "svm_rbf": 2, "random_forest": 2, "mlp_64_32": 2}
    jobs.sort(key=lambda spec: -weight.get(spec["learner"], 1))
    return jobs, data


def run_jobs(jobs, data, workers):
    """Every job runs in its own freshly spawned process (max_tasks_per_child=1).

    TinyKAN training is sensitive to float-level differences that depend on process state
    (import order, earlier allocations): the same fold trained in two differently prepared
    processes can end in different models. A fresh process per job makes each result
    independent of which worker ran which job before, so reruns are reproducible.
    """
    payloads = [(spec, *data[spec["dataset"]]) for spec in jobs]
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, workers), max_tasks_per_child=1) as pool:
        for result in pool.map(models.run_cv_job_from_arrays, payloads):
            results[result["name"]] = result
            print(f"  {result['name']}: {result['seconds']:.1f} s", flush=True)
    for spec in jobs:
        results[spec["name"]]["spec"] = spec
    return results


# ----------------------------------------------------------------------------- section A

SIZE_UNITS = {
    "tinykan": "trainable parameters", "logreg": "trainable parameters", "mlp_16": "trainable parameters",
    "mlp_64_32": "trainable parameters", "knn": "stored training values (n_train x d), fold mean",
    "svm_rbf": "support-vector values + dual coefficients + intercept + Platt A,B, fold mean",
    "random_forest": "tree nodes, fold mean", "hist_gb": "tree nodes, fold mean",
}
MODEL_LABEL = {
    "tinykan": "TinyKAN", "logreg": "LogReg (L2)", "knn": "kNN (k=25)", "svm_rbf": "SVM-RBF",
    "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting", "mlp_16": "MLP (16)",
    "mlp_64_32": "MLP (64, 32)",
}


def section_a(ctx):
    rows = []

    def add(version, grouping, learner, source, p, y, groups, size, settings):
        metrics = lm.binary_metrics(y, p, THRESHOLD)
        rows.append({
            "version": version, "grouping": grouping, "model": MODEL_LABEL[learner], "learner": learner,
            "source": source, "size_count": size, "size_unit": SIZE_UNITS[learner],
            "n_windows": metrics["n"], "n_groups": int(pd.Series(groups).nunique()),
            "prevalence": metrics["prevalence"], "auc": metrics["auc"], "pr_auc": metrics["pr_auc"],
            "balanced_accuracy": metrics["balanced_accuracy"], "f1": metrics["f1"], "recall": metrics["recall"],
            "specificity": metrics["specificity"], "precision": metrics["precision"],
            "candidate_rate": metrics["candidate_rate"], "tn": metrics["tn"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tp": metrics["tp"], "settings": settings,
        })

    res = ctx["results"]
    y1, g1 = ctx["y1"], ctx["v1"]["recording_id"].to_numpy()
    add("v1", "recording", "tinykan", "frozen out-of-fold predictions (notebook 12), not retrained",
        ctx["p"]["v1_tinykan"], y1, g1, 297, "(14, 12, 1), 8 knots, [-3, 3], 80 epochs, seed 42")
    add("v1", "recording", "logreg", "existing out-of-fold predictions (notebook 11, same folds)",
        ctx["p"]["v1_logreg"], y1, g1, 15, models.BASELINE_SETTINGS["logreg"])
    for learner in ("knn", "svm_rbf", "random_forest", "hist_gb", "mlp_16", "mlp_64_32"):
        result = res[f"v1_{learner}"]
        add("v1", "recording", learner, "retrained here on the same folds (analysis model)",
            result["probabilities"], y1, g1, float(np.mean(result["fold_sizes"])), models.BASELINE_SETTINGS[learner])
    y2, g2 = ctx["y2"], ctx["v2"]["recording_id"].to_numpy()
    add("v2", "recording", "tinykan", "existing out-of-fold predictions (train_tinykan_v2.py), not retrained",
        ctx["p"]["v2_tinykan"], y2, g2, 321, "(16, 12, 1), 8 knots, [-3, 3], 100 epochs, seed 42")
    add("v2", "recording", "logreg", "existing out-of-fold predictions (train_tinykan_v2.py, same folds)",
        ctx["p"]["v2_logreg"], y2, g2, 17, models.BASELINE_SETTINGS["logreg"])
    for learner in ("knn", "svm_rbf", "random_forest", "hist_gb", "mlp_16", "mlp_64_32"):
        result = res[f"v2_{learner}"]
        add("v2", "recording", learner, "retrained here on the same folds (analysis model)",
            result["probabilities"], y2, g2, float(np.mean(result["fold_sizes"])), models.BASELINE_SETTINGS[learner])
    gp = ctx["v2"]["patient_id"].to_numpy()
    add("v2", "patient", "tinykan", "existing out-of-fold predictions (patient-grouped folds), not retrained",
        ctx["p"]["v2_tinykan_patient"], y2, gp, 321, "(16, 12, 1), 8 knots, [-3, 3], 100 epochs, seed 42")
    add("v2", "patient", "logreg", "retrained here on the patient-grouped folds (analysis model)",
        res["v2patient_logreg"]["probabilities"], y2, gp, 17, models.BASELINE_SETTINGS["logreg"])
    table = pd.DataFrame(rows)
    write_csv(table, ctx["out"] / "a_baselines.csv")

    checks = {
        "v1_logreg_retrained_vs_existing_max_abs_diff": float(np.max(np.abs(res["v1_logreg"]["probabilities"] - ctx["p"]["v1_logreg"]))),
        "v2_logreg_retrained_vs_existing_max_abs_diff": float(np.max(np.abs(res["v2_logreg"]["probabilities"] - ctx["p"]["v2_logreg"]))),
    }
    ctx["checks"]["section_a"] = checks
    head = {}
    for (version, grouping), block in table.groupby(["version", "grouping"], sort=False):
        best = block.sort_values("auc", ascending=False).iloc[0]
        head[f"{version}_{grouping}"] = {
            "tinykan_auc": float(block.loc[block.learner == "tinykan", "auc"].iloc[0]),
            "logreg_auc": float(block.loc[block.learner == "logreg", "auc"].iloc[0]),
            "best_auc_model": best["model"], "best_auc": float(best["auc"]),
            "tinykan_rank_by_auc": int((block["auc"] > block.loc[block.learner == "tinykan", "auc"].iloc[0]).sum() + 1),
            "models": int(len(block)),
        }
    ctx["tables"]["a"] = table
    return head


# ----------------------------------------------------------------------------- section B

def section_b(ctx, plt):
    series = [("v1", "TinyKAN", ctx["y1"], ctx["p"]["v1_tinykan"]), ("v1", "LogReg", ctx["y1"], ctx["p"]["v1_logreg"]),
              ("v2", "TinyKAN", ctx["y2"], ctx["p"]["v2_tinykan"]), ("v2", "LogReg", ctx["y2"], ctx["p"]["v2_logreg"])]
    rows = []
    for version, model, y, p in series:
        for row in lm.threshold_sweep(y, p, SWEEP_THRESHOLDS):
            rows.append({"version": version, "model": model, "reported_operating_point": bool(np.isclose(row["threshold"], THRESHOLD)), **row})
    table = pd.DataFrame(rows)
    write_csv(table, ctx["out"] / "b_operating_point_sweep.csv")

    existing = pd.read_csv(P["operating_point_sweep"])
    existing = existing[existing["mode"] == "window"].copy()
    existing["threshold"] = existing["threshold"].round(2)
    mine = table[(table.version == "v1") & (table.model == "TinyKAN")].set_index("threshold")
    theirs = existing[existing.model == "TinyKAN"].set_index("threshold").loc[list(SWEEP_THRESHOLDS)]
    tiny_count_diff = int(np.max(np.abs(mine[["tn", "fp", "fn", "tp"]].to_numpy() - theirs[["TN", "FP", "FN", "TP"]].to_numpy())))
    mine_lr = table[(table.version == "v1") & (table.model == "LogReg")].set_index("threshold")
    theirs_lr = existing[existing.model == "LogReg"].set_index("threshold").loc[list(SWEEP_THRESHOLDS)]
    lr_count_diff = int(np.max(np.abs(mine_lr[["tn", "fp", "fn", "tp"]].to_numpy() - theirs_lr[["TN", "FP", "FN", "TP"]].to_numpy())))
    ctx["checks"]["section_b"] = {
        "v1_tinykan_counts_max_abs_diff_vs_openox_operating_point_sweep": tiny_count_diff,
        "v1_logreg_counts_max_abs_diff_vs_openox_operating_point_sweep": lr_count_diff,
        "note": "The existing sweep's LogReg rows are the calibrate_operating_point.py reconstruction on sklearn-1.9 GroupKFold folds, "
                "not the notebook-11 predictions on the TinyKAN folds used here.",
    }

    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.3), sharex=True, sharey=True, layout="constrained")
    metrics = (("recall", "Recall"), ("specificity", "Specificity"), ("balanced_accuracy", "Balanced accuracy"),
               ("candidate_rate", "Candidate rate"))
    for r, version in enumerate(("v1", "v2")):
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            for model, color, style, marker in (("TinyKAN", BLUE, "-", "o"), ("LogReg", ORANGE, "--", "s")):
                block = table[(table.version == version) & (table.model == model)]
                ax.plot(block["threshold"], block[metric], color=color, linestyle=style, marker=marker,
                        markersize=3.2, markeredgecolor="white", markeredgewidth=0.4, label=model)
            ax.axvline(THRESHOLD, color=INK2, linewidth=0.7, linestyle=":")
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel("v1: as-coded labels, 14 IR" if version == "v1" else "v2: header-aligned, 16 inputs", fontsize=6.5)
            if r == 1:
                ax.set_xlabel("Threshold on P")
    axes[0, 2].text(THRESHOLD + 0.01, 0.06, "reported 0.50", fontsize=6, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    save_figure(fig, ctx["out"] / "fig_b_operating_point_sensitivity.png")

    def at(version, model, threshold):
        return table[(table.version == version) & (table.model == model) & np.isclose(table.threshold, threshold)].iloc[0]

    head = {}
    for version in ("v1", "v2"):
        for model in ("TinyKAN", "LogReg"):
            row = at(version, model, THRESHOLD)
            head[f"{version}_{model}_at_0_50"] = {k: float(row[k]) for k in ("recall", "specificity", "balanced_accuracy", "f1", "candidate_rate")}
        block = table[(table.version == version) & (table.model == "TinyKAN")]
        head[f"{version}_TinyKAN_balanced_accuracy_range_0_30_0_70"] = [float(block.balanced_accuracy.min()), float(block.balanced_accuracy.max())]
        head[f"{version}_TinyKAN_best_balanced_accuracy_threshold_descriptive_only"] = float(block.loc[block.balanced_accuracy.idxmax(), "threshold"])
    row030 = at("v1", "TinyKAN", 0.30)
    head["v1_TinyKAN_at_0_30_footnote"] = {"recall": float(row030.recall), "specificity": float(row030.specificity)}
    ctx["tables"]["b"] = table
    return head


# ----------------------------------------------------------------------------- section C

def section_c(ctx):
    comparisons = [
        ("v1", "recording", ctx["y1"], ctx["v1"]["recording_id"].to_numpy(), ctx["folds1"],
         {"TinyKAN": ctx["p"]["v1_tinykan"], "LogReg": ctx["p"]["v1_logreg"]}),
        ("v2", "recording", ctx["y2"], ctx["v2"]["recording_id"].to_numpy(), ctx["folds2"],
         {"TinyKAN": ctx["p"]["v2_tinykan"], "LogReg": ctx["p"]["v2_logreg"]}),
        ("v2", "patient", ctx["y2"], ctx["v2"]["patient_id"].to_numpy(), ctx["folds2p"],
         {"TinyKAN": ctx["p"]["v2_tinykan_patient"], "LogReg": ctx["results"]["v2patient_logreg"]["probabilities"]}),
    ]
    from scipy import stats

    fold_rows, summary_rows, ci_rows, paired_rows = [], [], [], []
    head = {}
    for index, (version, grouping, y, groups, folds, probs) in enumerate(comparisons):
        key = f"{version}_{grouping}"
        fold_auc = {}
        for model, p in probs.items():
            per = []
            for fold in sorted(np.unique(folds)):
                mask = folds == fold
                m = lm.binary_metrics(y[mask], p[mask], THRESHOLD)
                per.append(m)
                fold_rows.append({"version": version, "grouping": grouping, "model": model, "fold": int(fold),
                                  "n_windows": m["n"], "n_groups": int(pd.Series(groups[mask]).nunique()),
                                  "prevalence": m["prevalence"], "auc": m["auc"], "pr_auc": m["pr_auc"],
                                  "recall": m["recall"], "specificity": m["specificity"],
                                  "balanced_accuracy": m["balanced_accuracy"]})
            fold_auc[model] = np.array([m["auc"] for m in per])
            row = {"version": version, "grouping": grouping, "model": model, "folds": len(per)}
            for metric in ("auc", "recall", "specificity", "balanced_accuracy"):
                mean, sd = lm.mean_sd([m[metric] for m in per])
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
            summary_rows.append(row)
        t_result = stats.ttest_rel(fold_auc["TinyKAN"], fold_auc["LogReg"])
        diff = fold_auc["TinyKAN"] - fold_auc["LogReg"]
        paired_rows.append({"version": version, "grouping": grouping, "comparison": "TinyKAN - LogReg", "metric": "auc",
                            "method": "paired t-test on fold AUCs (n=5 folds; weak)", "difference": float(diff.mean()),
                            "ci_low": float("nan"), "ci_high": float("nan"), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                            "statistic": float(t_result.statistic), "p_value": float(t_result.pvalue),
                            "fraction_resamples_le_0": float("nan"), "note": "folds won by TinyKAN: %d of %d" % (int((diff > 0).sum()), len(diff))})

        point, draws = lm.group_bootstrap(y, probs, groups, N_BOOT, BOOTSTRAP_SEED + index, THRESHOLD)
        for model in probs:
            for metric in ("auc", "recall", "specificity"):
                low, high = lm.percentile_ci(draws[model][metric])
                ci_rows.append({"version": version, "grouping": grouping, "resampled_unit": grouping, "model": model,
                                "metric": metric, "point": point[model][metric], "ci_low": low, "ci_high": high,
                                "n_boot": N_BOOT, "seed": BOOTSTRAP_SEED + index,
                                "n_units": int(pd.Series(groups).nunique())})
        for metric in ("auc", "recall", "specificity"):
            delta = draws["TinyKAN"][metric] - draws["LogReg"][metric]
            low, high = lm.percentile_ci(delta)
            paired_rows.append({"version": version, "grouping": grouping, "comparison": "TinyKAN - LogReg", "metric": metric,
                                "method": f"paired {grouping} bootstrap ({N_BOOT} resamples, 95% percentile CI)",
                                "difference": point["TinyKAN"][metric] - point["LogReg"][metric],
                                "ci_low": low, "ci_high": high, "se": float(np.nanstd(delta, ddof=1)),
                                "statistic": float("nan"), "p_value": float("nan"),
                                "fraction_resamples_le_0": float(np.mean(delta <= 0)), "note": ""})
        delong = lm.delong_paired(y, probs["TinyKAN"], probs["LogReg"])
        paired_rows.append({"version": version, "grouping": grouping, "comparison": "TinyKAN - LogReg", "metric": "auc",
                            "method": "paired DeLong, windows treated as independent (anti-conservative here)",
                            "difference": delong["difference"], "ci_low": delong["difference"] - 1.959964 * delong["se_difference"],
                            "ci_high": delong["difference"] + 1.959964 * delong["se_difference"], "se": delong["se_difference"],
                            "statistic": delong["z"], "p_value": delong["p_value"], "fraction_resamples_le_0": float("nan"),
                            "note": "ignores within-recording correlation"})
        auc_ci = {r["model"]: [r["ci_low"], r["ci_high"]] for r in ci_rows
                  if r["version"] == version and r["grouping"] == grouping and r["metric"] == "auc"}
        boot_diff = [r for r in paired_rows if r["version"] == version and r["grouping"] == grouping
                     and r["metric"] == "auc" and r["method"].startswith("paired " + grouping)][0]
        head[key] = {
            "tinykan_auc": point["TinyKAN"]["auc"], "tinykan_auc_ci": auc_ci["TinyKAN"],
            "logreg_auc": point["LogReg"]["auc"], "logreg_auc_ci": auc_ci["LogReg"],
            "auc_difference": boot_diff["difference"], "auc_difference_ci": [boot_diff["ci_low"], boot_diff["ci_high"]],
            "fold_t_test_p_n5": float(t_result.pvalue), "delong_p_window_level": delong["p_value"],
        }
    write_csv(pd.DataFrame(fold_rows), ctx["out"] / "c_per_fold.csv")
    write_csv(pd.DataFrame(summary_rows), ctx["out"] / "c_fold_summary.csv")
    write_csv(pd.DataFrame(ci_rows), ctx["out"] / "c_bootstrap_ci.csv")
    write_csv(pd.DataFrame(paired_rows), ctx["out"] / "c_paired_comparison.csv")

    # Seed spread: retrained analysis models, exact CV recipe, same folds.
    seed_rows = []
    reproduction = {}
    for version, y, frozen in (("v1", ctx["y1"], ctx["p"]["v1_tinykan"]), ("v2", ctx["y2"], ctx["p"]["v2_tinykan"])):
        for seed in SEEDS:
            p = ctx["results"][f"{version}_tinykan_seed{seed}"]["probabilities"]
            m = lm.binary_metrics(y, p, THRESHOLD)
            seed_rows.append({"version": version, "seed": seed, "model": "TinyKAN (analysis retrain)",
                              "recipe": "notebook 12 CV recipe, 80 epochs" if version == "v1" else "train_tinykan_v2.py CV recipe, 100 epochs",
                              "auc": m["auc"], "pr_auc": m["pr_auc"], "recall": m["recall"], "specificity": m["specificity"],
                              "balanced_accuracy": m["balanced_accuracy"], "candidate_rate": m["candidate_rate"],
                              "max_abs_diff_vs_reported_oof": float(np.max(np.abs(p - frozen)))})
        p42 = ctx["results"][f"{version}_tinykan_seed42"]["probabilities"]
        folds = ctx["folds1"] if version == "v1" else ctx["folds2"]
        per_fold = {int(f): float(np.max(np.abs(p42[folds == f] - frozen[folds == f]))) for f in sorted(np.unique(folds))}
        reproduction[version] = {
            "reference": "openox_tinykan_predictions.csv (notebook 12)" if version == "v1" else "predictions_b_v2_tinykan16.csv",
            "max_abs_diff": float(np.max(np.abs(p42 - frozen))),
            "max_abs_diff_per_fold": per_fold,
            "folds_within_tolerance": [f for f, d in per_fold.items() if d <= REPRODUCTION_TOLERANCE],
            "tolerance": REPRODUCTION_TOLERANCE,
            "candidate_flag_disagreements_at_0_50": int(np.sum((p42 >= THRESHOLD) != (frozen >= THRESHOLD))),
            "auc_retrain_minus_reference": lm.weighted_auc(y, p42) - lm.weighted_auc(y, frozen),
        }
    seeds = pd.DataFrame(seed_rows)
    write_csv(seeds, ctx["out"] / "c_seed_runs.csv")
    ctx["tables"]["c_seed_runs"] = seeds
    seed_summary = []
    for version, block in seeds.groupby("version"):
        row = {"version": version, "seeds": " ".join(str(s) for s in block.seed), "n_seeds": len(block)}
        for metric in ("auc", "recall", "specificity", "balanced_accuracy"):
            row[f"{metric}_mean"], row[f"{metric}_sd"] = lm.mean_sd(block[metric])
            row[f"{metric}_min"], row[f"{metric}_max"] = float(block[metric].min()), float(block[metric].max())
        seed_summary.append(row)
        head[f"{version}_seed_spread"] = {k: row[k] for k in row if k.endswith(("_mean", "_sd", "_min", "_max"))}
    write_csv(pd.DataFrame(seed_summary), ctx["out"] / "c_seed_summary.csv")
    ctx["checks"]["seed_42_reproduces_frozen_oof"] = reproduction
    head["fold_summary"] = {f"{r['version']}_{r['grouping']}_{r['model']}": {"auc": [r["auc_mean"], r["auc_sd"]],
                            "recall": [r["recall_mean"], r["recall_sd"]], "specificity": [r["specificity_mean"], r["specificity_sd"]]}
                            for r in summary_rows}
    ctx["tables"]["c_ci"] = pd.DataFrame(ci_rows)
    ctx["tables"]["c_paired"] = pd.DataFrame(paired_rows)
    ctx["tables"]["c_seed"] = pd.DataFrame(seed_summary)
    ctx["tables"]["c_fold"] = pd.DataFrame(summary_rows)
    return head


# ----------------------------------------------------------------------------- section D

def explain_table(header, raw):
    model = models.HeaderKAN(header)
    scaled = model.standardize(raw)
    probability, attribution = model.mean_perturbation(scaled)
    return model, scaled, probability, attribution


def section_d(ctx, plt):
    from evaluate_xai_faithfulness import FrozenKAN
    import plot_tinykan_architecture_splines as spline_plot

    head, checks = {}, {}
    explained = {}
    for version, header, frame, features in (("v1", P["v1_header"], ctx["v1"], IR_FEATURES),
                                              ("v2", P["v2_header"], ctx["v2"], V2_FEATURES)):
        raw = frame.loc[:, features].to_numpy(np.float32)
        model, scaled, probability, attribution = explain_table(header, raw)
        explained[version] = (model, scaled, probability, attribution, frame, features)

    # Replica checks against existing references.
    model1, scaled1, prob1, attr1 = explained["v1"][:4]
    frozen = FrozenKAN(P["v1_header"])
    aopc_inputs = np.load(P["aopc_inputs"])
    checks["v1_replica_vs_FrozenKAN_max_abs_p"] = float(np.max(np.abs(frozen.predict(frozen.standardize(ctx["v1"].loc[:, IR_FEATURES].to_numpy(np.float32))) - prob1)))
    checks["v1_replica_vs_aopc_inputs_max_abs_p"] = float(np.max(np.abs(aopc_inputs["probability"] - prob1)))
    checks["v1_replica_vs_aopc_inputs_max_abs_delta"] = float(np.max(np.abs(aopc_inputs["attribution"] - attr1)))
    model2, scaled2, prob2, attr2 = explained["v2"][:4]
    report = json.loads(P["v2_final_report"].read_text(encoding="utf-8"))["training_set_metrics_at_0_50"]
    tn, fp, fn, tp = lm.confusion_at(ctx["y2"], prob2, THRESHOLD)
    checks["v2_replica_resubstitution_confusion"] = [int(tn), int(fp), int(fn), int(tp)]
    checks["v2_final_model_report_confusion"] = [report["tn"], report["fp"], report["fn"], report["tp"]]
    checks["v2_replica_matches_final_model_report"] = checks["v2_replica_resubstitution_confusion"] == checks["v2_final_model_report_confusion"]
    for version in ("v1", "v2"):
        model = explained[version][0]
        weights = spline_plot.parse_weights(Path(P[f"{version}_header"]).read_text(encoding="utf-8"))
        grid = np.linspace(-3.0, 3.0, 401)
        diffs = [np.max(np.abs(spline_plot.phi(grid, weights["theta1"][j], weights["centers1"], weights["step1"]) - model.phi(grid, 1, j)))
                 for j in range(model.hidden)]
        diffs.append(np.max(np.abs(spline_plot.phi(grid, weights["theta2"][0], weights["centers2"], weights["step2"]) - model.phi(grid, 2, 0))))
        checks[f"{version}_phi_vs_plot_tinykan_architecture_splines_max_abs"] = float(max(diffs))

    # (1) Case studies: shipped header on its own training table (in-sample), median-P window per cell.
    case_rows = []
    for version, (model, scaled, probability, attribution, frame, features) in explained.items():
        y = frame["label"].to_numpy(int)
        candidate = probability >= THRESHOLD
        cells = {"TP": (y == 1) & candidate, "FN": (y == 1) & ~candidate, "FP": (y == 0) & candidate, "TN": (y == 0) & ~candidate}
        order = lm.firmware_abs_order(attribution)
        rank = np.empty_like(order)
        rows_idx = np.arange(len(order))[:, None]
        rank[rows_idx, order] = np.arange(1, order.shape[1] + 1)[None, :]
        picked = {}
        for cell, mask in cells.items():
            index = lm.median_member(probability, mask)
            picked[cell] = index
            for f, name in enumerate(features):
                case_rows.append({
                    "version": version, "cell": cell, "cell_size": int(mask.sum()), "row_index": index,
                    "recording_id": frame.loc[index, "recording_id"], "start_time_seconds": float(frame.loc[index, "start_time_seconds"]),
                    "y_true": int(y[index]), "probability": float(probability[index]), "median_spo2": float(frame.loc[index, "median_spo2"]),
                    "feature": name, "gloss": GLOSS[name], "raw_value": float(frame.loc[index, name]),
                    "standardized_value": float(scaled[index, f]), "delta": float(attribution[index, f]),
                    "abs_rank": int(rank[index, f]), "in_top3_payload": bool(rank[index, f] <= 3),
                    "direction": "toward risk" if attribution[index, f] > 0 else ("toward normal" if attribution[index, f] < 0 else "unchanged"),
                })
        head[f"{version}_case_studies"] = {cell: {"row_index": idx, "probability": float(probability[idx]),
                                                   "top3": [features[i] for i in order[idx, :3]],
                                                   "nonzero_deltas": int(np.sum(np.abs(attribution[idx]) > ZERO_DELTA)),
                                                   "n_features": len(features)} for cell, idx in picked.items()}
        head[f"{version}_in_sample_confusion_at_0_50"] = {cell: int(mask.sum()) for cell, mask in cells.items()}
    cases = pd.DataFrame(case_rows)
    write_csv(cases, ctx["out"] / "d_case_studies.csv")
    for version in ("v1", "v2"):
        plot_case_studies(plt, cases[cases.version == version], version, ctx["out"] / f"fig_d_case_studies_{version}.png")

    # (2) Global ranking split by y_true.
    global_rows = []
    for version, (model, scaled, probability, attribution, frame, features) in explained.items():
        y = frame["label"].to_numpy(int)
        order = lm.firmware_abs_order(attribution)
        top1 = order[:, 0]
        top3 = np.zeros_like(attribution, dtype=bool)
        top3[np.arange(len(order))[:, None], order[:, :3]] = True
        for f, name in enumerate(features):
            row = {"version": version, "feature": name, "gloss": GLOSS[name],
                   "mean_abs_delta_all": float(np.mean(np.abs(attribution[:, f])))}
            for label in (1, 0):
                mask = y == label
                row[f"mean_abs_delta_y{label}"] = float(np.mean(np.abs(attribution[mask, f])))
                row[f"mean_signed_delta_y{label}"] = float(np.mean(attribution[mask, f]))
                row[f"top1_share_y{label}"] = float(np.mean(top1[mask] == f))
                row[f"top3_share_y{label}"] = float(np.mean(top3[mask, f]))
            global_rows.append(row)
        block = pd.DataFrame([r for r in global_rows if r["version"] == version])
        head[f"{version}_global"] = {
            "top_mean_abs_delta_y1": block.sort_values("mean_abs_delta_y1", ascending=False).feature.head(3).tolist(),
            "top_mean_abs_delta_y0": block.sort_values("mean_abs_delta_y0", ascending=False).feature.head(3).tolist(),
            "top1_most_common_y1": [block.loc[block.top1_share_y1.idxmax(), "feature"], float(block.top1_share_y1.max())],
            "top1_most_common_y0": [block.loc[block.top1_share_y0.idxmax(), "feature"], float(block.top1_share_y0.max())],
        }
    global_table = pd.DataFrame(global_rows)
    write_csv(global_table, ctx["out"] / "d_global_attribution.csv")
    plot_global(plt, global_table, ctx["out"] / "fig_d_global_attribution.png")

    # Degenerate explanations: zero deltas (the output spline is clamped on both passes), so the
    # firmware fills top-3 slots by feature index; and top-1 drivers that oppose the decision.
    degeneracy_rows = []
    for version, (model, scaled, probability, attribution, frame, features) in explained.items():
        y = frame["label"].to_numpy(int)
        candidate = probability >= THRESHOLD
        nonzero = np.sum(np.abs(attribution) > ZERO_DELTA, axis=1)
        top1_delta = attribution[np.arange(len(attribution)), lm.firmware_abs_order(attribution)[:, 0]]
        clamped = np.abs(model.output_preactivation(scaled)) > 3.0
        opposes = np.where(candidate, top1_delta < -ZERO_DELTA, top1_delta > ZERO_DELTA)
        subsets = {"all": np.ones(len(y), bool), "candidate (P >= 0.50)": candidate, "non-candidate (P < 0.50)": ~candidate,
                   "TP (in-sample)": (y == 1) & candidate, "FN (in-sample)": (y == 1) & ~candidate,
                   "FP (in-sample)": (y == 0) & candidate, "TN (in-sample)": (y == 0) & ~candidate}
        for subset, mask in subsets.items():
            degeneracy_rows.append({
                "version": version, "subset": subset, "n": int(mask.sum()), "n_features": len(features),
                "fraction_all_deltas_zero": float(np.mean(nonzero[mask] == 0)),
                "fraction_top3_has_zero_delta_slot": float(np.mean(nonzero[mask] < 3)),
                "median_nonzero_deltas": float(np.median(nonzero[mask])),
                "fraction_output_preactivation_clamped": float(np.mean(clamped[mask])),
                "fraction_top1_opposes_decision": float(np.mean(opposes[mask])),
            })
        rows_v = [r for r in degeneracy_rows if r["version"] == version]
        head[f"{version}_degeneracy"] = {r["subset"]: {k: r[k] for k in ("n", "fraction_top3_has_zero_delta_slot",
                                                                         "fraction_output_preactivation_clamped",
                                                                         "fraction_top1_opposes_decision")} for r in rows_v}
    write_csv(pd.DataFrame(degeneracy_rows), ctx["out"] / "d_explanation_degeneracy.csv")

    # (3) Spline inspection with class-wise pre-activation distributions.
    spline_rows, shape_rows = [], []
    for version, (model, scaled, probability, attribution, frame, features) in explained.items():
        y = frame["label"].to_numpy(int)
        u_hidden = model.hidden_preactivation(scaled)
        u_out = model.output_preactivation(scaled)
        units = [(f"h{j}", u_hidden[:, j]) for j in range(model.hidden)] + [("output", u_out)]
        for unit, u in units:
            for label in (1, 0):
                values = u[y == label]
                spline_rows.append({"version": version, "unit": unit, "y_true": label, "n": int(len(values)),
                                    "mean_u": float(values.mean()), "median_u": float(np.median(values)),
                                    "q05_u": float(np.quantile(values, 0.05)), "q95_u": float(np.quantile(values, 0.95)),
                                    "fraction_clamped_low": float(np.mean(values < -3.0)),
                                    "fraction_clamped_high": float(np.mean(values > 3.0))})
        all_clamped = np.all(np.abs(u_hidden) > 3.0, axis=1)
        head[f"{version}_windows_with_all_hidden_units_clamped"] = [int(all_clamped.sum()), float(all_clamped.mean())]
        plot_splines(plt, model, units, y, version, ctx["out"] / f"fig_d_splines_{version}.png")
        # Output spline shape: where P peaks inside [-3, 3] versus its value at the clamp edges.
        grid = np.linspace(-3.0, 3.0, 6001)
        phi_out = model.phi(grid, 2, 0)
        p_out = 1.0 / (1.0 + np.exp(-phi_out))
        shape_rows.append({"version": version, "u_at_max_p": float(grid[np.argmax(p_out)]), "max_p": float(p_out.max()),
                           "p_at_u_plus3_clamp": float(p_out[-1]), "u_at_min_p": float(grid[np.argmin(p_out)]),
                           "min_p": float(p_out.min()), "p_at_u_minus3_clamp": float(p_out[0]),
                           "monotone_nondecreasing": bool(np.all(np.diff(phi_out) >= -1e-12)),
                           "fraction_windows_u_above_3": float(np.mean(u_out > 3.0)),
                           "fraction_windows_u_below_minus3": float(np.mean(u_out < -3.0))})
    write_csv(pd.DataFrame(spline_rows), ctx["out"] / "d_spline_preactivation.csv")
    write_csv(pd.DataFrame(shape_rows), ctx["out"] / "d_output_spline_shape.csv")
    head["output_spline_shape"] = shape_rows

    # (4) AOPC: cite the write-once v1 run; recompute v1 in memory only as a replica check; new v2 run.
    summary = json.loads(P["aopc_summary"].read_text(encoding="utf-8"))
    cited = []
    for subset, block in summary["summaries"].items():
        for ranking in ("firmware_absolute", "random_mean", "signed_risk", "reverse_signed"):
            method = block["methods"][ranking]
            gain = summary["paired_gains_vs_random_mean"][subset].get(ranking, {}).get("deletion_aopc", {})
            cited.append({"source": "outputs/xai-faithfulness/openox-aopc-v1/summary.json (write-once, cited)",
                          "version": "v1", "subset": subset, "windows": block["windows"], "ranking": ranking,
                          "deletion_aopc_window_mean": method["deletion_aopc"]["window_mean"],
                          "insertion_aopc_window_mean": method["insertion_aopc"]["window_mean"],
                          "deletion_gain_vs_random_mean": gain.get("window_mean", float("nan")),
                          "fraction_better_than_own_random_mean": gain.get("fraction_better_than_own_random_mean", float("nan"))})
    write_csv(add_predicted_class_column(pd.DataFrame(cited)), ctx["out"] / "d_aopc_v1_cited.csv")
    v1_recomputed = aopc_run(model1, scaled1, prob1, attr1, ctx["y1"])
    checks["v1_aopc_recomputed_in_memory_firmware_absolute_all"] = v1_recomputed["all"]["firmware_absolute"]["deletion_aopc"]
    checks["v1_aopc_cited_firmware_absolute_all"] = summary["summaries"]["all"]["methods"]["firmware_absolute"]["deletion_aopc"]["window_mean"]
    checks["v1_aopc_recomputed_in_memory_random_mean_all"] = v1_recomputed["all"]["random_mean"]["deletion_aopc"]
    checks["v1_aopc_cited_random_mean_all"] = summary["summaries"]["all"]["methods"]["random_mean"]["deletion_aopc"]["window_mean"]
    v2_aopc = aopc_run(model2, scaled2, prob2, attr2, ctx["y2"])
    v2_rows = []
    for subset, block in v2_aopc.items():
        for ranking, values in block.items():
            v2_rows.append({"source": "new run in this directory: shipped v2 header, in-sample, seed 20260911, 5 random repeats",
                            "version": "v2", "subset": subset, "windows": values["windows"], "ranking": ranking,
                            "deletion_aopc_window_mean": values["deletion_aopc"], "insertion_aopc_window_mean": values["insertion_aopc"],
                            "deletion_gain_vs_random_mean": values.get("deletion_gain", float("nan")),
                            "fraction_better_than_own_random_mean": values.get("fraction_better", float("nan"))})
    write_csv(add_predicted_class_column(pd.DataFrame(v2_rows)), ctx["out"] / "d_aopc_v2_new_run.csv")
    for version, table in (("v1", pd.DataFrame(cited)), ("v2", pd.DataFrame(v2_rows))):
        def get(subset, ranking, column="deletion_aopc_window_mean"):
            return float(table[(table.subset == subset) & (table.ranking == ranking)][column].iloc[0])
        head[f"{version}_aopc"] = {
            "firmware_absolute_all": get("all", "firmware_absolute"), "random_mean_all": get("all", "random_mean"),
            "firmware_absolute_candidate": get("candidate_p_ge_0_50", "firmware_absolute"),
            "random_mean_candidate": get("candidate_p_ge_0_50", "random_mean"),
            "firmware_absolute_noncandidate": get("noncandidate_p_lt_0_50", "firmware_absolute"),
            "random_mean_noncandidate": get("noncandidate_p_lt_0_50", "random_mean"),
            "fraction_better_all": get("all", "firmware_absolute", "fraction_better_than_own_random_mean"),
            "fraction_better_noncandidate": get("noncandidate_p_lt_0_50", "firmware_absolute", "fraction_better_than_own_random_mean"),
        }
    ctx["checks"]["section_d"] = checks
    ctx["explained"] = explained
    return head


def add_predicted_class_column(table):
    """Derived column: deletion AOPC with the predicted class as target instead of the fixed P(risk).

    With target 1 - P on non-candidate windows every curve is mirrored, so the AOPC is exactly the
    negative of the P(risk) value; on candidate windows the two targets coincide. Mixed subsets
    (all, label_1, label_0) have no single predicted class and are left empty.
    """
    table = table.copy()
    sign = table["subset"].map({"candidate_p_ge_0_50": 1.0, "noncandidate_p_lt_0_50": -1.0})
    table["deletion_aopc_predicted_class_target"] = sign * table["deletion_aopc_window_mean"]
    table["predicted_class_note"] = np.where(sign.notna(), "derived arithmetic from the P(risk) column, not a new run",
                                             "undefined for a mixed subset")
    return table


def aopc_run(model, scaled, probability, attribution, y):
    from evaluate_xai_faithfulness import curve_metrics, perturbation_curves, ranked_orders

    orders = ranked_orders(attribution, seed=AOPC_SEED, random_repeats=AOPC_RANDOM_REPEATS)
    metrics = {}
    for name, order in orders.items():
        deletion, insertion = perturbation_curves(model.predict, scaled, order)
        metrics[name] = curve_metrics(deletion, insertion)
    random_names = [n for n in orders if n.startswith("random_")]
    metrics["random_mean"] = {key: np.mean([metrics[n][key] for n in random_names], axis=0) for key in metrics[random_names[0]]}
    subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": probability >= THRESHOLD,
               "noncandidate_p_lt_0_50": probability < THRESHOLD, "label_1": y == 1, "label_0": y == 0}
    out = {}
    for subset, mask in subsets.items():
        out[subset] = {}
        for ranking in ("firmware_absolute", "random_mean", "signed_risk", "reverse_signed"):
            values = {"windows": int(mask.sum()),
                      "deletion_aopc": float(np.mean(metrics[ranking]["deletion_aopc"][mask])),
                      "insertion_aopc": float(np.mean(metrics[ranking]["insertion_aopc"][mask]))}
            if ranking != "random_mean":
                gain = metrics[ranking]["deletion_aopc"][mask] - metrics["random_mean"]["deletion_aopc"][mask]
                values["deletion_gain"] = float(np.mean(gain))
                values["fraction_better"] = float(np.mean(gain > 1e-12))
            out[subset][ranking] = values
    return out


def short_id(recording_id):
    return str(recording_id)[:8]


def feature_label(name):
    return f"{name}: {GLOSS[name]}"


def plot_case_studies(plt, cases, version, path):
    features = list(dict.fromkeys(cases.feature))
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 1.6 + 0.32 * len(features)), sharey=True, layout="constrained")
    labels = [feature_label(name) for name in features]
    for ax, cell in zip(axes.ravel(), ("TP", "FN", "FP", "TN")):
        block = cases[cases.cell == cell].set_index("feature").loc[features]
        y_pos = np.arange(len(features))[::-1]
        deltas = block["delta"].to_numpy()
        for yy, d, top in zip(y_pos, deltas, block["abs_rank"].to_numpy()):
            positive = d > 0
            ax.barh(yy, d, height=0.62, color=BLUE if positive else ORANGE, edgecolor="white", linewidth=0.4,
                    hatch=None if positive else "////")
        limit = max(np.max(np.abs(deltas)) * 1.35, 1e-4)
        ax.set_xlim(-limit, limit)
        for yy, d, top in zip(y_pos, deltas, block["abs_rank"].to_numpy()):
            if top <= 3:
                text = f"#{top}" if abs(d) > ZERO_DELTA else f"#{top} (Δ = 0, index tie-break)"
                ax.text(d + (0.03 * limit if d >= 0 else -0.03 * limit), yy, text, va="center",
                        ha="left" if d >= 0 else "right", fontsize=5.8, color=INK)
        ax.axvline(0, color=INK2, linewidth=0.6)
        first = block.iloc[0]
        nonzero = int(np.sum(np.abs(deltas) > ZERO_DELTA))
        ax.set_title(f"{cell}: P = {first.probability:.3f}, y = {int(first.y_true)}, {nonzero} of {len(features)} Δ non-zero\n"
                     f"ref. ScalcO2 {first.median_spo2:.1f}, rec. {short_id(first.recording_id)}, t = {first.start_time_seconds:.0f} s",
                     fontsize=6.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"$\Delta_i$ = P(x) $-$ P(x with feature $i$ at training mean)", fontsize=6.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, edgecolor="white", label="raised P vs training-mean replacement (toward risk)"),
               Patch(facecolor=ORANGE, edgecolor="white", hatch="////", label="lowered P vs training-mean replacement (toward normal)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=6.3)
    save_figure(fig, path)


def plot_global(plt, table, path):
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 6.2), layout="constrained",
                             gridspec_kw={"height_ratios": [14, 16]})
    for r, version in enumerate(("v1", "v2")):
        block = table[table.version == version].sort_values("mean_abs_delta_all", ascending=True)
        y_pos = np.arange(len(block))
        labels = [feature_label(f) for f in block.feature]
        for c, (prefix, xlabel) in enumerate((("mean_abs_delta", r"Mean $|\Delta_i|$"),
                                              ("top1_share", r"Share of windows with feature top-1 by $|\Delta|$"))):
            ax = axes[r, c]
            ax.barh(y_pos + 0.2, block[f"{prefix}_y1"], height=0.38, color=BLUE, edgecolor="white", linewidth=0.4,
                    label="y = 1 (Low-Oxygen Risk label)")
            ax.barh(y_pos - 0.2, block[f"{prefix}_y0"], height=0.38, color=ORANGE, edgecolor="white", linewidth=0.4,
                    hatch="////", label="y = 0")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels if c == 0 else [], fontsize=6)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(xlabel)
            model = "TinyKAN v1 (14, 12, 1)" if version == "v1" else "TinyKAN v2 (16, 12, 1)"
            ax.set_title(f"{model}: {'mean |Δ| by class' if c == 0 else 'top-1 share by class'}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    save_figure(fig, path)


def plot_splines(plt, model, units, y, version, path):
    from matplotlib.gridspec import GridSpec

    cols = 5
    rows = int(np.ceil(len(units) / cols))
    fig = plt.figure(figsize=(COL2, 1.6 * rows + 0.6), layout="constrained")
    grid = GridSpec(rows * 2, cols, figure=fig, height_ratios=[2.0, 1.0] * rows)
    u_grid = np.linspace(-3.0, 3.0, 401)
    bins = np.linspace(-3.0, 3.0, 31)
    for k, (unit, u) in enumerate(units):
        r, c = divmod(k, cols)
        ax_phi = fig.add_subplot(grid[2 * r, c])
        ax_den = fig.add_subplot(grid[2 * r + 1, c], sharex=ax_phi)
        layer, index = (2, 0) if unit == "output" else (1, int(unit[1:]))
        ax_phi.plot(u_grid, model.phi(u_grid, layer, index), color=INK, linewidth=1.1)
        low = np.mean(u < -3.0)
        high = np.mean(u > 3.0)
        name = unit if unit != "output" else "output"
        ax_phi.set_title(f"{name}  (u<-3: {pct(low, 0)}, u>3: {pct(high, 0)})", fontsize=6.0, pad=2)
        ax_phi.tick_params(labelbottom=False)
        clipped = np.clip(u, -3.0, 3.0)
        for label, color, style in ((1, BLUE, "-"), (0, ORANGE, "--")):
            values = clipped[y == label]
            ax_den.hist(values, bins=bins, density=True, histtype="step", color=color, linestyle=style, linewidth=0.9,
                        label="y = 1 (Low-Oxygen Risk label)" if label else "y = 0")
        ax_den.set_yticks([])
        ax_den.set_xlim(-3.0, 3.0)
        ax_den.grid(visible=False)
        if k + cols >= len(units):
            ax_den.set_xlabel("pre-activation u", fontsize=6)
        if c == 0:
            ax_phi.set_ylabel(r"$\varphi(u)$", fontsize=6.5)
            ax_den.set_ylabel("density", fontsize=6)
    handles, labels = ax_den.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=6.3)
    fig.suptitle(f"TinyKAN {version}: learned splines (top) and class-wise pre-activation u on the training windows "
                 "(bottom; u clamped to [-3, 3] as on the device).\nArchitecture inspection, not operator XAI; "
                 "hidden units are not detectors.", fontsize=6.3)
    save_figure(fig, path)


# ----------------------------------------------------------------------------- section E

def section_e(ctx):
    res = ctx["results"]
    rows = []

    def add(variant, learner, source, labels, inputs, n_inputs, folds_text, y, p, hidden=None, spline_range=None,
            knots=None, epochs=None, params=None, board_ms=None, board_note="not measured (never ran on the board)"):
        m = lm.binary_metrics(y, p, THRESHOLD)
        if learner == "TinyKAN":
            params = models.tinykan_param_count(n_inputs, hidden, knots)
            header_bytes = 4 * models.tinykan_header_floats(n_inputs, hidden, knots)
        else:
            header_bytes = float("nan")
        rows.append({"variant": variant, "learner": learner, "source": source, "labels": labels, "inputs": inputs,
                     "n_inputs": n_inputs, "hidden": hidden, "spline_range": None if spline_range is None else f"[{spline_range[0]:g}, {spline_range[1]:g}]",
                     "knots": knots, "epochs": epochs, "folds": folds_text, "rows": m["n"], "auc": m["auc"], "pr_auc": m["pr_auc"],
                     "recall_050": m["recall"], "specificity_050": m["specificity"], "balanced_accuracy_050": m["balanced_accuracy"],
                     "params": params, "fp32_weight_bytes": 4 * params, "header_constant_bytes": header_bytes,
                     "board_decision_block_median_ms": board_ms, "board_note": board_note})

    y1, y2 = ctx["y1"], ctx["y2"]
    v1f = "v1 recording-grouped split_index (5)"
    v2f = "v2 recording-grouped fold (5)"
    analysis = "analysis retrain (not shipped)"
    add("Shipped v1 recipe", "TinyKAN", "frozen OOF (notebook 12)", "as-coded", "14 IR", 14, v1f, y1, ctx["p"]["v1_tinykan"],
        12, (-3, 3), 8, 80, board_ms=ctx["board"]["v1"]["median_ms"],
        board_note="whole 250-sample decision block incl. 15 passes, synthetic USB, cycle-derived 64 MHz; shipped header is the all-rows 100-epoch fit")
    for name, label, hidden, rng, knots in (("v1_tinykan_h4", "v1 hidden 4", 4, (-3, 3), 8), ("v1_tinykan_h32", "v1 hidden 32", 32, (-3, 3), 8),
                                            ("v1_tinykan_range2", "v1 spline range [-2, 2]", 12, (-2, 2), 8),
                                            ("v1_tinykan_k6", "v1 6 knots", 12, (-3, 3), 6), ("v1_tinykan_k12", "v1 12 knots", 12, (-3, 3), 12)):
        add(label, "TinyKAN", analysis, "as-coded", "14 IR", 14, v1f, y1, res[name]["probabilities"], hidden, rng, knots, 80)
    add("As-coded labels, 14 IR, 100 epochs", "TinyKAN", "existing OOF predictions_a_prime_frozen_tinykan14 (sklearn-1.9 GroupKFold)",
        "as-coded", "14 IR", 14, "v1 rows, GroupKFold(5) sklearn 1.9 (not split_index)", y1, ctx["p"]["v1labels_tinykan14_e100"], 12, (-3, 3), 8, 100)
    add("Header-aligned labels, 14 IR", "TinyKAN", "existing OOF predictions_a_v2_tinykan14", "header-aligned", "14 IR", 14, v2f, y2,
        ctx["p"]["v2_tinykan14"], 12, (-3, 3), 8, 100)
    add("Shipped v2 recipe (14 IR + 2 red/IR)", "TinyKAN", "existing OOF predictions_b_v2_tinykan16", "header-aligned", "16", 16, v2f, y2,
        ctx["p"]["v2_tinykan"], 12, (-3, 3), 8, 100, board_ms=ctx["board"]["v2"]["median_ms"],
        board_note="r2 image, whole 250-sample block incl. 17 passes, synthetic USB, cycle-derived 64 MHz")
    for name, label, hidden in (("v2_tinykan_h4", "v2 hidden 4", 4), ("v2_tinykan_h32", "v2 hidden 32", 32)):
        add(label, "TinyKAN", analysis, "header-aligned", "16", 16, v2f, y2, res[name]["probabilities"], hidden, (-3, 3), 8, 100)
    add("2 red/IR inputs only", "TinyKAN", analysis, "header-aligned", "ratio_of_ratios + ir_perfusion_index", 2, v2f, y2,
        res["v2_tinykan_redir2"]["probabilities"], 12, (-3, 3), 8, 100)
    add("LogReg, as-coded, 14 IR", "LogReg", "existing OOF (notebook 11)", "as-coded", "14 IR", 14, v1f, y1, ctx["p"]["v1_logreg"], params=15,
        board_note="not measured (LogReg was never deployed)")
    add("LogReg, header-aligned, 14 IR", "LogReg", analysis, "header-aligned", "14 IR", 14, v2f, y2, res["v2_logreg_ir14"]["probabilities"], params=15,
        board_note="not measured (LogReg was never deployed)")
    add("LogReg, header-aligned, 16", "LogReg", "existing OOF predictions_c_v2_logreg16", "header-aligned", "16", 16, v2f, y2, ctx["p"]["v2_logreg"], params=17,
        board_note="not measured (LogReg was never deployed)")
    add("LogReg, header-aligned, 2 red/IR only", "LogReg", analysis, "header-aligned", "ratio_of_ratios + ir_perfusion_index", 2, v2f, y2,
        res["v2_logreg_redir2"]["probabilities"], params=3, board_note="not measured (LogReg was never deployed)")
    table = pd.DataFrame(rows)
    write_csv(table, ctx["out"] / "e_ablation.csv")

    sweep = pd.read_csv(P["tinykan_sweep"])
    ref_rows = []
    recompute = {(12, -3.0): ctx["p"]["v1_tinykan"], (4, -3.0): res["v1_tinykan_h4"]["probabilities"],
                 (32, -3.0): res["v1_tinykan_h32"]["probabilities"], (12, -2.0): res["v1_tinykan_range2"]["probabilities"]}
    for _, r in sweep.sort_values(["hidden_dim", "spline_range_min"]).iterrows():
        key = (int(r.hidden_dim), float(r.spline_range_min))
        row = {"hidden_dim": int(r.hidden_dim), "spline_range": f"[{r.spline_range_min:g}, {r.spline_range_max:g}]",
               "param_count": int(r.param_count), "sweep_best_f1_threshold": float(r.best_threshold),
               "sweep_recall_at_best_f1": float(r.recall), "sweep_f1_at_best_f1": float(r.f1),
               "auc_recomputed": float("nan"), "recall_050_recomputed": float("nan"), "specificity_050_recomputed": float("nan"),
               "recomputed_from": ""}
        if key in recompute:
            m = lm.binary_metrics(y1, recompute[key], THRESHOLD)
            row.update({"auc_recomputed": m["auc"], "recall_050_recomputed": m["recall"], "specificity_050_recomputed": m["specificity"],
                        "recomputed_from": "frozen OOF" if key == (12, -3.0) else "analysis retrain in this directory"})
            mb = lm.binary_metrics(y1, recompute[key], float(r.best_threshold))
            row["sweep_recall_reproduced"] = mb["recall"]
        ref_rows.append(row)
    write_csv(pd.DataFrame(ref_rows), ctx["out"] / "e_sweep_bestf1_reference.csv")

    def auc_of(variant):
        return float(table.loc[table.variant == variant, "auc"].iloc[0])
    head = {
        "v1_auc_hidden_4_12_32": [auc_of("v1 hidden 4"), auc_of("Shipped v1 recipe"), auc_of("v1 hidden 32")],
        "v2_auc_hidden_4_12_32": [auc_of("v2 hidden 4"), auc_of("Shipped v2 recipe (14 IR + 2 red/IR)"), auc_of("v2 hidden 32")],
        "v1_auc_range_2_vs_3": [auc_of("v1 spline range [-2, 2]"), auc_of("Shipped v1 recipe")],
        "v1_auc_knots_6_8_12": [auc_of("v1 6 knots"), auc_of("Shipped v1 recipe"), auc_of("v1 12 knots")],
        "label_alignment_auc_as_coded_vs_header_aligned_14IR": [auc_of("As-coded labels, 14 IR, 100 epochs"), auc_of("Header-aligned labels, 14 IR")],
        "inputs_auc_14IR_16_2redIR": [auc_of("Header-aligned labels, 14 IR"), auc_of("Shipped v2 recipe (14 IR + 2 red/IR)"), auc_of("2 red/IR inputs only")],
        "logreg_auc_v1_14_v2_14_v2_16_v2_2": [auc_of("LogReg, as-coded, 14 IR"), auc_of("LogReg, header-aligned, 14 IR"),
                                              auc_of("LogReg, header-aligned, 16"), auc_of("LogReg, header-aligned, 2 red/IR only")],
    }
    ctx["tables"]["e"] = table
    return head


# ----------------------------------------------------------------------------- section F

def read_build_sizes(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ram = re.findall(r"RAM:.*?used (\d+) bytes from (\d+) bytes", text)
    flash = re.findall(r"Flash:.*?used (\d+) bytes from (\d+) bytes", text)
    if not ram or not flash:
        raise RuntimeError(f"No size lines in {log_path}")
    return int(flash[-1][0]), int(ram[-1][0])


def board_numbers():
    timing_v1 = json.loads(P["timing_v1"].read_text(encoding="utf-8"))
    timing_v2 = json.loads(P["timing_v2_r2"].read_text(encoding="utf-8"))
    timing_v2_old = json.loads(P["timing_v2_20261015"].read_text(encoding="utf-8"))
    parity_v1 = json.loads(P["parity_v1_locked"].read_text(encoding="utf-8"))
    parity_v2 = json.loads(P["parity_v2_r2"].read_text(encoding="utf-8"))
    receipt_v1 = json.loads(P["v1_live_receipt"].read_text(encoding="utf-8"))["images"]["nano33ble"]
    md = P["on_device_timing_md"].read_text(encoding="utf-8")
    flash_t1 = int(re.search(r"\*\*([\d,]+) bytes flash\*\*", md).group(1).replace(",", ""))
    ram_t1 = int(re.search(r"\*\*([\d,]+) bytes static RAM\*\*", md).group(1).replace(",", ""))
    flash_l2, ram_l2 = read_build_sizes(P["v2_live_r2_build_log"])
    flash_t2, ram_t2 = read_build_sizes(P["v2_timing_r2_build_log"])

    def lat(report):
        s = report["latency_summary"]
        return {"median_ms": s["subsequent_decisions"]["median_us"] / 1000, "p95_ms": s["subsequent_decisions"]["p95_us"] / 1000,
                "max_ms": s["subsequent_decisions"]["max_us"] / 1000, "n": s["subsequent_decisions"]["count"],
                "first_median_ms": s["first_decision_after_reset"]["median_us"] / 1000,
                "stack_watermark_bytes": max(m["main_stack_peak_used_bytes"] for m in report["observations"]["memory"]),
                "stack_reserved_bytes": report["observations"]["memory"][0]["main_stack_reserved_bytes"],
                "physical_gpio_verified": report.get("physical_gpio_verified"), "parity_complete_field": report.get("parity_complete")}
    v1 = {**lat(timing_v1), "live_flash": receipt_v1["flash_bytes"], "live_ram": receipt_v1["static_ram_bytes"],
          "timing_flash": flash_t1, "timing_ram": ram_t1,
          "parity_pass": parity_v1["summary"]["windows"] - parity_v1["summary"]["failed_windows"],
          "parity_total": parity_v1["summary"]["windows"], "parity_original_1e4_failed": parity_v1["summary"]["original_absolute_1e_4_failed_windows"],
          "parity_max_p_err": parity_v1["summary"]["max_probability_absolute_error"],
          "parity_candidate_disagreements": parity_v1["summary"]["candidate_disagreements"],
          "parity_complete": parity_v1.get("parity_complete")}
    v2 = {**lat(timing_v2), "live_flash": flash_l2, "live_ram": ram_l2, "timing_flash": flash_t2, "timing_ram": ram_t2,
          "parity_pass": parity_v2["summary"]["decisions"] - parity_v2["summary"]["failed"], "parity_total": parity_v2["summary"]["decisions"],
          "parity_original_1e4_failed": parity_v2["summary"]["original_absolute_1e_4_failed_vs_python"],
          "parity_max_p_err": parity_v2["summary"]["max_board_host_probability_error"],
          "parity_candidate_disagreements": None, "parity_complete": parity_v2.get("parity_complete"),
          "old_image_median_ms": timing_v2_old["latency_summary"]["subsequent_decisions"]["median_us"] / 1000}
    return {"v1": v1, "v2": v2}


def section_f(ctx):
    b = ctx["board"]
    rows = []
    for version, n_in, passes in (("v1", 14, 15), ("v2", 16, 17)):
        d = b[version]
        params = models.tinykan_param_count(n_in, 12, 8)
        rows.append({
            "model": f"TinyKAN {version} ({n_in}, 12, 1)", "deployed": "yes", "board": "Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F, 64 MHz)",
            "arithmetic": "native FP32 (ADR-0004)", "params": params, "fp32_weight_bytes": 4 * params,
            "header_constant_bytes": 4 * models.tinykan_header_floats(n_in, 12, 8),
            "forward_passes_per_decision": passes,
            "live_image_flash_bytes": d["live_flash"], "live_image_static_ram_bytes": d["live_ram"],
            "live_image_size_source": ("outputs/firmware-v1-rollback/rollback-receipt.json (rebuilt v1 live image)" if version == "v1"
                                       else "outputs/firmware-v2-f1a-r2/build.log (r2 live image; built, not flashed)"),
            "timing_image_flash_bytes": d["timing_flash"], "timing_image_static_ram_bytes": d["timing_ram"],
            "timing_image_size_source": ("docs/on-device-timing.md (linker estimate)" if version == "v1" else "outputs/timing-v2-prepared-r2/build.log"),
            "decision_block_median_ms": d["median_ms"], "decision_block_p95_ms": d["p95_ms"], "decision_block_max_ms": d["max_ms"],
            "decision_blocks_n": d["n"], "first_decision_median_ms": d["first_median_ms"],
            "latency_scope": "whole 250-sample block: pushes, filtering, features, baseline inference, perturbation passes, sort, debounce/payload; "
                             "DWT cycle-derived at 64 MHz on accelerated synthetic USB input; not inference time; D2 GPIO unmeasured",
            "latency_source": "outputs/parity/timing-board-v1/report.json" if version == "v1" else "outputs/timing-v2-board-20261022-r2/report.json",
            "fifo_headroom_vs_640ms": 640.0 / d["max_ms"],
            "board_parity": f"{d['parity_pass']}/{d['parity_total']} adopted checks pass",
            "board_parity_original_1e4_failed": f"{d['parity_original_1e4_failed']}/{d['parity_total']}",
            "board_parity_max_probability_error": d["parity_max_p_err"],
            "board_parity_source": ("outputs/parity/qualification/locked-board-replay-v1/report.json" if version == "v1"
                                    else "outputs/parity/v2-board-20261022-r2/report.json"),
            "parity_complete": ("false (report field)" if d["parity_complete"] is False else
                                "not a field in the board report; host r2 run 7/71,462 fail, parity_complete false (docs/v2-flash-runbook.md)"),
            "physical_gpio_verified": str(d["physical_gpio_verified"]).lower(),
            "main_stack_watermark_bytes": f"{d['stack_watermark_bytes']}/{d['stack_reserved_bytes']} (timing image; not peak RAM)",
            "peak_ram": "not measured (board-gated; stack watermark and heap snapshots are not whole-device peak RAM)",
            "energy": "not measured (no power instrumentation)",
        })
    for version, n_in in (("v1", 14), ("v2", 16)):
        rows.append({"model": f"LogReg {version} ({n_in} inputs)", "deployed": "no (host comparator)", "board": "not deployed",
                     "arithmetic": "host float64", "params": n_in + 1, "fp32_weight_bytes": 4 * (n_in + 1),
                     **{k: "not measured (never deployed)" for k in (
                         "header_constant_bytes", "forward_passes_per_decision", "live_image_flash_bytes", "live_image_static_ram_bytes",
                         "live_image_size_source", "timing_image_flash_bytes", "timing_image_static_ram_bytes", "timing_image_size_source",
                         "decision_block_median_ms", "decision_block_p95_ms", "decision_block_max_ms", "decision_blocks_n",
                         "first_decision_median_ms", "latency_scope", "latency_source", "fifo_headroom_vs_640ms", "board_parity",
                         "board_parity_original_1e4_failed", "board_parity_max_probability_error", "board_parity_source",
                         "parity_complete", "physical_gpio_verified", "main_stack_watermark_bytes", "peak_ram", "energy")}})
    table = pd.DataFrame(rows)
    table.to_csv(ctx["out"] / "f_deployment.csv", index=False, lineterminator="\n")
    return {version: {"median_ms": b[version]["median_ms"], "max_ms": b[version]["max_ms"],
                      "live_flash": b[version]["live_flash"], "live_static_ram": b[version]["live_ram"],
                      "parity": f"{b[version]['parity_pass']}/{b[version]['parity_total']}"} for version in ("v1", "v2")}


# ----------------------------------------------------------------------------- section G

DISTANCE_BINS = [(-np.inf, -10), (-10, -5), (-5, -2), (-2, -1), (1, 2), (2, 5), (5, 10), (10, np.inf)]


def distance_label(lo, hi):
    def f(v):
        return "-inf" if v == -np.inf else ("inf" if v == np.inf else f"{v:+g}")
    return f"[{f(lo)}, {f(hi)})"


def section_g(ctx, plt):
    rows, conc_rows, perf_rows = [], [], []
    curves = {}
    head = {}
    for version, frame, p in (("v1", ctx["v1"], ctx["p"]["v1_tinykan"]), ("v2", ctx["v2"], ctx["p"]["v2_tinykan"])):
        y = frame["label"].to_numpy(int)
        pred = p >= THRESHOLD
        distance = frame["median_spo2"].to_numpy(float) - 90.0
        inside = int(np.sum((distance >= -1) & (distance < 1)))
        for lo, hi in DISTANCE_BINS:
            mask = (distance >= lo) & (distance < hi)
            n = int(mask.sum())
            errors = int(np.sum(pred[mask] != (y[mask] == 1)))
            rows.append({"version": version, "distance_bin": distance_label(lo, hi), "bin_low": lo, "bin_high": hi,
                         "true_class": "y = 1 (ScalcO2 < 90)" if hi <= 0 else "y = 0", "n": n, "errors": errors,
                         "error_type": "FN" if hi <= 0 else "FP", "error_rate": errors / n if n else float("nan"),
                         "mean_probability": float(np.mean(p[mask])) if n else float("nan")})
        head[f"{version}_windows_within_1_of_cut"] = inside
        recordings = frame["recording_id"].to_numpy()
        per = pd.DataFrame({"recording": recordings, "fp": (~(y == 1)) & pred, "fn": (y == 1) & ~pred})
        per = per.groupby("recording").agg(fp=("fp", "sum"), fn=("fn", "sum"))
        per["errors"] = per.fp + per.fn
        for kind in ("fp", "fn", "errors"):
            counts = per[kind].to_numpy()
            conc_rows.append({"version": version, "error_type": kind.upper() if kind != "errors" else "FP+FN",
                              "total_errors": int(counts.sum()), "recordings_total": int(len(counts)),
                              "recordings_with_any": int(np.sum(counts > 0)),
                              "recordings_holding_50pct": lm.groups_holding_share(counts, 0.5),
                              "recordings_holding_80pct": lm.groups_holding_share(counts, 0.8),
                              "largest_single_recording_share": float(counts.max() / counts.sum()) if counts.sum() else float("nan")})
            curves[(version, kind)] = lm.concentration_curve(counts)
        if version == "v2":
            quartile = pd.qcut(frame["ir_perfusion_index"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
            edges = np.quantile(frame["ir_perfusion_index"], [0, 0.25, 0.5, 0.75, 1.0])
            for qi, q in enumerate(quartile.cat.categories):
                mask = (quartile == q).to_numpy()
                m = lm.binary_metrics(y[mask], p[mask], THRESHOLD)
                perf_rows.append({"version": version, "quartile": q, "ir_perfusion_index_low": edges[qi], "ir_perfusion_index_high": edges[qi + 1],
                                  "n": m["n"], "prevalence": m["prevalence"], "error_rate": 1 - m["accuracy"],
                                  "fn_rate_among_y1": 1 - m["recall"], "fp_rate_among_y0": 1 - m["specificity"],
                                  "recall": m["recall"], "specificity": m["specificity"], "auc": m["auc"]})
    dist = pd.DataFrame(rows)
    conc = pd.DataFrame(conc_rows)
    perf = pd.DataFrame(perf_rows)
    write_csv(dist, ctx["out"] / "g_error_by_spo2_distance.csv")
    write_csv(conc, ctx["out"] / "g_error_concentration.csv")
    write_csv(perf, ctx["out"] / "g_error_by_perfusion_quartile.csv")

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    x = np.arange(len(DISTANCE_BINS))
    for offset, version, color, hatch in ((-0.2, "v1", BLUE, None), (0.2, "v2", ORANGE, "////")):
        block = dist[dist.version == version]
        ax.bar(x + offset, block.error_rate, width=0.38, color=color, edgecolor="white", linewidth=0.4, hatch=hatch,
               label=f"TinyKAN {version}")
    ax.set_xticks(x)
    ax.set_xticklabels([distance_label(lo, hi).replace("inf", "∞") for lo, hi in DISTANCE_BINS], rotation=35, ha="right", fontsize=5.8)
    ax.set_xlabel("Reference median ScalcO2 - 90 (left: y = 1, errors FN; right: y = 0, errors FP)", fontsize=6.2)
    ax.set_ylabel("Error rate at 0.50")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("(a) Error rate vs distance from the cut")
    ax = axes[1]
    for (version, kind), (cx, cy) in curves.items():
        if kind == "errors":
            continue
        ax.plot(cx, cy, color=BLUE if version == "v1" else ORANGE, linestyle="-" if kind == "fp" else "--",
                marker="o" if version == "v1" else "s", markevery=0.1, markersize=2.8,
                label=f"{version} {kind.upper()}")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6)
    ax.axhline(0.5, color=MUTED, linewidth=0.5, linestyle=":")
    ax.set_xlabel("Share of recordings (most errors first)")
    ax.set_ylabel("Cumulative share of errors")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.set_title("(b) Concentration by recording")
    ax = axes[2]
    xq = np.arange(len(perf))
    ax.bar(xq - 0.2, perf.fn_rate_among_y1, width=0.38, color=BLUE, edgecolor="white", linewidth=0.4, label="FN rate (y = 1)")
    ax.bar(xq + 0.2, perf.fp_rate_among_y0, width=0.38, color=ORANGE, edgecolor="white", linewidth=0.4, hatch="////", label="FP rate (y = 0)")
    ax.set_xticks(xq)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("v2 ir_perfusion_index quartile")
    ax.set_ylabel("Rate at 0.50")
    ax.set_ylim(0, max(0.3, float(np.nanmax(np.r_[perf.fn_rate_among_y1, perf.fp_rate_among_y0])) * 1.2))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("(c) v2 by perfusion quartile")
    save_figure(fig, ctx["out"] / "fig_g_error_analysis.png")

    for version in ("v1", "v2"):
        c = conc[(conc.version == version)].set_index("error_type")
        block = dist[dist.version == version]
        head[f"{version}_concentration"] = {k: {"total": int(c.loc[k, "total_errors"]), "with_any": int(c.loc[k, "recordings_with_any"]),
                                                "holding_50pct": int(c.loc[k, "recordings_holding_50pct"]),
                                                "recordings_total": int(c.loc[k, "recordings_total"])} for k in ("FP", "FN", "FP+FN")}
        head[f"{version}_error_rate_by_distance"] = dict(zip(block.distance_bin, [float(v) for v in block.error_rate]))
    head["v2_perfusion_quartiles"] = {r["quartile"]: {"error_rate": r["error_rate"], "fn_rate": r["fn_rate_among_y1"], "fp_rate": r["fp_rate_among_y0"]}
                                      for r in perf_rows}
    ctx["tables"]["g_conc"] = conc
    ctx["tables"]["g_dist"] = dist
    ctx["tables"]["g_perf"] = perf
    return head


# ----------------------------------------------------------------------------- paper sentences

def paper_sentences(ctx, heads):
    a = ctx["tables"]["a"]
    lines = ["# Paper-ready sentences (generated; each cites the CSV it was computed from)", "",
             "Generated by scripts/analyze_lamaakal_style.py from the tables in this directory. Edit wording in the",
             "script, not here, so the numbers cannot drift from the CSVs.", ""]

    def row(table, **query):
        mask = np.ones(len(table), bool)
        for k, v in query.items():
            mask &= (table[k] == v).to_numpy()
        if mask.sum() != 1:
            raise RuntimeError(f"Expected one row for {query}, found {mask.sum()}")
        return table[mask].iloc[0]

    # A
    lines += ["## A. Baselines (a_baselines.csv)", ""]
    for version, desc in (("v1", "the reported v1 inputs (14 IR features, as-coded labels)"),
                          ("v2", "the v2 inputs (14 IR + 2 red/IR, header-aligned labels)")):
        block = a[(a.version == version) & (a.grouping == "recording")]
        tk = row(block, learner="tinykan")
        lr = row(block, learner="logreg")
        best = block.sort_values("auc", ascending=False).iloc[0]
        worst = block.sort_values("auc").iloc[0]
        rank = int((block.auc > tk.auc).sum() + 1)
        lines.append(
            f"- On {desc}, recording-grouped 5-fold out-of-fold windows ({int(tk.n_windows):,} windows, {int(tk.n_groups)} recordings), "
            f"TinyKAN reached AUC {fmt(tk.auc)} with recall {fmt(tk.recall)} and specificity {fmt(tk.specificity)} at 0.50; "
            f"logistic regression reached AUC {fmt(lr.auc)} (recall {fmt(lr.recall)}, specificity {fmt(lr.specificity)}). "
            f"Across {len(block)} models on the same folds, AUC ranged from {fmt(worst.auc)} ({worst.model}) to {fmt(best.auc)} ({best.model}); "
            f"TinyKAN ranked {rank} of {len(block)} by AUC.")
    pat = a[(a.version == "v2") & (a.grouping == "patient")]
    lines.append(f"- With patient-grouped folds on v2, TinyKAN AUC was {fmt(row(pat, learner='tinykan').auc)} and logistic regression "
                 f"{fmt(row(pat, learner='logreg').auc)}.")
    lines += ["- No raw-waveform 1D-CNN was run: it needs a raw-window export that this repository does not have.", ""]

    # B
    bt = ctx["tables"]["b"]
    t1 = row(bt, version="v1", model="TinyKAN", threshold=0.5)
    t1_030 = row(bt, version="v1", model="TinyKAN", threshold=0.3)
    t1_070 = row(bt, version="v1", model="TinyKAN", threshold=0.7)
    t2 = row(bt, version="v2", model="TinyKAN", threshold=0.5)
    t1_045 = row(bt, version="v1", model="TinyKAN", threshold=0.45)
    t1_055 = row(bt, version="v1", model="TinyKAN", threshold=0.55)
    t1_060 = row(bt, version="v1", model="TinyKAN", threshold=0.6)
    v2_block = bt[(bt.version == "v2") & (bt.model == "TinyKAN")]
    lines += ["## B. Operating-point sensitivity (b_operating_point_sweep.csv)", "",
              f"- The v1 operating point is fragile: v1 TinyKAN recall is {fmt(t1_045.recall)} at 0.45, {fmt(t1.recall)} at 0.50, "
              f"{fmt(t1_055.recall)} at 0.55 and {fmt(t1_060.recall)} at 0.60, because its out-of-fold probabilities are compressed around 0.5. "
              f"v2 TinyKAN balanced accuracy stays between {fmt(v2_block.balanced_accuracy.min())} and {fmt(v2_block.balanced_accuracy.max())} "
              "across 0.30-0.70.",
              f"- Between thresholds 0.30 and 0.70, v1 TinyKAN recall fell from {fmt(t1_030.recall)} to {fmt(t1_070.recall)} while specificity "
              f"rose from {fmt(t1_030.specificity)} to {fmt(t1_070.specificity)}; at the reported 0.50 it was {fmt(t1.recall)} and {fmt(t1.specificity)} "
              f"(balanced accuracy {fmt(t1.balanced_accuracy)}).",
              f"- The 97.1% recall quoted for v1 is the 0.30 cut (recall {fmt(t1_030.recall)}, specificity {fmt(t1_030.specificity)}); it is not the operating point.",
              f"- At 0.50, v2 TinyKAN gave recall {fmt(t2.recall)} and specificity {fmt(t2.specificity)}. The sweep is descriptive; 0.50 was fixed before it and is not re-selected from it.", ""]

    # C
    ci = ctx["tables"]["c_ci"]
    paired = ctx["tables"]["c_paired"]
    seeds = ctx["tables"]["c_seed"]
    folds = ctx["tables"]["c_fold"]
    lines += ["## C. Robustness and uncertainty (c_bootstrap_ci.csv, c_paired_comparison.csv, c_fold_summary.csv, c_seed_summary.csv)", ""]
    for version, grouping in (("v1", "recording"), ("v2", "recording")):
        tk_auc = row(ci, version=version, grouping=grouping, model="TinyKAN", metric="auc")
        lr_auc = row(ci, version=version, grouping=grouping, model="LogReg", metric="auc")
        tk_rec = row(ci, version=version, grouping=grouping, model="TinyKAN", metric="recall")
        tk_spec = row(ci, version=version, grouping=grouping, model="TinyKAN", metric="specificity")
        boot = row(paired, version=version, grouping=grouping, metric="auc", method=f"paired {grouping} bootstrap ({N_BOOT} resamples, 95% percentile CI)")
        fold_tk = row(folds, version=version, grouping=grouping, model="TinyKAN")
        lines.append(
            f"- {version}: a recording-level bootstrap ({N_BOOT} resamples of {int(tk_auc.n_units)} recordings) gives TinyKAN AUC {fmt(tk_auc.point)} "
            f"(95% CI {fmt(tk_auc.ci_low)}–{fmt(tk_auc.ci_high)}), recall {fmt(tk_rec.point)} ({fmt(tk_rec.ci_low)}–{fmt(tk_rec.ci_high)}) and specificity "
            f"{fmt(tk_spec.point)} ({fmt(tk_spec.ci_low)}–{fmt(tk_spec.ci_high)}); logistic regression AUC is {fmt(lr_auc.point)} "
            f"({fmt(lr_auc.ci_low)}–{fmt(lr_auc.ci_high)}). The paired TinyKAN minus logistic-regression AUC difference is "
            f"{boot.difference:+.3f} (95% CI {boot.ci_low:+.3f} to {boot.ci_high:+.3f}). Across the 5 folds TinyKAN AUC was "
            f"{fmt(fold_tk.auc_mean)} ± {fmt(fold_tk.auc_sd)} (mean ± sd).")
    seed_runs = ctx["tables"]["c_seed_runs"]
    for version in ("v1", "v2"):
        s = row(seeds, version=version)
        lines.append(f"- Retraining the {version} TinyKAN recipe with {int(s.n_seeds)} seeds ({s.seeds}) on the same folds gave out-of-fold AUC "
                     f"{fmt(s.auc_mean)} ± {fmt(s.auc_sd)}, recall {fmt(s.recall_mean)} ± {fmt(s.recall_sd)} and specificity "
                     f"{fmt(s.specificity_mean)} ± {fmt(s.specificity_sd)} at 0.50. These are analysis models, not the shipped model.")
    v1_runs = seed_runs[seed_runs.version == "v1"]
    s42 = row(v1_runs, seed=42)
    lines.append(f"- The reported v1 seed (42) has the lowest AUC ({fmt(s42.auc)}) and the highest recall ({fmt(s42.recall)}) of the six v1 seeds; "
                 f"the other five give recall {fmt(v1_runs[v1_runs.seed != 42].recall.min())}-{fmt(v1_runs[v1_runs.seed != 42].recall.max())} and "
                 f"specificity {fmt(v1_runs[v1_runs.seed != 42].specificity.min())}-{fmt(v1_runs[v1_runs.seed != 42].specificity.max())}. "
                 "The reported 0.50 trade-off is one draw from a seed-sensitive split, not a stable property of the recipe.")
    lines += ["- Windows from one recording are correlated, so window-level DeLong p-values in c_paired_comparison.csv are anti-conservative; "
              "the fold-level t-test has n = 5. The recording bootstrap is the comparison to report.", ""]

    # D
    hd = heads["D"]
    lines += ["## D. Explanation case studies and spline inspection (d_*.csv)", ""]
    for version in ("v1", "v2"):
        cs = hd[f"{version}_case_studies"]
        lines.append(f"- For the shipped {version} header on its own training windows (in-sample), the median-probability TP window "
                     f"(P = {fmt(cs['TP']['probability'])}, {cs['TP']['nonzero_deltas']} of {cs['TP']['n_features']} Δ non-zero) has top-3 |Δ| "
                     f"features {', '.join(cs['TP']['top3'])}; the median FP window (P = {fmt(cs['FP']['probability'])}, "
                     f"{cs['FP']['nonzero_deltas']} of {cs['FP']['n_features']} Δ non-zero) has {', '.join(cs['FP']['top3'])}. A feature is "
                     "'toward normal' when replacing it with its training mean raises P; it does not mean the patient is normal.")
    shape = {r["version"]: r for r in hd["output_spline_shape"]}
    s2 = shape["v2"]
    lines.append(f"- The v2 output spline is not monotone: P peaks at {fmt(s2['max_p'])} near u = {s2['u_at_max_p']:.2f} and is "
                 f"{fmt(s2['p_at_u_plus3_clamp'])} at the +3 clamp, where {pct(s2['fraction_windows_u_above_3'])} of v2 windows sit "
                 "(d_output_spline_shape.csv). A replacement that pulls a clamped window back just inside the range raises P, so on "
                 "saturated risk windows the largest |Δ| is often a feature that lowered P.")
    for version in ("v1", "v2"):
        gl = hd[f"{version}_global"]
        lines.append(f"- {version}: the most frequent top-1 feature by |Δ| is {gl['top1_most_common_y1'][0]} on y = 1 windows "
                     f"({pct(gl['top1_most_common_y1'][1])}) and {gl['top1_most_common_y0'][0]} on y = 0 windows ({pct(gl['top1_most_common_y0'][1])}). "
                     f"A top driver does not identify an ill patient.")
    deg = hd["v2_degeneracy"]["candidate (P >= 0.50)"]
    deg_tp = hd["v2_degeneracy"]["TP (in-sample)"]
    lines.append(f"- On {pct(deg['fraction_top3_has_zero_delta_slot'])} of v2 candidate windows (P >= 0.50) at least one of the three transmitted "
                 f"features has Δ = 0 and is placed by feature index, because the output spline is clamped on both passes "
                 f"({pct(deg['fraction_output_preactivation_clamped'])} of candidate windows have a clamped output pre-activation). "
                 f"On {pct(deg['fraction_top1_opposes_decision'])} of v2 candidate windows the top-1 feature lowered P (d_explanation_degeneracy.csv).")
    a1, a2 = hd["v1_aopc"], hd["v2_aopc"]
    lines.append(f"- Host deletion AOPC (write-once v1 run): firmware-absolute ranking {a1['firmware_absolute_all']:.4f} vs random {a1['random_mean_all']:.4f} "
                 f"over all windows; on non-candidate windows firmware-absolute is {a1['firmware_absolute_noncandidate']:.4f} vs random "
                 f"{a1['random_mean_noncandidate']:.4f}, i.e. worse than random.")
    lines.append(f"- A new in-sample v2 run gives firmware-absolute {a2['firmware_absolute_all']:.4f} vs random {a2['random_mean_all']:.4f} over all windows, "
                 f"{a2['firmware_absolute_candidate']:.4f} vs {a2['random_mean_candidate']:.4f} on candidate windows, "
                 f"and {a2['firmware_absolute_noncandidate']:.4f} vs {a2['random_mean_noncandidate']:.4f} on non-candidate windows (d_aopc_v2_new_run.csv).")
    lines.append(f"- Both AOPC runs fix the target at P(risk). On non-candidate windows a predicted-class target (1 - P) mirrors every curve, so the "
                 f"values change sign (v1 {-a1['firmware_absolute_noncandidate']:.4f} vs {-a1['random_mean_noncandidate']:.4f}; v2 "
                 f"{-a2['firmware_absolute_noncandidate']:.4f} vs {-a2['random_mean_noncandidate']:.4f}); 'worse than random on non-candidates' "
                 "is a statement about the fixed P(risk) target, and should be written with it.")
    lines += ["- The spline figures are architecture inspection, not operator XAI; hidden units are not detectors.", ""]

    # E
    he = heads["E"]
    lines += ["## E. Ablation (e_ablation.csv)", "",
              f"- Hidden width does not move v1: out-of-fold AUC {', '.join(fmt(v) for v in he['v1_auc_hidden_4_12_32'])} for 4, 12 and 32 hidden units "
              f"(v2: {', '.join(fmt(v) for v in he['v2_auc_hidden_4_12_32'])}). Capacity is not the ceiling.",
              f"- Header-aligned labels lift the 14-IR TinyKAN from AUC {fmt(he['label_alignment_auc_as_coded_vs_header_aligned_14IR'][0])} to "
              f"{fmt(he['label_alignment_auc_as_coded_vs_header_aligned_14IR'][1])} (different rows and folds), and adding the two red/IR inputs lifts it to "
              f"{fmt(he['inputs_auc_14IR_16_2redIR'][1])}; the two red/IR inputs alone give {fmt(he['inputs_auc_14IR_16_2redIR'][2])}.",
              f"- Logistic regression on the two red/IR inputs alone reaches AUC {fmt(he['logreg_auc_v1_14_v2_14_v2_16_v2_2'][3])}, against "
              f"{fmt(he['logreg_auc_v1_14_v2_14_v2_16_v2_2'][2])} on all 16 inputs and {fmt(he['logreg_auc_v1_14_v2_14_v2_16_v2_2'][1])} on the 14 IR "
              "features: most of the v2 gain comes from the red/IR relation, not from the spline network.", ""]

    # F
    hf = heads["F"]
    lines += ["## F. Deployment (f_deployment.csv)", "",
              f"- On the Nano 33 BLE Sense (nRF52840, 64 MHz), the whole 250-sample decision block, including the explanation passes, has a cycle-derived "
              f"median of {hf['v1']['median_ms']:.1f} ms for v1 and {hf['v2']['median_ms']:.1f} ms for v2 on synthetic USB input; this is not inference "
              f"time, and GPIO timing, peak RAM and energy were not measured.",
              f"- Board parity: v1 {hf['v1']['parity']} locked windows pass the adopted local-scale checks; v2 r2 {hf['v2']['parity']} decisions pass. "
              "parity_complete stays false.", ""]

    # G
    conc = ctx["tables"]["g_conc"]
    lines += ["## G. Error analysis (g_*.csv)", ""]
    for version in ("v1", "v2"):
        fp = row(conc, version=version, error_type="FP")
        fn = row(conc, version=version, error_type="FN")
        lines.append(f"- {version}: half of the {int(fp.total_errors):,} false positives sit in {int(fp.recordings_holding_50pct)} of "
                     f"{int(fp.recordings_total)} recordings, and half of the {int(fn.total_errors):,} false negatives in {int(fn.recordings_holding_50pct)}.")
    dist = ctx["tables"]["g_dist"]
    perf = ctx["tables"]["g_perf"]
    near_fn = row(dist, version="v2", distance_bin="[-2, -1)")
    far_fn = row(dist, version="v2", distance_bin="[-inf, -10)")
    near_fp = row(dist, version="v2", distance_bin="[+1, +2)")
    far_fp = row(dist, version="v2", distance_bin="[+10, inf)")
    v1_fp_rates = dist[(dist.version == "v1") & (dist.error_type == "FP")].error_rate
    lines.append(f"- v2 errors concentrate near the cut: the FN rate is {fmt(near_fn.error_rate)} for reference ScalcO2 1-2 points below 90 and "
                 f"{fmt(far_fn.error_rate)} more than 10 below; the FP rate is {fmt(near_fp.error_rate)} 1-2 points above and {fmt(far_fp.error_rate)} "
                 f"more than 10 above ({int(far_fp.n)} windows). v1 shows no such gradient: its FP rate is {fmt(v1_fp_rates.min())}-{fmt(v1_fp_rates.max())} "
                 "in every bin above 90.")
    q1 = row(perf, quartile="Q1 (lowest)")
    q4 = row(perf, quartile="Q4 (highest)")
    lines.append(f"- v2 error rate is highest in the lowest ir_perfusion_index quartile ({fmt(q1.error_rate)}, mostly FP: FP rate {fmt(q1.fp_rate_among_y0)}) "
                 f"and the highest quartile ({fmt(q4.error_rate)}, mostly FN: FN rate {fmt(q4.fn_rate_among_y1)}).")
    lines.append("- v1 distances are to the as-coded (time-misaligned) reference, so v1 error-by-distance does not measure closeness to a true saturation cut.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--jobs", type=int, default=8, help="worker processes for the retraining jobs")
    parser.add_argument("--job-cache", type=Path, help="development only: reuse/save retraining results in this pickle "
                                                       "(the committed run and the rerun check do not use it)")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    write_once_before = {str(p.relative_to(ROOT).as_posix()): tree_hashes(p) for p in WRITE_ONCE}
    input_hashes = {key: sha256(path) for key, path in P.items()}

    v1, v1_kan, v1_lr, v2, v2_oof, aprime, checks = load_inputs()
    ctx = {"v1": v1, "v2": v2, "checks": checks, "tables": {}}
    ctx["y1"] = v1["label"].to_numpy(int)
    ctx["y2"] = v2["label"].to_numpy(int)
    ctx["folds1"] = v1_kan["split_index"].to_numpy(int)
    ctx["folds2"] = v2_oof["v2_tinykan16_oof"]["fold"].to_numpy(int)
    ctx["folds2p"] = v2_oof["v2_tinykan16_patient_oof"]["fold"].to_numpy(int)
    ctx["p"] = {
        "v1_tinykan": v1_kan["y_prob"].to_numpy(float), "v1_logreg": v1_lr["probability_low_oxygen_risk"].to_numpy(float),
        "v2_tinykan": v2_oof["v2_tinykan16_oof"]["y_prob"].to_numpy(float), "v2_logreg": v2_oof["v2_logreg16_oof"]["y_prob"].to_numpy(float),
        "v2_tinykan_patient": v2_oof["v2_tinykan16_patient_oof"]["y_prob"].to_numpy(float),
        "v2_tinykan14": v2_oof["v2_tinykan14_oof"]["y_prob"].to_numpy(float),
        "v1labels_tinykan14_e100": aprime["y_prob"].to_numpy(float),
    }
    ctx["board"] = board_numbers()

    jobs, data = build_jobs(v1, v1_kan, v2, v2_oof)
    if args.job_cache and args.job_cache.exists():
        import pickle
        ctx["results"] = pickle.loads(args.job_cache.read_bytes())
        print(f"Loaded {len(ctx['results'])} cached jobs from {args.job_cache} (development run)", flush=True)
    else:
        print(f"Running {len(jobs)} cross-validation jobs with {args.jobs} workers ...", flush=True)
        ctx["results"] = run_jobs(jobs, data, args.jobs)
        if args.job_cache:
            import pickle
            args.job_cache.write_bytes(pickle.dumps(ctx["results"]))
    ctx["job_cache_used"] = bool(args.job_cache)

    out.mkdir(parents=True)
    ctx["out"] = out
    plt = setup_style()
    heads = {}
    print("Section A", flush=True)
    heads["A"] = section_a(ctx)
    print("Section B", flush=True)
    heads["B"] = section_b(ctx, plt)
    print("Section C", flush=True)
    heads["C"] = section_c(ctx)
    print("Section D", flush=True)
    heads["D"] = section_d(ctx, plt)
    print("Section E", flush=True)
    heads["E"] = section_e(ctx)
    print("Section F", flush=True)
    heads["F"] = section_f(ctx)
    print("Section G", flush=True)
    heads["G"] = section_g(ctx, plt)

    for version, frame, n_rows in (("v1", v1, len(v1)), ("v2", v2, len(v2))):
        columns = {"row_index": np.arange(n_rows), "recording_id": frame["recording_id"].to_numpy(),
                   "fold": ctx["folds1"] if version == "v1" else ctx["folds2"], "y_true": frame["label"].to_numpy(int)}
        for name in sorted(ctx["results"]):
            if name.startswith(version + "_") or (version == "v2" and name.startswith("v2patient_")):
                columns[name] = ctx["results"][name]["probabilities"]
        if version == "v2":
            columns["fold_patient"] = ctx["folds2p"]
        write_gzip_csv(pd.DataFrame(columns), out / f"oof_{version}_analysis_models.csv.gz")

    (out / "paper_sentences.md").write_text(paper_sentences(ctx, heads), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(
        "# analysis-lamaakal-style-v1\n\n"
        "Lamaakal-style results analysis of the frozen TinyKAN v1 (reported) and v2 (comparison). v3 is out of scope.\n"
        "Generated by `scripts/analyze_lamaakal_style.py`; write-up and limits in `docs/analysis-lamaakal-style.md`.\n"
        "Retrained models in this directory are analysis models only; they are not the shipped model and never replace a header.\n"
        "`summary.json` holds input hashes, seeds, versions, checks and every headline number.\n",
        encoding="utf-8", newline="\n")

    write_once_after = {str(p.relative_to(ROOT).as_posix()): tree_hashes(p) for p in WRITE_ONCE}
    if write_once_after != write_once_before:
        raise RuntimeError("A write-once directory changed during the run")

    import matplotlib
    import scipy
    import sklearn
    import torch
    summary = {
        "created_utc": created,
        "duration_seconds": round(time.perf_counter() - started, 1),
        "script": "scripts/analyze_lamaakal_style.py",
        "scope": "host only; v1 reported, v2 comparison, v3 excluded; no board I/O; no firmware, header or existing outputs/ writes",
        "job_cache_used": ctx["job_cache_used"],
        "threshold": THRESHOLD, "debounce_N": 2,
        "seeds": {"tinykan_retrain": list(SEEDS), "bootstrap_base": BOOTSTRAP_SEED, "n_boot": N_BOOT,
                  "aopc": AOPC_SEED, "aopc_random_repeats": AOPC_RANDOM_REPEATS, "baselines": models.BASELINE_SEED},
        "recipes": {"v1_tinykan_cv": {**V1_RECIPE, "lr": 0.01, "weight_decay": 1e-4, "source": "notebook 12 / scripts/sweep_kan.py"},
                    "v2_tinykan_cv": {**V2_RECIPE, "lr": 0.01, "weight_decay": 1e-4, "source": "scripts/train_tinykan_v2.py"},
                    "baselines": models.BASELINE_SETTINGS, "svm_subsample": models.SVM_SUBSAMPLE},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                        "matplotlib": matplotlib.__version__, "executable": sys.executable,
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "inputs_sha256": {str(P[k].relative_to(ROOT).as_posix()): v for k, v in input_hashes.items()},
        "write_once_unchanged": True,
        "write_once_sha256": write_once_after,
        "checks": ctx["checks"],
        "job_seconds": {name: round(r["seconds"], 1) for name, r in sorted(ctx["results"].items())},
        "headlines": heads,
        "outputs_sha256": {f.name: sha256(f) for f in sorted(out.iterdir()) if f.is_file()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Done in {summary['duration_seconds']} s -> {out}")
    return 0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not serialisable: {type(value)}")


if __name__ == "__main__":
    raise SystemExit(main())

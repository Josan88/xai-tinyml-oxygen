"""Physiological-input TinyKAN and concept-bottleneck prototype (host only, no board).

Pre-registration: docs/physio-concept-preregistration.md (committed before any of this ran).

Stages (run separately or end to end; every stage writes new files only):
  1        beat features from the v2 waveforms -> physio_features.csv.gz + Stage 1 checks
  2        P5 / P3 / V16+P5 x TinyKAN, LogReg, HistGB on the v2 folds; the v2-16 TinyKAN
           seed-42 reproduction check runs first and stops the stage if it fails
  3        explanation checks on the P5 all-window model and the shipped v2 header; the v2
           AOPC same-code check runs first and stops the stage if it fails
  4        sequential concept bottleneck (estimated ScalcO2 -> risk), cross-fitted
  summary  summary.json + README.md from the stage files
  rerun    fresh Stage 1 into build/physio/rerun + uncached retrain of a cheap job subset -> rerun_check.json
  superseded  compare an earlier run's outputs (--against DIR) with the current ones -> superseded_check.json

Rules kept here: no board I/O, no firmware or header writes, no write into an existing
output file or a write-once directory (hashed before and after each stage). Every trained
model is an analysis model. median_spo2, min_spo2, label_min_spo2 and label are never
model inputs (asserted for every model); median_spo2 is only the concept target in stage 4.

Run from the repository root with the interpreter that has torch 2.12 / sklearn 1.9:
  py -3.14 scripts/analyze_physio_concept.py --stage 1
  py -3.14 scripts/analyze_physio_concept.py --stage all
TinyKAN jobs run in freshly spawned processes (as analyze_lamaakal_style.run_jobs) and are
cached under build/physio/jobs so that a stage can be split across runs.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import analyze_lamaakal_style as las  # noqa: E402
import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402
import physio_features as pf  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "physio-concept-v1"
CACHE_DEFAULT = ROOT / "build" / "physio" / "jobs"
RERUN_DIR = ROOT / "build" / "physio" / "rerun"
CACHE_VERSION = 1

P = {
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v2_npz": ROOT / "outputs/openox-raw-windows-v1/v2_ir_red_windows.npz",
    "npz_sha256sums": ROOT / "outputs/openox-raw-windows-v1/SHA256SUMS",
    "v2_tinykan16_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "v2_tinykan16_patient_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
    "v2_logreg16_oof": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
    "artefact_rows": ROOT / "outputs/openox-v3/artefact_rows.csv",
    "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "las_aopc_v2": ROOT / "outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv",
    "las_global": ROOT / "outputs/analysis-lamaakal-style-v1/d_global_attribution.csv",
    "las_degeneracy": ROOT / "outputs/analysis-lamaakal-style-v1/d_explanation_degeneracy.csv",
    "las_bootstrap_ci": ROOT / "outputs/analysis-lamaakal-style-v1/c_bootstrap_ci.csv",
    "firmware_main": ROOT / "firmware/Arduino Nano 33/src/main.cpp",
    "dashboard_core": ROOT / "dashboard/dashboard_core.js",
}
WRITE_ONCE = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/ptt-ppg/subject-1-v1",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
    ROOT / "outputs/analysis-lamaakal-style-v1",
    ROOT / "outputs/analysis-addenda-v1",
)
PROTECTED = WRITE_ONCE + (
    ROOT / "outputs/openox-v2",
    ROOT / "outputs/openox-raw-windows-v1",
    ROOT / "outputs/openox-v3/artefact_rows.csv",
    ROOT / "firmware/Arduino Nano 33/include",
)

N_ROWS, N_RECORDINGS, N_PATIENTS = 48662, 154, 87
THRESHOLD = 0.50
P5 = pf.FEATURES                                   # r_beat, pi_beat, hr_bpm, sqi_template, sqi_ibi_ratio
P3 = ("r_beat", "pi_beat", "hr_bpm")
V16 = las.V2_FEATURES
V16P5 = V16 + P5
FEATURE_SETS = {"P5": P5, "P3": P3, "V16+P5": V16P5, "v2-16": V16}
FORBIDDEN_INPUTS = ("median_spo2", "min_spo2", "label_min_spo2", "label")


# ----------------------------------------------------------------------------- utilities

class StopAnalysis(RuntimeError):
    pass


def guard_inputs(features):
    """Label-leakage guard: no label or saturation column may be a model input."""
    leaked = sorted(set(features) & set(FORBIDDEN_INPUTS))
    if leaked:
        raise AssertionError(f"Label leakage: {leaked} used as model inputs")
    return tuple(features)


def sha256(path):
    return las.sha256(path)


def protected_hashes():
    return {p.relative_to(ROOT).as_posix(): las.tree_hashes(p) for p in PROTECTED}


def new_file(out, name):
    path = out / name
    if path.exists():
        raise FileExistsError(f"{path} exists; this script never overwrites an output file")
    return path


def write_csv(frame, out, name):
    las.write_csv(frame, new_file(out, name))


def write_gzip_csv(frame, out, name):
    las.write_gzip_csv(frame, new_file(out, name))


def write_json(payload, out, name):
    new_file(out, name).write_text(json.dumps(payload, indent=2, default=las._json_default) + "\n", encoding="utf-8")


def read_json(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def environment():
    import scipy
    import sklearn
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch_version,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def spearman(a, b):
    from scipy import stats
    return float(stats.spearmanr(a, b).statistic)


# ----------------------------------------------------------------------------- data

def load_v2():
    v2 = pd.read_csv(P["v2_features"])
    v2["recording_id"] = v2["recording_id"].astype(str)
    if len(v2) != N_ROWS or v2["recording_id"].nunique() != N_RECORDINGS or v2["patient_id"].nunique() != N_PATIENTS:
        raise StopAnalysis("v2 table is not the 48,662-window, 154-recording, 87-patient export")
    oof = {}
    for key in ("v2_tinykan16_oof", "v2_tinykan16_patient_oof", "v2_logreg16_oof"):
        frame = pd.read_csv(P[key]).sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(frame["row_index"], np.arange(N_ROWS)):
            raise StopAnalysis(f"{key} does not cover every v2 row once")
        if not (frame["recording_id"].astype(str).to_numpy() == v2["recording_id"].to_numpy()).all():
            raise StopAnalysis(f"{key} recording IDs differ from the v2 table")
        if not (frame["y_true"].to_numpy() == v2["label"].to_numpy()).all():
            raise StopAnalysis(f"{key} labels differ from the v2 table")
        oof[key] = frame
    folds_rec = oof["v2_tinykan16_oof"]["fold"].to_numpy(int)
    folds_pat = oof["v2_tinykan16_patient_oof"]["fold"].to_numpy(int)
    if not np.array_equal(oof["v2_logreg16_oof"]["fold"].to_numpy(int), folds_rec):
        raise StopAnalysis("predictions_c_v2_logreg16.csv folds differ from the recording folds")
    if pd.Series(folds_rec).groupby(v2["recording_id"].to_numpy()).nunique().max() != 1:
        raise StopAnalysis("A recording spans several recording folds")
    if pd.Series(folds_pat).groupby(v2["patient_id"].to_numpy()).nunique().max() != 1:
        raise StopAnalysis("A patient spans several patient folds")
    checks = {"rows": N_ROWS, "recordings": N_RECORDINGS, "patients": N_PATIENTS,
              "prevalence": float(v2["label"].mean()),
              "folds": "recording: fold of predictions_b_v2_tinykan16.csv; patient: fold of "
                       "predictions_b_v2_tinykan16_patient.csv; row_index-aligned, labels and recording IDs equal, "
                       "no recording (patient) spans two recording (patient) folds",
              "recordings_spanning_patient_folds_note": "recording-grouped folds let a patient appear in train and test"}
    return v2, folds_rec, folds_pat, oof, checks


def load_physio(out):
    table = pd.read_csv(out / "physio_features.csv.gz")
    if not np.array_equal(table["row_index"], np.arange(N_ROWS)):
        raise StopAnalysis("physio_features.csv.gz is not row_index-aligned")
    return table


# ----------------------------------------------------------------------------- stage 1

def npz_sha_check():
    expected = {}
    for line in P["npz_sha256sums"].read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        expected[name.lstrip("*")] = digest
    actual = sha256(P["v2_npz"])
    if expected.get(P["v2_npz"].name) != actual:
        raise StopAnalysis(f"v2_ir_red_windows.npz SHA-256 {actual} differs from SHA256SUMS")
    return actual


def load_waveforms(v2, rows=None):
    z = np.load(P["v2_npz"])
    arrays = {key: z[key] for key in ("ir", "red", "ir_dc_mean", "red_dc_mean", "row_index", "recording_id", "label")}
    if not np.array_equal(arrays["row_index"], np.arange(N_ROWS)):
        raise StopAnalysis("npz row_index is not 0..N-1")
    if not (arrays["recording_id"].astype(str) == v2["recording_id"].to_numpy()).all():
        raise StopAnalysis("npz recording IDs differ from the v2 table")
    if not np.array_equal(arrays["label"].astype(int), v2["label"].to_numpy(int)):
        raise StopAnalysis("npz labels differ from the v2 table")
    if float(z["sample_rate_hz"]) != pf.SAMPLE_RATE_HZ or arrays["ir"].shape != (N_ROWS, 500):
        raise StopAnalysis("npz is not 500 samples at 50 Hz")
    if rows is not None:
        arrays = {key: value[rows] for key, value in arrays.items()}
    return arrays


def extract_table(v2, waves, rows):
    feats = pf.extract_many(waves["ir"], waves["red"], waves["ir_dc_mean"], waves["red_dc_mean"])
    table = pd.DataFrame({"row_index": rows, "recording_id": v2["recording_id"].to_numpy()[rows],
                          "start_time_seconds": v2["start_time_seconds"].to_numpy()[rows],
                          "n_beats": feats["n_beats"].astype(int), "beat_valid": feats["beat_valid"].astype(int),
                          "polarity": feats["polarity"].astype(int)})
    for name in P5:
        table[name] = feats[name]
    return table


def stage1(out):
    started = time.perf_counter()
    v2, _, _, _, load_checks = load_v2()
    npz_sha = npz_sha_check()
    waves = load_waveforms(v2)
    print("Stage 1: extracting beat features from 48,662 windows ...", flush=True)
    table = extract_table(v2, waves, np.arange(N_ROWS))
    out.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(table, out, "physio_features.csv.gz")
    table = load_physio(out)            # downstream uses the written (7 significant digit) table
    y = v2["label"].to_numpy(int)
    valid = table["beat_valid"].to_numpy(bool)

    # Row alignment: recompute the two v2 red/IR features from the stored arrays.
    ir = waves["ir"].astype(np.float64)
    red = waves["red"].astype(np.float64)
    pi_npz = ir.std(axis=1) / waves["ir_dc_mean"]
    ror_npz = (red.std(axis=1) / waves["red_dc_mean"]) / pi_npz
    alignment = {}
    for name, value in (("ratio_of_ratios", ror_npz), ("ir_perfusion_index", pi_npz)):
        ref = v2[name].to_numpy(float)
        err = np.abs(value - ref)
        alignment[name] = {"max_abs_error": float(err.max()), "max_rel_error": float(np.max(err / np.abs(ref))),
                           "formula": "std(filtered)/dc as export_openox_windows_v2.two_wavelength_features, from the stored float32 arrays"}
    alignment["aligned"] = bool(all(alignment[n]["max_rel_error"] < 1e-4 for n in ("ratio_of_ratios", "ir_perfusion_index")))
    if not alignment["aligned"]:
        raise StopAnalysis(f"npz rows do not reproduce the v2 table: {alignment}")

    # Validity, plausibility, agreement with the window features.
    hr = table["hr_bpm"].to_numpy(float)
    pol0 = table["polarity"].to_numpy() == 0
    few = table["n_beats"].to_numpy() < pf.MIN_BEATS
    invalid_reasons = {
        "flat_or_nonpositive_dc": int(np.sum(~valid & pol0)),
        "fewer_than_4_beats": int(np.sum(~valid & ~pol0 & few)),
        "no_segment_with_positive_ir_ac": int(np.sum(~valid & ~pol0 & ~few)),
    }
    invalid_block = {
        "beat_valid": int(valid.sum()), "beat_invalid": int((~valid).sum()), "reasons": invalid_reasons,
        "prevalence_beat_invalid": float(y[~valid].mean()) if (~valid).any() else float("nan"),
        "prevalence_beat_valid": float(y[valid].mean()),
        "recordings_with_invalid_windows": int(v2.loc[~valid, "recording_id"].nunique()),
        "artefact_rows_among_invalid": int(np.isin(np.flatnonzero(~valid), pd.read_csv(P["artefact_rows"])["v2_row_index"]).sum()),
    }
    checks = {
        "npz_sha256": npz_sha, "npz_sha_matches_SHA256SUMS": True, "load": load_checks,
        "row_alignment": alignment, "validity": invalid_block,
        "hr_within_40_180_share_of_valid": float(np.mean((hr[valid] >= 40) & (hr[valid] <= 180))),
        "hr_quantiles_valid": dict(zip(("q01", "q05", "q50", "q95", "q99"), np.quantile(hr[valid], [.01, .05, .5, .95, .99]).tolist())),
        "spearman_hr_vs_60x_dominant_frequency": spearman(hr[valid], 60.0 * v2["dominant_frequency_hz"].to_numpy(float)[valid]),
        "spearman_r_beat_vs_ratio_of_ratios": spearman(table["r_beat"].to_numpy(float)[valid], v2["ratio_of_ratios"].to_numpy(float)[valid]),
        "spearman_pi_beat_vs_ir_perfusion_index": spearman(table["pi_beat"].to_numpy(float)[valid], v2["ir_perfusion_index"].to_numpy(float)[valid]),
    }
    hz_match = np.abs(hr[valid] - 60.0 * v2["dominant_frequency_hz"].to_numpy(float)[valid])
    checks["share_hr_within_10_bpm_of_60x_dominant_frequency"] = float(np.mean(hz_match <= 10.0))

    # Polarity per window and per recording.
    pol = table["polarity"].to_numpy(int)
    per_rec = pd.DataFrame({"recording_id": v2["recording_id"], "pol": pol, "valid": valid})
    per_rec = per_rec[per_rec.pol != 0].groupby("recording_id")["pol"].agg(lambda s: float(np.mean(s == 1)))
    majority = np.maximum(per_rec, 1 - per_rec)
    checks["polarity"] = {
        "rule": "s = +1 if P99(diff(x)) >= -P1(diff(x)) else -1 (steepest slope = systolic upstroke points up)",
        "share_windows_plus1": float(np.mean(pol[pol != 0] == 1)), "share_windows_minus1": float(np.mean(pol[pol != 0] == -1)),
        "recordings_majority_plus1": int(np.sum(per_rec >= 0.5)), "recordings_majority_minus1": int(np.sum(per_rec < 0.5)),
        "recordings_with_majority_share_below_0_90": int(np.sum(majority < 0.90)),
        "median_recording_majority_share": float(np.median(majority)),
    }

    # SQI vs the ADR-0007 artefact rows (frozen sensor buffer replays).
    art = pd.read_csv(P["artefact_rows"])
    is_art = np.zeros(N_ROWS, bool)
    is_art[art["v2_row_index"].to_numpy(int)] = True
    sqi_rows = []
    for name, natural in (("sqi_template", -1.0), ("sqi_ibi_ratio", 1.0), ("hr_bpm", 1.0), ("pi_beat", 1.0), ("r_beat", 1.0)):
        v = table[name].to_numpy(float)
        m = valid
        sqi_rows.append({"feature": name, "windows_beat_valid": int(m.sum()), "artefact_rows_beat_valid": int((m & is_art).sum()),
                         "auc_raw_value_flags_artefact": lm.weighted_auc(is_art[m].astype(int), v[m]),
                         "natural_direction": "lower = worse quality" if natural < 0 else "higher = more irregular / larger",
                         "auc_natural_direction": lm.weighted_auc(is_art[m].astype(int), natural * v[m]),
                         "median_artefact": float(np.median(v[m & is_art])), "median_other": float(np.median(v[m & ~is_art])),
                         "q05_artefact": float(np.quantile(v[m & is_art], .05)), "q95_artefact": float(np.quantile(v[m & is_art], .95))})
    sqi_table = pd.DataFrame(sqi_rows)
    write_csv(sqi_table, out, "s1_sqi_vs_artefact_rows.csv")
    checks["sqi_vs_artefact_rows"] = {
        "artefact_rows": int(is_art.sum()), "beat_invalid_share_artefact": float(np.mean(~valid[is_art])),
        "beat_invalid_share_other": float(np.mean(~valid[~is_art])),
        **{f"{r['feature']}_auc_natural_direction": r["auc_natural_direction"] for r in sqi_rows},
        "reading": "AUC 0.5 = no separation; the artefact rows are replayed 0.5 s sensor buffers, which are periodic",
    }

    # Per-feature description and univariate AUC (descriptive; compared with the scratch probe).
    desc_rows = []
    for name in P5:
        v = table[name].to_numpy(float)
        row = {"feature": name, "valid_windows": int(valid.sum()), "univariate_auc_valid": lm.weighted_auc(y[valid], v[valid])}
        for label in (1, 0):
            q = np.quantile(v[valid & (y == label)], [.25, .5, .75])
            row.update({f"y{label}_q25": float(q[0]), f"y{label}_median": float(q[1]), f"y{label}_q75": float(q[2])})
        desc_rows.append(row)
    for name in ("ratio_of_ratios", "ir_perfusion_index"):
        v = v2[name].to_numpy(float)
        desc_rows.append({"feature": f"v2 {name} (window, CSV)", "valid_windows": int(valid.sum()),
                          "univariate_auc_valid": lm.weighted_auc(y[valid], v[valid])})
    write_csv(pd.DataFrame(desc_rows), out, "s1_feature_description.csv")
    corr = table.loc[valid, list(P5)].corr(method="spearman")
    corr.insert(0, "feature", corr.index)
    write_csv(corr.reset_index(drop=True), out, "s1_feature_spearman.csv")
    checks["univariate_auc_valid"] = {r["feature"]: r["univariate_auc_valid"] for r in desc_rows}
    checks["detector"] = {"min_peak_distance_samples": pf.MIN_PEAK_DISTANCE, "prominence_std": pf.PROMINENCE_STD,
                          "min_beats": pf.MIN_BEATS, "polarity_percentile": pf.POLARITY_PERCENTILE,
                          "table_float_format": "%.7g (las.write_gzip_csv); stages 2-4 read the written table"}
    checks["seconds"] = round(time.perf_counter() - started, 1)
    write_json(checks, out, "s1_checks.json")
    print(json.dumps({k: checks[k] for k in ("validity", "hr_within_40_180_share_of_valid", "spearman_hr_vs_60x_dominant_frequency",
                                              "spearman_r_beat_vs_ratio_of_ratios", "polarity", "sqi_vs_artefact_rows",
                                              "univariate_auc_valid", "row_alignment")}, indent=1, default=las._json_default))
    return 0


# ----------------------------------------------------------------------------- jobs (one fresh process each)

RECIPE = {**las.V2_RECIPE, "spline_range": list(las.V2_RECIPE["spline_range"])}   # hidden 12, 8 knots, [-3, 3], 100 epochs
N_BOOT = las.N_BOOT
BOOT_SEED_REC = las.BOOTSTRAP_SEED + 1     # section C convention: index 1 = v2 recording-grouped
BOOT_SEED_PAT = las.BOOTSTRAP_SEED + 2     # index 2 = v2 patient-grouped
SEEDS = las.SEEDS                          # 42, 1, 2, 3, 4, 5


def impute(x, train):
    """NaN -> mean of the training rows' finite values, per column (0 after a scaler fitted on those rows)."""
    x = np.array(x, copy=True)
    means = np.full(x.shape[1], np.nan)
    for c in range(x.shape[1]):
        missing = ~np.isfinite(x[:, c])
        if missing.any():
            means[c] = x[train & ~missing, c].astype(np.float64).mean()
            x[missing, c] = means[c]
    return x, means


def train_kan_regressor(x_scaled, t_scaled, hidden, knots, spline_range, epochs, seed):
    """lamaakal_models.train_tinykan with a linear output and MSE (no sigmoid, no pos_weight)."""
    import torch
    import torch.nn as nn
    from train_tinykan_v2 import KANBinaryClassifier

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = KANBinaryClassifier((x_scaled.shape[1], hidden, 1), knots, 3, tuple(spline_range))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    t_t = torch.tensor(t_scaled, dtype=torch.float32).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), t_t) + 1e-5 * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    model.eval()
    return model


def kan_params(model):
    state = model.state_dict()
    return {"w1": state["layers.0.W"].numpy(), "b1": state["layers.0.b"].numpy(), "theta1": state["layers.0.theta"].numpy(),
            "centers1": model.layers[0].spline.centers.numpy(), "step1": np.float32(model.layers[0].spline.step.item()),
            "w2": state["layers.1.W"].numpy(), "b2": state["layers.1.b"].numpy(), "theta2": state["layers.1.theta"].numpy(),
            "centers2": model.layers[1].spline.centers.numpy(), "step2": np.float32(model.layers[1].spline.step.item()),
            "parameter_count": np.int64(sum(p.numel() for p in model.parameters()))}


def _cv_job(spec, x, y, folds):
    """lamaakal_models.run_cv_job with per-fold imputation of beat-invalid rows before the fold scaler."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    guard_inputs(spec["features"])
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    probabilities = np.full(len(y), np.nan)
    sizes = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        xi, _ = impute(x, train)
        scaler = StandardScaler()
        if spec["learner"] == "tinykan":
            x_train = scaler.fit_transform(xi[train]).astype(np.float32)
            x_test = scaler.transform(xi[test]).astype(np.float32)
            p, size = models._tinykan_fold(x_train, y[train], x_test, spec)
        else:
            x_train = scaler.fit_transform(xi[train].astype(np.float64))
            x_test = scaler.transform(xi[test].astype(np.float64))
            p, size = models._baseline_fold(x_train, y[train], x_test, spec["learner"])
        probabilities[test] = p
        sizes.append(size)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{spec['name']}: incomplete out-of-fold coverage")
    return {"probabilities": probabilities, "fold_sizes": np.array(sizes, dtype=float)}


def _repro_job(spec):
    import train_tinykan_v2 as tk2

    guard_inputs(tk2.V2_FEATURES)
    frame = pd.read_csv(tk2.V2_CSV)
    frame["recording_id"] = frame["recording_id"].astype(str)
    p, folds = tk2.out_of_fold(frame, tk2.V2_FEATURES, "recording_id", "tinykan")
    return {"probabilities": p, "folds": folds}


def _final_job(spec, x, y):
    """P5 all-window analysis model: impute with the all-window beat-valid mean, scaler on all rows, seed 42."""
    import torch
    from sklearn.preprocessing import StandardScaler

    guard_inputs(spec["features"])
    xi, means = impute(x, np.ones(len(y), bool))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(xi).astype(np.float32)
    model = models.train_tinykan(x_scaled, y, spec["hidden"], spec["knots"], tuple(spec["spline_range"]), spec["epochs"], spec["seed"])
    with torch.no_grad():
        logit = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    return {"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_, "imputation_mean": means, "torch_logit": logit,
            **kan_params(model)}


def _kanreg_job(spec, x, t, train):
    """Concept regressor: fit on rows where train is True, predict the other rows (in order)."""
    import torch
    from sklearn.preprocessing import StandardScaler

    guard_inputs(spec["features"])
    xi, _ = impute(x, train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(xi[train]).astype(np.float32)
    x_test = scaler.transform(xi[~train]).astype(np.float32)
    t_mean, t_std = float(t[train].mean()), float(t[train].std())
    model = train_kan_regressor(x_train, ((t[train] - t_mean) / t_std).astype(np.float32), spec["hidden"], spec["knots"],
                                spec["spline_range"], spec["epochs"], spec["seed"])
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32)).numpy().ravel().astype(np.float64)
        fit = model(torch.tensor(x_train, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    return {"prediction": pred * t_std + t_mean, "train_rmse": np.float64(np.sqrt(np.mean((fit * t_std + t_mean - t[train]) ** 2))),
            "parameter_count": np.int64(sum(p.numel() for p in model.parameters()))}


JOB_KINDS = {"cv": _cv_job, "repro": _repro_job, "final": _final_job, "kanreg": _kanreg_job}


def run_job(payload):
    kind, spec, arrays = payload
    started = time.perf_counter()
    result = JOB_KINDS[kind](spec, *arrays)
    result["seconds"] = np.float64(time.perf_counter() - started)
    return spec["name"], result


def job_key(kind, spec, arrays):
    digest = hashlib.sha256(json.dumps({"cache_version": CACHE_VERSION, "kind": kind, "spec": spec}, sort_keys=True).encode())
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_jobs_cached(payloads, workers, cache_dir):
    """Run each job in a freshly spawned process (max_tasks_per_child=1, as las.run_jobs); cache finished jobs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, pending = {}, []
    for kind, spec, arrays in payloads:
        key = job_key(kind, spec, arrays)
        path = cache_dir / f"{spec['name']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["_key"]) == key:
                    results[spec["name"]] = {k: z[k] for k in z.files if k != "_key"}
                    continue
        pending.append((kind, spec, arrays, key, path))
    if pending:
        weight = {"repro": 9, "final": 5, "kanreg": 4}
        pending.sort(key=lambda job: -(weight.get(job[0], 0) + (3 if job[1].get("learner") == "tinykan" else 0)))
        print(f"  running {len(pending)} job(s) on {min(workers, len(pending))} worker(s); {len(results)} cached", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending))), max_tasks_per_child=1) as pool:
            futures = {pool.submit(run_job, (kind, spec, arrays)): (key, path) for kind, spec, arrays, key, path in pending}
            for future in as_completed(futures):
                key, path = futures[future]
                name, result = future.result()
                np.savez(path, _key=np.array(key), **{k: np.asarray(v) for k, v in result.items()})
                results[name] = result
                print(f"  {name}: {float(result['seconds']):.1f} s", flush=True)
    return results


# ----------------------------------------------------------------------------- stage 2

def feature_matrix(v2, table, features):
    guard_inputs(features)
    columns = [(table[c] if c in P5 else v2[c]).to_numpy(np.float64) for c in features]
    return np.column_stack(columns).astype(np.float32)


def class_columns(y, p):
    row = {}
    for label, tag in ((1, "pos"), (0, "neg")):
        q = np.quantile(p[y == label], [.25, .5, .75])
        row.update({f"p_{tag}_q25": float(q[0]), f"p_{tag}_median": float(q[1]), f"p_{tag}_q75": float(q[2]),
                    f"share_p_ge_0_50_{tag}": float(np.mean(p[y == label] >= THRESHOLD))})
    return row


def metric_row(meta, y, p, mask, subset, threshold=THRESHOLD):
    m = lm.binary_metrics(y[mask], p[mask], threshold)
    row = {**meta, "subset": subset, "n_windows": m["n"], "prevalence": m["prevalence"], "auc": m["auc"], "pr_auc": m["pr_auc"],
           "recall": m["recall"], "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
           "candidate_rate": m["candidate_rate"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}
    if threshold == THRESHOLD:
        row.update(class_columns(y[mask], p[mask]))
    return row


def paired_rows(pairs, point, draws, grouping, seed):
    rows = []
    for a, b in pairs:
        delta = draws[a]["auc"] - draws[b]["auc"]
        low, high = lm.percentile_ci(delta)
        rows.append({"grouping": grouping, "comparison": f"{a} - {b}", "metric": "auc",
                     "difference": point[a]["auc"] - point[b]["auc"], "ci_low": low, "ci_high": high,
                     "fraction_resamples_le_0": float(np.mean(delta <= 0)), "n_boot": N_BOOT, "seed": seed,
                     "method": f"paired {grouping} bootstrap, 95% percentile CI (lamaakal_metrics.group_bootstrap)"})
    return rows


def stage2(out, args):
    v2, folds_rec, folds_pat, oof, _ = load_v2()
    table = load_physio(out)
    y = v2["label"].to_numpy(np.int64)
    valid = table["beat_valid"].to_numpy(bool)
    ref = oof["v2_tinykan16_oof"]["y_prob"].to_numpy(float)

    # 2a: reproduction check, alone and first.
    print("Stage 2a: v2-16 TinyKAN seed 42 through train_tinykan_v2.out_of_fold ...", flush=True)
    repro = run_jobs_cached([("repro", {"name": "repro_v2_16_tinykan_seed42", "features": list(V16)}, ())],
                            1, args.cache)["repro_v2_16_tinykan_seed42"]
    p42 = repro["probabilities"]
    diff = np.abs(p42 - ref)
    check = {
        "reference": "outputs/openox-v2/predictions_b_v2_tinykan16.csv", "path": "train_tinykan_v2.out_of_fold(v2, V2_FEATURES, 'recording_id', 'tinykan') in a fresh process",
        "folds_regenerated_equal_reference_fold_column": bool(np.array_equal(repro["folds"].astype(int), folds_rec)),
        "reference_auc": lm.weighted_auc(y, ref), "retrain_auc": lm.weighted_auc(y, p42),
        "max_abs_probability_diff": float(diff.max()),
        "max_abs_probability_diff_per_fold": {int(f): float(diff[folds_rec == f].max()) for f in range(1, 6)},
        "bitwise_identical": bool(np.array_equal(p42, ref)),
        "candidate_flag_disagreements_at_0_50": int(np.sum((p42 >= THRESHOLD) != (ref >= THRESHOLD))),
        "tolerance_auc": 0.002, "seconds": float(repro["seconds"]),
    }
    check["auc_difference"] = check["retrain_auc"] - check["reference_auc"]
    check["reproduces"] = bool(check["folds_regenerated_equal_reference_fold_column"] and abs(check["auc_difference"]) <= 0.002)
    out_file = out / "s2_reproduction.json"
    if out_file.exists():
        if read_json(out, "s2_reproduction.json") != json.loads(json.dumps(check, default=las._json_default)):
            raise StopAnalysis("s2_reproduction.json exists and differs from this run's check")
    else:
        write_json(check, out, "s2_reproduction.json")
    print(json.dumps(check, indent=1), flush=True)
    if not check["reproduces"]:
        raise StopAnalysis("v2-16 seed-42 reproduction failed; Stage 2 stopped (see s2_reproduction.json)")
    if args.stage_part == "2a":
        return 0

    # 2b: models.
    payloads, registry = [], []

    def add(fs, learner, grouping, seed=None):
        name = f"{fs}_{learner}_{grouping}" + (f"_seed{seed}" if learner == "tinykan" else "")
        spec = {"name": name, "feature_set": fs, "features": list(FEATURE_SETS[fs]), "learner": learner, "grouping": grouping}
        if learner == "tinykan":
            spec.update({**RECIPE, "seed": seed})
        payloads.append(("cv", spec, (feature_matrix(v2, table, FEATURE_SETS[fs]), y, folds_rec if grouping == "recording" else folds_pat)))
        registry.append({"key": name, "feature_set": fs, "learner": learner, "grouping": grouping, "seed": seed,
                         "source": "trained here (analysis model)"})

    for fs in ("P5", "P3", "V16+P5"):
        for learner in ("tinykan", "logreg", "hist_gb"):
            add(fs, learner, "recording", 42 if learner == "tinykan" else None)
    for seed in SEEDS[1:]:
        add("P5", "tinykan", "recording", seed)
    add("v2-16", "hist_gb", "recording")
    for learner in ("tinykan", "logreg", "hist_gb"):
        add("P5", learner, "patient", 42 if learner == "tinykan" else None)
    add("v2-16", "logreg", "patient")
    add("v2-16", "hist_gb", "patient")
    print(f"Stage 2b: {len(payloads)} model jobs ...", flush=True)
    results = run_jobs_cached(payloads, args.jobs, args.cache)

    probs = {r["key"]: results[r["key"]]["probabilities"] for r in registry}
    sizes = {r["key"]: float(np.mean(results[r["key"]]["fold_sizes"])) for r in registry}
    existing = [("v2-16_tinykan_recording_seed42", "tinykan", "recording", 42, ref, "existing OOF predictions_b_v2_tinykan16.csv"),
                ("v2-16_logreg_recording", "logreg", "recording", None, oof["v2_logreg16_oof"]["y_prob"].to_numpy(float),
                 "existing OOF predictions_c_v2_logreg16.csv"),
                ("v2-16_tinykan_patient_seed42", "tinykan", "patient", 42, oof["v2_tinykan16_patient_oof"]["y_prob"].to_numpy(float),
                 "existing OOF predictions_b_v2_tinykan16_patient.csv")]
    for key, learner, grouping, seed, p, source in existing:
        registry.append({"key": key, "feature_set": "v2-16", "learner": learner, "grouping": grouping, "seed": seed, "source": source})
        probs[key] = p
        sizes[key] = models.tinykan_param_count(16, 12, 8) if learner == "tinykan" else 17.0

    boot = {}
    for grouping, groups, seed in (("recording", v2["recording_id"].to_numpy(), BOOT_SEED_REC),
                                   ("patient", v2["patient_id"].to_numpy(), BOOT_SEED_PAT)):
        keys = [r["key"] for r in registry if r["grouping"] == grouping]
        boot[grouping] = lm.group_bootstrap(y, {k: probs[k] for k in keys}, groups, N_BOOT, seed, THRESHOLD)

    rows = []
    for r in registry:
        n_in = len(FEATURE_SETS[r["feature_set"]])
        size_unit = {"tinykan": "trainable parameters", "logreg": "trainable parameters", "hist_gb": "tree nodes, fold mean"}[r["learner"]]
        meta = {**{k: r[k] for k in ("feature_set", "learner", "grouping", "seed", "source")}, "model_key": r["key"], "n_inputs": n_in,
                "size": models.tinykan_param_count(n_in, 12, 8) if r["learner"] == "tinykan" else sizes[r["key"]],
                "size_unit": size_unit, "forwards_per_explained_decision": n_in + 1}
        point, draws = boot[r["grouping"]]
        low, high = lm.percentile_ci(draws[r["key"]]["auc"])
        rows.append({**metric_row(meta, y, probs[r["key"]], np.ones(N_ROWS, bool), "all"), "auc_ci_low": low, "auc_ci_high": high})
        rows.append(metric_row(meta, y, probs[r["key"]], valid, "beat-valid"))
    metrics_table = pd.DataFrame(rows)

    pairs_rec = [("P5_tinykan_recording_seed42", "v2-16_tinykan_recording_seed42"),
                 ("P3_tinykan_recording_seed42", "P5_tinykan_recording_seed42"),
                 ("V16+P5_tinykan_recording_seed42", "v2-16_tinykan_recording_seed42"),
                 ("P5_logreg_recording", "v2-16_logreg_recording"),
                 ("P5_hist_gb_recording", "v2-16_hist_gb_recording"),
                 ("V16+P5_hist_gb_recording", "v2-16_hist_gb_recording"),
                 ("P5_tinykan_recording_seed42", "P5_logreg_recording"),
                 ("P5_tinykan_recording_seed42", "P5_hist_gb_recording")]
    pairs_pat = [("P5_tinykan_patient_seed42", "v2-16_tinykan_patient_seed42"),
                 ("P5_logreg_patient", "v2-16_logreg_patient"),
                 ("P5_hist_gb_patient", "v2-16_hist_gb_patient"),
                 ("P5_tinykan_patient_seed42", "P5_logreg_patient")]
    paired = pd.DataFrame(paired_rows(pairs_rec, *boot["recording"], "recording", BOOT_SEED_REC)
                          + paired_rows(pairs_pat, *boot["patient"], "patient", BOOT_SEED_PAT))

    seed_rows = []
    for seed in SEEDS:
        key = f"P5_tinykan_recording_seed{seed}"
        m = lm.binary_metrics(y, probs[key], THRESHOLD)
        seed_rows.append({"feature_set": "P5", "grouping": "recording", "seed": seed, "auc": m["auc"], "pr_auc": m["pr_auc"],
                          "recall": m["recall"], "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
                          "candidate_rate": m["candidate_rate"]})
    seeds = pd.DataFrame(seed_rows)
    seed_summary = {metric: dict(zip(("mean", "sd"), lm.mean_sd(seeds[metric])))
                    for metric in ("auc", "pr_auc", "recall", "specificity", "balanced_accuracy", "candidate_rate")}

    # Cross-check: same bootstrap seed and groups as analyze_lamaakal_style section C -> same v2-16 TinyKAN CI.
    las_ci = pd.read_csv(P["las_bootstrap_ci"])
    ci_check = {}
    for grouping, key in (("recording", "v2-16_tinykan_recording_seed42"), ("patient", "v2-16_tinykan_patient_seed42")):
        row = las_ci[(las_ci.version == "v2") & (las_ci.grouping == grouping) & (las_ci.model == "TinyKAN") & (las_ci.metric == "auc")].iloc[0]
        mine = lm.percentile_ci(boot[grouping][1][key]["auc"])
        ci_check[grouping] = {"las_section_c": [float(row.ci_low), float(row.ci_high)], "this_run": list(mine),
                              "max_abs_diff": float(max(abs(mine[0] - row.ci_low), abs(mine[1] - row.ci_high)))}

    primary = paired.iloc[0]
    reading = ("kept (lower bound above -0.02)" if primary.ci_low > -0.02 else "not kept (lower bound at or below -0.02)")
    if primary.ci_low > 0:
        reading += "; higher on this table, recording-grouped"
    pat = paired[paired.comparison == "P5_tinykan_patient_seed42 - v2-16_tinykan_patient_seed42"].iloc[0]
    oof_frame = pd.DataFrame({"row_index": np.arange(N_ROWS), "fold_recording": folds_rec, "fold_patient": folds_pat,
                              "y_true": y, "beat_valid": valid.astype(int),
                              **{k: probs[k] for k in sorted(probs)}})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_table, out, "s2_metrics.csv")
    write_csv(paired, out, "s2_paired_auc_difference.csv")
    write_csv(seeds, out, "s2_p5_tinykan_seeds.csv")
    write_gzip_csv(oof_frame, out, "oof_stage2.csv.gz")
    headline_keys = ["P5_tinykan_recording_seed42", "P3_tinykan_recording_seed42", "V16+P5_tinykan_recording_seed42",
                     "v2-16_tinykan_recording_seed42", "P5_logreg_recording", "P5_hist_gb_recording", "v2-16_logreg_recording",
                     "v2-16_hist_gb_recording", "V16+P5_logreg_recording", "V16+P5_hist_gb_recording", "P3_logreg_recording",
                     "P3_hist_gb_recording", "P5_tinykan_patient_seed42", "v2-16_tinykan_patient_seed42", "P5_logreg_patient",
                     "P5_hist_gb_patient", "v2-16_logreg_patient", "v2-16_hist_gb_patient"]
    allrows = metrics_table[metrics_table.subset == "all"].set_index("model_key")
    validrows = metrics_table[metrics_table.subset == "beat-valid"].set_index("model_key")
    s2 = {
        "reproduction": check,
        "bootstrap_ci_matches_las_section_c": ci_check,
        "headline_all_windows": {k: {c: float(allrows.loc[k, c]) for c in ("auc", "auc_ci_low", "auc_ci_high", "pr_auc", "recall",
                                                                           "specificity", "balanced_accuracy", "candidate_rate", "size")}
                                 for k in headline_keys},
        "headline_beat_valid_auc": {k: float(validrows.loc[k, "auc"]) for k in headline_keys},
        "p5_tinykan_seed_spread": seed_summary,
        "primary_reading": {"comparison": primary.comparison, "difference": float(primary.difference),
                            "ci": [float(primary.ci_low), float(primary.ci_high)], "reading": reading,
                            "rule": "pre-registered: kept if paired recording-bootstrap lower bound > -0.02; 'higher' only if > 0"},
        "patient_reading": {"difference": float(pat.difference), "ci": [float(pat.ci_low), float(pat.ci_high)]},
        "beat_invalid_windows": int((~valid).sum()),
        "job_seconds": {k: float(results[k]["seconds"]) for k in sorted(results)},
    }
    write_json(s2, out, "s2.json")
    print(json.dumps({k: s2[k] for k in ("primary_reading", "patient_reading", "p5_tinykan_seed_spread", "bootstrap_ci_matches_las_section_c")}, indent=1))
    print(metrics_table[metrics_table.subset == "all"][["model_key", "auc", "auc_ci_low", "auc_ci_high", "recall", "specificity", "candidate_rate"]].round(4).to_string(index=False))
    return 0


# ----------------------------------------------------------------------------- stage 3

GLOSS = {**las.GLOSS, "r_beat": "red/IR beat amplitude ratio (R)", "pi_beat": "IR beat perfusion index",
         "hr_bpm": "pulse rate", "sqi_template": "beat-shape consistency", "sqi_ibi_ratio": "beat-interval max/min"}
EXPLANATION_ZERO = 0.0005        # dashboard_core.js EXPLANATION_ZERO (prints as 0.000)
ICE_GRID = np.linspace(-3.0, 3.0, 61)
ICE_TOL = 1e-6
SIGN_CHECKS = (("P5", "r_beat", "primary"), ("P5", "hr_bpm", "secondary"), ("v2", "ratio_of_ratios", "reference"))
WEIGHT_KEYS = {"SCALER_MEAN": "scaler_mean", "SCALER_STD": "scaler_std", "KAN1_W": "w1", "KAN1_B": "b1", "KAN1_THETA": "theta1",
               "KAN1_CENTERS": "centers1", "KAN1_STEP": "step1", "KAN2_W": "w2", "KAN2_B": "b2", "KAN2_THETA": "theta2",
               "KAN2_CENTERS": "centers2", "KAN2_STEP": "step2"}


class ArrayKAN(models.HeaderKAN):
    """lamaakal_models.HeaderKAN built from saved analysis-model arrays instead of a header file.

    Same arithmetic as the header replica: constants rounded to FP32 and evaluated in float64,
    raw inputs rounded to FP32 before standardisation, u clamped to [-3, 3]. No header is written.
    """

    def __init__(self, arrays):  # noqa: D107 - deliberately does not call HeaderKAN.__init__ (no header file)
        self.p = {key: np.asarray(arrays[name], dtype=np.float32).astype(np.float64) for key, name in WEIGHT_KEYS.items()}
        self.n_inputs = self.p["SCALER_MEAN"].shape[0]
        self.hidden = self.p["KAN1_W"].shape[0]
        self.knots = self.p["KAN1_CENTERS"].shape[0]


def perturb_against(model, scaled, reference):
    """delta_i = P(x) - P(x with standardized input i set to reference[i]); reference 0 = the device method."""
    probability = model.predict(scaled)
    attribution = np.empty_like(scaled, dtype=np.float64)
    for feature in range(scaled.shape[1]):
        changed = np.array(scaled, dtype=np.float64, copy=True)
        changed[:, feature] = reference[feature]
        attribution[:, feature] = probability - model.predict(changed)
    return probability, attribution


def global_rows(name, features, attribution, y):
    order = lm.firmware_abs_order(attribution)
    top1 = order[:, 0]
    top3 = np.zeros_like(attribution, dtype=bool)
    top3[np.arange(len(order))[:, None], order[:, :3]] = True
    rows = []
    for f, feature in enumerate(features):
        row = {"version": name, "feature": feature, "gloss": GLOSS[feature], "mean_abs_delta_all": float(np.mean(np.abs(attribution[:, f])))}
        for label in (1, 0):
            mask = y == label
            row[f"mean_abs_delta_y{label}"] = float(np.mean(np.abs(attribution[mask, f])))
            row[f"mean_signed_delta_y{label}"] = float(np.mean(attribution[mask, f]))
            row[f"top1_share_y{label}"] = float(np.mean(top1[mask] == f))
            row[f"top3_share_y{label}"] = float(np.mean(top3[mask, f]))
        rows.append(row)
    return rows


def tvd(block):
    return float(0.5 * np.sum(np.abs(block["top1_share_y1"].to_numpy() - block["top1_share_y0"].to_numpy())))


def degeneracy_rows(name, reference_label, model, scaled, probability, attribution, y, n_features):
    """d_explanation_degeneracy.csv columns (same code as analyze_lamaakal_style section D) plus the dashboard withhold rule."""
    candidate = probability >= THRESHOLD
    nonzero = np.sum(np.abs(attribution) > las.ZERO_DELTA, axis=1)
    lead = attribution[np.arange(len(attribution)), lm.firmware_abs_order(attribution)[:, 0]]
    clamped = np.abs(model.output_preactivation(scaled)) > 3.0
    opposes = np.where(candidate, lead < -las.ZERO_DELTA, lead > las.ZERO_DELTA)
    zero = ~(np.abs(lead) >= EXPLANATION_ZERO)
    gate_opposes = ~zero & np.where(candidate, lead < 0, lead > 0)
    subsets = {"all": np.ones(len(y), bool), "candidate (P >= 0.50)": candidate, "non-candidate (P < 0.50)": ~candidate,
               "TP (in-sample)": (y == 1) & candidate, "FN (in-sample)": (y == 1) & ~candidate,
               "FP (in-sample)": (y == 0) & candidate, "TN (in-sample)": (y == 0) & ~candidate}
    rows = []
    for subset, mask in subsets.items():
        rows.append({"version": name, "reference": reference_label, "subset": subset, "n": int(mask.sum()), "n_features": n_features,
                     "fraction_all_deltas_zero": float(np.mean(nonzero[mask] == 0)),
                     "fraction_top3_has_zero_delta_slot": float(np.mean(nonzero[mask] < 3)),
                     "median_nonzero_deltas": float(np.median(nonzero[mask])),
                     "fraction_output_preactivation_clamped": float(np.mean(clamped[mask])),
                     "fraction_top1_opposes_decision": float(np.mean(opposes[mask])),
                     "dashboard_withheld": float(np.mean((zero | gate_opposes)[mask])),
                     "dashboard_withheld_zero": float(np.mean(zero[mask])),
                     "dashboard_withheld_opposes": float(np.mean(gate_opposes[mask]))})
    return rows


def sign_rows(name, feature, role, reference_label, z, delta, reference, y, candidate):
    eligible = np.abs(delta) > EXPLANATION_ZERO
    agree = np.sign(delta) == np.sign(z - reference)
    subsets = {"all": np.ones(len(y), bool), "label_1": y == 1, "label_0": y == 0,
               "candidate": candidate, "non-candidate": ~candidate}
    rows = []
    for subset, mask in subsets.items():
        e = mask & eligible
        rows.append({"version": name, "feature": feature, "role": role, "reference": reference_label, "subset": subset,
                     "windows": int(mask.sum()), "eligible_abs_delta_gt_0_0005": int(e.sum()),
                     "share_eligible": float(e.sum() / max(mask.sum(), 1)),
                     "sign_agreement_share": float(np.mean(agree[e])) if e.any() else float("nan"),
                     "expected": "+ (above-reference value raises the score)"})
    return rows


def ice_curves(model, scaled, feature):
    curves = np.empty((len(scaled), len(ICE_GRID)))
    for g, value in enumerate(ICE_GRID):
        changed = np.array(scaled, dtype=np.float64, copy=True)
        changed[:, feature] = value
        curves[:, g] = model.predict(changed)
    return curves


def ice_rows(name, feature, role, curves, y, candidate):
    steps = np.diff(curves, axis=1)
    monotone = np.all(steps >= -ICE_TOL, axis=1)
    decreasing = np.all(steps <= ICE_TOL, axis=1)
    span = curves.max(axis=1) - curves.min(axis=1)
    max_drop = np.maximum(0.0, -steps.min(axis=1))
    z_at_max = ICE_GRID[np.argmax(curves, axis=1)]
    subsets = {"all": np.ones(len(y), bool), "label_1": y == 1, "label_0": y == 0, "candidate": candidate, "non-candidate": ~candidate}
    return [{"version": name, "feature": feature, "role": role, "subset": subset, "windows": int(mask.sum()),
             "share_non_decreasing": float(np.mean(monotone[mask])), "share_non_increasing": float(np.mean(decreasing[mask])),
             "share_flat_span_lt_0_0005": float(np.mean(span[mask] < EXPLANATION_ZERO)),
             "median_span": float(np.median(span[mask])), "median_p_at_z_minus3": float(np.median(curves[mask, 0])),
             "median_p_at_z_plus3": float(np.median(curves[mask, -1])), "grid": "z = -3..3, 61 points", "tolerance": ICE_TOL,
             # Post hoc, descriptive only (added after seeing share_non_decreasing; the pre-registered reading is unchanged).
             "posthoc_median_largest_single_step_drop": float(np.median(max_drop[mask])),
             "posthoc_q90_largest_single_step_drop": float(np.quantile(max_drop[mask], 0.9)),
             "posthoc_share_largest_drop_lt_0_05": float(np.mean(max_drop[mask] < 0.05)),
             "posthoc_share_p_at_plus3_ge_p_at_minus3": float(np.mean(curves[mask, -1] >= curves[mask, 0])),
             "posthoc_median_z_at_curve_max": float(np.median(z_at_max[mask]))}
            for subset, mask in subsets.items()]


def output_spline_shape(name, model):
    """As analyze_lamaakal_style section D d_output_spline_shape.csv: where P peaks inside [-3, 3] vs the clamp edges."""
    grid = np.linspace(-3.0, 3.0, 6001)
    p = 1.0 / (1.0 + np.exp(-model.phi(grid, 2, 0)))
    return {"version": name, "u_at_max_p": float(grid[np.argmax(p)]), "max_p": float(p.max()), "p_at_u_plus3_clamp": float(p[-1]),
            "u_at_min_p": float(grid[np.argmin(p)]), "min_p": float(p.min()), "p_at_u_minus3_clamp": float(p[0]),
            "overshoot_above_clamp_value": float(p.max() - p[-1])}


def reading(share):
    return "consistent" if share >= 0.90 else ("mixed" if share >= 0.50 else "contradicts")


def aopc_table(name, result, source):
    rows = []
    for subset, block in result.items():
        for ranking, values in block.items():
            rows.append({"source": source, "version": name, "subset": subset, "windows": values["windows"], "ranking": ranking,
                         "deletion_aopc_window_mean": values["deletion_aopc"], "insertion_aopc_window_mean": values["insertion_aopc"],
                         "deletion_gain_vs_random_mean": values.get("deletion_gain", float("nan")),
                         "fraction_better_than_own_random_mean": values.get("fraction_better", float("nan"))})
    return pd.DataFrame(rows)


def plot_top1(plt, table, path):
    fig, axes = plt.subplots(1, 2, figsize=(las.COL2, 3.0), layout="constrained",
                             gridspec_kw={"width_ratios": [5, 16]})
    for ax, name, title in ((axes[0], "P5", "P5 TinyKAN (analysis model)"), (axes[1], "v2", "v2 shipped header (16 inputs)")):
        block = table[table.version == name].reset_index(drop=True)
        pos = np.arange(len(block))
        ax.barh(pos - 0.2, block["top1_share_y1"], height=0.38, color=las.BLUE, label="label 1 (ScalcO2 < 90)")
        ax.barh(pos + 0.2, block["top1_share_y0"], height=0.38, color=las.ORANGE, hatch="////", edgecolor="white",
                linewidth=0, label="label 0")
        ax.set_yticks(pos, block["feature"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of windows where the input is top-1 (|delta|)")
        ax.set_title(f"{title}\nTVD(label 1, label 0) = {tvd(block):.2f}")
    axes[1].legend(loc="lower right")
    las.save_figure(fig, path)


def plot_ice(plt, panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(las.COL2, 2.6), layout="constrained", sharey=True)
    rng = np.random.default_rng(20261030)
    for ax, (title, curves, y) in zip(axes, panels):
        for label, color in ((1, las.BLUE), (0, las.ORANGE)):
            rows = np.flatnonzero(y == label)
            pick = rng.choice(rows, size=min(60, len(rows)), replace=False)
            for i in pick:
                ax.plot(ICE_GRID, curves[i], color=color, linewidth=0.3, alpha=0.35)
            ax.plot(ICE_GRID, np.median(curves[rows], axis=0), color=color, linewidth=1.6,
                    linestyle="-" if label == 1 else "--", label=f"label {label}: median curve")
        ax.axhline(THRESHOLD, color=las.INK2, linewidth=0.6, linestyle=":")
        ax.axvline(0.0, color=las.MUTED, linewidth=0.6)
        ax.set_xlabel("Standardized input z (others fixed)")
        ax.set_title(title)
    axes[0].set_ylabel("P (risk)")
    axes[0].legend(loc="upper left")
    las.save_figure(fig, path)


def stage3(out, args):
    v2, _, _, _, _ = load_v2()
    table = load_physio(out)
    y = v2["label"].to_numpy(np.int64)
    valid = table["beat_valid"].to_numpy(bool)
    checks, head = {}, {}

    # Same-code check first: the shipped v2 header through this code path.
    print("Stage 3: AOPC same-code check on the shipped v2 header ...", flush=True)
    import train_tinykan_v2 as tk2
    v2_weights = tk2.parse_header(P["v2_header"])           # parsed with the brief's loader as well, for a parity check
    model_v2 = models.HeaderKAN(P["v2_header"])
    raw_v2 = v2.loc[:, V16].to_numpy(np.float32)
    scaled_v2 = model_v2.standardize(raw_v2)
    prob_v2, attr_v2 = model_v2.mean_perturbation(scaled_v2)
    checks["v2_headerkan_vs_header_probabilities_max_abs"] = float(np.max(np.abs(tk2.header_probabilities(v2_weights, raw_v2) - prob_v2)))
    aopc_v2 = las.aopc_run(model_v2, scaled_v2, prob_v2, attr_v2, y)
    mine = aopc_table("v2", aopc_v2, "")
    ref = pd.read_csv(P["las_aopc_v2"])
    merged = ref.merge(mine, on=["subset", "ranking"], suffixes=("_ref", "_new"))
    cols = ["deletion_aopc_window_mean", "insertion_aopc_window_mean", "deletion_gain_vs_random_mean", "fraction_better_than_own_random_mean"]
    diffs = {c: float(np.nanmax(np.abs(merged[f"{c}_ref"] - merged[f"{c}_new"]))) for c in cols}
    same = {"reference": "outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv", "cells": int(len(merged)),
            "max_abs_diff": diffs, "tolerance": 1e-8,
            "firmware_absolute_all": float(mine[(mine.subset == "all") & (mine.ranking == "firmware_absolute")].deletion_aopc_window_mean.iloc[0]),
            "random_mean_all": float(mine[(mine.subset == "all") & (mine.ranking == "random_mean")].deletion_aopc_window_mean.iloc[0])}
    same["reproduces"] = bool(len(merged) == len(ref) and max(diffs.values()) <= 1e-8)
    checks["aopc_same_code_v2"] = same
    print(json.dumps(same, indent=1), flush=True)
    if not same["reproduces"]:
        out.mkdir(parents=True, exist_ok=True)
        write_json(checks, out, "s3_stopped.json")
        raise StopAnalysis("v2 AOPC same-code check failed; Stage 3 stopped")
    deg_ref = pd.read_csv(P["las_degeneracy"])
    deg_v2 = pd.DataFrame(degeneracy_rows("v2", "training mean (device)", model_v2, scaled_v2, prob_v2, attr_v2, y, 16))
    m = deg_ref[deg_ref.version == "v2"].merge(deg_v2, on="subset", suffixes=("_ref", "_new"))
    checks["degeneracy_same_code_v2_max_abs_diff"] = float(max(np.max(np.abs(m[f"{c}_ref"] - m[f"{c}_new"])) for c in
                                                               ("fraction_all_deltas_zero", "fraction_top3_has_zero_delta_slot",
                                                                "fraction_output_preactivation_clamped", "fraction_top1_opposes_decision")))

    # P5 all-window analysis model (fresh process, cached).
    print("Stage 3: P5 all-window model (seed 42) ...", flush=True)
    x5 = feature_matrix(v2, table, P5)
    spec = {"name": "P5_tinykan_final_all_windows_seed42", "features": list(P5), **RECIPE, "seed": 42}
    final = run_jobs_cached([("final", spec, (x5, y))], 1, args.cache)[spec["name"]]
    model_p5 = ArrayKAN(final)
    # Beat-invalid inputs take the exported (FP32) training mean, i.e. exactly 0 after standardization. Training
    # imputed with the float64 beat-valid mean; the two differ by FP32 rounding only (imputation_vs_scaler_mean_max_abs).
    raw_p5 = np.where(np.isfinite(x5), x5, model_p5.p["SCALER_MEAN"].astype(np.float32)[None, :]).astype(np.float32)
    checks["imputation_vs_scaler_mean_max_abs"] = float(np.max(np.abs(final["imputation_mean"] - final["scaler_mean"])))
    scaled_p5 = model_p5.standardize(raw_p5)
    if not np.all(scaled_p5[~valid] == 0.0):
        raise StopAnalysis("Imputed P5 rows do not standardize to exactly 0")
    prob_p5, attr_p5 = model_p5.mean_perturbation(scaled_p5)
    torch_p = 1.0 / (1.0 + np.exp(-final["torch_logit"]))
    checks["p5_replica_vs_torch_max_abs_p"] = float(np.max(np.abs(prob_p5 - torch_p)))
    checks["p5_replica_vs_torch_flag_disagreements"] = int(np.sum((prob_p5 >= THRESHOLD) != (torch_p >= THRESHOLD)))
    m5 = lm.binary_metrics(y, prob_p5, THRESHOLD)
    checks["p5_final_resubstitution_at_0_50"] = {k: m5[k] for k in ("auc", "recall", "specificity", "candidate_rate", "tn", "fp", "fn", "tp")}
    m2 = lm.binary_metrics(y, prob_v2, THRESHOLD)
    checks["v2_header_resubstitution_at_0_50"] = {k: m2[k] for k in ("auc", "recall", "specificity", "candidate_rate", "tn", "fp", "fn", "tp")}

    out.mkdir(parents=True, exist_ok=True)
    np.savez(new_file(out, "p5_final_model.npz"), features=np.array(P5), **{k: np.asarray(final[k]) for k in
             ("scaler_mean", "scaler_std", "imputation_mean", "w1", "b1", "theta1", "centers1", "step1", "w2", "b2", "theta2",
              "centers2", "step2")})
    write_json({"model": "P5 TinyKAN all-window analysis model; not a header, never replaces kan_model_weights*.h",
                "features_in_order": list(P5), "recipe": {**RECIPE, "seed": 42, "lr": 0.01, "weight_decay": 1e-4,
                                                          "loss": "pos_weight BCE + 1e-5 smoothness, grad clip 10"},
                "training_rows": N_ROWS, "imputation": "beat-invalid rows -> all-window beat-valid mean (imputation_mean), 0 after scaling",
                "imputation_mean": final["imputation_mean"].tolist(), "scaler_mean": final["scaler_mean"].tolist(),
                "scaler_std": final["scaler_std"].tolist(), "parameter_count": int(final["parameter_count"]),
                "replica": "ArrayKAN = lamaakal_models.HeaderKAN arithmetic on these arrays (FP32 constants in float64)",
                "replica_vs_torch_max_abs_p": checks["p5_replica_vs_torch_max_abs_p"],
                "resubstitution_at_0_50": checks["p5_final_resubstitution_at_0_50"],
                "resubstitution_note": "in-sample on the training rows; not a generalisation estimate (see s2_metrics.csv)"},
               out, "p5_final_model.json")

    # 1. AOPC (P5 new run; v2 recomputed above, identical to the cited CSV).
    print("Stage 3: AOPC, class separation, degeneracy, sign agreement, ICE ...", flush=True)
    aopc_p5 = las.aopc_run(model_p5, scaled_p5, prob_p5, attr_p5, y)
    aopc = pd.concat([aopc_table("P5", aopc_p5, "new run here: P5 all-window analysis model, in-sample, seed 20260911, 5 random repeats"),
                      aopc_table("v2", aopc_v2, "shipped v2 header through this code path (equal to d_aopc_v2_new_run.csv within 1e-8)")])
    aopc = aopc[aopc.ranking.isin(["firmware_absolute", "random_mean", "signed_risk"])].reset_index(drop=True)
    write_csv(las.add_predicted_class_column(aopc), out, "s3_aopc.csv")

    def cell(name, subset, ranking, column="deletion_aopc_window_mean"):
        return float(aopc[(aopc.version == name) & (aopc.subset == subset) & (aopc.ranking == ranking)][column].iloc[0])

    for name in ("P5", "v2"):
        head[f"{name}_aopc"] = {f"{subset}_{ranking}": cell(name, subset, ranking)
                                for subset in ("all", "candidate_p_ge_0_50", "noncandidate_p_lt_0_50", "label_1", "label_0")
                                for ranking in ("firmware_absolute", "random_mean", "signed_risk")}
        head[f"{name}_aopc"]["fraction_better_all"] = cell(name, "all", "firmware_absolute", "fraction_better_than_own_random_mean")

    # 2. Class separation.
    g = pd.DataFrame(global_rows("P5", P5, attr_p5, y) + global_rows("v2", V16, attr_v2, y))
    write_csv(g, out, "s3_global_attribution.csv")
    v1 = pd.read_csv(P["las_global"])
    v1 = v1[v1.version == "v1"]
    sep = []
    for name, block, n in (("P5", g[g.version == "P5"], 5), ("v2", g[g.version == "v2"], 16), ("v1 (d_global_attribution.csv, read only)", v1, 14)):
        top_y1 = block.loc[block.top1_share_y1.idxmax()]
        top_y0 = block.loc[block.top1_share_y0.idxmax()]
        sep.append({"model": name, "n_features": n, "tvd_top1_label1_vs_label0": tvd(block),
                    "top1_most_common_y1": top_y1.feature, "top1_most_common_y1_share": float(top_y1.top1_share_y1),
                    "top1_most_common_y0": top_y0.feature, "top1_most_common_y0_share": float(top_y0.top1_share_y0),
                    "note": "top-3 covers 3 of n inputs; TVD on the top-1 distribution, not thresholded"})
    write_csv(pd.DataFrame(sep), out, "s3_class_separation.csv")
    head["class_separation"] = sep
    plt = las.setup_style()
    plot_top1(plt, g, out / "fig_top1_share_by_class.png")

    # 3-5. Degeneracy, sign agreement, ICE; reference = training mean (device) and label-0 median (sensitivity).
    explained = {"P5": (model_p5, scaled_p5, prob_p5, attr_p5, P5), "v2": (model_v2, scaled_v2, prob_v2, attr_v2, V16)}
    references = {}
    deg_rows, sign_out, ice_out = [], [], []
    deg_rows += degeneracy_rows("P5", "training mean (device)", model_p5, scaled_p5, prob_p5, attr_p5, y, 5)
    deg_rows += deg_v2.to_dict("records")
    sens = {}
    for name, (model, scaled, prob, attr, features) in explained.items():
        ref0 = np.median(scaled[y == 0], axis=0)
        references[name] = {f: {"training_mean_raw": float(model.p["SCALER_MEAN"][i]),
                                "label0_median_z": float(ref0[i]),
                                "label0_median_raw": float(model.p["SCALER_MEAN"][i] + ref0[i] * model.p["SCALER_STD"][i])}
                            for i, f in enumerate(features)}
        _, attr_ref = perturb_against(model, scaled, ref0)
        sens[name] = (ref0, attr_ref)
        deg_rows += degeneracy_rows(name, "label-0 median (sensitivity, host only)", model, scaled, prob, attr_ref, y, len(features))
    candidate = {"P5": prob_p5 >= THRESHOLD, "v2": prob_v2 >= THRESHOLD}
    curves_for_plot = {}
    for name, feature, role in SIGN_CHECKS:
        model, scaled, prob, attr, features = explained[name]
        f = features.index(feature)
        sign_out += sign_rows(name, feature, role, "training mean (device)", scaled[:, f], attr[:, f], 0.0, y, candidate[name])
        ref0, attr_ref = sens[name]
        sign_out += sign_rows(name, feature, role, "label-0 median (sensitivity, host only)", scaled[:, f], attr_ref[:, f], ref0[f], y,
                              candidate[name])
        curves = ice_curves(model, scaled, f)
        ice_out += ice_rows(name, feature, role, curves, y, candidate[name])
        curves_for_plot[(name, feature)] = curves
    deg = pd.DataFrame(deg_rows)
    signs = pd.DataFrame(sign_out)
    ice = pd.DataFrame(ice_out)
    write_csv(deg, out, "s3_degeneracy.csv")
    write_csv(signs, out, "s3_sign_agreement.csv")
    write_csv(ice, out, "s3_ice_monotonicity.csv")
    shapes = pd.DataFrame([output_spline_shape("P5", model_p5), output_spline_shape("v2", model_v2)])
    write_csv(shapes, out, "s3_output_spline_shape.csv")
    head["output_spline_shape"] = shapes.to_dict("records")
    plot_ice(plt, [("P5 r_beat (primary)", curves_for_plot[("P5", "r_beat")], y),
                   ("P5 hr_bpm (secondary)", curves_for_plot[("P5", "hr_bpm")], y),
                   ("v2 ratio_of_ratios (reference)", curves_for_plot[("v2", "ratio_of_ratios")], y)],
             out / "fig_ice_r_beat.png")

    # Per-window outputs (P5 deltas both references; v2 lead feature).
    order5 = lm.firmware_abs_order(attr_p5)
    order2 = lm.firmware_abs_order(attr_v2)
    per = {"row_index": np.arange(N_ROWS), "y_true": y, "beat_valid": valid.astype(int), "p5_probability": prob_p5}
    per.update({f"p5_delta_{f}": attr_p5[:, i] for i, f in enumerate(P5)})
    per.update({f"p5_delta_label0ref_{f}": sens["P5"][1][:, i] for i, f in enumerate(P5)})
    per.update({"p5_top1": np.array(P5)[order5[:, 0]], "v2_probability": prob_v2, "v2_top1": np.array(V16)[order2[:, 0]],
                "v2_delta_ratio_of_ratios": attr_v2[:, V16.index("ratio_of_ratios")],
                "v2_delta_label0ref_ratio_of_ratios": sens["v2"][1][:, V16.index("ratio_of_ratios")]})
    write_gzip_csv(pd.DataFrame(per), out, "oof_stage3_explanations.csv.gz")

    def sign_share(name, feature, reference_label, subset="all"):
        r = signs[(signs.version == name) & (signs.feature == feature) & (signs.reference == reference_label) & (signs.subset == subset)].iloc[0]
        return {"share": float(r.sign_agreement_share), "eligible": int(r.eligible_abs_delta_gt_0_0005), "windows": int(r.windows),
                "reading": reading(float(r.sign_agreement_share))}

    for name, feature, role in SIGN_CHECKS:
        ice_all = ice[(ice.version == name) & (ice.feature == feature) & (ice.subset == "all")].iloc[0]
        head[f"{name}_{feature}"] = {
            "role": role,
            "sign_agreement_training_mean": {s: sign_share(name, feature, "training mean (device)", s)
                                             for s in ("all", "label_1", "label_0", "candidate", "non-candidate")},
            "sign_agreement_label0_median": {s: sign_share(name, feature, "label-0 median (sensitivity, host only)", s)
                                             for s in ("all", "label_1", "label_0", "candidate", "non-candidate")},
            "ice_share_non_decreasing": float(ice_all.share_non_decreasing),
            "ice_reading": reading(float(ice_all.share_non_decreasing)),
            "ice_share_flat": float(ice_all.share_flat_span_lt_0_0005),
            "ice_posthoc_descriptive": {c.removeprefix("posthoc_"): float(ice_all[c]) for c in ice.columns if c.startswith("posthoc_")},
        }

    def deg_cell(name, reference_label, subset, column):
        return float(deg[(deg.version == name) & (deg.reference == reference_label) & (deg.subset == subset)][column].iloc[0])

    for name in ("P5", "v2"):
        for ref_label, tag in (("training mean (device)", "device"), ("label-0 median (sensitivity, host only)", "label0ref")):
            head[f"{name}_degeneracy_{tag}"] = {
                subset: {c: deg_cell(name, ref_label, subset, c) for c in
                         ("fraction_top3_has_zero_delta_slot", "fraction_output_preactivation_clamped", "fraction_top1_opposes_decision",
                          "dashboard_withheld", "dashboard_withheld_zero", "dashboard_withheld_opposes")}
                for subset in ("all", "candidate (P >= 0.50)", "non-candidate (P < 0.50)")}
    head["references"] = references
    head["beat_invalid_rows_all_deltas_zero_in_p5"] = int(np.sum(np.all(attr_p5[~valid] == 0, axis=1)))
    s3 = {"checks": checks, "headline": head,
          "explainer": "device method: standardized input -> 0 (training mean), delta = P - P_perturbed, firmware-absolute ranking; "
                       "n+1 forwards (P5: 6, v2: 17)",
          "wording": "AOPC is host, in-sample, on the exported model, functionally grounded only; deltas are not effect sizes"}
    write_json(s3, out, "s3.json")
    print(json.dumps({k: head[k] for k in ("P5_aopc", "v2_aopc", "class_separation", "P5_r_beat", "P5_hr_bpm", "v2_ratio_of_ratios")},
                     indent=1, default=las._json_default))
    print(json.dumps(checks, indent=1, default=las._json_default))
    return 0


# ----------------------------------------------------------------------------- stage 4

TARGET_CLIP = (50.0, 100.0)
CONCEPT_CUT = 90.0
GIVEN = ("hr_bpm", "pi_beat", "sqi_template", "sqi_ibi_ratio")
HEAD_INPUTS = ("s_hat",) + GIVEN
HEAD_VARIANTS = {"full": HEAD_INPUTS, "s_hat_only": ("s_hat",), "s_hat+hr_bpm": ("s_hat", "hr_bpm"),
                 "s_hat+pi_beat": ("s_hat", "pi_beat"), "s_hat+sqi_template": ("s_hat", "sqi_template"),
                 "s_hat+sqi_ibi_ratio": ("s_hat", "sqi_ibi_ratio"), "given_only": GIVEN}
# Oracle sanity check only (requested by the brief): the true clipped median_spo2 as a head input. It is the one
# sanctioned exception to the leakage guard, is never part of any other model, and is trivially near-perfect.
ORACLE_INPUT = "oracle_true_scalco2_target"
ORACLE_VARIANTS = {"oracle_full": (ORACLE_INPUT,) + GIVEN, "oracle_only": (ORACLE_INPUT,)}
BOTTLENECK_THRESHOLD = float(np.nextafter(0.0, 1.0))      # score = 90 - s_hat; flag = score > 0 = s_hat < 90
RANGES = (("lt_80", -np.inf, 80.0), ("80_to_90", 80.0, 90.0), ("ge_90", 90.0, np.inf))
CONCEPT_PREDICTORS = ("kan_p5", "ridge_p5", "ridge_v16p5", "constant")


def guard_head(columns, oracle=False):
    guard_inputs(columns)
    if ORACLE_INPUT in columns and not oracle:
        raise AssertionError("The oracle target may only enter the oracle sanity-check head")
    return tuple(columns)


def head_fit(h_train, y_train, h_test):
    """BASELINE_SETTINGS['logreg'] on standardized concepts; returns P, coefficients, intercept, standardized test concepts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    a = scaler.fit_transform(h_train)
    b = scaler.transform(h_test)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=models.BASELINE_SEED)
    model.fit(a, y_train)
    return model.predict_proba(b)[:, 1], model.coef_[0].copy(), float(model.intercept_[0]), b


def ridge_predict(x, t, train):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    xi, _ = impute(x.astype(np.float64), train)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0).fit(scaler.fit_transform(xi[train]), t[train])
    return model.predict(scaler.transform(xi[~train]))


def concept_quality(pred, grouping, subset, s, t):
    d = s - t
    sd = float(np.std(d, ddof=1))
    row = {"concept_predictor": pred, "grouping": grouping, "subset": subset, "n": int(len(t)), "mae": float(np.mean(np.abs(d))),
           "rmse": float(np.sqrt(np.mean(d ** 2))), "bias": float(np.mean(d)), "sd_of_difference": sd,
           "bland_altman_low": float(np.mean(d) - 1.959964 * sd), "bland_altman_high": float(np.mean(d) + 1.959964 * sd),
           "pearson_r": float(np.corrcoef(s, t)[0, 1]) if np.std(s) > 0 else float("nan")}
    for name, lo, hi in RANGES:
        m = (t >= lo) & (t < hi)
        row[f"n_{name}"] = int(m.sum())
        row[f"rmse_{name}"] = float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("nan")
        row[f"bias_{name}"] = float(np.mean(d[m])) if m.any() else float("nan")
    return row


def plot_concept(plt, s, t, path):
    fig, ax = plt.subplots(figsize=(las.COL1, 3.0), layout="constrained")
    hb = ax.hexbin(t, s, gridsize=60, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    lims = [TARGET_CLIP[0], TARGET_CLIP[1]]
    ax.plot(lims, lims, color=las.INK2, linewidth=0.8, linestyle="--", label="identity")
    ax.axhline(CONCEPT_CUT, color=las.ORANGE, linewidth=0.8, linestyle=":")
    ax.axvline(CONCEPT_CUT, color=las.ORANGE, linewidth=0.8, linestyle=":", label="90 (label cut)")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Reference window-median ScalcO2 (clipped 50-100)")
    ax.set_ylabel("Estimated ScalcO2 s_hat (out-of-fold)")
    ax.set_title("Concept s_hat, KAN regressor on P5,\nrecording-grouped out-of-fold (host only)")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="windows (log)")
    las.save_figure(fig, path)


def stage4(out, args):
    from sklearn.model_selection import GroupKFold

    v2, folds_rec, folds_pat, _, _ = load_v2()
    table = load_physio(out)
    y = v2["label"].to_numpy(np.int64)
    valid = table["beat_valid"].to_numpy(bool)
    s2oof = pd.read_csv(out / "oof_stage2.csv.gz")
    target_raw = v2["median_spo2"].to_numpy(float)              # concept target only, never an input
    t = np.clip(target_raw, *TARGET_CLIP)
    x5 = feature_matrix(v2, table, P5)
    x21 = feature_matrix(v2, table, V16P5)
    given_raw = feature_matrix(v2, table, GIVEN).astype(np.float64)
    groupings = {"recording": (folds_rec, v2["recording_id"].to_numpy()), "patient": (folds_pat, v2["patient_id"].to_numpy())}
    checks = {"target": "median_spo2 clipped to [50, 100]; window-median ScalcO2 (end-tidal O2 derived), not pulse-oximeter SpO2",
              "rows_clipped_low": int(np.sum(target_raw < TARGET_CLIP[0])), "rows_clipped_high": int(np.sum(target_raw > TARGET_CLIP[1])),
              "label_equals_target_below_90": bool(np.array_equal(y, (target_raw < 90).astype(int))),
              "leakage_guard": "guard_inputs asserted for P5, V16+P5, GIVEN and every head; oracle head is the single sanctioned exception"}

    # Cross-fitting splits: inner GroupKFold(5) inside each outer training fold, grouped like the outer folds.
    splits, payloads = {}, []
    kan_spec = {"features": list(P5), **RECIPE, "seed": 42, "loss": "MSE on the per-fold standardized target, linear output"}
    for grouping, (folds, groups) in groupings.items():
        for k in range(1, 6):
            train = folds != k
            payloads.append(("kanreg", {**kan_spec, "name": f"kanreg_{grouping}_outer{k}"}, (x5, t, train)))
            rows = np.flatnonzero(train)
            inner = list(GroupKFold(n_splits=5).split(rows, groups=groups[rows]))
            for j, (itr, ite) in enumerate(inner, start=1):
                if set(groups[rows[itr]]) & set(groups[rows[ite]]):
                    raise StopAnalysis("Inner fold group leakage")
                mask = np.zeros(len(rows), bool)
                mask[itr] = True
                payloads.append(("kanreg", {**kan_spec, "name": f"kanreg_{grouping}_outer{k}_inner{j}"}, (x5[rows], t[rows], mask)))
            splits[(grouping, k)] = (rows, inner)
    print(f"Stage 4: {len(payloads)} concept-regressor jobs ...", flush=True)
    results = run_jobs_cached(payloads, args.jobs, args.cache)

    quality_rows, coef_rows, task_probs, bottleneck_scores, explain = [], [], {}, {}, {}
    s_store = {}
    for grouping, (folds, groups) in groupings.items():
        s_outer = {pred: np.full(N_ROWS, np.nan) for pred in CONCEPT_PREDICTORS}
        heads = {}
        contrib = np.full((N_ROWS, len(HEAD_INPUTS)), np.nan)
        logit = np.full(N_ROWS, np.nan)
        for k in range(1, 6):
            test = folds == k
            train = ~test
            rows, inner = splits[(grouping, k)]
            s_outer["kan_p5"][test] = results[f"kanreg_{grouping}_outer{k}"]["prediction"]
            s_outer["ridge_p5"][test] = ridge_predict(x5, t, train)
            s_outer["ridge_v16p5"][test] = ridge_predict(x21, t, train)
            s_outer["constant"][test] = t[train].mean()
            s_inner = {pred: np.full(len(rows), np.nan) for pred in ("kan_p5", "ridge_p5", "ridge_v16p5")}
            for j, (itr, ite) in enumerate(inner, start=1):
                mask = np.zeros(len(rows), bool)
                mask[itr] = True
                s_inner["kan_p5"][ite] = results[f"kanreg_{grouping}_outer{k}_inner{j}"]["prediction"]
                s_inner["ridge_p5"][ite] = ridge_predict(x5[rows], t[rows], mask)
                s_inner["ridge_v16p5"][ite] = ridge_predict(x21[rows], t[rows], mask)
            if not all(np.isfinite(v).all() for v in s_inner.values()):
                raise StopAnalysis("Inner out-of-fold s_hat incomplete")
            given_i, _ = impute(given_raw, train)

            def matrix(columns, s_values, index):
                block = []
                for c in columns:
                    if c == "s_hat":
                        block.append(s_values)
                    elif c == ORACLE_INPUT:
                        block.append(t[index])
                    else:
                        block.append(given_i[index, GIVEN.index(c)])
                return np.column_stack(block)

            jobs = [("kan_p5", name, cols, False) for name, cols in HEAD_VARIANTS.items()]
            jobs += [("ridge_p5", "full", HEAD_INPUTS, False), ("ridge_v16p5", "full", HEAD_INPUTS, False)]
            jobs += [("oracle", name, cols, True) for name, cols in ORACLE_VARIANTS.items()]
            for pred, name, cols, oracle in jobs:
                guard_head(cols, oracle=oracle)
                s_tr = s_inner[pred] if pred in s_inner else None
                s_te = s_outer[pred][test] if pred in s_outer else None
                p, coef, intercept, z = head_fit(matrix(cols, s_tr, train), y[train], matrix(cols, s_te, test))
                key = f"head_{name}" if pred in ("kan_p5", "oracle") else f"head_full_{pred}"
                heads.setdefault(key, np.full(N_ROWS, np.nan))[test] = p
                coef_rows.append({"grouping": grouping, "concept_predictor": pred, "head": name, "fold": str(k),
                                  "training": "cross-fitted inner out-of-fold s_hat" if pred != "oracle" else "oracle (true target)",
                                  "intercept": intercept, **{f"coef_{c}": float(w) for c, w in zip(cols, coef)}})
                if pred == "kan_p5" and name == "full":
                    contrib[test] = z * coef[None, :]
                    logit[test] = intercept + contrib[test].sum(axis=1)
        for pred in CONCEPT_PREDICTORS:
            if not np.isfinite(s_outer[pred]).all():
                raise StopAnalysis(f"{grouping} {pred} outer s_hat incomplete")
            for subset, mask in (("all", np.ones(N_ROWS, bool)), ("beat-valid", valid)):
                quality_rows.append(concept_quality(pred, grouping, subset, s_outer[pred][mask], t[mask]))
        # Pooled head: all windows, every s_hat out-of-fold (outer), given concepts imputed with the all-window mean.
        given_all, _ = impute(given_raw, np.ones(N_ROWS, bool))
        _, coef, intercept, _ = head_fit(np.column_stack([s_outer["kan_p5"], given_all]), y, np.column_stack([s_outer["kan_p5"], given_all]))
        coef_rows.append({"grouping": grouping, "concept_predictor": "kan_p5", "head": "full", "fold": "pooled",
                          "training": "all windows, outer out-of-fold s_hat", "intercept": intercept,
                          **{f"coef_{c}": float(w) for c, w in zip(HEAD_INPUTS, coef)}})
        s2_tag = "recording_seed42" if grouping == "recording" else "patient_seed42"
        task_probs[grouping] = {**heads, "tinykan_p5": s2oof[f"P5_tinykan_{s2_tag}"].to_numpy(float),
                                "tinykan_v2_16": s2oof[f"v2-16_tinykan_{s2_tag}"].to_numpy(float)}
        bottleneck_scores[grouping] = {"bottleneck_kan_p5": CONCEPT_CUT - s_outer["kan_p5"],
                                       "bottleneck_ridge_p5": CONCEPT_CUT - s_outer["ridge_p5"],
                                       "bottleneck_ridge_v16p5": CONCEPT_CUT - s_outer["ridge_v16p5"]}
        explain[grouping] = (contrib, logit)
        s_store[grouping] = s_outer

    # Task metrics with paired bootstrap (same seeds and groups as Stage 2, so resamples are shared).
    task_rows, paired = [], []
    head_summary = {}
    for grouping, (folds, groups) in groupings.items():
        seed = BOOT_SEED_REC if grouping == "recording" else BOOT_SEED_PAT
        point_a, draws_a = lm.group_bootstrap(y, task_probs[grouping], groups, N_BOOT, seed, THRESHOLD)
        point_b, draws_b = lm.group_bootstrap(y, bottleneck_scores[grouping], groups, N_BOOT, seed, BOTTLENECK_THRESHOLD)
        point, draws = {**point_a, **point_b}, {**draws_a, **draws_b}
        for key, p in {**task_probs[grouping], **bottleneck_scores[grouping]}.items():
            bottleneck = key.startswith("bottleneck")
            threshold = BOTTLENECK_THRESHOLD if bottleneck else THRESHOLD
            meta = {"grouping": grouping, "model": key,
                    "operating_point": "flag = s_hat < 90 (score 90 - s_hat)" if bottleneck else "P >= 0.50",
                    "note": ("oracle: trivially near-perfect, the label is median < 90" if key.startswith("head_oracle") else
                             "Stage 2 out-of-fold" if key.startswith("tinykan") else "")}
            low, high = lm.percentile_ci(draws[key]["auc"])
            task_rows.append({**metric_row(meta, y, p, np.ones(N_ROWS, bool), "all", threshold), "auc_ci_low": low, "auc_ci_high": high})
            task_rows.append(metric_row(meta, y, p, valid, "beat-valid", threshold))
        pairs = [("head_full", "tinykan_p5"), ("head_full", "tinykan_v2_16"), ("bottleneck_kan_p5", "head_full"),
                 ("head_full", "head_s_hat_only"), ("head_s_hat+hr_bpm", "head_s_hat_only"), ("head_s_hat+pi_beat", "head_s_hat_only"),
                 ("head_s_hat+sqi_template", "head_s_hat_only"), ("head_s_hat+sqi_ibi_ratio", "head_s_hat_only"),
                 ("head_given_only", "head_full"), ("head_full_ridge_v16p5", "head_full"), ("head_full_ridge_p5", "head_full")]
        paired += paired_rows(pairs, point, draws, grouping, seed)
        head_summary[grouping] = {k: point[k]["auc"] for k in point}
    tasks = pd.DataFrame(task_rows)
    paired = pd.DataFrame(paired)
    quality = pd.DataFrame(quality_rows)
    coefs = pd.DataFrame(coef_rows)

    # Concept-level explanation: exact logit contributions w_k * z_k of the linear head (reference: training mean, z = 0).
    exp_rows, exp_head = [], {}
    for grouping, (contrib, logit) in explain.items():
        p_head = task_probs[grouping]["head_full"]
        flagged = p_head >= THRESHOLD
        order = lm.firmware_abs_order(contrib)
        top = order[:, 0]
        top_c = contrib[np.arange(N_ROWS), top]
        opposes = np.where(flagged, top_c < 0, top_c > 0)
        shares = {}
        for c, name in enumerate(HEAD_INPUTS):
            row = {"grouping": grouping, "concept": name}
            for label in (1, 0):
                row[f"top_share_y{label}"] = float(np.mean(top[y == label] == c))
                row[f"mean_contribution_y{label}"] = float(np.mean(contrib[y == label, c]))
            row["top_share_flagged"] = float(np.mean(top[flagged] == c))
            row["top_share_unflagged"] = float(np.mean(top[~flagged] == c))
            exp_rows.append(row)
            shares[name] = (row["top_share_y1"], row["top_share_y0"])
        logit_check = float(np.max(np.abs(1.0 / (1.0 + np.exp(-logit)) - p_head)))
        exp_head[grouping] = {"top_concept_share_y1_y0": shares,
                              "tvd_top_concept_label1_vs_label0": float(0.5 * sum(abs(a - b) for a, b in shares.values())),
                              "share_top_contribution_opposes_head_decision": float(np.mean(opposes)),
                              "share_top_opposes_flagged": float(np.mean(opposes[flagged])),
                              "share_top_opposes_unflagged": float(np.mean(opposes[~flagged])),
                              "contributions_reconstruct_head_probability_max_abs": logit_check}
    explanation = pd.DataFrame(exp_rows)

    # Concept sign (pre-registered): s_hat coefficient negative in every fold of both groupings and pooled.
    # Column access by key: DataFrame.head is a method, so coefs.head == "full" is always False.
    full = coefs[(coefs["concept_predictor"] == "kan_p5") & (coefs["head"] == "full")]
    s_only = coefs[(coefs["concept_predictor"] == "kan_p5") & (coefs["head"] == "s_hat_only")]
    if len(full) != 2 * 6 or len(s_only) != 2 * 5:     # 5 outer folds + pooled per grouping; s_hat_only has no pooled row
        raise StopAnalysis(f"Concept sign rows: full {len(full)} (expected 12), s_hat_only {len(s_only)} (expected 10)")
    sign_ok = bool((full["coef_s_hat"] < 0).all())

    out.mkdir(parents=True, exist_ok=True)
    write_csv(quality, out, "s4_concept_quality.csv")
    write_csv(tasks, out, "s4_task_metrics.csv")
    write_csv(paired, out, "s4_paired_auc_difference.csv")
    write_csv(coefs, out, "s4_head_coefficients.csv")
    write_csv(explanation, out, "s4_concept_explanation.csv")
    per = {"row_index": np.arange(N_ROWS), "fold_recording": folds_rec, "fold_patient": folds_pat, "y_true": y, "beat_valid": valid.astype(int)}
    for grouping in groupings:
        per.update({f"s_hat_{pred}_{grouping}": s_store[grouping][pred] for pred in ("kan_p5", "ridge_p5", "ridge_v16p5")})
        per.update({f"p_head_full_{grouping}": task_probs[grouping]["head_full"]})
        per.update({f"contribution_{c}_{grouping}": explain[grouping][0][:, i] for i, c in enumerate(HEAD_INPUTS)})
    write_gzip_csv(pd.DataFrame(per), out, "oof_stage4.csv.gz")
    plt = las.setup_style()
    plot_concept(plt, s_store["recording"]["kan_p5"], t, out / "fig_concept_scatter.png")

    def qcell(pred, grouping, subset="all"):
        r = quality[(quality.concept_predictor == pred) & (quality.grouping == grouping) & (quality.subset == subset)].iloc[0]
        return {c: float(r[c]) for c in ("mae", "rmse", "bias", "bland_altman_low", "bland_altman_high", "pearson_r",
                                         "rmse_lt_80", "rmse_80_to_90", "rmse_ge_90", "n_lt_80", "n_80_to_90", "n_ge_90")}

    def tcell(grouping, model):
        r = tasks[(tasks.grouping == grouping) & (tasks.model == model) & (tasks.subset == "all")].iloc[0]
        return {c: float(r[c]) for c in ("auc", "auc_ci_low", "auc_ci_high", "pr_auc", "recall", "specificity", "candidate_rate")}

    kan_params_count = int(results["kanreg_recording_outer1"]["parameter_count"])
    s4 = {
        "checks": checks,
        "concept_quality": {f"{pred}_{g}": qcell(pred, g) for pred in CONCEPT_PREDICTORS for g in groupings},
        "task": {f"{g}_{m}": tcell(g, m) for g in groupings for m in
                 ("head_full", "head_s_hat_only", "head_given_only", "bottleneck_kan_p5", "bottleneck_ridge_v16p5",
                  "head_full_ridge_p5", "head_full_ridge_v16p5", "tinykan_p5", "tinykan_v2_16", "head_oracle_full", "head_oracle_only")},
        "head_coefficients_full": full.drop(columns=["concept_predictor", "head", "training"]).to_dict("records"),
        "s_hat_only_coefficients": s_only[["grouping", "fold", "coef_s_hat"]].to_dict("records"),
        "concept_sign_negative_everywhere": sign_ok,
        "concept_sign_rule": "pre-registered: coef on s_hat < 0 in the pooled head and in every outer fold of both groupings",
        "concept_explanation": exp_head,
        "paired_auc": paired[["grouping", "comparison", "difference", "ci_low", "ci_high"]].to_dict("records"),
        "cost_counted_not_measured": {
            "concept_regressor_parameters": kan_params_count, "head_parameters": len(HEAD_INPUTS) + 1,
            "forwards_per_decision": "1 KAN regressor forward + 5 multiply-adds (head)",
            "concept_explanation_cost": "logit contributions w_k z_k come from the head forward itself (exact, linear)",
            "explaining_s_hat_by_inputs": f"n + 1 = {len(P5) + 1} regressor forwards with the device method (not run here)",
            "tinykan_p5_parameters": models.tinykan_param_count(5, 12, 8),
        },
        "statement": "s_hat is an internal host concept, uncalibrated for the MAX30102 (R-to-saturation is sensor-specific; "
                     "OpenOximetry-to-MAX30102 transfer untested). It is not an oxygen reading; the device output stays a risk flag.",
        "job_seconds_total": float(sum(float(r["seconds"]) for r in results.values())),
    }
    write_json(s4, out, "s4.json")
    print(json.dumps({k: s4[k] for k in ("checks", "concept_quality", "task", "concept_sign_negative_everywhere", "concept_explanation")},
                     indent=1, default=las._json_default))
    print(paired[["grouping", "comparison", "difference", "ci_low", "ci_high"]].round(4).to_string(index=False))
    print(full.round(3).to_string(index=False))
    return 0


# ----------------------------------------------------------------------------- comparison helpers

TIMING_KEYS = {"seconds", "job_seconds", "job_seconds_total"}
NOT_COMPARED = {"summary.json", "README.md", "rerun_check.json", "superseded_check.json"}


def _leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaves(value, f"{prefix}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _leaves(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _timing(path):
    return any(part.split("[")[0] in TIMING_KEYS for part in path.split("/"))


def compare_json(a, b):
    la = {k: v for k, v in _leaves(a) if not _timing(k)}
    lb = {k: v for k, v in _leaves(b) if not _timing(k)}
    numeric_diff, mismatched = 0.0, []
    for key in sorted(set(la) & set(lb)):
        x, y = la[key], lb[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            if not (np.isnan(x) and np.isnan(y)):
                numeric_diff = max(numeric_diff, abs(float(x) - float(y)))
        elif x != y:
            mismatched.append(key)
    return {"kind": "json", "only_in_a": sorted(set(la) - set(lb)), "only_in_b": sorted(set(lb) - set(la)),
            "max_abs_numeric_diff": numeric_diff, "non_numeric_mismatches": mismatched[:20],
            "n_non_numeric_mismatches": len(mismatched), "timing_keys_ignored": sorted(TIMING_KEYS)}


def compare_frames(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {"kind": "csv", "same_shape_and_columns": False, "columns_only_in_a": sorted(set(a.columns) - set(b.columns)),
                "columns_only_in_b": sorted(set(b.columns) - set(a.columns)), "rows": [len(a), len(b)]}
    numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    diff, nan_equal = 0.0, True
    for c in numeric:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        nan_equal &= bool(np.array_equal(np.isnan(x), np.isnan(y)))
        both = ~np.isnan(x) & ~np.isnan(y)
        if both.any():
            diff = max(diff, float(np.max(np.abs(x[both] - y[both]))))
    text = [c for c in a.columns if c not in numeric]
    text_equal = all(bool((a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all()) for c in text)
    return {"kind": "csv", "same_shape_and_columns": True, "rows": len(a), "max_abs_numeric_diff": diff,
            "nan_pattern_equal": nan_equal, "text_columns_equal": text_equal}


def compare_npz(a, b):
    with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
        if sorted(za.files) != sorted(zb.files):
            return {"kind": "npz", "same_keys": False}
        diff, text_equal = 0.0, True
        for k in za.files:
            if za[k].dtype.kind in "fiu":
                diff = max(diff, float(np.max(np.abs(za[k].astype(float) - zb[k].astype(float)))) if za[k].size else 0.0)
            else:
                text_equal &= bool(np.array_equal(za[k], zb[k]))
        return {"kind": "npz", "same_keys": True, "max_abs_numeric_diff": diff, "text_arrays_equal": text_equal}


def compare_dirs(a, b, names=None):
    """Compare every file present in both directories (figures by SHA-256 only, not required to match)."""
    a, b = Path(a), Path(b)
    files_a = {p.name for p in a.iterdir() if p.is_file()} - NOT_COMPARED
    files_b = {p.name for p in b.iterdir() if p.is_file()} - NOT_COMPARED
    common = sorted(files_a & files_b) if names is None else [n for n in names if n in files_a & files_b]
    result = {}
    for name in common:
        pa, pb = a / name, b / name
        if name.endswith((".csv", ".csv.gz")):
            result[name] = compare_frames(pd.read_csv(pa), pd.read_csv(pb))
        elif name.endswith(".json"):
            result[name] = compare_json(json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8")))
        elif name.endswith(".npz"):
            result[name] = compare_npz(pa, pb)
        else:
            result[name] = {"kind": "binary (figure)", "sha256_equal": sha256(pa) == sha256(pb)}
    return {"files": result, "only_in_a": sorted(files_a - files_b), "only_in_b": sorted(files_b - files_a)}


def max_numeric_diff(comparison):
    return max((v.get("max_abs_numeric_diff", 0.0) for v in comparison["files"].values()), default=0.0)


# ----------------------------------------------------------------------------- rerun (cheap subset, uncached)

RERUN_SUBSET = ("P5_logreg_recording", "P5_hist_gb_recording", "P5_tinykan_recording_seed42", "kanreg_recording_outer1")


def rerun(out, args):
    """Fresh Stage 1 plus an uncached retrain of a cheap subset of jobs, compared with the outputs and the job cache."""
    if RERUN_DIR.exists() and any(RERUN_DIR.iterdir()):
        raise StopAnalysis(f"{RERUN_DIR} is not empty; move it aside first (the rerun never overwrites)")
    started = time.perf_counter()
    print("Rerun: Stage 1 into build/physio/rerun ...", flush=True)
    code = stage1(RERUN_DIR)
    if code:
        return code
    s1_names = ("physio_features.csv.gz", "s1_feature_description.csv", "s1_feature_spearman.csv", "s1_sqi_vs_artefact_rows.csv",
                "s1_checks.json")
    stage1_cmp = compare_dirs(out, RERUN_DIR, s1_names)

    v2, folds_rec, _, _, _ = load_v2()
    table = load_physio(RERUN_DIR)                       # the freshly extracted table
    y = v2["label"].to_numpy(np.int64)
    payloads = []
    for fs, learner in (("P5", "logreg"), ("P5", "hist_gb"), ("P5", "tinykan")):
        spec = {"name": f"{fs}_{learner}_recording" + ("_seed42" if learner == "tinykan" else ""), "feature_set": fs,
                "features": list(FEATURE_SETS[fs]), "learner": learner, "grouping": "recording"}
        if learner == "tinykan":
            spec.update({**RECIPE, "seed": 42})
        payloads.append(("cv", spec, (feature_matrix(v2, table, FEATURE_SETS[fs]), y, folds_rec)))
    t = np.clip(v2["median_spo2"].to_numpy(float), *TARGET_CLIP)
    payloads.append(("kanreg", {"features": list(P5), **RECIPE, "seed": 42, "loss": "MSE on the per-fold standardized target, linear output",
                                "name": "kanreg_recording_outer1"}, (feature_matrix(v2, table, P5), t, folds_rec != 1)))
    keys = {}
    for kind, spec, arrays in payloads:
        cached = Path(args.cache) / f"{spec['name']}.npz"
        with np.load(cached, allow_pickle=False) as z:
            keys[spec["name"]] = {"cache_key_equal": str(z["_key"]) == job_key(kind, spec, arrays)}
    print(f"Rerun: {len(payloads)} uncached job(s) ...", flush=True)
    fresh = run_jobs_cached(payloads, min(args.jobs, len(payloads)), RERUN_DIR / "jobs")
    jobs = {}
    for name in RERUN_SUBSET:
        with np.load(Path(args.cache) / f"{name}.npz", allow_pickle=False) as z:
            field = "prediction" if name.startswith("kanreg") else "probabilities"
            old, new = z[field].astype(float), fresh[name][field].astype(float)
        jobs[name] = {**keys[name], "field": field, "bitwise_identical": bool(np.array_equal(old, new)),
                      "max_abs_diff": float(np.max(np.abs(old - new)))}
        if field == "probabilities":
            jobs[name]["auc_cached"], jobs[name]["auc_fresh"] = lm.weighted_auc(y, old), lm.weighted_auc(y, new)
    check = {"what": "fresh Stage 1 into build/physio/rerun compared with the outputs; uncached retrain of a cheap job subset "
                     "compared with build/physio/jobs (the cache every stage read)",
             "stage1": stage1_cmp, "stage1_max_abs_numeric_diff": max_numeric_diff(stage1_cmp), "jobs": jobs,
             "environment": environment(), "seconds": round(time.perf_counter() - started, 1)}
    write_json(check, out, "rerun_check.json")
    print(json.dumps({"stage1_max_abs_numeric_diff": check["stage1_max_abs_numeric_diff"], "jobs": jobs}, indent=1, default=las._json_default))
    return 0


def superseded(out, args):
    """Compare a directory of superseded outputs (an earlier run of the same stages) with the current outputs."""
    if args.against is None:
        raise StopAnalysis("--against DIR is required for --stage superseded")
    against = args.against if args.against.is_absolute() else ROOT / args.against
    cmp = compare_dirs(against, out)
    check = {"against": against.relative_to(ROOT).as_posix() if against.is_relative_to(ROOT) else str(against),
             "note": args.note, "comparison": cmp, "max_abs_numeric_diff_all_common_files": max_numeric_diff(cmp)}
    write_json(check, out, "superseded_check.json")
    print(json.dumps({"max_abs_numeric_diff_all_common_files": check["max_abs_numeric_diff_all_common_files"],
                      "only_in_superseded": cmp["only_in_a"], "only_in_current": cmp["only_in_b"],
                      "json_key_changes": {k: (v["only_in_a"][:5], v["only_in_b"][:5]) for k, v in cmp["files"].items()
                                           if v.get("kind") == "json" and (v["only_in_a"] or v["only_in_b"])},
                      "csv_shape_changes": [k for k, v in cmp["files"].items() if v.get("kind") == "csv" and not v["same_shape_and_columns"]]},
                     indent=1))
    return 0


# ----------------------------------------------------------------------------- summary

STAGE_FILES = ("s1_checks.json", "s2.json", "s3.json", "s4.json")
PREREG = ROOT / "docs" / "physio-concept-preregistration.md"
CODE_FILES = (SCRIPTS / "analyze_physio_concept.py", SCRIPTS / "physio_features.py", ROOT / "tests" / "test_physio_features.py", PREREG)


def _git(*args):
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _n(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _auc_ci(d, nd=3):
    return f"{d['auc']:.{nd}f} [{d['auc_ci_low']:.{nd}f}, {d['auc_ci_high']:.{nd}f}]"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def summary(out, args):
    missing = [f for f in STAGE_FILES if not (out / f).exists()]
    if missing:
        raise StopAnalysis(f"summary needs every stage file; missing {missing}")
    s1, s2, s3, s4 = (read_json(out, f) for f in STAGE_FILES)
    h3 = s3["headline"]
    full = s4["head_coefficients_full"]
    readings = {
        "a_prediction_kept": {**s2["primary_reading"], "patient_secondary": s2["patient_reading"]},
        "b_direction": {key: {"role": h3[key]["role"],
                              "sign_agreement_device_reference": h3[key]["sign_agreement_training_mean"]["all"],
                              "ice_share_non_decreasing": h3[key]["ice_share_non_decreasing"], "ice_reading": h3[key]["ice_reading"],
                              "sign_agreement_label0_reference_sensitivity": h3[key]["sign_agreement_label0_median"]["all"]}
                        for key in ("P5_r_beat", "P5_hr_bpm", "v2_ratio_of_ratios")},
        "c_concept_sign": {"holds": s4["concept_sign_negative_everywhere"], "rule": s4["concept_sign_rule"],
                           "coef_s_hat_min": min(r["coef_s_hat"] for r in full), "coef_s_hat_max": max(r["coef_s_hat"] for r in full),
                           "rows": len(full)},
    }
    optional = {name: read_json(out, name) for name in ("rerun_check.json", "superseded_check.json") if (out / name).exists()}
    payload = {
        "study": "physiological-input TinyKAN and concept-bottleneck prototype (host only, no board, final report only)",
        "preregistration": {"path": PREREG.relative_to(ROOT).as_posix(),
                            "first_commit": _git("log", "--diff-filter=A", "--format=%H %cI", "--", PREREG.relative_to(ROOT).as_posix())},
        "readings": readings,
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for k, p in P.items() if p.is_file()},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in CODE_FILES},
        "outputs_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in {"summary.json", "README.md"}},
        "checks": {"rerun_check": {"stage1_max_abs_numeric_diff": optional["rerun_check.json"]["stage1_max_abs_numeric_diff"],
                                   "jobs": optional["rerun_check.json"]["jobs"]} if "rerun_check.json" in optional else None,
                   "superseded_check_max_abs_numeric_diff": optional["superseded_check.json"]["max_abs_numeric_diff_all_common_files"]
                   if "superseded_check.json" in optional else None},
        "git_head_at_summary": _git("rev-parse", "HEAD"),
        "environment": environment(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(payload, out, "summary.json")
    new_file(out, "README.md").write_text(readme(s1, s2, s3, s4, readings, optional), encoding="utf-8")
    print(json.dumps(readings, indent=1, default=las._json_default))
    return 0


def readme(s1, s2, s3, s4, readings, optional):
    h2, h3, a = s2["headline_all_windows"], s3["headline"], readings["a_prediction_kept"]
    rd = readings["b_direction"]
    task, cq = s4["task"], s4["concept_quality"]
    L = ["# Physiological-input TinyKAN and concept bottleneck (host only)", "",
         "Pre-registration: `docs/physio-concept-preregistration.md`, committed before any feature was computed or model trained. "
         "Host only, laptop, no board. Results go to the final report only; the IEEE paper is not edited. "
         "Generated by `scripts/analyze_physio_concept.py --stage summary` from the stage files in this directory.", "",
         "Label: window-median ScalcO2 < 90 (ScalcO2 is calculated from end-tidal O2; it is not a pulse-oximeter SpO2). "
         "The learned concept `s_hat` (estimated ScalcO2) is an internal host quantity, uncalibrated for the MAX30102. "
         "It is not an oxygen reading; the device output stays a risk flag.", "",
         "## Pre-registered readings", "",
         _table(["Question", "Rule (fixed in advance)", "Result", "Reading"], [
             ["(a) Prediction kept", "paired recording-bootstrap 95% CI of AUC(P5 TinyKAN) - AUC(v2-16 TinyKAN): lower bound > -0.02; "
              "'higher' only if > 0", f"{a['difference']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]; patient-grouped "
              f"{a['patient_secondary']['difference']:+.4f} [{a['patient_secondary']['ci'][0]:+.4f}, {a['patient_secondary']['ci'][1]:+.4f}]",
              a["reading"]],
             *[[f"(b) Direction, {k.replace('_', ' ', 1)} ({v['role']})", "sign agreement (device reference) and ICE non-decreasing share: "
                ">= 0.90 consistent, 0.50-0.90 mixed, < 0.50 contradicts",
                f"sign {v['sign_agreement_device_reference']['share']:.3f}; ICE {v['ice_share_non_decreasing']:.3f}",
                f"sign {v['sign_agreement_device_reference']['reading']}; ICE {v['ice_reading']}"] for k, v in rd.items()],
             ["(c) Concept sign", "coefficient on s_hat < 0 in the pooled head and every outer fold, both groupings",
              f"{readings['c_concept_sign']['rows']} rows, coef {readings['c_concept_sign']['coef_s_hat_min']:.3f} to "
              f"{readings['c_concept_sign']['coef_s_hat_max']:.3f}", "holds" if readings["c_concept_sign"]["holds"] else "fails"]]), "",
         "## Stage 1: beat features", "",
         f"- Beat-valid {s1['validity']['beat_valid']:,} of 48,662 windows; {s1['validity']['beat_invalid']} invalid (all with fewer than 4 beats), "
         f"invalid-window prevalence {s1['validity']['prevalence_beat_invalid']:.2f}.",
         f"- HR within 40-180 bpm on {s1['hr_within_40_180_share_of_valid']:.2%} of valid windows; Spearman HR vs 60 x dominant frequency "
         f"{s1['spearman_hr_vs_60x_dominant_frequency']:.3f}; r_beat vs ratio_of_ratios {s1['spearman_r_beat_vs_ratio_of_ratios']:.3f}; "
         f"pi_beat vs ir_perfusion_index {s1['spearman_pi_beat_vs_ir_perfusion_index']:.3f}.",
         "- Univariate AUC (valid windows): " + ", ".join(f"{k} {v:.3f}" for k, v in s1["univariate_auc_valid"].items()) + ".",
         f"- The {s1['sqi_vs_artefact_rows']['artefact_rows']:,} ADR-0007 artefact rows (replayed sensor buffers) look like perfect beats to the "
         f"SQIs: template SQI AUC {s1['sqi_vs_artefact_rows']['sqi_template_auc_natural_direction']:.3f} and IBI-ratio AUC "
         f"{s1['sqi_vs_artefact_rows']['sqi_ibi_ratio_auc_natural_direction']:.3f} in the quality direction. These SQIs do not detect that artefact.", "",
         "## Stage 2: prediction (out-of-fold, all windows, threshold 0.50)", "",
         _table(["Model", "Grouping", "AUC [95% CI]", "Recall", "Specificity", "Size"],
                [[k.rsplit("_", 2)[0] if "seed" in k else k.rsplit("_", 1)[0], "patient" if "patient" in k else "recording",
                  _auc_ci(h2[k]), _n(h2[k]["recall"]), _n(h2[k]["specificity"]), f"{h2[k]['size']:.0f}"]
                 for k in ("P5_tinykan_recording_seed42", "P3_tinykan_recording_seed42", "V16+P5_tinykan_recording_seed42",
                           "v2-16_tinykan_recording_seed42", "P5_logreg_recording", "v2-16_logreg_recording", "P5_hist_gb_recording",
                           "v2-16_hist_gb_recording", "P5_tinykan_patient_seed42", "v2-16_tinykan_patient_seed42")]), "",
         f"P5 TinyKAN over six seeds: AUC {s2['p5_tinykan_seed_spread']['auc']['mean']:.4f} +- {s2['p5_tinykan_seed_spread']['auc']['sd']:.4f}. "
         f"Size: TinyKAN trainable parameters; LogReg coefficients plus intercept; HistGB fold-mean tree nodes. "
         f"Reproduction check: v2-16 TinyKAN seed 42 retrained to AUC {s2['reproduction']['retrain_auc']:.4f} vs {s2['reproduction']['reference_auc']:.4f} "
         f"(not bitwise; within the 0.002 tolerance).", "",
         "## Stage 3: explanation checks (device method, in-sample, host)", "",
         _table(["Check", "P5 (5 inputs)", "v2 header (16 inputs)"], [
             ["Deletion AOPC, all windows: firmware-absolute vs random",
              f"{h3['P5_aopc']['all_firmware_absolute']:.4f} vs {h3['P5_aopc']['all_random_mean']:.4f}",
              f"{h3['v2_aopc']['all_firmware_absolute']:.4f} vs {h3['v2_aopc']['all_random_mean']:.4f}"],
             ["Deletion AOPC, candidates (P >= 0.50)",
              f"{h3['P5_aopc']['candidate_p_ge_0_50_firmware_absolute']:.4f} vs {h3['P5_aopc']['candidate_p_ge_0_50_random_mean']:.4f}",
              f"{h3['v2_aopc']['candidate_p_ge_0_50_firmware_absolute']:.4f} vs {h3['v2_aopc']['candidate_p_ge_0_50_random_mean']:.4f}"],
             *[[f"Top-1 feature ({c['model']})", f"{c['top1_most_common_y1']} {c['top1_most_common_y1_share']:.1%} (label 1) / "
                f"{c['top1_most_common_y0']} {c['top1_most_common_y0_share']:.1%} (label 0), TVD {c['tvd_top1_label1_vs_label0']:.3f}", ""]
               for c in h3["class_separation"]],
             ["Candidates whose top bar opposes the flag (device reference)",
              f"{h3['P5_degeneracy_device']['candidate (P >= 0.50)']['fraction_top1_opposes_decision']:.1%}",
              f"{h3['v2_degeneracy_device']['candidate (P >= 0.50)']['fraction_top1_opposes_decision']:.1%}"],
             ["Candidates whose top bar opposes the flag (label-0 median reference, sensitivity)",
              f"{h3['P5_degeneracy_label0ref']['candidate (P >= 0.50)']['fraction_top1_opposes_decision']:.1%}",
              f"{h3['v2_degeneracy_label0ref']['candidate (P >= 0.50)']['fraction_top1_opposes_decision']:.1%}"],
             ["Candidates with a zero top-3 slot / output pre-activation clamped",
              f"{h3['P5_degeneracy_device']['candidate (P >= 0.50)']['fraction_top3_has_zero_delta_slot']:.1%} / "
              f"{h3['P5_degeneracy_device']['candidate (P >= 0.50)']['fraction_output_preactivation_clamped']:.1%}",
              f"{h3['v2_degeneracy_device']['candidate (P >= 0.50)']['fraction_top3_has_zero_delta_slot']:.1%} / "
              f"{h3['v2_degeneracy_device']['candidate (P >= 0.50)']['fraction_output_preactivation_clamped']:.1%}"]]), "",
         "Sign agreement by subset (share of windows with |delta| > 0.0005 where an above-reference value raises the score):", "",
         _table(["Input", "Reference", "all", "label 1", "label 0", "candidate", "non-candidate"],
                [[k, ref_name, *[_n(h3[k][ref][s]["share"]) for s in ("all", "label_1", "label_0", "candidate", "non-candidate")]]
                 for k in ("P5_r_beat", "P5_hr_bpm", "v2_ratio_of_ratios")
                 for ref, ref_name in (("sign_agreement_training_mean", "training mean (device)"),
                                       ("sign_agreement_label0_median", "label-0 median (sensitivity)"))]), "",
         "Post hoc, descriptive only (added after the pre-registered ICE share was seen; no reading changes). Each ICE curve sweeps one "
         "input over z = -3..3 with the window's other inputs fixed:", "",
         _table(["Input", "Median largest single-step drop", "Share with largest drop < 0.05", "Share P(z=+3) >= P(z=-3)", "Median z at curve max"],
                [[k, _n(h3[k]["ice_posthoc_descriptive"]["median_largest_single_step_drop"], 4),
                  _n(h3[k]["ice_posthoc_descriptive"]["share_largest_drop_lt_0_05"]),
                  _n(h3[k]["ice_posthoc_descriptive"]["share_p_at_plus3_ge_p_at_minus3"]),
                  _n(h3[k]["ice_posthoc_descriptive"]["median_z_at_curve_max"], 1)]
                 for k in ("P5_r_beat", "P5_hr_bpm", "v2_ratio_of_ratios")]), "",
         _table(["Output spline", "u at max P", "max P", "P at the u = +3 clamp", "Overshoot above the clamp value"],
                [[r["version"], _n(r["u_at_max_p"], 2), _n(r["max_p"]), _n(r["p_at_u_plus3_clamp"]), _n(r["overshoot_above_clamp_value"], 4)]
                 for r in h3["output_spline_shape"]]), "",
         "## Stage 4: concept bottleneck (sequential, cross-fitted)", "",
         _table(["Concept predictor", "Grouping", "MAE", "RMSE", "Bias", "Bland-Altman limits", "Pearson r"],
                [[k.rsplit("_", 1)[0], k.rsplit("_", 1)[1], _n(v["mae"], 2), _n(v["rmse"], 2), _n(v["bias"], 2),
                  f"{v['bland_altman_low']:.1f} to {v['bland_altman_high']:.1f}", _n(v["pearson_r"])] for k, v in cq.items()]), "",
         "Units are ScalcO2 percentage points against the clipped window median. RMSE by reference range for the KAN regressor "
         f"(recording-grouped): < 80 {cq['kan_p5_recording']['rmse_lt_80']:.2f}, 80-89 {cq['kan_p5_recording']['rmse_80_to_90']:.2f}, "
         f">= 91 {cq['kan_p5_recording']['rmse_ge_90']:.2f}.", "",
         _table(["Task model", "Recording AUC [95% CI]", "Patient AUC [95% CI]", "Recall (rec.)", "Specificity (rec.)"],
                [[m, _auc_ci(task[f"recording_{m}"]), _auc_ci(task[f"patient_{m}"]), _n(task[f"recording_{m}"]["recall"]),
                  _n(task[f"recording_{m}"]["specificity"])]
                 for m in ("head_full", "head_s_hat_only", "bottleneck_kan_p5", "head_given_only", "head_full_ridge_p5",
                           "head_full_ridge_v16p5", "tinykan_p5", "tinykan_v2_16", "head_oracle_full")]), "",
         "`head_full`: LogReg on [s_hat, hr_bpm, pi_beat, sqi_template, sqi_ibi_ratio], trained on inner out-of-fold s_hat. "
         "`bottleneck_kan_p5`: flag = s_hat < 90. `head_oracle_full` uses the true target and is trivially perfect, because the label is the median < 90.", "",
         "Concept-level explanation (exact logit contributions of the linear head): " + "; ".join(
             f"{g}: s_hat is the top concept on {v['top_concept_share_y1_y0']['s_hat'][0]:.1%} of label-1 and "
             f"{v['top_concept_share_y1_y0']['s_hat'][1]:.1%} of label-0 windows, top contribution opposes the head decision on "
             f"{v['share_top_contribution_opposes_head_decision']:.1%} ({v['share_top_opposes_flagged']:.1%} of flagged)"
             for g, v in s4["concept_explanation"].items()) + ".", "",
         f"Cost (counted, not measured): concept regressor {s4['cost_counted_not_measured']['concept_regressor_parameters']} parameters, "
         f"head {s4['cost_counted_not_measured']['head_parameters']}; one regressor forward plus five multiply-adds per decision.", "",
         "## Checks and provenance", ""]
    if "rerun_check.json" in optional:
        rc = optional["rerun_check.json"]

        def job_text(name, v):
            text = "bitwise identical" if v["bitwise_identical"] else f"max abs diff {v['max_abs_diff']:.3g}"
            return f"{name} {text}" + ("" if v["cache_key_equal"] else " (cache key differs)")

        L.append(f"- `rerun_check.json`: fresh Stage 1 max abs numeric difference {rc['stage1_max_abs_numeric_diff']:.3g}; uncached retrain: "
                 + "; ".join(job_text(k, v) for k, v in rc["jobs"].items()) + ".")
    if "superseded_check.json" in optional:
        sc = optional["superseded_check.json"]
        L.append(f"- `superseded_check.json`: {sc['note']} Max abs numeric difference over every common file: "
                 f"{sc['max_abs_numeric_diff_all_common_files']:.3g}.")
    L += ["- Stage 4 deviation: the first Stage 4 run selected the concept-sign rows with `coefs.head`, which is the DataFrame method, so the "
          "selection was empty and the rule passed vacuously. Those Stage 4 outputs were deleted. The selection now uses "
          "`coefs[\"head\"]` and stops unless it finds 12 rows; Stage 4 was rerun from the same cached jobs.",
          "- ICE: the `posthoc_*` columns in `s3_ice_monotonicity.csv`, `ice_posthoc_descriptive` in `s3.json`, and `s3_output_spline_shape.csv` "
          "were added after the pre-registered ICE share was seen. They describe the curves and do not change any reading.",
          "- Inputs, code and output SHA-256 are in `summary.json`.", "",
          "## Wording rules (from the pre-registration)", "",
          "- AOPC is host, in-sample, on the exported model, and functionally grounded only. No explanation is called faithful without that qualification.",
          "- Mean-perturbation deltas are not effect sizes.",
          "- `s_hat` is uncalibrated for the MAX30102: R-to-saturation is sensor-specific, and transfer from OpenOximetry to the MAX30102 is untested.",
          "- 'Recording-grouped' lets a patient appear in train and test; patient-grouped is the secondary reading.", ""]
    return "\n".join(L)


STAGES = {"2": stage2, "3": stage3, "4": stage4, "summary": summary, "rerun": rerun, "superseded": superseded}


# ----------------------------------------------------------------------------- main

def run_stage(stage, out, args):
    before = protected_hashes()
    try:
        code = STAGES[stage](out, args) if stage != "1" else stage1(out)
    except StopAnalysis as stop:
        print(f"STOP: {stop}")
        return 3
    if protected_hashes() != before:
        raise RuntimeError("A protected or write-once path changed during the run")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("1", "2", "3", "4", "summary", "rerun", "superseded", "all"))
    parser.add_argument("--stage-part", default="all", choices=("all", "2a"),
                        help="2a = run only the Stage 2 reproduction check")
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--jobs", type=int, default=14, help="worker processes for model jobs")
    parser.add_argument("--against", type=Path, default=None, help="superseded: directory of an earlier run to compare with")
    parser.add_argument("--note", default="", help="superseded: one sentence on why those outputs were superseded")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache = args.cache if args.cache.is_absolute() else ROOT / args.cache
    stages = ("1", "2", "3", "4", "summary") if args.stage == "all" else (args.stage,)
    for stage in stages:
        code = run_stage(stage, out, args)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

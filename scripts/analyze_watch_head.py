"""Below-95 Watch head beside the shipped below-90 head (host only, laptop, no board).

Pre-registration: docs/watch-head-preregistration.md (first commit c2a6e8c, 7 Nov 2026, before
any model in this study was trained). It is the specification; this script implements it.

Stages (run in this order; every stage re-asserts the frozen inputs, and every stage after
`repro` re-checks the reproduction from the job cache and stops if it fails):
  repro      v2-16 TinyKAN on the original label through train_tinykan_v2.out_of_fold;
             folds must equal the stored recording folds, AUC within 0.002 of 0.907528
  cv         head B (label median < 95) through out_of_fold, recording- and patient-grouped
  seeds      head B, seeds 42, 1, 2, 3, 4, 5 through lamaakal_models.run_cv_job (recording)
  secondary  B_cond on S (recording and patient), LogReg-16 and HistGB-16 on y95 (recording)
  summary    gate, tables, summary.json and README.md from the cached jobs (trains nothing)
  final      head B on all windows, seed 42, as run_final but with y95; refuses unless
             gate.json says pass
  all        repro, cv, seeds, secondary, summary (not final)

Output dir outputs/watch-head-v1/ is write-once: every file refuses to exist already. No board
I/O, no firmware or header writes, no edit of any existing output. Head A is never retrained:
its probabilities are the stored out-of-fold vectors.

Run from the repository root with the interpreter that has torch 2.12 / sklearn 1.9:
  py -3.14 scripts/analyze_watch_head.py --stage repro
  py -3.14 scripts/analyze_watch_head.py --stage cv --jobs 2
  ...
Smoke mode (testing the code only): --epochs N (N != 100) with --out under build/watch-head/;
it uses its own job cache, refuses outputs/ and the official cache, and is marked smoke.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
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
import lamaakal_models as models  # noqa: E402
import train_tinykan_v2 as tk2  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "watch-head-v1"
CACHE_DEFAULT = ROOT / "build" / "watch-head" / "jobs"
SMOKE_ROOT = ROOT / "build" / "watch-head"
SMOKE_CACHE_ROOT = ROOT / "build" / "watch-head" / "smoke-jobs"
CACHE_VERSION = 1
PREREG = ROOT / "docs" / "watch-head-preregistration.md"

P = {
    "v2_features": tk2.V2_CSV,
    "head_a_recording_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "head_a_patient_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
    "v2_header_reference_only": tk2.HEADER_V2,
}
CODE_FILES = (SCRIPTS / "analyze_watch_head.py", SCRIPTS / "train_tinykan_v2.py", SCRIPTS / "lamaakal_models.py",
              SCRIPTS / "lamaakal_metrics.py", ROOT / "tests" / "test_watch_head.py", PREREG)
PROTECTED = (ROOT / "outputs/openox-v2", ROOT / "firmware/Arduino Nano 33/include", ROOT / "outputs/physio-concept-v1",
             ROOT / "outputs/analysis-lamaakal-style-v1", PREREG)

# Frozen facts from the pre-registration ("Data, labels and folds"); asserted at the start of every stage.
N_ROWS, N_RECORDINGS, N_PATIENTS = 48662, 154, 87
BAND_EXPECTED = {"L": (28795, 149, 85), "M": (7785, 151, 86), "N": (12082, 154, 87)}   # windows, recordings, patients
Y95_POSITIVE = 36580
S_SIZE = 19867
DROP_94_96, S_KEEP_M, S_KEEP_N = 5358, 5550, 8959
FOLD_COUNTS = {"recording": [(1448, 2611), (1735, 2252), (1455, 2495), (1511, 2380), (1636, 2344)],
               "patient": [(1730, 2304), (1549, 2494), (1354, 2521), (1488, 2520), (1664, 2243)]}
HEAD_A_EXPECTED = {"recording": {"recall": 0.89269, "specificity": 0.86802, "auc": 0.907528},
                   "patient": {"recall": 0.89932, "specificity": 0.86354, "auc": 0.906936}}
REPRO_AUC_TOLERANCE = 0.002
SAME_PATH_TOLERANCE = 1e-5          # analyze_lamaakal_style.REPRODUCTION_TOLERANCE

THRESHOLD = 0.50
G_RECALL, G_SPECIFICITY = 0.80, 0.50
N_BOOT = 2000
BOOT_SEED_REC, BOOT_SEED_PAT = 20261025, 20261026
SEEDS = (42, 1, 2, 3, 4, 5)
EPOCHS = tk2.N_EPOCHS               # 100
V2_FEATURES = tk2.V2_FEATURES
FORBIDDEN_INPUTS = ("median_spo2", "min_spo2", "label_min_spo2", "label")
OUTPUT_FILES = ("oof_watch.csv.gz", "confusion_3x3.csv", "metrics.csv", "seeds.csv", "sensitivity_94_96.csv",
                "gate.json", "summary.json", "README.md")
FINAL_FILES = ("watch_final_model.npz", "watch_final_model.json")


class StopAnalysis(RuntimeError):
    pass


# ----------------------------------------------------------------------------- pure helpers (tested)

def bands(median_spo2):
    """L < 89, M [91, 95), N >= 95. [89, 91) was dropped at export; any such (or NaN) row is an error."""
    s = np.asarray(median_spo2, dtype=float)
    out = np.full(len(s), "", dtype="<U1")
    out[s < 89] = "L"
    out[(s >= 91) & (s < 95)] = "M"
    out[s >= 95] = "N"
    if (out == "").any():
        raise ValueError(f"{int((out == '').sum())} windows have median in [89, 91) or are NaN")
    return out


def y95(median_spo2):
    return (np.asarray(median_spo2, dtype=float) < 95).astype(np.int64)


def in_94_96(median_spo2):
    s = np.asarray(median_spo2, dtype=float)
    return (s >= 94) & (s <= 96)


def state_rule(p_a, p_b, threshold=THRESHOLD):
    """LOW if P_A >= threshold; else WATCH if P_B >= threshold; else NONE."""
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    return np.where(p_a >= threshold, "LOW", np.where(p_b >= threshold, "WATCH", "NONE"))


def rule_block(m, p_a, p_b, threshold=THRESHOLD):
    """Recall on M (alerted = WATCH or LOW) and specificity on N (NONE), float64, unrounded."""
    m = np.asarray(m).astype(int)
    states = state_rule(p_a, p_b, threshold)
    n_pos, n_neg = int((m == 1).sum()), int((m == 0).sum())
    tp = int(((states != "NONE") & (m == 1)).sum())
    tn = int(((states == "NONE") & (m == 0)).sum())
    watch_n = int(((states == "WATCH") & (m == 0)).sum())
    low_n = int(((states == "LOW") & (m == 0)).sum())
    recall = tp / n_pos if n_pos else float("nan")
    specificity = tn / n_neg if n_neg else float("nan")
    recall_pass, spec_pass = bool(recall >= G_RECALL), bool(specificity >= G_SPECIFICITY)
    return {"n_M": n_pos, "n_N": n_neg, "M_alerted": tp, "N_none": tn, "N_watch": watch_n, "N_low": low_n,
            "recall": recall, "specificity": specificity,
            "false_watch_rate_N": watch_n / n_neg if n_neg else float("nan"),
            "low_share_N": low_n / n_neg if n_neg else float("nan"),
            "recall_pass": recall_pass, "specificity_pass": spec_pass, "pass": recall_pass and spec_pass}


def _gate(m, p_a_rec, p_b_rec, p_a_pat, p_b_pat, rec_groups, n_boot=N_BOOT, seed=BOOT_SEED_REC, threshold=THRESHOLD):
    m = np.asarray(m).astype(int)
    p_a_rec, p_b_rec = np.asarray(p_a_rec, dtype=float), np.asarray(p_b_rec, dtype=float)
    p_a_pat, p_b_pat = np.asarray(p_a_pat, dtype=float), np.asarray(p_b_pat, dtype=float)
    g1 = {**rule_block(m, p_a_rec, p_b_rec, threshold), "grouping": "recording", "scores": "P_A_rec, P_B_rec"}
    g3 = {**rule_block(m, p_a_pat, p_b_pat, threshold), "grouping": "patient", "scores": "P_A_pat, P_B_pat"}
    point, draws = lm.group_bootstrap(m, {"P_B_rec": p_b_rec, "P_A_rec": p_a_rec, "alert_rec": np.maximum(p_a_rec, p_b_rec)},
                                      rec_groups, n_boot, seed, threshold)
    delta = draws["P_B_rec"]["auc"] - draws["P_A_rec"]["auc"]
    finite = np.isfinite(delta)
    low, high = lm.percentile_ci(delta)
    cis = {k: lm.percentile_ci(draws[k]["auc"]) for k in draws}
    g2 = {"auc_P_B_rec": point["P_B_rec"]["auc"], "auc_P_A_rec": point["P_A_rec"]["auc"],
          "auc_ci_P_B_rec": list(cis["P_B_rec"]), "auc_ci_P_A_rec": list(cis["P_A_rec"]),
          "auc_alert_rec": point["alert_rec"]["auc"], "auc_ci_alert_rec": list(cis["alert_rec"]),
          "difference": point["P_B_rec"]["auc"] - point["P_A_rec"]["auc"], "ci_low": low, "ci_high": high,
          "share_resamples_delta_le_0": float(np.mean(delta[finite] <= 0)), "non_finite_draws": int((~finite).sum()),
          "n_boot": int(n_boot), "seed": int(seed), "groups": "recording_id",
          "method": "lamaakal_metrics.group_bootstrap (paired), lamaakal_metrics.percentile_ci (2.5th, 97.5th)",
          "pass": bool(low > 0)}
    verdict = "pass" if (g1["pass"] and g2["pass"] and g3["pass"]) else "fail"
    gate_result = {"n_S": int(len(m)), "threshold": threshold, "truth": "M (median in [91, 95)) on S (median >= 91)",
                   "G1": g1, "G2": g2, "G3": g3, "verdict": verdict,
                   "rule": "pass = G1 and G2 and G3; G1/G3 recall >= 0.80 and specificity >= 0.50 (state rule at 0.50); "
                           "G2 lower bound of the paired recording-bootstrap 95% CI of AUC(P_B_rec) - AUC(P_A_rec) > 0"}
    return gate_result, draws


def gate(m, p_a_rec, p_b_rec, p_a_pat, p_b_pat, rec_groups, n_boot=N_BOOT, seed=BOOT_SEED_REC, threshold=THRESHOLD):
    """Pre-registered ship gate on S (truth = M). Arrays are already restricted to S."""
    return _gate(m, p_a_rec, p_b_rec, p_a_pat, p_b_pat, rec_groups, n_boot, seed, threshold)[0]


def check_label_swap(frame, copy, expected_positive=Y95_POSITIVE):
    """The label-swap copy differs from the frame only in `label`, which is median < 95 (int64)."""
    if list(copy.columns) != list(frame.columns) or not copy.index.equals(frame.index):
        raise AssertionError("label-swap copy changed the column order or the index")
    if not copy.drop(columns="label").equals(frame.drop(columns="label")):
        raise AssertionError("label-swap copy changed a column other than label")
    if copy["label"].dtype != np.int64:
        raise AssertionError("label-swap label is not int64")
    if not np.array_equal(copy["label"].to_numpy(), y95(frame["median_spo2"].to_numpy())):
        raise AssertionError("label-swap label is not median_spo2 < 95")
    if expected_positive is not None and int(copy["label"].sum()) != expected_positive:
        raise AssertionError(f"label-swap label sums to {int(copy['label'].sum())}, expected {expected_positive}")
    if not (frame["label"].to_numpy() == (frame["median_spo2"].to_numpy() < 90).astype(int)).all():
        raise AssertionError("original label is not median_spo2 < 90 on every row")


def label_swap_copy(frame, expected_positive=Y95_POSITIVE):
    copy = frame.copy()
    copy["label"] = (copy["median_spo2"] < 95).astype(np.int64)
    check_label_swap(frame, copy, expected_positive)
    return copy


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


def protected_hashes():
    return {p.relative_to(ROOT).as_posix(): tree_hashes(p) for p in PROTECTED}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def new_file(out, name):
    path = out / name
    if path.exists():
        raise FileExistsError(f"{path} exists; this script never overwrites an output file")
    return path


def write_csv(frame, out, name):
    frame.to_csv(new_file(out, name), index=False, float_format="%.8g", lineterminator="\n")


def write_gzip_csv(frame, out, name):
    """Full float64 precision (%.17g), so the gate can be recomputed exactly from the file; gzip mtime 0."""
    path = new_file(out, name)
    text = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(text)


def write_json(payload, out, name):
    new_file(out, name).write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def environment():
    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def _git(*args):
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def guard_inputs(features):
    leaked = sorted(set(features) & set(FORBIDDEN_INPUTS))
    if leaked:
        raise AssertionError(f"Label leakage: {leaked} used as model inputs")
    return tuple(features)


# ----------------------------------------------------------------------------- frozen inputs

def regenerate_folds(groups):
    """GroupKFold(5) as train_tinykan_v2.out_of_fold numbers it (1..5 in split order)."""
    from sklearn.model_selection import GroupKFold
    folds = np.zeros(len(groups), dtype=int)
    for fold, (_, test) in enumerate(GroupKFold(n_splits=tk2.N_SPLITS).split(np.zeros((len(groups), 1)), groups=groups), start=1):
        folds[test] = fold
    return folds


def load():
    """Read and assert every frozen input; StopAnalysis on the first failure."""
    v2 = pd.read_csv(P["v2_features"])
    v2["recording_id"] = v2["recording_id"].astype(str)
    if len(v2) != N_ROWS or v2["recording_id"].nunique() != N_RECORDINGS or v2["patient_id"].nunique() != N_PATIENTS:
        raise StopAnalysis("v2 table is not the 48,662-window, 154-recording, 87-patient export")
    s = v2["median_spo2"].to_numpy(float)
    if not (v2["label"].to_numpy() == (s < 90).astype(int)).all():
        raise StopAnalysis("label == (median_spo2 < 90) does not hold on every row")
    if round(float(s.min()), 2) != 3.96 or float(s.max()) != 100.0:
        raise StopAnalysis(f"median_spo2 range {s.min()}..{s.max()} is not 3.96..100.0")
    if int(((s >= 89) & (s < 91)).sum()) != 0:
        raise StopAnalysis("windows with median in [89, 91) exist")
    band = bands(s)
    band_counts = {}
    for name, expected in BAND_EXPECTED.items():
        rows = band == name
        got = (int(rows.sum()), int(v2.loc[rows, "recording_id"].nunique()), int(v2.loc[rows, "patient_id"].nunique()))
        band_counts[name] = dict(zip(("windows", "recordings", "patients"), got))
        if got != expected:
            raise StopAnalysis(f"band {name}: {got} != pre-registered {expected}")
    y = y95(s)
    if int(y.sum()) != Y95_POSITIVE:
        raise StopAnalysis(f"y95 sums to {int(y.sum())}, not {Y95_POSITIVE}")
    S = s >= 91
    if int(S.sum()) != S_SIZE or not np.array_equal(y[S], (band[S] == "M").astype(np.int64)):
        raise StopAnalysis("S is not 19,867 windows with y95 == M")
    drop = in_94_96(s)
    if (int(drop.sum()), int(((band == "M") & ~drop).sum()), int(((band == "N") & ~drop).sum())) != (DROP_94_96, S_KEEP_M, S_KEEP_N):
        raise StopAnalysis("[94, 96] sensitivity counts differ from the pre-registration")

    oof = {}
    expected_columns = ["row_index", "recording_id", "patient_id", "start_time_seconds", "fold", "y_true", "y_prob"]
    for key in ("head_a_recording_oof", "head_a_patient_oof"):
        frame = pd.read_csv(P[key])
        if list(frame.columns) != expected_columns:
            raise StopAnalysis(f"{key} columns {list(frame.columns)} differ from {expected_columns}")
        frame = frame.sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(frame["row_index"].to_numpy(), np.arange(N_ROWS)):
            raise StopAnalysis(f"{key}: row_index is not 0..48661")
        if not (frame["recording_id"].astype(str).to_numpy() == v2["recording_id"].to_numpy()).all():
            raise StopAnalysis(f"{key}: recording_id differs from the v2 table")
        if not (frame["y_true"].to_numpy() == v2["label"].to_numpy()).all():
            raise StopAnalysis(f"{key}: y_true differs from label")
        oof[key] = frame
    fold_rec = oof["head_a_recording_oof"]["fold"].to_numpy(int)
    fold_pat = oof["head_a_patient_oof"]["fold"].to_numpy(int)
    rec_ids, pat_ids = v2["recording_id"].to_numpy(), v2["patient_id"].to_numpy()
    if pd.Series(fold_rec).groupby(rec_ids).nunique().max() != 1:
        raise StopAnalysis("a recording spans two recording folds")
    if pd.Series(fold_pat).groupby(pat_ids).nunique().max() != 1:
        raise StopAnalysis("a patient spans two patient folds")
    regen = {"recording": bool(np.array_equal(regenerate_folds(rec_ids), fold_rec)),
             "patient": bool(np.array_equal(regenerate_folds(pat_ids), fold_pat))}
    if not all(regen.values()):
        raise StopAnalysis(f"GroupKFold(5) does not regenerate the stored folds: {regen}")
    fold_counts = {}
    for grouping, folds in (("recording", fold_rec), ("patient", fold_pat)):
        got = [(int(((folds == f) & S & (band == "M")).sum()), int(((folds == f) & S & (band == "N")).sum())) for f in range(1, 6)]
        fold_counts[grouping] = got
        if got != FOLD_COUNTS[grouping]:
            raise StopAnalysis(f"{grouping} fold (M, N) counts on S {got} != {FOLD_COUNTS[grouping]}")

    p_a_rec = oof["head_a_recording_oof"]["y_prob"].to_numpy(float)
    p_a_pat = oof["head_a_patient_oof"]["y_prob"].to_numpy(float)
    label = v2["label"].to_numpy(int)
    head_a = {}
    for grouping, p in (("recording", p_a_rec), ("patient", p_a_pat)):
        mtr = lm.binary_metrics(label, p, THRESHOLD)
        got = {"recall": mtr["recall"], "specificity": mtr["specificity"], "auc": mtr["auc"]}
        exp = HEAD_A_EXPECTED[grouping]
        ok = (round(got["recall"], 5) == exp["recall"] and round(got["specificity"], 5) == exp["specificity"]
              and round(got["auc"], 6) == exp["auc"])
        head_a[grouping] = {**got, "expected": exp, "reproduces": bool(ok),
                            "comparison": "recall/specificity to 5 decimals, AUC to 6, label at 0.50, stored vector"}
        if not ok:
            raise StopAnalysis(f"head A stored {grouping} vector does not reproduce {exp}: {got}")

    checks = {"rows": N_ROWS, "recordings": N_RECORDINGS, "patients": N_PATIENTS,
              "label_is_median_below_90_every_row": True, "median_range": [float(s.min()), float(s.max())],
              "bands": band_counts, "band_89_91_windows": 0, "y95_positive": int(y.sum()), "S_windows": int(S.sum()),
              "y95_equals_M_on_S": True,
              "sensitivity_94_96": {"dropped": int(drop.sum()), "S_keeps_M": S_KEEP_M, "S_keeps_N": S_KEEP_N},
              "stored_vectors": "row_index 0..48661 after sort, recording_id equals the table, y_true == label (both files)",
              "no_recording_spans_two_recording_folds": True, "no_patient_spans_two_patient_folds": True,
              "groupkfold_regenerates_stored_folds": regen, "fold_M_N_on_S": fold_counts, "head_a_stored": head_a}
    return {"v2": v2, "median": s, "band": band, "y95": y, "S": S, "drop": drop, "fold_rec": fold_rec, "fold_pat": fold_pat,
            "p_a_rec": p_a_rec, "p_a_pat": p_a_pat, "label": label, "rec_ids": rec_ids, "pat_ids": pat_ids,
            "x": v2.loc[:, list(guard_inputs(V2_FEATURES))].to_numpy(np.float32), "checks": checks}


# ----------------------------------------------------------------------------- jobs (one fresh process each)

def kan_params(model):
    state = model.state_dict()
    return {"w1": state["layers.0.W"].numpy(), "b1": state["layers.0.b"].numpy(), "theta1": state["layers.0.theta"].numpy(),
            "centers1": model.layers[0].spline.centers.numpy(), "step1": np.float32(model.layers[0].spline.step.item()),
            "w2": state["layers.1.W"].numpy(), "b2": state["layers.1.b"].numpy(), "theta2": state["layers.1.theta"].numpy(),
            "centers2": model.layers[1].spline.centers.numpy(), "step2": np.float32(model.layers[1].spline.step.item()),
            "parameter_count": np.int64(sum(p.numel() for p in model.parameters()))}


def _read_v2():
    frame = pd.read_csv(tk2.V2_CSV)
    frame["recording_id"] = frame["recording_id"].astype(str)
    return frame


def _repro_job(spec):
    """analyze_physio_concept._repro_job: out_of_fold on the original label, recording-grouped."""
    guard_inputs(tk2.V2_FEATURES)
    p, folds = tk2.out_of_fold(_read_v2(), tk2.V2_FEATURES, "recording_id", "tinykan", spec["epochs"])
    return {"probabilities": p, "folds": folds}


def _head_b_job(spec):
    """Head B: out_of_fold on the label-swap copy (label = median < 95); seed 42 via train_kan."""
    guard_inputs(tk2.V2_FEATURES)
    frame = _read_v2()
    copy = label_swap_copy(frame)
    p, folds = tk2.out_of_fold(copy, tk2.V2_FEATURES, spec["group_column"], "tinykan", spec["epochs"])
    return {"probabilities": p, "folds": folds}


def _cv_job(spec, x, y, folds):
    result = models.run_cv_job(spec, x, y, folds)
    return {"probabilities": np.asarray(result["probabilities"], dtype=np.float64),
            "fold_sizes": np.asarray(result["fold_sizes"], dtype=float)}


def _final_job(spec):
    """train_tinykan_v2.run_final's training (scaler on all rows, train_kan, seed 42) with y95."""
    import torch
    from sklearn.preprocessing import StandardScaler

    guard_inputs(tk2.V2_FEATURES)
    copy = label_swap_copy(_read_v2())
    x_raw = copy.loc[:, tk2.V2_FEATURES].to_numpy(dtype=np.float32)
    y = copy["label"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)
    model = tk2.train_kan(x_scaled, y, spec["epochs"])
    with torch.no_grad():
        logit = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    return {"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_, "torch_logit": logit, **kan_params(model)}


JOB_KINDS = {"repro": _repro_job, "head_b": _head_b_job, "cv": _cv_job, "final": _final_job}


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


def run_jobs_cached(payloads, workers, cache_dir, train=True):
    """Each job in a freshly spawned process (max_tasks_per_child=1); finished jobs cached by key.

    train=False only reads the cache and stops if a job is missing (used by summary).
    """
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
    if pending and not train:
        raise StopAnalysis(f"jobs not in the cache {cache_dir}: {[job[1]['name'] for job in pending]}; run their stage first")
    if pending:
        weight = {"repro": 9, "head_b": 9, "final": 5}
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


def tinykan_spec(name, seed, epochs):
    return {"name": name, "learner": "tinykan", "hidden": tk2.HIDDEN_DIM, "knots": tk2.N_SPLINE_KNOTS,
            "spline_range": [float(v) for v in tk2.SPLINE_RANGE], "epochs": int(epochs), "seed": int(seed)}


def payloads_for(stage, data, args):
    sha_v2 = data["sha_v2"]
    x, y, S = data["x"], data["y95"], data["S"]
    if stage == "repro":
        return [("repro", {"name": "repro_label90_recording_seed42", "epochs": args.epochs, "input_sha256": sha_v2}, ())]
    if stage == "cv":
        return [("head_b", {"name": f"B_y95_{g}_seed42", "group_column": col, "epochs": args.epochs, "input_sha256": sha_v2}, ())
                for g, col in (("recording", "recording_id"), ("patient", "patient_id"))]
    if stage == "seeds":
        return [("cv", tinykan_spec(f"B_y95_recording_runcv_seed{s}", s, args.epochs), (x, y, data["fold_rec"])) for s in SEEDS]
    if stage == "secondary":
        jobs = [("cv", tinykan_spec(f"Bcond_S_{g}_seed42", 42, args.epochs), (x[S], y[S], folds[S]))
                for g, folds in (("recording", data["fold_rec"]), ("patient", data["fold_pat"]))]
        jobs += [("cv", {"name": f"{learner}_y95_recording", "learner": learner, "settings": models.BASELINE_SETTINGS[learner]},
                  (x, y, data["fold_rec"])) for learner in ("logreg", "hist_gb")]
        return jobs
    if stage == "final":
        return [("final", {"name": "final_B_y95_all_seed42", "epochs": args.epochs, "input_sha256": sha_v2}, ())]
    raise ValueError(stage)


# ----------------------------------------------------------------------------- stages

def repro_check(data, result, args):
    ref, folds_ref, label = data["p_a_rec"], data["fold_rec"], data["label"]
    p = result["probabilities"]
    diff = np.abs(p - ref)
    check = {
        "reference": "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
        "path": "train_tinykan_v2.out_of_fold(v2, V2_FEATURES, 'recording_id', 'tinykan') in a fresh process, original label",
        "epochs": args.epochs,
        "folds_equal_stored_recording_folds": bool(np.array_equal(result["folds"].astype(int), folds_ref)),
        "reference_auc": lm.weighted_auc(label, ref), "retrain_auc": lm.weighted_auc(label, p),
        "max_abs_probability_diff": float(diff.max()),
        "max_abs_probability_diff_per_fold": {int(f): float(diff[folds_ref == f].max()) for f in range(1, 6)},
        "bitwise_identical": bool(np.array_equal(p, ref)),
        "candidate_flag_disagreements_at_0_50": int(np.sum((p >= THRESHOLD) != (ref >= THRESHOLD))),
        "tolerance_auc": REPRO_AUC_TOLERANCE, "seconds": float(result["seconds"]),
    }
    check["auc_difference"] = check["retrain_auc"] - check["reference_auc"]
    if args.smoke:
        check["auc_check"] = f"not applicable (smoke run, epochs={args.epochs})"
        check["reproduces"] = check["folds_equal_stored_recording_folds"]
    else:
        check["auc_check"] = "pass" if abs(check["auc_difference"]) <= REPRO_AUC_TOLERANCE else "fail"
        check["reproduces"] = bool(check["folds_equal_stored_recording_folds"] and check["auc_check"] == "pass")
    return check


def require_repro(data, args, train=False):
    payloads = payloads_for("repro", data, args)
    result = run_jobs_cached(payloads, 1, args.cache, train=train)[payloads[0][1]["name"]]
    check = repro_check(data, result, args)
    if not check["reproduces"]:
        print(json.dumps(check, indent=1), flush=True)
        raise StopAnalysis("reproduction check failed; the run stops (no other stage may run)")
    return check


def stage_repro(data, args):
    print("Stage repro: v2-16 TinyKAN, original label, through train_tinykan_v2.out_of_fold ...", flush=True)
    check = require_repro(data, args, train=True)
    print(json.dumps(check, indent=1), flush=True)
    return 0


def head_b_vectors(data, results):
    out, checks = {}, {}
    for grouping, stored in (("recording", data["fold_rec"]), ("patient", data["fold_pat"])):
        r = results[f"B_y95_{grouping}_seed42"]
        equal = bool(np.array_equal(r["folds"].astype(int), stored))
        checks[grouping] = {"folds_equal_stored": equal, "seconds": float(r["seconds"])}
        if not equal:
            raise StopAnalysis(f"head B {grouping} folds differ from the stored {grouping} fold column")
        out[grouping] = r["probabilities"]
    return out, checks


def stage_cv(data, args):
    require_repro(data, args)
    print("Stage cv: head B (median < 95) through out_of_fold, recording and patient ...", flush=True)
    results = run_jobs_cached(payloads_for("cv", data, args), args.jobs, args.cache)
    vectors, _ = head_b_vectors(data, results)
    S, m = data["S"], (data["band"][data["S"]] == "M").astype(int)
    for grouping, p in vectors.items():
        print(f"  P_B {grouping}: AUC on S {lm.weighted_auc(m, p[S]):.4f}, all vs y95 {lm.weighted_auc(data['y95'], p):.4f}", flush=True)
    return 0


def stage_seeds(data, args):
    require_repro(data, args)
    print("Stage seeds: head B, seeds 42, 1-5, lamaakal_models.run_cv_job, recording folds ...", flush=True)
    results = run_jobs_cached(payloads_for("seeds", data, args), args.jobs, args.cache)
    S, m = data["S"], (data["band"][data["S"]] == "M").astype(int)
    for s in SEEDS:
        print(f"  seed {s}: AUC on S {lm.weighted_auc(m, results[f'B_y95_recording_runcv_seed{s}']['probabilities'][S]):.4f}", flush=True)
    return 0


def stage_secondary(data, args):
    require_repro(data, args)
    print("Stage secondary: B_cond on S (recording, patient), LogReg-16 and HistGB-16 on y95 ...", flush=True)
    results = run_jobs_cached(payloads_for("secondary", data, args), args.jobs, args.cache)
    for name, r in results.items():
        print(f"  {name}: {len(r['probabilities'])} windows, {float(r['seconds']):.1f} s", flush=True)
    return 0


# ----------------------------------------------------------------------------- summary

def _nan(n):
    return np.full(n, np.nan)


def score_row(score, model, grouping, subset, truth, y, p, ci=None, ci_seed=None, p_a=None, source=""):
    m = lm.binary_metrics(y, p, THRESHOLD)
    row = {"score": score, "model": model, "grouping": grouping, "subset": subset, "truth": truth, "n": m["n"],
           "n_pos": int(np.sum(y == 1)), "prevalence": m["prevalence"], "auc": m["auc"],
           "auc_ci_low": ci[0] if ci else np.nan, "auc_ci_high": ci[1] if ci else np.nan, "bootstrap_seed": ci_seed,
           "recall_score_ge_0_50": m["recall"], "specificity_score_ge_0_50": m["specificity"],
           "state_rule_recall": np.nan, "state_rule_specificity": np.nan, "state_rule_false_watch_N": np.nan,
           "state_rule_low_share_N": np.nan, "source": source}
    if p_a is not None:
        rb = rule_block(y, p_a, p)
        row.update({"state_rule_recall": rb["recall"], "state_rule_specificity": rb["specificity"],
                    "state_rule_false_watch_N": rb["false_watch_rate_N"], "state_rule_low_share_N": rb["low_share_N"]})
    return row


def confusion_rows(band, p_a, p_b, grouping):
    states = state_rule(p_a, p_b)
    rows = []
    for b in ("L", "M", "N"):
        sel = band == b
        n = int(sel.sum())
        counts = {st: int((states[sel] == st).sum()) for st in ("LOW", "WATCH", "NONE")}
        rows.append({"grouping": grouping, "true_band": b, "n": n, **counts,
                     **{f"share_{st}": counts[st] / n for st in ("LOW", "WATCH", "NONE")}})
    if sum(r["n"] for r in rows) != N_ROWS:
        raise StopAnalysis("L + M + N != 48,662")
    return rows


def sensitivity_rows(full, reduced):
    rows = []
    spec = [("G1_recall", "G1", "recall", ">= 0.80", "recall_pass"), ("G1_specificity", "G1", "specificity", ">= 0.50", "specificity_pass"),
            ("G2_difference", "G2", "difference", "reported", None), ("G2_ci_low", "G2", "ci_low", "> 0", "pass"),
            ("G2_ci_high", "G2", "ci_high", "reported", None),
            ("G2_share_resamples_delta_le_0", "G2", "share_resamples_delta_le_0", "reported", None),
            ("G2_non_finite_draws", "G2", "non_finite_draws", "reported", None),
            ("G3_recall", "G3", "recall", ">= 0.80", "recall_pass"), ("G3_specificity", "G3", "specificity", ">= 0.50", "specificity_pass"),
            ("n_M", "G1", "n_M", "reported", None), ("n_N", "G1", "n_N", "reported", None)]
    for name, block, key, rule, flag in spec:
        rows.append({"quantity": name, "rule": rule, "all_S": full[block][key], "excluding_94_96": reduced[block][key],
                     "passes_all_S": full[block][flag] if flag else None,
                     "passes_excluding_94_96": reduced[block][flag] if flag else None})
    rows.append({"quantity": "verdict", "rule": "G1 and G2 and G3", "all_S": full["verdict"], "excluding_94_96": reduced["verdict"],
                 "passes_all_S": full["verdict"] == "pass", "passes_excluding_94_96": reduced["verdict"] == "pass"})
    return rows


def stage_summary(data, args):
    out = args.out
    existing = [n for n in OUTPUT_FILES if (out / n).exists()]
    if existing:
        raise FileExistsError(f"{out} already holds {existing}; this script never overwrites an output file")
    repro = require_repro(data, args)
    payloads = sum((payloads_for(st, data, args) for st in ("cv", "seeds", "secondary")), [])
    results = run_jobs_cached(payloads, 1, args.cache, train=False)
    print("Stage summary: computing the gate and the reported tables from the cached jobs ...", flush=True)

    vectors, cv_checks = head_b_vectors(data, results)
    p_b_rec, p_b_pat = vectors["recording"], vectors["patient"]
    p_a_rec, p_a_pat = data["p_a_rec"], data["p_a_pat"]
    S, band, y, drop = data["S"], data["band"], data["y95"], data["drop"]
    m = (band[S] == "M").astype(int)
    rec_S, pat_S = data["rec_ids"][S], data["pat_ids"][S]
    bcond = {}
    for grouping in ("recording", "patient"):
        full = _nan(N_ROWS)
        full[S] = results[f"Bcond_S_{grouping}_seed42"]["probabilities"]
        bcond[grouping] = full
    comps = {learner: results[f"{learner}_y95_recording"]["probabilities"] for learner in ("logreg", "hist_gb")}
    seeds = {s: results[f"B_y95_recording_runcv_seed{s}"]["probabilities"] for s in SEEDS}

    # Gate (exactly the pre-registered call) and the [94, 96] sensitivity (same seed, re-scoring only).
    gate_result, gate_draws = _gate(m, p_a_rec[S], p_b_rec[S], p_a_pat[S], p_b_pat[S], rec_S)
    keep = ~drop[S]
    sens = gate(m[keep], p_a_rec[S][keep], p_b_rec[S][keep], p_a_pat[S][keep], p_b_pat[S][keep], rec_S[keep])

    # Reported CIs: the same recording call (same seed, groups, n_boot) with more scores, so paired with G2.
    rec_scores = {"P_B_rec": p_b_rec[S], "P_A_rec": p_a_rec[S], "alert_rec": np.maximum(p_a_rec, p_b_rec)[S],
                  "B_cond_rec": bcond["recording"][S], "logreg_y95_rec": comps["logreg"][S], "hist_gb_y95_rec": comps["hist_gb"][S]}
    point_r, draws_r = lm.group_bootstrap(m, rec_scores, rec_S, N_BOOT, BOOT_SEED_REC, THRESHOLD)
    if not np.array_equal(draws_r["P_B_rec"]["auc"], gate_draws["P_B_rec"]["auc"], equal_nan=True):
        raise StopAnalysis("reported recording bootstrap is not paired with the G2 call")
    pat_scores = {"P_B_pat": p_b_pat[S], "P_A_pat": p_a_pat[S], "alert_pat": np.maximum(p_a_pat, p_b_pat)[S],
                  "B_cond_pat": bcond["patient"][S]}
    point_p, draws_p = lm.group_bootstrap(m, pat_scores, pat_S, N_BOOT, BOOT_SEED_PAT, THRESHOLD)
    ci = {k: lm.percentile_ci(v["auc"]) for k, v in {**draws_r, **draws_p}.items()}

    paired_rows = []
    for grouping, draws, point, seed, names in (
            ("recording", draws_r, point_r, BOOT_SEED_REC, [("P_B_rec", "P_A_rec"), ("hist_gb_y95_rec", "P_B_rec"),
                                                             ("logreg_y95_rec", "P_B_rec"), ("B_cond_rec", "P_B_rec")]),
            ("patient", draws_p, point_p, BOOT_SEED_PAT, [("P_B_pat", "P_A_pat"), ("B_cond_pat", "P_B_pat")])):
        for a, b in names:
            delta = draws[a]["auc"] - draws[b]["auc"]
            lo, hi = lm.percentile_ci(delta)
            paired_rows.append({"grouping": grouping, "comparison": f"{a} - {b}", "difference": point[a]["auc"] - point[b]["auc"],
                                "ci_low": lo, "ci_high": hi, "share_le_0": float(np.mean(delta[np.isfinite(delta)] <= 0)),
                                "non_finite": int((~np.isfinite(delta)).sum()), "seed": seed, "gated": a == "P_B_rec" and b == "P_A_rec"})

    # Seeds and the same-path check.
    sp_diff = np.abs(seeds[42] - p_b_rec)
    same_path = {"compared": "seed 42 via lamaakal_models.run_cv_job vs primary P_B_rec via train_tinykan_v2.out_of_fold",
                 "bitwise_identical": bool(np.array_equal(seeds[42], p_b_rec)), "max_abs_diff": float(sp_diff.max()),
                 "max_abs_diff_per_fold": {int(f): float(sp_diff[data["fold_rec"] == f].max()) for f in range(1, 6)},
                 "auc_on_S_difference": lm.weighted_auc(m, seeds[42][S]) - lm.weighted_auc(m, p_b_rec[S]),
                 "tolerance_max_abs_diff": SAME_PATH_TOLERANCE}
    same_path["within_tolerance"] = bool(same_path["max_abs_diff"] <= SAME_PATH_TOLERANCE)
    seed_note = "same path as the primary head (within tolerance)" if same_path["within_tolerance"] else \
        "not the same path as the primary head; the spread is not head B's spread"
    seed_rows = []
    for s in SEEDS:
        rb = rule_block(m, p_a_rec[S], seeds[s][S])
        seed_rows.append({"seed": s, "grouping": "recording", "path": "lamaakal_models.run_cv_job",
                          "auc_S": lm.weighted_auc(m, seeds[s][S]), "auc_all_y95": lm.weighted_auc(y, seeds[s]),
                          "g1_recall": rb["recall"], "g1_specificity": rb["specificity"],
                          "false_watch_rate_N": rb["false_watch_rate_N"], "note": seed_note})
    seeds_table = pd.DataFrame(seed_rows)
    seed_spread = {k: dict(zip(("mean", "sd"), lm.mean_sd(seeds_table[k]))) for k in ("auc_S", "g1_recall", "g1_specificity")}

    # metrics.csv
    S_sub, S_truth = "S (median >= 91)", "M (median in [91, 95))"
    all_sub, y95_truth, l90_truth = "all windows", "y95 (median < 95)", "label (median < 90)"
    rows = []
    for grouping, pa, tag in (("recording", p_a_rec, "rec"), ("patient", p_a_pat, "pat")):
        rows.append(score_row(f"P_A_{tag}", "head A (shipped, stored OOF)", grouping, all_sub, l90_truth, data["label"], pa,
                              source="stored vector; reproduction of the shipped head"))
        rows.append(score_row(f"P_A_{tag}", "head A used as a score", grouping, S_sub, S_truth, m, pa[S], ci[f"P_A_{tag}"],
                              BOOT_SEED_REC if tag == "rec" else BOOT_SEED_PAT, source="stored vector"))
    for grouping, pa, pb, tag in (("recording", p_a_rec, p_b_rec, "rec"), ("patient", p_a_pat, p_b_pat, "pat")):
        seed = BOOT_SEED_REC if tag == "rec" else BOOT_SEED_PAT
        rows.append(score_row(f"P_B_{tag}", "head B TinyKAN y95 seed 42", grouping, S_sub, S_truth, m, pb[S], ci[f"P_B_{tag}"], seed,
                              p_a=pa[S], source="train_tinykan_v2.out_of_fold on the label-swap copy"))
        rows.append(score_row(f"P_B_{tag}", "head B TinyKAN y95 seed 42", grouping, all_sub, y95_truth, y, pb,
                              source="train_tinykan_v2.out_of_fold on the label-swap copy"))
        rows.append(score_row(f"alert_{tag}", "max(P_A, P_B) = state rule", grouping, S_sub, S_truth, m, np.maximum(pa, pb)[S],
                              ci[f"alert_{tag}"], seed, p_a=pa[S], source="alerted <=> max(P_A, P_B) >= 0.50"))
        rows.append(score_row(f"B_cond_{tag}", "B_cond TinyKAN on S, seed 42 (reported, never gated)", grouping, S_sub, S_truth, m,
                              bcond[grouping][S], ci[f"B_cond_{tag}"], seed, p_a=pa[S],
                              source="lamaakal_models.run_cv_job on S, stored folds restricted to S"))
    for learner, label_name in (("logreg", "LogReg-16"), ("hist_gb", "HistGB-16")):
        rows.append(score_row(f"{learner}_y95_rec", f"{label_name} y95 (reported, never gated)", "recording", S_sub, S_truth, m,
                              comps[learner][S], ci[f"{learner}_y95_rec"], BOOT_SEED_REC, p_a=p_a_rec[S],
                              source=f"lamaakal_models.run_cv_job; {models.BASELINE_SETTINGS[learner]}"))
        rows.append(score_row(f"{learner}_y95_rec", f"{label_name} y95 (reported, never gated)", "recording", all_sub, y95_truth, y,
                              comps[learner], source=f"lamaakal_models.run_cv_job; {models.BASELINE_SETTINGS[learner]}"))
    metrics_table = pd.DataFrame(rows)

    confusion = pd.DataFrame(confusion_rows(band, p_a_rec, p_b_rec, "recording") + confusion_rows(band, p_a_pat, p_b_pat, "patient"))
    sensitivity = pd.DataFrame(sensitivity_rows(gate_result, sens))
    oof = pd.DataFrame({"row_index": np.arange(N_ROWS), "recording_id": data["rec_ids"], "patient_id": data["pat_ids"],
                        "median_spo2": data["median"], "band": band, "P_A_rec": p_a_rec, "P_A_pat": p_a_pat,
                        "P_B_rec": p_b_rec, "P_B_pat": p_b_pat, "B_cond_rec": bcond["recording"], "B_cond_pat": bcond["patient"],
                        "fold_rec": data["fold_rec"], "fold_pat": data["fold_pat"]})

    run_meta = {"smoke": args.smoke, "epochs": args.epochs}
    gate_payload = {**run_meta, **gate_result, "excluding_94_96_verdict": sens["verdict"]}
    false_watch = {grouping: {k: gate_result[g][k] for k in ("false_watch_rate_N", "low_share_N", "N_watch", "N_low", "n_N")}
                   for grouping, g in (("recording", "G1"), ("patient", "G3"))}
    readings = {
        "verdict": gate_result["verdict"],
        "G1": {k: gate_result["G1"][k] for k in ("recall", "specificity", "pass")},
        "G2": {k: gate_result["G2"][k] for k in ("difference", "ci_low", "ci_high", "share_resamples_delta_le_0", "non_finite_draws", "pass")},
        "G3": {k: gate_result["G3"][k] for k in ("recall", "specificity", "pass")},
        "excluding_94_96": {"verdict": sens["verdict"], "G1": [sens["G1"]["recall"], sens["G1"]["specificity"]],
                            "G2": [sens["G2"]["difference"], sens["G2"]["ci_low"], sens["G2"]["ci_high"]],
                            "G3": [sens["G3"]["recall"], sens["G3"]["specificity"]]},
        "false_watch_on_N": false_watch,
        "p_b_auc_on_S": {"recording": [point_r["P_B_rec"]["auc"], *ci["P_B_rec"]], "patient": [point_p["P_B_pat"]["auc"], *ci["P_B_pat"]]},
        "p_b_auc_all_y95": {"recording": lm.weighted_auc(y, p_b_rec), "patient": lm.weighted_auc(y, p_b_pat)},
        "seed_spread": seed_spread,
    }
    checks = {"inputs": data["checks"], "reproduction": repro, "head_b_folds": cv_checks, "same_path": same_path,
              "label_swap": "asserted inside every head B job and the final job: copy differs only in label; label sums to 36,580; "
                            "original label == median < 90 on every row",
              "reported_recording_bootstrap_paired_with_G2": True, "paired_auc_differences": paired_rows}

    out.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(oof, out, "oof_watch.csv.gz")
    write_csv(confusion, out, "confusion_3x3.csv")
    write_csv(metrics_table, out, "metrics.csv")
    write_csv(seeds_table, out, "seeds.csv")
    write_csv(sensitivity, out, "sensitivity_94_96.csv")
    write_json(gate_payload, out, "gate.json")
    prereg_rel = PREREG.relative_to(ROOT).as_posix()
    first = _git("log", "--diff-filter=A", "--format=%H %cI", "--", prereg_rel)
    payload = {
        "study": "below-95 Watch head beside the shipped below-90 head (host only, no board, final report only)",
        **run_meta,
        "smoke_note": f"SMOKE RUN (epochs={args.epochs}): tests the code only; not a result" if args.smoke else None,
        "preregistration": {"path": prereg_rel, "first_commit": first.split()[0] if first else None,
                            "first_commit_date": first.split()[1] if first else None,
                            "commits_touching": _git("log", "--format=%H %cI", "--", prereg_rel),
                            "worktree_unmodified": _git("status", "--porcelain", "--", prereg_rel) == "",
                            "sha256": sha256(PREREG)},
        "readings": readings,
        "gate": gate_result,
        "seeds": {"head_b_primary": 42, "head_b_seeds": list(SEEDS), "bootstrap_recording": BOOT_SEED_REC,
                  "bootstrap_patient": BOOT_SEED_PAT, "n_boot": N_BOOT, "baseline_random_state": models.BASELINE_SEED},
        "recipe": {"features": list(V2_FEATURES), "hidden": tk2.HIDDEN_DIM, "knots": tk2.N_SPLINE_KNOTS, "degree": tk2.SPLINE_DEGREE,
                   "spline_range": list(tk2.SPLINE_RANGE), "lr": tk2.LEARNING_RATE, "weight_decay": tk2.WEIGHT_DECAY,
                   "epochs": args.epochs, "threshold": THRESHOLD, "baselines": {k: models.BASELINE_SETTINGS[k] for k in ("logreg", "hist_gb")}},
        "checks": checks,
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for k, p in P.items()},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in CODE_FILES},
        "outputs_sha256": {n: sha256(out / n) for n in OUTPUT_FILES if (out / n).is_file()},
        "job_seconds": {k: float(v["seconds"]) for k, v in sorted(results.items())},
        "job_cache": args.cache.relative_to(ROOT).as_posix() if args.cache.is_relative_to(ROOT) else str(args.cache),
        "git_head_at_summary": _git("rev-parse", "HEAD"),
        "environment": environment(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(payload, out, "summary.json")
    new_file(out, "README.md").write_text(readme(payload, gate_result, sens, confusion, metrics_table, seeds_table, seed_spread,
                                                 same_path, repro, paired_rows), encoding="utf-8")
    print(json.dumps(readings, indent=1, default=_json_default), flush=True)
    return 0


# ----------------------------------------------------------------------------- README

def _n(x, nd=3):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def readme(payload, g, sens, confusion, metrics_table, seeds_table, seed_spread, same_path, repro, paired_rows):
    g1, g2, g3 = g["G1"], g["G2"], g["G3"]
    ok = lambda b: "pass" if b else "fail"  # noqa: E731
    pre = payload["preregistration"]
    L = ["# Below-95 Watch head beside the shipped below-90 head (host only)", ""]
    if payload["smoke"]:
        L += [f"**SMOKE RUN (epochs = {payload['epochs']}). This tests the code only. Nothing here is a result.**", ""]
    L += [f"Pre-registration: `{pre['path']}`, first commit {pre['first_commit']} ({pre['first_commit_date']}), before any model in "
          "this study was trained. Host only, laptop, no board. Results go to the final report only; the IEEE paper is not edited. "
          "Generated by `scripts/analyze_watch_head.py --stage summary` from the cached jobs.", "",
          "Label: head B is trained on window-median ScalcO2 < 95 (ScalcO2 is SaO2 estimated from end-tidal O2; the CSV column is "
          "named `median_spo2`, but it is not a pulse-oximeter reading). The state it drives is called **Watch** (the below-95 pattern). "
          "It is not a saturation reading and not a diagnosis. Head A is the shipped below-90 head, read from its stored out-of-fold "
          "vectors and never retrained.", "",
          "State rule per window: LOW if P_A >= 0.50; else WATCH if P_B >= 0.50; else NONE. On S (median >= 91, n = "
          f"{g['n_S']:,}), recall = share of M (median in [91, 95)) alerted (WATCH or LOW); specificity = share of N (median >= 95) "
          "in NONE.", "",
          "## Pre-registered readings (ship gate, threshold 0.50 only)", "",
          _table(["Condition", "Rule (fixed in advance)", "Result", "Reading"], [
              ["G1 (recording-grouped)", "recall >= 0.80 and specificity >= 0.50, P_A_rec with P_B_rec",
               f"recall {g1['recall']:.4f}, specificity {g1['specificity']:.4f}", ok(g1["pass"])],
              ["G2 (paired recording bootstrap)", f"95% CI of AUC(P_B_rec) - AUC(P_A_rec) on S, {g2['n_boot']} resamples, "
               f"seed {g2['seed']}: lower bound > 0",
               f"{g2['difference']:+.4f} [{g2['ci_low']:+.4f}, {g2['ci_high']:+.4f}]; AUC P_B {g2['auc_P_B_rec']:.4f} vs P_A "
               f"{g2['auc_P_A_rec']:.4f}; share of resamples <= 0: {g2['share_resamples_delta_le_0']:.4f}; "
               f"non-finite draws {g2['non_finite_draws']}", ok(g2["pass"])],
              ["G3 (patient-grouped)", "recall >= 0.80 and specificity >= 0.50, P_A_pat with P_B_pat",
               f"recall {g3['recall']:.4f}, specificity {g3['specificity']:.4f}", ok(g3["pass"])],
              ["Verdict", "G1 and G2 and G3", "", f"**{g['verdict'].upper()}**"]]), ""]
    if g["verdict"] == "pass":
        L += ["Gate passed: `--stage final` may train head B on all 48,662 windows (seed 42). Any device image is a separate, "
              "host-only step; the board session waits for Jason.", ""]
    else:
        L += ["Gate failed: no final model and no firmware. The negative result goes to the final report.", ""]
    L += ["Gate quantities are float64 and compared unrounded; values above are rounded for display only (`gate.json` holds them in full).", "",
          "## States by true band (all 48,662 windows)", ""]
    for grouping in ("recording", "patient"):
        sub = confusion[confusion.grouping == grouping]
        L += [f"{grouping.capitalize()}-grouped (P_A_{grouping[:3]} with P_B_{grouping[:3]}):", "",
              _table(["True band", "n", "LOW", "WATCH", "NONE"],
                     [[r.true_band, f"{r.n:,}", f"{r.LOW:,} ({r.share_LOW:.3f})", f"{r.WATCH:,} ({r.share_WATCH:.3f})",
                       f"{r.NONE:,} ({r.share_NONE:.3f})"] for r in sub.itertuples()]), ""]
    L += ["False-Watch rate on N (WATCH share of N windows) with the LOW share beside it: "
          + "; ".join(f"{name} {b['false_watch_rate_N']:.4f} (LOW {b['low_share_N']:.4f})"
                      for name, b in (("recording", g1), ("patient", g3))) + ". Specificity on N = 1 - the two.", "",
          "## Scores (out-of-fold, threshold 0.50)", ""]
    disp = metrics_table.copy()
    L += [_table(["Score", "Model", "Grouping", "Subset / truth", "AUC [95% CI]", "Recall / spec. (score >= 0.50)",
                  "State rule recall / spec."],
                 [[r.score, r.model, r.grouping, f"{r.subset} / {r.truth}",
                   _n(r.auc, 4) + (f" [{_n(r.auc_ci_low, 4)}, {_n(r.auc_ci_high, 4)}]" if np.isfinite(r.auc_ci_low) else ""),
                   f"{_n(r.recall_score_ge_0_50)} / {_n(r.specificity_score_ge_0_50)}",
                   f"{_n(r.state_rule_recall)} / {_n(r.state_rule_specificity)}" if np.isfinite(r.state_rule_recall) else "-"]
                  for r in disp.itertuples()]), "",
          "Head A rows on all windows against `label` are the reproduction of the shipped head from the stored vectors "
          "(must equal recall 0.89269 / specificity 0.86802 recording, 0.89932 / 0.86354 patient). LogReg-16, HistGB-16 and "
          "B_cond are reported, never gated; in their state-rule columns the comparator's P replaces P_B. CIs: recording bootstrap "
          f"seed {BOOT_SEED_REC} (the G2 call, so paired), patient bootstrap seed {BOOT_SEED_PAT}, {N_BOOT} resamples.", "",
          _table(["Paired AUC difference on S", "Grouping", "Difference [95% CI]", "Share <= 0", "Gated"],
                 [[r["comparison"], r["grouping"], f"{r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                   f"{r['share_le_0']:.4f}", "G2" if r["gated"] else "no"] for r in paired_rows]), "",
          "## Seeds (recording-grouped, `lamaakal_models.run_cv_job`)", "",
          _table(["Seed", "AUC on S", "AUC all vs y95", "G1 recall", "G1 specificity", "False-Watch on N"],
                 [[r.seed, _n(r.auc_S, 4), _n(r.auc_all_y95, 4), _n(r.g1_recall, 4), _n(r.g1_specificity, 4),
                   _n(r.false_watch_rate_N, 4)] for r in seeds_table.itertuples()]), "",
          "Mean +- sd over six seeds: " + "; ".join(f"{k} {v['mean']:.4f} +- {v['sd']:.4f}" for k, v in seed_spread.items()) + ".", "",
          f"Same-path check (seed 42 here vs primary P_B_rec): max abs difference {same_path['max_abs_diff']:.3g} "
          f"(tolerance {SAME_PATH_TOLERANCE:g}), bitwise {same_path['bitwise_identical']}, AUC-on-S difference "
          f"{same_path['auc_on_S_difference']:+.2g}. "
          + ("Within tolerance: the seed spread describes head B." if same_path["within_tolerance"] else
             "**Not within tolerance: this is not the same path as the primary head, and the spread is not read as head B's spread.**"), "",
          "## Sensitivity: windows with median in [94, 96] removed (re-scoring only, no retraining)", "",
          _table(["Quantity", "All S", "Excluding [94, 96]"],
                 [["n M / n N", f"{g1['n_M']:,} / {g1['n_N']:,}", f"{sens['G1']['n_M']:,} / {sens['G1']['n_N']:,}"],
                  ["G1 recall / specificity", f"{g1['recall']:.4f} / {g1['specificity']:.4f}",
                   f"{sens['G1']['recall']:.4f} / {sens['G1']['specificity']:.4f}"],
                  ["G2 difference [CI]", f"{g2['difference']:+.4f} [{g2['ci_low']:+.4f}, {g2['ci_high']:+.4f}]",
                   f"{sens['G2']['difference']:+.4f} [{sens['G2']['ci_low']:+.4f}, {sens['G2']['ci_high']:+.4f}]"],
                  ["G3 recall / specificity", f"{g3['recall']:.4f} / {g3['specificity']:.4f}",
                   f"{sens['G3']['recall']:.4f} / {sens['G3']['specificity']:.4f}"],
                  ["Verdict", g["verdict"], sens["verdict"]]]), "",
          "The sensitivity does not change the verdict; it is the only check on the missing drop band at 95.", "",
          "## Checks and provenance", "",
          f"- Reproduction (stage `repro`): v2-16 TinyKAN on the original label retrained to AUC {repro['retrain_auc']:.6f} vs stored "
          f"{repro['reference_auc']:.6f} (difference {repro['auc_difference']:+.6f}, tolerance 0.002; AUC check: {repro['auc_check']}); "
          f"folds equal the stored folds: {repro['folds_equal_stored_recording_folds']}; bitwise {repro['bitwise_identical']}; max abs "
          f"difference {repro['max_abs_probability_diff']:.4g}; {repro['candidate_flag_disagreements_at_0_50']} flag flips at 0.50.",
          "- Frozen inputs (counts, bands, folds, head A reproduction) are asserted at the start of every stage; see `summary.json` `checks.inputs`.",
          "- The label-swap copy is asserted inside every head B job (only `label` changes; 36,580 positives).",
          "- `oof_watch.csv.gz` holds every vector at full float64 precision (`%.17g`).",
          "- Inputs, code, pre-registration and output SHA-256 are in `summary.json`.", "",
          "## Limits (fixed in advance)", "",
          "- ScalcO2 label noise against a 4-point band (concept regressor RMSE 6.04 on windows >= 90, bias -2.88).",
          "- No drop band at 95; the [94, 96] sensitivity is the only check on this.",
          "- Recording-grouped folds let a patient span folds (hence G3).",
          "- Oxygen cannot be varied live, so a live Watch at rest is a false alarm by assumption.",
          "- The scratch probe (HistGB / LogReg, P_A as a score) was seen before the pre-registration was written.",
          "- Any pass is window-level, on retrospective ScalcO2, not a live result.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- final

def stage_final(data, args):
    out = args.out
    gate_path = out / "gate.json"
    if not gate_path.is_file():
        raise StopAnalysis(f"{gate_path} does not exist; run --stage summary first")
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if bool(gate_payload.get("smoke")) != args.smoke or int(gate_payload.get("epochs", -1)) != args.epochs:
        raise StopAnalysis("gate.json was written by a run with a different --epochs / smoke setting")
    forced = False
    if gate_payload.get("verdict") != "pass":
        if not args.force_final_smoke:
            print(f"REFUSED: gate.json verdict is {gate_payload.get('verdict')!r}; no final model is trained and no file is written.")
            return 4
        forced = True
        print("Smoke only: --force-final-smoke trains the final model despite a failed gate, to exercise the writer.", flush=True)
    existing = [n for n in FINAL_FILES if (out / n).exists()]
    if existing:
        raise FileExistsError(f"{out} already holds {existing}")
    require_repro(data, args)
    payloads = payloads_for("final", data, args)
    result = run_jobs_cached(payloads, 1, args.cache)[payloads[0][1]["name"]]
    y = data["y95"]
    prob = 1.0 / (1.0 + np.exp(-result["torch_logit"]))
    params = {k: result[k] for k in ("scaler_mean", "scaler_std", "w1", "b1", "theta1", "centers1", "step1",
                                     "w2", "b2", "theta2", "centers2", "step2")}
    with new_file(out, "watch_final_model.npz").open("wb") as stream:
        np.savez(stream, **params)
    report = {
        "model": "head B (Watch, below-95 pattern), TinyKAN on the 16 v2 inputs, all 48,662 windows",
        "smoke": args.smoke, "epochs": args.epochs, "forced_despite_failed_gate_smoke_only": forced,
        "gate_json_sha256": sha256(gate_path), "gate_verdict": gate_payload.get("verdict"),
        "training": "train_tinykan_v2.run_final recipe (StandardScaler on all rows, train_tinykan_v2.train_kan, seed 42) with "
                    "label = median_spo2 < 95 (int64) on the label-swap copy",
        "features": list(V2_FEATURES), "kan_dims": [len(V2_FEATURES), tk2.HIDDEN_DIM, 1],
        "parameter_count": int(result["parameter_count"]), "seed": tk2.SEED, "threshold": THRESHOLD,
        "training_rows": N_ROWS, "training_prevalence_y95": float(y.mean()),
        "training_set_metrics_at_0_50": lm.binary_metrics(y, prob, THRESHOLD),
        "training_set_metrics_note": "resubstitution on the training rows; not a generalisation estimate",
        "npz_sha256": sha256(out / "watch_final_model.npz"),
        "npz_arrays": sorted(params),
        "scaler_mean": np.asarray(result["scaler_mean"]).tolist(), "scaler_std": np.asarray(result["scaler_std"]).tolist(),
        "inputs": {k: sha256(p) for k, p in P.items()},
        "seconds": float(result["seconds"]), "environment": environment(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "No header and no firmware are written here.",
    }
    write_json(report, out, "watch_final_model.json")
    print(json.dumps({k: report[k] for k in ("gate_verdict", "parameter_count", "training_set_metrics_at_0_50")}, indent=1,
                     default=_json_default), flush=True)
    return 0


# ----------------------------------------------------------------------------- main

STAGES = {"repro": stage_repro, "cv": stage_cv, "seeds": stage_seeds, "secondary": stage_secondary,
          "summary": stage_summary, "final": stage_final}
STAGE_ORDER = ("repro", "cv", "seeds", "secondary", "summary")


def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def configure(args):
    """Smoke guards: --epochs != 100 only into build/watch-head/, never outputs/ or the official cache."""
    args.smoke = args.epochs != EPOCHS
    args.out = _resolve(args.out) if args.out else OUT_DEFAULT.resolve()
    if args.cache:
        args.cache = _resolve(args.cache)
    else:
        args.cache = (SMOKE_CACHE_ROOT / f"epochs-{args.epochs}").resolve() if args.smoke else CACHE_DEFAULT.resolve()
    official_cache, outputs_root = CACHE_DEFAULT.resolve(), (ROOT / "outputs").resolve()
    if args.smoke:
        if args.epochs < 1:
            raise SystemExit("--epochs must be >= 1")
        if args.out.is_relative_to(outputs_root) or not args.out.is_relative_to(SMOKE_ROOT.resolve()) or args.out == SMOKE_ROOT.resolve():
            raise SystemExit(f"REFUSED: a smoke run (--epochs {args.epochs}) writes only into a directory under build/watch-head/, "
                             f"not {args.out}")
        if args.cache.is_relative_to(official_cache):
            raise SystemExit("REFUSED: a smoke run may not use the official job cache build/watch-head/jobs")
        if args.out.is_relative_to(args.cache) or args.cache.is_relative_to(args.out):
            raise SystemExit("REFUSED: --out and the job cache must be separate directories")
    if args.force_final_smoke and not args.smoke:
        raise SystemExit("REFUSED: --force-final-smoke exists for smoke runs only")
    return args


def run_stage(stage, data, args):
    before = protected_hashes()
    try:
        code = STAGES[stage](data, args)
    except StopAnalysis as stop:
        print(f"STOP: {stop}", flush=True)
        return 3
    if protected_hashes() != before:
        raise RuntimeError("A protected path changed during the run")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER + ("final", "all"))
    parser.add_argument("--out", type=Path, default=None, help="output dir (default outputs/watch-head-v1; smoke: under build/watch-head/)")
    parser.add_argument("--cache", type=Path, default=None, help="job cache (default build/watch-head/jobs; smoke: build/watch-head/smoke-jobs/epochs-N)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="smoke only: anything but 100 marks the run as smoke")
    parser.add_argument("--jobs", type=int, default=min(14, os.cpu_count() or 1), help="worker processes for model jobs")
    parser.add_argument("--force-final-smoke", action="store_true", help="smoke only: run --stage final despite a failed gate")
    args = configure(parser.parse_args(argv))
    print(f"out {args.out}; cache {args.cache}; epochs {args.epochs}" + (" (SMOKE)" if args.smoke else ""), flush=True)
    try:
        data = load()
    except StopAnalysis as stop:
        print(f"STOP: {stop}", flush=True)
        return 3
    data["sha_v2"] = sha256(P["v2_features"])
    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        started = time.perf_counter()
        code = run_stage(stage, data, args)
        print(f"stage {stage}: exit {code}, {time.perf_counter() - started:.1f} s", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

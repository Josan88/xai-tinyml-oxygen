"""ScalcO2 (the label) against arterial co-oximetry SaO2 at OpenOximetry blood draws (host only).

Pre-registration: docs/scalco2-vs-sao2-preregistration.md (committed 456b006, 8 Nov 2026, before any
paired statistic). It is the specification; this script implements it. No model is trained or
loaded: S1 reads the stored out-of-fold prediction CSVs only. No board, no firmware.

Wording: ScalcO2 = SaO2 calculated from end-tidal O2 (the label; never "SpO2"); SaO2 = arterial
co-oximetry (`bloodgas.csv` so2); SpO2 = pulse-oximeter reading (`pulseoximeter.csv` saturation).

Gate (all draws, point estimates, unrounded float64):
  P1  ARMS_S (draws with 70 <= SaO2_draw <= 100) <= median ARMS_d of benchmark devices
  P2  kappa_S (ScalcO2_draw < 90 vs SaO2_draw < 90, all draws) >= median kappa_d of benchmark devices
  PASS iff P1 and P2 pass; otherwise FAIL naming the failed part(s).

Output dir outputs/scalco2-vs-sao2-v1/ is write-once: the script refuses to run if it exists and
creates it only after every input has loaded and every input assert has passed.

Run from the repository root:
  py -3.14 scripts/analyze_scalco2_vs_sao2.py --prereg-commit 456b006
Smoke mode (tests the code only, not a result): --smoke uses the first 20 encounters by sorted
encounter_id, skips the full-data count asserts, and writes only under build/scalco2-vs-sao2-v1/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lamaakal_metrics as lm  # noqa: E402

ROOT = SCRIPTS.parent
OPENOX = ROOT / "data" / "physionet.org" / "files" / "openox-repo" / "1.1.1"
OUT_DEFAULT = ROOT / "outputs" / "scalco2-vs-sao2-v1"
SMOKE_ROOT = ROOT / "build" / "scalco2-vs-sao2-v1"
PREREG = ROOT / "docs" / "scalco2-vs-sao2-preregistration.md"

P = {
    "bloodgas": OPENOX / "bloodgas.csv",
    "pulseoximeter": OPENOX / "pulseoximeter.csv",
    "v2_windows": ROOT / "outputs" / "openox-v2" / "window_features.csv",
    "tinykan_oof": ROOT / "outputs" / "openox-v2" / "predictions_b_v2_tinykan16.csv",
    "logreg_oof": ROOT / "outputs" / "openox-v2" / "predictions_c_v2_logreg16.csv",
    "eligible_encounters": ROOT / "outputs" / "openox_ppg_eligible_encounters.csv",
}
EXPECTED_SHA256 = {
    "bloodgas": "73c96a32e95ff1c4a483194eb4a35718cca963411ad46c3bba48bfe461e47afc",
    "pulseoximeter": "979a37049a94eca0fdcc2803d8a5fd205d0b21d47fda07aecaa5201d39e88d6d",
    "v2_windows": "d96d93989f1ffa85f72689a2e084c898dacf18d4eafce50712fb0dcf3ad1b3d6",
    "tinykan_oof": "e1449fcac7d32bd691d4e1168fa3d7254dffb63c832cf937124162aa546a1611",
    "logreg_oof": "4ce1e78f08e43e26452cc21b6e5aa1165abe32256b710f9a79dcb1e49e2c80b7",
    "eligible_encounters": "120734a1a4a460f962b58d5aaeb5b52dbbeeb9aa9bc909d184bacb9c888662b5",
}
ENCOUNTER_CSV = OPENOX / "encounter.csv"     # patient check only, not a gated input

# Counts fixed in the pre-registration (8 Nov); asserted on the full (non-smoke) run.
EXPECTED = {
    "rows_no_encounter_id": 257, "rows_both_numeric": 29295, "draws": 16709, "encounters": 748, "patients": 218,
    "v2_draws": 3458, "v2_rows": 6974, "v2_recordings": 154,
    "pulse_no_device": 190, "pulse_sample_number_nan": 11,
    "pulse_invalid_on_draws": {"non_numeric": 0, "above_100": 2, "below_range": 27},
    "valid_paired_readings": 79644, "valid_paired_readings_plan_rule": 79671,
    "device_labels": 68, "benchmark_p2_devices": 54,
    "v2_windows": 48662,
}

BELOW = 90.0
P1_LOW, P1_HIGH = 70.0, 100.0
MIN_READINGS = 100
VALID_LOW_PRIMARY = 10.0                  # 10 <= SpO2 <= 100 (pre-registration, pulse oximeters)
N_BOOT = 2000
SEED = 20261108
FLAG_THRESHOLD = 0.50
WINDOW_SECONDS = 10.0
NEAR_90 = 2.0
SMOKE_ENCOUNTERS = 20
BANDS = (("[70, 80)", 70.0, 80.0, False), ("[80, 90)", 80.0, 90.0, False), ("[90, 100]", 90.0, 100.0, True))
OUTPUT_FILES = ("draws.csv", "devices.csv", "bands.csv", "s1_windows.csv", "gate.json", "summary.json", "README.md")


class StopAnalysis(RuntimeError):
    pass


# ----------------------------------------------------------------------------- pure helpers (tested)

def both_numeric_rows(bloodgas):
    """Rows with an encounter_id and both so2 and ScalcO2 finite; also returns rows dropped for no encounter_id."""
    frame = bloodgas.loc[bloodgas["encounter_id"].notna()].copy()
    dropped = int(len(bloodgas) - len(frame))
    sample = pd.to_numeric(frame["sample"], errors="raise").to_numpy(dtype=float)
    assert np.isfinite(sample).all(), "bloodgas sample has NaN"
    assert (sample == np.round(sample)).all(), "bloodgas sample is not integer-valued"
    frame["sample"] = sample.astype(np.int64)
    frame["so2"] = pd.to_numeric(frame["so2"], errors="coerce").astype(float)
    frame["ScalcO2"] = pd.to_numeric(frame["ScalcO2"], errors="coerce").astype(float)
    both = np.isfinite(frame["so2"].to_numpy()) & np.isfinite(frame["ScalcO2"].to_numpy())
    rows = frame.loc[both, ["encounter_id", "patient_id", "sample", "so2", "ScalcO2"]].reset_index(drop=True)
    rows["encounter_id"] = rows["encounter_id"].astype(str)
    return rows, dropped


def aggregate_draws(bloodgas, encounter_patient=None):
    """Draw = (encounter_id, sample) with >= 1 both-numeric row; means of so2 and ScalcO2 over the same rows."""
    rows, dropped = both_numeric_rows(bloodgas)
    grouped = rows.groupby(["encounter_id", "sample"], sort=True)
    n_patients = grouped["patient_id"].nunique(dropna=False)
    assert (n_patients == 1).all(), f"{int((n_patients != 1).sum())} draws carry more than one patient_id"
    assert rows["patient_id"].notna().all(), "a both-numeric row has no patient_id"
    draws = grouped.agg(patient_id=("patient_id", "first"), n_rows=("so2", "size"),
                        SaO2_draw=("so2", "mean"), ScalcO2_draw=("ScalcO2", "mean")).reset_index()
    draws = draws[["encounter_id", "patient_id", "sample", "n_rows", "SaO2_draw", "ScalcO2_draw"]]
    draws["sample"] = draws["sample"].astype(np.int64)
    draws["n_rows"] = draws["n_rows"].astype(np.int64)
    draws["patient_id"] = draws["patient_id"].astype(str)
    if encounter_patient is not None:
        mapping = dict(encounter_patient)
        expected = draws["encounter_id"].map(lambda e: mapping.get(e))
        assert expected.notna().all(), "a draw's encounter is missing from encounter.csv"
        assert (expected.astype(str) == draws["patient_id"]).all(), "draw patient_id differs from encounter.csv"
    counts = {"rows_no_encounter_id": dropped, "rows_both_numeric": int(len(rows)), "draws": int(len(draws)),
              "encounters": int(draws["encounter_id"].nunique()), "patients": int(draws["patient_id"].nunique())}
    return draws, counts


def saturation_validity(saturation, rule="primary"):
    """(sat float64, reason) with reason in valid / non_numeric / above_100 / below_range.

    primary: valid iff 10 <= sat <= 100. plan (sensitivity only): valid iff 0 < sat <= 100.
    """
    sat = pd.to_numeric(pd.Series(saturation).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    reason = np.full(len(sat), "valid", dtype=object)
    finite = np.isfinite(sat)
    if rule == "primary":
        below = finite & (sat < VALID_LOW_PRIMARY)
    elif rule == "plan":
        below = finite & (sat <= 0.0)
    else:
        raise ValueError(f"unknown validity rule {rule!r}")
    reason[finite & (sat > 100.0)] = "above_100"
    reason[below] = "below_range"
    reason[~finite] = "non_numeric"
    return sat, reason


def device_label(device):
    """'2', '2.0' -> '2.00'; '56 .01' -> '56.01'; suffixes .01/.02 stay distinct devices."""
    device = pd.Series(device)
    return pd.Series([f"{float(str(v).replace(' ', '')):.2f}" for v in device], index=device.index, dtype=object)


def integer_model(label):
    label = pd.Series(label)
    return pd.Series([int(float(v)) for v in label], index=label.index, dtype=np.int64)


def classify_pulse(pulse, rule="primary"):
    """Drop rows with no device, then NaN sample_number; classify the rest. Returns (frame, counts)."""
    frame = pulse.copy()
    no_device = frame["device"].isna().to_numpy()
    frame = frame.loc[~no_device].copy()
    sample_number = pd.to_numeric(frame["sample_number"], errors="coerce").astype(float)
    nan_sample = sample_number.isna().to_numpy()
    frame = frame.loc[~nan_sample].copy()
    sample_number = sample_number.loc[~nan_sample].to_numpy()
    assert (sample_number == np.round(sample_number)).all(), "sample_number is not integer-valued"
    frame["sample_number"] = sample_number.astype(np.int64)
    frame["encounter_id"] = frame["encounter_id"].astype(str)
    sat, reason = saturation_validity(frame["saturation"], rule)
    frame["spo2"] = sat
    frame["reason"] = reason
    frame["device_label"] = device_label(frame["device"]).to_numpy()
    frame["model"] = integer_model(frame["device_label"]).to_numpy()
    counts = {"no_device": int(no_device.sum()), "sample_number_nan": int(nan_sample.sum()),
              "invalid_non_numeric": int((reason == "non_numeric").sum()),
              "invalid_above_100": int((reason == "above_100").sum()),
              "invalid_below_range": int((reason == "below_range").sum()), "rule": rule}
    return frame.reset_index(drop=True), counts


def clean_pulse(pulse, rule="primary"):
    frame, counts = classify_pulse(pulse, rule)
    clean = frame.loc[frame["reason"] == "valid", ["encounter_id", "device_label", "model", "sample_number", "spo2"]]
    clean = clean.reset_index(drop=True)
    assert not clean.duplicated(["encounter_id", "device_label", "sample_number"]).any(), \
        "(encounter_id, device_label, sample_number) repeats after cleaning"
    counts["valid"] = int(len(clean))
    return clean, counts


def pair_readings(clean, draws):
    """Inner join of cleaned pulse readings to draws on encounter_id and sample_number == sample."""
    pairs = clean.rename(columns={"sample_number": "sample"}).merge(
        draws, on=["encounter_id", "sample"], how="inner", validate="many_to_one")
    return pairs.sort_values(["encounter_id", "sample", "device_label"], kind="mergesort").reset_index(drop=True)


def in_p1(sao2):
    s = np.asarray(sao2, dtype=float)
    return (s >= P1_LOW) & (s <= P1_HIGH)


def below90(x):
    return (np.asarray(x, dtype=float) < BELOW).astype(np.int64)


def arms(x, r, weights=None):
    d = np.asarray(x, dtype=float) - np.asarray(r, dtype=float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
    total = w.sum()
    return float(np.sqrt(np.sum(w * d ** 2) / total)) if total > 0 else float("nan")


def bias(x, r, weights=None):
    d = np.asarray(x, dtype=float) - np.asarray(r, dtype=float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
    total = w.sum()
    return float(np.sum(w * d) / total) if total > 0 else float("nan")


def bland_altman(x, r):
    """Plain Bland-Altman (ignores repeated draws per patient): n, bias, SD (ddof 1), 95 % LoA, ARMS."""
    d = np.asarray(x, dtype=float) - np.asarray(r, dtype=float)
    n = int(len(d))
    b = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n": n, "bias": b, "sd": sd, "loa_low": b - 1.96 * sd, "loa_high": b + 1.96 * sd,
            "arms": float(np.sqrt(np.mean(d ** 2))) if n else float("nan")}


def kappa_binary(y1, y2, sample_weight=None):
    y1, y2 = np.asarray(y1).astype(np.int64), np.asarray(y2).astype(np.int64)
    if len(y1) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(y1, y2, labels=[0, 1], sample_weight=sample_weight))


def kappa_below(x, r, sample_weight=None):
    """Cohen's kappa between x < 90 and r < 90 (strict), labels [0, 1], unweighted kappa."""
    return kappa_binary(below90(x), below90(r), sample_weight)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def sens_spec(test, truth, weights=None):
    """Sensitivity and specificity of a 0/1 test against 0/1 truth; NaN on an empty denominator."""
    test = np.asarray(test).astype(bool)
    truth = np.asarray(truth).astype(bool)
    w = np.ones(len(test)) if weights is None else np.asarray(weights, dtype=float)
    return (_ratio(np.sum(w * (test & truth)), np.sum(w * truth)),
            _ratio(np.sum(w * (~test & ~truth)), np.sum(w * ~truth)))


def device_table(pairs, column="device_label", min_n=MIN_READINGS):
    """Per device: valid paired readings on P1 draws and on all draws, benchmark flags, ARMS_d, bias_d, kappa_d."""
    rows = []
    keys = sorted(pairs[column].unique(), key=float)
    p1 = in_p1(pairs["SaO2_draw"].to_numpy())
    for key in keys:
        mine = (pairs[column] == key).to_numpy()
        x, r = pairs["spo2"].to_numpy()[mine], pairs["SaO2_draw"].to_numpy()[mine]
        in_range = p1[mine]
        n_p1, n_all = int(in_range.sum()), int(mine.sum())
        rows.append({"device": key, "model": int(float(key)), "n_p1": n_p1, "n_all": n_all,
                     "benchmark_p1": n_p1 >= min_n, "benchmark_p2": n_all >= min_n,
                     "arms_d": arms(x[in_range], r[in_range]), "bias_d": bias(x[in_range], r[in_range]),
                     "kappa_d": kappa_below(x, r)})
    columns = ["device", "model", "n_p1", "n_all", "benchmark_p1", "benchmark_p2", "arms_d", "bias_d", "kappa_d"]
    return pd.DataFrame(rows, columns=columns)


def gate(arms_s, device_arms, kappa_s, device_kappa):
    """P1: ARMS_S <= median device ARMS; P2: kappa_S >= median finite device kappa; float64, unrounded."""
    a = np.asarray(device_arms, dtype=float)
    k = np.asarray(device_kappa, dtype=float)
    assert np.isfinite(a).all(), "a benchmark device has a non-finite ARMS"
    k_finite = k[np.isfinite(k)]
    median_a = float(np.median(a)) if len(a) else float("nan")
    median_k = float(np.median(k_finite)) if len(k_finite) else float("nan")
    p1 = bool(float(arms_s) <= median_a)
    p2 = bool(float(kappa_s) >= median_k)
    failed = [name for name, ok in (("P1", p1), ("P2", p2)) if not ok]
    return {
        "P1": {"ARMS_S": float(arms_s), "device_median": median_a, "n_devices": int(len(a)), "rule": "ARMS_S <= median",
               "pass": p1},
        "P2": {"kappa_S": float(kappa_s), "device_median": median_k, "n_devices": int(len(k_finite)),
               "n_excluded_nonfinite": int(len(k) - len(k_finite)), "rule": "kappa_S >= median", "pass": p2},
        "verdict": "PASS" if not failed else "FAIL",
        "failed_parts": failed,
    }


def rank_among(value, others, higher_is_better):
    """1 + number of (finite) devices strictly better than value."""
    o = np.asarray(others, dtype=float)
    o = o[np.isfinite(o)]
    better = o > value if higher_is_better else o < value
    return int(1 + better.sum())


def bootstrap_counts(patient_ids, n_boot=N_BOOT, seed=SEED):
    """(groups, group_index, B x n_groups counts) with np.unique-sorted string patient ids."""
    groups, group_index = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    n_groups = int(group_index.max()) + 1
    return groups, group_index, lm.group_bootstrap_counts(group_index, n_groups, n_boot, seed)


def elapsed_2hz(frame):
    """As export_openox_windows_v2.py l.221-226: keep valid-Timestamp rows; elapsed from the first one.

    The reference-column notna filter (l.227) is not applied.
    """
    out = frame.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.loc[out["Timestamp"].notna()].copy()
    t0 = out["Timestamp"].iloc[0]
    out["elapsed_seconds"] = (out["Timestamp"] - t0).dt.total_seconds()
    return out.reset_index(drop=True)


def marker_rows(hz2):
    markers = hz2.loc[hz2["Sample"].notna(), ["Sample", "elapsed_seconds"]].copy()
    sample = pd.to_numeric(markers["Sample"], errors="raise").to_numpy(dtype=float)
    assert (sample == np.round(sample)).all(), "a 2 Hz Sample marker is not integer-valued"
    markers["Sample"] = sample.astype(np.int64)
    return markers.reset_index(drop=True)


def draw_time(markers, sample):
    """(t_draw, n_marker_rows): median elapsed seconds over rows whose Sample equals the draw's sample."""
    times = markers.loc[markers["Sample"].to_numpy() == int(sample), "elapsed_seconds"].to_numpy(dtype=float)
    return (float(np.median(times)) if len(times) else float("nan")), int(len(times))


def match_window(t_draw, starts, ends, delta):
    """(status, position) with spans [start + delta, end + delta); two candidates -> nearest centre, tie -> earlier."""
    if not np.isfinite(t_draw):
        return "no marker", None
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    lo, hi = starts + delta, ends + delta
    candidates = np.flatnonzero((lo <= t_draw) & (t_draw < hi))
    assert len(candidates) <= 2, f"{len(candidates)} windows contain t_draw {t_draw}"
    if len(candidates) == 0:
        return "no window", None
    if len(candidates) == 1:
        return "matched", int(candidates[0])
    distance = np.abs((starts[candidates] + ends[candidates]) / 2 + delta - t_draw)
    if distance[0] == distance[1]:
        return "matched", int(candidates[np.argmin(starts[candidates])])
    return "matched", int(candidates[np.argmin(distance)])


def join_oof(windows, oof, positions):
    """Stored out-of-fold rows at row_index == window-table position; asserts the keys agree."""
    positions = np.asarray(positions, dtype=np.int64)
    got = oof.set_index("row_index", drop=False).loc[positions].reset_index(drop=True)
    expected = windows.iloc[positions].reset_index(drop=True)
    assert (got["recording_id"].astype(str).to_numpy() == expected["recording_id"].astype(str).to_numpy()).all(), \
        "out-of-fold recording_id differs from the window table"
    assert np.array_equal(got["start_time_seconds"].to_numpy(dtype=float),
                          expected["start_time_seconds"].to_numpy(dtype=float)), \
        "out-of-fold start_time_seconds differs from the window table"
    return got


# ----------------------------------------------------------------------------- utilities

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    return value


def new_file(out, name):
    path = out / name
    if path.exists():
        raise FileExistsError(f"{path} exists; this script never overwrites an output file")
    return path


def write_csv(frame, out, name):
    frame.to_csv(new_file(out, name), index=False, float_format="%.10g", lineterminator="\n")


def write_json(payload, out, name):
    text = json.dumps(_clean_json(payload), indent=2, allow_nan=False) + "\n"
    new_file(out, name).write_text(text, encoding="utf-8", newline="\n")


def environment():
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__}


def ci_from(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    low, high = lm.percentile_ci(values, 0.95) if finite.any() else (float("nan"), float("nan"))
    return {"ci_low": low, "ci_high": high, "n_nonfinite": int((~finite).sum())}


def bootstrap(patient_ids, statistics, n_boot=N_BOOT, seed=SEED):
    """statistics: name -> f(w) with w the per-unit weight vector; returns name -> CI block."""
    if len(patient_ids) == 0:
        return {name: {"ci_low": None, "ci_high": None, "n_nonfinite": None} for name in statistics}, 0
    groups, group_index, counts = bootstrap_counts(patient_ids, n_boot, seed)
    values = {name: np.empty(n_boot) for name in statistics}
    for b in range(n_boot):
        w = counts[b][group_index]
        for name, fn in statistics.items():
            values[name][b] = fn(w)
    return {name: ci_from(v) for name, v in values.items()}, int(len(groups))


def _quartiles(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return dict(zip(("min", "q1", "median", "q3", "max"), (float(x) for x in q)))


# ----------------------------------------------------------------------------- load

def load(smoke):
    """Read and assert every input (SHA-256 first); StopAnalysis on the first failure."""
    hashes = {}
    for key, path in P.items():
        hashes[key] = sha256(path)
        if hashes[key] != EXPECTED_SHA256[key]:
            raise StopAnalysis(f"SHA-256 mismatch for {path}: {hashes[key]} != {EXPECTED_SHA256[key]}")
    print("input SHA-256 match", flush=True)

    bloodgas = pd.read_csv(P["bloodgas"], dtype={"encounter_id": str, "patient_id": str}, low_memory=False)
    pulse = pd.read_csv(P["pulseoximeter"], dtype={"encounter_id": str, "device": str, "saturation": str})
    pulse["sample_number"] = pd.to_numeric(pulse["sample_number"], errors="coerce").astype(float)
    encounters = pd.read_csv(ENCOUNTER_CSV, usecols=["patient_id", "encounter_id"], dtype=str)
    assert not encounters["encounter_id"].duplicated().any(), "encounter.csv repeats an encounter_id"
    encounter_patient = dict(zip(encounters["encounter_id"], encounters["patient_id"]))
    windows = pd.read_csv(P["v2_windows"], dtype={"recording_id": str, "patient_id": str})
    oof = {name: pd.read_csv(P[key], dtype={"recording_id": str, "patient_id": str})
           for name, key in (("tinykan", "tinykan_oof"), ("logreg", "logreg_oof"))}
    eligible = pd.read_csv(P["eligible_encounters"], dtype={"encounter_id": str})

    smoke_encounters = None
    if smoke:
        smoke_encounters = sorted(bloodgas["encounter_id"].dropna().astype(str).unique())[:SMOKE_ENCOUNTERS]
        keep = set(smoke_encounters)
        bloodgas = bloodgas.loc[bloodgas["encounter_id"].isna() | bloodgas["encounter_id"].isin(keep)].reset_index(drop=True)
        pulse = pulse.loc[pulse["encounter_id"].isna() | pulse["encounter_id"].isin(keep)].reset_index(drop=True)

    # Blood gas -> draws
    draws, draw_counts = aggregate_draws(bloodgas, encounter_patient)
    rows, _ = both_numeric_rows(bloodgas)
    v2_recordings = set(windows["recording_id"].unique())
    draws["in_v2"] = draws["encounter_id"].isin(v2_recordings)
    rows["in_v2"] = rows["encounter_id"].isin(v2_recordings)
    draw_counts.update({"v2_draws": int(draws["in_v2"].sum()), "v2_rows": int(draws.loc[draws["in_v2"], "n_rows"].sum()),
                        "v2_recordings": int(draws.loc[draws["in_v2"], "encounter_id"].nunique()),
                        "v2_patients": int(draws.loc[draws["in_v2"], "patient_id"].nunique())})

    # Pulse oximeters
    draw_keys = draws[["encounter_id", "sample"]]
    pulse_counts, pairs = {}, {}
    for rule in ("primary", "plan"):
        classified, counts = classify_pulse(pulse, rule)
        on_draw = classified.rename(columns={"sample_number": "sample"}).merge(draw_keys, on=["encounter_id", "sample"],
                                                                               how="inner")
        counts["invalid_on_draws"] = {r: int((on_draw["reason"] == r).sum()) for r in ("non_numeric", "above_100", "below_range")}
        clean, clean_counts = clean_pulse(pulse, rule)
        counts["valid"] = clean_counts["valid"]
        pairs[rule] = pair_readings(clean, draws)
        counts["valid_paired_readings"] = int(len(pairs[rule]))
        counts["device_labels_on_pairs"] = int(pairs[rule]["device_label"].nunique())
        counts["models_on_pairs"] = int(pairs[rule]["model"].nunique())
        pulse_counts[rule] = counts
    with_device = pulse.loc[pulse["device"].notna(), "device"]
    pulse_counts["raw_rows"] = int(len(pulse))
    pulse_counts["device_labels"] = int(device_label(with_device).nunique())
    pulse_counts["integer_models"] = int(integer_model(device_label(with_device)).nunique())
    pulse_counts["raw_device_spellings"] = int(with_device.nunique())
    draws = draws.merge(pairs["primary"].groupby(["encounter_id", "sample"]).size().rename("n_valid_pulse_readings")
                        .reset_index(), on=["encounter_id", "sample"], how="left")
    draws["n_valid_pulse_readings"] = draws["n_valid_pulse_readings"].fillna(0).astype(np.int64)

    # Window table and stored out-of-fold rows (S1 step 8)
    n_windows = len(windows)
    assert np.allclose(windows["end_time_seconds"] - windows["start_time_seconds"], WINDOW_SECONDS, rtol=0, atol=0), \
        "a window is not 10.0 s long"
    assert not windows.duplicated(["recording_id", "start_time_seconds"]).any()
    for name, frame in oof.items():
        assert np.array_equal(frame["row_index"].to_numpy(), np.arange(n_windows)), f"{name}: row_index is not 0..N-1"
        assert (frame["recording_id"].to_numpy() == windows["recording_id"].to_numpy()).all(), f"{name}: recording_id"
        assert (frame["patient_id"].to_numpy() == windows["patient_id"].to_numpy()).all(), f"{name}: patient_id"
        assert np.array_equal(frame["start_time_seconds"].to_numpy(float), windows["start_time_seconds"].to_numpy(float)), \
            f"{name}: start_time_seconds"
        assert (frame["y_true"].to_numpy() == windows["label"].to_numpy()).all(), f"{name}: y_true != label"

    # S1: draw times and window matching
    assert not eligible["encounter_id"].duplicated().any()
    eligible = eligible.set_index("encounter_id")
    s1_records = []
    v2_draws = draws.loc[draws["in_v2"]].sort_values(["encounter_id", "sample"], kind="mergesort")
    for recording_id, recording_draws in v2_draws.groupby("encounter_id", sort=True):
        assert recording_id in eligible.index, f"{recording_id}: no eligible-encounters row (delta)"
        row = eligible.loc[recording_id]
        delta = float(row["ppg_start_time_seconds"] - row["hz2_first_time_seconds"])
        hz2 = elapsed_2hz(pd.read_csv(Path(str(row["hz2_csv_path"])), usecols=["Sample", "Timestamp"]))
        markers = marker_rows(hz2)
        mine = np.flatnonzero(windows["recording_id"].to_numpy() == recording_id)
        starts = windows["start_time_seconds"].to_numpy(dtype=float)[mine]
        ends = windows["end_time_seconds"].to_numpy(dtype=float)[mine]
        for draw in recording_draws.itertuples(index=False):
            t, n_markers = draw_time(markers, draw.sample)
            status, position = match_window(t, starts, ends, delta)
            s1_records.append({"encounter_id": draw.encounter_id, "patient_id": draw.patient_id, "sample": draw.sample,
                               "SaO2_draw": draw.SaO2_draw, "t_draw": t, "n_marker_rows": n_markers, "status": status,
                               "row_index": int(mine[position]) if position is not None else None})
    s1 = pd.DataFrame(s1_records, columns=["encounter_id", "patient_id", "sample", "SaO2_draw", "t_draw", "n_marker_rows",
                                           "status", "row_index"])
    matched = s1["status"].to_numpy() == "matched"
    positions = s1.loc[matched, "row_index"].to_numpy(dtype=np.int64)
    tk = join_oof(windows, oof["tinykan"], positions)
    lr = join_oof(windows, oof["logreg"], positions)
    assert (tk["y_true"].to_numpy() == lr["y_true"].to_numpy()).all()
    s1["row_index"] = pd.array([int(v) if v is not None and v == v else None for v in s1["row_index"]], dtype="Int64")
    for column, values in (("recording_id", tk["recording_id"].to_numpy()),
                           ("start_time_seconds", tk["start_time_seconds"].to_numpy(dtype=float)),
                           ("y_true", tk["y_true"].to_numpy()), ("y_prob_tinykan", tk["y_prob"].to_numpy(dtype=float)),
                           ("y_prob_logreg", lr["y_prob"].to_numpy(dtype=float))):
        full = pd.Series([None] * len(s1), dtype=object)
        full[matched] = values
        s1[column] = full
    s1["y_true"] = pd.array([None if v is None else int(v) for v in s1["y_true"]], dtype="Int64")
    for column in ("start_time_seconds", "y_prob_tinykan", "y_prob_logreg"):
        s1[column] = pd.to_numeric(s1[column]).astype(float)
    s1 = s1[["encounter_id", "patient_id", "sample", "t_draw", "n_marker_rows", "status", "row_index", "recording_id",
             "start_time_seconds", "y_true", "y_prob_tinykan", "y_prob_logreg", "SaO2_draw"]]
    matched_rows = s1.loc[matched, "row_index"]
    s1_counts = {"draws_in_v2_recordings": int(len(s1)), "no_marker": int((s1["status"] == "no marker").sum()),
                 "no_window": int((s1["status"] == "no window").sum()), "matched": int(matched.sum()),
                 "windows_matched_by_more_than_one_draw": int((matched_rows.value_counts() > 1).sum()),
                 "recordings": int(v2_draws["encounter_id"].nunique()),
                 "patients_matched": int(s1.loc[matched, "patient_id"].nunique())}

    if not smoke:
        checks = {
            "rows_no_encounter_id": draw_counts["rows_no_encounter_id"], "rows_both_numeric": draw_counts["rows_both_numeric"],
            "draws": draw_counts["draws"], "encounters": draw_counts["encounters"], "patients": draw_counts["patients"],
            "v2_draws": draw_counts["v2_draws"], "v2_rows": draw_counts["v2_rows"], "v2_recordings": draw_counts["v2_recordings"],
            "pulse_no_device": pulse_counts["primary"]["no_device"],
            "pulse_sample_number_nan": pulse_counts["primary"]["sample_number_nan"],
            "pulse_invalid_on_draws": pulse_counts["primary"]["invalid_on_draws"],
            "valid_paired_readings": pulse_counts["primary"]["valid_paired_readings"],
            "valid_paired_readings_plan_rule": pulse_counts["plan"]["valid_paired_readings"],
            "device_labels": pulse_counts["device_labels"],
            "benchmark_p2_devices": int((pairs["primary"]["device_label"].value_counts() >= MIN_READINGS).sum()),
            "v2_windows": n_windows,
        }
        for key, expected in EXPECTED.items():
            if checks[key] != expected:
                raise StopAnalysis(f"count {key} = {checks[key]}, pre-registered {expected}")
        print("pre-registered counts match", flush=True)

    return {"hashes": hashes, "draws": draws, "rows": rows, "pairs": pairs, "draw_counts": draw_counts,
            "pulse_counts": pulse_counts, "s1": s1, "s1_counts": s1_counts, "smoke_encounters": smoke_encounters}


# ----------------------------------------------------------------------------- analysis

def device_summary(pairs, column, scalc_arms, scalc_kappa):
    table = device_table(pairs, column=column)
    bench_p1 = table.loc[table["benchmark_p1"]]
    bench_p2 = table.loc[table["benchmark_p2"]]
    p1_pairs = pairs.loc[pairs[column].isin(bench_p1["device"]) & in_p1(pairs["SaO2_draw"].to_numpy())]
    p2_pairs = pairs.loc[pairs[column].isin(bench_p2["device"])]
    comparison = gate(scalc_arms, bench_p1["arms_d"].to_numpy(), scalc_kappa, bench_p2["kappa_d"].to_numpy())
    summary = {
        "unit": column, "devices_with_readings": int(len(table)),
        "benchmark_p1_devices": int(len(bench_p1)), "benchmark_p2_devices": int(len(bench_p2)),
        "arms_d_distribution_p1": _quartiles(bench_p1["arms_d"]),
        "kappa_d_distribution_p2": _quartiles(bench_p2["kappa_d"]),
        "kappa_d_nonfinite_p2": int((~np.isfinite(bench_p2["kappa_d"].to_numpy(dtype=float))).sum()),
        "pooled_arms_p1": arms(p1_pairs["spo2"], p1_pairs["SaO2_draw"]) if len(p1_pairs) else float("nan"),
        "pooled_readings_p1": int(len(p1_pairs)),
        "pooled_kappa_p2": kappa_below(p2_pairs["spo2"], p2_pairs["SaO2_draw"]),
        "pooled_readings_p2": int(len(p2_pairs)),
        "scalco2_rank_arms": rank_among(scalc_arms, bench_p1["arms_d"], higher_is_better=False),
        "scalco2_rank_kappa": rank_among(scalc_kappa, bench_p2["kappa_d"], higher_is_better=True),
        "comparison": comparison,
    }
    return table, summary


def population(draws, pairs, rows, name):
    x = draws["ScalcO2_draw"].to_numpy(dtype=float)
    r = draws["SaO2_draw"].to_numpy(dtype=float)
    p1 = in_p1(r)
    xb, rb = below90(x), below90(r)
    disagree = xb != rb
    near = np.abs(r - BELOW) <= NEAR_90
    arms_s = arms(x[p1], r[p1])
    kappa_s = kappa_below(x, r)
    sens, spec = sens_spec(xb, rb)
    band_rows = []
    for band, lo, hi, closed in BANDS:
        mask = (r >= lo) & ((r <= hi) if closed else (r < hi))
        ba = bland_altman(x[mask], r[mask])
        band_rows.append({"population": name, "band": band, "n": ba["n"], "bias": ba["bias"], "sd": ba["sd"],
                          "loa_low": ba["loa_low"], "loa_high": ba["loa_high"], "arms": ba["arms"]})
    cis, n_groups = bootstrap(draws["patient_id"].to_numpy(), {
        "ARMS_S": lambda w: arms(x[p1], r[p1], w[p1]),
        "bias_p1": lambda w: bias(x[p1], r[p1], w[p1]),
        "kappa_S": lambda w: kappa_below(x, r, w),
        "sensitivity": lambda w: sens_spec(xb, rb, w)[0],
        "specificity": lambda w: sens_spec(xb, rb, w)[1],
    })
    table, devices = device_summary(pairs, "device_label", arms_s, kappa_s)
    rx, rr = rows["ScalcO2"].to_numpy(dtype=float), rows["so2"].to_numpy(dtype=float)
    row_p1 = in_p1(rr)
    row_sens, row_spec = sens_spec(below90(rx), below90(rr))
    result = {
        "draws": int(len(draws)), "encounters": int(draws["encounter_id"].nunique()),
        "patients": int(draws["patient_id"].nunique()), "p1_draws": int(p1.sum()),
        "draws_sao2_below_70": int((r < P1_LOW).sum()), "draws_sao2_above_100": int((r > P1_HIGH).sum()),
        "ARMS_S": arms_s, "kappa_S": kappa_s, "sensitivity": sens, "specificity": spec,
        "bland_altman_p1": bland_altman(x[p1], r[p1]), "bland_altman_all_draws": bland_altman(x, r),
        "cross_table": {"both_below_90": int((xb & rb).sum()), "scalco2_only_below_90": int((xb & ~rb.astype(bool)).sum()),
                        "sao2_only_below_90": int((~xb.astype(bool) & rb.astype(bool)).sum()),
                        "both_at_or_above_90": int(((xb == 0) & (rb == 0)).sum())},
        "disagreements": {"total": int(disagree.sum()), "sao2_within_2_of_90": int((disagree & near).sum()),
                          "sao2_more_than_2_from_90": int((disagree & ~near).sum())},
        "bootstrap": {"n_patients": n_groups, "B": N_BOOT, "seed": SEED, **cis},
        "devices": devices,
        "row_level": {"rows": int(len(rows)), "p1_rows": int(row_p1.sum()), "ARMS_S": arms(rx[row_p1], rr[row_p1]),
                      "kappa_S": kappa_below(rx, rr), "sensitivity": row_sens, "specificity": row_spec},
    }
    return result, band_rows, table


def s1_results(s1):
    m = s1.loc[s1["status"] == "matched"]
    truth = below90(m["SaO2_draw"].to_numpy(dtype=float))
    y_true = m["y_true"].to_numpy(dtype=np.int64) if len(m) else np.zeros(0, dtype=np.int64)
    stats = {}
    for model in ("tinykan", "logreg"):
        flag = (m[f"y_prob_{model}"].to_numpy(dtype=float) >= FLAG_THRESHOLD).astype(np.int64)
        stats[f"{model}_recall_vs_sao2"] = (lambda f: lambda w=None: sens_spec(f, truth, w)[0])(flag)
        stats[f"{model}_specificity_vs_sao2"] = (lambda f: lambda w=None: sens_spec(f, truth, w)[1])(flag)
        stats[f"{model}_recall_vs_y_true"] = (lambda f: lambda w=None: sens_spec(f, y_true, w)[0])(flag)
        stats[f"{model}_specificity_vs_y_true"] = (lambda f: lambda w=None: sens_spec(f, y_true, w)[1])(flag)
    stats["y_true_kappa_vs_sao2"] = lambda w=None: kappa_binary(y_true, truth, w)
    stats["y_true_sensitivity_vs_sao2"] = lambda w=None: sens_spec(y_true, truth, w)[0]
    stats["y_true_specificity_vs_sao2"] = lambda w=None: sens_spec(y_true, truth, w)[1]
    point = {name: fn() for name, fn in stats.items()}
    cis, n_groups = bootstrap(m["patient_id"].to_numpy(), stats)
    return {"flag_rule": "y_prob >= 0.50, single window (not N = 2)", "matched_draws": int(len(m)),
            "matched_draws_sao2_below_90": int(truth.sum()), "matched_y_true_positive": int(y_true.sum()),
            "point": point, "bootstrap": {"n_patients": n_groups, "B": N_BOOT, "seed": SEED, **cis}}


INTERPRETATIONS = [
    "Headline bias, SD and 95 % LoA (and the bootstrapped bias) are on the P1 draws (70 <= SaO2_draw <= 100), "
    "the same draws as ARMS_S; all-draw values are reported beside them (bland_altman_all_draws).",
    "Row-level sensitivity: ARMS_S on both-numeric rows with 70 <= so2 <= 100; kappa, sensitivity and specificity "
    "on all both-numeric rows.",
    "Device tables on the v2 subset use only pulse readings at v2-subset draws, with the same >= 100 rule; their "
    "P1/P2 comparison is reported, not gated.",
    "Smoke mode keeps the first 20 encounter_ids (string sort) of bloodgas.csv and the pulse rows of those encounters.",
]


def _fmt(value, nd=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{value:.{nd}f}"


def _ci(block, name, nd=3):
    c = block["bootstrap"][name]
    return f"[{_fmt(c['ci_low'], nd)}, {_fmt(c['ci_high'], nd)}]"


def readme(summary, gate_payload, band_rows):
    L = ["# ScalcO2 (the label) against arterial SaO2 at blood draws", ""]
    if summary["smoke"]:
        L += ["**SMOKE RUN (first 20 encounters). This tests the code only. Nothing here is a result.**", ""]
    L += [f"Pre-registration: `docs/scalco2-vs-sao2-preregistration.md` (commit {summary['prereg_commit']}). "
          "Script: `scripts/analyze_scalco2_vs_sao2.py`. Host only; no model trained or loaded; no board.", "",
          "The label of every reported Header-aligned model number is window-median ScalcO2 < 90. ScalcO2 is SaO2 "
          "calculated from end-tidal O2 (Severinghaus 1979), not a pulse-oximeter reading. This file compares it with "
          "arterial co-oximetry SaO2 at the OpenOximetry blood draws, using the pulse oximeters tested at the same draws "
          "(their readings, SpO2) as the yardstick. It is a check of the label, not a hypoxia result, not a test of the "
          "MAX30102 or the board, and not an ISO 80601-2-61 accuracy claim.", ""]
    g = gate_payload
    verdict = g["verdict"] + (f" (failed: {', '.join(g['failed_parts'])})" if g["failed_parts"] else "")
    L += ["## Gate (all draws, point estimates)", "",
          "| Part | ScalcO2 | Median benchmark device | Devices | Pass |", "|---|---|---|---|---|",
          f"| P1 ARMS vs SaO2 (SaO2 70-100), lower is better | {_fmt(g['P1']['ARMS_S'])} | {_fmt(g['P1']['device_median'])} "
          f"| {g['P1']['n_devices']} | {'yes' if g['P1']['pass'] else 'no'} |",
          f"| P2 kappa, below 90 vs SaO2 below 90, higher is better | {_fmt(g['P2']['kappa_S'])} | "
          f"{_fmt(g['P2']['device_median'])} | {g['P2']['n_devices']} (+{g['P2']['n_excluded_nonfinite']} non-finite excluded) "
          f"| {'yes' if g['P2']['pass'] else 'no'} |", "",
          f"**Verdict: {verdict}.** " + (
              "At blood draws, the label (ScalcO2) tracks arterial SaO2 at least as closely as the median pulse oximeter "
              "tested in the same lab." if g["verdict"] == "PASS" else
              "At blood draws, the label (ScalcO2) does not track arterial SaO2 as closely as the median pulse oximeter "
              "tested in the same lab; on the failed part(s) it is noisier than a typical oximeter at draws."),
          "Neither outcome changes the shipped TinyKAN, its 0.50 threshold, N = 2, or any reported number.", ""]
    L += ["## Readings", ""]
    for key, title in (("all", "All draws (gate population)"), ("v2", "Draws in the 154 v2 recordings (reported, not gated)")):
        p = summary["populations"][key]
        d = p["devices"]
        L += [f"### {title}", "",
              f"- Draws {_fmt(p['draws'])} ({_fmt(p['encounters'])} encounters, {_fmt(p['patients'])} patients); "
              f"P1 draws {_fmt(p['p1_draws'])}; SaO2 < 70: {_fmt(p['draws_sao2_below_70'])}; > 100: {_fmt(p['draws_sao2_above_100'])}.",
              f"- ScalcO2 ARMS {_fmt(p['ARMS_S'])} {_ci(p, 'ARMS_S')}; bias {_fmt(p['bland_altman_p1']['bias'])} "
              f"{_ci(p, 'bias_p1')}; SD {_fmt(p['bland_altman_p1']['sd'])}; 95 % LoA {_fmt(p['bland_altman_p1']['loa_low'])} "
              f"to {_fmt(p['bland_altman_p1']['loa_high'])} (P1 draws; patient-bootstrap 95 % CIs in brackets).",
              f"- ScalcO2 < 90 vs SaO2 < 90: kappa {_fmt(p['kappa_S'])} {_ci(p, 'kappa_S')}; sensitivity "
              f"{_fmt(p['sensitivity'])} {_ci(p, 'sensitivity')}; specificity {_fmt(p['specificity'])} {_ci(p, 'specificity')}. "
              f"Disagreements {_fmt(p['disagreements']['total'])}: {_fmt(p['disagreements']['sao2_within_2_of_90'])} with "
              f"SaO2 within 2 of 90, {_fmt(p['disagreements']['sao2_more_than_2_from_90'])} further.",
              f"- Benchmark devices: {d['benchmark_p1_devices']} for P1, {d['benchmark_p2_devices']} for P2. Device ARMS "
              f"median {_fmt(d['arms_d_distribution_p1']['median'])} (range {_fmt(d['arms_d_distribution_p1']['min'])}-"
              f"{_fmt(d['arms_d_distribution_p1']['max'])}); device kappa median {_fmt(d['kappa_d_distribution_p2']['median'])}. "
              f"Pooled all-device ARMS {_fmt(d['pooled_arms_p1'])}, kappa {_fmt(d['pooled_kappa_p2'])}. ScalcO2 ranks "
              f"{d['scalco2_rank_arms']} of {d['benchmark_p1_devices'] + 1} on ARMS and {d['scalco2_rank_kappa']} of "
              f"{d['benchmark_p2_devices'] + 1} on kappa (1 = best).",
              f"- Row level (each both-numeric blood-gas row): ARMS {_fmt(p['row_level']['ARMS_S'])}, kappa "
              f"{_fmt(p['row_level']['kappa_S'])}.", ""]
    L += ["### ScalcO2 ARMS by SaO2 band", "", "| Population | SaO2 band | n | Bias | SD | 95 % LoA | ARMS |",
          "|---|---|---|---|---|---|---|"]
    for b in band_rows:
        L.append(f"| {b['population']} | {b['band']} | {_fmt(b['n'])} | {_fmt(b['bias'])} | {_fmt(b['sd'])} | "
                 f"{_fmt(b['loa_low'])} to {_fmt(b['loa_high'])} | {_fmt(b['arms'])} |")
    L += [""]
    s = summary["sensitivities"]
    L += ["### Sensitivities (reported; never change the verdict)", ""]
    for key, title in (("integer_model", "Devices pooled by integer model number"),
                       ("validity_0_to_100", "Pulse-oximeter validity 0 < SpO2 <= 100 (plan rule)")):
        c = s[key]["comparison"]
        L.append(f"- {title}: P1 median {_fmt(c['P1']['device_median'])} over {c['P1']['n_devices']} "
                 f"({'ScalcO2 at or below' if c['P1']['pass'] else 'ScalcO2 above'}); P2 median {_fmt(c['P2']['device_median'])} "
                 f"over {c['P2']['n_devices']} ({'ScalcO2 at or above' if c['P2']['pass'] else 'ScalcO2 below'}); ranks "
                 f"{s[key]['scalco2_rank_arms']} (ARMS), {s[key]['scalco2_rank_kappa']} (kappa).")
    L += [""]
    c1 = summary["s1_counts"]
    r1 = summary["s1"]
    pt = r1["point"]
    L += ["## S1: stored out-of-fold flags at draws (descriptive, not gated)", "",
          f"Draws in v2 recordings {_fmt(c1['draws_in_v2_recordings'])}: no marker {_fmt(c1['no_marker'])}, no window "
          f"{_fmt(c1['no_window'])}, matched {_fmt(c1['matched'])} ({_fmt(c1['windows_matched_by_more_than_one_draw'])} windows "
          f"matched by more than one draw). Flag = y_prob >= 0.50 on the single matched window (not N = 2).", "",
          "| Model | Recall vs SaO2 < 90 | Specificity vs SaO2 < 90 | Recall vs label | Specificity vs label |",
          "|---|---|---|---|---|"]
    for model, title in (("tinykan", "TinyKAN (Header-aligned model)"), ("logreg", "Logistic regression")):
        L.append(f"| {title} | {_fmt(pt[f'{model}_recall_vs_sao2'])} {_ci(r1, f'{model}_recall_vs_sao2')} | "
                 f"{_fmt(pt[f'{model}_specificity_vs_sao2'])} {_ci(r1, f'{model}_specificity_vs_sao2')} | "
                 f"{_fmt(pt[f'{model}_recall_vs_y_true'])} | {_fmt(pt[f'{model}_specificity_vs_y_true'])} |")
    L += ["", f"Label (window-median ScalcO2 < 90) vs SaO2 < 90 at matched draws: kappa {_fmt(pt['y_true_kappa_vs_sao2'])} "
          f"{_ci(r1, 'y_true_kappa_vs_sao2')}, sensitivity {_fmt(pt['y_true_sensitivity_vs_sao2'])}, specificity "
          f"{_fmt(pt['y_true_specificity_vs_sao2'])}.", ""]
    L += ["## Limits (fixed in advance)", "",
          "- Draws are taken on stable plateaus with the participant motionless. Agreement at draws does not describe "
          "transitions, where end-tidal O2 leads arterial saturation; many windows sit in transitions.",
          "- Borderline v2 windows (median 89-91) were dropped at export, so S1 under-represents draws near 90.",
          "- The PPG is not time-synced to the 2 Hz stream (PhysioNet); header alignment is better, not exact. S1 inherits it.",
          "- If ScalcO2 were fitted to the blood samples, agreement would be optimistic; the paper does not say it is.",
          "- The benchmark devices are devices under test in a lab study, of mixed quality and several finger sites; the "
          "median device is a benchmark, not a standard. No ISO 80601-2-61 limit is used as the gate.",
          "- Nothing here tests the MAX30102 or the board.",
          "- Device identity is inferred from label spelling (`.01`/`.02` kept separate); the integer-model sensitivity is "
          "the only check on that choice.",
          "- Bland-Altman limits ignore repeated draws per patient; the patient bootstrap is the only correction.", ""]
    return "\n".join(L) + "\n"


def analyse(data, args):
    draws, rows, pairs = data["draws"], data["rows"], data["pairs"]
    populations, band_rows = {}, []
    all_result, all_bands, devices_all = population(draws, pairs["primary"], rows, "all draws")
    populations["all"] = all_result
    band_rows += all_bands
    v2 = draws["in_v2"].to_numpy()
    v2_pairs = pairs["primary"].loc[pairs["primary"]["in_v2"].to_numpy()]
    v2_result, v2_bands, _ = population(draws.loc[v2].reset_index(drop=True), v2_pairs.reset_index(drop=True),
                                        rows.loc[rows["in_v2"].to_numpy()].reset_index(drop=True), "v2 subset")
    populations["v2"] = v2_result
    band_rows += v2_bands

    bench_p1 = devices_all.loc[devices_all["benchmark_p1"]]
    bench_p2 = devices_all.loc[devices_all["benchmark_p2"]]
    gate_result = gate(all_result["ARMS_S"], bench_p1["arms_d"].to_numpy(), all_result["kappa_S"],
                       bench_p2["kappa_d"].to_numpy())
    gate_payload = {"population": "all draws, all encounters", "smoke": args.smoke,
                    "prereg_commit": args.prereg_commit_full, **gate_result}

    _, model_summary = device_summary(pairs["primary"], "model", all_result["ARMS_S"], all_result["kappa_S"])
    _, plan_summary = device_summary(pairs["plan"], "device_label", all_result["ARMS_S"], all_result["kappa_S"])
    print(f"gate verdict computed ({'smoke' if args.smoke else 'full'})", flush=True)
    s1 = s1_results(data["s1"])
    return populations, band_rows, devices_all, gate_payload, {"integer_model": model_summary,
                                                               "validity_0_to_100": plan_summary}, s1


# ----------------------------------------------------------------------------- CLI

def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def configure(args):
    args.out = _resolve(args.out) if args.out else (SMOKE_ROOT / "smoke").resolve() if args.smoke else OUT_DEFAULT.resolve()
    if args.smoke:
        smoke_root = SMOKE_ROOT.resolve()
        if args.out == OUT_DEFAULT.resolve() or not args.out.is_relative_to(smoke_root) or args.out == smoke_root:
            raise SystemExit(f"REFUSED: a smoke run writes only into a directory under build/scalco2-vs-sao2-v1/, not {args.out}")
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; the output dir is write-once")
    if _git("cat-file", "-e", f"{args.prereg_commit}^{{commit}}") is None:
        raise SystemExit(f"REFUSED: --prereg-commit {args.prereg_commit} does not resolve to a commit")
    rel = PREREG.relative_to(ROOT).as_posix()
    if _git("cat-file", "-e", f"{args.prereg_commit}:{rel}") is None:
        raise SystemExit(f"REFUSED: {rel} is not in commit {args.prereg_commit}")
    args.prereg_commit_full = _git("rev-parse", f"{args.prereg_commit}^{{commit}}")
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prereg-commit", required=True, help="commit of docs/scalco2-vs-sao2-preregistration.md")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default outputs/scalco2-vs-sao2-v1; smoke: under build/scalco2-vs-sao2-v1/)")
    parser.add_argument("--smoke", action="store_true", help="first 20 encounters only; tests the code, not a result")
    args = configure(parser.parse_args(argv))
    print(f"out {args.out}" + (" (SMOKE)" if args.smoke else ""), flush=True)
    started = time.perf_counter()
    try:
        data = load(args.smoke)
    except StopAnalysis as stop:
        print(f"STOP: {stop}", flush=True)
        return 3
    print(f"inputs loaded and asserted, {time.perf_counter() - started:.1f} s", flush=True)
    populations, band_rows, devices_all, gate_payload, sensitivities, s1 = analyse(data, args)

    if args.out.exists():
        raise FileExistsError(f"{args.out} appeared during the run; the output dir is write-once")
    args.out.mkdir(parents=True)
    out = args.out
    draws = data["draws"][["encounter_id", "patient_id", "sample", "n_rows", "SaO2_draw", "ScalcO2_draw", "in_v2",
                           "n_valid_pulse_readings"]]
    write_csv(draws, out, "draws.csv")
    devices = devices_all.rename(columns={"device": "device_label", "n_p1": "valid_paired_readings_p1_range",
                                          "n_all": "valid_paired_readings_all_draws"})
    write_csv(devices, out, "devices.csv")
    write_csv(pd.DataFrame(band_rows), out, "bands.csv")
    write_csv(data["s1"], out, "s1_windows.csv")
    write_json(gate_payload, out, "gate.json")

    rel = PREREG.relative_to(ROOT).as_posix()
    summary = {
        "smoke": args.smoke,
        "smoke_note": "SMOKE RUN (first 20 encounters): tests the code only; not a result" if args.smoke else None,
        "smoke_encounters": data["smoke_encounters"],
        "prereg_commit": args.prereg_commit_full,
        "prereg": {"path": rel, "sha256": sha256(PREREG), "commits_touching": _git("log", "--format=%H %cI", "--", rel),
                   "worktree_unmodified": _git("status", "--porcelain", "--", rel) == ""},
        "git_head_at_run": _git("rev-parse", "HEAD"),
        "versions": environment(),
        "B": N_BOOT, "seed": SEED,
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": data["hashes"][k]} for k, p in P.items()},
        "also_read": {"encounter_csv": ENCOUNTER_CSV.relative_to(ROOT).as_posix(),
                      "hz2_csvs": "hz2_csv_path of the eligible-encounters row of each v2 recording (S1)"},
        "counts": {"bloodgas": data["draw_counts"], "pulseoximeter": data["pulse_counts"],
                   "benchmark_devices": {"P1": int(devices_all["benchmark_p1"].sum()),
                                         "P2": int(devices_all["benchmark_p2"].sum())}},
        "s1_counts": data["s1_counts"],
        "gate": gate_payload,
        "populations": populations,
        "sensitivities": sensitivities,
        "s1": s1,
        "interpretations": INTERPRETATIONS,
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                        for p in (Path(__file__).resolve(), ROOT / "tests" / "test_scalco2_vs_sao2.py",
                                  SCRIPTS / "lamaakal_metrics.py")},
        "outputs_sha256": {n: sha256(out / n) for n in ("draws.csv", "devices.csv", "bands.csv", "s1_windows.csv",
                                                        "gate.json")},
    }
    write_json(summary, out, "summary.json")
    new_file(out, "README.md").write_text(readme(_clean_json(summary), _clean_json(gate_payload), _clean_json(band_rows)),
                                          encoding="utf-8", newline="\n")
    print(f"wrote {', '.join(OUTPUT_FILES)} to {out} in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

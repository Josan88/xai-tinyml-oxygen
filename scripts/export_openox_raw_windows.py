"""Raw-waveform window export for the frozen OpenOximetry tables (host only).

For every row of the two frozen window tables this script exports the exact filtered
waveform the features were computed from, so that a raw-waveform comparator (1D-CNN) can be
trained on the same windows, labels and folds as TinyKAN:

  v1  outputs/openox_canonical_window_features.csv  (notebook 10, as-coded labels)
      IR only, 500 samples per window.
  v2  outputs/openox-v2/window_features.csv          (export_openox_windows_v2.py)
      IR and red, 500 samples per window, plus the per-window means of the 50 Hz
      resampled raw (unfiltered) IR and red, which ratio_of_ratios and
      ir_perfusion_index need and the bandpassed waveform does not carry.

Pipeline (imported, not re-implemented, from scripts/export_openox_windows_v2.py, whose IR
path is notebook 10's resample_and_filter split into two functions): IR channel chosen by
choose_ppg_channel, scipy.signal.resample of the whole recording to 50 Hz, 4th-order
Butterworth 0.5-8 Hz sosfiltfilt over the whole recording, window [start, start + 500)
with start = start_time_seconds * 50. Row keys (recording_id, start_time_seconds, label)
are copied from the frozen CSV in its row order, so row i of each .npz is row i of its CSV.

Join check (mandatory, over ALL rows): the 14 IR features (and ratio_of_ratios,
ir_perfusion_index for v2) are recomputed with the same feature code
(compute_window_features, two_wavelength_features) from
  (a) the float64 filtered waveform before storage (pipeline reproduction), and
  (b) the stored float32 arrays read back from the written .npz (what a model sees),
and compared with the frozen CSV. The script stops without writing the output directory
if any row is missing, any window falls outside its recording, or (a) fails its tolerance.

Writes (write-once) outputs/openox-raw-windows-v1/{v1_ir_windows.npz,
v2_ir_red_windows.npz, feature_reproduction.csv, export_report.json, SHA256SUMS, README.md}.
No board I/O; nothing in firmware/ or in an existing outputs/ directory is written.

Run: C:\\Python314\\python.exe scripts/export_openox_raw_windows.py [--workers 6]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_openox_windows_v2 as v2pipe  # noqa: E402  (the pipeline, imported unchanged)

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "openox-raw-windows-v1"
V1_CSV = ROOT / "outputs" / "openox_canonical_window_features.csv"
V2_CSV = ROOT / "outputs" / "openox-v2" / "window_features.csv"
PASSLIST_CSV = v2pipe.PASSLIST_CSV
ELIGIBLE_CSV = v2pipe.ELIGIBLE_CSV
NOTEBOOK_10 = ROOT / "notebooks" / "10_openox_canonical_window_export.ipynb"
PIPELINE_SCRIPT = SCRIPTS / "export_openox_windows_v2.py"

WINDOW = int(round(v2pipe.WINDOW_SECONDS * v2pipe.TARGET_RATE_HZ))      # 500
IR_FEATURES = v2pipe.IR_FEATURE_NAMES
RED_IR_FEATURES = v2pipe.EXTRA_FEATURE_NAMES
V2_FEATURES = IR_FEATURES + RED_IR_FEATURES

# Write-once directories (docs/analysis-lamaakal-style.md) plus the frozen tables and the
# analysis outputs this job reads. Hashed before and after; the run fails if any changes.
PROTECTED = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/ptt-ppg/subject-1-v1",
    ROOT / "outputs/ptt_ppg_motion_spike.csv",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
    ROOT / "outputs/analysis-lamaakal-style-v1",
    ROOT / "outputs/openox-v2",
    ROOT / "outputs/openox_canonical_window_features.csv",
)

# Tolerances, fixed before the run. "Scale-aware" relative error divides |delta| by a
# per-row scale in the feature's own units, so that features whose true value is near zero
# (the mean of a bandpassed window, skewness) are not judged by a vanishing denominator:
# amplitude-unit features use the window's frozen std, dimensionless shape features use 1,
# and every other feature uses its own frozen |value|.
AMPLITUDE_FEATURES = ("mean", "minimum", "maximum", "value_range", "rms", "std", "derivative_std")
SHAPE_FEATURES = ("skewness", "kurtosis")
TOL_FLOAT64 = 1e-9     # pipeline reproduction: must hold on every row and feature, else STOP
TOL_FLOAT32 = 1e-4     # stored float32 arrays: reported; rows above it are listed


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
    return {p.relative_to(ROOT).as_posix(): tree_hashes(p) for p in PROTECTED if p.exists()}


def feature_matrix(windows64, names, extra=None):
    """Features of each row with the frozen feature code; windows are float64."""
    rows = []
    for i, values in enumerate(windows64):
        features = v2pipe.compute_window_features(values, v2pipe.TARGET_RATE_HZ)
        if extra is not None:
            features.update(extra(i))
        rows.append([features[name] for name in names])
    return np.asarray(rows, dtype=np.float64).reshape(len(windows64), len(names))


def scale_aware_error(names, recomputed, frozen):
    """|delta| and |delta| / per-row scale, both n x n_features."""
    delta = np.abs(recomputed - frozen)
    scale = np.empty_like(frozen)
    std_column = frozen[:, names.index("std")]
    for j, name in enumerate(names):
        if name in AMPLITUDE_FEATURES:
            scale[:, j] = np.abs(std_column)
        elif name in SHAPE_FEATURES:
            scale[:, j] = 1.0
        else:
            scale[:, j] = np.abs(frozen[:, j])
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(delta == 0, 0.0, delta / np.where(scale > 0, scale, np.nan))
    relative = np.where(np.isfinite(relative), relative, np.inf)
    return delta, relative


# ----------------------------------------------------------------------------- per recording

def process_recording(task):
    """Filter one recording and cut the requested windows. Runs in a worker process.

    task = (recording_id, hea_path, v1_starts, v2_starts), starts in 50 Hz samples.
    Returns float32 windows, float64-path features, and per-window raw means for v2.
    """
    recording_id, hea_path, v1_starts, v2_starts = task
    hea_path = Path(hea_path)
    channel_count, source_rate_hz, channel_labels = v2pipe.read_header(hea_path)
    frames = v2pipe.read_ppg_frames(hea_path.with_suffix(".dat"), channel_count)
    ir_index = v2pipe.choose_ppg_channel(channel_labels)
    ir_resampled = v2pipe.resample_to_target(frames[:, ir_index], source_rate_hz)
    ir_filtered = v2pipe.bandpass(ir_resampled)
    n = len(ir_filtered)
    result = {"recording_id": recording_id, "samples_50hz": int(n), "source_rate_hz": source_rate_hz,
              "ir_channel": channel_labels[ir_index]}

    for starts in (v1_starts, v2_starts):
        if len(starts) and (np.min(starts) < 0 or np.max(starts) + WINDOW > n):
            raise RuntimeError(f"{recording_id}: window outside the filtered recording (n={n})")

    ir1 = np.stack([ir_filtered[s:s + WINDOW] for s in v1_starts]) if len(v1_starts) else np.empty((0, WINDOW))
    result["v1_ir"] = ir1.astype(np.float32)
    result["v1_features64"] = feature_matrix(ir1, list(IR_FEATURES))

    if len(v2_starts):
        red_index = v2pipe.find_channel(channel_labels, ("red signal", "red"))
        if red_index is None:
            raise RuntimeError(f"{recording_id}: no red channel in {channel_labels}")
        red_resampled = v2pipe.resample_to_target(frames[:, red_index], source_rate_hz)
        red_filtered = v2pipe.bandpass(red_resampled)
        if len(red_filtered) != n:
            raise RuntimeError(f"{recording_id}: red and IR lengths differ")
        ir2 = np.stack([ir_filtered[s:s + WINDOW] for s in v2_starts])
        red2 = np.stack([red_filtered[s:s + WINDOW] for s in v2_starts])
        ir_dc = np.array([np.mean(ir_resampled[s:s + WINDOW]) for s in v2_starts])
        red_dc = np.array([np.mean(red_resampled[s:s + WINDOW]) for s in v2_starts])

        def extra(i, s=v2_starts):
            start = s[i]
            return v2pipe.two_wavelength_features(red_resampled[start:start + WINDOW], red_filtered[start:start + WINDOW],
                                                  ir_resampled[start:start + WINDOW], ir_filtered[start:start + WINDOW])

        result["v2_ir"] = ir2.astype(np.float32)
        result["v2_red"] = red2.astype(np.float32)
        result["v2_ir_dc_mean"] = ir_dc
        result["v2_red_dc_mean"] = red_dc
        result["v2_features64"] = feature_matrix(ir2, list(V2_FEATURES), extra)
        result["red_channel"] = channel_labels[red_index]
    return result


def two_wavelength_from_stored(ir, red, ir_dc, red_dc):
    """two_wavelength_features from stored arrays: the function only uses the raw window mean.

    two_wavelength_features(red_raw, red_filtered, ir_raw, ir_filtered) reads np.mean(raw) and
    np.std(filtered). The stored per-window raw mean is passed as a length-1 raw array, so the
    same function computes the same quantities.
    """
    out = v2pipe.two_wavelength_features(np.array([red_dc]), red.astype(np.float64),
                                         np.array([ir_dc]), ir.astype(np.float64))
    return out


# ----------------------------------------------------------------------------- main

def load_tables():
    v1 = pd.read_csv(V1_CSV)
    v1["recording_id"] = v1["recording_id"].astype(str)
    v2 = pd.read_csv(V2_CSV)
    v2["recording_id"] = v2["recording_id"].astype(str)
    v2["patient_id"] = v2["patient_id"].astype(str)
    for name, frame in (("v1", v1), ("v2", v2)):
        start_index = np.rint(frame["start_time_seconds"].to_numpy() * v2pipe.TARGET_RATE_HZ).astype(np.int64)
        if not np.allclose(start_index / v2pipe.TARGET_RATE_HZ, frame["start_time_seconds"].to_numpy(), rtol=0, atol=1e-9):
            raise RuntimeError(f"{name}: a window start is not on the 50 Hz grid")
        if not np.allclose(frame["end_time_seconds"] - frame["start_time_seconds"], v2pipe.WINDOW_SECONDS):
            raise RuntimeError(f"{name}: a window is not 10 s long")
        if frame.duplicated(["recording_id", "start_time_seconds"]).any():
            raise RuntimeError(f"{name}: duplicate (recording_id, start_time_seconds) keys")
        frame["start_index"] = start_index
    return v1, v2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-recordings", type=int, default=None,
                        help="development only: process the first N recordings (refuses the default output)")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    if args.limit_recordings is not None and out == OUT_DEFAULT:
        print("STOP: --limit-recordings is for development and may not write the default output.")
        return 2

    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    before = protected_hashes()
    input_hashes = {p.relative_to(ROOT).as_posix(): sha256(p)
                    for p in (V1_CSV, V2_CSV, PASSLIST_CSV, ELIGIBLE_CSV, PIPELINE_SCRIPT, NOTEBOOK_10)}

    v1, v2 = load_tables()
    eligible = pd.read_csv(ELIGIBLE_CSV)
    passlist = pd.read_csv(PASSLIST_CSV)
    work = eligible.merge(passlist[["encounter_id", "reference_column"]], on="encounter_id", how="inner")
    hea_by_recording = dict(zip(work["encounter_id"].astype(str), work["ppg_hea_path"].astype(str)))

    recordings = sorted(set(v1["recording_id"]) | set(v2["recording_id"]))
    missing = [r for r in recordings if r not in hea_by_recording or not Path(hea_by_recording[r]).is_file()]
    if missing:
        print(f"STOP: {len(missing)} recordings have no waveform on disk, e.g. {missing[:3]}")
        return 3
    if args.limit_recordings is not None:
        recordings = recordings[: args.limit_recordings]
        v1 = v1[v1["recording_id"].isin(recordings)].reset_index(drop=True)
        v2 = v2[v2["recording_id"].isin(recordings)].reset_index(drop=True)

    v1_groups = {r: block.index.to_numpy() for r, block in v1.groupby("recording_id", sort=False)}
    v2_groups = {r: block.index.to_numpy() for r, block in v2.groupby("recording_id", sort=False)}
    tasks = []
    for r in recordings:
        i1 = v1_groups.get(r, np.empty(0, dtype=int))
        i2 = v2_groups.get(r, np.empty(0, dtype=int))
        tasks.append((r, hea_by_recording[r], v1.loc[i1, "start_index"].to_numpy(), v2.loc[i2, "start_index"].to_numpy()))

    n1, n2 = len(v1), len(v2)
    v1_ir = np.full((n1, WINDOW), np.nan, dtype=np.float32)
    v1_f64 = np.full((n1, len(IR_FEATURES)), np.nan)
    v2_ir = np.full((n2, WINDOW), np.nan, dtype=np.float32)
    v2_red = np.full((n2, WINDOW), np.nan, dtype=np.float32)
    v2_ir_dc = np.full(n2, np.nan)
    v2_red_dc = np.full(n2, np.nan)
    v2_f64 = np.full((n2, len(V2_FEATURES)), np.nan)
    v1_filled = np.zeros(n1, dtype=bool)
    v2_filled = np.zeros(n2, dtype=bool)
    recording_info = []

    print(f"Exporting {n1} v1 and {n2} v2 windows from {len(recordings)} recordings, {args.workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for k, result in enumerate(pool.map(process_recording, tasks), start=1):
            r = result["recording_id"]
            i1 = v1_groups.get(r, np.empty(0, dtype=int))
            i2 = v2_groups.get(r, np.empty(0, dtype=int))
            v1_ir[i1] = result["v1_ir"]
            v1_f64[i1] = result["v1_features64"]
            v1_filled[i1] = True
            if len(i2):
                v2_ir[i2] = result["v2_ir"]
                v2_red[i2] = result["v2_red"]
                v2_ir_dc[i2] = result["v2_ir_dc_mean"]
                v2_red_dc[i2] = result["v2_red_dc_mean"]
                v2_f64[i2] = result["v2_features64"]
                v2_filled[i2] = True
            recording_info.append({"recording_id": r, "samples_50hz": result["samples_50hz"],
                                   "source_rate_hz": result["source_rate_hz"], "ir_channel": result["ir_channel"],
                                   "red_channel": result.get("red_channel", ""), "v1_windows": int(len(i1)),
                                   "v2_windows": int(len(i2))})
            if k % 20 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} recordings", flush=True)

    if not (v1_filled.all() and v2_filled.all()):
        print(f"STOP: unmatched rows: v1 {int((~v1_filled).sum())}, v2 {int((~v2_filled).sum())}")
        return 4
    for name, array in (("v1_ir", v1_ir), ("v2_ir", v2_ir), ("v2_red", v2_red)):
        if not np.isfinite(array).all():
            print(f"STOP: non-finite values in {name}")
            return 4

    # --- write to a temporary sibling directory, then verify from disk, then rename.
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    np.savez_compressed(
        staging / "v1_ir_windows.npz",
        ir=v1_ir,
        row_index=np.arange(n1, dtype=np.int64),
        recording_id=v1["recording_id"].to_numpy().astype("U64"),
        start_time_seconds=v1["start_time_seconds"].to_numpy(np.float64),
        start_index=v1["start_index"].to_numpy(np.int64),
        label=v1["label"].to_numpy(np.int8),
        sample_rate_hz=np.float64(v2pipe.TARGET_RATE_HZ),
    )
    np.savez_compressed(
        staging / "v2_ir_red_windows.npz",
        ir=v2_ir,
        red=v2_red,
        ir_dc_mean=v2_ir_dc,
        red_dc_mean=v2_red_dc,
        row_index=np.arange(n2, dtype=np.int64),
        recording_id=v2["recording_id"].to_numpy().astype("U64"),
        patient_id=v2["patient_id"].to_numpy().astype("U64"),
        start_time_seconds=v2["start_time_seconds"].to_numpy(np.float64),
        start_index=v2["start_index"].to_numpy(np.int64),
        label=v2["label"].to_numpy(np.int8),
        sample_rate_hz=np.float64(v2pipe.TARGET_RATE_HZ),
    )

    # --- join check from the written files.
    print("Join check: recomputing features from the stored float32 arrays ...", flush=True)
    # Read fully and close: an open NpzFile holds a handle that blocks the rename on Windows.
    with np.load(staging / "v1_ir_windows.npz", allow_pickle=False) as handle:
        stored1 = {key: handle[key] for key in handle.files}
    with np.load(staging / "v2_ir_red_windows.npz", allow_pickle=False) as handle:
        stored2 = {key: handle[key] for key in handle.files}
    key_checks = {
        "v1_keys_equal_csv_in_order": bool(
            np.array_equal(stored1["recording_id"], v1["recording_id"].to_numpy().astype("U64"))
            and np.array_equal(stored1["start_time_seconds"], v1["start_time_seconds"].to_numpy(np.float64))
            and np.array_equal(stored1["label"], v1["label"].to_numpy(np.int8))),
        "v2_keys_equal_csv_in_order": bool(
            np.array_equal(stored2["recording_id"], v2["recording_id"].to_numpy().astype("U64"))
            and np.array_equal(stored2["start_time_seconds"], v2["start_time_seconds"].to_numpy(np.float64))
            and np.array_equal(stored2["label"], v2["label"].to_numpy(np.int8))),
        "v1_rows": int(stored1["ir"].shape[0]), "v2_rows": int(stored2["ir"].shape[0]),
    }
    v1_f32 = feature_matrix(stored1["ir"].astype(np.float64), list(IR_FEATURES))
    ir2s, red2s, irdc, reddc = stored2["ir"], stored2["red"], stored2["ir_dc_mean"], stored2["red_dc_mean"]
    v2_f32 = feature_matrix(ir2s.astype(np.float64), list(V2_FEATURES),
                            lambda i: two_wavelength_from_stored(ir2s[i], red2s[i], irdc[i], reddc[i]))

    reproduction_rows, report_features, failures = [], {}, []
    for table, frame, names, f64, f32 in (("v1", v1, list(IR_FEATURES), v1_f64, v1_f32),
                                          ("v2", v2, list(V2_FEATURES), v2_f64, v2_f32)):
        frozen = frame.loc[:, names].to_numpy(np.float64)
        d64, r64 = scale_aware_error(names, f64, frozen)
        d32, r32 = scale_aware_error(names, f32, frozen)
        report_features[table] = {}
        for j, name in enumerate(names):
            row = {
                "table": table, "feature": name, "rows": int(len(frame)),
                "scale": ("window std" if name in AMPLITUDE_FEATURES else "1 (dimensionless)" if name in SHAPE_FEATURES else "own |value|"),
                "float64_max_abs_delta": float(d64[:, j].max()),
                "float64_max_scaled_delta": float(r64[:, j].max()),
                "float64_rows_over_tol": int((r64[:, j] > TOL_FLOAT64).sum()),
                "float32_max_abs_delta": float(d32[:, j].max()),
                "float32_max_scaled_delta": float(r32[:, j].max()),
                "float32_rows_over_tol": int((r32[:, j] > TOL_FLOAT32).sum()),
                "float32_rows_not_exact": int((d32[:, j] > 0).sum()),
                "frozen_abs_max": float(np.abs(frozen[:, j]).max()),
            }
            reproduction_rows.append(row)
            report_features[table][name] = {k: row[k] for k in row if k.startswith(("float64", "float32"))}
            if row["float64_rows_over_tol"]:
                failures.append(f"{table}.{name}: {row['float64_rows_over_tol']} rows over {TOL_FLOAT64} (float64 path)")
    pd.DataFrame(reproduction_rows).to_csv(staging / "feature_reproduction.csv", index=False,
                                           float_format="%.6g", lineterminator="\n")

    if failures or not (key_checks["v1_keys_equal_csv_in_order"] and key_checks["v2_keys_equal_csv_in_order"]):
        print("STOP: the export does not reproduce the frozen tables:")
        for line in failures:
            print("  " + line)
        print(f"Staging directory left for inspection: {staging}")
        return 5

    after = protected_hashes()
    if after != before:
        print("STOP: a protected directory changed during the run")
        return 6

    import scipy
    files = {f.name: sha256(f) for f in sorted(staging.iterdir()) if f.is_file()}
    sizes = {f.name: f.stat().st_size for f in sorted(staging.iterdir()) if f.is_file()}
    report = {
        "created_utc": created,
        "duration_seconds": None,
        "script": "scripts/export_openox_raw_windows.py",
        "scope": "host only; no board I/O; no firmware, header or existing outputs/ writes; v3 out of scope",
        "pipeline": {
            "source": "imported from scripts/export_openox_windows_v2.py (IR path identical to notebook 10 resample_and_filter)",
            "ir_channel": "choose_ppg_channel: first of 'ir signal', 'ir', 'red signal', 'red'",
            "resample": "scipy.signal.resample of the whole recording to 50 Hz",
            "filter": "Butterworth order 4, 0.5-8 Hz bandpass, sosfiltfilt over the whole recording",
            "window": "samples [start_index, start_index + 500), start_index = round(start_time_seconds * 50)",
            "v2_red": "same resample and filter as IR; v2 only",
            "v2_dc_means": "ir_dc_mean / red_dc_mean = mean of the 50 Hz resampled raw (unfiltered) window; "
                           "ratio_of_ratios and ir_perfusion_index divide by them",
        },
        "arrays": {
            "v1_ir_windows.npz": {"ir": "float32 (52090, 500) filtered IR", "keys": "row_index, recording_id, start_time_seconds, start_index, label"},
            "v2_ir_red_windows.npz": {"ir": "float32 (48662, 500) filtered IR", "red": "float32 (48662, 500) filtered red",
                                      "ir_dc_mean": "float64 (48662,)", "red_dc_mean": "float64 (48662,)",
                                      "keys": "row_index, recording_id, patient_id, start_time_seconds, start_index, label"},
            "row_order": "row i of each .npz is row i of its frozen CSV (keys copied in order and checked)",
        },
        "rows": {"v1": n1, "v2": n2},
        "recordings": {"v1": int(v1["recording_id"].nunique()), "v2": int(v2["recording_id"].nunique()),
                       "union": len(recordings)},
        "key_checks": key_checks,
        "join_check": {
            "method": "features recomputed with compute_window_features / two_wavelength_features over all rows; "
                      "float64 = filtered waveform before storage; float32 = arrays read back from the written .npz "
                      "and cast to float64 before the feature code",
            "scaled_delta": "|delta| / scale; scale = frozen window std for mean, minimum, maximum, value_range, rms, std, "
                            "derivative_std; 1 for skewness, kurtosis; own frozen |value| otherwise",
            "tolerance_float64_must_hold": TOL_FLOAT64,
            "tolerance_float32_reported": TOL_FLOAT32,
            "all_rows_matched": True,
            "float64_path_passes": True,
            "per_feature": report_features,
            "max_abs_delta_float32_over_all_features": {
                t: max(v["float32_max_abs_delta"] for v in report_features[t].values()) for t in report_features},
            "max_scaled_delta_float32_over_all_features": {
                t: max(v["float32_max_scaled_delta"] for v in report_features[t].values()) for t in report_features},
            "max_scaled_delta_float64_over_all_features": {
                t: max(v["float64_max_scaled_delta"] for v in report_features[t].values()) for t in report_features},
        },
        "recordings_detail": sorted(recording_info, key=lambda r: r["recording_id"]),
        "inputs_sha256": input_hashes,
        "outputs_sha256": files,
        "outputs_bytes": sizes,
        "protected_unchanged": True,
        "protected_sha256": after,
        "environment": {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    }
    report["duration_seconds"] = round(time.perf_counter() - started, 1)
    (staging / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (staging / "README.md").write_text(
        "# openox-raw-windows-v1\n\n"
        "Filtered PPG windows for every row of the two frozen OpenOximetry window tables, written by\n"
        "`scripts/export_openox_raw_windows.py`. Row i of `v1_ir_windows.npz` is row i of\n"
        "`outputs/openox_canonical_window_features.csv`; row i of `v2_ir_red_windows.npz` is row i of\n"
        "`outputs/openox-v2/window_features.csv`. The feature reproduction check over all rows is in\n"
        "`feature_reproduction.csv` and `export_report.json`. The `.npz` files are not committed (size);\n"
        "regenerate them with the script and compare against `SHA256SUMS`. Write-up: `docs/analysis-1dcnn.md`.\n",
        encoding="utf-8", newline="\n")
    all_files = {f.name: sha256(f) for f in sorted(staging.iterdir()) if f.is_file()}
    (staging / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in all_files.items()),
                                        encoding="utf-8", newline="\n")
    staging.rename(out)
    print(json.dumps({"rows": report["rows"], "recordings": report["recordings"],
                      "max_abs_delta_float32": report["join_check"]["max_abs_delta_float32_over_all_features"],
                      "max_scaled_delta_float32": report["join_check"]["max_scaled_delta_float32_over_all_features"],
                      "max_scaled_delta_float64": report["join_check"]["max_scaled_delta_float64_over_all_features"],
                      "bytes": sizes, "seconds": report["duration_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

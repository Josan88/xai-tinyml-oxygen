"""Independent audit of the OpenOximetry label alignment used by notebook 10.

Notebook 10 shifts each PPG window by signed_delta = ppg_start_time_seconds -
hz2_first_time_seconds (first 2 Hz ROW timestamp) but measures reference
elapsed time from the first NON-NULL ScalcO2 row. This script re-derives, from
raw data only, (1) the leading-null gap per recording, (2) whether the frozen
CSV median_spo2 is reproduced under the as-coded alignment versus header
alignment (elapsed from the first 2 Hz row), and (3) a physiological check that
uses the red/IR ratio-of-ratios only to compare alignments, never to tune one.

PhysioNet OpenOximetry v1.1.1 states the raw PPG is not time-synced to the other
waveforms, so header alignment is better than the as-coded alignment, not exact.

Writes (write-once): outputs/label-alignment-v1/{report.json, README.md,
per_recording.csv, lag_scan.csv}. Reads frozen inputs only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ELIGIBLE_CSV = OUTPUTS / "openox_ppg_eligible_encounters.csv"
PASSLIST_CSV = OUTPUTS / "openox_reference_encounter_passlist.csv"
FROZEN_CSV = OUTPUTS / "openox_canonical_window_features.csv"
OUT_DIR = OUTPUTS / "label-alignment-v1"

REFERENCE_COLUMN = "ScalcO2"
MIN_REFERENCE_SAMPLES_PER_WINDOW = 10
LOW_SPO2_THRESHOLD = 90.0
BORDERLINE_MARGIN = 1.0
RATIO_BAND_HZ = (0.5, 5.0)
RATIO_FILTER_ORDER = 4
MATCH_TOLERANCE = 1e-6
LAGS_SECONDS = np.arange(-600.0, 600.0 + 1e-9, 5.0)
GAP_FLAG_SECONDS = 10.0


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_ppg_header(hea_path: Path) -> tuple[int, float, int, str, list[str]]:
    lines = [line.strip() for line in hea_path.open("r", encoding="utf-8", errors="ignore") if line.strip()]
    tokens = lines[0].split()
    channel_count = int(tokens[1])
    labels = [" ".join(line.split()[8:]) for line in lines[1 : 1 + channel_count]]
    return channel_count, float(tokens[2]), int(tokens[3]), tokens[4], labels


def channel_index(labels: list[str], names: tuple[str, ...]) -> int:
    lowered = [label.strip().lower() for label in labels]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    raise ValueError(f"None of {names} in {labels}")


def read_frames(dat_path: Path, channel_count: int) -> np.ndarray:
    raw = np.fromfile(dat_path, dtype=np.int32)
    if raw.size % channel_count:
        raise ValueError(f"{dat_path.name}: {raw.size} int32 values not divisible by {channel_count} channels")
    return raw.reshape(-1, channel_count)


def seconds_of_day(clock: str) -> float:
    hours, minutes, seconds = clock.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def window_medians(elapsed: np.ndarray, values: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """Median and count of values with starts <= elapsed < ends (notebook 10 semantics)."""
    lo = np.searchsorted(elapsed, starts, side="left")
    hi = np.searchsorted(elapsed, ends, side="left")
    counts = hi - lo
    width = int(counts.max()) if len(counts) else 0
    if width == 0:
        return np.full(len(starts), np.nan), np.full(len(starts), np.nan), counts
    offsets = np.arange(width)
    index = lo[:, None] + offsets[None, :]
    valid = offsets[None, :] < counts[:, None]
    gathered = np.where(valid, values[np.clip(index, 0, len(values) - 1)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows for empty windows
        medians = np.nanmedian(gathered, axis=1)
    minimums = np.where(counts > 0, np.min(np.where(valid, gathered, np.inf), axis=1), np.nan)
    return medians, minimums, counts


def spearman_per_recording(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    values = {}
    for recording_id, group in frame.groupby("recording_id"):
        group = group[np.isfinite(group[x]) & np.isfinite(group[y])]
        if len(group) < 5 or group[x].nunique() < 2 or group[y].nunique() < 2:
            continue
        values[recording_id] = stats.spearmanr(group[x], group[y]).statistic
    return pd.Series(values, dtype=float)


def labelled_auc(score: np.ndarray, median_ref: np.ndarray, counts: np.ndarray) -> tuple[float, int, float]:
    keep = (
        np.isfinite(score)
        & np.isfinite(median_ref)
        & (counts >= MIN_REFERENCE_SAMPLES_PER_WINDOW)
        & ~((median_ref >= LOW_SPO2_THRESHOLD - BORDERLINE_MARGIN) & (median_ref <= LOW_SPO2_THRESHOLD + BORDERLINE_MARGIN))
    )
    labels = (median_ref[keep] < LOW_SPO2_THRESHOLD).astype(int)
    if labels.min() == labels.max():
        return float("nan"), int(keep.sum()), float("nan")
    return float(roc_auc_score(labels, score[keep])), int(keep.sum()), float(labels.mean())


def render_readme(hermes: dict, measured: dict, report: dict) -> str:
    rows = [
        ("Leading-null gap, median (s)", hermes["gap_median_s"], f"{measured['gap_median_s']:.1f}"),
        ("Leading-null gap, max (s)", hermes["gap_max_s"], f"{measured['gap_max_s']:.1f}"),
        ("Recordings with gap > 10 s", f"{hermes['recordings_gap_gt_10s']}/162", f"{measured['recordings_gap_gt_10s']}/{measured['recordings']}"),
        ("CSV median_spo2 reproduced, as coded", f"{hermes['reproduce_as_coded']}/162", f"{measured['reproduce_as_coded']}/{measured['recordings']}"),
        ("CSV median_spo2 reproduced, header-aligned", f"{hermes['reproduce_header']}/162", f"{measured['reproduce_header']}/{measured['recordings']}"),
        ("Ratio-of-ratios AUC (<90), exported labels", hermes["auc_exported"], f"{measured['auc_exported']:.3f}"),
        ("Ratio-of-ratios AUC (<90), header-aligned", hermes["auc_header"], f"{measured['auc_header']:.3f}"),
        ("Median within-recording Spearman, exported", hermes["spearman_exported"], f"{measured['spearman_exported']:.3f}"),
        ("Median within-recording Spearman, header-aligned", hermes["spearman_header"], f"{measured['spearman_header']:.3f}"),
        ("Lag-scan peak (AUC / Spearman), s", hermes["lag_peak"], f"{measured['lag_peak_auc_s']:+.0f} / {measured['lag_peak_spearman_s']:+.0f}"),
    ]
    table = "\n".join(f"| {name} | {claimed} | {mine} |" for name, claimed, mine in rows)
    return f"""# OpenOximetry label-alignment audit (v1, write-once)

Generated by `scripts/audit_openox_label_alignment.py` on {report['created_utc'][:10]} from raw PhysioNet
OpenOximetry v1.1.1 files plus the frozen `outputs/openox_ppg_eligible_encounters.csv`,
`outputs/openox_reference_encounter_passlist.csv` and `outputs/openox_canonical_window_features.csv`.

**Defect checked.** Notebook 10 adds `signed_delta = ppg_start_time_seconds - hz2_first_time_seconds`
(first 2 Hz *row*) to each PPG window, but measures reference `elapsed_seconds` from the first *non-null*
ScalcO2 row. Each window's reference is therefore read later by that recording's leading-null gap.
Header alignment measures elapsed time from the first 2 Hz row instead.

| Quantity | Reported (Hermes) | This audit |
|---|---|---|
{table}

Ratio-of-ratios R = (std(bandpassed red)/mean(raw red)) / (std(bandpassed IR)/mean(raw IR)) per frozen
10 s window at the native 86 Hz, 4th-order Butterworth 0.5-5 Hz `sosfiltfilt`. R is used only to compare
the two alignments; it was not used to choose or tune an alignment. Lag sign: positive = reference read
later than header alignment.

Caveat: PhysioNet v1.1.1 states the raw PPG is not time-synced to the other waveforms. Header alignment is
better than the as-coded alignment, not exact.

Files: `report.json` (all numbers), `per_recording.csv` (gap and reproduction per recording),
`lag_scan.csv` (AUC and median within-recording Spearman for lags -600..+600 s, 5 s step).
"""


def main() -> int:
    if (OUT_DIR / "report.json").exists():
        raise FileExistsError(f"Write-once output exists: {OUT_DIR / 'report.json'}")

    eligible = pd.read_csv(ELIGIBLE_CSV)
    passlist = pd.read_csv(PASSLIST_CSV)
    work = eligible.merge(passlist[["encounter_id", "reference_column"]], on="encounter_id", how="inner")
    if set(work["reference_column"]) != {REFERENCE_COLUMN}:
        raise RuntimeError(f"Unexpected reference columns: {set(work['reference_column'])}")
    frozen = pd.read_csv(FROZEN_CSV)
    frozen["recording_id"] = frozen["recording_id"].astype(str)
    frozen_ids = set(frozen["recording_id"])

    sos = signal.butter(RATIO_FILTER_ORDER, RATIO_BAND_HZ, btype="bandpass", fs=86.0, output="sos")
    recording_rows = []
    window_frames = []
    references = {}

    for row in work.itertuples(index=False):
        recording_id = str(row.encounter_id)
        hea_path = Path(str(row.ppg_hea_path))
        hz2_path = Path(str(row.hz2_csv_path))
        channel_count, fs, sample_count, start_clock, labels = read_ppg_header(hea_path)

        ref = pd.read_csv(hz2_path, usecols=["Timestamp", REFERENCE_COLUMN])
        ref["Timestamp"] = pd.to_datetime(ref["Timestamp"], errors="coerce")
        ref[REFERENCE_COLUMN] = pd.to_numeric(ref[REFERENCE_COLUMN], errors="coerce")
        ref = ref.loc[ref["Timestamp"].notna()].reset_index(drop=True)
        first_row_time = ref["Timestamp"].iloc[0]
        non_null = ref.loc[ref[REFERENCE_COLUMN].notna()].reset_index(drop=True)
        first_value_time = non_null["Timestamp"].iloc[0]
        gap_seconds = (first_value_time - first_row_time).total_seconds()

        # Independent re-derivation of the header offset, checked against the frozen eligible CSV.
        first_row_clock = first_row_time.hour * 3600 + first_row_time.minute * 60 + first_row_time.second + first_row_time.microsecond / 1e6
        derived_delta = seconds_of_day(start_clock) - first_row_clock
        signed_delta = float(row.ppg_start_time_seconds - row.hz2_first_time_seconds)
        if abs(derived_delta - signed_delta) > 1e-3:
            raise RuntimeError(f"{recording_id}: derived delta {derived_delta} != eligible CSV delta {signed_delta}")
        if pd.Timestamp(row.hz2_first_timestamp) != first_row_time:
            raise RuntimeError(f"{recording_id}: hz2_first_timestamp is not the first 2 Hz row")

        elapsed_as_coded = (non_null["Timestamp"] - first_value_time).dt.total_seconds().to_numpy()
        elapsed_header = (non_null["Timestamp"] - first_row_time).dt.total_seconds().to_numpy()
        values = non_null[REFERENCE_COLUMN].to_numpy(dtype=np.float64)
        references[recording_id] = (elapsed_header, values, signed_delta)

        record = {
            "recording_id": recording_id,
            "in_frozen_csv": recording_id in frozen_ids,
            "source_fs_hz": fs,
            "ppg_samples": sample_count,
            "ppg_duration_s": sample_count / fs,
            "signed_delta_s": signed_delta,
            "leading_null_gap_s": gap_seconds,
            "hz2_rows": len(ref),
            "reference_non_null_rows": len(non_null),
        }
        if recording_id not in frozen_ids:
            recording_rows.append(record)
            continue

        windows = frozen.loc[frozen["recording_id"] == recording_id].copy()
        starts = windows["start_time_seconds"].to_numpy() + signed_delta
        ends = windows["end_time_seconds"].to_numpy() + signed_delta
        med_coded, min_coded, n_coded = window_medians(elapsed_as_coded, values, starts, ends)
        med_header, min_header, n_header = window_medians(elapsed_header, values, starts, ends)
        csv_median = windows["median_spo2"].to_numpy()
        csv_min = windows["min_spo2"].to_numpy()
        match_coded = (n_coded >= MIN_REFERENCE_SAMPLES_PER_WINDOW) & (np.abs(med_coded - csv_median) <= MATCH_TOLERANCE) & (np.abs(min_coded - csv_min) <= MATCH_TOLERANCE)
        match_header = (n_header >= MIN_REFERENCE_SAMPLES_PER_WINDOW) & (np.abs(med_header - csv_median) <= MATCH_TOLERANCE) & (np.abs(min_header - csv_min) <= MATCH_TOLERANCE)

        frames = read_frames(hea_path.with_suffix(".dat"), channel_count)
        if len(frames) != sample_count:
            raise RuntimeError(f"{recording_id}: header says {sample_count} frames, data has {len(frames)}")
        red_raw = frames[:, channel_index(labels, ("red signal", "red"))].astype(np.float64)
        ir_raw = frames[:, channel_index(labels, ("ir signal", "ir"))].astype(np.float64)
        red_bp = signal.sosfiltfilt(sos, red_raw)
        ir_bp = signal.sosfiltfilt(sos, ir_raw)
        ratios = []
        for start_s, end_s in zip(windows["start_time_seconds"], windows["end_time_seconds"]):
            a, b = int(round(start_s * fs)), int(round(end_s * fs))
            red_mean, ir_mean = red_raw[a:b].mean(), ir_raw[a:b].mean()
            if b > len(red_raw) or red_mean <= 0 or ir_mean <= 0:
                ratios.append(np.nan)
                continue
            ratios.append((red_bp[a:b].std() / red_mean) / (ir_bp[a:b].std() / ir_mean))
        windows["ratio_of_ratios"] = np.asarray(ratios, dtype=np.float64)
        windows["median_header"] = med_header
        windows["count_header"] = n_header
        windows["median_as_coded"] = med_coded
        windows["match_as_coded"] = match_coded
        windows["match_header"] = match_header
        window_frames.append(windows[["recording_id", "start_time_seconds", "end_time_seconds", "median_spo2", "label",
                                      "median_as_coded", "median_header", "count_header", "match_as_coded",
                                      "match_header", "ratio_of_ratios"]])
        record.update(
            {
                "frozen_windows": len(windows),
                "windows_reproduced_as_coded": int(match_coded.sum()),
                "windows_reproduced_header": int(match_header.sum()),
                "reproduces_as_coded": bool(match_coded.all()),
                "reproduces_header": bool(match_header.all()),
            }
        )
        recording_rows.append(record)
        print(f"{recording_id[:12]} gap={gap_seconds:7.1f}s coded={match_coded.mean():.3f} header={match_header.mean():.3f}", flush=True)

    per_recording = pd.DataFrame(recording_rows)
    windows = pd.concat(window_frames, ignore_index=True)
    in_frozen = per_recording.loc[per_recording["in_frozen_csv"]]
    gaps = in_frozen["leading_null_gap_s"]

    ratio = windows["ratio_of_ratios"].to_numpy()
    auc_exported = float(roc_auc_score(windows["label"][np.isfinite(ratio)], ratio[np.isfinite(ratio)]))
    auc_header, n_header_auc, prev_header = labelled_auc(ratio, windows["median_header"].to_numpy(), windows["count_header"].to_numpy())
    rho_exported = spearman_per_recording(windows, "ratio_of_ratios", "median_spo2")
    header_valid = windows.loc[windows["count_header"] >= MIN_REFERENCE_SAMPLES_PER_WINDOW]
    rho_header = spearman_per_recording(header_valid, "ratio_of_ratios", "median_header")

    # Lag scan relative to header alignment: positive lag = reference taken from later than the PPG window.
    lag_rows = []
    grouped = {rid: group for rid, group in windows.groupby("recording_id")}
    for lag in LAGS_SECONDS:
        medians_all, counts_all, ratios_all = [], [], []
        rhos = []
        for recording_id, group in grouped.items():
            elapsed, values, delta = references[recording_id]
            starts = group["start_time_seconds"].to_numpy() + delta + lag
            ends = group["end_time_seconds"].to_numpy() + delta + lag
            medians, _, counts = window_medians(elapsed, values, starts, ends)
            r = group["ratio_of_ratios"].to_numpy()
            ok = (counts >= MIN_REFERENCE_SAMPLES_PER_WINDOW) & np.isfinite(r) & np.isfinite(medians)
            if ok.sum() >= 5 and np.unique(r[ok]).size > 1 and np.unique(medians[ok]).size > 1:
                rhos.append(stats.spearmanr(r[ok], medians[ok]).statistic)
            medians_all.append(medians)
            counts_all.append(counts)
            ratios_all.append(r)
        auc, n_auc, _ = labelled_auc(np.concatenate(ratios_all), np.concatenate(medians_all), np.concatenate(counts_all))
        lag_rows.append({"lag_s": float(lag), "pooled_auc_lt90": auc, "auc_windows": n_auc,
                         "median_within_recording_spearman": float(np.median(rhos)), "spearman_recordings": len(rhos)})
    lag_scan = pd.DataFrame(lag_rows)
    best_auc_lag = float(lag_scan.loc[lag_scan["pooled_auc_lt90"].idxmax(), "lag_s"])
    best_rho_lag = float(lag_scan.loc[lag_scan["median_within_recording_spearman"].idxmin(), "lag_s"])

    hermes = {
        "gap_median_s": 234, "gap_max_s": 1520, "recordings_gap_gt_10s": 128, "recordings": 162,
        "reproduce_as_coded": 162, "reproduce_header": 17, "auc_exported": 0.619, "auc_header": 0.835,
        "spearman_exported": -0.30, "spearman_header": -0.81, "lag_peak": "near 0 / -30 s",
    }
    measured = {
        "gap_median_s": float(gaps.median()), "gap_max_s": float(gaps.max()),
        "recordings_gap_gt_10s": int((gaps > GAP_FLAG_SECONDS).sum()), "recordings": int(len(in_frozen)),
        "reproduce_as_coded": int(in_frozen["reproduces_as_coded"].sum()),
        "reproduce_header": int(in_frozen["reproduces_header"].sum()),
        "auc_exported": auc_exported, "auc_header": auc_header,
        "spearman_exported": float(rho_exported.median()), "spearman_header": float(rho_header.median()),
        "lag_peak_auc_s": best_auc_lag, "lag_peak_spearman_s": best_rho_lag,
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "notebook 10 measures reference elapsed time from the first non-null ScalcO2 row while signed_delta is "
                 "relative to the first 2 Hz row, shifting each window's reference later by the leading-null gap",
        "hermes_reported": hermes,
        "measured": measured,
        "gap_seconds": {"min": float(gaps.min()), "q25": float(gaps.quantile(0.25)), "median": float(gaps.median()),
                        "q75": float(gaps.quantile(0.75)), "max": float(gaps.max()),
                        "recordings_gap_zero": int((gaps == 0).sum()),
                        "recordings_gap_gt_10s": int((gaps > GAP_FLAG_SECONDS).sum())},
        "gap_all_173_work_encounters": {"median": float(per_recording["leading_null_gap_s"].median()),
                                         "max": float(per_recording["leading_null_gap_s"].max())},
        "reproduction": {
            "criterion": f"all frozen windows of a recording match median_spo2 and min_spo2 within {MATCH_TOLERANCE} with >= {MIN_REFERENCE_SAMPLES_PER_WINDOW} reference samples",
            "frozen_windows": int(len(windows)),
            "windows_reproduced_as_coded": int(windows["match_as_coded"].sum()),
            "windows_reproduced_header": int(windows["match_header"].sum()),
            "recordings_reproduced_as_coded": measured["reproduce_as_coded"],
            "recordings_reproduced_header": measured["reproduce_header"],
            "header_reproducing_recordings_with_zero_gap": int((in_frozen["reproduces_header"] & (in_frozen["leading_null_gap_s"] == 0)).sum()),
        },
        "ratio_of_ratios": {
            "definition": "(std(bandpassed red)/mean(raw red)) / (std(bandpassed IR)/mean(raw IR)) per frozen 10 s window, native 86 Hz",
            "filter": f"Butterworth order {RATIO_FILTER_ORDER} bandpass {RATIO_BAND_HZ} Hz, sosfiltfilt over whole recording",
            "finite_windows": int(np.isfinite(ratio).sum()),
            "auc_lt90_exported_labels": auc_exported,
            "auc_lt90_header_aligned": auc_header,
            "header_aligned_auc_windows": n_header_auc,
            "header_aligned_prevalence": prev_header,
            "median_within_recording_spearman_exported": float(rho_exported.median()),
            "median_within_recording_spearman_header": float(rho_header.median()),
            "spearman_recordings_exported": int(len(rho_exported)),
            "spearman_recordings_header": int(len(rho_header)),
            "use": "comparison of alignments only; never used to choose or tune an alignment",
        },
        "lag_scan": {"lags_s": [float(LAGS_SECONDS[0]), float(LAGS_SECONDS[-1]), 5.0],
                     "sign": "positive lag = reference taken later than header alignment",
                     "best_auc_lag_s": best_auc_lag, "best_auc": float(lag_scan["pooled_auc_lt90"].max()),
                     "best_spearman_lag_s": best_rho_lag,
                     "best_spearman": float(lag_scan["median_within_recording_spearman"].min()),
                     "at_zero": lag_scan.loc[lag_scan["lag_s"] == 0.0].iloc[0].to_dict()},
        "caveat": "PhysioNet OpenOximetry v1.1.1: raw PPG is not time-synced to other waveforms; header alignment is better, not exact.",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (ELIGIBLE_CSV, PASSLIST_CSV, FROZEN_CSV)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_recording.to_csv(OUT_DIR / "per_recording.csv", index=False)
    lag_scan.to_csv(OUT_DIR / "lag_scan.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text(render_readme(hermes, measured, report), encoding="utf-8")
    print(json.dumps({"hermes": hermes, "measured": measured}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

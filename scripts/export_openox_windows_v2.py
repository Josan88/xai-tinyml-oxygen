"""OpenOximetry v2 window export: notebook 10 pipeline with header-aligned reference time.

Identical to notebooks/10_openox_canonical_window_export.ipynb (50 Hz scipy.signal.resample,
4th-order Butterworth 0.5-8 Hz sosfiltfilt over the whole recording, 10 s window, 5 s stride,
>= 10 reference samples per window, >= 5 windows per encounter, median ScalcO2 < 90 label with
the +/-1 borderline drop, same 14 IR features) EXCEPT:

- reference elapsed time is measured from the first 2 Hz ROW (header alignment), matching the
  row-based signed_delta; notebook 10 measured it from the first non-null ScalcO2 row
  (see outputs/label-alignment-v1/);
- red is resampled and filtered identically to IR, adding ratio_of_ratios and ir_perfusion_index;
- windows inside which a Red Drive or IR Drive channel changes are dropped and counted.

Writes (write-once) outputs/openox-v2/{window_features.csv, export_audit.csv,
recording_summary.csv, export_report.json}.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
PASSLIST_CSV = OUTPUTS_DIR / "openox_reference_encounter_passlist.csv"
ELIGIBLE_CSV = OUTPUTS_DIR / "openox_ppg_eligible_encounters.csv"
FROZEN_CSV = OUTPUTS_DIR / "openox_canonical_window_features.csv"
ENCOUNTER_CSV = ROOT / "data/physionet.org/files/openox-repo/1.1.1/encounter.csv"
OUT_DIR = OUTPUTS_DIR / "openox-v2"
FEATURE_CSV = OUT_DIR / "window_features.csv"

TARGET_RATE_HZ = 50.0
LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 8.0
FILTER_ORDER = 4
WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 5.0
LOW_SPO2_THRESHOLD = 90.0
BORDERLINE_MARGIN = 1.0
MIN_REFERENCE_SAMPLES_PER_WINDOW = 10
MIN_WINDOWS_PER_ENCOUNTER = 5

IR_FEATURE_NAMES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
EXTRA_FEATURE_NAMES = ("ratio_of_ratios", "ir_perfusion_index")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_ppg_channel(channel_labels: list[str]) -> int:
    lowered = [str(label).strip().lower() for label in channel_labels]
    for preferred in ("ir signal", "ir", "red signal", "red"):
        for idx, label in enumerate(lowered):
            if preferred == label:
                return idx
    raise ValueError(f"Could not choose a signal channel from labels: {channel_labels}")


def find_channel(channel_labels: list[str], names: tuple[str, ...]) -> int | None:
    lowered = [str(label).strip().lower() for label in channel_labels]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return None


def read_ppg_frames(dat_path: Path, channel_count: int) -> np.ndarray:
    raw = np.memmap(dat_path, dtype=np.int32, mode="r")
    return np.asarray(raw).reshape(-1, channel_count)


def resample_to_target(values: np.ndarray, source_rate_hz: float) -> np.ndarray:
    target_length = int(round(len(values) * TARGET_RATE_HZ / source_rate_hz))
    return signal.resample(values.astype(np.float64), target_length)


def bandpass(resampled: np.ndarray) -> np.ndarray:
    sos = signal.butter(
        FILTER_ORDER,
        [LOW_CUT_HZ, HIGH_CUT_HZ],
        btype="bandpass",
        fs=TARGET_RATE_HZ,
        output="sos",
    )
    return signal.sosfiltfilt(sos, resampled)


# Copied verbatim from notebooks/10_openox_canonical_window_export.ipynb (cell 7).
def compute_window_features(values: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    minimum_value = float(np.min(values))
    maximum_value = float(np.max(values))
    value_range = maximum_value - minimum_value
    rms_value = float(np.sqrt(np.mean(np.square(values))))

    skewness_value = float(stats.skew(values, bias=False))
    if not np.isfinite(skewness_value):
        skewness_value = 0.0

    kurtosis_value = float(stats.kurtosis(values, fisher=True, bias=False))
    if not np.isfinite(kurtosis_value):
        kurtosis_value = 0.0

    energy_value = float(np.sum(np.square(values)))
    derivative_std_value = float(np.std(np.diff(values))) if len(values) > 1 else 0.0

    frequencies, power = signal.welch(values, fs=sample_rate_hz, nperseg=min(256, len(values)))
    if len(frequencies) <= 1 or len(power) <= 1:
        dominant_frequency_hz = 0.0
        bandpower_0_5_2_hz = 0.0
        bandpower_2_5_hz = 0.0
    else:
        positive_frequencies = frequencies[1:]
        positive_power = power[1:]
        dominant_frequency_hz = float(positive_frequencies[int(np.argmax(positive_power))]) if len(positive_power) else 0.0

        mask_0_5_2 = (frequencies >= 0.5) & (frequencies <= 2.0)
        mask_2_5 = (frequencies >= 2.0) & (frequencies <= 5.0)
        bandpower_0_5_2_hz = float(np.trapezoid(power[mask_0_5_2], frequencies[mask_0_5_2])) if np.any(mask_0_5_2) else 0.0
        bandpower_2_5_hz = float(np.trapezoid(power[mask_2_5], frequencies[mask_2_5])) if np.any(mask_2_5) else 0.0

    bandpower_ratio = bandpower_0_5_2_hz / bandpower_2_5_hz if bandpower_2_5_hz > 0.0 else 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "minimum": minimum_value,
        "maximum": maximum_value,
        "value_range": value_range,
        "rms": rms_value,
        "skewness": skewness_value,
        "kurtosis": kurtosis_value,
        "energy": energy_value,
        "derivative_std": derivative_std_value,
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "bandpower_0_5_2_hz": float(bandpower_0_5_2_hz),
        "bandpower_2_5_hz": float(bandpower_2_5_hz),
        "bandpower_ratio_0_5_2_to_2_5": float(bandpower_ratio),
    }


def two_wavelength_features(red_raw: np.ndarray, red_filtered: np.ndarray,
                            ir_raw: np.ndarray, ir_filtered: np.ndarray) -> dict[str, float]:
    """Inputs are one window of the 50 Hz resampled raw and bandpassed red/IR signals."""
    ir_mean = float(np.mean(ir_raw))
    red_mean = float(np.mean(red_raw))
    if ir_mean <= 0.0 or red_mean <= 0.0:
        return {"ratio_of_ratios": float("nan"), "ir_perfusion_index": float("nan")}
    ir_perfusion = float(np.std(ir_filtered)) / ir_mean
    red_perfusion = float(np.std(red_filtered)) / red_mean
    ratio = red_perfusion / ir_perfusion if ir_perfusion > 0.0 else float("nan")
    return {"ratio_of_ratios": ratio, "ir_perfusion_index": ir_perfusion}


def iter_window_starts(sample_count: int):
    window_size_samples = int(round(WINDOW_SECONDS * TARGET_RATE_HZ))
    stride_size_samples = int(round(STRIDE_SECONDS * TARGET_RATE_HZ))
    for start_index in range(0, sample_count - window_size_samples + 1, stride_size_samples):
        yield start_index, start_index + window_size_samples


def read_header(hea_path: Path) -> tuple[int, float, list[str]]:
    with hea_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    first_tokens = lines[0].split()
    channel_count = int(first_tokens[1])
    source_rate_hz = float(first_tokens[2])
    channel_labels = [" ".join(line.split()[8:]) if len(line.split()) >= 9 else "" for line in lines[1 : 1 + channel_count]]
    return channel_count, source_rate_hz, channel_labels


def main() -> int:
    if FEATURE_CSV.exists():
        raise FileExistsError(f"Write-once output exists: {FEATURE_CSV}")

    passlist_df = pd.read_csv(PASSLIST_CSV)
    eligible_df = pd.read_csv(ELIGIBLE_CSV)
    work_df = eligible_df.merge(passlist_df[["encounter_id", "reference_column"]], on="encounter_id", how="inner")
    patient_map = pd.read_csv(ENCOUNTER_CSV, usecols=["patient_id", "encounter_id"]).set_index("encounter_id")["patient_id"]
    delta_map = dict(zip(eligible_df["encounter_id"], eligible_df["ppg_start_time_seconds"] - eligible_df["hz2_first_time_seconds"]))

    borderline_lower = LOW_SPO2_THRESHOLD - BORDERLINE_MARGIN
    borderline_upper = LOW_SPO2_THRESHOLD + BORDERLINE_MARGIN
    feature_rows = []
    audit_rows = []

    for row in work_df.itertuples(index=False):
        recording_id = str(row.encounter_id)
        hea_path = Path(str(row.ppg_hea_path))
        dat_path = hea_path.with_suffix(".dat")
        channel_count, source_rate_hz, channel_labels = read_header(hea_path)

        ir_index = choose_ppg_channel(channel_labels)
        red_index = find_channel(channel_labels, ("red signal", "red"))
        if red_index is None:
            raise ValueError(f"{recording_id}: no red channel in {channel_labels}")
        drive_indices = [idx for idx in (find_channel(channel_labels, ("red drive",)), find_channel(channel_labels, ("ir drive",))) if idx is not None]

        frames = read_ppg_frames(dat_path, channel_count)
        ir_resampled = resample_to_target(frames[:, ir_index], source_rate_hz)
        red_resampled = resample_to_target(frames[:, red_index], source_rate_hz)
        ir_filtered = bandpass(ir_resampled)
        red_filtered = bandpass(red_resampled)
        drives = frames[:, drive_indices].astype(np.int64) if drive_indices else None

        ref_df = pd.read_csv(Path(str(row.hz2_csv_path)))
        ref_df["Timestamp"] = pd.to_datetime(ref_df["Timestamp"], errors="coerce")
        ref_df[row.reference_column] = pd.to_numeric(ref_df[row.reference_column], errors="coerce")
        ref_df = ref_df.loc[ref_df["Timestamp"].notna()].copy()
        # Header alignment: elapsed time from the first 2 Hz ROW, the same origin as signed_delta.
        t0 = ref_df["Timestamp"].iloc[0]
        ref_df["elapsed_seconds"] = (ref_df["Timestamp"] - t0).dt.total_seconds()
        ref_df = ref_df.loc[ref_df[row.reference_column].notna()].copy()

        counts = {"proposed": 0, "no_reference": 0, "too_few_reference": 0, "borderline": 0,
                  "drive_change": 0, "nonfinite_two_wavelength": 0, "kept": 0}
        signed_delta = delta_map.get(row.encounter_id, 0.0)
        for start_index, end_index in iter_window_starts(len(ir_filtered)):
            counts["proposed"] += 1
            start_time_seconds = start_index / TARGET_RATE_HZ
            end_time_seconds = end_index / TARGET_RATE_HZ
            start_time_adjusted = start_time_seconds + signed_delta
            end_time_adjusted = end_time_seconds + signed_delta

            window_ref = ref_df.loc[
                (ref_df["elapsed_seconds"] >= start_time_adjusted)
                & (ref_df["elapsed_seconds"] < end_time_adjusted),
                row.reference_column,
            ]
            if window_ref.empty:
                counts["no_reference"] += 1
                continue
            if len(window_ref) < MIN_REFERENCE_SAMPLES_PER_WINDOW:
                counts["too_few_reference"] += 1
                continue

            median_spo2 = float(window_ref.median())
            min_spo2 = float(window_ref.min())
            if borderline_lower <= median_spo2 <= borderline_upper:
                counts["borderline"] += 1
                continue

            if drives is not None:
                native_start = int(round(start_time_seconds * source_rate_hz))
                native_end = min(int(round(end_time_seconds * source_rate_hz)), len(drives))
                segment = drives[native_start:native_end]
                if np.any(segment != segment[0]):
                    counts["drive_change"] += 1
                    continue

            two_wavelength = two_wavelength_features(
                red_resampled[start_index:end_index], red_filtered[start_index:end_index],
                ir_resampled[start_index:end_index], ir_filtered[start_index:end_index],
            )
            if not all(np.isfinite(value) for value in two_wavelength.values()):
                counts["nonfinite_two_wavelength"] += 1
                continue

            counts["kept"] += 1
            feature_rows.append(
                {
                    "recording_id": recording_id,
                    "patient_id": patient_map.get(row.encounter_id, ""),
                    "start_time_seconds": float(start_time_seconds),
                    "end_time_seconds": float(end_time_seconds),
                    "sample_rate_hz": float(TARGET_RATE_HZ),
                    "median_spo2": median_spo2,
                    "min_spo2": min_spo2,
                    "label": int(median_spo2 < LOW_SPO2_THRESHOLD),
                    "label_min_spo2": int(min_spo2 < LOW_SPO2_THRESHOLD),
                    **compute_window_features(ir_filtered[start_index:end_index], TARGET_RATE_HZ),
                    **two_wavelength,
                }
            )
        audit_rows.append({"recording_id": recording_id, "drive_channels": len(drive_indices), **counts})
        print(f"{recording_id[:12]} kept={counts['kept']:4d} drive_drop={counts['drive_change']:3d}", flush=True)

    temp_df = pd.DataFrame(feature_rows)
    windows_by_encounter = temp_df.groupby("recording_id").size()
    passing = set(windows_by_encounter[windows_by_encounter >= MIN_WINDOWS_PER_ENCOUNTER].index)
    feature_df = temp_df[temp_df["recording_id"].isin(passing)].sort_values(["recording_id", "start_time_seconds"]).reset_index(drop=True)

    audit_df = pd.DataFrame(audit_rows)
    audit_df["passed_encounter_gate"] = audit_df["recording_id"].isin(passing)
    audit_df["final_kept_windows"] = np.where(audit_df["passed_encounter_gate"], audit_df["kept"], 0)

    recording_summary_df = (
        feature_df.groupby("recording_id")
        .agg(patient_id=("patient_id", "first"), window_count=("label", "size"), positive_window_count=("label", "sum"),
             min_median_spo2=("median_spo2", "min"), max_median_spo2=("median_spo2", "max"))
        .reset_index()
    )
    recording_summary_df["has_positive_window"] = recording_summary_df["positive_window_count"] > 0

    # Comparison with the frozen notebook 10 export on shared (recording, start) windows.
    frozen = pd.read_csv(FROZEN_CSV)
    frozen["recording_id"] = frozen["recording_id"].astype(str)
    shared = feature_df.merge(frozen, on=["recording_id", "start_time_seconds"], suffixes=("", "_frozen"))
    feature_rel_diff = {
        name: float(np.max(np.abs(shared[name] - shared[f"{name}_frozen"]) / np.maximum(np.abs(shared[f"{name}_frozen"]), 1e-12)))
        for name in IR_FEATURE_NAMES
    }
    agreement = float((shared["label"] == shared["label_frozen"]).mean())
    crosstab = pd.crosstab(shared["label_frozen"], shared["label"])
    drive_recordings = audit_df.loc[audit_df["drive_channels"] > 0]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": "header: reference elapsed seconds from the first 2 Hz row; window shifted by signed_delta = ppg_start - hz2_first_row",
        "rows": int(len(feature_df)),
        "recordings": int(feature_df["recording_id"].nunique()),
        "patients": int(feature_df["patient_id"].nunique()),
        "positive_rows": int(feature_df["label"].sum()),
        "prevalence": float(feature_df["label"].mean()),
        "recordings_with_positive_window": int(recording_summary_df["has_positive_window"].sum()),
        "recordings_dropped_by_encounter_gate": sorted(set(temp_df["recording_id"]) - passing),
        "work_encounters": int(len(work_df)),
        "drop_counts_all_encounters": {key: int(audit_df[key].sum()) for key in
                                       ("proposed", "no_reference", "too_few_reference", "borderline", "drive_change", "nonfinite_two_wavelength", "kept")},
        "drive_change_windows_dropped": int(audit_df["drive_change"].sum()),
        "drive_change_note": "counted only among windows that passed the reference and borderline checks; recordings without drive channels are not checked",
        "recordings_with_drive_channels": int(len(drive_recordings)),
        "recordings_without_drive_channels": int((audit_df["drive_channels"] == 0).sum()),
        "frozen_comparison": {
            "frozen_rows": int(len(frozen)),
            "frozen_recordings": int(frozen["recording_id"].nunique()),
            "frozen_prevalence": float(frozen["label"].mean()),
            "shared_windows": int(len(shared)),
            "shared_recordings": int(shared["recording_id"].nunique()),
            "label_agreement_on_shared": agreement,
            "label_crosstab_frozen_rows_by_v2_cols": {f"frozen_{i}": {f"v2_{j}": int(crosstab.loc[i, j]) for j in crosstab.columns} for i in crosstab.index},
            "max_relative_ir_feature_difference_on_shared": feature_rel_diff,
        },
        "parameters": {
            "target_rate_hz": TARGET_RATE_HZ, "band_hz": [LOW_CUT_HZ, HIGH_CUT_HZ], "filter_order": FILTER_ORDER,
            "window_s": WINDOW_SECONDS, "stride_s": STRIDE_SECONDS, "min_reference_samples": MIN_REFERENCE_SAMPLES_PER_WINDOW,
            "min_windows_per_encounter": MIN_WINDOWS_PER_ENCOUNTER, "label": "median ScalcO2 < 90",
            "borderline_drop": [LOW_SPO2_THRESHOLD - BORDERLINE_MARGIN, LOW_SPO2_THRESHOLD + BORDERLINE_MARGIN],
        },
        "two_wavelength_features": {
            "ratio_of_ratios": "(std(filtered red)/mean(raw resampled red)) / (std(filtered IR)/mean(raw resampled IR))",
            "ir_perfusion_index": "std(filtered IR)/mean(raw resampled IR)",
            "red_processing": "identical to IR: scipy.signal.resample to 50 Hz, Butterworth 4 0.5-8 Hz sosfiltfilt over whole recording",
        },
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (PASSLIST_CSV, ELIGIBLE_CSV, FROZEN_CSV, ENCOUNTER_CSV)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_CSV, index=False)
    audit_df.to_csv(OUT_DIR / "export_audit.csv", index=False)
    recording_summary_df.to_csv(OUT_DIR / "recording_summary.csv", index=False)
    (OUT_DIR / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "parameters", "two_wavelength_features")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

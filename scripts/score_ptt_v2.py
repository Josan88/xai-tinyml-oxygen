"""Offline healthy-transfer check of the v2 TinyKAN (16 inputs) on PTT-PPG subject 1 sit/walk/run.

Uses the v2 OpenOximetry window/filter/feature code (scripts/export_openox_windows_v2.py):
scipy.signal.resample to 50 Hz, Butterworth 4 0.5-8 Hz sosfiltfilt over the whole record,
10 s windows with 5 s stride, 14 IR features + ratio_of_ratios + ir_perfusion_index. Scores
with a NumPy forward pass of the parsed kan_model_weights_v2.h. No training, no labels:
activity start/end SpO2 is not a window label.

Wavelength caveat: the dataset README maps pleth_1/pleth_2 as IR/red in Methods but red/IR
in Data Description. Primary mapping follows Data Description (pleth_2 = IR, the channel v1
used); the swapped mapping is reported as a sensitivity row. A control row scores the shipped
v1 header (14 IR features) through the same v2 pipeline.

Writes (write-once) outputs/openox-v2/ptt-subject-1/{windows.csv, activity_summary.csv, report.json}.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    import wfdb
except ImportError:  # vendored copy used by scripts/evaluate_ptt_ppg.py runs
    sys.path.insert(0, str(ROOT / "outputs" / "ptt-ppg" / "python-deps"))
    import wfdb

from export_openox_windows_v2 import (
    IR_FEATURE_NAMES, TARGET_RATE_HZ, bandpass, compute_window_features, iter_window_starts,
    resample_to_target, sha256, two_wavelength_features,
)
from train_tinykan_v2 import DECISION_THRESHOLD, HEADER_V2, SHIPPED_HEADER, V2_FEATURES, header_probabilities, parse_header

DATA_ROOT = ROOT / "data/physionet.org/files/pulse-transit-time-ppg/1.1.0"
V1_REPORT = ROOT / "outputs/ptt-ppg/subject-1-v1/report.json"
OUT_DIR = ROOT / "outputs/openox-v2/ptt-subject-1"
ACTIVITIES = ("sit", "walk", "run")
MAPPINGS = {
    "data_description": {"red": "pleth_1", "ir": "pleth_2"},
    "methods_swapped": {"red": "pleth_2", "ir": "pleth_1"},
}


def read_checksums() -> dict[str, str]:
    expected = {}
    for line in (DATA_ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename.lstrip("*")] = digest
    return expected


def record_windows(record, mapping: dict[str, str]) -> list[dict]:
    signals = {}
    for role, channel in mapping.items():
        values = np.asarray(record.p_signal[:, record.sig_name.index(channel)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{record.record_name}: nonfinite samples in {channel}")
        resampled = resample_to_target(values, float(record.fs))
        signals[role] = (resampled, bandpass(resampled))
    red_raw, red_filtered = signals["red"]
    ir_raw, ir_filtered = signals["ir"]
    rows = []
    for start, end in iter_window_starts(len(ir_filtered)):
        rows.append({
            "start_time_seconds": start / TARGET_RATE_HZ, "end_time_seconds": end / TARGET_RATE_HZ,
            **compute_window_features(ir_filtered[start:end], TARGET_RATE_HZ),
            **two_wavelength_features(red_raw[start:end], red_filtered[start:end], ir_raw[start:end], ir_filtered[start:end]),
        })
    return rows


def main() -> int:
    if (OUT_DIR / "report.json").exists():
        raise FileExistsError(f"Write-once output exists: {OUT_DIR / 'report.json'}")
    checksums = read_checksums()
    v2_weights = parse_header(HEADER_V2)
    v1_weights = parse_header(SHIPPED_HEADER)
    if v2_weights["SCALER_MEAN"].shape != (len(V2_FEATURES),) or v1_weights["SCALER_MEAN"].shape != (len(IR_FEATURE_NAMES),):
        raise RuntimeError("Unexpected header input dimensions")

    rows = []
    provenance = {}
    for activity in ACTIVITIES:
        name = f"s1_{activity}"
        for suffix in (".hea", ".dat"):
            digest = sha256(DATA_ROOT / (name + suffix))
            if checksums.get(name + suffix) != digest:
                raise ValueError(f"Published SHA256 mismatch: {name + suffix}")
            provenance[name + suffix] = digest
        record = wfdb.rdrecord(str(DATA_ROOT / name), physical=True)
        for mapping_name, mapping in MAPPINGS.items():
            windows = record_windows(record, mapping)
            frame = pd.DataFrame(windows)
            x16 = frame.loc[:, V2_FEATURES].to_numpy(dtype=np.float32)
            x14 = frame.loc[:, IR_FEATURE_NAMES].to_numpy(dtype=np.float32)
            frame["p_v2"] = header_probabilities(v2_weights, x16)
            frame["p_v1_header_v2_pipeline"] = header_probabilities(v1_weights, x14)
            frame.insert(0, "mapping", mapping_name)
            frame.insert(1, "red_channel", mapping["red"])
            frame.insert(2, "ir_channel", mapping["ir"])
            frame.insert(3, "activity", activity)
            frame.insert(4, "recording_id", name)
            rows.append(frame)
        print(f"Scored {name}: {len(windows)} windows", flush=True)
    windows = pd.concat(rows, ignore_index=True)
    if not np.isfinite(windows[list(V2_FEATURES) + ["p_v2", "p_v1_header_v2_pipeline"]].to_numpy()).all():
        raise RuntimeError("Nonfinite features or probabilities; no imputation")

    v1 = {item["activity"]: item for item in json.loads(V1_REPORT.read_text(encoding="utf-8"))["summary"]}
    summary = []
    for (mapping_name, activity), group in windows.groupby(["mapping", "activity"], sort=False):
        summary.append({
            "mapping": mapping_name, "ir_channel": group["ir_channel"].iloc[0], "activity": activity, "windows": len(group),
            "v2_mean_p": float(group["p_v2"].mean()), "v2_median_p": float(group["p_v2"].median()),
            "v2_candidate_rate": float((group["p_v2"] >= DECISION_THRESHOLD).mean()),
            "v1_header_v2_pipeline_mean_p": float(group["p_v1_header_v2_pipeline"].mean()),
            "v1_header_v2_pipeline_candidate_rate": float((group["p_v1_header_v2_pipeline"] >= DECISION_THRESHOLD).mean()),
            "v1_published_mean_p": float(v1[activity]["mean_p"]), "v1_published_candidate_rate": float(v1[activity]["candidate_fraction"]),
            "median_ratio_of_ratios": float(group["ratio_of_ratios"].median()),
            "median_ir_perfusion_index": float(group["ir_perfusion_index"].median()),
        })
    summary_df = pd.DataFrame(summary)

    train = pd.read_csv(ROOT / "outputs/openox-v2/window_features.csv", usecols=["label", "ratio_of_ratios", "ir_perfusion_index"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline feasibility check on one healthy subject; no labels, no accuracy metrics",
        "model": str(HEADER_V2.relative_to(ROOT)), "model_sha256": sha256(HEADER_V2),
        "control_model": str(SHIPPED_HEADER.relative_to(ROOT)), "control_model_sha256": sha256(SHIPPED_HEADER),
        "pipeline": "export_openox_windows_v2: scipy.signal.resample 500->50 Hz, Butterworth 4 0.5-8 Hz sosfiltfilt whole record, 10 s / 5 s windows",
        "physical_conversion": "WFDB physical=True",
        "wavelength_mapping_caveat": "README Methods: pleth_1 IR, pleth_2 red; README Data Description: pleth_1 red, pleth_2 IR. Primary = data_description (v1 used pleth_2 as model input).",
        "v1_reference": {"source": str(V1_REPORT.relative_to(ROOT)), "pipeline": "resample_poly + firmware-twin per-window features, pleth_2, shipped header"},
        "threshold": DECISION_THRESHOLD,
        "summary": summary,
        "openox_v2_training_reference": {
            "ratio_of_ratios_median_label0": float(train.loc[train["label"] == 0, "ratio_of_ratios"].median()),
            "ratio_of_ratios_median_label1": float(train.loc[train["label"] == 1, "ratio_of_ratios"].median()),
            "ir_perfusion_index_median_label0": float(train.loc[train["label"] == 0, "ir_perfusion_index"].median()),
            "ir_perfusion_index_median_label1": float(train.loc[train["label"] == 1, "ir_perfusion_index"].median()),
        },
        "source_sha256": provenance,
        "wfdb": wfdb.__version__,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_csv(OUT_DIR / "windows.csv", index=False)
    summary_df.to_csv(OUT_DIR / "activity_summary.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary_df.round(4).to_string(index=False))
    print(json.dumps(report["openox_v2_training_reference"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

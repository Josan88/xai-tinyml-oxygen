"""Reproducible, bounded PTT-PPG activity feasibility evaluation (no training).

Default: subject 1, all complete sit/walk/run windows, pleth_2, threshold 0.50.
WFDB is only needed for real records; unit tests can inject a record reader.
Downloaded data and generated outputs stay in gitignored local directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from urllib.request import Request, urlopen

import numpy as np
import scipy
from scipy.signal import resample_poly

from check_probability_parity import run_probe
from score_tinykan_explanation_rubric import FEATURE_NAMES, model_attributions, parse_c_header
from verify_features import cpp_extract_features

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/"
DATASET_URL = "https://physionet.org/content/pulse-transit-time-ppg/1.1.0/"
DATA_ROOT = ROOT / "data/physionet.org/files/pulse-transit-time-ppg/1.1.0"
HEADER = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h"
ACTIVITIES = ("sit", "walk", "run")
FS = 50
WINDOW = 500
HOP = 250
THRESHOLD = 0.50
USER_AGENT = "fyrp-xai-tinyml/1.0 (research; pulse-transit-time-ppg 1.1.0 subset)"
CONTRACT_FILES = (
    HEADER, ROOT / "firmware/Arduino Nano 33/include/feature_extraction.h",
    ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
    ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h",
    Path(__file__), ROOT / "scripts/verify_features.py",
    ROOT / "scripts/score_tinykan_explanation_rubric.py",
    ROOT / "scripts/check_probability_parity.py", ROOT / "tests/probability_probe.cpp",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_record_name(name):
    match = re.fullmatch(r"s([1-9]|1[0-9]|2[0-2])_(sit|walk|run)", name)
    if match is None:
        raise ValueError(f"Invalid PTT-PPG 1.1.0 record name: {name!r}")
    return f"s{match[1]}", match[2]


def read_checksums(data_root):
    manifest = Path(data_root) / "SHA256SUMS.txt"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA256SUMS entry")
        expected[filename.lstrip("*")] = digest
    return expected


def verify_source(path, checksums):
    path = Path(path)
    actual = sha256(path)
    if checksums.get(path.name) != actual:
        raise ValueError(f"Published SHA256 mismatch or absent entry: {path.name}")
    return actual


def fetch_bytes(name, limit, timeout=120):
    request = Request(BASE_URL + name, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Download exceeded bounded file size: {name}")
    return data


def fetch_subset(data_root, records):
    """Fetch only named records plus metadata; never download the full dataset."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for name in records:
        parse_record_name(name)
    names = ["SHA256SUMS.txt", "LICENSE.txt", "README.txt", "RECORDS"]
    names += [name + suffix for name in records for suffix in (".hea", ".dat")]
    checksums = None
    for name in names:
        destination = data_root / name
        if name == "SHA256SUMS.txt" and destination.exists():
            checksums = read_checksums(data_root)
        stale = False
        if destination.exists() and checksums is not None and name != "SHA256SUMS.txt":
            try:
                verify_source(destination, checksums)
            except ValueError:
                stale = True
        if not destination.exists() or stale:
            limit = 12 * 1024 * 1024 if name.endswith(".dat") else 128 * 1024
            data = fetch_bytes(name, limit)
            if checksums is not None and hashlib.sha256(data).hexdigest() != checksums.get(name):
                raise ValueError(f"Published SHA256 mismatch: {name}")
            temporary = destination.with_name(name + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            print(f"Fetched {name}: {len(data):,} bytes", flush=True)
        if name == "SHA256SUMS.txt":
            checksums = read_checksums(data_root)
        else:
            verify_source(destination, checksums)


def resample_ppg_50hz(samples, source_fs):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("PPG must be a nonempty, finite one-dimensional signal; no gap interpolation")
    if not np.isfinite(source_fs) or source_fs <= 0:
        raise ValueError("Source sample rate must be positive and finite")
    ratio = Fraction(str(FS / float(source_fs))).limit_denominator(100000)
    if not np.isclose(float(ratio) * source_fs, FS, rtol=0, atol=1e-8):
        raise ValueError("Source sample rate cannot be represented accurately for polyphase resampling")
    if ratio == 1:
        return samples.copy()
    # Anti-alias FIR; preserve local DC at boundaries instead of zero padding.
    return resample_poly(samples, ratio.numerator, ratio.denominator, padtype="line")


def iter_windows(samples):
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("Window input must be one-dimensional")
    for start in range(0, len(samples) - WINDOW + 1, HOP):
        yield start, samples[start:start + WINDOW]


def load_record(path, channel, reader=None):
    path = Path(path)
    subject, activity = parse_record_name(path.name)
    if reader is None:
        try:
            import wfdb
        except ImportError as exc:
            raise RuntimeError("Install wfdb in the Python environment to read PTT-PPG records") from exc
        reader = wfdb.rdrecord
    record = reader(str(path), physical=True)
    metadata = dict(re.findall(r"<([^>]+)>:\s*([^<]+)", " ".join(record.comments)))
    metadata = {key: value.strip() for key, value in metadata.items()}
    if metadata.get("activity") != activity or metadata.get("filename") != path.name:
        raise ValueError(f"Record filename and activity metadata disagree: {path.name}")
    if record.sig_name.count(channel) != 1:
        raise ValueError(f"Expected exactly one {channel} in {path.name}")
    if not np.isfinite(record.fs) or record.fs <= 0:
        raise ValueError(f"Invalid source sample rate in {path.name}")
    values = np.asarray(record.p_signal)
    if values.ndim != 2 or values.shape != (record.sig_len, len(record.sig_name)):
        raise ValueError(f"Invalid waveform dimensions in {path.name}")
    index = record.sig_name.index(channel)
    ppg = values[:, index].astype(np.float64)
    missing_ppg = int(np.count_nonzero(~np.isfinite(ppg)))
    if missing_ppg:
        raise ValueError(f"{path.name}: {missing_ppg} missing PPG samples; refusing to interpolate")
    axes = ("a_x", "a_y", "a_z")
    have_accel = all(record.sig_name.count(axis) == 1 for axis in axes)
    acceleration = values[:, [record.sig_name.index(axis) for axis in axes]] if have_accel else None
    audit = {
        "recording_id": path.name, "subject_id": subject, "activity": activity,
        "source_fs_hz": float(record.fs), "source_samples": int(record.sig_len),
        "source_duration_s": record.sig_len / record.fs, "source_channel_count": len(record.sig_name),
        "ppg_channel": channel, "ppg_units": record.units[index],
        "ppg_adc_gain": float(record.adc_gain[index]), "ppg_baseline": int(record.baseline[index]),
        "ppg_missing_samples": missing_ppg, "ppg_min": float(ppg.min()), "ppg_max": float(ppg.max()),
        "acceleration_channels": list(axes) if have_accel else [],
        "acceleration_units": [record.units[record.sig_name.index(axis)] for axis in axes] if have_accel else [],
        "acceleration_missing_samples": int(np.count_nonzero(~np.isfinite(acceleration))) if have_accel else None,
    }
    return ppg, acceleration, audit


def motion_context(acceleration, source_fs, start):
    if acceleration is None:
        return {"imu_rms_header_units": None, "imu_dynamic_rms_header_units": None}
    left = round(start / FS * source_fs)
    right = round((start + WINDOW) / FS * source_fs)
    segment = np.asarray(acceleration[left:right], dtype=np.float64)
    if len(segment) != right - left or not np.isfinite(segment).all():
        return {"imu_rms_header_units": None, "imu_dynamic_rms_header_units": None}
    # Vector-magnitude RMS, not per-axis RMS. The first includes gravity/DC.
    centered = segment - segment.mean(axis=0)
    return {"imu_rms_header_units": float(np.sqrt(np.mean(np.sum(segment ** 2, axis=1)))),
            "imu_dynamic_rms_header_units": float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))}


def score_windows(windows, weights, probe=None):
    raw = np.asarray(windows, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != WINDOW or not np.isfinite(raw).all():
        raise ValueError("Expected finite full 500-sample windows")
    if probe is not None:
        native = run_probe(Path(probe).resolve(), "raw", raw)
        for result in native:
            features, deltas = result[1:15], result[29:43]
            order = sorted(range(14), key=lambda i: (-abs(float(deltas[i])), i))
            top3 = [(FEATURE_NAMES[i], float(deltas[i])) for i in order[:3]]
            # Reuse the independent existing FP32 helper on identical features.
            reference_p, _ = model_attributions(features, weights)
            yield float(result[0]), features, top3, reference_p
    else:
        for window in raw:
            features = cpp_extract_features(window)
            features = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite extracted features; no invalid-window imputation")
            p, top3 = model_attributions(features, weights)
            yield p, features, top3, p


def summarize(rows, subject_id=None):
    result = []
    for activity in ACTIVITIES:
        subset = [row for row in rows if row["activity"] == activity]
        if not subset:
            continue
        probabilities = np.array([row["p"] for row in subset])
        top1 = Counter(row["top1_feature"] for row in subset)
        result.append({
            **({"subject_id": subject_id} if subject_id else {}),
            "activity": activity, "subjects": len({row["subject_id"] for row in subset}),
            "recordings": len({row["recording_id"] for row in subset}), "windows": len(subset),
            "mean_p": float(probabilities.mean()), "median_p": float(np.median(probabilities)),
            "p05": float(np.quantile(probabilities, .05)), "p95": float(np.quantile(probabilities, .95)),
            "candidate_windows": sum(row["candidate"] for row in subset),
            "candidate_fraction": float(np.mean(probabilities >= THRESHOLD)),
            "top1_histogram": json.dumps(dict(sorted(top1.items())), sort_keys=True),
        })
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to emit a result CSV with no real windows")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data_root, records, channel, output, probe=None, reader=None):
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in CONTRACT_FILES}
    weights = parse_c_header(HEADER)
    checksums = read_checksums(data_root)
    rows, audits = [], []
    for name in records:
        path = Path(data_root) / name
        provenance = {suffix: verify_source(path.with_suffix(suffix), checksums) for suffix in (".hea", ".dat")}
        ppg, acceleration, audit = load_record(path, channel, reader=reader)
        resampled = resample_ppg_50hz(ppg, audit["source_fs_hz"])
        windows = list(iter_windows(resampled))
        if not windows:
            raise ValueError(f"No full 10-second windows: {name}")
        for (start, _), (p, features, top3, reference_p) in zip(
                windows, score_windows([window for _, window in windows], weights, probe), strict=True):
            row = {"subject_id": audit["subject_id"], "recording_id": name, "activity": audit["activity"],
                   "channel": channel, "start_time_s": start / FS, "end_time_s": (start + WINDOW) / FS,
                   "p": p, "candidate": int(p >= THRESHOLD), "same_features_python_p": reference_p,
                   "same_features_probability_error": abs(p - reference_p)}
            for rank, (feature, delta) in enumerate(top3, 1):
                row[f"top{rank}_feature"] = feature
                row[f"top{rank}_delta"] = delta
            row.update(motion_context(acceleration, audit["source_fs_hz"], start))
            row.update({f"feature_{name}": float(value) for name, value in zip(FEATURE_NAMES, features)})
            rows.append(row)
        audit.update({"resampled_samples": len(resampled), "windows": len(windows),
                      "tail_samples_after_last_window": len(resampled) - (windows[-1][0] + WINDOW),
                      "source_url": BASE_URL + name + ".hea", "source_sha256": provenance})
        audits.append(audit)
        print(f"Evaluated {name}: {len(windows)} complete windows", flush=True)
    if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("Model/evaluation source changed during the run; no result files written")
    summary = summarize(rows)
    per_subject = [item for subject in sorted({row["subject_id"] for row in rows})
                   for item in summarize([row for row in rows if row["subject_id"] == subject], subject)]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_url": DATASET_URL,
        "scope": "bounded activity feasibility spike; selected subjects only, not a validation cohort",
        "subjects": sorted({row["subject_id"] for row in rows}), "records": records, "channel": channel,
        "channel_wavelength_caveat": "pleth_1/2 and pleth_4/5 wavelength mapping conflicts between dataset Methods and Data Description",
        "window_samples": WINDOW, "hop_samples": HOP, "target_fs_hz": FS, "threshold": THRESHOLD,
        "backend": "compiled firmware probability_probe raw mode; independent per-window filtering" if probe else "verify_features NumPy feature twin and FP32 NumPy TinyKAN; not board execution",
        "resampling": "scipy.signal.resample_poly of full physical-unit record to 50 Hz, padtype=line; float32 before feature extraction",
        "physical_conversion": "WFDB physical=True: (digital - header baseline) / header ADC gain; do not use stored 12-bit counts as sensor units",
        "ranking": "descending absolute mean-perturbation probability delta; ties by feature index; signed deltas retained",
        "consecutive_state": "not evaluated; each window is scored independently; candidate only",
        "accuracy_metrics": "not computed; activity start/end SpO2 is not a window label",
        "motion_context": "source-sample vector RMS; dynamic RMS subtracts each axis mean inside each window; header units only, not instrument-calibrated; never a model input",
        "summary": summary, "recording_audit": audits,
        "max_same_features_probability_error": max(row["same_features_probability_error"] for row in rows),
        "same_features_candidate_disagreements": sum((row["p"] >= THRESHOLD) != (row["same_features_python_p"] >= THRESHOLD) for row in rows),
        "source_hashes": source_hashes, "published_checksum_manifest_sha256": sha256(Path(data_root) / "SHA256SUMS.txt"),
        "probe": None if probe is None else {"path": str(Path(probe).resolve()), "sha256": sha256(probe)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "wfdb": getattr(sys.modules.get("wfdb"), "__version__", None)},
    }
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError(f"Write-once output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "windows.csv", rows)
    write_csv(output / "activity_summary.csv", summary)
    write_csv(output / "subject_activity_summary.csv", per_subject)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1], help="Explicit IDs from 1 to 22; default 1 only")
    parser.add_argument("--channel", choices=[f"pleth_{i}" for i in range(1, 7)], default="pleth_2")
    parser.add_argument("--download", action="store_true", help="Fetch only selected subjects' waveform/header files and provenance metadata")
    parser.add_argument("--probe", type=Path, help="Existing native probability_probe executable to use actual C++ FP32 feature/inference code")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ptt-ppg/subject-1-v1")
    parser.add_argument("--success-csv", type=Path, default=ROOT / "outputs/ptt_ppg_motion_spike.csv")
    args = parser.parse_args()
    if any(subject < 1 or subject > 22 for subject in args.subjects) or len(set(args.subjects)) != len(args.subjects):
        parser.error("--subjects must be unique IDs in 1..22")
    if not HEADER.is_file():
        parser.error(f"Frozen weights missing: {HEADER}")
    if args.probe is not None and not args.probe.is_file():
        parser.error(f"Native probe missing: {args.probe}")
    records = [f"s{subject}_{activity}" for subject in sorted(args.subjects) for activity in ACTIVITIES]
    if args.download:
        fetch_subset(args.data_root, records)
    report = evaluate(args.data_root, records, args.channel, args.output, args.probe)
    if args.success_csv is not None:
        if args.success_csv.exists():
            raise FileExistsError(f"Write-once success CSV exists: {args.success_csv}")
        args.success_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.success_csv, report["summary"])
    print(json.dumps({"output": str(args.output.resolve()), "subjects": report["subjects"],
                      "activity_summary": report["summary"],
                      "success_csv": None if args.success_csv is None else str(args.success_csv.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

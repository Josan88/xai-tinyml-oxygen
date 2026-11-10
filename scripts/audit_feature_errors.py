"""Read-only FP32 stage audit and broader five-second-context replay.

All recorded inputs and per-window outputs stay under ignored outputs/. This
does not replace the original feature gate, retune the model, or flash a board.
The proposed diagnostic tolerances are fixed here before the expanded run.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import scipy
from scipy import signal

from check_probability_parity import (ROOT, HEADER, NOTEBOOK, NAMES,
                                      canonical_feature_function, reference, run_probe)
from check_training_context import run_context_probe
from verify_features import parse_c_header

PROPOSAL = {"status": "diagnostic_only_not_adopted", "standardized_absolute_tolerance": 1e-4,
            "fp32_spacing_allowance": 8, "probability_absolute_tolerance": 1e-4,
            "attribution_absolute_tolerance": 1e-5, "candidate_disagreements_allowed": 0,
            "dominant_frequency_bin_disagreements_allowed": 0,
            "feature_rule": "abs(actual-reference) <= 1e-4 * scaler_std + 8 * abs(spacing(float32(reference))); dominant frequency must match exactly",
            "rationale": "Feature budget is one ten-thousandth of the training scale plus eight output spacings. These are engineering screening budgets, not proven error bounds or clinical acceptance criteria."}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fp32(values):
    # Restore the exact float represented by the native nine-digit serialization.
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def run_audit_probe(executable, mode, rows):
    lines = []
    for row in rows:
        if mode == "raw":
            samples, offset = row
            prefix = f"{len(samples)} {offset} "
        else:
            samples, prefix = row, ""
        lines.append(prefix + " ".join(format(float(x), ".9g") for x in fp32(samples)))
    result = subprocess.run([str(executable), mode], input="\n".join(lines) + "\n",
                            text=True, capture_output=True, check=True, timeout=120)
    values = np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
    if values.shape != (len(rows), 514) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid audit probe output: {values.shape}")
    return fp32(values)


def proposal_feature_mask(expected, actual, scales):
    spacing = np.abs(np.spacing(np.asarray(expected, dtype=np.float32)).astype(np.float64))
    budget = PROPOSAL["standardized_absolute_tolerance"] * scales + PROPOSAL["fp32_spacing_allowance"] * spacing
    passed = np.abs(actual - expected) <= budget
    passed[..., 10] = actual[..., 10] == expected[..., 10]
    return passed


def feature_summary(expected, actual, scales):
    error = np.abs(actual - expected)
    nearest = fp32(expected)
    rounding = np.abs(nearest - expected)
    spacing = np.abs(np.spacing(np.asarray(expected, dtype=np.float32)).astype(np.float64))
    proposed = proposal_feature_mask(expected, actual, scales)
    return [{"feature": name, "comparisons": len(expected),
             "max_absolute_error": float(error[:, i].max()),
             "max_standardized_error": float((error[:, i] / scales[i]).max()),
             "p99_standardized_error": float(np.quantile(error[:, i] / scales[i], .99)),
             "strict_1e_4_failures": int(np.sum(error[:, i] > 1e-4)),
             "nearest_fp32_already_exceeds_1e_4": int(np.sum(rounding[:, i] > 1e-4)),
             "max_nearest_fp32_rounding_error": float(rounding[:, i].max()),
             "max_fp32_spacing": float(spacing[:, i].max()),
             "proposal_failures": int(np.sum(~proposed[:, i]))} for i, name in enumerate(NAMES)]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-probe", required=True, type=Path)
    parser.add_argument("--context-probe", required=True, type=Path)
    parser.add_argument("--probability-probe", required=True, type=Path)
    parser.add_argument("--recordings", type=int, default=0, help="0 means every recording; positive limit for smoke checks")
    parser.add_argument("--windows-per-recording", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--board-report", type=Path)
    args = parser.parse_args()
    if args.recordings < 0 or args.windows_per_recording < 1:
        parser.error("recordings must be nonnegative and windows-per-recording positive")
    output = args.output or ROOT / "outputs/parity/feature_audit" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Keep this local recorded-data audit under the ignored outputs/ directory")
    output.mkdir(parents=True, exist_ok=True)
    feature_csv = ROOT / "outputs/openox_canonical_window_features.csv"
    eligible_csv = ROOT / "outputs/openox_ppg_eligible_encounters.csv"
    with feature_csv.open(newline="") as stream:
        retained = list(csv.DictReader(stream))
    grouped = {}
    for row in retained:
        grouped.setdefault(row["recording_id"], []).append(row)
    ids = list(grouped)
    prior_ids = {ids[i] for i in np.linspace(0, len(ids) - 1, min(8, len(ids)), dtype=int)}
    selected = ids[:args.recordings] if args.recordings else ids
    with eligible_csv.open(newline="") as stream:
        eligible = {r["encounter_id"]: r for r in csv.DictReader(stream)}
    parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
    scales = parameters["scaler_std"]
    if np.any(scales <= 0):
        raise ValueError("Invalid scaler standard deviations")
    compute = canonical_feature_function()
    def features(samples):
        row = compute(np.asarray(samples, dtype=np.float64), 50.)
        return np.array([row[name] for name in NAMES])
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    stages = {name: [[], []] for name in ("input_quantization", "filter_calculation", "feature_calculation",
                                         "end_to_end_finite_context", "finite_vs_full_context")}
    windows, provenance, details, skipped = [], [], [], []
    artifact_paths = [Path(__file__), HEADER, NOTEBOOK, Path(__file__).with_name("check_probability_parity.py"),
                      Path(__file__).with_name("check_training_context.py"), Path(__file__).with_name("verify_features.py"),
                      ROOT / "tests/feature_audit_probe.cpp", ROOT / "tests/context_probe.cpp",
                      ROOT / "tests/probability_probe.cpp", ROOT / "tests/CMakeLists.txt",
                      ROOT / "firmware/Arduino Nano 33/include/feature_extraction.h",
                      ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h",
                      ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
                      feature_csv, eligible_csv, args.audit_probe, args.context_probe, args.probability_probe]
    hashes = {str(path.resolve()): sha256(path) for path in artifact_paths}
    (output / "proposal.json").write_text(json.dumps(PROPOSAL, indent=2) + "\n")
    for record_number, recording_id in enumerate(selected):
        header = Path(eligible[recording_id]["ppg_hea_path"])
        lines = [line.strip() for line in header.read_text().splitlines() if line.strip()]
        tokens = lines[0].split()
        channels, rate = int(tokens[1]), float(tokens[2])
        labels = [" ".join(line.split()[8:]).lower() for line in lines[1:1 + channels]]
        channel = next(labels.index(name) for name in ("ir signal", "ir", "red signal", "red") if name in labels)
        path = header.with_suffix(".dat")
        frames = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, channels)
        resampled = signal.resample(frames[:, channel].astype(np.float64), round(len(frames) * 50 / rate))
        full_filtered = signal.sosfiltfilt(sos, resampled)
        recording_rows = grouped[recording_id]
        chosen = [recording_rows[i] for i in np.linspace(0, len(recording_rows) - 1,
                                                        min(args.windows_per_recording, len(recording_rows)), dtype=int)]
        pending, raw_contexts, full_rows, finite_rows, quantized_rows, stored_rows = [], [], [], [], [], []
        for row in chosen:
            start = round(float(row["start_time_seconds"]) * 50)
            if start + 750 > len(resampled):
                skipped.append({"recording_id": recording_id, "start_sample": start, "reason": "missing_250_future_samples"})
                continue
            if start % 250:
                raise ValueError("Retained window is off the deployment stride")
            left = max(0, start - 250)
            raw = resampled[left:start + 750]
            offset = start - left
            full_rows.append(features(full_filtered[start:start + 500]))
            stored_rows.append([float(row[name]) for name in NAMES])
            finite_rows.append(features(signal.sosfiltfilt(sos, raw)[offset:offset + 500]))
            quantized_rows.append(features(signal.sosfiltfilt(sos, fp32(raw))[offset:offset + 500]))
            raw_contexts.append((raw, offset))
            pending.append({"recording_id": recording_id, "recording_number": record_number + 1,
                            "cohort": "previously_sampled_recording" if recording_id in prior_ids else "new_recording",
                            "start_sample": start})
        if not pending:
            raise RuntimeError("No context-eligible windows in selected recording")
        audited = run_audit_probe(args.audit_probe.resolve(), "raw", raw_contexts)
        native_filtered, native_features = audited[:, :500], audited[:, 500:]
        on_native_filter = np.array([features(row) for row in native_filtered])
        full_rows, finite_rows, quantized_rows, stored_rows = map(np.asarray, (full_rows, finite_rows, quantized_rows, stored_rows))
        if not all(np.isfinite(x).all() for x in (full_rows, finite_rows, quantized_rows, stored_rows, on_native_filter)):
            raise RuntimeError("Nonfinite feature data; audit stopped")
        full_p, full_xai = reference(full_rows, parameters)
        finite_p, finite_xai = reference(quantized_rows, parameters)
        context64_p, _ = reference(finite_rows, parameters)
        native_inference = fp32(run_probe(args.probability_probe.resolve(), "features", native_features))
        streamed = run_context_probe(args.context_probe.resolve(), resampled[:max(r["start_sample"] for r in pending) + 750])
        by_start = {int(row[2]): fp32(row) for row in streamed}
        proposed_features = proposal_feature_mask(quantized_rows, native_features, scales)
        for i, row in enumerate(pending):
            actual = by_start.get(row["start_sample"])
            if actual is None or not np.array_equal(actual[6:20], native_features[i]) or actual[5] != native_inference[i, 0]:
                raise RuntimeError("Audit stages disagree with the actual streaming pipeline")
            if actual[1] != row["start_sample"] + 750 or actual[3] != row["start_sample"] + 500 or actual[4] != 250:
                raise RuntimeError("Deployment timing mismatch")
            native_p = native_inference[i, 0]
            native_xai = native_inference[i, 29:]
            p_error = abs(native_p - finite_p[i])
            xai_error = float(np.max(np.abs(native_xai - finite_xai[i])))
            candidate_agrees = bool((native_p >= .5) == (finite_p[i] >= .5))
            top3_native = np.argsort(-np.abs(native_xai), kind="stable")[:3]
            top3_finite = np.argsort(-np.abs(finite_xai[i]), kind="stable")[:3]
            row.update(full_probability=float(full_p[i]), finite_context_probability=float(finite_p[i]),
                       cpp_probability=float(native_p), numerical_probability_error=float(p_error),
                       numerical_xai_error=xai_error, numerical_candidate_agrees=candidate_agrees,
                       numerical_top3_indices_agree=bool(np.array_equal(top3_native, top3_finite)),
                       full_context_probability_error=float(abs(native_p - full_p[i])),
                       pure_context_probability_error=float(abs(context64_p[i] - full_p[i])),
                       full_context_xai_error=float(np.max(np.abs(native_xai - full_xai[i]))),
                       full_context_candidate_agrees=bool((native_p >= .5) == (full_p[i] >= .5)),
                       stored_features_reconstructed=bool(np.allclose(full_rows[i], stored_rows[i], atol=1e-6, rtol=1e-7)),
                       strict_raw_feature_pass=bool(np.all(np.abs(native_features[i] - quantized_rows[i]) <= 1e-4)),
                       proposal_feature_pass=bool(proposed_features[i].all()),
                       proposal_numerical_pass=bool(proposed_features[i].all() and p_error <= 1e-4 and xai_error <= 1e-5 and candidate_agrees))
            for j, name in enumerate(NAMES):
                details.append({"recording_id": recording_id, "start_sample": row["start_sample"], "feature": name,
                                "full_recording": float(full_rows[i, j]), "finite_context": float(finite_rows[i, j]),
                                "finite_context_fp32_input": float(quantized_rows[i, j]),
                                "python_on_cpp_filtered_samples": float(on_native_filter[i, j]),
                                "cpp": float(native_features[i, j]), "scaler_std": float(scales[j])})
        stage_pairs = {"input_quantization": (finite_rows, quantized_rows),
                       "filter_calculation": (quantized_rows, on_native_filter),
                       "feature_calculation": (on_native_filter, native_features),
                       "end_to_end_finite_context": (quantized_rows, native_features),
                       "finite_vs_full_context": (full_rows, finite_rows)}
        for name, pair in stage_pairs.items():
            for index in range(2):
                stages[name][index].extend(pair[index])
        provenance.append({"recording_id": recording_id, "header_sha256": sha256(header), "data_sha256": sha256(path),
                           "path": str(path.relative_to(ROOT)), "source_rate": rate, "channel": channel,
                           "source_samples": len(frames), "resampled_samples": len(resampled),
                           "replay_input_sha256": hashlib.sha256(np.asarray(resampled[:int(streamed[-1][1])], dtype="<f4").tobytes()).hexdigest()})
        windows.extend(pending)
        write_csv(output / "windows.csv", windows)
        print(f"Audited recording {record_number + 1}/{len(selected)}; {len(windows)} retained windows", flush=True)
    summaries = {name: feature_summary(np.array(pair[0]), np.array(pair[1]), scales) for name, pair in stages.items()}
    def aggregate(rows):
        return {"windows": len(rows), "recordings": len({r["recording_id"] for r in rows}),
                "max_numerical_probability_error": max(r["numerical_probability_error"] for r in rows),
                "max_numerical_xai_error": max(r["numerical_xai_error"] for r in rows),
                "numerical_candidate_disagreements": sum(not r["numerical_candidate_agrees"] for r in rows),
                "numerical_top3_index_disagreements": sum(not r["numerical_top3_indices_agree"] for r in rows),
                "max_full_context_probability_error": max(r["full_context_probability_error"] for r in rows),
                "max_pure_context_probability_error": max(r["pure_context_probability_error"] for r in rows),
                "full_context_errors_ge_0_01": sum(r["full_context_probability_error"] >= .01 for r in rows),
                "full_context_candidate_disagreements": sum(not r["full_context_candidate_agrees"] for r in rows),
                "strict_raw_feature_failed_windows": sum(not r["strict_raw_feature_pass"] for r in rows),
                "proposal_feature_failed_windows": sum(not r["proposal_feature_pass"] for r in rows),
                "proposal_numerical_failed_windows": sum(not r["proposal_numerical_pass"] for r in rows),
                "stored_feature_reconstruction_failures": sum(not r["stored_features_reconstructed"] for r in rows)}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "completed_diagnostic",
              "threshold": .5, "debounce_windows": 2, "future_context_samples": 250,
              "dataset_recordings": len(ids), "dataset_retained_windows": len(retained),
              "selection": f"All {len(selected)} selected recording IDs in CSV order, up to {args.windows_per_recording} evenly spaced retained windows per recording; only complete 250-sample future context",
              "scope": "Recorded-data host audit. Full-recording resampling remains an offline operation. No board run, sensor validation, clinical performance estimate or tolerance adoption.",
              "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
              "summary": aggregate(windows),
              "cohorts": {cohort: aggregate([r for r in windows if r["cohort"] == cohort]) for cohort in sorted({r["cohort"] for r in windows})},
              "proposal": PROPOSAL, "feature_stages": summaries, "skipped_windows": skipped,
              "provenance": provenance, "sha256": hashes,
              "worst_context_windows": sorted(windows, key=lambda r: r["full_context_probability_error"], reverse=True)[:10],
              "context_candidate_disagreements": [r for r in windows if not r["full_context_candidate_agrees"]]}
    if args.board_report:
        board = json.loads(args.board_report.read_text())
        expected, actual = [], []
        for case, rows in board["board_results"].items():
            for ref, result in zip(board["reference"][case], rows, strict=True):
                expected.append(ref["values"][1:])
                actual.append(result["actual"]["values"][1:])
        report["saved_board_feature_audit"] = {"source": str(args.board_report.resolve()), "sha256": sha256(args.board_report),
                                                "windows": len(expected), "features": feature_summary(np.array(expected), np.array(actual), scales)}
    write_csv(output / "feature_values.csv", details)
    write_csv(output / "feature_summary.csv", [{"stage": name, **row} for name, rows in summaries.items() for row in rows])
    write_csv(output / "skipped_windows.csv", skipped)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "summary": report["summary"], "cohorts": report["cohorts"]}, indent=2))
    # This is an audit, not a new acceptance gate. Integrity failure still fails.
    return int(bool(report["summary"]["stored_feature_reconstruction_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())

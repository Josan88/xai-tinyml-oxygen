"""Inspect feature precision without changing deployment or acceptance gates.

Uses the existing expanded audit and reconstructs selected worst cases. The
high-precision calculation is an offline control using the firmware's rounded
coefficients and initial states, not a proposed firmware execution mode.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from scipy import signal

from audit_feature_errors import fp32, run_audit_probe, sha256
from check_probability_parity import ROOT, HEADER, NAMES, canonical_feature_function, reference, run_probe
from verify_features import parse_c_header


CANDIDATE = {
    "status": "draft_not_adopted", "relative_or_dimensionless_budget": 1e-4,
    "output_spacing_allowance": 8, "probability_budget": 1e-4, "attribution_budget": 1e-5,
    "feature_rule": "abs(error) <= 1e-4 * local_scale + 8 * abs(spacing(float32(reference)))",
    "local_scale": {
        "mean": "reference RMS",
        "minimum_maximum": "max(abs(reference feature), reference RMS)",
        "std_range_rms_energy_derivative_bandpowers": "abs(reference feature)",
        "skewness_kurtosis_bandpower_ratio": "max(1, abs(reference feature))",
        "dominant_frequency": "exact bin equality, no tolerance",
    },
    "rank_rule": "Check all 14 deltas within 1e-5; require order only for reference magnitude gaps > 2e-5, including selected versus unselected features",
    "notes": "Engineering proposal, not a formal FP32 error bound. The 1e-4 relative budget is 0.01 percent; dimensionless features have a near-zero floor. Full-recording context differences and input quantization are separate checks. Screening this already-examined sample is not independent validation.",
}


def candidate_feature_mask(expected, actual):
    scale = np.abs(expected).copy()
    scale[:, 0] = expected[:, 5]
    for index in (2, 3):
        scale[:, index] = np.maximum(scale[:, index], expected[:, 5])
    for index in (6, 7, 13):
        scale[:, index] = np.maximum(1., scale[:, index])
    spacing = np.abs(np.spacing(np.asarray(expected, dtype=np.float32)).astype(float))
    passed = np.abs(actual - expected) <= 1e-4 * scale + 8 * spacing
    passed[:, 10] = expected[:, 10] == actual[:, 10]
    return passed


def exported_filter_constants(path):
    text = path.read_text()
    def read(name, count):
        match = re.search(r"\b" + name + r"\s*\[[^;]+?=\s*\{(.*?)\};", text, re.S)
        if match is None:
            raise ValueError(f"Missing {name}")
        values = re.findall(r"(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)f", match[1])
        if len(values) != count:
            raise ValueError(f"Unexpected {name} size")
        return fp32([float(x) for x in values])
    return read("FILTER_SOS", 24).reshape(4, 6), read("zi", 8).reshape(4, 2)


def high_precision_filter(raw, offset, sos, zi):
    centered = fp32(raw) - float(np.float32(raw[0]))
    padded = np.concatenate((-centered[27:0:-1], centered,
                             2 * centered[-1] - centered[-2:-29:-1]))
    forward, _ = signal.sosfilt(sos, padded, zi=zi * padded[0])
    reverse = forward[::-1].copy()
    backward, _ = signal.sosfilt(sos, reverse, zi=zi * reverse[0])
    return backward[::-1][27 + offset:27 + offset + 500]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-probe", required=True, type=Path)
    parser.add_argument("--probability-probe", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/parity/precision_review" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Recorded-data results must stay under ignored outputs/")
    source_report = json.loads((args.audit / "report.json").read_text())
    feature_header = ROOT / "firmware/Arduino Nano 33/include/feature_extraction.h"
    for path in (HEADER, feature_header, args.audit_probe, args.probability_probe,
                 ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h"):
        if sha256(path) != source_report["sha256"][str(path.resolve())]:
            raise ValueError(f"Source audit no longer matches {path.name}")
    with (args.audit / "feature_values.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["recording_id"], int(row["start_sample"])), {})[row["feature"]] = row
    keys = list(grouped)
    def array(column):
        return np.array([[float(grouped[key][name][column]) for name in NAMES] for key in keys])
    expected, actual = array("finite_context_fp32_input"), array("cpp")
    parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
    scales = parameters["scaler_std"]
    errors = np.abs(actual - expected)
    candidate_pass = candidate_feature_mask(expected, actual)
    selected, normalized = {}, []
    def select(index, reason):
        selected.setdefault(keys[int(index)], []).append(reason)
    for i, name in enumerate(NAMES):
        if np.max(errors[:, i]) == 0:
            continue
        select(np.argmax(errors[:, i] / scales[i]), name + ": max training-scale error")
        relative = errors[:, i] / np.maximum(np.abs(expected[:, i]), np.finfo(float).tiny)
        select(np.argmax(relative), name + ": max relative error")
        nearest_error = np.abs(fp32(expected[:, i]) - expected[:, i])
        normalized.append({"feature": name, "max_relative_error": float(relative.max()),
                           "p99_relative_error": float(np.quantile(relative, .99)),
                           "max_training_scale_error": float((errors[:, i] / scales[i]).max()),
                           "nearest_fp32_fails_original_gate": int(np.sum(nearest_error > 1e-4))})
    mean_by_rms = errors[:, 0] / expected[:, 5]
    select(np.argmax(mean_by_rms), "mean: max error divided by reference RMS")
    with (args.audit / "windows.csv").open(newline="") as stream:
        windows = list(csv.DictReader(stream))
    for name in ("numerical_probability_error", "numerical_xai_error", "full_context_probability_error"):
        row = max(windows, key=lambda r: float(r[name]))
        selected.setdefault((row["recording_id"], int(row["start_sample"])), []).append(name)
    for row in windows:
        if row["numerical_top3_indices_agree"] != "True":
            selected.setdefault((row["recording_id"], int(row["start_sample"])), []).append("top-three order disagreement")
    with (ROOT / "outputs/openox_ppg_eligible_encounters.csv").open(newline="") as stream:
        eligible = {row["encounter_id"]: row for row in csv.DictReader(stream)}
    compute = canonical_feature_function()
    def features(values):
        result = compute(values, 50.)
        return np.array([result[name] for name in NAMES])
    sos64 = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    sos_export, zi_export = exported_filter_constants(feature_header)
    output.mkdir(parents=True, exist_ok=True)
    results, provenance = [], []
    record_ids = list(dict.fromkeys(key[0] for key in selected))
    for record_number, recording_id in enumerate(record_ids):
        header = Path(eligible[recording_id]["ppg_hea_path"])
        lines = [line.strip() for line in header.read_text().splitlines() if line.strip()]
        channels, rate = int(lines[0].split()[1]), float(lines[0].split()[2])
        labels = [" ".join(line.split()[8:]).lower() for line in lines[1:1 + channels]]
        channel = next(labels.index(name) for name in ("ir signal", "ir", "red signal", "red") if name in labels)
        path = header.with_suffix(".dat")
        source = next(row for row in source_report["provenance"] if row["recording_id"] == recording_id)
        if sha256(path) != source["data_sha256"] or sha256(header) != source["header_sha256"]:
            raise ValueError("Recorded source has changed since the expanded audit")
        frames = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, channels)
        resampled = signal.resample(frames[:, channel].astype(np.float64), round(len(frames) * 50 / rate))
        record_keys = [key for key in selected if key[0] == recording_id]
        for key in record_keys:
            start = key[1]
            left = max(0, start - 250)
            raw, offset = fp32(resampled[left:start + 750]), start - left
            native = run_audit_probe(args.audit_probe.resolve(), "raw", [(raw, offset)])[0]
            saved = actual[keys.index(key)]
            np.testing.assert_array_equal(native[500:], saved)
            reference_filtered = signal.sosfiltfilt(sos64, raw)[offset:offset + 500]
            fixed_filtered = high_precision_filter(raw, offset, sos_export, zi_export)
            reference_features = features(reference_filtered)
            np.testing.assert_allclose(reference_features, expected[keys.index(key)], rtol=1e-12, atol=1e-12)
            fixed_features = features(fixed_filtered)
            float_filter_features = features(native[:500])
            feature_rows = np.array([reference_features, fixed_features, float_filter_features, native[500:]])
            probabilities, deltas = reference(feature_rows, parameters)
            native_inference = fp32(run_probe(args.probability_probe.resolve(), "features", [native[500:]]))[0]
            stage_names = ("canonical_constants_fp64", "exported_constants_fp64", "native_filter_python_features", "native_features_python_inference")
            per_stage = {name: {"probability": float(probabilities[i]),
                               "probability_error": float(abs(probabilities[i] - probabilities[0])),
                               "max_xai_error": float(np.max(np.abs(deltas[i] - deltas[0])))}
                         for i, name in enumerate(stage_names)}
            per_stage["native_complete"] = {"probability": float(native_inference[0]),
                                             "probability_error": float(abs(native_inference[0] - probabilities[0])),
                                             "max_xai_error": float(np.max(np.abs(native_inference[29:] - deltas[0])))}
            rms = reference_features[5]
            results.append({"recording_id": recording_id, "start_sample": start, "selection_reasons": selected[key],
                            "raw_context_range": float(np.ptp(raw)), "reference_rms": float(rms),
                            "context_range_over_target_rms": float(np.ptp(raw) / rms),
                            "max_filter_error_over_target_rms": float(np.max(np.abs(native[:500] - reference_filtered)) / rms),
                            "rounded_constant_filter_error_over_target_rms": float(np.max(np.abs(fixed_filtered - reference_filtered)) / rms),
                            "stages": per_stage, "feature_names": NAMES, "feature_values": feature_rows.tolist(),
                            "reference_attributions": deltas[0].tolist(), "native_attributions": native_inference[29:].tolist()})
        provenance.append(source)
        print(f"Reconstructed {record_number + 1}/{len(record_ids)} recordings; {len(results)} selected cases", flush=True)
    stage_summary = {name: {"max_probability_error": max(row["stages"][name]["probability_error"] for row in results),
                            "max_xai_error": max(row["stages"][name]["max_xai_error"] for row in results)}
                     for name in results[0]["stages"]}
    report = {"status": "review_only_no_gate_adoption", "source_audit": str(args.audit.resolve()),
              "source_report_sha256": sha256(args.audit / "report.json"), "script_sha256": sha256(__file__),
              "selection": "Union of worst relative and training-scale error per feature, worst mean/RMS, probability, XAI, context, and ranking cases; deliberately selected, not validation holdout",
              "windows_in_source": len(keys), "selected_cases": len(results), "selected_recordings": len(record_ids),
              "max_mean_error_over_reference_rms": float(mean_by_rms.max()),
              "candidate": CANDIDATE,
              "candidate_screen": {
                  "feature_failed_windows": int(np.sum(~np.all(candidate_pass, axis=1))),
                  "feature_failures": {name: int(np.sum(~candidate_pass[:, i])) for i, name in enumerate(NAMES)},
                  "probability_failed_windows": sum(float(row["numerical_probability_error"]) > 1e-4 for row in windows),
                  "attribution_failed_windows": sum(float(row["numerical_xai_error"]) > 1e-5 for row in windows),
                  "candidate_disagreements": sum(row["numerical_candidate_agrees"] != "True" for row in windows),
                  "failed_window_keys": [{"recording_id": key[0], "start_sample": key[1],
                                          "features": [name for i, name in enumerate(NAMES) if not candidate_pass[j, i]]}
                                         for j, key in enumerate(keys) if not candidate_pass[j].all()],
              },
              "feature_error_summary": normalized, "selected_case_stage_summary": stage_summary,
              "provenance": provenance, "cases": results}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "selected_cases": len(results), "stage_summary": stage_summary}, indent=2))


if __name__ == "__main__":
    main()

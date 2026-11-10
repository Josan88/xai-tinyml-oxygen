"""Compare compiled firmware with independent NumPy/SciPy calculations.

Does not train, export weights, retune, flash, or access the board. A failing
end-to-end comparison exits 1; same-feature inference can be checked separately.
Build tests/probability_probe.cpp with C++17 and the firmware include directory.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import scipy
from scipy import signal, stats

from verify_features import numpy_kan_predict, parse_c_header

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h"
NOTEBOOK = ROOT / "notebooks/10_openox_canonical_window_export.ipynb"
NAMES = ["mean", "std", "minimum", "maximum", "value_range", "rms",
         "skewness", "kurtosis", "energy", "derivative_std",
         "dominant_frequency_hz", "bandpower_0_5_2_hz", "bandpower_2_5_hz",
         "bandpower_ratio_0_5_2_to_2_5"]


def canonical_feature_function():
    # Execute only this function definition, never notebook I/O or export cells.
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "compute_window_features":
                namespace = {"np": np, "signal": signal, "stats": stats}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
                return namespace[node.name]
    raise RuntimeError("Canonical feature function not found")


def run_probe(executable, mode, rows):
    rows = np.asarray(rows, dtype=np.float32)
    payload = "\n".join(" ".join(format(float(x), ".9g") for x in row) for row in rows) + "\n"
    result = subprocess.run([str(executable), mode], input=payload, text=True,
                            capture_output=True, check=True)
    output = np.asarray([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
    if output.shape != (len(rows), 43) or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid probe output: {output.shape}")
    return output


def reference(rows, parameters):
    probabilities, attributions = [], []
    for row in rows:
        probability = numpy_kan_predict(row.astype(np.float64), parameters)[1]
        deltas = []
        for i in range(14):
            changed = row.astype(np.float64).copy()
            changed[i] = parameters["scaler_mean"][i]
            deltas.append(probability - numpy_kan_predict(changed, parameters)[1])
        probabilities.append(probability)
        attributions.append(deltas)
    return np.array(probabilities), np.array(attributions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parity")
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--inference-only", action="store_true")
    gate.add_argument("--probability-only", action="store_true",
                      help="Gate identical-feature inference and raw-window probability/candidate agreement; keep strict feature failures visible")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = parse_c_header(HEADER)
    # Quantize constants exactly as stored in the FP32 firmware header.
    parameters = {key: np.asarray(value, dtype=np.float32).astype(np.float64)
                  for key, value in parameters.items()}
    feature_csv = ROOT / "outputs/openox_canonical_window_features.csv"
    feature_rows = []
    with feature_csv.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index % 251 == 0:
                feature_rows.append([float(row[name]) for name in NAMES])
    feature_rows = np.asarray(feature_rows, dtype=np.float32)
    if not len(feature_rows) or not np.isfinite(feature_rows).all():
        raise ValueError("No finite canonical feature rows")
    native = run_probe(args.probe.resolve(), "features", feature_rows)
    python_p, python_xai = reference(feature_rows, parameters)
    probability_error = np.abs(native[:, 0] - python_p)
    xai_error = np.abs(native[:, 29:] - python_xai)
    compute_features = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    t = np.arange(500) / 50.
    cases = []
    raw_windows = []
    # Deterministic signals cover differing edge values, frequency bands and DC levels.
    for dc in (0., 2000., 100000.):
        for frequency in (1.2, 2.2, 3.0):
            raw = (dc + 100 * np.sin(2 * np.pi * frequency * t)
                   + 20 * np.sin(2 * np.pi * .1 * t)
                   + 5 * (np.sin(13 * t + .5) + np.cos(29 * t + 1.2))).astype(np.float32)
            raw_windows.append(raw)
            cases.append({"name": f"synthetic_dc{dc:g}_hz{frequency:g}"})
    raw_windows = np.array(raw_windows)
    np.savetxt(args.output / "synthetic_windows.csv", raw_windows, delimiter=",", fmt="%.9g")
    raw_windows = list(raw_windows)
    sources = []
    manifest = ROOT / "outputs/openox_sample_window_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["ppg_window_fully_in_bounds"].lower() != "true":
                continue
            path = Path(row["ppg_dat_path"])
            if not path.is_file():
                continue
            labels = row["ppg_channel_labels"].split(" | ")
            channel = next(i for i, label in enumerate(labels) if label.lower() == "ir signal")
            start, end = int(row["ppg_window_start_index"]), int(row["ppg_window_end_index"])
            frames = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, len(labels))
            segment = np.array(frames[start:end, channel], dtype=np.float64)
            resampled = signal.resample(segment, round(len(segment) * 50 / float(row["ppg_sampling_rate_hz"])))
            if len(resampled) < 500:
                continue
            case_name = f"openox_manifest_{len(sources)}"
            raw_windows.append(resampled[:500].astype(np.float32))
            context_features = np.array(list(compute_features(signal.sosfiltfilt(sos, resampled)[:500], 50.).values()))
            cases.append({"name": case_name,
                          "python_manifest_context_probability": float(numpy_kan_predict(context_features, parameters)[1])})
            sources.append({"case": case_name, "path": str(path.relative_to(ROOT)),
                            "channel": channel, "start": start, "end": end,
                            "resampling": "Fourier resample full manifest segment to 50 Hz; first 500 samples",
                            "raw_segment_sha256": hashlib.sha256(segment.tobytes()).hexdigest()})
            if len(sources) == 8:
                break
    if len(sources) != 8:
        raise RuntimeError(f"Expected eight recorded replay cases; found {len(sources)}")
    raw_windows = np.array(raw_windows)
    np.savetxt(args.output / "replay_windows.csv", raw_windows, delimiter=",", fmt="%.9g")
    native_raw = run_probe(args.probe.resolve(), "raw", raw_windows)
    feature_details = []
    for case, raw, actual in zip(cases, raw_windows, native_raw):
        raw64 = raw.astype(np.float64)
        filtered = signal.sosfiltfilt(sos, raw64)
        features = np.array(list(compute_features(filtered, 50.).values()))
        expected = numpy_kan_predict(features, parameters)[1]
        _, expected_xai = reference(features[None, :], parameters)
        # Feeding the actual C++ features into Python isolates KAN arithmetic.
        same_features_p = numpy_kan_predict(actual[1:15], parameters)[1]
        unpadded = signal.sosfiltfilt(sos, raw64, padtype=None, padlen=0)
        unpadded_features = np.array(list(compute_features(unpadded, 50.).values()))
        unpadded_p = numpy_kan_predict(unpadded_features, parameters)[1]
        case.update(python_window_probability=float(expected),
                    cpp_pipeline_probability=float(actual[0]),
                    python_on_cpp_features_probability=float(same_features_p),
                    python_unpadded_filter_probability=float(unpadded_p),
                    absolute_probability_error=float(abs(expected - actual[0])),
                    max_attribution_error=float(np.max(np.abs(expected_xai[0] - actual[29:]))),
                    candidate_agrees=bool((expected >= .5) == (actual[0] >= .5)))
        for i, name in enumerate(NAMES):
            feature_details.append({"case": case["name"], "feature": name,
                                    "python": float(features[i]), "cpp": float(actual[i + 1]),
                                    "absolute_error": float(abs(features[i] - actual[i + 1]))})
    with (args.output / "feature_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_details[0]))
        writer.writeheader()
        writer.writerows(feature_details)
    inference_pass = bool(probability_error.max() < .01 and xai_error.max() < .01)
    raw_pass = all(case["absolute_probability_error"] < .01 and case["candidate_agrees"] for case in cases)
    feature_pass = all(row["absolute_error"] <= 1e-4 for row in feature_details)
    paths = [HEADER, NOTEBOOK, feature_csv, manifest, Path(__file__),
             ROOT / "scripts/verify_features.py", ROOT / "tests/probability_probe.cpp",
             ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h",
             ROOT / "firmware/Arduino Nano 33/include/feature_extraction.h",
             ROOT / "firmware/Arduino Nano 33/include/kan_inference.h", args.probe.resolve()]
    report = {
        "scope": "Host C++ FP32 vs NumPy using exported constants; SciPy per-window preprocessing. No board verification or original training checkpoint comparison.",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "threshold": float(parameters["threshold"]),
        "same_feature_inference": {"rows": len(feature_rows), "sampling": "Every 251st canonical CSV row, starting at zero",
            "max_probability_error": float(probability_error.max()),
            "max_attribution_error": float(xai_error.max()),
            "candidate_disagreements": int(np.count_nonzero((native[:, 0] >= .5) != (python_p >= .5))),
            "pass": inference_pass},
        "raw_window_cases": cases,
        "raw_window_probability_pass": raw_pass,
        "raw_window_feature_pass_at_1e_4": feature_pass,
        "recorded_input_sources": sources,
        "limitations": ["Training notebook filters longer recordings before cutting windows; per-window SciPy is a diagnostic reference, not proof of training preprocessing equivalence.",
                        "No saved PyTorch checkpoint was found for the exported weights.",
                        "Synthetic and OpenOximetry replay inputs are not board still, wiggle, or walk captures."],
        "sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                   hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"same_feature_inference": report["same_feature_inference"],
                      "raw_window_probability_pass": raw_pass,
                      "raw_window_feature_pass_at_1e_4": feature_pass, "cases": cases}, indent=2))
    return 0 if inference_pass and (args.inference_only or (raw_pass and (args.probability_only or feature_pass))) else 1


if __name__ == "__main__":
    raise SystemExit(main())

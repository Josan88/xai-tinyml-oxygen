"""Read-only reconstruction of canonical full-recording versus window filtering."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import signal

from check_probability_parity import ROOT, NAMES, canonical_feature_function, run_probe
from verify_features import numpy_kan_predict, parse_c_header


def run_context_probe(executable, samples):
    payload = " ".join(format(float(x), ".9g") for x in np.asarray(samples, dtype=np.float32)) + "\n"
    result = subprocess.run([str(executable)], input=payload, text=True, capture_output=True, check=True)
    rows = [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
    if any(len(row) != 20 or not np.isfinite(row).all() for row in rows):
        raise RuntimeError("Invalid context probe output")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--context-probe", type=Path,
                        help="Optional BasicOxygenRiskPipeline<250> host probe for streaming context replay")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parity/context")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/openox_canonical_window_features.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = list(dict.fromkeys(row["recording_id"] for row in rows))
    selected_ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(8, len(ids)), dtype=int)]
    with (ROOT / "outputs/openox_ppg_eligible_encounters.csv").open(newline="") as stream:
        eligible = {row["encounter_id"]: row for row in csv.DictReader(stream)}
    parameters = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in
                  parse_c_header(ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h").items()}
    features = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    results, provenance, context_rows = [], [], []
    for recording_id in selected_ids:
        header = Path(eligible[recording_id]["ppg_hea_path"])
        lines = [line.strip() for line in header.read_text().splitlines() if line.strip()]
        tokens = lines[0].split()
        channels, rate = int(tokens[1]), float(tokens[2])
        labels = [" ".join(line.split()[8:]).lower() for line in lines[1:1 + channels]]
        channel = next(labels.index(name) for name in ("ir signal", "ir", "red signal", "red") if name in labels)
        path = header.with_suffix(".dat")
        frames = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, channels)
        # Exactly the canonical notebook's whole-recording resampling/filtering.
        resampled = signal.resample(frames[:, channel].astype(np.float64), round(len(frames) * 50 / rate))
        full_filtered = signal.sosfiltfilt(sos, resampled)
        recording_rows = [row for row in rows if row["recording_id"] == recording_id]
        chosen = [recording_rows[i] for i in np.linspace(0, len(recording_rows) - 1,
                                                        min(16, len(recording_rows)), dtype=int)]
        windows, pending, local_feature_rows = [], [], []
        for row in chosen:
            start = round(float(row["start_time_seconds"]) * 50)
            full_f = np.array(list(features(full_filtered[start:start + 500], 50.).values()))
            stored_f = np.array([float(row[name]) for name in NAMES])
            raw = resampled[start:start + 500].astype(np.float32)
            local_f = np.array(list(features(signal.sosfiltfilt(sos, raw.astype(np.float64)), 50.).values()))
            windows.append(raw)
            local_feature_rows.append(local_f)
            stored_p = float(numpy_kan_predict(stored_f, parameters)[1])
            full_p = float(numpy_kan_predict(full_f, parameters)[1])
            local_p = float(numpy_kan_predict(local_f, parameters)[1])
            python_5s5s = None
            # Read-only evaluation of finite context; not a deployment change.
            for history, future in ((100, 0), (250, 0), (500, 0), (100, 100), (250, 250), (500, 500)):
                left, right = max(0, start - history), min(len(resampled), start + 500 + future)
                contextual = signal.sosfiltfilt(sos, resampled[left:right])
                contextual_f = np.array(list(features(contextual[start - left:start - left + 500], 50.).values()))
                contextual_p = float(numpy_kan_predict(contextual_f, parameters)[1])
                context_rows.append({"history_seconds": history / 50, "future_seconds": future / 50,
                                     "absolute_probability_error": abs(full_p - contextual_p),
                                     "candidate_agrees": bool((full_p >= .5) == (contextual_p >= .5))})
                if history == 250 and future == 250:
                    python_5s5s = contextual_p
            pending.append({"recording_id": recording_id, "start_time_seconds": start / 50,
                            "stored_probability": stored_p, "full_recording_probability": full_p,
                            "per_window_python_probability": local_p,
                            "python_5s5s_probability": python_5s5s,
                            "stored_feature_reconstruction_pass": bool(np.allclose(full_f, stored_f, rtol=1e-7, atol=1e-6)),
                            "full_vs_window_probability_error": abs(full_p - local_p),
                            "full_vs_window_candidate_agrees": bool((full_p >= .5) == (local_p >= .5))})
        native = run_probe(args.probe.resolve(), "raw", windows)
        for row, actual, local_f in zip(pending, native, local_feature_rows):
            row["cpp_probability"] = float(actual[0])
            row["python_cpp_probability_error"] = abs(row["per_window_python_probability"] - actual[0])
            row["python_cpp_candidate_agrees"] = bool((row["per_window_python_probability"] >= .5) == (actual[0] >= .5))
            row["max_standardized_feature_error"] = float(np.max(np.abs(local_f - actual[1:15]) / parameters["scaler_std"]))
        if args.context_probe:
            need = max(round(row["start_time_seconds"] * 50) for row in pending) + 750
            streamed = run_context_probe(args.context_probe.resolve(), resampled[:min(len(resampled), need)])
            by_start = {int(row[2]): row for row in streamed}
            first = streamed[0] if streamed else None
            for row in pending:
                start = round(row["start_time_seconds"] * 50)
                actual = by_start.get(start)
                row["cpp_context_matched"] = actual is not None
                if actual is None:
                    continue
                row["cpp_context_probability"] = actual[5]
                row["cpp_context_total_samples"] = actual[1]
                row["cpp_context_delay_samples"] = actual[4]
                row["cpp_context_vs_python_5s5s_error"] = abs(actual[5] - row["python_5s5s_probability"])
                row["cpp_context_vs_full_error"] = abs(actual[5] - row["full_recording_probability"])
                row["cpp_context_candidate_agrees_5s5s"] = bool((actual[5] >= .5) == (row["python_5s5s_probability"] >= .5))
                row["cpp_context_candidate_agrees_full"] = bool((actual[5] >= .5) == (row["full_recording_probability"] >= .5))
            provenance.append({"recording_id": recording_id, "path": str(path.relative_to(ROOT)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "channel": channel, "source_rate": rate,
                               "first_decision_samples": None if first is None else first[1],
                               "first_window_start_sample": None if first is None else first[2],
                               "first_window_end_sample": None if first is None else first[3],
                               "first_decision_delay_samples": None if first is None else first[4]})
        else:
            provenance.append({"recording_id": recording_id, "path": str(path.relative_to(ROOT)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "channel": channel, "source_rate": rate})
        results.extend(pending)
    with (args.output / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {"recordings": len(selected_ids), "windows": len(results),
               "stored_feature_reconstruction_failures": sum(not r["stored_feature_reconstruction_pass"] for r in results),
               "max_full_vs_window_probability_error": max(r["full_vs_window_probability_error"] for r in results),
               "full_vs_window_candidate_disagreements": sum(not r["full_vs_window_candidate_agrees"] for r in results),
               "max_python_cpp_probability_error": max(r["python_cpp_probability_error"] for r in results),
               "python_cpp_candidate_disagreements": sum(not r["python_cpp_candidate_agrees"] for r in results),
               "max_standardized_feature_error": max(r["max_standardized_feature_error"] for r in results),
               "threshold": .5, "sampling": "8 evenly spaced recording IDs in CSV order; up to 16 evenly spaced retained windows each",
               "provenance": provenance}
    summary["finite_context_diagnostics"] = []
    for history, future in sorted({(r["history_seconds"], r["future_seconds"]) for r in context_rows}):
        subset = [r for r in context_rows if r["history_seconds"] == history and r["future_seconds"] == future]
        summary["finite_context_diagnostics"].append({"history_seconds": history, "future_seconds": future,
            "max_probability_error": max(r["absolute_probability_error"] for r in subset),
            "errors_ge_0_01": sum(r["absolute_probability_error"] >= .01 for r in subset),
            "candidate_disagreements": sum(not r["candidate_agrees"] for r in subset)})
    if args.context_probe:
        matched = [r for r in results if r.get("cpp_context_matched")]
        summary["context_pipeline"] = {
            "matched_windows": len(matched),
            "unmatched_windows": sum(not r.get("cpp_context_matched") for r in results),
            "first_decision_samples_ok": all(p.get("first_decision_samples") == 750
                                             and p.get("first_window_start_sample") == 0
                                             and p.get("first_window_end_sample") == 500
                                             and p.get("first_decision_delay_samples") == 250
                                             for p in provenance),
            "max_cpp_context_vs_python_5s5s_error": max((r["cpp_context_vs_python_5s5s_error"] for r in matched), default=None),
            "cpp_context_vs_python_5s5s_candidate_disagreements": sum(not r["cpp_context_candidate_agrees_5s5s"] for r in matched),
            "max_cpp_context_vs_full_error": max((r["cpp_context_vs_full_error"] for r in matched), default=None),
            "cpp_context_vs_full_candidate_disagreements": sum(not r["cpp_context_candidate_agrees_full"] for r in matched),
        }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    return 0 if not summary["stored_feature_reconstruction_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

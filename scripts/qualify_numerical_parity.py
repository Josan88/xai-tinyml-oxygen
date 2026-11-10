"""Prepare immutable cases and enforce the frozen contract, preserving absolute errors."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import signal

from parity_criteria import ROOT, CONTRACT, CONTRACT_SHA256, NAMES, check_window
from audit_feature_errors import fp32, sha256
from check_probability_parity import canonical_feature_function, reference, HEADER
from verify_features import parse_c_header, numpy_kan_predict


def local_output(path):
    path = path.resolve()
    if not path.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Restricted case artifacts must stay under outputs/")
    return path


def prepare(args):
    out = local_output(args.output)
    if out.exists():
        raise ValueError("Case directory already exists; refusing to overwrite")
    lock = json.loads(args.lock.read_text())
    if lock["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Frozen contract changed")
    audit = Path(lock["development_audit"])
    prior = json.loads((audit / "report.json").read_text())
    if sha256(audit / "report.json") != lock["development_report_sha256"]:
        raise ValueError("Development evidence changed")
    for path, digest in lock["metadata_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Locked source metadata changed")
    if sha256(HEADER) != lock["pre_repair_sha256"][str(HEADER.resolve())]:
        raise ValueError("Model/scaler constants changed")
    with (audit / "windows.csv").open(newline="") as stream:
        dev = list(csv.DictReader(stream))
    keys = ([(r["recording_id"], int(r["start_sample"])) for r in dev] if args.role == "development" else
            [(r["recording_id"], r["start_sample"]) for r in lock["windows"]])
    with (ROOT / "outputs/openox_ppg_eligible_encounters.csv").open(newline="") as stream:
        eligible = {r["encounter_id"]: r for r in csv.DictReader(stream)}
    parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
    compute = canonical_feature_function()
    def features(values):
        f = compute(values, 50.)
        return np.array([f[n] for n in NAMES])
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    raws, counts, feature_rows, first_probabilities, full_probabilities = [], [], [], [], []
    output_keys, provenance = [], []
    record_ids = list(dict.fromkeys(key[0] for key in keys))
    for number, recording_id in enumerate(record_ids):
        source = next(p for p in prior["provenance"] if p["recording_id"] == recording_id)
        header = Path(eligible[recording_id]["ppg_hea_path"])
        data_path = header.with_suffix(".dat")
        if sha256(data_path) != source["data_sha256"] or sha256(header) != source["header_sha256"]:
            raise ValueError("Recorded source changed")
        lines = [line.strip() for line in header.read_text().splitlines() if line.strip()]
        channels, rate = int(lines[0].split()[1]), float(lines[0].split()[2])
        labels = [" ".join(line.split()[8:]).lower() for line in lines[1:1 + channels]]
        channel = next(labels.index(name) for name in ("ir signal", "ir", "red signal", "red") if name in labels)
        frames = np.memmap(data_path, dtype=np.int32, mode="r").reshape(-1, channels)
        resampled = signal.resample(frames[:, channel].astype(np.float64), round(len(frames) * 50 / rate))
        full = signal.sosfiltfilt(sos, resampled)
        for key in (key for key in keys if key[0] == recording_id):
            start = key[1]
            left = max(0, start - 250)
            raw, offset = fp32(resampled[left:start + 750]), start - left
            if (len(raw), offset) not in ((750, 0), (1000, 250)):
                raise ValueError("Incomplete or off-stride context")
            f = features(signal.sosfiltfilt(sos, raw)[offset:offset + 500])
            first = features(signal.sosfiltfilt(sos, raw[:750])[:500])
            padded = np.zeros(1000, dtype=np.float32)
            padded[:len(raw)] = raw
            raws.append(padded)
            counts.append(len(raw))
            feature_rows.append(f)
            first_probabilities.append(numpy_kan_predict(first, parameters)[1])
            full_probabilities.append(numpy_kan_predict(features(full[start:start + 500]), parameters)[1])
            output_keys.append(key)
        provenance.append(source)
        print(f"Prepared {number + 1}/{len(record_ids)} recordings; {len(output_keys)} {args.role} cases", flush=True)
    feature_rows = np.array(feature_rows)
    probabilities, deltas = reference(feature_rows, parameters)
    arrays = {"raw": np.array(raws), "counts": np.array(counts), "features": feature_rows,
              "probabilities": probabilities, "deltas": deltas,
              "first_probabilities": np.array(first_probabilities), "full_probabilities": np.array(full_probabilities)}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite reference data; preparation failed")
    if args.role == "development":
        saved = {(r["recording_id"], int(r["start_sample"])): float(r["finite_context_probability"]) for r in dev}
        np.testing.assert_allclose(probabilities, [saved[key] for key in output_keys], rtol=0, atol=1e-12)
    out.mkdir(parents=True)
    with (out / "cases.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {"role": args.role, "created_utc": datetime.now(timezone.utc).isoformat(),
                "keys": [{"recording_id": k[0], "start_sample": k[1]} for k in output_keys],
                "counts": {"windows": len(output_keys), "recordings": len(record_ids)},
                "validation_lock": str(args.lock.resolve()), "validation_lock_sha256": sha256(args.lock),
                "contract_sha256": CONTRACT_SHA256, "model_sha256": sha256(HEADER),
                "source_audit": str(audit), "source_report_sha256": sha256(audit / "report.json"),
                "reference_code_sha256": {str(p.resolve()): sha256(p) for p in
                                           (Path(__file__), ROOT / "scripts/check_probability_parity.py", ROOT / "scripts/verify_features.py",
                                            ROOT / "notebooks/10_openox_canonical_window_export.ipynb")},
                "cases_sha256": sha256(out / "cases.npz"), "provenance": provenance}
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(out), "counts": manifest["counts"]}, indent=2))


def run_probe(path, raw, counts):
    payload = "\n".join(str(int(n)) + " " + " ".join(format(float(v), ".9g") for v in row[:int(n)])
                         for row, n in zip(raw, counts)) + "\n"
    result = subprocess.run([str(path.resolve())], input=payload, text=True, capture_output=True, check=True, timeout=180)
    rows = np.array([[float(v) for v in line.split()] for line in result.stdout.splitlines()])
    if rows.shape != (len(raw), 546) or not np.isfinite(rows).all():
        raise ValueError("Malformed or nonfinite qualification probe output")
    if not np.array_equal(rows[:, 0], np.arange(len(raw))):
        raise ValueError("Probe case order changed")
    return fp32(rows)


def run(args):
    out = local_output(args.output)
    if out.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")
    metadata = json.loads((args.cases / "manifest.json").read_text())
    if metadata["contract_sha256"] != CONTRACT_SHA256 or sha256(HEADER) != metadata["model_sha256"]:
        raise ValueError("Contract or model changed")
    if sha256(args.cases / "cases.npz") != metadata["cases_sha256"] or sha256(metadata["validation_lock"]) != metadata["validation_lock_sha256"]:
        raise ValueError("Immutable cases or validation lock changed")
    if metadata["role"] == "validation":
        if args.only_failures or not args.candidate:
            raise ValueError("Validation requires a frozen passing development candidate and cannot select failures")
        candidate = json.loads(args.candidate.read_text())
        if candidate["contract_sha256"] != CONTRACT_SHA256 or candidate["probe_sha256"] != sha256(args.probe):
            raise ValueError("Candidate does not match frozen contract/probe")
        if candidate["validation_lock_sha256"] != metadata["validation_lock_sha256"]:
            raise ValueError("Candidate validation lock differs")
        development_report = json.loads(Path(candidate["development_report"]).read_text())
        if sha256(candidate["development_report"]) != candidate["development_report_sha256"] or development_report["summary"]["failed_windows"]:
            raise ValueError("Candidate development evidence is not an unchanged pass")
        if development_report["role"] != "development" or development_report["subset"] or development_report["summary"]["windows"] != candidate["development_windows"]:
            raise ValueError("Candidate must pass the complete development set")
        for path, digest in candidate["source_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("Candidate source changed after freezing")
        if candidate["runner_sha256"] != sha256(__file__) or candidate["criteria_sha256"] != sha256(ROOT / "scripts/parity_criteria.py"):
            raise ValueError("Qualification code changed after freezing")
        receipt_path = Path(metadata["validation_lock"]).with_suffix(".use.json")
        receipt = {"first_native_use_utc": datetime.now(timezone.utc).isoformat(),
                   "candidate_sha256": sha256(args.candidate), "probe_sha256": sha256(args.probe),
                   "validation_lock_sha256": metadata["validation_lock_sha256"]}
        if receipt_path.exists():
            previous_use = json.loads(receipt_path.read_text())
            if previous_use["candidate_sha256"] != receipt["candidate_sha256"]:
                raise ValueError("Validation was already used by another candidate; it is no longer an independent set")
        else:
            with receipt_path.open("x") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")
    with np.load(args.cases / "cases.npz", allow_pickle=False) as cached:
        data = {k: cached[k] for k in cached.files}
    indices = np.arange(len(metadata["keys"]))
    if args.only_failures:
        previous = json.loads(args.only_failures.read_text())
        if previous["cases_sha256"] != metadata["cases_sha256"] or previous["role"] != "development":
            raise ValueError("Failure subset is from different development cases")
        indices = np.array([r["case_index"] for r in previous["windows"] if not r["pass"]], dtype=int)
    if not len(indices):
        raise ValueError("No cases selected")
    actual = run_probe(args.probe, data["raw"][indices], data["counts"][indices])
    results = []
    for number, case_index in enumerate(indices):
        row = actual[number]
        n = int(data["counts"][case_index])
        emissions, offset = (1, 0) if n == 750 else (2, 250)
        check = check_window(data["features"][case_index], row[9:23], data["probabilities"][case_index], row[8],
                             data["deltas"][case_index], row[32:46])
        expected_candidate = data["probabilities"][case_index] >= .5
        first_candidate = data["first_probabilities"][case_index] >= .5
        streak = (1 + int(first_candidate) if emissions == 2 else 1) if expected_candidate else 0
        timing = np.array_equal(row[3:8], [emissions - 1, n, offset, offset + 500, 250]) and row[1] == emissions
        states = np.array_equal(row[23:26], [int(expected_candidate), int(streak >= 2), streak]) and (row[2] >= .5) == first_candidate
        top = row[26:32].reshape(3, 2)
        expected_top = np.argsort(-np.abs(row[32:46]), kind="stable")[:3]
        payload_ok = np.array_equal(top[:, 0], expected_top) and np.array_equal(top[:, 1], row[32:46][expected_top])
        check.update(case_index=int(case_index), **metadata["keys"][case_index],
                     timing_pass=bool(timing), states_pass=bool(states), top3_payload_pass=bool(payload_ok),
                     full_context_probability_absolute_error=float(abs(row[8] - data["full_probabilities"][case_index])))
        check["pass"] = check["pass"] and bool(timing and states and payload_ok)
        results.append(check)
    summary = {"windows": len(results), "failed_windows": sum(not r["pass"] for r in results),
               "feature_failed_windows": sum(not r["features"]["pass"] for r in results),
               "probability_failed_windows": sum(not r["probability_pass"] for r in results),
               "attribution_failed_windows": sum(not r["attributions"]["delta_pass"] for r in results),
               "rank_failed_windows": sum(bool(r["attributions"]["rank_violations"]) for r in results),
               "candidate_disagreements": sum(not r["candidate_agrees"] for r in results),
               "streaming_failed_windows": sum(not (r["timing_pass"] and r["states_pass"] and r["top3_payload_pass"]) for r in results),
               "original_absolute_1e_4_failed_windows": sum(not r["features"]["original_absolute_1e_4_pass"] for r in results),
               "max_probability_absolute_error": max(r["probability_absolute_error"] for r in results),
               "max_attribution_absolute_error": max(r["attributions"]["max_absolute_error"] for r in results),
               "top3_order_changes": sum(r["attributions"]["top3_order_changed"] for r in results),
               "max_full_context_probability_absolute_error": max(r["full_context_probability_absolute_error"] for r in results)}
    out.mkdir(parents=True, exist_ok=False)
    with (out / "native.npz").open("xb") as stream:
        np.savez_compressed(stream, case_indices=indices, rows=actual)
    sources = [ROOT / "firmware/Arduino Nano 33/include" / name for name in
               ("feature_extraction.h", "oxygen_risk_pipeline.h", "kan_inference.h", "kan_model_weights.h")]
    report = {"status": "pass" if not summary["failed_windows"] else "failed", "parity_complete": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "role": metadata["role"], "subset": bool(args.only_failures),
              "contract": CONTRACT, "contract_sha256": CONTRACT_SHA256, "validation_lock_sha256": metadata["validation_lock_sha256"],
              "cases_sha256": metadata["cases_sha256"], "cases_manifest_sha256": sha256(args.cases / "manifest.json"),
              "probe_sha256": sha256(args.probe), "source_sha256": {str(p.resolve()): sha256(p) for p in sources},
              "runner_sha256": sha256(__file__), "criteria_sha256": sha256(ROOT / "scripts/parity_criteria.py"),
              "complete_case_count": len(metadata["keys"]),
              "summary": summary, "windows": results,
              "streaming_scope": "Each locked input context is replayed through a reset deployment pipeline. Final window and preceding warmup decision verify timing, candidate, N=2, and top-three payload. Longer continuous streams remain covered separately by existing regressions."}
    with (out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "summary": summary}, indent=2))
    return int(bool(summary["failed_windows"]))


def seal(args):
    report = json.loads(args.development_report.read_text())
    if report["role"] != "development" or report["subset"] or report["summary"]["failed_windows"]:
        raise ValueError("Only a complete passing development run can be frozen")
    if report["summary"]["windows"] != report["complete_case_count"]:
        raise ValueError("Development run is incomplete")
    if report["contract_sha256"] != CONTRACT_SHA256 or report["probe_sha256"] != sha256(args.probe):
        raise ValueError("Contract or executable changed")
    if report["runner_sha256"] != sha256(__file__) or report["criteria_sha256"] != sha256(ROOT / "scripts/parity_criteria.py"):
        raise ValueError("Qualification checks changed since development run")
    for path, digest in report["source_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Source changed since development run")
    if "100% tests passed, 0 tests failed" not in args.tests_log.read_text():
        raise ValueError("A passing CTest record is required")
    out = local_output(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidate = {"id": "development-candidate-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                 "development_report": str(args.development_report.resolve()), "development_report_sha256": sha256(args.development_report),
                 "development_windows": report["summary"]["windows"], "probe_sha256": sha256(args.probe),
                 "contract_sha256": CONTRACT_SHA256, "validation_lock_sha256": report["validation_lock_sha256"],
                 "source_sha256": report["source_sha256"], "runner_sha256": sha256(__file__),
                 "criteria_sha256": report["criteria_sha256"], "tests_log_sha256": sha256(args.tests_log),
                 "parity_complete": False}
    with out.open("x") as stream:
        json.dump(candidate, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"candidate": str(out), "sha256": sha256(out)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--role", choices=("development", "validation"), required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--probe", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--only-failures", type=Path)
    r.add_argument("--candidate", type=Path)
    s = sub.add_parser("seal")
    s.add_argument("--development-report", type=Path, required=True)
    s.add_argument("--probe", type=Path, required=True)
    s.add_argument("--tests-log", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "run": run, "seal": seal}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

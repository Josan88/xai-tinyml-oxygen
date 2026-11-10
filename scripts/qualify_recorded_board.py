"""Replay frozen recorded development cases on the context diagnostic.

No --port means host-cache verification only. Never uploads firmware.
Restricted waveforms stay under outputs/; this is not complete parity.
Validation cases are refused. Original absolute errors remain in the report.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from audit_feature_errors import sha256, fp32
from parity_criteria import ROOT, CONTRACT, CONTRACT_SHA256, check_window
from replay_context_board import (IDENTITY, parse_result, parse_all_deltas, write_command,
                                 receive, send_block, compare)
from qualify_numerical_parity import local_output


def verify_frozen(cases, native_report, native_npz, candidate_path, upload_path):
    metadata = json.loads((cases / "manifest.json").read_text())
    report = json.loads(native_report.read_text())
    candidate = json.loads(candidate_path.read_text())
    upload = json.loads(upload_path.read_text())
    if metadata["role"] != "development":
        raise ValueError("Recorded-data target qualification uses development cases only")
    if metadata["contract_sha256"] != CONTRACT_SHA256 or candidate["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Frozen contract changed")
    if sha256(cases / "cases.npz") != metadata["cases_sha256"]:
        raise ValueError("Immutable cases changed")
    if sha256(native_report) != candidate["development_report_sha256"]:
        raise ValueError("Native development report is not the frozen candidate")
    if report["role"] != "development" or report["subset"] or report["summary"]["failed_windows"]:
        raise ValueError("Frozen development evidence is not an unchanged pass")
    if report["summary"]["windows"] != candidate["development_windows"] or candidate["development_windows"] != len(metadata["keys"]):
        raise ValueError("Development case count changed")
    if report["cases_sha256"] != metadata["cases_sha256"] or report["parity_complete"]:
        raise ValueError("Native report does not match these cases")
    for path, digest in candidate["source_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Candidate source changed")
    archived = ROOT / "outputs/parity/qualification/all14-diagnostic-image/firmware.bin"
    if not upload["uploaded_and_verified"] or sha256(candidate_path) != upload["candidate_sha256"]:
        raise ValueError("Upload receipt does not establish this frozen candidate")
    if sha256(archived) != upload["image_sha256"]:
        raise ValueError("Archived diagnostic image differs from the verified upload")
    with np.load(native_npz, allow_pickle=False) as cached:
        native = {k: cached[k] for k in cached.files}
    if native["rows"].shape != (len(metadata["keys"]), 546) or not np.array_equal(native["case_indices"], np.arange(len(metadata["keys"]))):
        raise ValueError("Native rows do not cover the complete development set")
    return metadata, report, candidate, upload, native


def split_blocks(raw, count):
    if int(count) not in (750, 1000):
        raise ValueError("Incomplete or off-stride context")
    samples = np.asarray(raw[:int(count)], dtype=np.float32)
    return [samples[i:i + 250] for i in range(0, int(count), 250)]


def expected_final(row, count):
    emissions, offset = (1, 0) if int(count) == 750 else (2, 250)
    meta = [int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])]
    if meta != [emissions - 1, int(count), offset, offset + 500, 250] or int(row[1]) != emissions:
        raise ValueError("Native timing metadata is inconsistent")
    top = [[int(row[26 + i]), float(row[27 + i])] for i in (0, 2, 4)]
    return {"meta": meta, "values": fp32(row[8:23]).tolist(),
            "states": [int(row[23]), int(row[24]), int(row[25])], "top": top,
            "deltas": fp32(row[32:46]).tolist(), "first_probability": float(row[2]),
            "emissions": emissions}


def identify(connection):
    write_command(connection, "H\n")
    if receive(connection) != IDENTITY:
        raise RuntimeError("Unexpected firmware identity; no sample blocks sent")


def replay_case(connection, raw, count):
    write_command(connection, "R\n")
    if receive(connection) != "RESET 0":
        raise RuntimeError("Reset not acknowledged")
    write_command(connection, "A\n")
    if receive(connection) != "ERR no_decision":
        raise RuntimeError("Diagnostic lacks all-14 support or retained deltas after reset")
    observed = []
    for index, block in enumerate(split_blocks(raw, count)):
        line = send_block(connection, block)
        if index < 2:
            if line != f"BUFFER {(index + 1) * 250}":
                raise RuntimeError("Pipeline emitted before 750 samples")
            continue
        if not line.startswith("RESULT "):
            raise RuntimeError(f"Unexpected board response: {line}")
        observed.append(parse_result(line.split()[1:], hexadecimal=True))
    if len(observed) != (1 if int(count) == 750 else 2):
        raise RuntimeError("Unexpected decision count")
    write_command(connection, "A\n")
    observed[-1]["all_deltas"] = parse_all_deltas(receive(connection), observed[-1]["meta"][0])
    payload_pass = all(delta == observed[-1]["all_deltas"][index] for index, delta in observed[-1]["top"])
    return observed, payload_pass


def evaluate_case(python_features, python_probability, python_deltas, native_row, count, board_rows, payload_pass):
    expected = expected_final(native_row, count)
    actual = board_rows[-1]
    host = compare(actual, expected)
    python = check_window(python_features, actual["values"][1:], python_probability, actual["values"][0],
                          python_deltas, actual["all_deltas"])
    host_full = check_window(expected["values"][1:], actual["values"][1:], expected["values"][0], actual["values"][0],
                             expected["deltas"], actual["all_deltas"])
    first_ok = True
    if expected["emissions"] == 2:
        first = board_rows[0]
        first_candidate = int(expected["first_probability"] >= .5)
        first_ok = (first["meta"] == [0, 750, 0, 500, 250]
                    and abs(first["values"][0] - expected["first_probability"]) <= CONTRACT["probability_absolute"]
                    and first["states"] == [first_candidate, 0, first_candidate])
    passed = host["pass"] and python["pass"] and host_full["pass"] and payload_pass and first_ok
    return {"pass": bool(passed), "timing_state_pass": host["pass"] and first_ok,
            "top3_matches_all_deltas": bool(payload_pass), "board_vs_python": python, "board_vs_host": host_full,
            "board_vs_host_payload": host, "original_absolute_1e_4_pass": python["features"]["original_absolute_1e_4_pass"],
            "probability_absolute_error": python["probability_absolute_error"],
            "max_attribution_absolute_error": python["attributions"]["max_absolute_error"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--native-npz", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port")
    args = parser.parse_args()
    out = local_output(args.output)
    metadata, native_report, candidate, upload, native = verify_frozen(
        args.cases, args.native_report, args.native_npz, args.candidate, args.upload_receipt)
    with np.load(args.cases / "cases.npz", allow_pickle=False) as cached:
        data = {k: cached[k] for k in cached.files}
    cache_rows = []
    for i, key in enumerate(metadata["keys"]):
        expected = expected_final(native["rows"][i], data["counts"][i])
        check = check_window(data["features"][i], expected["values"][1:], data["probabilities"][i],
                             expected["values"][0], data["deltas"][i], expected["deltas"])
        check.update(case_index=i, timing_pass=True, **key)
        cache_rows.append(check)
    cache_failed = sum(not r["pass"] for r in cache_rows)
    stamp = datetime.now(timezone.utc).isoformat()
    report = {"status": "prepared_not_run", "parity_complete": False, "created_utc": stamp,
              "role": "development", "contract_sha256": CONTRACT_SHA256,
              "candidate_sha256": sha256(args.candidate), "cases_sha256": metadata["cases_sha256"],
              "native_report_sha256": sha256(args.native_report), "upload_receipt_sha256": sha256(args.upload_receipt),
              "image_sha256": upload["image_sha256"], "host_cache_failed_windows": cache_failed,
              "board_run_completed": False, "board_probability_and_state_verified": False,
              "protocol": IDENTITY,
              "scope": "Sparse recorded USB replay of frozen development contexts on the all-14 diagnostic. Not live sensing, DWT/GPIO latency, validation-set reuse, or complete parity.",
              "summary": None, "windows": []}
    print(json.dumps({"host_cache_failed_windows": cache_failed, "windows": len(cache_rows)}, indent=2))
    out.mkdir(parents=True)
    with (out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if cache_failed or not args.port:
        return int(bool(cache_failed))
    import serial
    results = []
    report["status"] = "running"
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            identify(connection)
            for i, key in enumerate(metadata["keys"]):
                board_rows, payload_pass = replay_case(connection, data["raw"][i], data["counts"][i])
                checked = evaluate_case(data["features"][i], data["probabilities"][i], data["deltas"][i],
                                        native["rows"][i], data["counts"][i], board_rows, payload_pass)
                checked.update(case_index=i, **key)
                results.append(checked)
                if (i + 1) % 50 == 0 or i + 1 == len(metadata["keys"]):
                    print(f"Board cases {i + 1}/{len(metadata['keys'])}; failed {sum(not r['pass'] for r in results)}",
                          flush=True)
        summary = {"windows": len(results), "failed_windows": sum(not r["pass"] for r in results),
                   "feature_failed_windows": sum(not r["board_vs_python"]["features"]["pass"] for r in results),
                   "probability_failed_windows": sum(not r["board_vs_python"]["probability_pass"] for r in results),
                   "attribution_failed_windows": sum(not r["board_vs_python"]["attributions"]["delta_pass"] for r in results),
                   "candidate_disagreements": sum(not r["board_vs_python"]["candidate_agrees"] for r in results),
                   "original_absolute_1e_4_failed_windows": sum(not r["original_absolute_1e_4_pass"] for r in results),
                   "max_probability_absolute_error": max(r["probability_absolute_error"] for r in results),
                   "max_attribution_absolute_error": max(r["max_attribution_absolute_error"] for r in results)}
        passed = summary["failed_windows"] == 0
        report.update(status="completed" if passed else "failed", board_run_completed=True,
                      board_probability_and_state_verified=passed, summary=summary, windows=results,
                      port=args.port)
    except Exception as error:
        report.update(status="failed", error=str(error), windows=results, port=args.port)
        raise
    finally:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "report.json").open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(out), "summary": report["summary"], "parity_complete": False}, indent=2))
    return int(not report["board_probability_and_state_verified"])


if __name__ == "__main__":
    raise SystemExit(main())

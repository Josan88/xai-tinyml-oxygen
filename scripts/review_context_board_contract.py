"""Review observed board checks; untransmitted attribution deltas remain unqualified."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from audit_feature_errors import sha256, fp32
from check_probability_parity import canonical_feature_function, reference, HEADER
from parity_criteria import ROOT, CONTRACT, CONTRACT_SHA256, NAMES, feature_check, attribution_check
from replay_context_board import fixtures, IDENTITY
from verify_features import parse_c_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "outputs").resolve()) or out.exists():
        raise ValueError("Use a new report under outputs/")
    replay = json.loads(args.replay_report.read_text())
    replay_hashes = {Path(path).as_posix(): digest for path, digest in replay["sha256"].items()}
    candidate = json.loads(args.candidate.read_text())
    upload = json.loads(args.upload_receipt.read_text())
    if not replay["board_run_completed"] or replay["protocol"] != IDENTITY:
        raise ValueError("Complete context diagnostic replay required")
    if not upload["uploaded_and_verified"] or upload["candidate_sha256"] != sha256(args.candidate):
        raise ValueError("Upload does not establish this frozen candidate")
    if candidate["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Contract changed")
    for path, digest in candidate["source_sha256"].items():
        if sha256(path) != digest or replay_hashes[Path(path).relative_to(ROOT).as_posix()] != digest:
            raise ValueError("Replay source differs from frozen candidate")
    image = "firmware/Arduino Nano 33/.pio/build/nano33ble_context_parity/firmware.bin"
    if replay_hashes[image] != upload["image_sha256"]:
        raise ValueError("Replay image differs from verified upload")
    all_14 = replay.get("all_14_deltas_requested", False)
    if all_14 and candidate["probe_sha256"] not in replay_hashes.values():
        raise ValueError("All-delta reference does not include the frozen qualification probe")
    samples_by_case = fixtures()
    if set(replay["board_results"]) != set(samples_by_case) or set(replay["reset_checks"]) != set(samples_by_case):
        raise ValueError("Replay cases or reset checks are incomplete")
    parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
    compute = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    rows = []
    for name, samples in samples_by_case.items():
        if hashlib.sha256(samples.tobytes()).hexdigest() != replay["input_sha256"][name]:
            raise ValueError("Synthetic input changed")
        actual_rows = replay["board_results"][name]
        if len(actual_rows) != (len(samples) - 500) // 250 or len(replay["reference"][name]) != len(actual_rows):
            raise ValueError("Decision count is incomplete")
        streak = 0
        for i, result in enumerate(actual_rows):
            board = result["actual"]
            host = replay["reference"][name][i]
            start, emitted = i * 250, i * 250 + 750
            left = max(0, start - 250)
            filtered = signal.sosfiltfilt(sos, fp32(samples[left:emitted]))
            computed = compute(filtered[start - left:start - left + 500], 50.)
            features = np.array([computed[n] for n in NAMES])
            probability, deltas = reference(features[None, :], parameters)
            p = float(probability[0])
            streak = min(255, streak + 1) if p >= CONTRACT["threshold"] else 0
            states = [int(p >= CONTRACT["threshold"]), int(streak >= CONTRACT["debounce"]), streak]
            feature_python = feature_check(features, board["values"][1:])
            feature_host = feature_check(host["values"][1:], board["values"][1:])
            p_error = abs(board["values"][0] - p)
            top_errors = [float(abs(delta - deltas[0, index])) for index, delta in board["top"]]
            top_indices = [index for index, _ in board["top"]]
            expected_top = np.argsort(-np.abs(deltas[0]), kind="stable")[:3].tolist()
            p_pass = bool(np.isfinite(p_error) and p_error <= CONTRACT["probability_absolute"])
            top_pass = bool(np.isfinite(top_errors).all() and max(top_errors) <= CONTRACT["attribution_absolute"])
            states_pass = board["states"] == states
            timing_pass = board["meta"] == [i, emitted, start, start + 500, 250]
            passed = bool(feature_python["pass"] and feature_host["pass"] and p_pass and top_pass
                          and top_indices == expected_top and states_pass and timing_pass and result["pass"])
            full_python, full_host = None, None
            if all_14:
                full_python = attribution_check(deltas[0], board["all_deltas"])
                full_host = attribution_check(replay["host_all_deltas"][name][i], board["all_deltas"])
                payload_pass = all(delta == board["all_deltas"][index] for index, delta in board["top"])
                passed = bool(passed and full_python["pass"] and full_host["pass"] and payload_pass)
            rows.append({"case": name, "window_index": i, "observed_checks_pass": passed,
                         "board_vs_python_features": feature_python, "board_vs_host_features": feature_host,
                         "board_vs_python_probability_absolute_error": p_error,
                         "board_vs_host_probability_absolute_error": result["probability_error"],
                         "board_vs_python_top3_absolute_errors": top_errors,
                         "board_vs_host_max_top3_absolute_error": result["max_top_attribution_error"],
                         "board_vs_python_all_deltas": full_python, "board_vs_host_all_deltas": full_host,
                         "top3_indices_match_python": top_indices == expected_top,
                         "states_pass": states_pass, "timing_pass": timing_pass})
    summary = {"decisions": len(rows), "observed_failed_decisions": sum(not r["observed_checks_pass"] for r in rows),
               "board_vs_python_original_absolute_1e_4_feature_failures": sum(not r["board_vs_python_features"]["original_absolute_1e_4_pass"] for r in rows),
               "board_vs_host_original_absolute_1e_4_feature_failures": sum(not r["board_vs_host_features"]["original_absolute_1e_4_pass"] for r in rows),
               "max_board_vs_python_probability_absolute_error": max(r["board_vs_python_probability_absolute_error"] for r in rows),
               "max_board_vs_host_probability_absolute_error": max(r["board_vs_host_probability_absolute_error"] for r in rows),
               "max_board_vs_python_top3_absolute_error": max(max(r["board_vs_python_top3_absolute_errors"]) for r in rows),
               "max_board_vs_host_top3_absolute_error": max(r["board_vs_host_max_top3_absolute_error"] for r in rows)}
    summary["max_board_vs_python_all_14_delta_absolute_error"] = max(r["board_vs_python_all_deltas"]["max_absolute_error"] for r in rows) if all_14 else None
    summary["max_board_vs_host_all_14_delta_absolute_error"] = max(r["board_vs_host_all_deltas"]["max_absolute_error"] for r in rows) if all_14 else None
    report = {"status": "observed_checks_passed" if not summary["observed_failed_decisions"] else "failed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "parity_complete": False,
              "board_all_14_attribution_deltas_observed": bool(all_14),
              "scope": "Continuous synthetic USB decisions: features, probability, transmitted top-three payload, timing and N=2/reset. All-14 native diagnostic recomputation is checked only when requested and present. This is a bounded synthetic replay, not recorded-data board or complete parity qualification. No sensor, runtime memory or latency measurement.",
              "contract_sha256": CONTRACT_SHA256, "candidate_sha256": sha256(args.candidate),
              "replay_report_sha256": sha256(args.replay_report), "upload_receipt_sha256": sha256(args.upload_receipt),
              "script_sha256": sha256(__file__), "summary": summary, "windows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    return int(bool(summary["observed_failed_decisions"]))


if __name__ == "__main__":
    raise SystemExit(main())

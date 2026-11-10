"""P5 (job F2, ADR-0008) synthetic timing replay. Never uploads firmware; opens a port only with --port.

Method of scripts/benchmark_context_board_v2.py: one D2 pulse and one DWT interval cover all 250 pushes of
a block, including the decision when due (both channels' context filtering, the beat features, 1 + 5 KAN
passes and the s_hat regressor). Talks only to firmware answering
  TIMING_REPLAY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR5=0 INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0
Inputs: the synthetic fixtures of outputs/parity/p5-board-prepared/ (the v2 fixtures, unchanged); every timed
decision is checked against the C++ host rows prepared there.

Stop rule (docs/xai-reference-label0-runbook.md, section P5, fixed before any flash): the run stops at the
first subsequent-decision block (total samples > 750) that takes >= 640 ms (the 32-record FIFO at 50 Hz),
with status "stopped_fifo_bound".

Functional check of every timed decision against its C++ host row (--functional-check):
  exact           (default) every transmitted field bit-equal in FP32
  parity-budgets  features, states, top-3 indices, beat_valid, n_beats bit-equal; probability (1e-4, exact 0.50
                  side), each top-3 value (1e-5) and s_hat under the pre-registered P5 parity budgets
                  (scripts/parity_criteria_p5.py)

  --dry-run   check archive, image, prepared set and host rows; write nothing, open no port
  no --port   write a fresh prepared report (status prepared_not_run) under outputs/
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_p5_host_parity as p5host  # noqa: E402
import p5_deployed_reference as ref  # noqa: E402
import parity_criteria_p5 as criteria  # noqa: E402
import replay_context_board_p5 as board  # noqa: E402
import replay_context_board_v2 as v2board  # noqa: E402
from benchmark_context_board import parse_memory, timing_summary, CLOCK_HZ  # noqa: E402
from record_v2_upload_receipt import archived_image, archive_key  # noqa: E402

IDENTITY = ("TIMING_REPLAY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR5=0 "
            "INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0")
IMAGE_ENV = "nano33ble_p5_context_timing"
ARCHIVE = ROOT / "outputs/firmware-p5/nano33ble_p5_context_timing"
FIFO_BOUND_MS = 640.0
TIMED_RESULT_WORDS = 23
FUNCTIONAL_CHECK_MODES = ("exact", "parity-budgets")
FIELD_CHECKS = ("features_equal", "states_equal", "top_indices_equal", "beat_valid_equal", "n_beats_equal",
                "probability_pass", "top_values_pass", "s_hat_pass")


def parse_timed(line, total_samples):
    words = line.split()
    if len(words) < 5 or words[0] != "TIMED" or total_samples < 250 or total_samples % 250:
        raise ValueError("Malformed timing response")
    cycles, elapsed_ms = int(words[1]), int(words[2])
    if not 0 < cycles < 2**32 or not 0 <= elapsed_ms < 60000:
        raise ValueError("Invalid or potentially wrapped timing interval")
    decision = None
    if total_samples < 750:
        if len(words) != 5 or words[3] != "BUFFER" or int(words[4]) != total_samples:
            raise ValueError("Unexpected warmup response")
    else:
        if words[3] != "RESULT" or len(words) != 4 + TIMED_RESULT_WORDS:
            raise ValueError("Missing or malformed decision")
        decision = board.parse_result(words[4:] + ["0"] * 5)   # no recomputed deltas on this image
        if decision["meta"] != [(total_samples - 750) // 250, total_samples, total_samples - 750, total_samples - 250, 250]:
            raise ValueError("Stale or inconsistent decision metadata")
    return {"cycles": cycles, "elapsed_ms": elapsed_ms, "microseconds": cycles * 1e6 / CLOCK_HZ, "decision": decision}


def functional_check(decision, cpp_row, mode="exact", target_std=None):
    """Transmitted fields against the C++ host row.

    exact: every field bit-equal (same code, same FP32 arithmetic).
    parity-budgets: features, states, top-3 indices, beat_valid and n_beats bit-equal; probability, the three
    top-3 values (against the host value for the same index) and s_hat under the pre-registered P5 budgets."""
    if mode not in FUNCTIONAL_CHECK_MODES:
        raise ValueError(f"Unknown functional check mode {mode!r}")
    host_index, host_value = [int(v) for v in cpp_row[p5host.TOP][0::2]], cpp_row[p5host.TOP][1::2]
    host_by_index = dict(zip(host_index, host_value))
    top_errors = [abs(float(t[1]) - float(host_by_index[t[0]])) if t[0] in host_by_index else None for t in decision["top"]]
    check = {"mode": mode,
             "features_equal": bool(np.array_equal(np.float32(decision["features"]), np.float32(cpp_row[p5host.F0:p5host.F1]))),
             "states_equal": decision["states"] == [int(v) for v in cpp_row[p5host.STATE]],
             "top_indices_equal": [t[0] for t in decision["top"]] == host_index,
             "beat_valid_equal": decision["beat_valid"] == int(cpp_row[p5host.BEAT_VALID]),
             "n_beats_equal": decision["n_beats"] == int(cpp_row[p5host.N_BEATS])}
    if mode == "exact":
        check.update(probability_pass=bool(np.float32(decision["probability"]) == np.float32(cpp_row[6])),
                     top_values_pass=all(bool(np.float32(t[1]) == np.float32(host_value[k])) for k, t in enumerate(decision["top"])),
                     s_hat_pass=bool(np.float32(decision["s_hat"]) == np.float32(cpp_row[p5host.S_HAT])))
    else:
        if target_std is None:
            raise ValueError("parity-budgets check requires target_std")
        probability = criteria.probability_check(float(cpp_row[6]), decision["probability"])
        s_hat = criteria.s_hat_check(float(cpp_row[p5host.S_HAT]), decision["s_hat"], target_std)
        budget = criteria.CONTRACT["attribution_absolute"]
        check.update(probability_pass=probability["pass"], candidate_agrees=probability["candidate_agrees"],
                     top_values_pass=all(e is not None and e <= budget for e in top_errors), top_value_budget=budget,
                     s_hat_pass=s_hat["pass"], s_hat_budget=s_hat["budget"])
    check.update(probability_error=abs(float(decision["probability"]) - float(cpp_row[6])),
                 top_value_errors=top_errors, max_top_value_error=None if None in top_errors else max(top_errors),
                 s_hat_error=abs(float(decision["s_hat"]) - float(cpp_row[p5host.S_HAT])))
    check["pass"] = all(check[k] for k in FIELD_CHECKS)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--upload-receipt", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--prepared", type=Path, default=board.PREPARED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--functional-check", choices=FUNCTIONAL_CHECK_MODES, default="exact")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error("--repetitions must be in 1..100")
    if args.port and not args.upload_receipt:
        parser.error("--port requires the receipt of the verified TIMING_REPLAY_P5 upload")
    if not args.dry_run:
        if not args.output:
            parser.error("--output required (or --dry-run)")
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to((ROOT / "outputs").resolve()):
            parser.error("Fresh --output under outputs/ required")
    image_path, image_sha256, sums = archived_image(IMAGE_ENV, args.archive)
    if board.sha256(ROOT / image_path) != image_sha256:
        raise SystemExit("Archived timing image does not match its SHA256SUMS")
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name in ("fixtures.npz", "expected_host.npz"):
        if board.sha256(prepared / name) != manifest["sha256"][name]:
            raise SystemExit(f"Prepared {name} changed")
    expected, fixtures_data, _ = v2board.load_expected(prepared)
    names = manifest["fixtures"]["names"]
    cpp = expected["fixture_cpp"]
    rows = [cpp[cpp[:, 0] == s] for s in range(len(names))]
    mode = args.functional_check
    target_std = ref.load_parameters()["target_std"] if mode == "parity-budgets" else None
    check = lambda decision, row: functional_check(decision, row, mode, target_std)  # noqa: E731
    loopback = [check(board.parse_result(board.result_words(r)), r)["pass"] for rs in rows for r in rs]
    host_pass = all(loopback)
    summary = {"protocol": IDENTITY, "image_env": IMAGE_ENV, "image_archive": archive_key(args.archive), "image_sha256": image_sha256,
               "prepared_dir": str(prepared.relative_to(ROOT).as_posix()), "decisions_per_repetition": int(sum(len(r) for r in rows)),
               "host_rows_loopback_pass": host_pass, "functional_check_mode": mode,
               "stop_rule": "any subsequent-decision block >= 640 ms", "fifo_bound_ms": FIFO_BOUND_MS}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "port_opened": False, "archive_and_image_ok": True, **summary}, indent=2))
        return 0 if host_pass else 1
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_not_run", "board_run_completed": False,
              "physical_gpio_verified": False, **summary, "repetitions": args.repetitions,
              "image_sums_sha256": board.sha256(sums), "prepared_manifest_sha256": board.sha256(prepared / "manifest.json"),
              "latency_summary": None, "observations": {},
              "scope": "Accelerated synthetic USB replay of paired blocks; DWT covers all 250 pushes per block. No sensor/BLE latency."}
    output.mkdir(parents=True)
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    save()
    if not host_pass:
        report["status"] = "host_failed"
        save()
        return 1
    if not args.port:
        print(f"Prepared {summary['decisions_per_repetition']} synthetic decisions x {args.repetitions}; board not run. {output / 'report.json'}")
        return 0
    upload = json.loads(args.upload_receipt.read_text())
    if not upload.get("uploaded_and_verified") or upload.get("image_sha256") != image_sha256:
        raise SystemExit("Upload receipt does not identify the archived TIMING_REPLAY_P5 image")
    import serial
    report.update(status="running", port=args.port, upload_receipt_sha256=board.sha256(args.upload_receipt))
    save()
    evidence = report["observations"]
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if v2board.receive_after(connection, "H\n") != IDENTITY:
                raise RuntimeError("Unexpected firmware identity; no sample or reset commands sent")
            evidence.update(blocks=[], calibration=[], memory=[])
            evidence["memory"].append({"stage": "before", **parse_memory(v2board.receive_after(connection, "M\n"))})
            for _ in range(8):
                words = v2board.receive_after(connection, "C\n").split()
                if len(words) != 3 or words[0] != "CAL":
                    raise ValueError("Malformed calibration response")
                evidence["calibration"].append({"cycles": int(words[1]), "elapsed_ms": int(words[2])})
            stopped = False
            for repetition in range(args.repetitions):
                for s, name in enumerate(names):
                    if v2board.receive_after(connection, "R\n") != "RESET 0":
                        raise RuntimeError("Reset not acknowledged")
                    if v2board.receive_after(connection, "B 00000000\n") != "ERR invalid_command":
                        raise RuntimeError("Timing image accepted a malformed block")
                    ir, red = fixtures_data[f"{name}_ir"], fixtures_data[f"{name}_red"]
                    for offset in range(0, len(ir), 250):
                        total = offset + 250
                        row = parse_timed(v2board.receive_after(connection, v2board.block_command(ir[offset:total], red[offset:total])), total)
                        row.update(case=name, repetition=repetition, total_samples=total)
                        if row["decision"] is not None:
                            row["functional_check"] = check(row["decision"], rows[s][(total - 750) // 250])
                            if not row["functional_check"]["pass"]:
                                report["functional_check_failure"] = row
                                raise RuntimeError("Timed decision differs from the C++ host pipeline")
                        evidence["blocks"].append(row)
                        if total > 750 and row["microseconds"] / 1000. >= FIFO_BOUND_MS:
                            stopped = True
                            break
                    if stopped:
                        break
                    evidence["memory"].append({"stage": "after_case", "case": name, "repetition": repetition,
                                               **parse_memory(v2board.receive_after(connection, "M\n"))})
                save()
                if stopped:
                    break
        worst = max(r["microseconds"] for r in evidence["blocks"]) / 1000.
        report.update(status="stopped_fifo_bound" if stopped else "completed", board_run_completed=not stopped,
                      latency_summary=timing_summary(evidence["blocks"]), max_block_ms=worst,
                      max_subsequent_decision_block_ms=max((r["microseconds"] / 1000. for r in evidence["blocks"]
                                                            if r["total_samples"] > 750), default=None))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["latency_summary"], indent=2))
    return 0 if not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())

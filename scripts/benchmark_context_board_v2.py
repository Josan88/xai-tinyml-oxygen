"""v2 (ADR-0006) synthetic timing replay. Without --port, prepare only. Never uploads firmware.

Same method as scripts/benchmark_context_board.py (v1) with paired IR/red blocks: one D2 pulse
and one DWT interval cover all 250 pushes of a block, including the normal decision and its
17-pass Mean-Perturbation XAI when due. Talks only to firmware answering
  TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED
Inputs are the frozen synthetic fixtures in outputs/parity/v2-board-prepared/fixtures.npz.

Defaults are the 15 Oct set (image from outputs/firmware-v2-prepared/build-receipt.json, host
expectations from outputs/parity/v2-board-prepared/). For the r2 image (22 Oct, fixed numeric
source): --archive outputs/timing-v2-prepared-r2 --prepared outputs/parity/v2-board-prepared-r2.
The fixtures are the same; the C++ host rows the board is checked against come from --prepared.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import parity_criteria_v2 as criteria  # noqa: E402
import replay_context_board_v2 as board  # noqa: E402
from benchmark_context_board import parse_memory, timing_summary, CLOCK_HZ  # noqa: E402

IDENTITY = ("TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 "
            "CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED")
IMAGE_ENV = "nano33ble_v2_context_timing"
# v1 measured subsequent decisions at median 249.702 ms (docs/on-device-timing.md). The FIFO
# holds 32 records = 640 ms at 50 Hz; a block near that bound would lose samples live.
FIFO_CAPACITY_MS = 640.0


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
        if words[3] != "RESULT":
            raise ValueError("Missing decision")
        decision = board.parse_result(words[4:])
        if decision["meta"] != [(total_samples - 750) // 250, total_samples, total_samples - 750, total_samples - 250, 250]:
            raise ValueError("Stale or inconsistent decision metadata")
    return {"cycles": cycles, "elapsed_ms": elapsed_ms, "microseconds": cycles * 1e6 / CLOCK_HZ, "decision": decision}


def functional_check(decision, cpp_row, python, index):
    """Transmitted fields only (this image has no A): probability, 16 features, states, top-3."""
    from check_v2_host_parity import F0, F1, STATE, TOP
    features = criteria.feature_check(python["features"][index], decision["features"])
    host_features = criteria.feature_check(cpp_row[F0:F1], decision["features"])
    p_error = abs(decision["probability"] - float(cpp_row[6]))
    top_ok = [t[0] for t in decision["top"]] == [int(v) for v in cpp_row[TOP][0::2]]
    top_error = max(abs(t[1] - float(cpp_row[TOP][1::2][k])) for k, t in enumerate(decision["top"]))
    states_ok = decision["states"] == [int(v) for v in cpp_row[STATE]]
    passed = (features["pass"] and host_features["pass"] and p_error <= criteria.CONTRACT["probability_absolute"]
              and top_ok and top_error <= criteria.CONTRACT["attribution_absolute"] and states_ok
              and (decision["probability"] >= .5) == (float(cpp_row[6]) >= .5))
    return {"pass": bool(passed), "probability_error_vs_host": p_error, "top3_indices_match": top_ok,
            "max_top3_delta_error_vs_host": top_error, "states_match": states_ok,
            "feature_check_vs_python": features, "feature_check_vs_host": host_features}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port")
    parser.add_argument("--upload-receipt", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="r2 image archive, e.g. outputs/timing-v2-prepared-r2 "
                                                     "(default: the 15 Oct build receipt)")
    parser.add_argument("--prepared", type=Path, default=board.PREPARED,
                        help="prepared host set with fixtures.npz and expected_host.npz (default: %(default)s)")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error("--repetitions must be in 1..100")
    if args.port and not args.upload_receipt:
        parser.error("--port requires the receipt of the verified TIMING_REPLAY_V2 upload")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Fresh --output under outputs/ required")
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    if bool(args.archive) != bool(manifest.get("image_archive")):
        parser.error("--archive (r2 image) and --prepared (host rows) must both be r2 or both be the 15 Oct defaults")
    try:
        image_path, image_sha256, receipt_path = board.archived_image(IMAGE_ENV, args.archive)
    except ValueError as error:
        parser.error(str(error))
    image = {"image_path": image_path, "firmware_bin_sha256": image_sha256}
    if board.sha256(ROOT / image["image_path"]) != image["firmware_bin_sha256"]:
        raise SystemExit("Archived timing image does not match its build receipt")
    for name in ("fixtures.npz", "expected_host.npz"):
        if board.sha256(prepared / name) != manifest["sha256"][name]:
            raise SystemExit(f"Prepared {name} changed")
    expected, fixtures_data, _ = board.load_expected(prepared)
    names = manifest["fixtures"]["names"]
    cpp = expected["fixture_cpp"]
    rows = [cpp[cpp[:, 0] == s] for s in range(len(names))]
    python = board.split_python(expected, "fixture", [len(r) for r in rows])
    host_checks = {name: [functional_check({"probability": float(r[6]), "features": r[7:23].tolist(),
                                            "states": [int(v) for v in r[23:26]],
                                            "top": [[int(r[26 + 2 * k]), float(r[27 + 2 * k])] for k in range(3)]},
                                           r, python[s], i) for i, r in enumerate(rows[s])]
                   for s, name in enumerate(names)}
    host_pass = all(c["pass"] for cs in host_checks.values() for c in cs)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_not_run",
        "board_run_completed": False, "physical_gpio_verified": False, "parity_complete": False,
        "protocol": IDENTITY, "image_env": IMAGE_ENV, "image_sha256": image["firmware_bin_sha256"],
        "build_receipt_sha256": board.sha256(receipt_path), "prepared_manifest_sha256": board.sha256(prepared / "manifest.json"),
        **({"image_archive": board.archive_key(args.archive), "prepared_dir": str(prepared.relative_to(ROOT).as_posix())}
           if args.archive else {}),
        "repetitions": args.repetitions, "decisions_per_repetition": sum(len(r) for r in rows),
        "host_pass": host_pass,
        "host_check_scope": "C++ host transmitted fields checked against themselves and against Python with the frozen criteria; the board run repeats this per timed decision.",
        "latency_summary": None, "observations": {},
        "fifo_capacity_ms": FIFO_CAPACITY_MS,
        "v1_reference": {"subsequent_decision_median_ms": 249.702, "subsequent_decision_max_ms": 250.797,
                         "first_decision_median_ms": 204.176, "source": "docs/on-device-timing.md"},
        "scope": "Accelerated synthetic USB replay of paired blocks. DWT covers all 250 pushes per block, including both channels' filtering, 16 features and 17 KAN passes when a decision is due. No sensor/BLE latency, no all-16 recomputation.",
    }
    output.mkdir(parents=True)
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    save()
    if not host_pass:
        report["status"] = "host_failed"
        save()
        return 1
    if not args.port:
        print(f"Prepared {report['decisions_per_repetition']} synthetic decisions x {args.repetitions}; board not run. {output / 'report.json'}")
        return 0
    upload = json.loads(args.upload_receipt.read_text())
    if not upload.get("uploaded_and_verified") or upload.get("image_sha256") != image["firmware_bin_sha256"]:
        raise SystemExit("Upload receipt does not identify the archived TIMING_REPLAY_V2 image")
    import serial
    report.update(status="running", port=args.port, upload_receipt_sha256=board.sha256(args.upload_receipt))
    save()
    evidence = report["observations"]
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if board.receive_after(connection, "H\n") != IDENTITY:
                raise RuntimeError("Unexpected firmware identity; no sample or reset commands sent")
            evidence.update(blocks=[], calibration=[], memory=[])
            evidence["memory"].append({"stage": "before", **parse_memory(board.receive_after(connection, "M\n"))})
            for _ in range(8):
                words = board.receive_after(connection, "C\n").split()
                if len(words) != 3 or words[0] != "CAL":
                    raise ValueError("Malformed calibration response")
                evidence["calibration"].append({"cycles": int(words[1]), "elapsed_ms": int(words[2])})
            for repetition in range(args.repetitions):
                for s, name in enumerate(names):
                    if board.receive_after(connection, "R\n") != "RESET 0":
                        raise RuntimeError("Reset not acknowledged")
                    for command in ("A\n", "B 00000000\n"):
                        if board.receive_after(connection, command) != "ERR invalid_command":
                            raise RuntimeError("Timing image accepted an unsupported command")
                    ir, red = fixtures_data[f"{name}_ir"], fixtures_data[f"{name}_red"]
                    for offset in range(0, len(ir), 250):
                        total = offset + 250
                        row = parse_timed(board.receive_after(connection, board.block_command(ir[offset:total], red[offset:total])), total)
                        row.update(case=name, repetition=repetition, total_samples=total)
                        if row["decision"] is not None:
                            index = (total - 750) // 250
                            row["functional_check"] = functional_check(row["decision"], rows[s][index], python[s], index)
                            if not row["functional_check"]["pass"]:
                                raise RuntimeError("Timed decision differs from the frozen host pipeline")
                        evidence["blocks"].append(row)
                    evidence["memory"].append({"stage": "after_case", "case": name, "repetition": repetition,
                                               **parse_memory(board.receive_after(connection, "M\n"))})
                save()
        summary = timing_summary(evidence["blocks"])
        worst = max(r["microseconds"] for r in evidence["blocks"]) / 1000.
        report.update(status="completed", board_run_completed=True, latency_summary=summary,
                      max_block_ms=worst, fifo_headroom_ratio=FIFO_CAPACITY_MS / worst)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["latency_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

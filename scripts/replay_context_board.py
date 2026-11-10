"""Validate the deployment context, debounce and XAI on host and USB board.

No --port means host validation/preparation only. Never uploads firmware.
All built-in inputs are synthetic; serial data are exact FP32 hex words.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import signal

from check_probability_parity import ROOT, canonical_feature_function, reference
from replay_board_parity import encode, decode
from verify_features import parse_c_header
from parity_criteria import attribution_check

IDENTITY = "CONTEXT_PARITY_V1 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50"


def waveform(frequency, count):
    t = np.arange(count, dtype=np.float64) / 50.
    return (2000 + 100 * np.sin(2 * np.pi * frequency * t)
            + 20 * np.sin(2 * np.pi * .1 * t)
            + 5 * (np.sin(13 * t + .5) + np.cos(29 * t + 1.2))).astype(np.float32)


def fixtures():
    return {"normal": waveform(1.2, 1750),
            "normal_risk_recovery": np.concatenate([waveform(1.2, 1000), waveform(2.2, 1250), waveform(1.2, 1500)]),
            "risk_after_reset": waveform(2.2, 1250)}


def parse_result(words, hexadecimal=False):
    if len(words) != 29:
        raise ValueError(f"Expected 29 fields; got {len(words)}")
    # The host prints max_digits10 (9 digits). Restore its exact FP32 value
    # before comparing with the board's bit-exact hex output.
    number = decode if hexadecimal else lambda word: decode(encode(float(word)))
    meta = [int(word) for word in words[:5]]
    values = [number(word) for word in words[5:20]]
    states = [int(word) for word in words[20:23]]
    top = [[int(words[i]), number(words[i + 1])] for i in (23, 25, 27)]
    if not all(map(math.isfinite, values + [item[1] for item in top])):
        raise ValueError("Non-finite result")
    if states[0] not in (0, 1) or states[1] not in (0, 1) or not 0 <= states[2] <= 255:
        raise ValueError("Invalid risk states")
    if len({item[0] for item in top}) != 3 or any(not 0 <= item[0] < 14 for item in top):
        raise ValueError("Invalid top attribution indices")
    return {"meta": meta, "values": values, "states": states, "top": top}


def host_results(executable, samples):
    payload = " ".join(format(float(x), ".9g") for x in samples) + "\n"
    result = subprocess.run([str(executable), "--extended"], input=payload,
                            text=True, capture_output=True, check=True, timeout=30)
    rows = [parse_result(line.split()) for line in result.stdout.splitlines()]
    if len(rows) != (len(samples) - 500) // 250:
        raise ValueError("Unexpected decision count")
    return rows


def python_check(samples, rows, parameters):
    compute = canonical_feature_function()
    sos = signal.butter(4, [.5, 8.], btype="bandpass", fs=50., output="sos")
    streak, details = 0, []
    for i, actual in enumerate(rows):
        start, end, emitted = i * 250, i * 250 + 500, i * 250 + 750
        left = max(0, start - 250)
        filtered = signal.sosfiltfilt(sos, samples[left:emitted].astype(np.float64))
        features = np.array(list(compute(filtered[start - left:start - left + 500], 50.).values()))
        probability, attrs = reference(features[None, :], parameters)
        p = float(probability[0])
        streak = min(255, streak + 1) if p >= .5 else 0
        states = [int(p >= .5), int(streak >= 2), streak]
        ranking = sorted(range(14), key=lambda index: -abs(attrs[0, index]))[:3]
        error = abs(p - actual["values"][0])
        attr_error = max(abs(item[1] - attrs[0, item[0]]) for item in actual["top"])
        passed = (actual["meta"] == [i, emitted, start, end, 250] and actual["states"] == states
                  and error < 1e-4 and attr_error < 1e-5
                  and [item[0] for item in actual["top"]] == ranking)
        details.append({"pass": bool(passed), "probability_error": error,
                        "max_top_attribution_error": float(attr_error), "expected_states": states})
    return details


def write_command(connection, command):
    data = command.encode("ascii")
    for offset in range(0, len(data), 64):
        piece = data[offset:offset + 64]
        if connection.write(piece) != len(piece):
            raise RuntimeError("Short USB write")
        connection.flush()
        time.sleep(.01)


def receive(connection):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        line = connection.readline().decode("ascii").strip()
        if line:
            return line
    raise RuntimeError("Diagnostic response timed out")


def send_block(connection, samples):
    if len(samples) != 250:
        raise ValueError("Block must contain exactly 250 samples")
    write_command(connection, "B " + " ".join(encode(float(value)) for value in samples) + "\n")
    return receive(connection)


def compare(actual, expected):
    p_error = abs(actual["values"][0] - expected["values"][0])
    attr_error = max(abs(a[1] - b[1]) for a, b in zip(actual["top"], expected["top"]))
    passed = (actual["meta"] == expected["meta"] and actual["states"] == expected["states"]
              and [a[0] for a in actual["top"]] == [a[0] for a in expected["top"]]
              and p_error < 1e-4 and attr_error < 1e-5)
    return {"pass": passed, "probability_error": p_error, "max_top_attribution_error": attr_error,
            "strict_raw_feature_pass": all(abs(a - b) <= 1e-4 for a, b in zip(actual["values"][1:], expected["values"][1:])),
            "actual": actual}


def parse_all_deltas(line, window_index):
    words = line.split()
    if len(words) != 16 or words[0] != "ATTR14" or int(words[1]) != window_index:
        raise ValueError("Malformed or stale all-delta response")
    values = [decode(word) for word in words[2:]]
    if not all(math.isfinite(v) and abs(v) <= 1 for v in values):
        raise ValueError("Invalid all-delta values")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--port")
    parser.add_argument("--qualification-probe", type=Path,
                        help="Observe all 14 board deltas and compare with the frozen native qualification probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "outputs/parity/context_board" / stamp
    output.mkdir(parents=True, exist_ok=True)
    samples_by_case = fixtures()
    parameters = {key: np.asarray(value, dtype=np.float32).astype(np.float64) for key, value in
                  parse_c_header(ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h").items()}
    expected = {name: host_results(args.probe.resolve(), samples) for name, samples in samples_by_case.items()}
    host_checks = {name: python_check(samples, expected[name], parameters) for name, samples in samples_by_case.items()}
    all_deltas = {}
    if args.qualification_probe:
        from qualify_numerical_parity import run_probe
        for name, samples in samples_by_case.items():
            contexts, counts = [], []
            for i in range(len(expected[name])):
                start = i * 250
                context = samples[max(0, start - 250):start + 750]
                padded = np.zeros(1000, dtype=np.float32)
                padded[:len(context)] = context
                contexts.append(padded)
                counts.append(len(context))
            native = run_probe(args.qualification_probe, np.array(contexts), np.array(counts))
            for i, row in enumerate(native):
                np.testing.assert_array_equal(row[8:23], expected[name][i]["values"])
            all_deltas[name] = native[:, 32:46].tolist()
    host_pass = all(row["pass"] for rows in host_checks.values() for row in rows)
    sources = [Path(__file__), args.probe.resolve(),
               ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h",
               ROOT / "firmware/Arduino Nano 33/include/feature_extraction.h",
               ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h",
               ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
               ROOT / "firmware/Arduino Nano 33/src/context_parity_replay.cpp",
               ROOT / "firmware/Arduino Nano 33/.pio/build/nano33ble_context_parity/firmware.bin"]
    if args.qualification_probe:
        sources.append(args.qualification_probe.resolve())
    report = {"timestamp_utc": stamp, "status": "prepared_not_run", "host_pass": host_pass,
              "board_run_completed": False, "board_probability_and_state_verified": False,
              "protocol": IDENTITY, "host_checks": host_checks, "reference": expected,
              "all_14_deltas_requested": bool(args.qualification_probe), "host_all_deltas": all_deltas,
              "sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                         hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()},
              "input_sha256": {name: hashlib.sha256(samples.tobytes()).hexdigest() for name, samples in samples_by_case.items()},
              "scope": "Synthetic USB replay; deployment context, metadata, XAI, N=2 and reset. No sensor data or latency measurement."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Independent host/SciPy check: {'PASS' if host_pass else 'FAIL'}; {sum(map(len, expected.values()))} decisions")
    print(f"Evidence: {output}")
    if not host_pass or not args.port:
        return 0 if host_pass else 1
    import serial
    board_results, reset_checks = {}, []
    report["status"] = "running"
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            write_command(connection, "H\n")
            if receive(connection) != IDENTITY:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            for name, samples in samples_by_case.items():
                write_command(connection, "R\n")
                if receive(connection) != "RESET 0":
                    raise RuntimeError("Reset not acknowledged")
                if args.qualification_probe:
                    write_command(connection, "A\n")
                    if receive(connection) != "ERR no_decision":
                        raise RuntimeError("Diagnostic lacks all-14 support or retained deltas after reset")
                # Exercise a mid-warmup reset with samples that must be discarded.
                if send_block(connection, waveform(2.2, 250)) != "BUFFER 250":
                    raise RuntimeError("Unexpected decision during warmup")
                write_command(connection, "B 00000000\n")
                if receive(connection) != "ERR invalid_command":
                    raise RuntimeError("Malformed block was not rejected")
                if send_block(connection, waveform(2.2, 250)) != "BUFFER 500":
                    raise RuntimeError("Malformed block changed pipeline state")
                write_command(connection, "R\n")
                if receive(connection) != "RESET 0":
                    raise RuntimeError("Mid-warmup reset not acknowledged")
                reset_checks.append(name)
                board_results[name] = []
                for block_index in range(len(samples) // 250):
                    line = send_block(connection, samples[block_index * 250:(block_index + 1) * 250])
                    if block_index < 2:
                        if line != f"BUFFER {(block_index + 1) * 250}":
                            raise RuntimeError("Pipeline emitted before 750 samples")
                        continue
                    if not line.startswith("RESULT "):
                        raise RuntimeError(f"Unexpected board response: {line}")
                    actual = parse_result(line.split()[1:], hexadecimal=True)
                    checked = compare(actual, expected[name][block_index - 2])
                    if args.qualification_probe:
                        write_command(connection, "A\n")
                        observed = parse_all_deltas(receive(connection), actual["meta"][0])
                        full_check = attribution_check(all_deltas[name][block_index - 2], observed)
                        payload_pass = all(delta == observed[index] for index, delta in actual["top"])
                        actual["all_deltas"] = observed
                        checked.update(all_delta_check=full_check, top3_matches_all_deltas=payload_pass)
                        checked["pass"] = checked["pass"] and full_check["pass"] and payload_pass
                    board_results[name].append(checked)
        passed = all(row["pass"] for rows in board_results.values() for row in rows)
        report.update(status="completed", board_run_completed=True,
                      board_probability_and_state_verified=passed)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(port=args.port, board_results=board_results, reset_checks=reset_checks)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"On-board context/state gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

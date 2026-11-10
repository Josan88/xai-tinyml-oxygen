"""Prepare replay goldens, or compare the explicit diagnostic USB firmware.

No port means preparation only. This script never uploads firmware.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def encode(value):
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def decode(word):
    if len(word) > 8:
        raise ValueError("Oversized FP32 word")
    return struct.unpack('<f', struct.pack('<I', int(word, 16)))[0]


def write_window(connection, window):
    payload = (" ".join(encode(value) for value in window) + "\n").encode("ascii")
    offset = 0
    while offset < len(payload):
        connection.write(payload[offset:offset + 64])
        flush = getattr(connection, "flush", None)
        if callable(flush):
            flush()
        offset += 64
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--windows", type=Path, default=ROOT / "tests/fixtures/parity_windows.csv")
    parser.add_argument("--port", help="Explicit USB port of the diagnostic firmware; omit to prepare only")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parity/board")
    args = parser.parse_args()
    with args.windows.open(newline="") as stream:
        windows = [[decode(encode(float(x))) for x in row] for row in csv.reader(stream)]
    if not windows or any(len(row) != 500 or not all(map(math.isfinite, row)) for row in windows):
        raise ValueError("Expected finite 500-sample windows")
    probe = subprocess.run([str(args.probe.resolve()), "raw"],
                           input="\n".join(" ".join(format(x, '.9g') for x in row) for row in windows) + "\n",
                           capture_output=True, text=True, check=True)
    reference = [[float(x) for x in line.split()] for line in probe.stdout.splitlines()]
    if len(reference) != len(windows) or any(len(row) != 43 or not all(map(math.isfinite, row)) for row in reference):
        raise ValueError("Invalid host reference")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "prepared_not_run", "windows": len(windows), "board_run_completed": False,
              "board_probability_verified": False,
              "fixture_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "reference": reference,
              "protocol": "PARITY_READY_V1; 500 FP32 hex words per line; each window resets debounce"}
    (args.output / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.port:
        print(f"Prepared {len(windows)} windows. No board connection or upload attempted.")
        return 0
    import serial  # Only required for an actual board run.
    results = []
    with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            line = connection.readline().strip()
            if line in (b"PARITY_READY_V1", b"ERR invalid_window"):
                break
        for index, (window, expected) in enumerate(zip(windows, reference)):
            write_window(connection, window)
            deadline = time.monotonic() + 15
            words = []
            while time.monotonic() < deadline:
                line = connection.readline().decode('ascii').strip()
                if line.startswith("ERR"):
                    raise RuntimeError(line)
                if line.startswith("RESULT "):
                    words = line.split()
                    break
            if len(words) != 46:
                raise RuntimeError("Incomplete board result")
            actual = [decode(word) for word in words[1:44]]
            if not all(map(math.isfinite, actual)) or words[44] not in ('0', '1') or words[45] != '0':
                raise RuntimeError("Invalid board values or reset-window confirmation state")
            probability_error = abs(actual[0] - expected[0])
            attribution_error = max(abs(a - b) for a, b in zip(actual[29:], expected[29:]))
            candidate_agrees = (words[44] == '1') == (expected[0] >= .5)
            results.append({"window": index, "probability_error": probability_error,
                            "max_attribution_error": attribution_error,
                            "candidate_agrees": candidate_agrees,
                            "strict_raw_feature_pass": all(abs(a - b) <= 1e-4 for a, b in zip(actual[1:15], expected[1:15])),
                            "probability_pass": probability_error < 1e-4 and attribution_error < 1e-5 and candidate_agrees,
                            "board_values": actual})
    passed = all(row["probability_pass"] for row in results)
    report.update(status="completed", board_run_completed=True,
                  board_probability_verified=passed, port=args.port, results=results)
    (args.output / "board_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Board probability/attribution gate: {'PASS' if passed else 'FAIL'} on {len(results)} windows")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

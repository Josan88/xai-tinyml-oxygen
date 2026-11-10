"""v2 (ADR-0006) context-parity board replay: prepare (host only) and run (board, only with --port).

Never uploads firmware. `prepare` opens no serial port. `run` opens the named port only after
the image, receipts and prepared set verify, and only talks to firmware that answers
  CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED
Protocol: H identifies; R resets; B <500 FP32 hex words> ingests 250 IR/red pairs
(ir0 red0 ... ir249 red249); A returns ATTR16 <window> <16 deltas> for the last decision.
RESULT has 32 words: window total start end delay probability f0..f15 candidate confirmed
streak (index delta) x3 two_wavelength_valid.

prepare writes a fresh outputs/parity/v2-board-prepared/ (status prepared_not_run):
  fixtures   three synthetic paired streams (normal, normal_risk_recovery, risk_after_reset)
  locked     recorded OpenOximetry v2 contexts, hash-selected (seed below; no labels, features
             or outcomes enter selection), up to 4 per recording, non-overlapping
and the host expectations for every emitted decision from the compiled C++ v2 probe and the
Python deployed-path reference, checked against each other with parity_criteria_v2.

Defaults are the 15 Oct set: prepared directory outputs/parity/v2-board-prepared/, image from
outputs/firmware-v2-prepared/build-receipt.json. For the r2 image (22 Oct, fixed numeric source):
  prepare --archive outputs/firmware-v2-prepared-r2 --prepared outputs/parity/v2-board-prepared-r2
  run     --prepared outputs/parity/v2-board-prepared-r2
`run` takes the image archive from the prepared manifest, so a prepared set is only ever judged
against the image it was prepared for.

Label-0-median explainer reference (30 Oct; docs/xai-reference-label0-runbook.md):
  prepare --xai-reference label0 --probe build/ref0/v2_stream_probe_ref0.exe \
          --archive outputs/firmware-v2-ref0/nano33ble_v2_context_parity_ref0 \
          --prepared outputs/parity/v2-board-prepared-ref0
  run     --prepared outputs/parity/v2-board-prepared-ref0 ...
The image env becomes nano33ble_v2_context_parity_ref0 and the expected identity the v2 identity
plus " XAI_REF=LABEL0"; both are written to the manifest, and `run` takes them from there (old
manifests carry the v2 values, so their runs are unchanged). `run --dry-run` checks the prepared
set, archive, image and (if given) upload receipt, then stops without opening a port.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_v2_host_parity as host  # noqa: E402
import parity_criteria_v2 as criteria  # noqa: E402
import v2_deployed_reference as ref  # noqa: E402
from replay_board_parity import encode, decode  # noqa: E402
from record_v2_upload_receipt import archived_image, archive_key  # noqa: E402

IDENTITY = "CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED"
# --xai-reference -> (H identity, image env). "mean" is the shipped v2 context-parity image.
PROFILES = {"mean": (IDENTITY, "nano33ble_v2_context_parity"),
            "label0": (IDENTITY + " XAI_REF=LABEL0", "nano33ble_v2_context_parity_ref0")}
PREPARED = ROOT / "outputs/parity/v2-board-prepared"
RECEIPT = ROOT / "outputs/firmware-v2-prepared/build-receipt.json"
IMAGE_ENV = "nano33ble_v2_context_parity"
SEED = "v2-board-locked-replay-2026-10-15"
PER_RECORDING = 4
RESULT_WORDS = 32
HEADER = ROOT / "firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# --- synthetic paired fixtures -------------------------------------------------------------
def waveform(frequency, count, dc, ac):
    t = np.arange(count, dtype=np.float64) / 50.
    return (dc + ac * np.sin(2 * np.pi * frequency * t) + .2 * ac * np.sin(2 * np.pi * .1 * t)
            + .05 * ac * (np.sin(13 * t + .5) + np.cos(29 * t + 1.2))).astype(np.float32)


def pair(frequency, count, ratio):
    """IR at DC 2000 / AC 100 (the v1 fixture scale); red at DC 1500 with ratio_of_ratios ~ratio."""
    return waveform(frequency, count, 2000., 100.), waveform(frequency, count, 1500., 75. * ratio)


def fixtures():
    normal, risk = (1.2, .6), (2.2, 1.0)  # v2 P ~0.07 and ~0.89 (checked at preparation)
    parts = {"normal": [(normal, 1750)],
             "normal_risk_recovery": [(normal, 1000), (risk, 1250), (normal, 1500)],
             "risk_after_reset": [(risk, 1250)]}
    out = {}
    for name, pieces in parts.items():
        streams = [pair(f, n, r) for (f, r), n in pieces]
        out[name] = (np.concatenate([s[0] for s in streams]), np.concatenate([s[1] for s in streams]))
    return out


# --- host expectations ---------------------------------------------------------------------
def host_expectations(probe, streams, scratch, reference=None):
    """C++ rows and Python reference per stream, plus their frozen-criteria comparison."""
    records = [{"samples": len(ir), "ir": ir, "red": red} for ir, red in streams]
    host.write_segments(scratch, records)
    cpp, banner = host.run_cpp(probe, scratch)
    scratch.unlink()
    p = ref.load_parameters()
    out = []
    for segment, (ir, red) in enumerate(streams):
        rows = cpp[cpp[:, 0] == segment]
        python = ref.stream(ir, red, p, reference)
        checks = host.compare({"recording_id": str(segment), "python": python}, rows)
        out.append({"cpp": rows, "python": python, "checks": checks})
    return out, banner


def checks_summary(expectations):
    rows = [c for e in expectations for c in e["checks"]]
    return {"decisions": len(rows), "failed": sum(not r["pass"] for r in rows),
            "attribution_failed": sum(not r["attribution_delta_pass"] for r in rows),
            "feature_failed": sum(not r["feature_pass"] for r in rows),
            "probability_failed": sum(not r["probability_pass"] for r in rows),
            "candidate_disagreements": sum(not r["candidate_agrees"] for r in rows),
            "states_or_metadata_failed": sum(not (r["states_pass"] and r["metadata_pass"]) for r in rows),
            "original_absolute_1e_4_failed": sum(not r["original_absolute_1e_4_pass"] for r in rows),
            "max_probability_absolute_error": max(r["probability_absolute_error"] for r in rows),
            "max_attribution_absolute_error": max(r["max_attribution_absolute_error"] for r in rows)}


def select_locked(export_df, samples):
    """Hash-ordered, outcome-blind, non-overlapping contexts; up to PER_RECORDING per recording."""
    keys = []
    for rid in dict.fromkeys(export_df["recording_id"]):
        starts = sorted(set((export_df.loc[export_df["recording_id"] == rid, "start_time_seconds"] * 50).round().astype(int)))
        candidates = [s for s in starts if s % 250 == 0 and s + ref.FIRST <= samples[rid]]
        candidates.sort(key=lambda s: hashlib.sha256(f"{SEED}:{rid}:{s}".encode()).hexdigest())
        chosen = []
        for s in candidates:
            span = (max(0, s - 250), s + 750)
            if all(max(span[0], c[0]) >= min(span[1], c[1]) for c in chosen):
                chosen.append(span)
            if len(chosen) == PER_RECORDING:
                break
        keys.extend((rid, span[1] - 750) for span in sorted(chosen))
    return keys


def prepare(args):
    out = args.prepared.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if not out.is_relative_to((ROOT / "outputs/parity").resolve()):
        raise SystemExit("--prepared must be under outputs/parity/")
    identity, image_env = PROFILES[args.xai_reference]
    if args.xai_reference != "mean" and not args.archive:
        raise SystemExit("--xai-reference label0 needs --archive (the ref0 image has no 15 Oct receipt)")
    reference = ref.load_reference(args.xai_reference)
    image_path, image_sha256, receipt_path = archived_image(image_env, args.archive)
    image = {"image_path": image_path, "firmware_bin_sha256": image_sha256}
    if sha256(ROOT / image["image_path"]) != image["firmware_bin_sha256"]:
        raise SystemExit("Archived context-parity image does not match its build receipt")
    probe = args.probe.resolve()
    scratch = ROOT / "build/v2/board-prepare-segments.f32"
    scratch.unlink(missing_ok=True)

    synthetic = fixtures()
    synthetic_expect, banner = host_expectations(probe, list(synthetic.values()), scratch, reference)
    if ("xai_reference=label0" in banner) != (args.xai_reference == "label0"):
        raise SystemExit(f"Probe banner does not match --xai-reference {args.xai_reference}: {banner!r}")
    synthetic_probabilities = {name: e["python"]["probability"].round(4).tolist() for name, e in zip(synthetic, synthetic_expect)}

    export_df = pd.read_csv(ROOT / "outputs/openox-v2/window_features.csv", low_memory=False)
    export_df["recording_id"] = export_df["recording_id"].astype(str)
    eligible = pd.read_csv(ROOT / "outputs/openox_ppg_eligible_encounters.csv").set_index("encounter_id")
    streams, sources = {}, {}
    for number, rid in enumerate(dict.fromkeys(export_df["recording_id"]), start=1):
        ir, red, source = host.recording_streams(eligible.loc[rid].to_dict())
        streams[rid] = (ir.astype(np.float32), red.astype(np.float32))
        sources[rid] = source
        if number % 25 == 0:
            print(f"  loaded {number} recordings", flush=True)
    keys = select_locked(export_df, {rid: len(s[0]) for rid, s in streams.items()})
    contexts, counts = [], []
    for rid, start in keys:
        left = max(0, start - 250)
        ir, red = streams[rid]
        contexts.append((ir[left:start + 750], red[left:start + 750]))
        counts.append(start + 750 - left)
    locked_expect, _ = host_expectations(probe, contexts, scratch, reference)

    out.mkdir(parents=True)
    raw = np.zeros((len(contexts), 2, 1000), dtype=np.float32)
    for k, (ir, red) in enumerate(contexts):
        raw[k, 0, :len(ir)], raw[k, 1, :len(red)] = ir, red
    with (out / "cases.npz").open("xb") as stream:
        np.savez_compressed(stream, raw=raw, counts=np.array(counts), starts=np.array([k[1] for k in keys]))
    with (out / "fixtures.npz").open("xb") as stream:
        np.savez_compressed(stream, **{f"{name}_ir": s[0] for name, s in synthetic.items()},
                            **{f"{name}_red": s[1] for name, s in synthetic.items()})
    pack = lambda expect, field: np.concatenate([e[field] for e in expect])  # noqa: E731
    with (out / "expected_host.npz").open("xb") as stream:
        np.savez_compressed(stream,
                            fixture_cpp=pack(synthetic_expect, "cpp"), locked_cpp=pack(locked_expect, "cpp"),
                            **{f"{part}_python_{k}": np.concatenate([e["python"][k] for e in expect])
                               for part, expect in (("fixture", synthetic_expect), ("locked", locked_expect))
                               for k in ("features", "probability", "deltas", "candidate", "confirmed", "streak", "meta", "valid")})
    failed_cases = sorted({int(c["recording_id"]) for e in locked_expect for c in e["checks"] if not c["pass"]})
    manifest = {
        "status": "prepared_not_run", "board_run_completed": False, "parity_complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": identity,
        "image_env": image_env, "image_sha256": image["firmware_bin_sha256"], "build_receipt_sha256": sha256(receipt_path),
        **({} if reference is None else {
            "xai_reference": args.xai_reference, "xai_reference_values": reference.tolist(),
            "xai_reference_header": str(ref.REFERENCE_HEADER_V2.relative_to(ROOT).as_posix()),
            "xai_reference_header_sha256": sha256(ref.REFERENCE_HEADER_V2)}),
        **({"image_archive": archive_key(args.archive), "build_receipt": str(receipt_path.relative_to(ROOT).as_posix())}
           if args.archive else {}),
        "probe_sha256": sha256(probe), "probe_banner": banner,
        "header_sha256": sha256(HEADER),
        **({"probe_compiler": args.probe_compiler} if args.probe_compiler else {}),
        "fixtures": {"names": list(synthetic), "samples": {n: len(s[0]) for n, s in synthetic.items()},
                     "python_probabilities": synthetic_probabilities,
                     "host_check": checks_summary(synthetic_expect)},
        "locked": {"selection_seed": SEED, "per_recording_max": PER_RECORDING,
                   "selection_policy": "Export windows with start % 250 == 0 and 5 s future context; SHA-256(seed:recording:start) order; first non-overlapping filter-input spans [max(0,start-250), start+750). No labels, features or model outcomes used.",
                   "scope": "Windows already evaluated by the complete v2 host run; a board-versus-frozen-host numerical check, not a holdout.",
                   "cases": len(keys), "recordings": len({k[0] for k in keys}),
                   "contexts_750": int(sum(c == 750 for c in counts)), "contexts_1000": int(sum(c == 1000 for c in counts)),
                   "keys": [{"recording_id": k[0], "start_sample": int(k[1])} for k in keys],
                   "host_check": checks_summary(locked_expect),
                   "cases_failing_host_check": failed_cases},
        "criteria": criteria.CRITERIA_DESCRIPTION,
        "board_gate": "Each board decision must pass parity_criteria_v2.check_window against BOTH the Python reference and the C++ host (all 16 deltas via A), with exact metadata, candidate/confirmed/streak, top-3 payload and two_wavelength_valid. Cases already failing host-vs-Python are reported, not excluded.",
        "sha256": {"cases.npz": sha256(out / "cases.npz"), "fixtures.npz": sha256(out / "fixtures.npz"),
                   "expected_host.npz": sha256(out / "expected_host.npz"),
                   **{str(p.relative_to(ROOT)): sha256(p) for p in
                      (Path(__file__), ROOT / "scripts/v2_deployed_reference.py", ROOT / "scripts/parity_criteria_v2.py",
                       ROOT / "scripts/check_v2_host_parity.py", ROOT / "tests/v2_stream_probe.cpp",
                       ROOT / "firmware/Arduino Nano 33/src/context_parity_replay_v2.cpp")},
                   **({} if reference is None else {str(p.relative_to(ROOT)): sha256(p) for p in
                                                    (ROOT / "scripts/xai_reference.py", ref.REFERENCE_HEADER_V2)})},
        "sources": {rid: sources[rid] for rid in dict.fromkeys(k[0] for k in keys)},
    }
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, default=float)
        stream.write("\n")
    print(json.dumps({"output": str(out), "fixtures": manifest["fixtures"]["host_check"],
                      "fixture_probabilities": synthetic_probabilities,
                      "locked": {k: manifest["locked"][k] for k in ("cases", "recordings", "contexts_750", "contexts_1000", "host_check", "cases_failing_host_check")}},
                     indent=2, default=float))
    return 0


# --- board protocol (used only by `run`) ---------------------------------------------------
def parse_result(words):
    if len(words) != RESULT_WORDS:
        raise ValueError(f"Expected {RESULT_WORDS} RESULT words; got {len(words)}")
    meta = [int(w) for w in words[:5]]
    values = [decode(w) for w in words[5:22]]
    states = [int(w) for w in words[22:25]]
    top = [[int(words[i]), decode(words[i + 1])] for i in (25, 27, 29)]
    valid = int(words[31])
    if not all(map(math.isfinite, values + [t[1] for t in top])):
        raise ValueError("Non-finite result")
    if states[0] not in (0, 1) or states[1] not in (0, 1) or not 0 <= states[2] <= 255 or valid not in (0, 1):
        raise ValueError("Invalid states")
    if len({t[0] for t in top}) != 3 or any(not 0 <= t[0] < 16 for t in top):
        raise ValueError("Invalid top attribution indices")
    return {"meta": meta, "probability": values[0], "features": values[1:], "states": states, "top": top, "valid": valid}


def parse_all_deltas(line, window_index):
    words = line.split()
    if len(words) != 18 or words[0] != "ATTR16" or int(words[1]) != window_index:
        raise ValueError(f"Malformed or stale ATTR16 response: {line[:60]}")
    values = [decode(w) for w in words[2:]]
    if not all(math.isfinite(v) and abs(v) <= 1 for v in values):
        raise ValueError("Invalid all-delta values")
    return values


def write_command(connection, command):
    data = command.encode("ascii")
    for offset in range(0, len(data), 64):
        piece = data[offset:offset + 64]
        if connection.write(piece) != len(piece):
            raise RuntimeError("Short USB write")
        connection.flush()
        time.sleep(.01)


def receive(connection, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = connection.readline().decode("ascii").strip()
        if line:
            return line
    raise RuntimeError("Diagnostic response timed out")


def block_command(ir, red):
    if len(ir) != 250 or len(red) != 250:
        raise ValueError("Block must contain exactly 250 IR/red pairs")
    words = []
    for a, b in zip(ir, red):
        words += [encode(float(a)), encode(float(b))]
    return "B " + " ".join(words) + "\n"


def judge(board, deltas, cpp_row, python, index):
    """Board decision vs Python reference and vs C++ host, frozen criteria both ways."""
    cpp_deltas = cpp_row[host.D0:host.D0 + 16]
    vs_python = criteria.check_window(python["features"][index], board["features"], python["probability"][index],
                                      board["probability"], python["deltas"][index], deltas)
    vs_host = criteria.check_window(cpp_row[host.F0:host.F1], board["features"], cpp_row[6], board["probability"],
                                    cpp_deltas, deltas)
    payload = (np.array_equal([t[0] for t in board["top"]], np.argsort(-np.abs(np.float32(deltas)), kind="stable")[:3])
               and all(np.float32(d) == np.float32(deltas[i]) for i, d in board["top"]))
    exact = (board["meta"] == python["meta"][index].tolist()
             and board["states"] == [int(python["candidate"][index]), int(python["confirmed"][index]), int(python["streak"][index])]
             and board["states"] == [int(v) for v in cpp_row[host.STATE]]
             and board["valid"] == int(python["valid"][index]))
    return {"pass": bool(vs_python["pass"] and vs_host["pass"] and payload and exact),
            "vs_python": vs_python, "vs_host": vs_host, "top3_payload_pass": bool(payload), "exact_fields_pass": bool(exact),
            "board_host_probability_error": abs(board["probability"] - float(cpp_row[6])),
            "board_host_max_delta_error": float(np.max(np.abs(np.array(deltas) - cpp_deltas)))}


def replay_stream(connection, ir, red, cpp_rows, python):
    """Reset, replay one paired stream in 250-pair blocks, judge every emitted decision."""
    write_command(connection, "R\n")
    if receive(connection) != "RESET 0":
        raise RuntimeError("Reset not acknowledged")
    write_command(connection, "A\n")
    if receive(connection) != "ERR no_decision":
        raise RuntimeError("Deltas retained after reset")
    results, decision = [], 0
    for block in range(len(ir) // 250):
        line = receive_after(connection, block_command(ir[block * 250:(block + 1) * 250], red[block * 250:(block + 1) * 250]))
        total = (block + 1) * 250
        if total < ref.FIRST:
            if line != f"BUFFER {total}":
                raise RuntimeError(f"Unexpected warmup response: {line[:60]}")
            continue
        if not line.startswith("RESULT "):
            raise RuntimeError(f"Unexpected board response: {line[:60]}")
        board = parse_result(line.split()[1:])
        write_command(connection, "A\n")
        deltas = parse_all_deltas(receive(connection), board["meta"][0])
        verdict = judge(board, deltas, cpp_rows[decision], python, decision)
        verdict.update(board=board, board_deltas=deltas)
        results.append(verdict)
        decision += 1
    if decision != len(python["probability"]):
        raise RuntimeError("Decision count differs from host expectation")
    return results


def receive_after(connection, command):
    write_command(connection, command)
    return receive(connection)


def protocol_self_checks(connection):
    """Mid-warmup reset and malformed-block rejection, as the v1 replay does."""
    ir, red = pair(2.2, 250, 1.0)
    write_command(connection, "R\n")
    if receive(connection) != "RESET 0":
        raise RuntimeError("Reset not acknowledged")
    if receive_after(connection, block_command(ir, red)) != "BUFFER 250":
        raise RuntimeError("Unexpected decision during warmup")
    if receive_after(connection, "B 00000000\n") != "ERR invalid_command":
        raise RuntimeError("Malformed block was not rejected")
    if receive_after(connection, block_command(ir, red)) != "BUFFER 500":
        raise RuntimeError("Malformed block changed pipeline state")


def verify_prepared(upload_receipt, prepared=PREPARED):
    """upload_receipt None (dry run only): everything except the receipt is checked."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name in ("cases.npz", "fixtures.npz", "expected_host.npz"):
        if sha256(prepared / name) != manifest["sha256"][name]:
            raise ValueError(f"Prepared {name} changed")
    archive = manifest.get("image_archive")
    image_env = manifest.get("image_env", IMAGE_ENV)
    image_path, image_sha256, receipt_path = archived_image(image_env, ROOT / archive if archive else None)
    if sha256(receipt_path) != manifest["build_receipt_sha256"]:
        raise ValueError("Build receipt changed since preparation")
    if image_sha256 != manifest["image_sha256"] or sha256(ROOT / image_path) != manifest["image_sha256"]:
        raise ValueError("Archived image changed")
    if upload_receipt is None:
        return manifest
    upload = json.loads(upload_receipt.read_text())
    if not upload.get("uploaded_and_verified") or upload.get("image_sha256") != manifest["image_sha256"]:
        raise ValueError("Upload receipt does not identify the prepared context-parity v2 image")
    return manifest


def load_expected(prepared=PREPARED):
    with np.load(prepared / "expected_host.npz") as e:
        expected = {k: e[k] for k in e.files}
    with np.load(prepared / "fixtures.npz") as f:
        fixtures_data = {k: f[k] for k in f.files}
    with np.load(prepared / "cases.npz") as c:
        cases = {k: c[k] for k in c.files}
    return expected, fixtures_data, cases


def split_python(expected, part, segment_rows):
    """Re-slice the packed Python arrays by the C++ segment column."""
    out, start = [], 0
    for n in segment_rows:
        out.append({k: expected[f"{part}_python_{k}"][start:start + n] for k in
                    ("features", "probability", "deltas", "candidate", "confirmed", "streak", "meta", "valid")})
        start += n
    return out


def run(args):
    if not args.dry_run and (not args.port or not args.upload_receipt):
        raise SystemExit("run needs --port and --upload-receipt for the flashed CONTEXT_PARITY_V2 image")
    if not args.dry_run and not args.output:
        raise SystemExit("Fresh --output under outputs/parity/ required")
    output = args.output.resolve() if args.output else None
    if output is not None and (output.exists() or not output.is_relative_to((ROOT / "outputs/parity").resolve())):
        raise SystemExit("Fresh --output under outputs/parity/ required")
    prepared = args.prepared.resolve()
    manifest = verify_prepared(args.upload_receipt, prepared)
    identity = manifest.get("protocol", IDENTITY)
    expected, fixtures_data, cases = load_expected(prepared)
    names = manifest["fixtures"]["names"]
    fixture_cpp, locked_cpp = expected["fixture_cpp"], expected["locked_cpp"]
    fixture_rows = [fixture_cpp[fixture_cpp[:, 0] == s] for s in range(len(names))]
    locked_rows = [locked_cpp[locked_cpp[:, 0] == s] for s in range(len(cases["counts"]))]
    fixture_python = split_python(expected, "fixture", [len(r) for r in fixture_rows])
    locked_python = split_python(expected, "locked", [len(r) for r in locked_rows])
    selected = range(len(cases["counts"])) if args.locked == "all" else range(min(int(args.locked), len(cases["counts"])))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "port_opened": False, "prepared_dir": str(prepared.relative_to(ROOT).as_posix()),
                          "prepared_hashes_ok": True, "image_env": manifest.get("image_env", IMAGE_ENV),
                          "image_archive": manifest.get("image_archive"), "image_sha256": manifest["image_sha256"],
                          "archive_and_image_ok": True, "expected_identity": identity,
                          "xai_reference": manifest.get("xai_reference", "mean"),
                          "upload_receipt": str(args.upload_receipt) if args.upload_receipt else "not given (not checked)",
                          "fixture_decisions": int(sum(len(r) for r in fixture_rows)),
                          "locked_cases_selected": len(selected),
                          "locked_decisions_selected": int(sum(len(locked_rows[k]) for k in selected))}, indent=2))
        return 0
    report = {"status": "running", "port": args.port, "protocol": identity, "image_sha256": manifest["image_sha256"],
              "upload_receipt_sha256": sha256(args.upload_receipt), "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
              "prepared_dir": str(prepared.relative_to(ROOT).as_posix()), "started_utc": datetime.now(timezone.utc).isoformat(), "fixtures": {}, "locked": []}
    output.mkdir(parents=True)
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if receive_after(connection, "H\n") != identity:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            protocol_self_checks(connection)
            for k, name in enumerate(names):
                report["fixtures"][name] = replay_stream(connection, fixtures_data[f"{name}_ir"], fixtures_data[f"{name}_red"],
                                                         fixture_rows[k], fixture_python[k])
                save()
            for k in selected:
                n = int(cases["counts"][k])
                verdicts = replay_stream(connection, cases["raw"][k, 0, :n], cases["raw"][k, 1, :n], locked_rows[k], locked_python[k])
                report["locked"].append({"case": k, **manifest["locked"]["keys"][k], "decisions": verdicts})
                if k % 25 == 0:
                    save()
        rows = [v for vs in report["fixtures"].values() for v in vs] + [v for c in report["locked"] for v in c["decisions"]]
        report.update(status="completed", board_run_completed=True, summary={
            "decisions": len(rows), "failed": sum(not r["pass"] for r in rows),
            "max_board_host_probability_error": max(r["board_host_probability_error"] for r in rows),
            "max_board_host_delta_error": max(r["board_host_max_delta_error"] for r in rows),
            "original_absolute_1e_4_failed_vs_python": sum(not r["vs_python"]["features"]["original_absolute_1e_4_pass"] for r in rows)})
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if not report["summary"]["failed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="host only; writes a fresh prepared directory")
    p.add_argument("--probe", type=Path, required=True, help="v2_stream_probe executable")
    p.add_argument("--prepared", type=Path, default=PREPARED, help="fresh output directory (default: %(default)s)")
    p.add_argument("--archive", type=Path, help="r2 image archive, e.g. outputs/firmware-v2-prepared-r2 "
                                                "(default: the 15 Oct build receipt)")
    p.add_argument("--probe-compiler", help="free text recorded in the manifest, e.g. the zig version")
    p.add_argument("--xai-reference", choices=tuple(PROFILES), default="mean",
                   help="explainer reference of the image and probe (default: mean, the shipped image)")
    r = sub.add_parser("run", help="board replay; needs the flashed CONTEXT_PARITY_V2 image")
    r.add_argument("--port", help="application port (required unless --dry-run)")
    r.add_argument("--upload-receipt", type=Path, help="required unless --dry-run")
    r.add_argument("--dry-run", action="store_true", help="verify prepared set, archive, image and receipt; open no port")
    r.add_argument("--locked", default="all", help="'all' or the number of locked cases to replay")
    r.add_argument("--prepared", type=Path, default=PREPARED, help="prepared directory (default: %(default)s)")
    r.add_argument("--output", type=Path, help="fresh directory under outputs/parity/ (required unless --dry-run)")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

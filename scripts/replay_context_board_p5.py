"""P5 (job F2, ADR-0008) context-parity board replay: prepare (host only) and run (board, only with --port).

Never uploads firmware. `prepare` opens no serial port. `run` opens the named port only after the
image, receipts and prepared set verify, and only talks to firmware that answers
  CONTEXT_PARITY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0
Protocol (src/context_parity_replay_p5.cpp): H identifies; R resets; B <500 FP32 hex words> ingests 250
IR/red pairs. RESULT has 28 words: window total start end delay probability f0..f4 candidate confirmed
streak (index delta) x3 beat_valid n_beats s_hat delta0..delta4. There is no A command.

prepare writes a fresh outputs/parity/p5-board-prepared/ (status prepared_not_run) with the v2 board set's
inputs, unchanged: the three synthetic paired fixtures and the locked OpenOximetry contexts selected by
replay_context_board_v2.select_locked (same seed, same outcome-blind rule). Host expectations for every
decision come from the compiled C++ P5 probe and scripts/p5_deployed_reference.py, checked against each
other with the pre-registered criteria (check_p5_host_parity.compare).

Board gate (docs/p5-parity-criteria.md, board section): every board decision against BOTH the Python
reference and the C++ host. (a) n_beats and beat_valid identical to both (the board does not send peak
positions); (b) inputs within the feature budgets of Python when both are beat-valid (bit-equal to the
imputation mean when not) and bit-equal to the C++ host; (c) probability 1e-4, the five deltas 1e-5,
ranking 2e-5, exact 0.50, against both; (d) s_hat against both; (e) metadata, Candidate/Confirmed/streak
and the top-3 payload exact. Cases already failing host-vs-Python count as board failures (reported,
not excluded).
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_p5_host_parity as p5host  # noqa: E402
import check_v2_host_parity as host  # noqa: E402
import p5_deployed_reference as ref  # noqa: E402
import parity_criteria_p5 as criteria  # noqa: E402
import replay_context_board_v2 as v2board  # noqa: E402
from record_v2_upload_receipt import archived_image, archive_key  # noqa: E402
from replay_board_parity import decode  # noqa: E402

IDENTITY = "CONTEXT_PARITY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0"
IMAGE_ENV = "nano33ble_p5_context_parity"
ARCHIVE = ROOT / "outputs/firmware-p5/nano33ble_p5_context_parity"
PREPARED = ROOT / "outputs/parity/p5-board-prepared"
RESULT_WORDS = 28
PYTHON_KEYS = ("features", "probability", "deltas", "s_hat", "candidate", "confirmed", "streak", "meta", "valid", "n_beats")


def sha256(path):
    return host.sha256(path)


def host_expectations(probe, streams, scratch):
    records = [{"samples": len(ir), "ir": ir, "red": red} for ir, red in streams]
    scratch.unlink(missing_ok=True)
    host.write_segments(scratch, records)
    cpp, banner = p5host.run_cpp(probe, scratch)
    scratch.unlink()
    p = ref.load_parameters()
    out = []
    for segment, (ir, red) in enumerate(streams):
        rows = cpp[cpp[:, 0] == segment]
        python = ref.stream(ir, red, p)
        checks = p5host.compare({"recording_id": str(segment), "python": python}, rows, p)
        out.append({"cpp": rows, "python": python, "checks": checks})
    return out, banner


def checks_summary(expectations):
    rows = [c for e in expectations for c in e["checks"]]
    return {"decisions": len(rows), "failed": sum(not r["pass"] for r in rows),
            **{f"{k}_failed": sum(not r[f"{k}_pass"] for r in rows)
               for k in ("a_discrete", "b_feature", "c", "d_s_hat", "e_exact")},
            "candidate_disagreements": sum(not r["candidate_agrees"] for r in rows),
            "max_probability_absolute_error": max(r["probability_absolute_error"] for r in rows),
            "max_attribution_absolute_error": max(r["max_attribution_absolute_error"] for r in rows),
            "max_s_hat_absolute_error": max(r["s_hat_absolute_error"] for r in rows)}


def prepare(args):
    out = args.prepared.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if not out.is_relative_to((ROOT / "outputs/parity").resolve()):
        raise SystemExit("--prepared must be under outputs/parity/")
    image_path, image_sha256, receipt_path = archived_image(IMAGE_ENV, args.archive)
    if sha256(ROOT / image_path) != image_sha256:
        raise SystemExit("Archived context-parity image does not match its SHA256SUMS")
    probe = args.probe.resolve()
    scratch = ROOT / "build/p5/board-prepare-segments.f32"
    synthetic = v2board.fixtures()
    synthetic_expect, banner = host_expectations(probe, list(synthetic.values()), scratch)
    if "xai_reference=label0" not in banner:
        raise SystemExit(f"Probe was not built with -DOXY_XAI_REF_LABEL0: {banner!r}")
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
    keys = v2board.select_locked(export_df, {rid: len(s[0]) for rid, s in streams.items()})
    contexts, counts = [], []
    for rid, start in keys:
        left = max(0, start - 250)
        ir, red = streams[rid]
        contexts.append((ir[left:start + 750], red[left:start + 750]))
        counts.append(start + 750 - left)
    locked_expect, _ = host_expectations(probe, contexts, scratch)
    out.mkdir(parents=True)
    raw = np.zeros((len(contexts), 2, 1000), dtype=np.float32)
    for k, (ir, red) in enumerate(contexts):
        raw[k, 0, :len(ir)], raw[k, 1, :len(red)] = ir, red
    with (out / "cases.npz").open("xb") as stream:
        np.savez_compressed(stream, raw=raw, counts=np.array(counts), starts=np.array([k[1] for k in keys]))
    with (out / "fixtures.npz").open("xb") as stream:
        np.savez_compressed(stream, **{f"{name}_ir": s[0] for name, s in synthetic.items()},
                            **{f"{name}_red": s[1] for name, s in synthetic.items()})
    with (out / "expected_host.npz").open("xb") as stream:
        np.savez_compressed(stream, fixture_cpp=np.concatenate([e["cpp"] for e in synthetic_expect]),
                            locked_cpp=np.concatenate([e["cpp"] for e in locked_expect]),
                            **{f"{part}_python_{k}": np.concatenate([e["python"][k] for e in expect])
                               for part, expect in (("fixture", synthetic_expect), ("locked", locked_expect)) for k in PYTHON_KEYS})
    failed_cases = sorted({int(c["recording_id"]) for e in locked_expect for c in e["checks"] if not c["pass"]})
    manifest = {
        "status": "prepared_not_run", "board_run_completed": False, "parity_complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": IDENTITY, "image_env": IMAGE_ENV,
        "image_archive": archive_key(args.archive), "image_sha256": image_sha256,
        "build_receipt": str(receipt_path.relative_to(ROOT).as_posix()), "build_receipt_sha256": sha256(receipt_path),
        "xai_reference": "label0", "xai_reference_values": ref.load_parameters()["reference"].tolist(),
        "probe_sha256": sha256(probe), "probe_banner": banner, **({"probe_compiler": args.probe_compiler} if args.probe_compiler else {}),
        "fixtures": {"names": list(synthetic), "samples": {n: len(s[0]) for n, s in synthetic.items()},
                     "python_probabilities": {n: e["python"]["probability"].round(4).tolist() for n, e in zip(synthetic, synthetic_expect)},
                     "host_check": checks_summary(synthetic_expect)},
        "locked": {"selection_seed": v2board.SEED, "per_recording_max": v2board.PER_RECORDING,
                   "selection_policy": "replay_context_board_v2.select_locked, unchanged: the v2 board set's contexts.",
                   "scope": "Windows already evaluated by the complete P5 host run; a board-versus-frozen-host check, not a holdout.",
                   "cases": len(keys), "recordings": len({k[0] for k in keys}),
                   "keys": [{"recording_id": k[0], "start_sample": int(k[1])} for k in keys],
                   "host_check": checks_summary(locked_expect), "cases_failing_host_check": failed_cases},
        "criteria": criteria.CRITERIA_DESCRIPTION,
        "board_gate": __doc__.split("Board gate")[1].strip(),
        "sha256": {"cases.npz": sha256(out / "cases.npz"), "fixtures.npz": sha256(out / "fixtures.npz"),
                   "expected_host.npz": sha256(out / "expected_host.npz"),
                   **{str(p.relative_to(ROOT).as_posix()): sha256(p) for p in
                      (Path(__file__), ROOT / "scripts/p5_deployed_reference.py", ROOT / "scripts/parity_criteria_p5.py",
                       ROOT / "scripts/check_p5_host_parity.py", ROOT / "tests/p5_stream_probe.cpp",
                       ROOT / "firmware/Arduino Nano 33/src/context_parity_replay_p5.cpp", ref.REFERENCE_HEADER)}},
        "sources": {rid: sources[rid] for rid in dict.fromkeys(k[0] for k in keys)},
    }
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, default=float)
        stream.write("\n")
    print(json.dumps({"output": str(out), "fixtures": manifest["fixtures"]["host_check"],
                      "locked": {k: manifest["locked"][k] for k in ("cases", "recordings", "host_check", "cases_failing_host_check")}},
                     indent=2, default=float))
    return 0


def parse_result(words):
    if len(words) != RESULT_WORDS:
        raise ValueError(f"Expected {RESULT_WORDS} RESULT words; got {len(words)}")
    meta = [int(w) for w in words[:5]]
    probability = decode(words[5])
    features = [decode(w) for w in words[6:11]]
    states = [int(w) for w in words[11:14]]
    top = [[int(words[i]), decode(words[i + 1])] for i in (14, 16, 18)]
    beat_valid, n_beats = int(words[20]), int(words[21])
    s_hat = decode(words[22])
    deltas = [decode(w) for w in words[23:28]]
    if not all(map(math.isfinite, [probability, s_hat] + features + deltas + [t[1] for t in top])):
        raise ValueError("Non-finite result")
    if states[0] not in (0, 1) or states[1] not in (0, 1) or not 0 <= states[2] <= 255 or beat_valid not in (0, 1):
        raise ValueError("Invalid states")
    if len({t[0] for t in top}) != 3 or any(not 0 <= t[0] < 5 for t in top) or not 0 <= n_beats <= 30:
        raise ValueError("Invalid top attribution indices or beat count")
    return {"meta": meta, "probability": probability, "features": features, "states": states, "top": top,
            "beat_valid": beat_valid, "n_beats": n_beats, "s_hat": s_hat, "deltas": deltas}


def judge(board, cpp_row, python, index, p):
    """Board decision vs the Python reference and vs the C++ host (classes a-e)."""
    valid_py = bool(python["valid"][index])
    discrete = (bool(board["beat_valid"]) == valid_py == bool(cpp_row[p5host.BEAT_VALID])
                and board["n_beats"] == int(python["n_beats"][index]) == int(cpp_row[p5host.N_BEATS]))
    host_features_equal = bool(np.array_equal(np.float32(board["features"]), np.float32(cpp_row[p5host.F0:p5host.F1])))
    if discrete and valid_py:
        features_vs_python = criteria.feature_check(python["features"][index], board["features"])["pass"]
    else:
        features_vs_python = discrete and bool(np.array_equal(np.float32(board["features"]), np.float32(p["imputation"])))
    c_py = (criteria.probability_check(python["probability"][index], board["probability"])["pass"]
            and criteria.attribution_check(python["deltas"][index], board["deltas"])["pass"])
    c_host = (criteria.probability_check(cpp_row[6], board["probability"])["pass"]
              and criteria.attribution_check(cpp_row[p5host.D0:p5host.D0 + 5], board["deltas"])["pass"])
    d_ok = (criteria.s_hat_check(python["s_hat"][index], board["s_hat"], p["target_std"])["pass"]
            and criteria.s_hat_check(cpp_row[p5host.S_HAT], board["s_hat"], p["target_std"])["pass"])
    deltas32 = np.float32(board["deltas"])
    payload = ([t[0] for t in board["top"]] == np.argsort(-np.abs(deltas32), kind="stable")[:3].tolist()
               and all(np.float32(d) == deltas32[i] for i, d in board["top"]))
    states_py = [int(python["candidate"][index]), int(python["confirmed"][index]), int(python["streak"][index])]
    exact = (board["meta"] == python["meta"][index].tolist() and board["states"] == states_py
             and board["states"] == [int(v) for v in cpp_row[p5host.STATE]] and payload)
    return {"pass": bool(discrete and features_vs_python and host_features_equal and c_py and c_host and d_ok and exact),
            "a_discrete_pass": bool(discrete), "b_features_vs_python_pass": bool(features_vs_python),
            "b_features_equal_host": host_features_equal, "c_vs_python_pass": bool(c_py), "c_vs_host_pass": bool(c_host),
            "d_s_hat_pass": bool(d_ok), "e_exact_pass": bool(exact),
            "board_host_probability_error": abs(board["probability"] - float(cpp_row[6])),
            "board_host_max_delta_error": float(np.max(np.abs(np.array(board["deltas"]) - cpp_row[p5host.D0:p5host.D0 + 5]))),
            "board_host_s_hat_error": abs(board["s_hat"] - float(cpp_row[p5host.S_HAT]))}


def replay_stream(connection, ir, red, cpp_rows, python, p):
    v2board.write_command(connection, "R\n")
    if v2board.receive(connection) != "RESET 0":
        raise RuntimeError("Reset not acknowledged")
    results, decision = [], 0
    for block in range(len(ir) // 250):
        line = v2board.receive_after(connection, v2board.block_command(ir[block * 250:(block + 1) * 250],
                                                                       red[block * 250:(block + 1) * 250]))
        total = (block + 1) * 250
        if total < ref.FIRST:
            if line != f"BUFFER {total}":
                raise RuntimeError(f"Unexpected warmup response: {line[:60]}")
            continue
        if not line.startswith("RESULT "):
            raise RuntimeError(f"Unexpected board response: {line[:60]}")
        board = parse_result(line.split()[1:])
        verdict = judge(board, cpp_rows[decision], python, decision, p)
        verdict.update(board=board)
        results.append(verdict)
        decision += 1
    if decision != len(python["probability"]):
        raise RuntimeError("Decision count differs from host expectation")
    return results


def result_words(row):
    """The 28 RESULT words the P5 parity image prints for a C++ host row (FP32 as hex)."""
    from replay_board_parity import encode
    f = lambda v: encode(float(np.float32(v)))  # noqa: E731
    words = [str(int(v)) for v in row[1:6]] + [f(row[6])] + [f(v) for v in row[p5host.F0:p5host.F1]]
    words += [str(int(v)) for v in row[p5host.STATE]]
    for k in range(3):
        words += [str(int(row[p5host.TOP][2 * k])), f(row[p5host.TOP][2 * k + 1])]
    words += [str(int(row[p5host.BEAT_VALID])), str(int(row[p5host.N_BEATS])), f(row[p5host.S_HAT])]
    return words + [f(v) for v in row[p5host.D0:p5host.D0 + 5]]


def loopback_check(rows_by_stream, python_by_stream):
    """No port: each C++ host row, encoded as the board would print it, must parse and pass judge()."""
    p = ref.load_parameters()
    verdicts = [judge(parse_result(result_words(row)), row, python, i, p)
                for rows, python in zip(rows_by_stream, python_by_stream) for i, row in enumerate(rows)]
    return {"decisions": len(verdicts), "failed": sum(not v["pass"] for v in verdicts)}


def verify_prepared(upload_receipt, prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name in ("cases.npz", "fixtures.npz", "expected_host.npz"):
        if sha256(prepared / name) != manifest["sha256"][name]:
            raise ValueError(f"Prepared {name} changed")
    image_path, image_sha256, receipt_path = archived_image(manifest["image_env"], ROOT / manifest["image_archive"])
    if sha256(receipt_path) != manifest["build_receipt_sha256"]:
        raise ValueError("Archive SHA256SUMS changed since preparation")
    if image_sha256 != manifest["image_sha256"] or sha256(ROOT / image_path) != manifest["image_sha256"]:
        raise ValueError("Archived image changed")
    if upload_receipt is not None:
        upload = json.loads(upload_receipt.read_text())
        if not upload.get("uploaded_and_verified") or upload.get("image_sha256") != manifest["image_sha256"]:
            raise ValueError("Upload receipt does not identify the prepared P5 context-parity image")
    return manifest


def split_python(expected, part, segment_rows):
    out, start = [], 0
    for n in segment_rows:
        out.append({k: expected[f"{part}_python_{k}"][start:start + n] for k in PYTHON_KEYS})
        start += n
    return out


def run(args):
    if not args.dry_run and (not args.port or not args.upload_receipt or not args.output):
        raise SystemExit("run needs --port, --upload-receipt and a fresh --output (or --dry-run)")
    output = args.output.resolve() if args.output else None
    if output is not None and (output.exists() or not output.is_relative_to((ROOT / "outputs/parity").resolve())):
        raise SystemExit("Fresh --output under outputs/parity/ required")
    prepared = args.prepared.resolve()
    manifest = verify_prepared(args.upload_receipt, prepared)
    expected, fixtures_data, cases = v2board.load_expected(prepared)
    names = manifest["fixtures"]["names"]
    fixture_cpp, locked_cpp = expected["fixture_cpp"], expected["locked_cpp"]
    fixture_rows = [fixture_cpp[fixture_cpp[:, 0] == s] for s in range(len(names))]
    locked_rows = [locked_cpp[locked_cpp[:, 0] == s] for s in range(len(cases["counts"]))]
    fixture_python = split_python(expected, "fixture", [len(r) for r in fixture_rows])
    locked_python = split_python(expected, "locked", [len(r) for r in locked_rows])
    selected = range(len(cases["counts"])) if args.locked == "all" else range(min(int(args.locked), len(cases["counts"])))
    if args.dry_run:
        loopback = loopback_check(fixture_rows + [locked_rows[k] for k in selected],
                                  fixture_python + [locked_python[k] for k in selected])
        print(json.dumps({"dry_run": True, "port_opened": False, "loopback": loopback, "prepared_dir": str(prepared.relative_to(ROOT).as_posix()),
                          "prepared_hashes_ok": True, "image_env": manifest["image_env"], "image_archive": manifest["image_archive"],
                          "image_sha256": manifest["image_sha256"], "archive_and_image_ok": True, "expected_identity": manifest["protocol"],
                          "upload_receipt": str(args.upload_receipt) if args.upload_receipt else "not given (not checked)",
                          "fixture_decisions": int(sum(len(r) for r in fixture_rows)), "locked_cases_selected": len(selected),
                          "locked_decisions_selected": int(sum(len(locked_rows[k]) for k in selected)),
                          "locked_cases_failing_host_check": manifest["locked"]["cases_failing_host_check"]}, indent=2))
        return 0
    p = ref.load_parameters()
    report = {"status": "running", "port": args.port, "protocol": manifest["protocol"], "image_sha256": manifest["image_sha256"],
              "upload_receipt_sha256": sha256(args.upload_receipt), "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
              "prepared_dir": str(prepared.relative_to(ROOT).as_posix()), "started_utc": datetime.now(timezone.utc).isoformat(),
              "fixtures": {}, "locked": []}
    output.mkdir(parents=True)
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if v2board.receive_after(connection, "H\n") != manifest["protocol"]:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            v2board.protocol_self_checks(connection)
            for k, name in enumerate(names):
                report["fixtures"][name] = replay_stream(connection, fixtures_data[f"{name}_ir"], fixtures_data[f"{name}_red"],
                                                         fixture_rows[k], fixture_python[k], p)
                save()
            for k in selected:
                n = int(cases["counts"][k])
                verdicts = replay_stream(connection, cases["raw"][k, 0, :n], cases["raw"][k, 1, :n], locked_rows[k], locked_python[k], p)
                report["locked"].append({"case": k, **manifest["locked"]["keys"][k], "decisions": verdicts})
                if k % 25 == 0:
                    save()
        rows = [v for vs in report["fixtures"].values() for v in vs] + [v for c in report["locked"] for v in c["decisions"]]
        report.update(status="completed", board_run_completed=True, summary={
            "decisions": len(rows), "failed": sum(not r["pass"] for r in rows),
            **{f"{k}_failed": sum(not r[k] for r in rows) for k in
               ("a_discrete_pass", "b_features_vs_python_pass", "b_features_equal_host", "c_vs_python_pass", "c_vs_host_pass",
                "d_s_hat_pass", "e_exact_pass")},
            "max_board_host_probability_error": max(r["board_host_probability_error"] for r in rows),
            "max_board_host_delta_error": max(r["board_host_max_delta_error"] for r in rows),
            "max_board_host_s_hat_error": max(r["board_host_s_hat_error"] for r in rows)})
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
    pp = sub.add_parser("prepare", help="host only; writes a fresh prepared directory")
    pp.add_argument("--probe", type=Path, required=True, help="p5_stream_probe (-DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0)")
    pp.add_argument("--prepared", type=Path, default=PREPARED)
    pp.add_argument("--archive", type=Path, default=ARCHIVE)
    pp.add_argument("--probe-compiler", help="free text recorded in the manifest, e.g. the zig version")
    r = sub.add_parser("run", help="board replay; needs the flashed CONTEXT_PARITY_P5 image")
    r.add_argument("--port")
    r.add_argument("--upload-receipt", type=Path)
    r.add_argument("--dry-run", action="store_true", help="verify prepared set, archive, image and receipt; open no port")
    r.add_argument("--locked", default="all", help="'all' or the number of locked cases to replay")
    r.add_argument("--prepared", type=Path, default=PREPARED)
    r.add_argument("--output", type=Path)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

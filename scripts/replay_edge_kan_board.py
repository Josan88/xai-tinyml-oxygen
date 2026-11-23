"""Edge-KAN board parity and timing (docs/edge-kan-board-preregistration.md).

Never uploads firmware. Opens --port only after the image archive and upload receipt verify, and only
talks to firmware that answers the expected identity.

  parity  CONTEXT_PARITY_EKAN image: replays the v2 r2 prepared set (3 fixtures, 611 locked contexts,
          1,243 decisions) and judges every decision by the pre-registered gate.
  timing  TIMING_REPLAY_EKAN image: fixture streams x --repetitions, calibrations and a memory snapshot.

Host edge expectations come from tests/edge_kan_probe.cpp built as in train_edge_kan.py parity
(outputs/edge-kan-v1/parity/edge_kan_probe_ekan16.exe), run on the C++ host v2 features.

--ref0 (docs/edge-kan-label0-board-preregistration.md): the label-0-reference images
(outputs/firmware-ekan-ref0/), identities ending " XAI_REF=LABEL0", the flag-on probe
(outputs/edge-kan-label0/host-parity/edge_kan_probe_ekan16_ref0.exe), output under
outputs/edge-kan-label0/, and one more check per decision: the board's edge probability and deltas
against the Python float64 label-0 deltas on the same C++ host features (1e-4, 1e-5).

--fpu (docs/module8-review-fpu-plan.md, Module 8 review M11): the hardware-FPU builds of the same envs
(outputs/firmware-ekan-fpu/<env>_fpu), same identities, probe and gates, output under outputs/edge-kan-fpu/;
the pipeline summary adds a decision-by-decision comparison with the soft-float board run of 12 Nov.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_v2_host_parity as host  # noqa: E402
import parity_criteria_v2 as criteria  # noqa: E402
from record_v2_upload_receipt import archived_image, sha256  # noqa: E402
from replay_board_parity import decode  # noqa: E402
from board_serial import block_command, receive_after  # noqa: E402

PREPARED = ROOT / "outputs/parity/v2-board-prepared-r2"
PROBE = ROOT / "outputs/edge-kan-v1/parity/edge_kan_probe_ekan16.exe"
PREREG = ROOT / "docs/edge-kan-board-preregistration.md"
PIPELINE_PREREG = ROOT / "docs/edge-kan-operating-point-preregistration.md"
OUT_ROOT = ROOT / "outputs/edge-kan-v1"
ARCHIVE_ROOT = ROOT / "outputs/firmware-ekan"
REF0 = False
PROFILES = {
    "parity": ("nano33ble_ekan_context_parity",
               "CONTEXT_PARITY_EKAN WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16"),
    "timing": ("nano33ble_ekan_context_timing",
               "TIMING_REPLAY_EKAN WINDOW=500 CONTEXT=250 STRIDE=250 N=2 CLOCK_HZ=64000000 GPIO=D2 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16"),
}
FIRST = 750
CLOCK_HZ = 64_000_000
FIFO_MS = 640.0
N = 16


# --- host side ------------------------------------------------------------------------------
def edge_host(features):
    """Run the host edge probe on float32 feature rows: (P, deltas[16], top3[3]) incremental."""
    features = np.asarray(features, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        with src.open("wb") as f:
            np.array([len(features), N], dtype="<u4").tofile(f)
            features.astype("<f4").tofile(f)
        subprocess.run([str(PROBE), str(src), str(dst)], check=True, capture_output=True)
        out = np.fromfile(dst, dtype="<f4").reshape(len(features), 2 * (1 + N + 3))
    return out[:, 0].astype(float), out[:, 1:1 + N].astype(float), out[:, 1 + N:1 + N + 3].astype(int)


def load_prepared():
    with np.load(PREPARED / "expected_host.npz") as e:
        expected = {k: e[k] for k in e.files}
    with np.load(PREPARED / "fixtures.npz") as f:
        fixtures = {k: f[k] for k in f.files}
    with np.load(PREPARED / "cases.npz") as c:
        cases = {k: c[k] for k in c.files}
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    names = manifest["fixtures"]["names"]
    fixture_cpp, locked_cpp = expected["fixture_cpp"], expected["locked_cpp"]
    streams = []
    for s, name in enumerate(names):
        streams.append({"kind": "fixture", "name": name, "ir": fixtures[f"{name}_ir"], "red": fixtures[f"{name}_red"],
                        "cpp": fixture_cpp[fixture_cpp[:, 0] == s]})
    for k in range(len(cases["counts"])):
        n = int(cases["counts"][k])
        streams.append({"kind": "locked", "case": k, **manifest["locked"]["keys"][k],
                        "ir": cases["raw"][k, 0, :n], "red": cases["raw"][k, 1, :n], "cpp": locked_cpp[locked_cpp[:, 0] == k]})
    return streams, manifest


def verify_image(env, receipt):
    archive = ARCHIVE_ROOT / env
    image, recorded, _ = archived_image(env, archive)
    if sha256(ROOT / image) != recorded:
        raise SystemExit("Archived image does not match its SHA256SUMS")
    r = json.loads(Path(receipt).read_text())
    if r.get("env") != env or r.get("image_sha256") != recorded or not r.get("uploaded_and_verified"):
        raise SystemExit("Upload receipt does not name this image")
    return recorded


def configure_ref0():
    """Switch every path, env and identity to the label-0-reference job."""
    global PROBE, PREREG, PIPELINE_PREREG, OUT_ROOT, ARCHIVE_ROOT, REF0, PIPELINE_ENV, PIPELINE_IDENTITY
    PROBE = ROOT / "outputs/edge-kan-label0/host-parity/edge_kan_probe_ekan16_ref0.exe"
    PREREG = PIPELINE_PREREG = ROOT / "docs/edge-kan-label0-board-preregistration.md"
    OUT_ROOT = ROOT / "outputs/edge-kan-label0"
    ARCHIVE_ROOT = ROOT / "outputs/firmware-ekan-ref0"
    PROFILES["parity"] = ("nano33ble_ekan_context_parity_ref0", PROFILES["parity"][1] + " XAI_REF=LABEL0")
    PIPELINE_ENV = "nano33ble_ekan_pipeline_timing_ref0"
    PIPELINE_IDENTITY = PIPELINE_IDENTITY + " XAI_REF=LABEL0"
    REF0 = True


FPU = False


def configure_fpu():
    """Switch archives, envs and output to the hardware-FPU builds (docs/module8-review-fpu-plan.md).

    Same firmware identities, prepared set, host probe and gates as the soft-float jobs; only the image differs.
    """
    global PREREG, PIPELINE_PREREG, OUT_ROOT, ARCHIVE_ROOT, FPU, PIPELINE_ENV
    PREREG = PIPELINE_PREREG = ROOT / "docs/module8-review-fpu-plan.md"
    OUT_ROOT = ROOT / "outputs/edge-kan-fpu"
    ARCHIVE_ROOT = ROOT / "outputs/firmware-ekan-fpu"
    for job in ("parity", "timing"):
        PROFILES[job] = (PROFILES[job][0] + "_fpu", PROFILES[job][1])
    PIPELINE_ENV = PIPELINE_ENV + "_fpu"
    FPU = True


def python_label0(features):
    """Python float64 edge P and label-0 deltas (EdgeModel, perturb_against) on FP32 feature rows."""
    import re
    import edge_kan_paper_analyses as eka
    import analyze_physio_concept as apc
    text = (ROOT / "firmware/Arduino Nano 33/include/edge_kan_xai_reference.h").read_text(encoding="utf-8")
    body = text[text.index("EKAN_XAI_REF_Z[16] = {"):]
    body = body[:body.index("};")]
    ref = np.array([float(v) for v in re.findall(r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)f", body)], dtype=np.float32)
    if len(ref) != N:
        raise SystemExit("could not parse the 16 label-0 reference values")
    model = eka.EdgeModel(eka.P["edge_header"])
    z = model.standardize(np.asarray(features, dtype=np.float32))
    p, d = apc.perturb_against(model, z, ref.astype(np.float64))
    return np.asarray(p, float), np.asarray(d, float)


def python_check(board_p, board_d, py_p, py_d):
    dp = abs(board_p - py_p)
    dd = float(np.max(np.abs(np.asarray(board_d, float) - py_d)))
    return {"pass": bool(dp <= 1e-4 and dd <= 1e-5), "probability_error": float(dp), "max_delta_error": dd}


# --- serial protocol -----------------------------------------------------------------------




def ask(connection, command, timeout=20):
    return receive_after(connection, command, timeout)




def blocks(stream):
    for b in range(len(stream["ir"]) // 250):
        yield (b + 1) * 250, block_command(stream["ir"][b * 250:(b + 1) * 250], stream["red"][b * 250:(b + 1) * 250])


def reset(connection):
    if ask(connection, "R\n") != "RESET 0":
        raise RuntimeError("Reset not acknowledged")


# --- parity ----------------------------------------------------------------------------------
def parse_parity(words):
    if len(words) != 32 + 1 + N + 3:
        raise ValueError(f"Expected 52 RESULT words; got {len(words)}")
    out = {"meta": [int(w) for w in words[:5]], "tk_probability": decode(words[5]),
           "features": [decode(w) for w in words[6:22]], "states": [int(w) for w in words[22:25]],
           "tk_top": [[int(words[i]), decode(words[i + 1])] for i in (25, 27, 29)], "valid": int(words[31]),
           "edge_probability": decode(words[32]), "edge_deltas": [decode(w) for w in words[33:33 + N]],
           "edge_top": [int(w) for w in words[33 + N:]]}
    values = [out["tk_probability"], out["edge_probability"], *out["features"], *out["edge_deltas"]]
    if not all(map(math.isfinite, values)):
        raise ValueError("Non-finite result")
    return out


def judge(board, cpp_row, edge_p, edge_d, py=None):
    feats = criteria.feature_check(cpp_row[host.F0:host.F1], board["features"])
    tk_error = abs(board["tk_probability"] - float(cpp_row[6]))
    exact = (board["meta"] == [int(v) for v in cpp_row[1:6]]
             and board["states"] == [int(v) for v in cpp_row[host.STATE]]
             and board["valid"] == int(cpp_row[host.VALID]))
    gate1 = bool(feats["pass"] and tk_error <= 1e-4 and exact)
    edge = criteria.check_window(cpp_row[host.F0:host.F1], board["features"], edge_p, board["edge_probability"],
                                 edge_d, board["edge_deltas"])
    own_top = np.argsort(-np.abs(np.float32(board["edge_deltas"])), kind="stable")[:3].tolist()
    gate3 = board["edge_top"] == own_top
    gate4 = python_check(board["edge_probability"], board["edge_deltas"], *py) if py is not None else None
    return {"pass": bool(gate1 and edge["pass"] and gate3 and (gate4 is None or gate4["pass"])), "gate1_pipeline": gate1,
            "gate2_edge": bool(edge["pass"]), "gate3_payload": bool(gate3), "gate4_python_label0": gate4,
            "features_pass": bool(feats["pass"]), "tk_probability_error": tk_error,
            "exact_fields": bool(exact), "edge_probability_error": edge["probability_absolute_error"],
            "edge_max_delta_error": edge["attributions"]["max_absolute_error"],
            "edge_candidate_agrees": edge["candidate_agrees"], "edge_rank_violations": edge["attributions"]["rank_violations"],
            "edge_top3_set_changed": edge["attributions"]["top3_set_changed"],
            "edge_top3_order_changed": edge["attributions"]["top3_order_changed"],
            "host_edge_probability": float(edge_p), "board": board}


def run_parity(args, streams):
    env, identity = PROFILES["parity"]
    image_sha = verify_image(env, args.upload_receipt)
    selected = [s for s in streams if s["kind"] == "fixture"] + \
               [s for s in streams if s["kind"] == "locked"][:None if args.locked == "all" else int(args.locked)]
    host_rows = np.concatenate([s["cpp"] for s in selected])
    hp, hd, _ = edge_host(host_rows[:, host.F0:host.F1])
    pyp, pyd = python_label0(host_rows[:, host.F0:host.F1]) if REF0 else (None, None)
    output = args.output
    output.mkdir(parents=True)
    report = {"status": "running", "port": args.port, "protocol": identity, "image_sha256": image_sha,
              "upload_receipt_sha256": sha256(args.upload_receipt), "preregistration_sha256": sha256(PREREG),
              "prepared_manifest_sha256": sha256(PREPARED / "manifest.json"), "probe_sha256": sha256(PROBE),
              "locked_selected": args.locked, "started_utc": datetime.now(timezone.utc).isoformat(), "streams": []}
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    import serial
    row = 0
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if ask(connection, "H\n") != identity:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            reset(connection)
            if ask(connection, "B 00000000\n") != "ERR invalid_command":
                raise RuntimeError("Malformed block was not rejected")
            for n_stream, stream in enumerate(selected):
                reset(connection)
                verdicts, k = [], 0
                for total, command in blocks(stream):
                    line = ask(connection, command)
                    if total < FIRST:
                        if line != f"BUFFER {total}":
                            raise RuntimeError(f"Unexpected warmup response: {line[:60]}")
                        continue
                    if not line.startswith("RESULT "):
                        raise RuntimeError(f"Unexpected board response: {line[:60]}")
                    verdicts.append(judge(parse_parity(line.split()[1:]), stream["cpp"][k], hp[row], hd[row],
                                          (pyp[row], pyd[row]) if REF0 else None))
                    k += 1
                    row += 1
                if k != len(stream["cpp"]):
                    raise RuntimeError("Decision count differs from host expectation")
                report["streams"].append({key: stream[key] for key in stream if key not in ("ir", "red", "cpp")} | {"decisions": verdicts})
                if n_stream % 25 == 0:
                    save()
                    print(f"{n_stream + 1}/{len(selected)} streams, {sum(not v['pass'] for s in report['streams'] for v in s['decisions'])} failed", flush=True)
        rows = [v for s in report["streams"] for v in s["decisions"]]
        bf = np.array([r["board"]["features"] for r in rows], dtype=np.float32)
        bp, bd, bt = edge_host(bf)
        arith_p = np.abs(bp - [r["board"]["edge_probability"] for r in rows])
        arith_d = np.abs(bd - [r["board"]["edge_deltas"] for r in rows])
        report.update(status="completed", summary={
            "decisions": len(rows), "failed": sum(not r["pass"] for r in rows),
            "gate1_pipeline_failed": sum(not r["gate1_pipeline"] for r in rows),
            "gate2_edge_failed": sum(not r["gate2_edge"] for r in rows),
            "gate3_payload_failed": sum(not r["gate3_payload"] for r in rows),
            **({"gate4_python_label0_failed": sum(not r["gate4_python_label0"]["pass"] for r in rows),
                "max_edge_probability_error_vs_python_label0": max(r["gate4_python_label0"]["probability_error"] for r in rows),
                "max_edge_delta_error_vs_python_label0": max(r["gate4_python_label0"]["max_delta_error"] for r in rows)}
               if REF0 else {}),
            "edge_decision_disagreements_at_0_50": sum(not r["edge_candidate_agrees"] for r in rows),
            "max_edge_probability_error_vs_host": max(r["edge_probability_error"] for r in rows),
            "max_edge_delta_error_vs_host": max(r["edge_max_delta_error"] for r in rows),
            "edge_top3_set_changed_vs_host": sum(r["edge_top3_set_changed"] for r in rows),
            "edge_top3_order_changed_vs_host": sum(r["edge_top3_order_changed"] for r in rows),
            "max_tinykan_probability_error_vs_host": max(r["tk_probability_error"] for r in rows),
            "arithmetic_only_on_board_features": {
                "max_probability_error": float(arith_p.max()), "max_delta_error": float(arith_d.max()),
                "top3_disagreements": int(sum(list(t) != r["board"]["edge_top"] for t, r in zip(bt, rows)))},
            "edge_candidates_board": sum(r["board"]["edge_probability"] >= 0.5 for r in rows),
            "tinykan_candidates_board": sum(r["board"]["tk_probability"] >= 0.5 for r in rows)})
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if not report["summary"]["failed"] else 1


# --- timing ----------------------------------------------------------------------------------
def run_timing(args, streams):
    env, identity = PROFILES["timing"]
    image_sha = verify_image(env, args.upload_receipt)
    fixtures = [s for s in streams if s["kind"] == "fixture"]
    host_rows = np.concatenate([s["cpp"] for s in fixtures])
    hp, _, _ = edge_host(host_rows[:, host.F0:host.F1])
    output = args.output
    output.mkdir(parents=True)
    report = {"status": "running", "port": args.port, "protocol": identity, "image_sha256": image_sha,
              "upload_receipt_sha256": sha256(args.upload_receipt), "preregistration_sha256": sha256(PREREG),
              "repetitions": args.repetitions, "kernel_repetitions_on_board": 5, "clock_hz": CLOCK_HZ,
              "started_utc": datetime.now(timezone.utc).isoformat(), "calibration": [], "memory": [], "blocks": [], "decisions": []}
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if ask(connection, "H\n") != identity:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            for _ in range(8):
                w = ask(connection, "C\n").split()
                report["calibration"].append(int(w[1]))
            report["memory"].append(ask(connection, "M\n"))
            for rep in range(args.repetitions):
                row = 0
                for stream in fixtures:
                    reset(connection)
                    k = 0
                    for total, command in blocks(stream):
                        w = ask(connection, command).split()
                        if w[0] != "TIMED":
                            raise RuntimeError(f"Unexpected response: {' '.join(w)[:60]}")
                        block = {"rep": rep, "stream": stream["name"], "total": total, "cycles": int(w[1]), "ms": int(w[2]),
                                 "decision": w[3] == "RESULT"}
                        report["blocks"].append(block)
                        if w[3] != "RESULT":
                            continue
                        r = w[4:]
                        if len(r) != 13:
                            raise ValueError(f"Expected 13 timing RESULT words; got {len(r)}")
                        cpp = stream["cpp"][k]
                        tk_p, ek_p = decode(r[2]), decode(r[3])
                        lo_hi = [int(x) for x in r[4:12]]
                        d = {"rep": rep, "stream": stream["name"], "window": int(r[0]), "block_cycles": block["cycles"],
                             "tk_probability": tk_p, "edge_probability": ek_p,
                             "tk_forward": lo_hi[0:2], "tk_xai": lo_hi[2:4], "edge_forward": lo_hi[4:6], "edge_xai": lo_hi[6:8],
                             "tk_copy_bit_equal": r[12] == "1",
                             "tk_ok": abs(tk_p - float(cpp[6])) <= 1e-4, "edge_ok": abs(ek_p - hp[row]) <= 1e-4}
                        report["decisions"].append(d)
                        k += 1
                        row += 1
                save()
                print(f"repetition {rep + 1}/{args.repetitions}", flush=True)
            report["memory"].append(ask(connection, "M\n"))
        us = lambda c: c / CLOCK_HZ * 1e6  # noqa: E731
        ds = report["decisions"]
        stat = lambda xs: {"median_us": float(np.median([us(x) for x in xs])), "max_us": float(max(us(x) for x in xs)),  # noqa: E731
                           "min_us": float(min(us(x) for x in xs))}
        derived = [d["block_cycles"] - d["tk_xai"][0] + d["edge_xai"][0] for d in ds]
        report.update(status="completed", summary={
            "decisions_timed": len(ds), "checks_failed": sum(not (d["tk_ok"] and d["edge_ok"] and d["tk_copy_bit_equal"]) for d in ds),
            "calibration_cycles_max": max(report["calibration"]),
            "decision_block_v2_pipeline": stat([d["block_cycles"] for d in ds]),
            "nondecision_block": stat([b["cycles"] for b in report["blocks"] if not b["decision"]]),
            "tinykan_forward_min_cycles": stat([d["tk_forward"][0] for d in ds]),
            "tinykan_xai_17pass_min_cycles": stat([d["tk_xai"][0] for d in ds]),
            "edge_forward_min_cycles": stat([d["edge_forward"][0] for d in ds]),
            "edge_xai_incremental_min_cycles": stat([d["edge_xai"][0] for d in ds]),
            "kernel_max_over_reps_us": {k: float(max(us(d[k][1]) for d in ds)) for k in ("tk_forward", "tk_xai", "edge_forward", "edge_xai")},
            "derived_edge_decision_block": {**stat(derived), "note": "block - TinyKAN XAI min + edge XAI min; derived, not measured"},
            "fifo_budget_ms": FIFO_MS})
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if not report["summary"]["checks_failed"] else 1


# --- pipeline: the integrated image (oxygen_risk_pipeline.h with -DOXY_MODEL_EKAN) ------------------
PIPELINE_ENV = "nano33ble_ekan_pipeline_timing"
PIPELINE_IDENTITY = ("TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 "
                     "GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16")


def edge_states(probabilities):
    """Candidate/Confirmed/streak of one stream from reset, as the pipeline's debounce (N = 2, 0.50)."""
    streak, out = 0, []
    for p in probabilities:
        candidate = p >= 0.5
        streak = min(streak + 1, 255) if candidate else 0
        out.append([int(candidate), int(streak >= 2), streak])
    return out


def judge_integrated(words, total, cpp_row, edge_p, edge_d, states, py=None):
    """One RESULT of the integrated image against the C++ host features and the host edge engine."""
    meta = [int(w) for w in words[:5]]
    p = decode(words[5])
    feats = [decode(w) for w in words[6:22]]
    got_states = [int(w) for w in words[22:25]]
    top = [(int(words[i]), decode(words[i + 1])) for i in (25, 27, 29)]
    valid = int(words[31])
    feature = criteria.feature_check(cpp_row[host.F0:host.F1], feats)
    expected_top = np.argsort(-np.abs(np.float32(edge_d)), kind="stable")[:3].tolist()
    mags = np.abs(edge_d)
    order_ok = all(i == j or abs(mags[i] - mags[j]) <= criteria.CONTRACT["ranking_gap"]
                   for (i, _), j in zip(top, expected_top))
    delta_ok = all(abs(d - edge_d[i]) <= criteria.CONTRACT["attribution_absolute"] for i, d in top)
    checks = {
        "meta": meta == [(total - FIRST) // 250, total, total - FIRST, total - 250, 250] and meta == [int(v) for v in cpp_row[1:6]],
        "features": bool(feature["pass"]),
        "probability": abs(p - edge_p) <= criteria.CONTRACT["probability_absolute"] and (p >= .5) == (edge_p >= .5),
        "states": got_states == states,
        "top3_order": bool(order_ok), "top3_deltas": bool(delta_ok),
        "valid": valid == int(cpp_row[host.VALID]),
    }
    if py is not None:  # --ref0: P and the three sent deltas against the Python float64 label-0 values
        py_p, py_d = py
        checks["python_label0"] = bool(abs(p - py_p) <= 1e-4 and all(abs(d - py_d[i]) <= 1e-5 for i, d in top))
    return {"pass": all(checks.values()), "checks": checks, "probability": p, "host_edge_probability": float(edge_p),
            "probability_error": abs(p - edge_p), "top3": [i for i, _ in top], "host_top3": expected_top,
            "top3_identical": [i for i, _ in top] == expected_top, "states": got_states}


SOFT_PIPELINE = ROOT / "outputs/edge-kan-v1/board-pipeline-20261112/report.json"


def compare_soft_float(rows):
    """FPU board decisions against the soft-float board run (same prepared set, same stream order)."""
    soft = json.loads(SOFT_PIPELINE.read_text())
    srows = [v for s in soft["functional"] for v in s["decisions"]]
    if len(srows) != len(rows):
        return {"comparable": False, "note": f"{len(rows)} FPU vs {len(srows)} soft-float decisions"}
    dp = [abs(a["probability"] - b["probability"]) for a, b in zip(rows, srows)]
    return {"comparable": True, "decisions": len(rows), "source": SOFT_PIPELINE.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(SOFT_PIPELINE),
            "probability_bit_identical": sum(a["probability"] == b["probability"] for a, b in zip(rows, srows)),
            "max_probability_difference": max(dp),
            "candidate_flag_differences_at_0_50": sum((a["probability"] >= .5) != (b["probability"] >= .5)
                                                      for a, b in zip(rows, srows)),
            "confirmed_state_differences": sum(a["states"][1] != b["states"][1] for a, b in zip(rows, srows)),
            "top3_differences": sum(a["top3"] != b["top3"] for a, b in zip(rows, srows))}


def run_pipeline(args, streams):
    image_sha = verify_image(PIPELINE_ENV, args.upload_receipt)
    selected = [s for s in streams if s["kind"] == "fixture"] + \
               [s for s in streams if s["kind"] == "locked"][:None if args.locked == "all" else int(args.locked)]
    fixtures = [s for s in streams if s["kind"] == "fixture"]
    expect = {}
    for s in selected:
        p, d, _ = edge_host(s["cpp"][:, host.F0:host.F1])
        py = python_label0(s["cpp"][:, host.F0:host.F1]) if REF0 else None
        expect[id(s)] = (p, d, edge_states(p), py)
    output = args.output
    output.mkdir(parents=True)
    report = {"status": "running", "port": args.port, "protocol": PIPELINE_IDENTITY, "image_env": PIPELINE_ENV,
              "image_sha256": image_sha, "upload_receipt_sha256": sha256(args.upload_receipt),
              "preregistration_sha256": sha256(PIPELINE_PREREG),
              "probe_sha256": sha256(PROBE), "locked_selected": args.locked, "repetitions": args.repetitions,
              "started_utc": datetime.now(timezone.utc).isoformat(), "calibration": [], "memory": [],
              "functional": [], "timing_blocks": []}
    save = lambda: (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")  # noqa: E731
    import serial

    def replay(connection, stream, timing_rep=None):
        reset(connection)
        p, d, st, py = expect[id(stream)]
        verdicts, k = [], 0
        for total, command in blocks(stream):
            w = ask(connection, command).split()
            if w[0] != "TIMED":
                raise RuntimeError(f"Unexpected response: {' '.join(w)[:60]}")
            cycles = int(w[1])
            if total < FIRST:
                if w[3:] != ["BUFFER", str(total)]:
                    raise RuntimeError(f"Unexpected warmup response: {' '.join(w)[:60]}")
                window = None
            else:
                if w[3] != "RESULT" or len(w) != 4 + 32:
                    raise RuntimeError(f"Unexpected decision response: {' '.join(w)[:60]}")
                v = judge_integrated(w[4:], total, stream["cpp"][k], p[k], d[k], st[k],
                                     (py[0][k], py[1][k]) if py is not None else None)
                verdicts.append(v)
                window = k
                k += 1
            if timing_rep is not None:
                report["timing_blocks"].append({"rep": timing_rep, "stream": stream["name"], "total": total,
                                                "cycles": cycles, "ms": int(w[2]), "window": window,
                                                "pass": None if window is None else verdicts[-1]["pass"]})
        if k != len(stream["cpp"]):
            raise RuntimeError("Decision count differs from host expectation")
        return verdicts

    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            if ask(connection, "H\n") != PIPELINE_IDENTITY:
                raise RuntimeError("Unexpected firmware identity; no sample blocks sent")
            report["memory"].append(ask(connection, "M\n"))
            for _ in range(8):
                report["calibration"].append(int(ask(connection, "C\n").split()[1]))
            for rep in range(args.repetitions):
                for stream in fixtures:
                    replay(connection, stream, timing_rep=rep)
                save()
                print(f"timing repetition {rep + 1}/{args.repetitions}", flush=True)
            for n, stream in enumerate(selected):
                verdicts = replay(connection, stream)
                report["functional"].append({key: stream[key] for key in stream if key not in ("ir", "red", "cpp")} | {"decisions": verdicts})
                if n % 50 == 0:
                    save()
                    print(f"functional {n + 1}/{len(selected)} streams, "
                          f"{sum(not v['pass'] for s in report['functional'] for v in s['decisions'])} failed", flush=True)
            report["memory"].append(ask(connection, "M\n"))
        rows = [v for s in report["functional"] for v in s["decisions"]]
        tb = report["timing_blocks"]
        us = lambda c: c / CLOCK_HZ * 1e6  # noqa: E731

        def stat(values):
            v = np.array([us(x) for x in values])
            return {"count": len(v), "min_us": float(v.min()), "median_us": float(np.median(v)),
                    "p95_us": float(np.percentile(v, 95, method="linear")), "max_us": float(v.max())}
        worst_ms = max(us(b["cycles"]) for b in tb) / 1000
        report.update(status="completed", summary={
            "functional_decisions": len(rows), "functional_failed": sum(not r["pass"] for r in rows),
            "failed_by_check": {c: sum(not r["checks"][c] for r in rows) for c in rows[0]["checks"]},
            "max_probability_error": max(r["probability_error"] for r in rows),
            "top3_not_identical_to_host": sum(not r["top3_identical"] for r in rows),
            "candidates_board": sum(r["probability"] >= .5 for r in rows),
            "confirmed_board": sum(r["states"][1] for r in rows),
            "timed_decisions_failed": sum(b["pass"] is False for b in tb),
            "calibration_cycles_max": max(report["calibration"]),
            "latency": {"warmup_blocks": stat([b["cycles"] for b in tb if b["window"] is None]),
                        "first_decision_after_reset": stat([b["cycles"] for b in tb if b["window"] == 0]),
                        "subsequent_decisions": stat([b["cycles"] for b in tb if b["window"] not in (None, 0)])},
            "max_block_ms": worst_ms, "fifo_headroom_ratio": FIFO_MS / worst_ms,
            "v2_reference": {"subsequent_decision_median_ms": 444.3, "p95_ms": 445.9,
                             "source": "IEEE draft 2026-11-12a; outputs/timing-v2-board-20261022-r2"},
            **({"training_mean_edge_reference": {"subsequent_decision_median_ms": 433.9, "p95_ms": 435.4,
                                                 "source": "outputs/edge-kan-v1/board-pipeline-20261112"}} if REF0 else {}),
            **({"soft_float_reference": {"subsequent_decision_median_ms": 433.9, "p95_ms": 435.4,
                                         "source": "outputs/edge-kan-v1/board-pipeline-20261112"},
                "vs_soft_float_board": compare_soft_float(rows)} if FPU else {})})
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if not (report["summary"]["functional_failed"] or report["summary"]["timed_decisions_failed"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("parity", "timing", "pipeline", "host-check"))
    parser.add_argument("--port")
    parser.add_argument("--upload-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--locked", default="all")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--ref0", action="store_true", help="label-0-reference images (docs/edge-kan-label0-board-preregistration.md)")
    parser.add_argument("--fpu", action="store_true", help="hardware-FPU images (docs/module8-review-fpu-plan.md)")
    args = parser.parse_args()
    if args.ref0 and args.fpu:
        parser.error("--ref0 and --fpu are separate jobs")
    if args.ref0:
        configure_ref0()
    if args.fpu:
        configure_fpu()
    streams, _ = load_prepared()
    if args.job == "host-check":
        rows = np.concatenate([s["cpp"] for s in streams])
        p, d, t = edge_host(rows[:, host.F0:host.F1])
        print(json.dumps({"streams": len(streams), "decisions": len(rows), "edge_candidates_host": int((p >= .5).sum()),
                          "tinykan_candidates_host": int((rows[:, 6] >= .5).sum()), "max_abs_delta": float(np.abs(d).max())}, indent=2))
        return 0
    if not (args.port and args.upload_receipt and args.output):
        raise SystemExit("needs --port, --upload-receipt and a fresh --output")
    args.output = args.output.resolve()
    if args.output.exists() or not args.output.is_relative_to(OUT_ROOT.resolve()):
        raise SystemExit(f"Fresh --output under {OUT_ROOT.relative_to(ROOT).as_posix()}/ required")
    return {"parity": run_parity, "timing": run_timing, "pipeline": run_pipeline}[args.job](args, streams)


if __name__ == "__main__":
    sys.exit(main())

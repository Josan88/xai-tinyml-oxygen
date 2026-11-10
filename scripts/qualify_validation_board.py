"""Replay the already evaluated, locked numerical set on the unchanged Nano image.

Host-only preflight is the default. No firmware upload, repair, subset selection,
new independence claim, or numerical budget override is supported.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from audit_feature_errors import sha256
from parity_criteria import ROOT, CONTRACT_SHA256, check_window
from qualify_numerical_parity import local_output
from qualify_recorded_board import identify, replay_case, evaluate_case, expected_final


BASE = ROOT / "outputs/parity/qualification"
CANDIDATE = BASE / "frozen-candidate-v1.json"
CASES = BASE / "validation-v1"
NATIVE = BASE / "locked-validation"
LOCK = ROOT / "outputs/parity/validation_lock/numerical-validation-v1.json"
IMAGE = BASE / "all14-diagnostic-image/firmware.bin"
IMAGE_SHA256 = "30d390c7ad19cc335f8453124e14fbeef7986bd9bb258c5a60dde27fd76b9193"
CANDIDATE_SHA256 = "1cfd4ed28b5aad114d1064a1a7323673b8725e434700e7b6303b5009ace74d0b"


def verify_metadata(candidate, manifest, report, lock, first_use, candidate_digest):
    """Reject changed provenance or incomplete/foreign validation evidence."""
    if candidate_digest != CANDIDATE_SHA256:
        raise ValueError("This runner requires the previously frozen candidate v1")
    if any(x["contract_sha256"] != CONTRACT_SHA256 for x in (candidate, manifest, report, lock)):
        raise ValueError("Frozen numerical contract changed")
    if manifest["role"] != "validation" or report["role"] != "validation" or report["subset"]:
        raise ValueError("Complete locked validation evidence is required")
    if first_use["candidate_sha256"] != candidate_digest:
        raise ValueError("Validation native first use belongs to another candidate")
    if any(x["validation_lock_sha256"] != candidate["validation_lock_sha256"]
           for x in (manifest, report, first_use)):
        raise ValueError("Validation lock identity changed")
    if manifest["keys"] != lock["windows"]:
        raise ValueError("Validation keys/order differ from the write-once lock")
    count = len(manifest["keys"])
    if report["summary"]["windows"] != count or report["complete_case_count"] != count:
        raise ValueError("Native validation evidence is incomplete")
    if report["summary"]["failed_windows"] or report["parity_complete"]:
        raise ValueError("Unexpected native qualification state")
    if report["cases_sha256"] != manifest["cases_sha256"]:
        raise ValueError("Native evidence belongs to different cases")
    if report["source_sha256"] != candidate["source_sha256"] or report["probe_sha256"] != candidate["probe_sha256"]:
        raise ValueError("Native evidence is not from the frozen executable/source")


def preflight():
    read = lambda p: json.loads(p.read_text())
    candidate, manifest, report, lock = map(read, (CANDIDATE, CASES / "manifest.json", NATIVE / "report.json", LOCK))
    first_use = read(LOCK.with_suffix(".use.json"))
    verify_metadata(candidate, manifest, report, lock, first_use, sha256(CANDIDATE))
    expected = {LOCK: candidate["validation_lock_sha256"], CASES / "cases.npz": manifest["cases_sha256"],
                CASES / "manifest.json": report["cases_manifest_sha256"], IMAGE: IMAGE_SHA256,
                Path(candidate["development_report"]): candidate["development_report_sha256"],
                ROOT / "scripts/qualify_numerical_parity.py": candidate["runner_sha256"],
                ROOT / "scripts/parity_criteria.py": candidate["criteria_sha256"]}
    for path, digest in list(expected.items()) + [(Path(p), h) for p, h in candidate["source_sha256"].items()]:
        if sha256(path) != digest:
            raise ValueError(f"Frozen artifact changed: {path}")
    with np.load(CASES / "cases.npz", allow_pickle=False) as cache:
        data = {k: cache[k] for k in cache.files}
    with np.load(NATIVE / "native.npz", allow_pickle=False) as cache:
        native = {k: cache[k] for k in cache.files}
    count = len(manifest["keys"])
    if native["rows"].shape != (count, 546) or not np.array_equal(native["case_indices"], np.arange(count)):
        raise ValueError("Native rows are incomplete or reordered")
    if any(len(v) != count or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Malformed reference cache")
    for i, row in enumerate(native["rows"]):
        expected_row = expected_final(row, data["counts"][i])
        if not check_window(data["features"][i], expected_row["values"][1:], data["probabilities"][i],
                            expected_row["values"][0], data["deltas"][i], expected_row["deltas"])["pass"]:
            raise ValueError(f"Native cache does not reproduce passing locked case {i}")
    return manifest, data, native


def summarize(rows):
    return {"windows": len(rows), "failed_windows": sum(not r["pass"] for r in rows),
            "candidate_disagreements": sum(not r["board_vs_python"]["candidate_agrees"] for r in rows),
            "timing_state_failures": sum(not r["timing_state_pass"] for r in rows),
            "original_absolute_1e_4_failed_windows": sum(not r["original_absolute_1e_4_pass"] for r in rows),
            "max_probability_absolute_error": max((r["probability_absolute_error"] for r in rows), default=None),
            "max_attribution_absolute_error": max((r["max_attribution_absolute_error"] for r in rows), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port")
    parser.add_argument("--upload-receipt", type=Path)
    args = parser.parse_args()
    out = local_output(args.output)
    if out.exists():
        raise ValueError("Output already exists; preserve previous evidence")
    if args.port and not args.upload_receipt:
        parser.error("Board replay requires the current verified upload receipt")
    manifest, data, native = preflight()
    report = {"status": "prepared_not_run", "role": "validation", "parity_complete": False,
              "board_run_completed": False, "created_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_sha256": sha256(CANDIDATE), "image_sha256": sha256(IMAGE),
              "validation_lock_sha256": sha256(LOCK), "contract_sha256": CONTRACT_SHA256,
              "cases_sha256": sha256(CASES / "cases.npz"), "native_npz_sha256": sha256(NATIVE / "native.npz"),
              "native_report_sha256": sha256(NATIVE / "report.json"), "runner_sha256": sha256(__file__),
              "scope": "All locked sparse contexts on unchanged candidate; already evaluated native evidence, not a fresh holdout or continuous recorded stream.",
              "expected_windows": len(manifest["keys"]), "windows": [], "summary": None}
    if args.port:
        upload = json.loads(args.upload_receipt.read_text())
        if (not upload["uploaded_and_verified"] or upload["image_sha256"] != IMAGE_SHA256
                or upload["candidate_sha256"] != CANDIDATE_SHA256):
            raise ValueError("Upload receipt does not establish the exact archived image")
        report["upload_receipt_sha256"] = sha256(args.upload_receipt)
        report["port"] = args.port
    out.mkdir(parents=True)
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    print(f"Preflight passed: {len(manifest['keys'])} locked cases", flush=True)
    if not args.port:
        return 0
    import serial
    rows = []
    try:
        with serial.Serial(args.port, 115200, timeout=1, write_timeout=5) as connection:
            connection.reset_input_buffer()
            identify(connection)
            report["status"] = "running"
            report["board_replay_started_utc"] = datetime.now(timezone.utc).isoformat()
            save()
            with (out / "windows.jsonl").open("x") as evidence:
                for i, key in enumerate(manifest["keys"]):
                    board, payload = replay_case(connection, data["raw"][i], data["counts"][i])
                    row = evaluate_case(data["features"][i], data["probabilities"][i], data["deltas"][i],
                                        native["rows"][i], data["counts"][i], board, payload)
                    row.update(case_index=i, **key)
                    rows.append(row)
                    evidence.write(json.dumps(row, allow_nan=False) + "\n")
                    evidence.flush()
                    if (i + 1) % 50 == 0 or i + 1 == len(manifest["keys"]):
                        print(f"Locked board cases {i + 1}/{len(manifest['keys'])}; failed {sum(not r['pass'] for r in rows)}", flush=True)
        report.update(status="completed" if all(r["pass"] for r in rows) else "failed", board_run_completed=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(windows=rows, summary=summarize(rows), completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())

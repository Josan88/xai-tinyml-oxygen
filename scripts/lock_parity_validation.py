"""Lock unseen numerical validation windows before arithmetic repair; never overwrite."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from parity_criteria import ROOT, CONTRACT, CONTRACT_SHA256
from audit_feature_errors import sha256

SEED = "native-fp32-local-scale-v1-validation-2026-09-10"


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def span(start):
    return max(0, start - 250), start + 750


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Keep restricted metadata under outputs/")
    if args.output.exists():
        parser.error("Validation lock already exists; refusing to replace it")
    csv_path = ROOT / "outputs/openox_canonical_window_features.csv"
    eligible_path = ROOT / "outputs/openox_ppg_eligible_encounters.csv"
    with csv_path.open(newline="") as stream:
        rows = [{"recording_id": r["recording_id"], "start_sample": round(float(r["start_time_seconds"]) * 50)}
                for r in csv.DictReader(stream)]
    with eligible_path.open(newline="") as stream:
        eligible = {r["encounter_id"]: r for r in csv.DictReader(stream)}
    excluded = {}
    with (args.audit / "windows.csv").open(newline="") as stream:
        development = list(csv.DictReader(stream))
    for r in development:
        excluded.setdefault(r["recording_id"], []).append(span(int(r["start_sample"])))
    # These canonical feature vectors were also used by the earlier inference audit.
    for r in rows[::251]:
        excluded.setdefault(r["recording_id"], []).append(span(r["start_sample"]))
    manifest_path = ROOT / "outputs/openox_sample_window_manifest.csv"
    historical_segments = 0
    with manifest_path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            if r["ppg_window_fully_in_bounds"].lower() != "true" or not Path(r["ppg_dat_path"]).is_file():
                continue
            rate = float(r["ppg_sampling_rate_hz"])
            first, last = int(r["ppg_window_start_index"]), int(r["ppg_window_end_index"])
            if round((last - first) * 50 / rate) < 500:
                continue
            excluded.setdefault(r["encounter_id"], []).append((math.floor(first * 50 / rate), math.ceil(last * 50 / rate)))
            historical_segments += 1
            if historical_segments == 8:
                break
    if historical_segments != 8:
        raise ValueError("Cannot reconstruct the eight historical recorded-segment exclusions")
    selected, per_recording = [], []
    audit_report = json.loads((args.audit / "report.json").read_text())
    for recording_id in dict.fromkeys(r["recording_id"] for r in rows):
        source = next(p for p in audit_report["provenance"] if p["recording_id"] == recording_id)
        header = Path(eligible[recording_id]["ppg_hea_path"])
        # Use verified source dimensions, not feature values or model outcomes, for eligibility.
        if sha256(header.with_suffix(".dat")) != source["data_sha256"] or sha256(header) != source["header_sha256"]:
            raise ValueError("Recorded source changed since development audit")
        candidates = [r for r in rows if r["recording_id"] == recording_id
                      and r["start_sample"] % 250 == 0
                      and r["start_sample"] + 750 <= source["resampled_samples"]
                      and not any(overlaps(span(r["start_sample"]), used) for used in excluded.get(recording_id, []))]
        candidates.sort(key=lambda r: hashlib.sha256(f"{SEED}:{recording_id}:{r['start_sample']}".encode()).hexdigest())
        chosen = []
        for r in candidates:
            if not any(overlaps(span(r["start_sample"]), span(c["start_sample"])) for c in chosen):
                chosen.append(r)
            if len(chosen) == 16:
                break
        selected.extend(sorted(chosen, key=lambda r: r["start_sample"]))
        per_recording.append({"recording_id": recording_id, "selected": len(chosen), "eligible": len(candidates)})
    if not selected:
        raise ValueError("No unseen nonoverlapping validation windows available")
    source_paths = [ROOT / "firmware/Arduino Nano 33/include" / name for name in
                    ("feature_extraction.h", "oxygen_risk_pipeline.h", "kan_inference.h", "kan_model_weights.h")]
    report = {"id": "numerical-validation-v1", "locked_utc": datetime.now(timezone.utc).isoformat(),
              "status": "locked_before_repair", "contract": CONTRACT, "contract_sha256": CONTRACT_SHA256,
              "selection_seed": SEED, "selection_policy": "Up to 16 deterministic hash-selected retained windows per recording; no context-sample overlap with development audit, historical every-251st feature check, first eight historical recorded segments, or other selected windows",
              "scope": "Unseen windows within previously examined recordings, for numerical validation only; not independent-subject or clinical validation. Do not inspect native validation outcomes until the development repair is frozen.",
              "windows": selected, "counts": {"windows": len(selected), "recordings": sum(r["selected"] > 0 for r in per_recording)},
              "per_recording": per_recording, "development_audit": str(args.audit.resolve()),
              "development_report_sha256": sha256(args.audit / "report.json"),
              "metadata_sha256": {str(p.resolve()): sha256(p) for p in (csv_path, eligible_path, manifest_path)},
              "pre_repair_sha256": {str(p.resolve()): sha256(p) for p in source_paths},
              "lock_script_sha256": sha256(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256(args.output), "counts": report["counts"],
                      "contract_sha256": CONTRACT_SHA256}, indent=2))


if __name__ == "__main__":
    main()

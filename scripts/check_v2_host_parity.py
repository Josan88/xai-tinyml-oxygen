"""v2 (ADR-0006) host parity on real OpenOximetry v2 recordings. Host only: no board, no serial.

Inputs: every recording in outputs/openox-v2/window_features.csv. Raw red and IR are resampled
to 50 Hz exactly as scripts/export_openox_windows_v2.py does (whole-recording Fourier resample),
quantized to FP32, and streamed as paired samples, one reset per recording, through
  (a) the C++ deployed pipeline, tests/v2_stream_probe.cpp built with -DOXY_MODEL_V2, and
  (b) scripts/v2_deployed_reference.py (Python reference of the same deployed path).
Every decision is checked with scripts/parity_criteria_v2.py (frozen v1 budgets, 16 inputs),
plus metadata, ADR-0005 states over the continuous stream, top-3 payload and the
two-wavelength validity flag.

Separately, at export windows the deployed path can decide (5 s future context available),
deployed probabilities are compared with the offline export probabilities (export features
through the same v2 model): this is the whole-recording versus context-buffer filtering gap,
the same boundary v1 has, and it is reported, not folded into any tolerance.

Writes a fresh directory (refuses to overwrite) under outputs/parity/v2-host/.

--xai-reference label0 checks a probe built with -DOXY_XAI_REF_LABEL0 against the Python
reference with the label-0-median explainer reference (v2_deployed_reference.load_reference);
budgets and everything else unchanged. Its output goes to a fresh outputs/parity/v2-host-ref0-*/.
Default (mean) is the unchanged training-mean run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_openox_windows_v2 as export  # noqa: E402
import parity_criteria_v2 as criteria  # noqa: E402
import v2_deployed_reference as ref  # noqa: E402

V2_DIR = ROOT / "outputs/openox-v2"
OUT_ROOT = ROOT / "outputs/parity/v2-host"
REF0_PREFIX = "v2-host-ref0-"     # outputs/parity/v2-host-ref0-<date>/ for --xai-reference label0
# cpp row: segment + 48 fields (see tests/v2_stream_probe.cpp)
COLS = 49
F0, F1 = 7, 23            # 16 features
STATE = slice(23, 26)     # candidate confirmed streak
TOP = slice(26, 32)       # 3 x (index, delta)
VALID = 32
D0 = 33                   # 16 deltas


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def recording_streams(row):
    """Raw IR/red at 50 Hz exactly as the v2 export, plus the export's whole-recording filter."""
    hea_path = Path(str(row["ppg_hea_path"]))
    channel_count, source_rate_hz, labels = export.read_header(hea_path)
    ir_index = export.choose_ppg_channel(labels)
    red_index = export.find_channel(labels, ("red signal", "red"))
    frames = export.read_ppg_frames(hea_path.with_suffix(".dat"), channel_count)
    ir = export.resample_to_target(frames[:, ir_index], source_rate_hz)
    red = export.resample_to_target(frames[:, red_index], source_rate_hz)
    source = {"header": str(hea_path.relative_to(ROOT)), "header_sha256": sha256(hea_path),
              "data_sha256": sha256(hea_path.with_suffix(".dat")), "source_rate_hz": source_rate_hz,
              "ir_channel": labels[ir_index], "red_channel": labels[red_index]}
    return ir, red, source


def process_recording(task):
    recording_id, row, export_rows, xai_reference = task
    p = ref.load_parameters()
    ir, red, source = recording_streams(row)
    ir_full, red_full = export.bandpass(ir), export.bandpass(red)
    # Offline reconstruction of the stored export rows (same features, whole-recording filter).
    recon = []
    for start_s, stored in export_rows:
        s = int(round(start_s * 50))
        f = export.compute_window_features(ir_full[s:s + 500], 50.)
        tw = export.two_wavelength_features(red[s:s + 500], red_full[s:s + 500], ir[s:s + 500], ir_full[s:s + 500])
        values = np.array([f[n] for n in export.IR_FEATURE_NAMES] + [tw[n] for n in export.EXTRA_FEATURE_NAMES])
        recon.append(np.max(np.abs(values - stored) / np.maximum(np.abs(stored), 1e-12)))
    ir32, red32 = ir.astype(np.float32), red.astype(np.float32)
    python = ref.stream(ir32, red32, p, ref.load_reference(xai_reference))
    return {"recording_id": recording_id, "samples": len(ir32), "ir": ir32, "red": red32,
            "python": python, "reconstruction_max_relative": recon, "source": source}


def write_segments(path, recordings):
    with path.open("xb") as stream:
        for segment, rec in enumerate(recordings):
            pairs = np.empty(2 * rec["samples"], dtype="<f4")
            pairs[0::2], pairs[1::2] = rec["ir"], rec["red"]
            stream.write(np.array([segment, rec["samples"]], dtype="<u4").tobytes())
            stream.write(pairs.tobytes())


def run_cpp(probe, segments):
    result = subprocess.run([str(probe), str(segments)], capture_output=True, text=True, check=True, timeout=3600)
    rows = np.array(result.stdout.split(), dtype=np.float64)
    if rows.size % COLS:
        raise ValueError("Malformed probe output")
    rows = rows.reshape(-1, COLS)
    if not np.isfinite(rows).all():
        raise ValueError("Non-finite probe output")
    return rows, result.stderr.strip()


def run_v1_ir_check(v1_probe, rec, cpp_rows):
    """v1 deployed pipeline on the same IR stream: the 14 IR features must be bit-identical."""
    payload = " ".join(format(float(x), ".9g") for x in rec["ir"]) + "\n"
    out = subprocess.run([str(v1_probe)], input=payload, capture_output=True, text=True, check=True, timeout=600)
    v1 = np.array([[float(x) for x in line.split()] for line in out.stdout.splitlines()])
    if v1.shape != (len(cpp_rows), 20):
        return {"recording_id": rec["recording_id"], "pass": False, "reason": f"shape {v1.shape}"}
    same_meta = np.array_equal(v1[:, :5], cpp_rows[:, 1:6])
    identical = np.array_equal(v1[:, 6:20], cpp_rows[:, F0:F0 + 14])
    return {"recording_id": rec["recording_id"], "decisions": len(v1), "pass": bool(same_meta and identical),
            "max_abs_feature_difference": float(np.max(np.abs(v1[:, 6:20] - cpp_rows[:, F0:F0 + 14])))}


def compare(rec, cpp):
    py = rec["python"]
    n = len(py["probability"])
    if len(cpp) != n:
        raise ValueError(f"{rec['recording_id']}: {len(cpp)} native vs {n} Python decisions")
    out = []
    for i in range(n):
        row = cpp[i]
        check = criteria.check_window(py["features"][i], row[F0:F1], py["probability"][i], row[6],
                                      py["deltas"][i], row[D0:D0 + 16])
        states = [int(py["candidate"][i]), int(py["confirmed"][i]), int(py["streak"][i])]
        top_index = row[TOP][0::2].astype(int)
        cpp_deltas = row[D0:D0 + 16].astype(np.float32)
        payload = (np.array_equal(top_index, np.argsort(-np.abs(cpp_deltas), kind="stable")[:3])
                   and np.array_equal(row[TOP][1::2].astype(np.float32), cpp_deltas[top_index]))
        meta = np.array_equal(row[1:6], py["meta"][i])
        states_ok = np.array_equal(row[STATE], states)
        valid_ok = bool(row[VALID]) == bool(py["valid"][i])
        feature_errors = check["features"]["absolute_errors"]
        out.append({
            "recording_id": rec["recording_id"], "window_index": i, "start_sample": int(py["meta"][i][2]),
            "pass": bool(check["pass"] and meta and states_ok and payload and valid_ok),
            "numeric_pass": check["pass"], "feature_pass": check["features"]["pass"],
            "original_absolute_1e_4_pass": check["features"]["original_absolute_1e_4_pass"],
            "probability_pass": check["probability_pass"], "attribution_delta_pass": check["attributions"]["delta_pass"],
            "rank_violations": len(check["attributions"]["rank_violations"]),
            "top3_order_changed": check["attributions"]["top3_order_changed"],
            "top3_set_changed": check["attributions"]["top3_set_changed"],
            "candidate_agrees": check["candidate_agrees"], "metadata_pass": bool(meta), "states_pass": bool(states_ok),
            "top3_payload_pass": bool(payload), "two_wavelength_valid_agrees": valid_ok,
            "two_wavelength_valid": bool(py["valid"][i]),
            "python_probability": float(py["probability"][i]), "cpp_probability": float(row[6]),
            "probability_absolute_error": check["probability_absolute_error"],
            "max_attribution_absolute_error": check["attributions"]["max_absolute_error"],
            "failed_features": ";".join(check["features"]["failed_features"]),
            "original_absolute_failed_features": ";".join(check["features"]["original_absolute_1e_4_failed_features"]),
            **{f"abs_err_{name}": feature_errors[k] for k, name in enumerate(criteria.NAMES)},
            **{f"rel_err_{name}": check["features"]["relative_errors"][k] for k, name in enumerate(criteria.NAMES)},
            **{f"budget_{name}": check["features"]["budgets"][k] for k, name in enumerate(criteria.NAMES)},
        })
    return out


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {"n": int(len(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def label_metrics(y, flag):
    tp = int(np.sum(flag & (y == 1))); fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0))); fp = int(np.sum(flag & (y == 0)))
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True, help="v2_stream_probe executable (-DOXY_MODEL_V2)")
    parser.add_argument("--v1-probe", type=Path, help="v1 context_probe: check the 14 IR features are bit-identical")
    parser.add_argument("--v1-recordings", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first N recordings only (smoke run; not the reported result)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--xai-reference", choices=ref.XAI_REFERENCES, default="mean",
                        help="explainer reference the probe was built with (default: mean, the shipped images)")
    args = parser.parse_args()
    out = args.output.resolve()
    smoke_ok = args.limit and out.is_relative_to((ROOT / "build").resolve())
    if args.xai_reference == "mean":
        root_ok = out.is_relative_to(OUT_ROOT.resolve())
        root_text = "outputs/parity/v2-host/"
    else:
        root_ok = out.parent == (ROOT / "outputs/parity").resolve() and out.name.startswith(REF0_PREFIX)
        root_text = f"outputs/parity/{REF0_PREFIX}*/"
    if not (root_ok or smoke_ok):
        parser.error(f"Output must be {root_text} (smoke runs with --limit may use build/)")
    if out.exists():
        parser.error("Output already exists; refusing to overwrite")
    p = ref.load_parameters()
    kan_self_check = ref.self_check(p)
    reference = ref.load_reference(args.xai_reference)
    reference_self_check = None if reference is None else ref.self_check_reference(p, reference)

    export_df = pd.read_csv(V2_DIR / "window_features.csv")
    export_df["recording_id"] = export_df["recording_id"].astype(str)
    eligible = pd.read_csv(ROOT / "outputs/openox_ppg_eligible_encounters.csv").set_index("encounter_id")
    recording_ids = list(dict.fromkeys(export_df["recording_id"]))
    if args.limit:
        recording_ids = recording_ids[:args.limit]
    names = criteria.NAMES
    tasks = []
    for rid in recording_ids:
        rows = export_df.loc[export_df["recording_id"] == rid]
        stored = rows[names].to_numpy(dtype=np.float64)
        tasks.append((rid, eligible.loc[rid].to_dict(), list(zip(rows["start_time_seconds"], stored)), args.xai_reference))

    out.mkdir(parents=True)
    print(f"Preparing {len(tasks)} recordings with {args.workers} workers ...", flush=True)
    recordings = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, rec in enumerate(pool.map(process_recording, tasks, chunksize=1), start=1):
            recordings.append(rec)
            if number % 10 == 0 or number == len(tasks):
                print(f"  {number}/{len(tasks)} recordings", flush=True)
    segments = out / "segments.f32"
    write_segments(segments, recordings)
    print("Running C++ deployed pipeline ...", flush=True)
    cpp, probe_banner = run_cpp(args.probe.resolve(), segments)
    by_segment = {s: cpp[cpp[:, 0] == s] for s in range(len(recordings))}

    decisions = []
    for segment, rec in enumerate(recordings):
        decisions.extend(compare(rec, by_segment[segment]))
    decisions_df = pd.DataFrame(decisions)
    decisions_df.to_csv(out / "decisions.csv.gz", index=False, compression="gzip")

    # Deployed versus offline export at export windows with 5 s future context.
    lookup = {(d["recording_id"], d["start_sample"]): d for d in decisions}
    samples = {rec["recording_id"]: rec["samples"] for rec in recordings}
    offline_p, _ = ref.probabilities_and_deltas(export_df[names].to_numpy(dtype=np.float64), p)
    export_df["offline_probability"] = offline_p
    export_df["start_sample"] = (export_df["start_time_seconds"] * 50).round().astype(int)
    export_df = export_df.loc[export_df["recording_id"].isin(recording_ids)].copy()
    export_df["decidable"] = [s + ref.FIRST <= samples[r] for r, s in zip(export_df["recording_id"], export_df["start_sample"])]
    matched = export_df.loc[export_df["decidable"]].copy()
    matched["cpp_probability"] = [lookup[(r, s)]["cpp_probability"] for r, s in zip(matched["recording_id"], matched["start_sample"])]
    matched["python_probability"] = [lookup[(r, s)]["python_probability"] for r, s in zip(matched["recording_id"], matched["start_sample"])]
    off_flag = matched["offline_probability"].to_numpy() >= ref.THRESHOLD
    cpp_flag = matched["cpp_probability"].to_numpy() >= ref.THRESHOLD
    py_flag = matched["python_probability"].to_numpy() >= ref.THRESHOLD
    matched["cpp_flipped"] = off_flag != cpp_flag
    matched["python_flipped"] = off_flag != py_flag
    matched["abs_dp_cpp"] = np.abs(matched["cpp_probability"] - matched["offline_probability"])
    matched[["recording_id", "start_time_seconds", "label", "offline_probability", "cpp_probability",
             "python_probability", "abs_dp_cpp", "cpp_flipped", "python_flipped"]].to_csv(
        out / "deployed_vs_offline.csv.gz", index=False, compression="gzip")
    y = matched["label"].to_numpy()
    flips = int(matched["cpp_flipped"].sum())

    v1_ir = []
    if args.v1_probe:
        picks = np.unique(np.linspace(0, len(recordings) - 1, min(args.v1_recordings, len(recordings))).astype(int))
        for segment in picks:
            v1_ir.append(run_v1_ir_check(args.v1_probe.resolve(), recordings[segment], by_segment[segment]))

    n = len(decisions_df)
    feature_max = {name: float(decisions_df[f"abs_err_{name}"].max()) for name in names}
    feature_max_rel = {name: float(decisions_df[f"rel_err_{name}"].max()) for name in names}
    feature_fail_counts = {name: int(decisions_df["failed_features"].str.split(";").apply(lambda s: name in s).sum()) for name in names}
    original_fail_counts = {name: int(decisions_df["original_absolute_failed_features"].str.split(";").apply(lambda s: name in s).sum()) for name in names}
    recon = np.concatenate([np.asarray(r["reconstruction_max_relative"]) for r in recordings])
    summary = {
        "recordings": len(recordings), "samples_per_channel": int(sum(samples.values())), "decisions": n,
        "failed_decisions": int((~decisions_df["pass"]).sum()),
        "numeric_failed_decisions": int((~decisions_df["numeric_pass"]).sum()),
        "feature_failed_decisions": int((~decisions_df["feature_pass"]).sum()),
        "original_absolute_1e_4_failed_decisions": int((~decisions_df["original_absolute_1e_4_pass"]).sum()),
        "probability_failed_decisions": int((~decisions_df["probability_pass"]).sum()),
        "attribution_failed_decisions": int((~decisions_df["attribution_delta_pass"]).sum()),
        "rank_failed_decisions": int((decisions_df["rank_violations"] > 0).sum()),
        "top3_order_changes": int(decisions_df["top3_order_changed"].sum()),
        "top3_set_changes": int(decisions_df["top3_set_changed"].sum()),
        "candidate_disagreements": int((~decisions_df["candidate_agrees"]).sum()),
        "metadata_failed": int((~decisions_df["metadata_pass"]).sum()),
        "states_failed": int((~decisions_df["states_pass"]).sum()),
        "top3_payload_failed": int((~decisions_df["top3_payload_pass"]).sum()),
        "two_wavelength_valid_disagreements": int((~decisions_df["two_wavelength_valid_agrees"]).sum()),
        "two_wavelength_invalid_decisions": int((~decisions_df["two_wavelength_valid"]).sum()),
        "max_probability_absolute_error": float(decisions_df["probability_absolute_error"].max()),
        "max_attribution_absolute_error": float(decisions_df["max_attribution_absolute_error"].max()),
        "feature_max_absolute_error": feature_max, "feature_max_relative_error": feature_max_rel,
        "feature_local_scale_failure_counts": feature_fail_counts,
        "feature_original_absolute_1e_4_failure_counts": original_fail_counts,
        "confirmed_decisions_python": int(sum(int(r["python"]["confirmed"].sum()) for r in recordings)),
    }
    offline = {
        "scope": "Export windows the deployed path can decide (start + 750 <= resampled length). Offline = v2 export features (whole-recording filter) through the same FP32-quantized v2 model; deployed = context-buffer filtering.",
        "export_windows_in_recordings": int(len(export_df)), "decidable_windows": int(len(matched)),
        "tail_windows_without_future_context": int((~export_df["decidable"]).sum()),
        "candidate_flips_cpp": flips, "flip_rate_cpp": flips / len(matched), "flip_rate_cpp_wilson95": wilson(flips, len(matched)),
        "candidate_flips_python": int(matched["python_flipped"].sum()),
        "flips_offline_candidate_to_deployed_not": int(np.sum(off_flag & ~cpp_flag)),
        "flips_offline_not_to_deployed_candidate": int(np.sum(~off_flag & cpp_flag)),
        "abs_dp_cpp": quantiles(matched["abs_dp_cpp"]),
        "abs_dp_cpp_by_label": {str(k): quantiles(g["abs_dp_cpp"]) for k, g in matched.groupby("label")},
        "flip_rate_by_label": {str(k): float(g["cpp_flipped"].mean()) for k, g in matched.groupby("label")},
        "label_metrics_at_0_50_resubstitution": {
            "note": "The v2 header was fitted on all v2 rows; these are training-set (resubstitution) counts, shown only to size the preprocessing shift, not a generalisation estimate.",
            "offline": label_metrics(y, off_flag), "deployed_cpp": label_metrics(y, cpp_flag)},
        "offline_reconstruction": {"windows": int(len(recon)), "max_relative_difference": float(recon.max()),
                                   "windows_above_1e-9": int(np.sum(recon > 1e-9))},
    }
    sources = [Path(__file__), ROOT / "scripts/v2_deployed_reference.py", ROOT / "scripts/parity_criteria_v2.py",
               ROOT / "scripts/parity_criteria.py", ROOT / "scripts/export_openox_windows_v2.py",
               ROOT / "scripts/check_probability_parity.py", ROOT / "scripts/verify_features.py",
               ROOT / "tests/v2_stream_probe.cpp", ROOT / "tests/fixtures/numerical_parity_contract.json",
               V2_DIR / "window_features.csv", ROOT / "notebooks/10_openox_canonical_window_export.ipynb"]
    sources += [ROOT / "firmware/Arduino Nano 33/include" / name for name in
                ("feature_extraction.h", "oxygen_risk_pipeline.h", "kan_inference.h", "kan_model_weights_v2.h")]
    if reference is not None:
        sources += [ROOT / "scripts/xai_reference.py", ref.REFERENCE_HEADER_V2]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if not args.limit else "smoke_subset",
        "board_run": False, "parity_complete": False,
        "scope": "Host only. Compiled C++ OxygenRiskPipeline (OXY_MODEL_V2, BasicOxygenRiskPipeline<250>) vs the Python deployed-path reference on continuous real OpenOximetry v2 recordings; every decision of every recording, one reset per recording. Not board evidence.",
        "host_compiler": "zig c++ (clang) -std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu",
        "probe_banner": probe_banner, "criteria": criteria.CRITERIA_DESCRIPTION,
        "vectorized_kan_self_check_max_error": kan_self_check,
        **({} if reference is None else {
            "xai_reference": args.xai_reference,
            "xai_reference_values": reference.tolist(),
            "xai_reference_self_check_max_error": reference_self_check,
            "xai_reference_note": "Probe built with -DOXY_XAI_REF_LABEL0; Python deltas set input i to KAN_XAI_REF_Z[i] "
                                  "(kan_xai_reference_v2.h, FP32) instead of 0. Budgets unchanged."}),
        "summary": summary, "deployed_vs_offline": offline, "v1_ir_feature_identity": v1_ir,
        "worst_decisions": decisions_df.sort_values("probability_absolute_error", ascending=False).head(10)[
            ["recording_id", "window_index", "python_probability", "cpp_probability", "probability_absolute_error",
             "max_attribution_absolute_error", "failed_features"]].to_dict("records"),
        "probe_sha256": sha256(args.probe), "segments_sha256": sha256(segments),
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "sources": [r["source"] | {"recording_id": r["recording_id"], "samples_50hz": r["samples"]} for r in recordings],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__},
    }
    with (out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        stream.write("\n")
    segments.unlink()  # 150+ MB, fully reproducible from the recorded sources
    print(json.dumps({"summary": {k: v for k, v in summary.items() if not isinstance(v, dict)},
                      "deployed_vs_offline": {k: offline[k] for k in ("decidable_windows", "candidate_flips_cpp",
                                                                       "flip_rate_cpp", "flip_rate_cpp_wilson95", "abs_dp_cpp")},
                      "v1_ir_feature_identity": v1_ir}, indent=2))
    print(f"Evidence: {out}")
    return 0 if not summary["failed_decisions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

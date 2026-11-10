"""P5 host parity (job F2) on the v2 host-parity recordings. Host only: no board, no serial.

Pre-registration: docs/p5-parity-criteria.md (commit f0f655a); budgets in scripts/parity_criteria_p5.py.
Inputs: every recording of outputs/openox-v2/window_features.csv, raw red/IR resampled to 50 Hz as
scripts/export_openox_windows_v2.py does (check_v2_host_parity.recording_streams), FP32, streamed as
paired samples, one reset per recording, through
  (a) the C++ deployed pipeline, tests/p5_stream_probe.cpp built with -DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0, and
  (b) scripts/p5_deployed_reference.py.
Classes per decision: (a) polarity, integer peak list, n_beats, beat_valid identical; (b) feature budgets
on (a)-pass decisions beat-valid on both sides (both invalid: inputs bit-equal to the imputation mean);
(c) probability 1e-4, 5 deltas 1e-5, ranking 2e-5, exact 0.50 candidate; (d) s_hat; (e) metadata,
Candidate/Confirmed/streak, top-3 payload. A decision fails if it fails any class.

Cause of every (a) mismatch (fixed in the pre-registration): the probe is re-run with a dump list and
writes the C++ FP32 filtered IR/red window; the extractor stages run on that window and on Python's own
float64 window; the first stage that differs is the cause, reported with its margin / window std.
Port check: Python extract on the C++ window must give the C++ discrete fields, and features whose FP32
rounding is the C++ value (within 0.5 FP32 spacing + 1e-12 x max(1, |f|)); on every mismatch plus a
SHA-256-selected sample of 200 decisions.

Deployed versus offline (descriptive): at export windows the deployed path can decide, the C++ deployed
inputs against outputs/physio-concept-v1/physio_features.csv.gz, and the P5 Candidate flip share (offline
P from the same replica on the offline inputs).

Writes a fresh outputs/parity/p5-host-<date>/ (smoke runs with --limit may use build/).
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
from scipy.signal._peak_finding_utils import _local_maxima_1d, _select_by_peak_distance, _peak_prominences

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_v2_host_parity as host  # noqa: E402
import parity_criteria_p5 as criteria  # noqa: E402
import p5_deployed_reference as ref  # noqa: E402
import physio_features as pf  # noqa: E402

OUT_PREFIX = "p5-host-"
COLS = 63
F0, F1 = 7, 12            # 5 model inputs
STATE = slice(12, 15)     # candidate confirmed streak
TOP = slice(15, 21)       # 3 x (index, delta)
BEAT_VALID, N_BEATS, POLARITY, TIE, S_HAT = 21, 22, 23, 24, 25
D0 = 26                   # 5 deltas
DC_IR, DC_RED = 31, 32
PEAKS = slice(33, 63)
PORT_SAMPLE = 200
PORT_SEED = "p5-host-port-check-2026-10-30"


def process_recording(task):
    recording_id, row = task
    p = ref.load_parameters()
    ir, red, source = host.recording_streams(row)
    ir32, red32 = ir.astype(np.float32), red.astype(np.float32)
    return {"recording_id": recording_id, "samples": len(ir32), "ir": ir32, "red": red32,
            "python": ref.stream(ir32, red32, p), "source": source}


def run_cpp(probe, segments, extra=()):
    result = subprocess.run([str(probe), str(segments), *map(str, extra)], capture_output=True, text=True, check=True,
                            timeout=7200)
    rows = np.array(result.stdout.split(), dtype=np.float64)
    if rows.size % COLS:
        raise ValueError("Malformed probe output")
    rows = rows.reshape(-1, COLS)
    if not np.isfinite(rows).all():
        raise ValueError("Non-finite probe output")
    return rows, result.stderr.strip()


def compare(rec, cpp, p):
    py = rec["python"]
    n = len(py["probability"])
    if len(cpp) != n:
        raise ValueError(f"{rec['recording_id']}: {len(cpp)} native vs {n} Python decisions")
    out = []
    for i in range(n):
        row, beat = cpp[i], py["beats"][i]
        cpp_peaks = [int(v) for v in row[PEAKS] if v >= 0]
        discrete = (int(row[POLARITY]) == beat["polarity"] and cpp_peaks == beat["peaks"][:30]
                    and len(beat["peaks"]) <= 30 and int(row[N_BEATS]) == beat["n_beats"]
                    and bool(row[BEAT_VALID]) == beat["beat_valid"])
        both_valid = discrete and beat["beat_valid"]
        both_invalid = discrete and not beat["beat_valid"]
        if both_valid:
            fcheck = criteria.feature_check(py["features"][i], row[F0:F1])
            feature_pass = fcheck["pass"]
        else:
            fcheck = None
            feature_pass = (bool(np.array_equal(np.float32(row[F0:F1]), np.float32(p["imputation"])))
                            if both_invalid else False)
        pcheck = criteria.probability_check(py["probability"][i], row[6])
        acheck = criteria.attribution_check(py["deltas"][i], row[D0:D0 + 5])
        scheck = criteria.s_hat_check(py["s_hat"][i], row[S_HAT], p["target_std"])
        cpp_deltas = row[D0:D0 + 5].astype(np.float32)
        top_index = row[TOP][0::2].astype(int)
        payload = (np.array_equal(top_index, np.argsort(-np.abs(cpp_deltas), kind="stable")[:3])
                   and np.array_equal(row[TOP][1::2].astype(np.float32), cpp_deltas[top_index]))
        meta = np.array_equal(row[1:6], py["meta"][i])
        states = np.array_equal(row[STATE], [int(py["candidate"][i]), int(py["confirmed"][i]), int(py["streak"][i])])
        c_pass = pcheck["pass"] and acheck["pass"]
        exact = bool(meta and states and payload)
        record = {
            "recording_id": rec["recording_id"], "window_index": i, "start_sample": int(py["meta"][i][2]),
            "pass": bool(discrete and feature_pass and c_pass and scheck["pass"] and exact),
            "a_discrete_pass": bool(discrete), "b_feature_pass": bool(feature_pass), "c_pass": bool(c_pass),
            "d_s_hat_pass": scheck["pass"], "e_exact_pass": exact,
            "python_beat_valid": beat["beat_valid"], "cpp_beat_valid": bool(row[BEAT_VALID]),
            "python_n_beats": beat["n_beats"], "cpp_n_beats": int(row[N_BEATS]),
            "python_polarity": beat["polarity"], "cpp_polarity": int(row[POLARITY]), "cpp_distance_tie": bool(row[TIE]),
            "probability_pass": pcheck["probability_pass"], "candidate_agrees": pcheck["candidate_agrees"],
            "attribution_delta_pass": acheck["delta_pass"], "rank_violations": len(acheck["rank_violations"]),
            "top3_order_changed": acheck["top3_order_changed"], "top3_set_changed": acheck["top3_set_changed"],
            "metadata_pass": bool(meta), "states_pass": bool(states), "top3_payload_pass": bool(payload),
            "python_probability": float(py["probability"][i]), "cpp_probability": float(row[6]),
            "probability_absolute_error": pcheck["absolute_error"],
            "max_attribution_absolute_error": acheck["max_absolute_error"],
            "python_s_hat": float(py["s_hat"][i]), "cpp_s_hat": float(row[S_HAT]),
            "s_hat_absolute_error": scheck["absolute_error"], "s_hat_display_differs": scheck["display_text_differs"],
            "failed_features": ";".join(fcheck["failed_features"]) if fcheck else "",
            **{f"abs_err_{name}": (fcheck["absolute_errors"][k] if fcheck else np.nan) for k, name in enumerate(criteria.NAMES)},
            **{f"rel_err_{name}": (fcheck["relative_errors"][k] if fcheck else np.nan) for k, name in enumerate(criteria.NAMES)},
        }
        out.append(record)
    return out


def stages(x, r, dc_ir, dc_red):
    """Extractor stages of physio_features on one float64 window (SciPy's own internals for find_peaks)."""
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    out = {"usable": bool(std > 0 and np.isfinite(x).all() and np.isfinite(r).all() and dc_ir > 0 and dc_red > 0), "std": std}
    if not out["usable"]:
        return out
    d = np.diff(x)
    q99, q01 = float(np.percentile(d, 99.0)), float(np.percentile(d, 1.0))
    s = 1 if q99 >= -q01 else -1
    y = np.ascontiguousarray(s * x)
    maxima = _local_maxima_1d(y)[0]
    kept = maxima[_select_by_peak_distance(maxima, y[maxima], float(pf.MIN_PEAK_DISTANCE))]
    prominences = _peak_prominences(y, kept, -1)[0]
    final = kept[prominences >= pf.PROMINENCE_STD * std]
    out.update(polarity=s, polarity_margin=abs(q99 + q01) / std, y=y, maxima=maxima.tolist(), kept=kept.tolist(),
               prominence=dict(zip(kept.tolist(), prominences.tolist())), final=final.tolist())
    return out


def cause(python_stage, cpp_stage, python_beat, cpp_beat):
    """First differing stage and its margin relative to the Python window's std."""
    a, b = python_stage, cpp_stage
    if a["usable"] != b["usable"]:
        return "1_usable_window", None
    std = a["std"]
    if a["polarity"] != b["polarity"]:
        return "2_polarity", a["polarity_margin"]
    if a["maxima"] != b["maxima"]:
        y = a["y"]
        flipped = sorted(set(a["maxima"]) ^ set(b["maxima"]))
        margins = [min(abs(y[i] - y[i - 1]), abs(y[i + 1] - y[i])) / std for i in flipped if 0 < i < len(y) - 1]
        return "3_local_maxima", min(margins) if margins else None
    if a["kept"] != b["kept"]:
        y = a["y"]
        flipped = sorted(set(a["kept"]) ^ set(b["kept"]))
        margins = [min(abs(y[i] - y[j]) for j in a["maxima"] if j != i and abs(j - i) < pf.MIN_PEAK_DISTANCE) / std
                   for i in flipped if any(j != i and abs(j - i) < pf.MIN_PEAK_DISTANCE for j in a["maxima"])]
        return "4_distance_selection", min(margins) if margins else None
    if a["final"] != b["final"]:
        flipped = sorted(set(a["final"]) ^ set(b["final"]))
        margins = [abs(a["prominence"][i] - std) / std for i in flipped if i in a["prominence"]]
        return "5_prominence_selection", min(margins) if margins else None
    if python_beat != cpp_beat:
        return "6_beat_valid_same_peaks", None
    return "none_found", None


def read_dump(path):
    raw = np.fromfile(path, dtype=np.uint8)
    record = 8 + 4 * (500 + 500 + 2)
    if raw.size % record:
        raise ValueError("Malformed dump")
    out = {}
    for k in range(raw.size // record):
        chunk = raw[k * record:(k + 1) * record]
        seg, win = (int(v) for v in chunk[:8].view("<u4"))
        values = chunk[8:].view("<f4")
        out[(seg, win)] = (values[:500].copy(), values[500:1000].copy(), float(values[1000]), float(values[1001]))
    return out


def port_check(window, cpp_row):
    ir, red, dc_ir, dc_red = window
    beat = pf.extract(ir.astype(np.float64), red.astype(np.float64), dc_ir, dc_red)
    usable = bool(np.std(ir.astype(np.float64)) > 0 and dc_ir > 0 and dc_red > 0)
    peaks = pf.detect_beats(ir.astype(np.float64))[1].tolist() if usable else []
    discrete = (peaks[:30] == [int(v) for v in cpp_row[PEAKS] if v >= 0] and beat["polarity"] == int(cpp_row[POLARITY])
                and beat["n_beats"] == int(cpp_row[N_BEATS]) and beat["beat_valid"] == bool(cpp_row[BEAT_VALID]))
    features_ok = True
    if beat["beat_valid"] and discrete:
        f = np.array([beat[n] for n in pf.FEATURES])
        cpp_f = cpp_row[F0:F1].astype(np.float32).astype(np.float64)   # the exact FP32 value (9 digits round-trip)
        tolerance = 0.5 * np.abs(np.spacing(cpp_f.astype(np.float32))).astype(np.float64) + 1e-12 * np.maximum(1.0, np.abs(f))
        features_ok = bool(np.all(np.abs(f - cpp_f) <= tolerance))
    return bool(discrete and features_ok), beat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True, help="p5_stream_probe (-DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first N recordings only (smoke run; not the reported result)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    out = args.output.resolve()
    root_ok = out.parent == (ROOT / "outputs/parity").resolve() and out.name.startswith(OUT_PREFIX)
    if not (root_ok or (args.limit and out.is_relative_to((ROOT / "build").resolve()))):
        parser.error(f"Output must be outputs/parity/{OUT_PREFIX}*/ (smoke runs with --limit may use build/)")
    if out.exists():
        parser.error("Output already exists; refusing to overwrite")
    p = ref.load_parameters()
    self_check = ref.self_check(p)

    export_df = pd.read_csv(ROOT / "outputs/openox-v2/window_features.csv", low_memory=False)
    export_df["recording_id"] = export_df["recording_id"].astype(str)
    eligible = pd.read_csv(ROOT / "outputs/openox_ppg_eligible_encounters.csv").set_index("encounter_id")
    recording_ids = list(dict.fromkeys(export_df["recording_id"]))
    if args.limit:
        recording_ids = recording_ids[:args.limit]
    tasks = [(rid, eligible.loc[rid].to_dict()) for rid in recording_ids]
    out.mkdir(parents=True)
    print(f"Preparing {len(tasks)} recordings with {args.workers} workers ...", flush=True)
    recordings = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, rec in enumerate(pool.map(process_recording, tasks, chunksize=1), start=1):
            recordings.append(rec)
            if number % 10 == 0 or number == len(tasks):
                print(f"  {number}/{len(tasks)} recordings", flush=True)
    segments = out / "segments.f32"
    host.write_segments(segments, recordings)
    print("Running C++ deployed pipeline ...", flush=True)
    cpp, banner = run_cpp(args.probe.resolve(), segments)
    by_segment = {s: cpp[cpp[:, 0] == s] for s in range(len(recordings))}
    decisions = []
    for segment, rec in enumerate(recordings):
        for d in compare(rec, by_segment[segment], p):
            decisions.append({"segment": segment, **d})
    df = pd.DataFrame(decisions)

    # Dumps: every (a) mismatch, plus a SHA-256 sample of decisions for the port check.
    mismatch_keys = [(int(r.segment), int(r.window_index)) for r in df.itertuples() if not r.a_discrete_pass]
    order = sorted(((int(r.segment), int(r.window_index)) for r in df.itertuples()),
                   key=lambda k: hashlib.sha256(f"{PORT_SEED}:{recordings[k[0]]['recording_id']}:{k[1]}".encode()).hexdigest())
    sample_keys = order[:PORT_SAMPLE]
    dump_list, dump_file = out / "dump_list.txt", out / "dump.f32"
    dump_list.write_text("".join(f"{s} {w}\n" for s, w in sorted(set(mismatch_keys) | set(sample_keys))))
    cpp2, _ = run_cpp(args.probe.resolve(), segments, (dump_list, dump_file))
    if not np.array_equal(cpp2, cpp):
        raise RuntimeError("Second probe run differs from the first")
    dumps = read_dump(dump_file)
    port_rows, mismatch_rows = [], []
    for key in sorted(dumps):
        seg, win = key
        ok, _ = port_check(dumps[key], by_segment[seg][win])
        port_rows.append({"recording_id": recordings[seg]["recording_id"], "window_index": win, "port_check_pass": ok,
                          "role": "mismatch" if key in set(mismatch_keys) else "sample"})
    for seg, win in mismatch_keys:
        rec = recordings[seg]
        start = win * ref.STRIDE
        py_window = ref.window_filter(rec["ir"].astype(np.float64), rec["red"].astype(np.float64), start)
        cpp_window = dumps[(seg, win)]
        py_stage = stages(*py_window)
        cpp_stage = stages(cpp_window[0].astype(np.float64), cpp_window[1].astype(np.float64), cpp_window[2], cpp_window[3])
        beat = rec["python"]["beats"][win]
        row = by_segment[seg][win]
        name, margin = cause(py_stage, cpp_stage, beat["beat_valid"], bool(row[BEAT_VALID]))
        d = df[(df.segment == seg) & (df.window_index == win)].iloc[0]
        mismatch_rows.append({"recording_id": rec["recording_id"], "window_index": win, "start_sample": start, "cause": name,
                              "margin_over_std": margin, "cpp_distance_tie": bool(row[TIE]),
                              "python_n_beats": beat["n_beats"], "cpp_n_beats": int(row[N_BEATS]),
                              "python_beat_valid": beat["beat_valid"], "cpp_beat_valid": bool(row[BEAT_VALID]),
                              "python_polarity": beat["polarity"], "cpp_polarity": int(row[POLARITY]),
                              "python_peaks": " ".join(map(str, beat["peaks"])),
                              "cpp_peaks": " ".join(str(int(v)) for v in row[PEAKS] if v >= 0),
                              "python_probability": float(d.python_probability), "cpp_probability": float(d.cpp_probability),
                              "probability_absolute_error": float(d.probability_absolute_error),
                              "candidate_agrees": bool(d.candidate_agrees), "c_pass": bool(d.c_pass),
                              "s_hat_absolute_error": float(d.s_hat_absolute_error)})
    mismatches = pd.DataFrame(mismatch_rows)
    ports = pd.DataFrame(port_rows)
    df.to_csv(out / "decisions.csv.gz", index=False, compression="gzip")
    mismatches.to_csv(out / "peak_mismatches.csv", index=False)
    ports.to_csv(out / "port_check.csv", index=False)

    # Deployed versus offline at decidable export windows.
    physio = pd.read_csv(ROOT / "outputs/physio-concept-v1/physio_features.csv.gz")
    export_df["start_sample"] = (export_df["start_time_seconds"] * 50).round().astype(int)
    for name in criteria.NAMES + ["n_beats", "beat_valid"]:
        export_df[f"offline_{name}"] = physio[name].to_numpy()
    offline_inputs = np.where(physio["beat_valid"].to_numpy(bool)[:, None], physio[criteria.NAMES].to_numpy(np.float64),
                              p["imputation"][None, :])
    export_df["offline_probability"] = ref.evaluate(offline_inputs, p)[0]
    samples = {rec["recording_id"]: rec["samples"] for rec in recordings}
    sub = export_df.loc[export_df["recording_id"].isin(recording_ids)].copy()
    sub["decidable"] = [s + ref.FIRST <= samples[r] for r, s in zip(sub["recording_id"], sub["start_sample"])]
    matched = sub.loc[sub["decidable"]].copy()
    seg_of = {rec["recording_id"]: k for k, rec in enumerate(recordings)}
    rows = [by_segment[seg_of[r]][(s // ref.STRIDE)] for r, s in zip(matched["recording_id"], matched["start_sample"])]
    if not all(int(r[3]) == s for r, s in zip(rows, matched["start_sample"])):
        raise RuntimeError("Export windows do not land on deployed decisions")
    rows = np.array(rows)
    matched["deployed_probability"] = rows[:, 6]
    matched["deployed_beat_valid"] = rows[:, BEAT_VALID].astype(bool)
    matched["deployed_n_beats"] = rows[:, N_BEATS].astype(int)
    for k, name in enumerate(criteria.NAMES):
        matched[f"deployed_{name}"] = rows[:, F0 + k]
    off_flag = matched["offline_probability"].to_numpy() >= ref.THRESHOLD
    dep_flag = matched["deployed_probability"].to_numpy() >= ref.THRESHOLD
    matched["flipped"] = off_flag != dep_flag
    matched[["recording_id", "start_time_seconds", "label", "offline_probability", "deployed_probability", "flipped",
             "offline_beat_valid", "deployed_beat_valid", "offline_n_beats", "deployed_n_beats"]
            + [f"{s}_{n}" for n in criteria.NAMES for s in ("offline", "deployed")]].to_csv(
        out / "deployed_vs_offline.csv.gz", index=False, compression="gzip")
    both = matched["offline_beat_valid"].astype(bool) & matched["deployed_beat_valid"]
    flips = int(matched["flipped"].sum())
    y = matched["label"].to_numpy()
    offline = {
        "scope": "Export windows the deployed path can decide (start + 750 <= resampled length). Offline = physio_features.csv.gz "
                 "(whole-recording filter, stored FP32 windows) through the same P5 replica; deployed = the C++ context-buffer path.",
        "decidable_windows": int(len(matched)), "tail_windows_without_future_context": int((~sub["decidable"]).sum()),
        "beat_valid_agreement": float((matched["offline_beat_valid"].astype(bool) == matched["deployed_beat_valid"]).mean()),
        "n_beats_agreement": float((matched["offline_n_beats"] == matched["deployed_n_beats"]).mean()),
        "both_beat_valid": int(both.sum()),
        "abs_diff_both_valid": {n: host.quantiles(np.abs(matched.loc[both, f"deployed_{n}"] - matched.loc[both, f"offline_{n}"]))
                                for n in criteria.NAMES},
        "rel_diff_both_valid": {n: host.quantiles(np.abs(matched.loc[both & (matched[f"offline_{n}"] != 0), f"deployed_{n}"] - matched.loc[both & (matched[f"offline_{n}"] != 0), f"offline_{n}"])
                                                  / np.abs(matched.loc[both & (matched[f"offline_{n}"] != 0), f"offline_{n}"])) for n in criteria.NAMES},
        "candidate_flips": flips, "flip_rate": flips / len(matched), "flip_rate_wilson95": host.wilson(flips, len(matched)),
        "flips_offline_candidate_to_deployed_not": int(np.sum(off_flag & ~dep_flag)),
        "flips_offline_not_to_deployed_candidate": int(np.sum(~off_flag & dep_flag)),
        "flip_rate_by_label": {str(k): float(g["flipped"].mean()) for k, g in matched.groupby("label")},
        "abs_dp": host.quantiles(np.abs(matched["deployed_probability"] - matched["offline_probability"])),
        "label_metrics_at_0_50_resubstitution": {
            "note": "The P5 model was fitted on all rows: training-set counts, shown only to size the preprocessing shift.",
            "offline": host.label_metrics(y, off_flag), "deployed_cpp": host.label_metrics(y, dep_flag)},
    }

    n = len(df)
    ok_a = int(df["a_discrete_pass"].sum())
    a_pass = df["a_discrete_pass"]
    both_valid = a_pass & df["python_beat_valid"]
    summary = {
        "recordings": len(recordings), "decisions": n, "failed_decisions": int((~df["pass"]).sum()),
        "a_peak_list_identical": ok_a, "a_peak_list_identical_share": ok_a / n, "a_wilson95": host.wilson(ok_a, n),
        "a_mismatches": n - ok_a,
        "a_mismatch_causes": mismatches["cause"].value_counts().to_dict() if len(mismatches) else {},
        "a_cpp_distance_tie_decisions": int(df["cpp_distance_tie"].sum()),
        "b_decisions_checked_both_valid": int(both_valid.sum()),
        "b_decisions_both_invalid": int((a_pass & ~df["python_beat_valid"]).sum()),
        "b_feature_failed": int((a_pass & ~df["b_feature_pass"]).sum()),
        "b_feature_failure_counts": {name: int(df.loc[both_valid, "failed_features"].str.split(";").apply(lambda s: name in s).sum())
                                     for name in criteria.NAMES},
        "b_feature_max_abs_error": {name: float(df.loc[both_valid, f"abs_err_{name}"].max()) for name in criteria.NAMES},
        "b_feature_max_rel_error": {name: float(df.loc[both_valid, f"rel_err_{name}"].max()) for name in criteria.NAMES},
        "c_failed_all": int((~df["c_pass"]).sum()), "c_failed_on_a_pass": int((a_pass & ~df["c_pass"]).sum()),
        "c_probability_failed": int((~df["probability_pass"]).sum()),
        "c_attribution_failed": int((~df["attribution_delta_pass"]).sum()),
        "c_rank_failed": int((df["rank_violations"] > 0).sum()),
        "c_candidate_disagreements": int((~df["candidate_agrees"]).sum()),
        "c_candidate_disagreements_on_a_pass": int((a_pass & ~df["candidate_agrees"]).sum()),
        "c_top3_order_changes": int(df["top3_order_changed"].sum()), "c_top3_set_changes": int(df["top3_set_changed"].sum()),
        "c_max_probability_error_on_a_pass": float(df.loc[a_pass, "probability_absolute_error"].max()),
        "c_max_attribution_error_on_a_pass": float(df.loc[a_pass, "max_attribution_absolute_error"].max()),
        "d_s_hat_failed": int((~df["d_s_hat_pass"]).sum()), "d_s_hat_failed_on_a_pass": int((a_pass & ~df["d_s_hat_pass"]).sum()),
        "d_max_s_hat_error_on_a_pass": float(df.loc[a_pass, "s_hat_absolute_error"].max()),
        "d_s_hat_display_text_differs": int(df["s_hat_display_differs"].sum()),
        "e_exact_failed": int((~df["e_exact_pass"]).sum()), "e_metadata_failed": int((~df["metadata_pass"]).sum()),
        "e_states_failed": int((~df["states_pass"]).sum()), "e_top3_payload_failed": int((~df["top3_payload_pass"]).sum()),
        "beat_invalid_python": int((~df["python_beat_valid"]).sum()), "beat_invalid_cpp": int((~df["cpp_beat_valid"]).sum()),
        "confirmed_python": int(sum(int(r["python"]["confirmed"].sum()) for r in recordings)),
        "port_check": {"windows": int(len(ports)), "passed": int(ports["port_check_pass"].sum()) if len(ports) else 0,
                       "mismatch_windows": int((ports["role"] == "mismatch").sum()) if len(ports) else 0},
    }
    sources = [Path(__file__), ROOT / "scripts/p5_deployed_reference.py", ROOT / "scripts/parity_criteria_p5.py",
               ROOT / "scripts/parity_criteria.py", ROOT / "scripts/physio_features.py", ROOT / "scripts/v2_deployed_reference.py",
               ROOT / "scripts/check_v2_host_parity.py", ROOT / "scripts/export_openox_windows_v2.py", ROOT / "scripts/xai_reference.py",
               ROOT / "tests/p5_stream_probe.cpp", ROOT / "tests/fixtures/numerical_parity_contract.json",
               ROOT / "docs/p5-parity-criteria.md", ROOT / "outputs/openox-v2/window_features.csv",
               ROOT / "outputs/physio-concept-v1/physio_features.csv.gz"]
    sources += [ROOT / "firmware/Arduino Nano 33/include" / name for name in
                ("feature_extraction.h", "oxygen_risk_pipeline.h", "kan_inference.h", "physio_features.h", "kan_model_weights_p5.h",
                 "kan_sat_regressor_p5.h", "kan_sat_inference_p5.h", "kan_xai_reference_p5.h")]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if not args.limit else "smoke_subset",
        "board_run": False, "parity_complete": summary["failed_decisions"] == 0 and not args.limit,
        "scope": "Host only. Compiled C++ OxygenRiskPipeline (OXY_MODEL_P5, OXY_XAI_REF_LABEL0, BasicOxygenRiskPipeline<250>) vs "
                 "the Python deployed-path reference on continuous real OpenOximetry v2 recordings; every decision, one reset per "
                 "recording. Not board evidence.",
        "host_compiler": "zig c++ (clang) -std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu",
        "probe_banner": banner, "criteria": criteria.CRITERIA_DESCRIPTION, "reference_self_check_max_error": self_check,
        "xai_reference_values": p["reference"].tolist(), "summary": summary, "peak_mismatches": mismatch_rows,
        "deployed_vs_offline": offline, "probe_sha256": host.sha256(args.probe), "segments_sha256": host.sha256(segments),
        "sha256": {str(path.relative_to(ROOT).as_posix()): host.sha256(path) for path in sources},
        "sources": [r["source"] | {"recording_id": r["recording_id"], "samples_50hz": r["samples"]} for r in recordings],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__},
    }
    with (out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        stream.write("\n")
    segments.unlink()
    dump_file.unlink()
    print(json.dumps({"summary": summary, "deployed_vs_offline": {k: offline[k] for k in (
        "decidable_windows", "beat_valid_agreement", "n_beats_agreement", "candidate_flips", "flip_rate", "flip_rate_wilson95")}},
        indent=2, default=float))
    return 0 if not summary["failed_decisions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

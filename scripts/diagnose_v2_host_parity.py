"""Read-only diagnosis of the decisions that failed a v2 host parity run. Changes nothing.

For every failed decision: (1) Python KAN on the C++ features (does the error come from the
KAN arithmetic or from the features?); (2) one-feature substitutions (which feature's C++
value moves the deltas outside the 1e-5 budget); (3) signal scale of the scored window
(filtered IR RMS and raw range across the 1,000-sample context); (4) the frozen v1 model on
the same 14 IR features (are these windows a v2-model effect?). Writes a fresh directory.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_v2_host_parity as host  # noqa: E402
import parity_criteria_v2 as criteria  # noqa: E402
import v2_deployed_reference as ref  # noqa: E402
from check_probability_parity import reference as v1_reference  # noqa: E402
from parity_criteria import attribution_check as v1_attribution_check  # noqa: E402
from verify_features import parse_c_header  # noqa: E402

V1_HEADER = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="outputs/parity/v2-host/<run> directory")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(host.OUT_ROOT.resolve()):
        parser.error("Fresh output directory under outputs/parity/v2-host/ required")
    decisions = pd.read_csv(args.run / "decisions.csv.gz", low_memory=False)
    failed = decisions.loc[~decisions["pass"]]
    eligible = pd.read_csv(ROOT / "outputs/openox_ppg_eligible_encounters.csv").set_index("encounter_id")
    p = ref.load_parameters()
    p1 = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in parse_c_header(V1_HEADER).items()}
    budget = criteria.CONTRACT["attribution_absolute"]
    rows = []
    for rid, group in failed.groupby("recording_id"):
        ir, red, _ = host.recording_streams(eligible.loc[rid].to_dict())
        ir32, red32 = ir.astype(np.float32), red.astype(np.float32)
        segments = out.parent / f".diag-{rid[:12]}.f32"
        host.write_segments(segments, [{"samples": len(ir32), "ir": ir32, "red": red32}])
        cpp, _ = host.run_cpp(args.probe.resolve(), segments)
        segments.unlink()
        python = ref.stream(ir32, red32, p)
        for i in group["window_index"].astype(int):
            cpp_f, py_f = cpp[i, host.F0:host.F1], python["features"][i]
            cpp_d, py_d = cpp[i, host.D0:host.D0 + 16], python["deltas"][i]
            _, on_cpp = ref.probabilities_and_deltas(cpp_f, p)
            kan_only = float(np.max(np.abs(on_cpp[0] - cpp_d)))
            substitution = {}
            for k, name in enumerate(criteria.NAMES):
                mixed = py_f.copy()
                mixed[k] = cpp_f[k]
                _, d = ref.probabilities_and_deltas(mixed, p)
                substitution[name] = float(np.max(np.abs(d[0] - py_d)))
            start = i * ref.STRIDE
            left = max(0, start - ref.CONTEXT)
            context = ir32[left:start + ref.FIRST].astype(np.float64)
            filtered = signal.sosfiltfilt(ref.SOS, context)[start - left:start - left + ref.WINDOW]
            scaled = (py_f - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)
            check = criteria.feature_check(py_f, cpp_f)
            v1_py_p, v1_py_d = v1_reference(py_f[None, :14], p1)
            v1_cpp_p, v1_cpp_d = v1_reference(cpp_f[None, :14], p1)
            v1_check = v1_attribution_check(v1_py_d[0], v1_cpp_d[0])
            rows.append({
                "recording_id": rid, "window_index": i, "python_probability": float(python["probability"][i]),
                "max_delta_error": float(np.max(np.abs(cpp_d - py_d))),
                "worst_delta_feature": criteria.NAMES[int(np.argmax(np.abs(cpp_d - py_d)))],
                "kan_arithmetic_only_error": kan_only,
                "largest_single_feature_substitution": max(substitution, key=substitution.get),
                "largest_single_feature_substitution_error": max(substitution.values()),
                "substitutions_over_budget": [n for n, v in substitution.items() if v > budget],
                "feature_failed": check["failed_features"],
                "relative_errors": dict(zip(criteria.NAMES, check["relative_errors"])),
                "filtered_ir_rms": float(np.sqrt(np.mean(filtered**2))),
                "raw_context_range": float(np.ptp(context)), "raw_target_range": float(np.ptp(ir32[start:start + 500])),
                "raw_target_mean": float(np.mean(ir32[start:start + 500])),
                "max_abs_standardized_input": float(np.max(np.abs(scaled))),
                "max_abs_standardized_ir_input": float(np.max(np.abs(scaled[:14]))),
                "v1_model_probability_error": float(abs(v1_py_p[0] - v1_cpp_p[0])),
                "v1_model_max_delta_error": v1_check["max_absolute_error"], "v1_model_attribution_pass": v1_check["pass"],
                "standardized_inputs": dict(zip(criteria.NAMES, scaled.tolist())),
            })
    df = pd.DataFrame(rows)
    out.mkdir(parents=True)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run": str(args.run.resolve()),
        "failed_decisions": int(len(df)), "recordings": int(df["recording_id"].nunique()),
        "max_kan_arithmetic_only_error": float(df["kan_arithmetic_only_error"].max()),
        "failures_explained_by_one_feature": int((df["largest_single_feature_substitution_error"] > budget).sum()),
        "largest_single_feature_substitution_counts": df["largest_single_feature_substitution"].value_counts().to_dict(),
        "worst_delta_feature_counts": df["worst_delta_feature"].value_counts().to_dict(),
        "filtered_ir_rms": host.quantiles(df["filtered_ir_rms"]),
        "raw_target_range": host.quantiles(df["raw_target_range"]),
        "raw_context_over_filtered_rms": host.quantiles(df["raw_context_range"] / np.maximum(df["filtered_ir_rms"], 1e-30)),
        "max_abs_standardized_input": host.quantiles(df["max_abs_standardized_input"]),
        "max_abs_standardized_ir_input": host.quantiles(df["max_abs_standardized_ir_input"]),
        "standardized_ir_perfusion_index": host.quantiles([w["standardized_inputs"]["ir_perfusion_index"] for w in rows]),
        "v1_model_on_same_ir_features": {
            "note": "Frozen v1 model on the identical 14 IR features (Python vs C++ values) of these windows.",
            "attribution_failures": int((~df["v1_model_attribution_pass"]).sum()),
            "max_delta_error": float(df["v1_model_max_delta_error"].max()),
            "max_probability_error": float(df["v1_model_probability_error"].max())},
        "feature_budget_failures": [{"recording_id": r["recording_id"], "window_index": r["window_index"],
                                     "features": r["feature_failed"]} for r in rows if r["feature_failed"]],
        "per_recording": df.groupby("recording_id").size().to_dict(),
    }
    with (out / "report.json").open("x") as stream:
        json.dump({"summary": summary, "windows": rows}, stream, indent=2, default=float)
        stream.write("\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    raise SystemExit(main())

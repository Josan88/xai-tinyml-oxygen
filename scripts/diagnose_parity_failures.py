"""Per-window stage diagnosis for every frozen-contract development failure."""
import argparse
import json
from pathlib import Path
import numpy as np

from parity_criteria import ROOT, CONTRACT_SHA256, NAMES, check_window, feature_check
from audit_feature_errors import fp32, sha256
from check_probability_parity import canonical_feature_function, reference, HEADER
from verify_features import parse_c_header
from review_feature_precision import exported_filter_constants, high_precision_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sources", type=Path,
                        help="Preserved original headers for diagnosis after an arithmetic repair")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "outputs").resolve()) or out.exists():
        raise ValueError("Use a new output directory under outputs/")
    baseline = json.loads((args.baseline / "report.json").read_text())
    metadata = json.loads((args.cases / "manifest.json").read_text())
    if baseline["role"] != "development" or baseline["contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Diagnosis requires matching development evidence")
    if sha256(args.cases / "cases.npz") != baseline["cases_sha256"]:
        raise ValueError("Cases changed")
    verified_sources = {}
    for path, digest in baseline["source_sha256"].items():
        source = Path(path)
        if sha256(source) != digest and args.baseline_sources:
            source = args.baseline_sources / source.name
        if not source.is_file() or sha256(source) != digest:
            raise ValueError("Baseline source changed; supply the matching preserved headers")
        verified_sources[Path(path).name] = source
    with np.load(args.cases / "cases.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(args.baseline / "native.npz", allow_pickle=False) as native:
        actual = {int(i): row for i, row in zip(native["case_indices"], native["rows"])}
    parameters = {k: fp32(v) for k, v in parse_c_header(HEADER).items()}
    compute = canonical_feature_function()
    sos, zi = exported_filter_constants(verified_sources["feature_extraction.h"])
    results = []
    for failed in (w for w in baseline["windows"] if not w["pass"]):
        i = failed["case_index"]
        row = actual[i]
        n = int(data["counts"][i])
        high_filtered = high_precision_filter(data["raw"][i, :n], 0 if n == 750 else 250, sos, zi)
        feature_sets = []
        for samples in (high_filtered, row[46:]):
            f = compute(samples, 50.)
            feature_sets.append(np.array([f[name] for name in NAMES]))
        feature_sets.append(row[9:23])
        probabilities, deltas = reference(np.array(feature_sets), parameters)
        stages = {}
        for j, name in enumerate(("rounded_constants_fp64_control", "native_filter_python_features", "native_features_python_inference")):
            stages[name] = check_window(data["features"][i], feature_sets[j], data["probabilities"][i], probabilities[j], data["deltas"][i], deltas[j])
        stages["native_complete"] = failed
        feature_math = feature_check(feature_sets[1], feature_sets[2])
        bp = {}
        for j, name in ((11, "lower_band"), (12, "upper_band"), (13, "ratio")):
            ref = data["features"][i, j]
            bp[name] = {"reference": float(ref), "cpp": float(row[9 + j]),
                        "filter_stage_signed_error": float(feature_sets[1][j] - ref),
                        "feature_math_signed_error": float(feature_sets[2][j] - feature_sets[1][j]),
                        "absolute_error": float(abs(row[9 + j] - ref)), "frozen_budget": failed["features"]["budgets"][j]}
        bp["division_only_absolute_error"] = float(abs(row[22] - row[20] / row[21])) if row[21] > 0 else None
        results.append({"case_index": i, **metadata["keys"][i], "baseline_failed_features": failed["features"]["failed_features"],
                        "baseline_failed_deltas": failed["attributions"]["failed_deltas"],
                        "stages": stages, "feature_math_only": feature_math, "bandpower_components": bp})
    summary = {"diagnosed_windows": len(results), "stage_failures": {},
               "feature_math_only_failed_windows": sum(not r["feature_math_only"]["pass"] for r in results)}
    for name in results[0]["stages"]:
        subset = [r["stages"][name] for r in results]
        summary["stage_failures"][name] = {"any": sum(not r["pass"] for r in subset),
            "features": sum(not r["features"]["pass"] for r in subset),
            "probability": sum(not r["probability_pass"] for r in subset),
            "attributions": sum(not r["attributions"]["delta_pass"] for r in subset),
            "max_probability_error": max(r["probability_absolute_error"] for r in subset),
            "max_attribution_error": max(r["attributions"]["max_absolute_error"] for r in subset)}
    out.mkdir(parents=True)
    report = {"status": "per_window_diagnosis", "contract_sha256": CONTRACT_SHA256,
              "baseline_report_sha256": sha256(args.baseline / "report.json"), "script_sha256": sha256(__file__),
              "baseline_native_sha256": sha256(args.baseline / "native.npz"),
              "verified_source_sha256": {str(p.resolve()): sha256(p) for p in verified_sources.values()},
              "scope": "Each development failure is evaluated independently at every stage. FP64 is an offline control only. Stage errors may cancel; aggregate counts do not assign a universal cause.",
              "summary": summary, "windows": results}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Label-0-median explainer reference: host analysis (pre-registered, docs/xai-reference-label0-preregistration.md).

Host only, in-sample, functionally grounded only. No board, no firmware or header writes. Writes a
fresh outputs/xai-reference-label0-v1/ (refuses to overwrite). Imports from analyze_lamaakal_style
and analyze_physio_concept; edits neither.

M0  same-code checks, run first; any failure stops the script before a new number is written:
    1 v2 training-mean AOPC (las.aopc_run) reproduces d_aopc_v2_new_run.csv within 1e-8, and the
      reference-aware deletion code here with reference 0 equals las.aopc_run's deletion cells;
    2 P5 training-mean AOPC reproduces s3_aopc.csv within 1e-8;
    3 withhold / opposing shares under the float64 label-0 median reproduce s3_degeneracy.csv
      within 1e-12 (both models, both references);
    4 KAN_XAI_REF_Z (kan_xai_reference_v2.h) equals the float64 label-0 medians rounded to FP32.
M1  deletion AOPC under the label-0 reference (deleted inputs -> FP32 label-0 median; ranking =
    firmware-absolute of the label-0-reference deltas; random 5 repeats, seed 20260911).
M2  withhold-rule share under both references (FP32 reference as the device holds it).

  py -3.14 scripts/analyze_xai_reference_label0.py
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import analyze_lamaakal_style as las  # noqa: E402
import analyze_physio_concept as apc  # noqa: E402
import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402
from evaluate_xai_faithfulness import ranked_orders  # noqa: E402
from xai_reference import parse_reference_header  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "xai-reference-label0-v1"
P = {
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "v2_reference_header": ROOT / "firmware/Arduino Nano 33/include/kan_xai_reference_v2.h",
    "p5_model": ROOT / "outputs/physio-concept-v1/p5_final_model.npz",
    "physio_features": ROOT / "outputs/physio-concept-v1/physio_features.csv.gz",
    "s3": ROOT / "outputs/physio-concept-v1/s3.json",
    "s3_aopc": ROOT / "outputs/physio-concept-v1/s3_aopc.csv",
    "s3_degeneracy": ROOT / "outputs/physio-concept-v1/s3_degeneracy.csv",
    "las_aopc_v2": ROOT / "outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv",
    "preregistration": ROOT / "docs/xai-reference-label0-preregistration.md",
}
CODE = [Path(__file__), SCRIPTS / "analyze_lamaakal_style.py", SCRIPTS / "analyze_physio_concept.py",
        SCRIPTS / "lamaakal_models.py", SCRIPTS / "lamaakal_metrics.py", SCRIPTS / "evaluate_xai_faithfulness.py",
        SCRIPTS / "xai_reference.py"]
V16 = list(las.V2_FEATURES)
P5 = list(apc.P5)
THRESHOLD = 0.50
SUBSETS = ("all", "candidate_p_ge_0_50", "noncandidate_p_lt_0_50", "label_1", "label_0")
AOPC_COLS = ("deletion_aopc_window_mean", "deletion_gain_vs_random_mean", "fraction_better_than_own_random_mean")
TOL_AOPC, TOL_SHARE = 1e-8, 1e-12
REF_LABELS = {"training_mean": "training mean (device)", "label0_float64": "label-0 median (sensitivity, host only)"}


class StopAnalysis(RuntimeError):
    pass


def subset_masks(probability, y):
    return {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": probability >= THRESHOLD,
            "noncandidate_p_lt_0_50": probability < THRESHOLD, "label_1": y == 1, "label_0": y == 0}


def deletion_curves(predict, scaled, order, reference):
    """P(risk) at k = 0..d, the top-k inputs (fixed order) set to reference[i]; reference 0 = the existing curves."""
    rows, dimensions = scaled.shape
    deleted = np.array(scaled, dtype=np.float64, copy=True)
    curve = np.empty((rows, dimensions + 1))
    curve[:, 0] = predict(deleted)
    row = np.arange(rows)
    for step in range(dimensions):
        feature = order[:, step]
        deleted[row, feature] = reference[feature]
        curve[:, step + 1] = predict(deleted)
    return curve


def aopc_deletion(model, scaled, probability, attribution, y, reference):
    """las.aopc_run's deletion cells with deletion to `reference` (same orders, seed, repeats, subsets)."""
    orders = ranked_orders(attribution, seed=las.AOPC_SEED, random_repeats=las.AOPC_RANDOM_REPEATS)
    per_window = {}
    for name, order in orders.items():
        curve = deletion_curves(model.predict, scaled, order, reference)
        per_window[name] = np.mean(curve[:, :1] - curve[:, 1:], axis=1)
    randoms = [n for n in orders if n.startswith("random_")]
    per_window["random_mean"] = np.mean([per_window[n] for n in randoms], axis=0)
    rows = []
    for subset, mask in subset_masks(probability, y).items():
        for ranking in ("firmware_absolute", "random_mean", "signed_risk", "reverse_signed"):
            row = {"subset": subset, "ranking": ranking, "windows": int(mask.sum()),
                   "deletion_aopc_window_mean": float(np.mean(per_window[ranking][mask]))}
            if ranking != "random_mean":
                gain = per_window[ranking][mask] - per_window["random_mean"][mask]
                row["deletion_gain_vs_random_mean"] = float(np.mean(gain))
                row["fraction_better_than_own_random_mean"] = float(np.mean(gain > 1e-12))
            else:
                row["deletion_gain_vs_random_mean"] = float("nan")
                row["fraction_better_than_own_random_mean"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def max_cell_diff(reference, mine, columns, rankings=None):
    ref = reference if rankings is None else reference[reference.ranking.isin(rankings)]
    merged = ref.merge(mine, on=["subset", "ranking"], suffixes=("_ref", "_new"))
    if len(merged) != len(ref):
        return len(merged), float("inf")
    return len(merged), float(max(np.nanmax(np.abs(merged[f"{c}_ref"] - merged[f"{c}_new"])) for c in columns))


def withhold_rows(model_name, reference_name, probability, attribution, y):
    """Dashboard withhold rule (dashboard_core.js; analyze_physio_concept.degeneracy_rows) on the five subsets."""
    candidate = probability >= THRESHOLD
    lead = attribution[np.arange(len(attribution)), lm.firmware_abs_order(attribution)[:, 0]]
    opposes = np.where(candidate, lead < -las.ZERO_DELTA, lead > las.ZERO_DELTA)
    zero = ~(np.abs(lead) >= apc.EXPLANATION_ZERO)
    gate_opposes = ~zero & np.where(candidate, lead < 0, lead > 0)
    masks = {**subset_masks(probability, y), "candidate (P >= 0.50)": candidate, "non-candidate (P < 0.50)": ~candidate}
    return [{"model": model_name, "reference": reference_name, "subset": subset, "n": int(mask.sum()),
             "fraction_top1_opposes_decision": float(np.mean(opposes[mask])),
             "dashboard_withheld": float(np.mean((zero | gate_opposes)[mask])),
             "dashboard_withheld_zero": float(np.mean(zero[mask])),
             "dashboard_withheld_opposes": float(np.mean(gate_opposes[mask]))} for subset, mask in masks.items()]


def load_models():
    v2 = pd.read_csv(P["v2_features"])
    y = v2["label"].to_numpy(np.int64)
    if len(v2) != apc.N_ROWS:
        raise StopAnalysis("v2 table is not the 48,662-window export")
    model_v2 = models.HeaderKAN(P["v2_header"])
    scaled_v2 = model_v2.standardize(v2.loc[:, V16].to_numpy(np.float32))
    table = pd.read_csv(P["physio_features"])
    if not np.array_equal(table["row_index"], np.arange(apc.N_ROWS)):
        raise StopAnalysis("physio_features.csv.gz is not row_index-aligned")
    with np.load(P["p5_model"]) as z:
        arrays = {k: z[k] for k in z.files}
    if list(arrays["features"]) != P5:
        raise StopAnalysis("p5_final_model.npz features differ from analyze_physio_concept.P5")
    model_p5 = apc.ArrayKAN(arrays)
    x5 = apc.feature_matrix(v2, table, P5)
    raw_p5 = np.where(np.isfinite(x5), x5, model_p5.p["SCALER_MEAN"].astype(np.float32)[None, :]).astype(np.float32)
    scaled_p5 = model_p5.standardize(raw_p5)
    return y, {"v2": (model_v2, scaled_v2), "P5": (model_p5, scaled_p5)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error(f"{out} exists; refusing to overwrite")
    protected_before = {p.relative_to(ROOT).as_posix(): las.tree_hashes(p) for p in apc.PROTECTED}
    y, explained = load_models()
    s3 = json.loads(P["s3"].read_text(encoding="utf-8"))["headline"]["references"]
    checks, aopc_tables, withhold, refs = {}, [], [], {}

    # M0.4 and the references.
    tag, ref_v2_header = parse_reference_header(P["v2_reference_header"])
    for name, (model, scaled) in explained.items():
        ref64 = np.median(scaled[y == 0], axis=0)
        features = V16 if name == "v2" else P5
        s3_values = np.array([s3[name][f]["label0_median_z"] for f in features])
        refs[name] = {"float64": ref64, "fp32": ref64.astype(np.float32).astype(np.float64),
                      "s3_max_abs_diff": float(np.max(np.abs(ref64 - s3_values)))}
    checks["m0_4_reference"] = {
        "v2_header_tag": tag,
        "v2_header_equals_fp32_of_float64_median_bitwise": bool(np.array_equal(
            ref_v2_header.view(np.uint32), refs["v2"]["float64"].astype(np.float32).view(np.uint32))),
        "v2_float64_median_vs_s3_max_abs": refs["v2"]["s3_max_abs_diff"],
        "P5_float64_median_vs_s3_max_abs": refs["P5"]["s3_max_abs_diff"],
        "P5_reference_source": "float64 label-0 median of the Stage 3 standardized P5 inputs (equal to s3.json headline.references.P5), rounded to FP32"}
    checks["m0_4_reference"]["pass"] = bool(checks["m0_4_reference"]["v2_header_equals_fp32_of_float64_median_bitwise"]
                                            and refs["v2"]["s3_max_abs_diff"] <= 1e-9 and refs["P5"]["s3_max_abs_diff"] <= 1e-9)

    # M0.1-M0.3: training-mean cells and the float64 label-0 shares against the existing outputs.
    print("M0 same-code checks ...", flush=True)
    base = {}
    for name, (model, scaled) in explained.items():
        prob, attr_mean = model.mean_perturbation(scaled)
        base[name] = (prob, attr_mean)
        las_table = apc.aopc_table(name, las.aopc_run(model, scaled, prob, attr_mean, y), "")
        mine0 = aopc_deletion(model, scaled, prob, attr_mean, y, np.zeros(scaled.shape[1]))
        n0, d0 = max_cell_diff(las_table, mine0, AOPC_COLS)
        if name == "v2":
            n, d = max_cell_diff(pd.read_csv(P["las_aopc_v2"]), las_table, AOPC_COLS + ("insertion_aopc_window_mean",))
            source = "outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv"
        else:
            s3a = pd.read_csv(P["s3_aopc"])
            s3a = s3a[s3a.version == "P5"]
            n, d = max_cell_diff(s3a, las_table, AOPC_COLS + ("insertion_aopc_window_mean",))
            source = "outputs/physio-concept-v1/s3_aopc.csv"
        cell = lambda t, s, r: float(t[(t.subset == s) & (t.ranking == r)].deletion_aopc_window_mean.iloc[0])  # noqa: E731
        checks[f"m0_aopc_{name}"] = {"reference_file": source, "cells_compared": n, "max_abs_diff": d,
                                     "reference_deletion_code_vs_las_aopc_run_cells": n0,
                                     "reference_deletion_code_vs_las_aopc_run_max_abs_diff": d0, "tolerance": TOL_AOPC,
                                     "firmware_absolute_all": cell(las_table, "all", "firmware_absolute"),
                                     "random_mean_all": cell(las_table, "all", "random_mean"),
                                     "pass": bool(d <= TOL_AOPC and d0 <= 1e-12)}
        mine0 = mine0.assign(model=name, reference="training_mean")
        aopc_tables.append(mine0)
        _, attr64 = apc.perturb_against(model, scaled, refs[name]["float64"])
        deg = pd.read_csv(P["s3_degeneracy"])
        worst, count_mismatches = 0.0, 0
        for ref_name, attribution in (("training_mean", attr_mean), ("label0_float64", attr64)):
            rows = pd.DataFrame(withhold_rows(name, ref_name, prob, attribution, y))
            want = deg[(deg.version == name) & (deg.reference == REF_LABELS[ref_name])]
            merged = want.merge(rows, on="subset", suffixes=("_ref", "_new"))
            if len(merged) != 3 or not (merged["n_ref"] == merged["n_new"]).all():
                raise StopAnalysis(f"s3_degeneracy.csv subsets not found for {name} / {ref_name}")
            for c in ("fraction_top1_opposes_decision", "dashboard_withheld", "dashboard_withheld_zero", "dashboard_withheld_opposes"):
                worst = max(worst, float(np.max(np.abs(merged[f"{c}_ref"] - merged[f"{c}_new"]))))
                # Every share is k / n; the CSV holds it to 8 significant digits (las.write_csv "%.8g").
                count_mismatches += int(np.sum(np.rint(merged[f"{c}_ref"] * merged["n_ref"]) != np.rint(merged[f"{c}_new"] * merged["n_new"])))
            withhold += rows[rows.subset.isin(SUBSETS)].to_dict("records")
        checks[f"m0_withhold_{name}"] = {
            "reference_file": "outputs/physio-concept-v1/s3_degeneracy.csv", "max_abs_diff": worst,
            "preregistered_tolerance": TOL_SHARE, "window_count_mismatches": count_mismatches,
            "deviation": "The pre-registered 1e-12 cannot be met: s3_degeneracy.csv stores shares as %.8g (rounding up to ~5e-9). "
                         "Replaced by an exact check on the window counts behind each share (k = share x n, all 12 cells per model); "
                         "the float difference is reported and must stay below 1e-8 (the AOPC tolerance). First run stopped on this "
                         "check before any new number was written (attempt1_stopped.json).",
            "pass": bool(count_mismatches == 0 and worst <= TOL_AOPC)}
    print(json.dumps(checks, indent=1, default=las._json_default), flush=True)
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed:
        out.mkdir(parents=True)
        (out / "stopped.json").write_text(json.dumps(checks, indent=2, default=las._json_default) + "\n", encoding="utf-8")
        raise StopAnalysis(f"Same-code checks failed: {failed}; no new number written")

    # M1 and M2 under the FP32 label-0 reference.
    print("M1/M2 label-0 reference ...", flush=True)
    for name, (model, scaled) in explained.items():
        prob, _ = base[name]
        reference = refs[name]["fp32"]
        _, attr_ref = apc.perturb_against(model, scaled, reference)
        aopc_tables.append(aopc_deletion(model, scaled, prob, attr_ref, y, reference).assign(model=name, reference="label0_fp32"))
        withhold += [r for r in withhold_rows(name, "label0_fp32", prob, attr_ref, y) if r["subset"] in SUBSETS]

    aopc = pd.concat(aopc_tables, ignore_index=True)[["model", "reference", "subset", "ranking", "windows", *AOPC_COLS]]
    withhold_df = pd.DataFrame(withhold)
    readings = []
    for name in explained:
        for subset in SUBSETS:
            for ref_name in ("training_mean", "label0_fp32"):
                block = aopc[(aopc.model == name) & (aopc.reference == ref_name) & (aopc.subset == subset)]
                fw = block[block.ranking == "firmware_absolute"].iloc[0]
                rnd = float(block[block.ranking == "random_mean"].deletion_aopc_window_mean.iloc[0])
                role = "primary" if (name, subset) == ("v2", "candidate_p_ge_0_50") else (
                    "secondary" if (name, subset) == ("v2", "all") else "descriptive")
                readings.append({"model": name, "subset": subset, "reference": ref_name, "role": role,
                                 "firmware_absolute": float(fw.deletion_aopc_window_mean), "random_mean": rnd,
                                 "fraction_better_than_own_random": float(fw.fraction_better_than_own_random_mean),
                                 "R1_firmware_absolute_beats_random": bool(fw.deletion_aopc_window_mean > rnd)})
    readings_df = pd.DataFrame(readings)
    fp = withhold_df[withhold_df.reference == "label0_fp32"].set_index(["model", "subset"])
    f64 = withhold_df[withhold_df.reference == "label0_float64"].set_index(["model", "subset"])
    r3 = (fp[["dashboard_withheld", "fraction_top1_opposes_decision"]] - f64[["dashboard_withheld", "fraction_top1_opposes_decision"]]).abs()
    checks["r3_fp32_vs_float64_withhold_max_abs"] = float(r3.to_numpy().max())

    out.mkdir(parents=True)
    las.write_csv(aopc, out / "aopc_deletion.csv")
    las.write_csv(withhold_df, out / "withhold.csv")
    las.write_csv(readings_df, out / "readings.csv")
    protected_after = {p.relative_to(ROOT).as_posix(): las.tree_hashes(p) for p in apc.PROTECTED}
    if protected_after != protected_before:
        raise StopAnalysis("A protected input or write-once directory changed during the run")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Host only, in-sample, functionally grounded only. Not board evidence.",
        "preregistration": "docs/xai-reference-label0-preregistration.md (commit d690ee5, before any number here)",
        "checks": checks, "references": {name: {"fp32": r["fp32"].tolist(), "float64": r["float64"].tolist(),
                                                "features": V16 if name == "v2" else P5} for name, r in refs.items()},
        "readings": readings, "withhold": withhold,
        "inputs_sha256": {k: las.sha256(v) for k, v in P.items()},
        "code_sha256": {p.relative_to(ROOT).as_posix(): las.sha256(p) for p in CODE},
        "protected_unchanged": True,
        "environment": apc.environment(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=las._json_default) + "\n", encoding="utf-8")
    print(readings_df.to_string())
    print(withhold_df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

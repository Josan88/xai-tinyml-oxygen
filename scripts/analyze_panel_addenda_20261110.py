"""Panel 10 Nov 2026 addenda: host analyses A1-A4 (pre-registered, docs/panel-2026-11-10-addenda-preregistration.md).

Host only. No board, no firmware or header writes. Writes a fresh outputs/panel-2026-11-10-addenda-v1/
(refuses to overwrite). Imports from the existing analysis modules and edits none.

A1  patient-bootstrap interval of the in-sample deletion-AOPC gain (firmware-absolute minus random), both references
A2  the same on clamped and unclamped candidates; zero-delta shares
A3  out-of-fold counterpart with five recording-fold analysis models
A4  RQ1 metrics with labels taken L seconds later or earlier (whole strides)

  py -3.14 scripts/analyze_panel_addenda_20261110.py
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
from analyze_xai_reference_label0 import deletion_curves  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "panel-2026-11-10-addenda-v1"
P = {
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "label0_aopc": ROOT / "outputs/xai-reference-label0-v1/aopc_deletion.csv",
    "preregistration": ROOT / "docs/panel-2026-11-10-addenda-preregistration.md",
}
V16 = list(las.V2_FEATURES)
THRESHOLD = 0.50
N_BOOT = 2000
BOOT_SEED = 20261110
CLAMP = 3.0
EXPECTED_CANDIDATES, EXPECTED_GRADIENT_ZERO = 28661, 11381
LAGS = (-30, -20, -15, -10, -5, 5, 10, 15, 20, 30)
TOL_AOPC, TOL_OOF = 1e-8, 1e-5


class StopAnalysis(RuntimeError):
    pass


def per_window_aopc(model, scaled, attribution, reference):
    """Per-window deletion AOPC for the firmware-absolute ranking and the mean of the five random orders."""
    orders = ranked_orders(attribution, seed=las.AOPC_SEED, random_repeats=las.AOPC_RANDOM_REPEATS)
    value = {}
    for name, order in orders.items():
        if name == "firmware_absolute" or name.startswith("random_"):
            curve = deletion_curves(model.predict, scaled, order, reference)
            value[name] = np.mean(curve[:, :1] - curve[:, 1:], axis=1)
    random_mean = np.mean([value[n] for n in value if n.startswith("random_")], axis=0)
    return value["firmware_absolute"], random_mean


def bootstrap_mean(values, mask, patient_index, counts):
    """Point mean of values[mask] and its patient-bootstrap 95 % interval (counts: B x n_patients multiplicities)."""
    n_patients = counts.shape[1]
    sums = np.bincount(patient_index[mask], weights=values[mask], minlength=n_patients)
    sizes = np.bincount(patient_index[mask], minlength=n_patients).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ sums) / (counts @ sizes)
    lo, hi = lm.percentile_ci(draws)
    return float(values[mask].mean()), lo, hi


def reading(lo, hi):
    return "beats random" if lo > 0 else ("worse than random" if hi < 0 else "not distinguishable from random")


def gain_rows(tag, reference_name, firmware, random_mean, masks, patient_index, n_patients, cell):
    rows = []
    for subset, mask in masks.items():
        counts = lm.group_bootstrap_counts(patient_index, n_patients, N_BOOT, BOOT_SEED + cell[0])
        cell[0] += 1
        gain = firmware - random_mean
        point, lo, hi = bootstrap_mean(gain, mask, patient_index, counts)
        _, f_lo, f_hi = bootstrap_mean(firmware, mask, patient_index, counts)
        rows.append({"analysis": tag, "reference": reference_name, "subset": subset, "windows": int(mask.sum()),
                     "firmware_absolute": float(firmware[mask].mean()), "firmware_ci_lo": f_lo, "firmware_ci_hi": f_hi,
                     "random_mean": float(random_mean[mask].mean()),
                     "gain": point, "gain_ci_lo": lo, "gain_ci_hi": hi,
                     "fraction_better_than_own_random": float(np.mean(gain[mask] > 1e-12)),
                     "reading": reading(lo, hi)})
    return rows


def lag_pairs(recording, start, lag):
    """Index pairs (i, j): window j is in the same recording and starts `lag` seconds after window i."""
    key = pd.MultiIndex.from_arrays([recording, np.round(start, 3)])
    lookup = pd.Series(np.arange(len(start)), index=key)
    if not lookup.index.is_unique:
        raise StopAnalysis("duplicate (recording, start time) windows")
    target = pd.MultiIndex.from_arrays([recording, np.round(start + lag, 3)])
    j = lookup.reindex(target).to_numpy()
    keep = np.isfinite(j)
    return np.flatnonzero(keep), j[keep].astype(int)


def rq1_metrics(y, p):
    m = lm.binary_metrics(y, p, THRESHOLD)
    return m["auc"], m["recall"], m["specificity"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error(f"{out} exists; refusing to overwrite")
    protected_before = apc.protected_hashes()

    v2 = pd.read_csv(P["v2_features"])
    oof = pd.read_csv(P["oof"])
    if len(v2) != 48662 or not np.array_equal(oof["row_index"], np.arange(len(v2))):
        raise StopAnalysis("inputs are not the row-aligned 48,662-window export")
    y = v2["label"].to_numpy(np.int64)
    raw = v2.loc[:, V16].to_numpy(np.float32)
    _, patient_index = np.unique(v2["patient_id"].astype(str).to_numpy(), return_inverse=True)
    n_patients = int(patient_index.max()) + 1
    cell = [0]
    checks, rows = {}, []

    # ---- In-sample, shipped model (A1, A2) ----
    model = models.HeaderKAN(P["v2_header"])
    scaled = model.standardize(raw)
    prob, attr_mean = model.mean_perturbation(scaled)
    ref0 = np.median(scaled[y == 0], axis=0).astype(np.float32).astype(np.float64)
    _, attr_ref0 = apc.perturb_against(model, scaled, ref0)
    in_sample = {"training_mean": per_window_aopc(model, scaled, attr_mean, np.zeros(16)),
                 "label0_fp32": per_window_aopc(model, scaled, attr_ref0, ref0)}
    candidate = prob >= THRESHOLD
    clamped = model.output_preactivation(scaled) >= CLAMP
    subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": candidate, "noncandidate_p_lt_0_50": ~candidate,
               "label_1": y == 1, "label_0": y == 0}

    stored = pd.read_csv(P["label0_aopc"])
    stored = stored[stored.model == "v2"]
    worst = 0.0
    for reference_name, (firmware, random_mean) in in_sample.items():
        for subset, mask in subsets.items():
            for ranking, values in (("firmware_absolute", firmware), ("random_mean", random_mean)):
                want = stored[(stored.reference == reference_name) & (stored.subset == subset) & (stored.ranking == ranking)]
                if len(want) != 1:
                    raise StopAnalysis(f"stored AOPC cell missing: {reference_name}/{subset}/{ranking}")
                worst = max(worst, abs(float(want.deletion_aopc_window_mean.iloc[0]) - float(values[mask].mean())))
    checks["m0_1_reproduces_label0_aopc_csv"] = {"cells": 20, "max_abs_diff": worst, "tolerance": TOL_AOPC, "pass": bool(worst <= TOL_AOPC)}
    # Amendment A1: 11,381 is the gradient-all-zero count (output clamp OR every hidden unit clamped), not the output-clamp count.
    all_hidden_clamped = (np.abs(model.hidden_preactivation(scaled)) >= CLAMP).all(axis=1)
    gradient_zero = clamped | (model.output_preactivation(scaled) <= -CLAMP) | all_hidden_clamped
    checks["m0_2_clamp_count"] = {"candidates": int(candidate.sum()), "output_clamped_candidates": int((candidate & clamped).sum()),
                                  "gradient_zero_candidates": int((candidate & gradient_zero).sum()),
                                  "expected_candidates_and_gradient_zero": [EXPECTED_CANDIDATES, EXPECTED_GRADIENT_ZERO],
                                  "pass": bool(candidate.sum() == EXPECTED_CANDIDATES
                                               and (candidate & gradient_zero).sum() == EXPECTED_GRADIENT_ZERO)}
    print(json.dumps(checks, indent=1), flush=True)
    if not all(c["pass"] for c in checks.values()):
        raise StopAnalysis("M0 in-sample checks failed; no new number written")

    # ---- A3 fold models; M0.3 before anything is written ----
    import torch
    from sklearn.preprocessing import StandardScaler
    import train_tinykan_v2 as t2

    folds = oof["fold"].to_numpy()
    oof_sample = {name: (np.full(len(y), np.nan), np.full(len(y), np.nan)) for name in in_sample}
    oof_prob = np.full(len(y), np.nan)
    oof_torch = np.full(len(y), np.nan)
    torch_diff = 0.0
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        scaler = StandardScaler()
        x_train = scaler.fit_transform(raw[~test]).astype(np.float32)
        x_test = scaler.transform(raw[test]).astype(np.float32)
        net = t2.train_kan(x_train, y[~test])
        oof_torch[test] = t2.kan_probabilities(net, x_test)
        torch_diff = max(torch_diff, float(np.max(np.abs(oof_torch[test] - oof.loc[test, "y_prob"].to_numpy()))))
        fold_model = apc.ArrayKAN({"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_, **apc.kan_params(net)})
        z_train, z_test = fold_model.standardize(raw[~test]), fold_model.standardize(raw[test])
        p_test, a_mean = fold_model.mean_perturbation(z_test)
        fold_ref0 = np.median(z_train[y[~test] == 0], axis=0).astype(np.float32).astype(np.float64)
        _, a_ref0 = apc.perturb_against(fold_model, z_test, fold_ref0)
        oof_prob[test] = p_test
        for name, attribution, reference in (("training_mean", a_mean, np.zeros(16)), ("label0_fp32", a_ref0, fold_ref0)):
            firmware, random_mean = per_window_aopc(fold_model, z_test, attribution, reference)
            oof_sample[name][0][test], oof_sample[name][1][test] = firmware, random_mean
        print(f"fold {fold}: held-out {int(test.sum())}, torch diff so far {torch_diff:.2e}", flush=True)
    # Amendment A2: v2 retraining is documented as not bit-reproducible (docs/analysis-lamaakal-style.md, Provenance item 5:
    # max |dP| 0.041, 17 candidate flags, AUC -1.3e-4). The check is that the fold models stay inside that documented envelope.
    stored_p = oof["y_prob"].to_numpy(float)
    torch_flags = int(np.sum((oof_torch >= THRESHOLD) != (stored_p >= THRESHOLD)))
    auc_diff = float(lm.weighted_auc(y, oof_torch) - lm.weighted_auc(y, stored_p))
    checks["m0_3_fold_models_reproduce_stored_oof"] = {
        "max_abs_probability_diff": torch_diff, "candidate_flag_differences": torch_flags, "auc_diff": auc_diff,
        "preregistered_tolerance": TOL_OOF, "amended_envelope": {"max_abs_probability_diff": 0.05, "candidate_flag_differences": 50, "abs_auc_diff": 1e-3},
        "pass": bool(torch_diff <= 0.05 and torch_flags <= 50 and abs(auc_diff) <= 1e-3)}
    print(json.dumps(checks["m0_3_fold_models_reproduce_stored_oof"], indent=1), flush=True)
    if not checks["m0_3_fold_models_reproduce_stored_oof"]["pass"]:
        raise StopAnalysis(f"M0.3 failed (max diff {torch_diff:.3e}); no new number written")

    # ---- A1, A2 ----
    for reference_name, (firmware, random_mean) in in_sample.items():
        rows += gain_rows("A1_in_sample_shipped", reference_name, firmware, random_mean, subsets, patient_index, n_patients, cell)
        strata = {"candidate_clamped": candidate & clamped, "candidate_unclamped": candidate & ~clamped}
        rows += gain_rows("A2_in_sample_shipped", reference_name, firmware, random_mean, strata, patient_index, n_patients, cell)
    max_abs = np.max(np.abs(attr_mean), axis=1)
    a2_shares = {"candidates": int(candidate.sum()), "clamped_candidates": int((candidate & clamped).sum()),
                 "training_mean_all_16_deltas_exact_zero": float(np.mean(max_abs[candidate] <= las.ZERO_DELTA)),
                 "training_mean_max_abs_delta_below_0_01": float(np.mean(max_abs[candidate] < 0.01)),
                 "training_mean_all_zero_among_clamped": float(np.mean(max_abs[candidate & clamped] <= las.ZERO_DELTA)),
                 "training_mean_max_abs_delta_below_0_01_among_unclamped": float(np.mean(max_abs[candidate & ~clamped] < 0.01))}

    # ---- A3 ----
    oof_candidate = oof_prob >= THRESHOLD
    oof_subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": oof_candidate, "noncandidate_p_lt_0_50": ~oof_candidate,
                   "label_1": y == 1, "label_0": y == 0}
    for reference_name, (firmware, random_mean) in oof_sample.items():
        rows += gain_rows("A3_out_of_fold_analysis_models", reference_name, firmware, random_mean, oof_subsets, patient_index, n_patients, cell)
    a3_note = {"held_out_candidates": int(oof_candidate.sum()),
               "arraykan_vs_stored_oof_max_abs": float(np.max(np.abs(oof_prob - oof["y_prob"].to_numpy()))),
               "candidate_flag_differences_vs_stored_oof": int(np.sum(oof_candidate != (oof["y_prob"].to_numpy() >= THRESHOLD)))}

    # ---- A4 ----
    recording = v2["recording_id"].astype(str).to_numpy()
    start = v2["start_time_seconds"].to_numpy(float)
    p_oof = oof["y_prob"].to_numpy(float)
    lag_rows = []
    for lag in LAGS:
        i, j = lag_pairs(recording, start, float(lag))
        auc0, rec0, spec0 = rq1_metrics(y[i], p_oof[i])
        auc1, rec1, spec1 = rq1_metrics(y[j], p_oof[i])
        point, draws = lm.group_bootstrap(y[i], {"p": p_oof[i]}, patient_index[i], N_BOOT, BOOT_SEED + cell[0])
        point_l, draws_l = lm.group_bootstrap(y[j], {"p": p_oof[i]}, patient_index[i], N_BOOT, BOOT_SEED + cell[0])
        cell[0] += 1
        row = {"lag_s": lag, "paired_windows": int(len(i)), "label_changed_share": float(np.mean(y[i] != y[j])),
               "auc_lag0": auc0, "auc_lagged": auc1, "recall_lag0": rec0, "recall_lagged": rec1,
               "specificity_lag0": spec0, "specificity_lagged": spec1}
        for metric in ("auc", "recall", "specificity"):
            lo, hi = lm.percentile_ci(draws_l["p"][metric] - draws["p"][metric])
            row[f"{metric}_diff"] = float(point_l["p"][metric] - point["p"][metric])
            row[f"{metric}_diff_ci_lo"], row[f"{metric}_diff_ci_hi"] = lo, hi
        lag_rows.append(row)

    table = pd.DataFrame(rows)
    lag_table = pd.DataFrame(lag_rows)
    if apc.protected_hashes() != protected_before:
        raise StopAnalysis("A protected input or write-once directory changed during the run")
    out.mkdir(parents=True)
    las.write_csv(table, out / "aopc_gain_intervals.csv")
    las.write_csv(lag_table, out / "label_lag_sensitivity.csv")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Host only. A1/A2 in-sample on the shipped header; A3 analysis models; A4 stored out-of-fold probabilities. Not board evidence.",
        "preregistration": "docs/panel-2026-11-10-addenda-preregistration.md (written before the run; not committed before it)",
        "bootstrap": {"unit": "patient", "patients": n_patients, "resamples": N_BOOT, "seed": BOOT_SEED, "interval": "95 % percentile"},
        "checks": checks, "a2_shares": a2_shares, "a3_note": a3_note,
        "aopc_gain_intervals": rows, "label_lag_sensitivity": lag_rows,
        "inputs_sha256": {k: las.sha256(v) for k, v in P.items()},
        "code_sha256": {Path(__file__).relative_to(ROOT).as_posix(): las.sha256(Path(__file__))},
        "protected_unchanged": True, "environment": apc.environment(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=las._json_default) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.to_string())
    print(json.dumps(a2_shares, indent=1))
    print(json.dumps(a3_note, indent=1))
    print(lag_table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

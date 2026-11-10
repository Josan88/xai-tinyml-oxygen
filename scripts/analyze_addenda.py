"""Two host-only addenda to the Lamaakal-style analysis (no board, no header writes).

Analysis 1  Patient-grouped evaluation of the v1 (baseline) TinyKAN.
            Reproduction check first: seed 42 on the existing recording-grouped folds must
            reproduce the frozen notebook-12 out-of-fold predictions at 0.50. Then
            GroupKFold(5) on patient_id over the same 52,090 v1 windows (labels as coded),
            v1 TinyKAN (V1_RECIPE, seeds 42, 1-5) and v1 LogReg-14, patient bootstrap CI.
Analysis 2  Logistic regression on the explanation-fidelity axis, plus cost facts.
            Same-code check first: the shipped v1 and v2 TinyKAN headers must reproduce the
            write-once v1 and the v2 deletion AOPC through this code path. Then an all-window
            LogReg (no exported one exists in the repository; analysis model) is explained
            with the same 1 + n mean-perturbation forwards and scored with the same AOPC.

Every retrained model here is an analysis model only. Nothing writes a weights header,
firmware, or an existing outputs/ directory; the shipped v1/v2 models, threshold 0.50, N=2,
labels and inputs are untouched. Imports, but never modifies, analyze_lamaakal_style.py and
lamaakal_models.py.

Run from the repository root with the interpreter that has torch 2.12 / sklearn 1.9:
  C:\\Python314\\python.exe scripts/analyze_addenda.py
  C:\\Python314\\python.exe scripts/analyze_addenda.py --output build/addenda-rerun
  C:\\Python314\\python.exe scripts/analyze_addenda.py --compare build/addenda-rerun
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_lamaakal_style as las  # noqa: E402
import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "analysis-addenda-v1"
LAS_DIR = ROOT / "outputs" / "analysis-lamaakal-style-v1"

P = {
    **las.P,
    "export_audit": ROOT / "outputs/openox_canonical_export_audit.csv",
    "encounter_coverage": ROOT / "outputs/openox_encounter_coverage.csv",
    "las_aopc_v1_cited": LAS_DIR / "d_aopc_v1_cited.csv",
    "las_aopc_v2_new_run": LAS_DIR / "d_aopc_v2_new_run.csv",
    "las_seed_runs": LAS_DIR / "c_seed_runs.csv",
    "las_seed_summary": LAS_DIR / "c_seed_summary.csv",
    "las_bootstrap_ci": LAS_DIR / "c_bootstrap_ci.csv",
    "las_deployment": LAS_DIR / "f_deployment.csv",
    "las_summary": LAS_DIR / "summary.json",
    "analyze_lamaakal_style_script": SCRIPTS / "analyze_lamaakal_style.py",
    "lamaakal_models_script": SCRIPTS / "lamaakal_models.py",
    "lamaakal_metrics_script": SCRIPTS / "lamaakal_metrics.py",
}
WRITE_ONCE = las.WRITE_ONCE + (LAS_DIR, SCRIPTS / "analyze_lamaakal_style.py", SCRIPTS / "lamaakal_models.py")

THRESHOLD = las.THRESHOLD
SEEDS = las.SEEDS
N_SPLITS = 5
N_BOOT = las.N_BOOT
# Section C seeds the bootstrap as BOOTSTRAP_SEED + comparison index (0 v1 recording,
# 1 v2 recording, 2 v2 patient = 20261026). v1 patient is the next comparison: index 3.
BOOTSTRAP_SEED_V1_PATIENT = las.BOOTSTRAP_SEED + 3

# Stated in the paper (Table I and the AOPC paragraph); checked, never used as inputs.
PAPER = {
    "v1_recording_tinykan": {"auc": 0.606, "recall": 0.767, "specificity": 0.394},
    "v1_recording_logreg": {"recall": 0.866, "specificity": 0.261},
    "v1_aopc_firmware_absolute_all": 0.0443, "v1_aopc_random_all": 0.0283,
    "v2_aopc_firmware_absolute_all": -0.3092, "v2_aopc_random_all": -0.1612,
}
REPRO_BRIEF = {"auc": 0.6063, "recall": 0.7668, "specificity": 0.3941}
AOPC_TOLERANCE = 1e-8  # same arithmetic, same orders; the cited CSVs print 8 significant digits, so round-off is up to ~5e-9


class StopAnalysis(RuntimeError):
    """A reproduction check failed; nothing is written."""


# ----------------------------------------------------------------------------- data

def v1_patient_ids(v1):
    """recording_id -> export audit encounter_id -> encounter coverage patient_id (one-to-one per recording)."""
    audit = pd.read_csv(P["export_audit"], dtype=str)
    coverage = pd.read_csv(P["encounter_coverage"], dtype=str)
    if audit["recording_id"].duplicated().any() or coverage["encounter_id"].duplicated().any():
        raise RuntimeError("Join keys are not unique")
    encounter = v1["recording_id"].map(audit.set_index("recording_id")["encounter_id"])
    patient = encounter.map(coverage.set_index("encounter_id")["patient_id"])
    if encounter.isna().any() or patient.isna().any() or (patient == "").any():
        raise RuntimeError("A v1 window has no encounter_id or patient_id")
    if (encounter != v1["recording_id"]).any():
        raise RuntimeError("recording_id and encounter_id differ; the documented one-to-one join does not hold")
    return patient.to_numpy()


def group_folds(n, groups):
    """GroupKFold(5), folds numbered 1..5, exactly as train_tinykan_v2.out_of_fold builds them."""
    from sklearn.model_selection import GroupKFold

    folds = np.zeros(n, dtype=int)
    dummy = np.zeros((n, 1))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=N_SPLITS).split(dummy, groups=groups), start=1):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Group leakage between train and test")
        folds[test] = fold
    if (folds == 0).any() or pd.Series(groups).groupby(folds).nunique().sum() != pd.Series(groups).nunique():
        raise RuntimeError("Fold assignment incomplete or a group spans folds")
    return folds


def fold_composition(version, grouping, folds, y, recordings, patients):
    rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        rows.append({"version": version, "grouping": grouping, "fold": int(fold), "windows": int(mask.sum()),
                     "patients": int(pd.Series(patients[mask]).nunique()),
                     "recordings": int(pd.Series(recordings[mask]).nunique()),
                     "prevalence": float(y[mask].mean())})
    return rows


# ----------------------------------------------------------------------------- analysis 1

def metric_row(label, grouping, model, seed, source, y, p):
    m = lm.binary_metrics(y, p, THRESHOLD)
    return {"version": "v1", "grouping": grouping, "model": model, "seed": seed, "source": source,
            "n_windows": m["n"], "prevalence": m["prevalence"], "auc": m["auc"], "pr_auc": m["pr_auc"],
            "recall": m["recall"], "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
            "candidate_rate": m["candidate_rate"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "label": label}


def reproduction_check(ctx, results):
    y = ctx["y1"]
    frozen_kan, frozen_lr = ctx["p_v1_tinykan"], ctx["p_v1_logreg"]
    p42, plr = results["v1rec_tinykan_seed42"]["probabilities"], results["v1rec_logreg"]["probabilities"]
    rows = [metric_row("frozen", "recording", "TinyKAN", 42, "openox_tinykan_predictions.csv y_prob, recomputed at 0.50 (stored y_pred is the 0.30 cut, ignored)", y, frozen_kan),
            metric_row("retrain", "recording", "TinyKAN", 42, "analysis retrain here, V1_RECIPE (80 epochs), split_index folds", y, p42),
            metric_row("frozen", "recording", "LogReg-14", None, "openox_logreg_local_explanations.csv (notebook 11, same folds)", y, frozen_lr),
            metric_row("retrain", "recording", "LogReg-14", None, "analysis retrain here, BASELINE_SETTINGS['logreg'], split_index folds", y, plr)]
    table = pd.DataFrame(rows)
    fk, rk, fl, rl = (rows[i] for i in range(4))
    check = {
        "tinykan_frozen_at_0_50": {k: fk[k] for k in ("auc", "recall", "specificity")},
        "tinykan_retrain_seed42_at_0_50": {k: rk[k] for k in ("auc", "recall", "specificity")},
        "tinykan_max_abs_probability_diff": float(np.max(np.abs(p42 - frozen_kan))),
        "tinykan_candidate_flag_disagreements_at_0_50": int(np.sum((p42 >= THRESHOLD) != (frozen_kan >= THRESHOLD))),
        "logreg_frozen_at_0_50": {k: fl[k] for k in ("auc", "recall", "specificity")},
        "logreg_retrain_at_0_50": {k: rl[k] for k in ("auc", "recall", "specificity")},
        "logreg_max_abs_probability_diff": float(np.max(np.abs(plr - frozen_lr))),
        "logreg_candidate_flag_disagreements_at_0_50": int(np.sum((plr >= THRESHOLD) != (frozen_lr >= THRESHOLD))),
        "brief_targets_4dp": REPRO_BRIEF,
        # LogReg uses the same criterion as TinyKAN (flags identical at 0.50, AUC within 1e-4); the first run's
        # 1e-6 probability bound was stricter than the TinyKAN check and failed at 1.3e-5 with identical metrics.
    }
    ok = (check["tinykan_candidate_flag_disagreements_at_0_50"] == 0
          and abs(rk["auc"] - fk["auc"]) < 1e-4
          and all(round(rk[k], 4) == v for k, v in REPRO_BRIEF.items())
          and all(round(fk[k], 4) == v for k, v in REPRO_BRIEF.items())
          and check["logreg_candidate_flag_disagreements_at_0_50"] == 0
          and abs(rl["auc"] - fl["auc"]) < 1e-4)
    check["reproduces"] = bool(ok)
    if not ok:
        raise StopAnalysis(f"Recording-fold reproduction failed: {json.dumps(check, indent=2)}")
    return table, check


def analysis_1(ctx, results):
    y = ctx["y1"]
    patients = ctx["patient1"]
    folds_p = ctx["folds1p"]
    out = ctx["out"]
    head = {}

    # Fold composition: v1 recording, v1 patient, v2 patient (reference).
    comp = []
    comp += fold_composition("v1", "recording (split_index, notebook 12)", ctx["folds1"], y, ctx["v1"]["recording_id"].to_numpy(), patients)
    comp += fold_composition("v1", "patient (GroupKFold(5) on patient_id, this run)", folds_p, y, ctx["v1"]["recording_id"].to_numpy(), patients)
    comp += fold_composition("v2", "patient (predictions_b_v2_tinykan16_patient.csv fold)", ctx["folds2p"], ctx["y2"],
                             ctx["v2"]["recording_id"].to_numpy(), ctx["v2"]["patient_id"].to_numpy())
    las.write_csv(pd.DataFrame(comp), out / "a1_fold_composition.csv")

    # Seed runs on patient folds.
    rows = []
    for seed in SEEDS:
        rows.append(metric_row("patient", "patient", "TinyKAN", seed, "analysis retrain, V1_RECIPE (80 epochs), patient folds", y,
                               results[f"v1patient_tinykan_seed{seed}"]["probabilities"]))
    rows.append(metric_row("patient", "patient", "LogReg-14", None, "analysis retrain, BASELINE_SETTINGS['logreg'], patient folds", y,
                           results["v1patient_logreg"]["probabilities"]))
    runs = pd.DataFrame(rows).drop(columns=["label"])
    las.write_csv(runs, out / "a1_patient_grouped_runs.csv")
    kan = runs[runs.model == "TinyKAN"]
    seed_summary = {"seeds": [int(s) for s in kan.seed]}
    for metric in ("auc", "pr_auc", "recall", "specificity", "balanced_accuracy", "candidate_rate"):
        mean, sd = lm.mean_sd(kan[metric])
        seed_summary[metric] = {"mean": mean, "sd": sd, "min": float(kan[metric].min()), "max": float(kan[metric].max())}
    las.write_csv(pd.DataFrame([{"version": "v1", "grouping": "patient", "model": "TinyKAN (analysis retrain)",
                                 "seeds": " ".join(str(s) for s in SEEDS), "n_seeds": len(SEEDS),
                                 **{f"{m}_{k}": v[k] for m, v in seed_summary.items() if m != "seeds" for k in ("mean", "sd", "min", "max")}}]),
                  out / "a1_patient_seed_summary.csv")
    head["v1_patient_tinykan_seed_spread"] = seed_summary

    # Patient bootstrap, paired TinyKAN (seed 42) vs LogReg-14, section C convention.
    probs = {"TinyKAN": results["v1patient_tinykan_seed42"]["probabilities"], "LogReg": results["v1patient_logreg"]["probabilities"]}
    point, draws = lm.group_bootstrap(y, probs, patients, N_BOOT, BOOTSTRAP_SEED_V1_PATIENT, THRESHOLD)
    ci_rows = []
    for model in probs:
        for metric in ("auc", "recall", "specificity"):
            low, high = lm.percentile_ci(draws[model][metric])
            ci_rows.append({"version": "v1", "grouping": "patient", "resampled_unit": "patient", "model": model, "metric": metric,
                            "point": point[model][metric], "ci_low": low, "ci_high": high, "n_boot": N_BOOT,
                            "seed": BOOTSTRAP_SEED_V1_PATIENT, "n_units": int(pd.Series(patients).nunique())})
    paired = []
    for metric in ("auc", "recall", "specificity"):
        delta = draws["TinyKAN"][metric] - draws["LogReg"][metric]
        low, high = lm.percentile_ci(delta)
        paired.append({"version": "v1", "grouping": "patient", "comparison": "TinyKAN - LogReg", "metric": metric,
                       "method": f"paired patient bootstrap ({N_BOOT} resamples, 95% percentile CI)",
                       "difference": point["TinyKAN"][metric] - point["LogReg"][metric], "ci_low": low, "ci_high": high,
                       "fraction_resamples_le_0": float(np.mean(delta <= 0)), "seed": BOOTSTRAP_SEED_V1_PATIENT})
    las.write_csv(pd.DataFrame(ci_rows), out / "a1_patient_bootstrap_ci.csv")
    las.write_csv(pd.DataFrame(paired), out / "a1_patient_paired_difference.csv")

    # Side by side with the cited recording-grouped v1 and patient-grouped v2 (analysis-lamaakal-style-v1).
    cited_ci = pd.read_csv(P["las_bootstrap_ci"])
    cited_seed = pd.read_csv(P["las_seed_summary"])
    side = []
    pr = {"TinyKAN": lm.binary_metrics(y, probs["TinyKAN"], THRESHOLD)["pr_auc"], "LogReg": lm.binary_metrics(y, probs["LogReg"], THRESHOLD)["pr_auc"]}
    for (version, grouping), source in ((("v1", "recording"), "cited: analysis-lamaakal-style-v1/c_bootstrap_ci.csv"),
                                        (("v1", "patient"), "this run"),
                                        (("v2", "patient"), "cited: analysis-lamaakal-style-v1/c_bootstrap_ci.csv")):
        table = pd.DataFrame(ci_rows) if source == "this run" else cited_ci[(cited_ci.version == version) & (cited_ci.grouping == grouping)]
        for model in ("TinyKAN", "LogReg"):
            block = table[table.model == model].set_index("metric")
            row = {"version": version, "grouping": grouping, "model": model, "source": source,
                   "n_units": int(block["n_units"].iloc[0]), "bootstrap_seed": int(block["seed"].iloc[0])}
            for metric in ("auc", "recall", "specificity"):
                row[metric] = float(block.loc[metric, "point"])
                row[f"{metric}_ci_low"] = float(block.loc[metric, "ci_low"])
                row[f"{metric}_ci_high"] = float(block.loc[metric, "ci_high"])
            row["pr_auc"] = pr[model] if source == "this run" else float("nan")
            side.append(row)
    las.write_csv(pd.DataFrame(side), out / "a1_grouping_comparison.csv")

    v1_rec_seed = cited_seed[cited_seed.version == "v1"].iloc[0]
    head["v1_patient_tinykan_seed42"] = {k: point["TinyKAN"][k] for k in ("auc", "recall", "specificity")} | {"pr_auc": pr["TinyKAN"]}
    head["v1_patient_logreg"] = {k: point["LogReg"][k] for k in ("auc", "recall", "specificity")} | {"pr_auc": pr["LogReg"]}
    head["v1_patient_ci"] = {f"{r['model']}_{r['metric']}": [r["ci_low"], r["ci_high"]] for r in ci_rows}
    head["v1_patient_paired_difference"] = {r["metric"]: {"difference": r["difference"], "ci": [r["ci_low"], r["ci_high"]],
                                                         "fraction_resamples_le_0": r["fraction_resamples_le_0"]} for r in paired}
    head["v1_recording_tinykan_seed_spread_cited"] = {m: [float(v1_rec_seed[f"{m}_mean"]), float(v1_rec_seed[f"{m}_sd"])]
                                                      for m in ("auc", "recall", "specificity")}

    # Patients split across folds under the recording grouping (why the change can matter).
    rec_spread = pd.Series(ctx["folds1"]).groupby(patients).nunique()
    head["v1_patients"] = int(len(rec_spread))
    head["v1_recordings"] = int(ctx["v1"]["recording_id"].nunique())
    head["v1_patients_with_more_than_one_recording"] = int((ctx["v1"].groupby(patients)["recording_id"].nunique() > 1).sum())
    head["v1_patients_spanning_several_recording_folds"] = int((rec_spread > 1).sum())
    head["v1_windows_of_patients_spanning_recording_folds"] = int(np.isin(patients, rec_spread[rec_spread > 1].index).sum())
    return head


# ----------------------------------------------------------------------------- analysis 2

class LinearModel:
    """P = sigmoid(z . w + b) on standardised inputs z; zero in z is the training mean."""

    def __init__(self, coef, intercept):
        self.w = np.asarray(coef, dtype=np.float64).ravel()
        self.b = float(intercept)

    def logit(self, scaled):
        return np.asarray(scaled, dtype=np.float64) @ self.w + self.b

    def predict(self, scaled):
        return 1.0 / (1.0 + np.exp(-self.logit(scaled)))


def fit_logreg_all(raw, y):
    """BASELINE_SETTINGS['logreg'] on all windows with its own StandardScaler, as run_cv_job does per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(raw, dtype=np.float32).astype(np.float64)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    sk = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=models.BASELINE_SEED)
    sk.fit(z, y)
    model = LinearModel(sk.coef_[0], sk.intercept_[0])
    check = float(np.max(np.abs(model.predict(z) - sk.predict_proba(z)[:, 1])))
    return model, scaler, z, check


def aopc_rows(result, version, model_label, source):
    rows = []
    for subset, block in result.items():
        for ranking, values in block.items():
            rows.append({"source": source, "version": version, "subset": subset, "windows": values["windows"], "ranking": ranking,
                         "deletion_aopc_window_mean": values["deletion_aopc"], "insertion_aopc_window_mean": values["insertion_aopc"],
                         "deletion_gain_vs_random_mean": values.get("deletion_gain", float("nan")),
                         "fraction_better_than_own_random_mean": values.get("fraction_better", float("nan"))})
    table = las.add_predicted_class_column(pd.DataFrame(rows))
    table.insert(0, "model", model_label)
    return table


def aopc_same_code_check(ctx):
    rows, check = [], {}
    tables = {}
    for version, header, frame, features, y, cited_key in (
            ("v1", P["v1_header"], ctx["v1"], las.IR_FEATURES, ctx["y1"], "las_aopc_v1_cited"),
            ("v2", P["v2_header"], ctx["v2"], las.V2_FEATURES, ctx["y2"], "las_aopc_v2_new_run")):
        model, scaled, probability, attribution = las.explain_table(header, frame.loc[:, features].to_numpy(np.float32))
        result = las.aopc_run(model, scaled, probability, attribution, y)
        mine = aopc_rows(result, version, f"TinyKAN {version} (shipped header)",
                         f"recomputed here through analyze_addenda.py: shipped {version} header, in-sample, seed 20260911, 5 random repeats")
        tables[version] = mine
        cited = pd.read_csv(P[cited_key])
        merged = mine.merge(cited, on=["subset", "ranking"], suffixes=("", "_cited"))
        if len(merged) != len(cited):
            raise StopAnalysis(f"{version}: AOPC cells do not line up with {cited_key}")
        diffs = {col: float(np.nanmax(np.abs(merged[col] - merged[f"{col}_cited"])))
                 for col in ("windows", "deletion_aopc_window_mean", "insertion_aopc_window_mean",
                             "deletion_gain_vs_random_mean", "fraction_better_than_own_random_mean")}
        fa = merged[(merged.subset == "all") & (merged.ranking == "firmware_absolute")].iloc[0]
        rm = merged[(merged.subset == "all") & (merged.ranking == "random_mean")].iloc[0]
        ok = all(v <= AOPC_TOLERANCE for v in diffs.values())
        paper_ok = (round(fa.deletion_aopc_window_mean, 4) == PAPER[f"{version}_aopc_firmware_absolute_all"]
                    and round(rm.deletion_aopc_window_mean, 4) == PAPER[f"{version}_aopc_random_all"])
        check[version] = {"reference": str(P[cited_key].relative_to(ROOT).as_posix()),
                          "firmware_absolute_all_recomputed": float(fa.deletion_aopc_window_mean),
                          "firmware_absolute_all_reference": float(fa.deletion_aopc_window_mean_cited),
                          "random_mean_all_recomputed": float(rm.deletion_aopc_window_mean),
                          "random_mean_all_reference": float(rm.deletion_aopc_window_mean_cited),
                          "max_abs_diff_over_all_cells": diffs, "tolerance": AOPC_TOLERANCE,
                          "matches_paper_4dp": bool(paper_ok), "reproduces": bool(ok and paper_ok)}
        for _, r in merged.iterrows():
            rows.append({"version": version, "subset": r.subset, "ranking": r.ranking,
                         "deletion_aopc_recomputed": r.deletion_aopc_window_mean, "deletion_aopc_reference": r.deletion_aopc_window_mean_cited,
                         "abs_diff": abs(r.deletion_aopc_window_mean - r.deletion_aopc_window_mean_cited)})
        if not check[version]["reproduces"]:
            raise StopAnalysis(f"AOPC same-code check failed for {version}: {json.dumps(check[version], indent=2)}")
    return pd.DataFrame(rows), check, tables


def analysis_2(ctx, kan_tables):
    out = ctx["out"]
    head, checks = {}, {}
    coef_rows, agreement_rows, lr_tables = [], [], []
    for version, frame, features, y in (("v1", ctx["v1"], las.IR_FEATURES, ctx["y1"]),
                                        ("v2", ctx["v2"], las.V2_FEATURES, ctx["y2"])):
        model, scaler, z, sk_check = fit_logreg_all(frame.loc[:, features].to_numpy(np.float32), y)
        checks[f"{version}_linear_model_vs_sklearn_predict_proba_max_abs"] = sk_check
        for i, name in enumerate(features):
            coef_rows.append({"version": version, "term": name, "coefficient_standardised": model.w[i],
                              "scaler_mean": scaler.mean_[i], "scaler_std": scaler.scale_[i]})
        coef_rows.append({"version": version, "term": "intercept", "coefficient_standardised": model.b,
                          "scaler_mean": float("nan"), "scaler_std": float("nan")})

        from evaluate_xai_faithfulness import mean_perturbation
        probability, attribution = mean_perturbation(model.predict, z)
        # Exact analytic alternative: one forward gives c_i = w_i z_i, and delta_i = P - sigmoid(logit - c_i).
        contribution = z * model.w[None, :]
        logit = model.logit(z)
        analytic = probability[:, None] - 1.0 / (1.0 + np.exp(-(logit[:, None] - contribution)))
        checks[f"{version}_mean_perturbation_vs_analytic_delta_max_abs"] = float(np.max(np.abs(analytic - attribution)))
        order_mp = lm.firmware_abs_order(attribution)
        order_logit = lm.firmware_abs_order(contribution)
        candidate = probability >= THRESHOLD
        for subset, mask in (("all", np.ones(len(y), bool)), ("candidate_p_ge_0_50", candidate), ("noncandidate_p_lt_0_50", ~candidate)):
            agreement_rows.append({"version": version, "subset": subset, "windows": int(mask.sum()),
                                   "top1_same": float(np.mean(order_mp[mask, 0] == order_logit[mask, 0])),
                                   "top3_same_ordered": float(np.mean(np.all(order_mp[mask, :3] == order_logit[mask, :3], axis=1))),
                                   "full_order_same": float(np.mean(np.all(order_mp[mask] == order_logit[mask], axis=1))),
                                   "comparison": "firmware-absolute order of probability deltas (n+1 forwards) vs order of |w_i z_i| (one forward)"})

        result = las.aopc_run(model, z, probability, attribution, y)
        lr_tables.append(aopc_rows(result, version, f"LogReg-{len(features)} {version} (all-window analysis model)",
                                   f"new run: all-window LogReg {version} (analysis model, BASELINE_SETTINGS['logreg'], own StandardScaler), "
                                   "in-sample, seed 20260911, 5 random repeats"))
        tn, fp, fn, tp = lm.confusion_at(y, probability, THRESHOLD)
        head[f"{version}_logreg_all_window_in_sample"] = {"auc": lm.weighted_auc(y, probability), "candidate_rate": float(candidate.mean()),
                                                          "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}

    lr = pd.concat(lr_tables, ignore_index=True)
    las.write_csv(lr.drop(columns=["model"]), out / "a2_aopc_logreg.csv")
    side = pd.concat([kan_tables["v1"], lr[lr.version == "v1"], kan_tables["v2"], lr[lr.version == "v2"]], ignore_index=True)
    las.write_csv(side, out / "a2_aopc_side_by_side.csv")
    las.write_csv(pd.DataFrame(coef_rows), out / "a2_logreg_coefficients.csv")
    las.write_csv(pd.DataFrame(agreement_rows), out / "a2_ranking_agreement_analytic.csv")

    def get(table, version, subset, ranking, column="deletion_aopc_window_mean"):
        block = table[(table.version == version) & (table.subset == subset) & (table.ranking == ranking)]
        return float(block[column].iloc[0])

    for version in ("v1", "v2"):
        for label, table in (("tinykan", kan_tables[version]), ("logreg", lr)):
            cell = {}
            for subset in ("all", "candidate_p_ge_0_50", "noncandidate_p_lt_0_50"):
                fa, rm, sr = (get(table, version, subset, r) for r in ("firmware_absolute", "random_mean", "signed_risk"))
                cell[subset] = {"windows": int(get(table, version, subset, "firmware_absolute", "windows")),
                                "firmware_absolute": fa, "random_mean": rm, "signed_risk": sr, "firmware_absolute_minus_random": fa - rm,
                                "fraction_better_than_own_random": get(table, version, subset, "firmware_absolute", "fraction_better_than_own_random_mean")}
            head[f"{version}_{label}_aopc"] = cell
    head["ranking_agreement_analytic"] = {f"{r['version']}_{r['subset']}": {k: r[k] for k in ("top1_same", "top3_same_ordered", "full_order_same")}
                                          for r in agreement_rows}
    head["logreg_coefficients"] = {v: {r["term"]: r["coefficient_standardised"] for r in coef_rows if r["version"] == v} for v in ("v1", "v2")}

    # Cost facts: counted from the model form; TinyKAN board numbers are cited from f_deployment.csv.
    deploy = pd.read_csv(P["las_deployment"])
    cost = []
    for version, n in (("v1", 14), ("v2", 16)):
        cost.append({"model": f"LogReg {version} ({n} inputs)", "params": n + 1, "fp32_weight_bytes": 4 * (n + 1),
                     "standardisation_ops_per_forward": f"{n} subtract + {n} multiply (precomputed 1/std)",
                     "multiply_adds_per_forward": n, "nonlinear_evals_per_forward": "1 sigmoid",
                     "forwards_per_explained_decision_mean_perturbation": n + 1,
                     "multiply_adds_per_explained_decision_mean_perturbation": (n + 1) * n,
                     "exact_analytic_alternative": f"1 forward; logit deltas w_i*z_i are the {n} products already formed in the dot product "
                                                   f"(0 extra multiply-adds); probability deltas need {n} extra sigmoids",
                     "board_decision_block_median_ms": "not measured (never deployed)", "flash_bytes": "not measured (never deployed)",
                     "static_ram_bytes": "not measured (never deployed)", "source": "counted from the model form"})
    for version, n in (("v1", 14), ("v2", 16)):
        row = deploy[deploy.model == f"TinyKAN {version} ({n}, 12, 1)"].iloc[0]
        h, k = 12, 8
        cost.append({"model": f"TinyKAN {version} ({n}, 12, 1)", "params": int(row.params), "fp32_weight_bytes": int(row.fp32_weight_bytes),
                     "standardisation_ops_per_forward": f"{n} subtract + {n} divide",
                     "multiply_adds_per_forward": n * h + h * k + h + k,
                     "nonlinear_evals_per_forward": f"{h * k + k} cubic B-spline basis evaluations ({h + 1} edges x {k} knots) + 1 sigmoid",
                     "forwards_per_explained_decision_mean_perturbation": int(row.forward_passes_per_decision),
                     "multiply_adds_per_explained_decision_mean_perturbation": int(row.forward_passes_per_decision) * (n * h + h * k + h + k),
                     "exact_analytic_alternative": "none: the spline network is not additive in its inputs, so per-feature deltas need a forward each",
                     "board_decision_block_median_ms": float(row.decision_block_median_ms),
                     "flash_bytes": int(row.live_image_flash_bytes), "static_ram_bytes": int(row.live_image_static_ram_bytes),
                     "source": "params/passes/board numbers cited from analysis-lamaakal-style-v1/f_deployment.csv (whole 250-sample "
                               "decision block, not inference time); multiply-adds counted from kan_inference.h loops (all 8 knots per edge), "
                               "excluding basis arithmetic"})
    las.write_csv(pd.DataFrame(cost), out / "a2_cost_facts.csv")
    head["cost_facts"] = {r["model"]: {k: r[k] for k in ("params", "multiply_adds_per_forward",
                                                         "forwards_per_explained_decision_mean_perturbation",
                                                         "board_decision_block_median_ms")} for r in cost}
    return head, checks


# ----------------------------------------------------------------------------- readme

def readme(heads, checks):
    a1, a2 = heads["analysis_1"], heads["analysis_2"]
    rc, ac = checks["reproduction_recording_folds"], checks["aopc_same_code"]
    t42, lr = a1["v1_patient_tinykan_seed42"], a1["v1_patient_logreg"]
    ci = a1["v1_patient_ci"]
    ss = a1["v1_patient_tinykan_seed_spread"]
    d = a1["v1_patient_paired_difference"]["auc"]

    def r(x, n=3):
        return f"{x:.{n}f}"

    lines = [
        "# analysis-addenda-v1", "",
        "Two host-only addenda to `outputs/analysis-lamaakal-style-v1/`. Generated by `scripts/analyze_addenda.py`;",
        "write-up and limits in `docs/analysis-addenda.md`. `summary.json` holds input SHA-256 hashes, seeds, library",
        "versions, every check and every headline number.", "",
        "**Analysis models only.** Every model trained here (patient-grouped v1 TinyKAN and LogReg, all-window LogReg) is an",
        "analysis model. None replaces a header; no `kan_model_weights*.h`, weight, threshold (0.50), N=2, label, input or",
        "recipe was changed. No board I/O.", "",
        "## Reproduction checks (both passed before any result was written)", "",
        f"- Recording folds, seed 42: retrain AUC {r(rc['tinykan_retrain_seed42_at_0_50']['auc'], 4)}, recall "
        f"{r(rc['tinykan_retrain_seed42_at_0_50']['recall'], 4)}, specificity {r(rc['tinykan_retrain_seed42_at_0_50']['specificity'], 4)} at 0.50 "
        f"vs frozen {r(rc['tinykan_frozen_at_0_50']['auc'], 4)} / {r(rc['tinykan_frozen_at_0_50']['recall'], 4)} / "
        f"{r(rc['tinykan_frozen_at_0_50']['specificity'], 4)}; max |dP| {rc['tinykan_max_abs_probability_diff']:.1e}, "
        f"{rc['tinykan_candidate_flag_disagreements_at_0_50']} candidate-flag disagreements. LogReg-14 retrain max |dP| "
        f"{rc['logreg_max_abs_probability_diff']:.1e} (recall {r(rc['logreg_retrain_at_0_50']['recall'], 4)}, specificity "
        f"{r(rc['logreg_retrain_at_0_50']['specificity'], 4)}).",
        f"- AOPC same code path: v1 firmware-absolute {ac['v1']['firmware_absolute_all_recomputed']:.4f} vs random "
        f"{ac['v1']['random_mean_all_recomputed']:.4f}; v2 {ac['v2']['firmware_absolute_all_recomputed']:.4f} vs "
        f"{ac['v2']['random_mean_all_recomputed']:.4f}; every cell within {AOPC_TOLERANCE:g} of the cited CSVs.", "",
        "## Analysis 1: v1 TinyKAN, patient-grouped (91 patients, GroupKFold(5) on patient_id)", "",
        f"- TinyKAN seed 42: AUC {r(t42['auc'])} (95% patient-bootstrap CI {r(ci['TinyKAN_auc'][0])}-{r(ci['TinyKAN_auc'][1])}), "
        f"PR-AUC {r(t42['pr_auc'])}, recall {r(t42['recall'])} ({r(ci['TinyKAN_recall'][0])}-{r(ci['TinyKAN_recall'][1])}), "
        f"specificity {r(t42['specificity'])} ({r(ci['TinyKAN_specificity'][0])}-{r(ci['TinyKAN_specificity'][1])}) at 0.50.",
        f"- LogReg-14: AUC {r(lr['auc'])} ({r(ci['LogReg_auc'][0])}-{r(ci['LogReg_auc'][1])}), PR-AUC {r(lr['pr_auc'])}, recall "
        f"{r(lr['recall'])} ({r(ci['LogReg_recall'][0])}-{r(ci['LogReg_recall'][1])}), specificity {r(lr['specificity'])} "
        f"({r(ci['LogReg_specificity'][0])}-{r(ci['LogReg_specificity'][1])}).",
        f"- Paired TinyKAN - LogReg AUC {d['difference']:+.3f} (95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}).",
        f"- Six seeds ({' '.join(str(s) for s in ss['seeds'])}): TinyKAN AUC {r(ss['auc']['mean'])} +- {r(ss['auc']['sd'])}, recall "
        f"{r(ss['recall']['mean'])} +- {r(ss['recall']['sd'])}, specificity {r(ss['specificity']['mean'])} +- {r(ss['specificity']['sd'])}.",
        "- Labels are as coded (notebook 10); this is a grouping change only, not a realignment.", "",
        "## Analysis 2: deletion AOPC, LogReg next to TinyKAN (host, in-sample, P(risk) target)", "",
        "| version | model | subset | windows | firmware-absolute | random | abs - random | signed |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for version in ("v1", "v2"):
        for label, name in (("tinykan", "TinyKAN"), ("logreg", "LogReg")):
            for subset in ("all", "candidate_p_ge_0_50", "noncandidate_p_lt_0_50"):
                c = a2[f"{version}_{label}_aopc"][subset]
                lines.append(f"| {version} | {name} | {subset} | {c['windows']:,} | {c['firmware_absolute']:.4f} | {c['random_mean']:.4f} | "
                             f"{c['firmware_absolute_minus_random']:+.4f} | {c['signed_risk']:.4f} |")
    lines += [
        "",
        "Subsets are by each model's own in-sample P, so the candidate sets differ between models. AOPC here is host,",
        "in-sample, on one all-window model per row; it is not board XAI and not clinical faithfulness. Neither explanation",
        "is called faithful.", "",
        "## Files", "",
        "- `a1_reproduction_recording_folds.csv`: frozen vs retrained v1 TinyKAN (seed 42) and LogReg-14 on recording folds.",
        "- `a1_fold_composition.csv`: windows, patients, recordings, prevalence per fold (v1 recording, v1 patient, v2 patient).",
        "- `a1_patient_grouped_runs.csv`, `a1_patient_seed_summary.csv`: v1 patient-grouped metrics at 0.50, per seed and mean +- sd.",
        "- `a1_patient_bootstrap_ci.csv`, `a1_patient_paired_difference.csv`: patient bootstrap (2,000, seed "
        f"{BOOTSTRAP_SEED_V1_PATIENT}).",
        "- `a1_grouping_comparison.csv`: v1 recording (cited), v1 patient (this run), v2 patient (cited).",
        "- `oof_v1_patient_models.csv.gz`: out-of-fold probabilities of every retrained v1 model, with both fold columns and patient_id.",
        "- `a2_aopc_same_code_check.csv`: TinyKAN AOPC recomputed here vs the cited CSVs.",
        "- `a2_aopc_logreg.csv`: LogReg AOPC in the column layout of `d_aopc_v1_cited.csv`.",
        "- `a2_aopc_side_by_side.csv`: TinyKAN and LogReg rows together, with a model column.",
        "- `a2_logreg_coefficients.csv`: coefficients (standardised space), intercept, scaler mean/std of both all-window LogRegs.",
        "- `a2_ranking_agreement_analytic.csv`: n+1-forward ranking vs the one-forward |w_i z_i| ranking.",
        "- `a2_cost_facts.csv`: parameters, multiply-adds, forwards per explained decision; LogReg board cost not measured (never deployed).",
        "- `rerun_check.json`: second full run compared file by file (written by `--compare`).", "",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main

def write_gzip_csv(frame, path):
    las.write_gzip_csv(frame, path)


def compare(committed, rerun):
    committed = committed if committed.is_absolute() else ROOT / committed
    rerun = rerun if rerun.is_absolute() else ROOT / rerun
    skip = {"summary.json", "rerun_check.json"}
    names = sorted(f.name for f in committed.iterdir() if f.is_file() and f.name not in skip)
    identical, different, numeric = [], [], {}
    for name in names:
        a, b = committed / name, rerun / name
        if not b.exists():
            different.append(name)
            continue
        if a.read_bytes() == b.read_bytes():
            identical.append(name)
            continue
        different.append(name)
        if name.endswith((".csv", ".csv.gz")):
            fa, fb = pd.read_csv(a), pd.read_csv(b)
            cols = [c for c in fa.columns if pd.api.types.is_numeric_dtype(fa[c]) and c in fb]
            numeric[name] = {c: float(np.nanmax(np.abs(fa[c].to_numpy(float) - fb[c].to_numpy(float)))) for c in cols
                             if not np.array_equal(fa[c].to_numpy(), fb[c].to_numpy(), equal_nan=True)}
    sa = json.loads((committed / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((rerun / "summary.json").read_text(encoding="utf-8"))
    report = {"written_by": "scripts/analyze_addenda.py --compare (after the committed run)",
              "committed_directory": committed.relative_to(ROOT).as_posix(), "rerun_directory": rerun.relative_to(ROOT).as_posix(),
              "files_compared": len(names), "byte_identical": identical, "different": different,
              "max_abs_numeric_diff_by_file_column": numeric,
              "headlines_identical": sa["headlines"] == sb["headlines"], "checks_identical": sa["checks"] == sb["checks"]}
    (committed / "rerun_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--jobs", type=int, default=8, help="worker processes for the retraining jobs")
    parser.add_argument("--compare", type=Path, help="compare --output (committed run) with this rerun directory and write rerun_check.json")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if args.compare:
        return compare(out, args.compare)
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    write_once_before = {str(p.relative_to(ROOT).as_posix()): las.tree_hashes(p) for p in WRITE_ONCE}
    input_hashes = {str(path.relative_to(ROOT).as_posix()): las.sha256(path) for path in P.values()}

    v1, v1_kan, v1_lr, v2, v2_oof, _aprime, load_checks = las.load_inputs()
    ctx = {"v1": v1, "v2": v2}
    ctx["y1"], ctx["y2"] = v1["label"].to_numpy(int), v2["label"].to_numpy(int)
    ctx["folds1"] = v1_kan["split_index"].to_numpy(int)
    ctx["folds2p"] = v2_oof["v2_tinykan16_patient_oof"]["fold"].to_numpy(int)
    ctx["p_v1_tinykan"] = v1_kan["y_prob"].to_numpy(float)
    ctx["p_v1_logreg"] = v1_lr["probability_low_oxygen_risk"].to_numpy(float)
    checks = {"load": load_checks}
    if len(v1) != 52090 or v1["recording_id"].nunique() != 162:
        raise RuntimeError("v1 table is not the 52,090-window, 162-recording export")

    # Patient join and folds.
    ctx["patient1"] = v1_patient_ids(v1)
    n_patients = int(pd.Series(ctx["patient1"]).nunique())
    if n_patients != 91:
        raise RuntimeError(f"Expected 91 v1 patients, found {n_patients}")
    v2_regen = group_folds(len(v2), v2["patient_id"].to_numpy())
    ctx["folds1p"] = group_folds(len(v1), ctx["patient1"])
    if pd.Series(ctx["folds1p"]).groupby(ctx["patient1"]).nunique().max() != 1:
        raise RuntimeError("A v1 patient spans two patient folds")
    v2_patient_of_recording = v2.groupby("recording_id")["patient_id"].first()
    shared = v1["recording_id"].isin(v2_patient_of_recording.index)
    v1_fold_of_patient = pd.Series(ctx["folds1p"]).groupby(ctx["patient1"]).first()
    v2_fold_of_patient = pd.Series(ctx["folds2p"]).groupby(v2["patient_id"].to_numpy()).first()
    common = v1_fold_of_patient.index.intersection(v2_fold_of_patient.index)
    checks["patient_folds"] = {
        "join": "v1 recording_id -> openox_canonical_export_audit.csv encounter_id (identical to recording_id) -> "
                "openox_encounter_coverage.csv patient_id; every one of 52,090 windows joined",
        "v1_patients": n_patients, "no_patient_spans_two_folds": True,
        "generator": "sklearn GroupKFold(n_splits=5) on patient_id, folds numbered 1..5, as train_tinykan_v2.out_of_fold",
        "generator_regenerates_v2_patient_fold_column_exactly": bool(np.array_equal(v2_regen, ctx["folds2p"])),
        "v1_patient_id_matches_v2_on_shared_recordings": bool((pd.Series(ctx["patient1"][shared.to_numpy()]) ==
                                                              v1.loc[shared, "recording_id"].map(v2_patient_of_recording).to_numpy()).all()),
        "patients_in_both_v1_and_v2": int(len(common)),
        "patients_in_both_with_same_fold_number": int((v1_fold_of_patient[common] == v2_fold_of_patient[common]).sum()),
        "difference_from_v2": "same generator and settings, but run over the v1 rows (52,090 windows, 91 patients) rather than the "
                              "v2 rows (48,662, 87); GroupKFold balances window counts, so patient-to-fold assignment differs",
    }
    if not checks["patient_folds"]["generator_regenerates_v2_patient_fold_column_exactly"]:
        raise RuntimeError("The fold generator does not regenerate the v2 patient folds; it is not the same generator")

    x1 = v1.loc[:, las.IR_FEATURES].to_numpy(np.float32)
    data = {"v1_recording": (x1, ctx["y1"].astype(np.int64), ctx["folds1"]),
            "v1_patient": (x1, ctx["y1"].astype(np.int64), ctx["folds1p"])}

    # Stage 1: reproduction checks (recording folds; AOPC same code path). Nothing is written if either fails.
    print("Stage 1: recording-fold reproduction (seed 42 TinyKAN, LogReg-14) ...", flush=True)
    stage1 = [{"name": "v1rec_tinykan_seed42", "dataset": "v1_recording", "learner": "tinykan", **las.V1_RECIPE, "seed": 42},
              {"name": "v1rec_logreg", "dataset": "v1_recording", "learner": "logreg"}]
    results = las.run_jobs(stage1, data, args.jobs)
    try:
        repro_table, checks["reproduction_recording_folds"] = reproduction_check(ctx, results)
        print("Stage 1: AOPC same-code check (shipped v1 and v2 headers) ...", flush=True)
        aopc_check_table, checks["aopc_same_code"], kan_tables = aopc_same_code_check(ctx)
    except StopAnalysis as stop:
        print(f"STOP: {stop}")
        return 3

    # Stage 2: patient-grouped jobs.
    print("Stage 2: patient-grouped v1 jobs ...", flush=True)
    stage2 = [{"name": f"v1patient_tinykan_seed{s}", "dataset": "v1_patient", "learner": "tinykan", **las.V1_RECIPE, "seed": s} for s in SEEDS]
    stage2.append({"name": "v1patient_logreg", "dataset": "v1_patient", "learner": "logreg"})
    results.update(las.run_jobs(stage2, data, args.jobs))

    out.mkdir(parents=True)
    ctx["out"] = out
    las.write_csv(repro_table, out / "a1_reproduction_recording_folds.csv")
    las.write_csv(aopc_check_table, out / "a2_aopc_same_code_check.csv")
    heads = {}
    print("Analysis 1", flush=True)
    heads["analysis_1"] = analysis_1(ctx, results)
    print("Analysis 2", flush=True)
    heads["analysis_2"], checks["analysis_2"] = analysis_2(ctx, kan_tables)

    columns = {"row_index": np.arange(len(v1)), "recording_id": v1["recording_id"].to_numpy(), "patient_id": ctx["patient1"],
               "fold_recording": ctx["folds1"], "fold_patient": ctx["folds1p"], "y_true": ctx["y1"]}
    for name in sorted(results):
        columns[name] = results[name]["probabilities"]
    write_gzip_csv(pd.DataFrame(columns), out / "oof_v1_patient_models.csv.gz")

    # Paper consistency (stated numbers vs this run and the cited tables).
    rc = checks["reproduction_recording_folds"]
    checks["paper_numbers"] = {
        "table_I_baseline_auc_0_606": round(rc["tinykan_frozen_at_0_50"]["auc"], 3) == PAPER["v1_recording_tinykan"]["auc"],
        "table_I_baseline_recall_spec_76_7_39_4": [round(100 * rc["tinykan_frozen_at_0_50"]["recall"], 1),
                                                   round(100 * rc["tinykan_frozen_at_0_50"]["specificity"], 1)],
        "table_I_logreg_recall_spec_86_6_26_1": [round(100 * rc["logreg_frozen_at_0_50"]["recall"], 1),
                                                 round(100 * rc["logreg_frozen_at_0_50"]["specificity"], 1)],
        "aopc_v1_0_0443_0_0283": checks["aopc_same_code"]["v1"]["matches_paper_4dp"],
        "aopc_v2_minus_0_3092_minus_0_1612": checks["aopc_same_code"]["v2"]["matches_paper_4dp"],
    }

    (out / "README.md").write_text(readme(heads, checks), encoding="utf-8", newline="\n")

    write_once_after = {str(p.relative_to(ROOT).as_posix()): las.tree_hashes(p) for p in WRITE_ONCE}
    if write_once_after != write_once_before:
        raise RuntimeError("A write-once path changed during the run")

    import scipy
    import sklearn
    import torch
    summary = {
        "created_utc": created, "duration_seconds": round(time.perf_counter() - started, 1),
        "script": "scripts/analyze_addenda.py",
        "scope": "host only; no board I/O; no header, firmware or existing outputs/ writes; retrained models are analysis models only",
        "threshold": THRESHOLD, "debounce_N": 2,
        "seeds": {"tinykan_retrain": list(SEEDS), "bootstrap_v1_patient": BOOTSTRAP_SEED_V1_PATIENT, "n_boot": N_BOOT,
                  "bootstrap_convention": "analyze_lamaakal_style section C: BOOTSTRAP_SEED (20261024) + comparison index; v1 patient is index 3",
                  "aopc": las.AOPC_SEED, "aopc_random_repeats": las.AOPC_RANDOM_REPEATS, "baselines": models.BASELINE_SEED},
        "recipes": {"v1_tinykan_cv": {**las.V1_RECIPE, "lr": 0.01, "weight_decay": 1e-4, "source": "analyze_lamaakal_style.V1_RECIPE (notebook 12)"},
                    "logreg": models.BASELINE_SETTINGS["logreg"],
                    "logreg_all_window": "BASELINE_SETTINGS['logreg'] fitted on all windows after its own StandardScaler (analysis model; "
                                         "the repository has no exported all-window LogReg)"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                        "executable": sys.executable,
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "inputs_sha256": input_hashes,
        "write_once_unchanged": True, "write_once_sha256": write_once_after,
        "checks": checks,
        "job_seconds": {name: round(r["seconds"], 1) for name, r in sorted(results.items())},
        "headlines": heads,
        "outputs_sha256": {f.name: las.sha256(f) for f in sorted(out.iterdir()) if f.is_file()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=las._json_default) + "\n", encoding="utf-8")
    print(f"Done in {summary['duration_seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

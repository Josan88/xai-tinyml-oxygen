"""Patient-grouped rerun with patient-level (clustered) bootstrap intervals.

Closes review finding W6 of docs/ieee-draft-panel-review-2026-11-03/seats/r1-phase2.md:

  1. The published paired bootstrap resamples *recordings*, although the 162 recordings
     come from 91 patients, so recordings sharing a patient are resampled independently
     and the intervals are too narrow by an unknown amount. Here every interval is also
     recomputed with a *patient* cluster bootstrap on the same frozen out-of-fold
     predictions, so the amount is no longer unknown.
  2. The patient-grouped rerun was reported for AUC only. Here each patient-grouped cell
     carries AUC, recall, specificity, precision, F1, balanced accuracy and accuracy.
  3. The intervals behind "neither interval includes zero" are printed for every metric
     of the paired TinyKAN - LogReg difference, in all four CV cells.

Nothing is retrained. Every probability vector is a frozen out-of-fold artefact already
in outputs/; only the resampling is new. Bootstrap machinery is scripts/lamaakal_metrics
(RankedScores weighted Mann-Whitney AUC + group multiplicities), extended here from
{auc, recall, specificity} to the full threshold-metric set at the 0.50 operating point.

Reads only; writes solely into --output (default: outputs/patient-bootstrap-v1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import lamaakal_metrics as lm  # noqa: E402

THRESHOLD = lm.REPORTED_THRESHOLD          # 0.50, the firmware cut; >= comparison
N_BOOT = 2000                              # same count as the published recording bootstrap
BOOT_SEED = 20261105                       # new base seed; cell index added, as in section C
CI_LEVEL = 0.95
METRICS = ("auc", "recall", "specificity", "precision", "f1", "balanced_accuracy", "accuracy")

P = {
    "v1_patient_oof": ROOT / "outputs/analysis-addenda-v1/oof_v1_patient_models.csv.gz",
    "v2_analysis_oof": ROOT / "outputs/analysis-lamaakal-style-v1/oof_v2_analysis_models.csv.gz",
    "v2_tinykan_rec": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "v2_tinykan_pat": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
    "v2_logreg_rec": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
    "published_rec_ci": ROOT / "outputs/analysis-lamaakal-style-v1/c_bootstrap_ci.csv",
    "published_rec_diff": ROOT / "outputs/analysis-lamaakal-style-v1/c_paired_comparison.csv",
    "published_pat_ci": ROOT / "outputs/analysis-addenda-v1/a1_patient_bootstrap_ci.csv",
    "published_pat_diff": ROOT / "outputs/analysis-addenda-v1/a1_patient_paired_difference.csv",
}

# Published point AUCs the frozen vectors must reproduce before anything is written.
# The two gzipped OOF files store 7 significant digits, so the bound is 1e-5, not 0.
POINT_REFERENCE = {
    ("v1", "recording", "TinyKAN"): 0.606302,
    ("v1", "recording", "LogReg"): 0.591991,
    ("v1", "patient", "TinyKAN"): 0.604118,
    ("v1", "patient", "LogReg"): 0.570956,
    ("v2", "recording", "TinyKAN"): 0.9075283627964827,
    ("v2", "recording", "LogReg"): 0.8630922610529321,
    ("v2", "patient", "TinyKAN"): 0.9069360588073916,
    ("v2", "patient", "LogReg"): 0.862958,
}
POINT_TOLERANCE = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- bootstrap

def _rates(tp, fn, fp, tn):
    """Vectorised metric set from resampled confusion counts (arrays over B draws)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
    return {"recall": recall, "specificity": specificity, "precision": precision,
            "f1": f1, "balanced_accuracy": 0.5 * (recall + specificity), "accuracy": accuracy}


def cluster_bootstrap(y, probabilities, groups, n_boot, seed, threshold=THRESHOLD):
    """Paired cluster bootstrap over `groups` for several models on the same windows.

    Resamples whole groups with replacement and carries the multiplicity as window
    weights, which is identical to replicating the drawn groups. Every model in
    `probabilities` sees the same resampled groups, so model differences are paired.
    Returns (point, draws, n_groups): point[name][metric], draws[name][metric] -> (B,).
    """
    y = np.asarray(y).astype(int)
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(group_index.max()) + 1
    counts = lm.group_bootstrap_counts(group_index, n_groups, n_boot, seed)   # B x n_groups
    point, draws = {}, {}
    for name, p in probabilities.items():
        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all():
            raise ValueError(f"{name}: non-finite probabilities")
        ranked = lm.RankedScores(y, p)
        pred = p >= threshold
        g = {key: np.bincount(group_index, weights=mask.astype(float), minlength=n_groups)
             for key, mask in (("tp", (y == 1) & pred), ("fn", (y == 1) & ~pred),
                               ("fp", (y == 0) & pred), ("tn", (y == 0) & ~pred))}
        drawn = _rates(counts @ g["tp"], counts @ g["fn"], counts @ g["fp"], counts @ g["tn"])
        aucs = np.empty(n_boot)
        for b in range(n_boot):
            aucs[b] = ranked.auc(counts[b][group_index])
        drawn["auc"] = aucs
        full = lm.binary_metrics(y, p, threshold)
        point[name] = {metric: float(full[metric]) for metric in METRICS}
        point[name]["pr_auc"] = float(full["pr_auc"])
        point[name]["prevalence"] = float(full["prevalence"])
        point[name]["n"] = int(full["n"])
        for key in ("tn", "fp", "fn", "tp"):
            point[name][key] = int(full[key])
        draws[name] = drawn
    return point, draws, n_groups


# ------------------------------------------------------------------------------ inputs

def load_cells():
    """Four CV cells x two models, each a frozen out-of-fold probability vector."""
    v1 = pd.read_csv(P["v1_patient_oof"]).sort_values("row_index").reset_index(drop=True)
    o2 = pd.read_csv(P["v2_analysis_oof"]).sort_values("row_index").reset_index(drop=True)
    k2r = pd.read_csv(P["v2_tinykan_rec"]).sort_values("row_index").reset_index(drop=True)
    k2p = pd.read_csv(P["v2_tinykan_pat"]).sort_values("row_index").reset_index(drop=True)
    l2r = pd.read_csv(P["v2_logreg_rec"]).sort_values("row_index").reset_index(drop=True)

    checks = {}
    if not np.array_equal(v1["row_index"], np.arange(len(v1))):
        raise RuntimeError("v1 patient OOF does not cover every canonical row once")
    for name, frame in (("tinykan_rec", k2r), ("tinykan_pat", k2p), ("logreg_rec", l2r)):
        if not np.array_equal(frame["row_index"], np.arange(len(o2))):
            raise RuntimeError(f"v2 {name} does not cover every v2 row once")
        if not (frame["y_true"].to_numpy() == o2["y_true"].to_numpy()).all():
            raise RuntimeError(f"v2 {name} labels differ from the analysis OOF table")
        if not (frame["recording_id"].astype(str).to_numpy()
                == o2["recording_id"].astype(str).to_numpy()).all():
            raise RuntimeError(f"v2 {name} recording IDs differ")
    if not (l2r["fold"].to_numpy() == k2r["fold"].to_numpy()).all():
        raise RuntimeError("v2 LogReg recording folds differ from TinyKAN")
    if not (k2p["patient_id"].to_numpy() == k2r["patient_id"].to_numpy()).all():
        raise RuntimeError("v2 patient IDs differ between prediction files")
    # No group may straddle the folds it is supposed to define.
    if v1.groupby("recording_id")["fold_recording"].nunique().max() != 1:
        raise RuntimeError("a v1 recording spans several recording folds")
    if v1.groupby("patient_id")["fold_patient"].nunique().max() != 1:
        raise RuntimeError("a v1 patient spans several patient folds")
    if k2r.groupby("recording_id")["fold"].nunique().max() != 1:
        raise RuntimeError("a v2 recording spans several recording folds")
    if k2p.groupby("patient_id")["fold"].nunique().max() != 1:
        raise RuntimeError("a v2 patient spans several patient folds")
    if o2.groupby(k2p["patient_id"])["fold_patient"].nunique().max() != 1:
        raise RuntimeError("a v2 patient spans several LogReg patient folds")
    if not (o2["fold_patient"].to_numpy() == k2p["fold"].to_numpy()).all():
        raise RuntimeError("v2 patient-grouped LogReg folds differ from TinyKAN's")
    checks["v2_patient_folds_shared"] = "LogReg and TinyKAN share the v2 patient fold column"

    # How much patient leakage the recording folds carry (the reason W6 exists).
    spanning = v1.groupby("patient_id")["fold_recording"].nunique()
    checks["v1_patients_spanning_recording_folds"] = int((spanning > 1).sum())
    checks["v1_windows_in_spanning_patients"] = int(
        v1["patient_id"].isin(spanning[spanning > 1].index).sum())
    spanning2 = pd.Series(k2r["fold"].to_numpy()).groupby(k2r["patient_id"].to_numpy()).nunique()
    checks["v2_patients_spanning_recording_folds"] = int((spanning2 > 1).sum())
    checks["v2_windows_in_spanning_patients"] = int(
        k2r["patient_id"].isin(spanning2[spanning2 > 1].index).sum())

    cells = [
        {"version": "v1", "cv_grouping": "recording",
         "y": v1["y_true"].to_numpy(), "patient": v1["patient_id"].astype(str).to_numpy(),
         "recording": v1["recording_id"].astype(str).to_numpy(),
         "p": {"TinyKAN": v1["v1rec_tinykan_seed42"].to_numpy(),
               "LogReg": v1["v1rec_logreg"].to_numpy()},
         "source": "analysis-addenda-v1/oof_v1_patient_models.csv.gz (v1rec_* columns; "
                   "split_index folds, TinyKAN V1_RECIPE seed 42, LogReg BASELINE_SETTINGS)"},
        {"version": "v1", "cv_grouping": "patient",
         "y": v1["y_true"].to_numpy(), "patient": v1["patient_id"].astype(str).to_numpy(),
         "recording": v1["recording_id"].astype(str).to_numpy(),
         "p": {"TinyKAN": v1["v1patient_tinykan_seed42"].to_numpy(),
               "LogReg": v1["v1patient_logreg"].to_numpy()},
         "source": "analysis-addenda-v1/oof_v1_patient_models.csv.gz (v1patient_* columns; "
                   "GroupKFold(5) on patient_id)"},
        {"version": "v2", "cv_grouping": "recording",
         "y": o2["y_true"].to_numpy(), "patient": k2r["patient_id"].astype(str).to_numpy(),
         "recording": o2["recording_id"].astype(str).to_numpy(),
         "p": {"TinyKAN": k2r["y_prob"].to_numpy(), "LogReg": l2r["y_prob"].to_numpy()},
         "source": "openox-v2/predictions_b_v2_tinykan16.csv and predictions_c_v2_logreg16.csv "
                   "(shipped recording-grouped out-of-fold predictions)"},
        {"version": "v2", "cv_grouping": "patient",
         "y": o2["y_true"].to_numpy(), "patient": k2p["patient_id"].astype(str).to_numpy(),
         "recording": o2["recording_id"].astype(str).to_numpy(),
         "p": {"TinyKAN": k2p["y_prob"].to_numpy(),
               "LogReg": o2["v2patient_logreg"].to_numpy()},
         "source": "openox-v2/predictions_b_v2_tinykan16_patient.csv and "
                   "analysis-lamaakal-style-v1/oof_v2_analysis_models.csv.gz (v2patient_logreg)"},
    ]
    # Seed spread of the v1 patient-grouped TinyKAN, for the seed-sensitivity row.
    seed_columns = {int(column.rsplit("seed", 1)[1]): column for column in v1.columns
                    if column.startswith("v1patient_tinykan_seed")}
    seeds = {seed: v1[column].to_numpy() for seed, column in sorted(seed_columns.items())}
    return cells, seeds, v1["y_true"].to_numpy(), v1["patient_id"].astype(str).to_numpy(), checks


def verify_points(cells):
    rows = []
    for cell in cells:
        for model, p in cell["p"].items():
            key = (cell["version"], cell["cv_grouping"], model)
            observed = lm.weighted_auc(cell["y"], p)
            expected = POINT_REFERENCE[key]
            delta = observed - expected
            if abs(delta) > POINT_TOLERANCE:
                raise RuntimeError(f"{key}: AUC {observed:.8f} differs from published "
                                   f"{expected:.8f} by {delta:.2e}")
            rows.append({"version": cell["version"], "cv_grouping": cell["cv_grouping"],
                         "model": model, "auc_recomputed": observed, "auc_published": expected,
                         "difference": delta, "tolerance": POINT_TOLERANCE,
                         "source": cell["source"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs" / "patient-bootstrap-v1"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    arguments = parser.parse_args()
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=True)
    n_boot = arguments.n_boot

    cells, seeds, y_seed, patient_seed, checks = load_cells()
    reproduction = verify_points(cells)
    reproduction.to_csv(out / "reproduction_check.csv", index=False, float_format="%.10g")
    print("point reproduction: max |delta AUC| = "
          f"{reproduction['difference'].abs().max():.2e} (bound {POINT_TOLERANCE:g})")

    ci_rows, diff_rows = [], []
    for index, cell in enumerate(cells):
        # Patient bootstrap everywhere; recording bootstrap too, where the published
        # interval used recordings, so the two widths can be compared like for like.
        units = ("patient", "recording") if cell["cv_grouping"] == "recording" else ("patient",)
        for unit_offset, unit in enumerate(units):
            seed = BOOT_SEED + 10 * index + unit_offset
            point, draws, n_groups = cluster_bootstrap(
                cell["y"], cell["p"], cell[unit], n_boot, seed)
            for model in cell["p"]:
                for metric in METRICS:
                    low, high = lm.percentile_ci(draws[model][metric], CI_LEVEL)
                    ci_rows.append({
                        "version": cell["version"], "cv_grouping": cell["cv_grouping"],
                        "resampled_unit": unit, "model": model, "metric": metric,
                        "point": point[model][metric], "ci_low": low, "ci_high": high,
                        "ci_width": high - low, "n_boot": n_boot, "seed": seed,
                        "n_units": n_groups, "n_windows": point[model]["n"],
                        "prevalence": point[model]["prevalence"],
                        "pr_auc_point": point[model]["pr_auc"],
                        "tn": point[model]["tn"], "fp": point[model]["fp"],
                        "fn": point[model]["fn"], "tp": point[model]["tp"],
                        "source": cell["source"]})
            for metric in METRICS:
                difference = draws["TinyKAN"][metric] - draws["LogReg"][metric]
                low, high = lm.percentile_ci(difference, CI_LEVEL)
                finite = difference[np.isfinite(difference)]
                diff_rows.append({
                    "version": cell["version"], "cv_grouping": cell["cv_grouping"],
                    "resampled_unit": unit, "comparison": "TinyKAN - LogReg", "metric": metric,
                    "difference": point["TinyKAN"][metric] - point["LogReg"][metric],
                    "ci_low": low, "ci_high": high, "ci_width": high - low,
                    "se": float(np.std(finite, ddof=1)),
                    "fraction_resamples_le_0": float(np.mean(finite <= 0)),
                    "excludes_zero": bool(low > 0 or high < 0),
                    "method": f"paired {unit} bootstrap ({n_boot} resamples, "
                              f"{int(100 * CI_LEVEL)}% percentile CI)",
                    "n_boot": n_boot, "seed": seed, "n_units": n_groups})
            print(f"  {cell['version']} {cell['cv_grouping']}-grouped CV, "
                  f"{unit} bootstrap ({n_groups} units): done")

    ci = pd.DataFrame(ci_rows)
    diff = pd.DataFrame(diff_rows)
    ci.to_csv(out / "bootstrap_ci.csv", index=False, float_format="%.8g")
    diff.to_csv(out / "paired_difference.csv", index=False, float_format="%.8g")

    # How much narrower the published recording bootstrap was, on identical predictions.
    key = ["version", "cv_grouping", "model", "metric"]
    rec = ci[ci.resampled_unit == "recording"].set_index(key)
    pat = ci[ci.resampled_unit == "patient"].set_index(key)
    shared = rec.index.intersection(pat.index)
    width = pd.DataFrame({
        "point": pat.loc[shared, "point"],
        "recording_ci_low": rec.loc[shared, "ci_low"], "recording_ci_high": rec.loc[shared, "ci_high"],
        "recording_ci_width": rec.loc[shared, "ci_width"],
        "patient_ci_low": pat.loc[shared, "ci_low"], "patient_ci_high": pat.loc[shared, "ci_high"],
        "patient_ci_width": pat.loc[shared, "ci_width"],
    }).reset_index()
    width["width_ratio_patient_over_recording"] = (width.patient_ci_width
                                                   / width.recording_ci_width)
    width.to_csv(out / "interval_width_comparison.csv", index=False, float_format="%.8g")

    keyd = ["version", "cv_grouping", "metric"]
    recd = diff[diff.resampled_unit == "recording"].set_index(keyd)
    patd = diff[diff.resampled_unit == "patient"].set_index(keyd)
    sharedd = recd.index.intersection(patd.index)
    width_diff = pd.DataFrame({
        "difference": patd.loc[sharedd, "difference"],
        "recording_ci_low": recd.loc[sharedd, "ci_low"], "recording_ci_high": recd.loc[sharedd, "ci_high"],
        "recording_ci_width": recd.loc[sharedd, "ci_width"],
        "recording_excludes_zero": recd.loc[sharedd, "excludes_zero"],
        "patient_ci_low": patd.loc[sharedd, "ci_low"], "patient_ci_high": patd.loc[sharedd, "ci_high"],
        "patient_ci_width": patd.loc[sharedd, "ci_width"],
        "patient_excludes_zero": patd.loc[sharedd, "excludes_zero"],
    }).reset_index()
    width_diff["width_ratio_patient_over_recording"] = (width_diff.patient_ci_width
                                                        / width_diff.recording_ci_width)
    width_diff.to_csv(out / "interval_width_comparison_differences.csv", index=False,
                      float_format="%.8g")

    # Seed spread of the v1 patient-grouped TinyKAN (point metrics, no bootstrap).
    seed_rows = []
    for seed, p in seeds.items():
        metrics = lm.binary_metrics(y_seed, p, THRESHOLD)
        seed_rows.append({"version": "v1", "cv_grouping": "patient", "model": "TinyKAN",
                          "seed": seed, **{metric: metrics[metric] for metric in METRICS},
                          "pr_auc": metrics["pr_auc"]})
    seed_frame = pd.DataFrame(seed_rows)
    summary_rows = []
    for metric in METRICS:
        mean, sd = lm.mean_sd(seed_frame[metric].to_numpy())
        summary_rows.append({"metric": metric, "mean": mean, "sd": sd,
                             "minimum": seed_frame[metric].min(),
                             "maximum": seed_frame[metric].max(),
                             "n_seeds": int(len(seed_frame))})
    seed_frame.to_csv(out / "seed_runs.csv", index=False, float_format="%.8g")
    pd.DataFrame(summary_rows).to_csv(out / "seed_summary.csv", index=False, float_format="%.8g")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Patient-grouped rerun with patient-level cluster bootstrap intervals; "
                   "closes W6 (bootstrap ignores patient clustering; patient-grouped rerun "
                   "covered AUC only; difference intervals not printed).",
        "retrained": False,
        "threshold": THRESHOLD,
        "n_boot": n_boot,
        "ci_level": CI_LEVEL,
        "bootstrap_base_seed": BOOT_SEED,
        "metrics": list(METRICS),
        "checks": checks,
        "point_reproduction_max_abs_delta_auc": float(reproduction["difference"].abs().max()),
        "point_reproduction_tolerance": POINT_TOLERANCE,
        "inputs": {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in P.values()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Skin-tone stratification of RQ1 on the frozen out-of-fold scores (host only).

Pre-registration: docs/skin-tone-stratification-preregistration.md (commit 75e9fa5, amendment A1).
Nothing is trained or retuned. Reads frozen prediction tables and the local OpenOximetry
encounter and spectrophotometer tables; writes solely into a new, write-once output folder.

S1  primary grouping: monk_dorsal in three bands (A-D light, E-G medium, H-J dark).
    Per group AUC, recall, specificity at 0.50 with patient-bootstrap intervals;
    primary contrast = recall, dark minus light.
S2  the same for logistic regression and for three further groupings (monk_forehead,
    Fitzpatrick, dorsal ITA thirds). Descriptive.
S3  per primary group, medians of two signal-strength inputs. Descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import lamaakal_metrics as lm  # noqa: E402

THRESHOLD = lm.REPORTED_THRESHOLD
N_BOOT = 2000
BOOT_SEED = 20261111
MIN_PATIENTS = 10
DATA = ROOT / "data/physionet.org/files/openox-repo/1.1.1"
P = {
    "tinykan": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "logreg": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
    "features": ROOT / "outputs/openox-v2/window_features.csv",
    "encounter": DATA / "encounter.csv",
    "spectro": DATA / "spectrophotometer.csv",
}
ORDER = ("light", "medium", "dark")
EXPECTED_DORSAL = {"light": 47, "medium": 63, "dark": 44}
PAPER = {"auc": "0.908", "recall": "89.3", "specificity": "86.8"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def monk_band(value):
    if pd.isna(value):
        return None
    level = "ABCDEFGHIJ".index(str(value).strip().upper()) + 1
    return "light" if level <= 4 else "medium" if level <= 7 else "dark"


def fitz_band(value):
    if pd.isna(value):
        return None
    return "light" if value <= 2 else "medium" if value <= 4 else "dark"


def recording_groupings(recordings):
    enc = pd.read_csv(P["encounter"]).set_index("encounter_id")
    if not set(recordings).issubset(enc.index) or enc.index.duplicated().any():
        raise RuntimeError("M0.2: a recording does not join exactly one encounter row")
    enc = enc.loc[recordings]
    out = pd.DataFrame(index=recordings)
    out["patient_id_encounter"] = enc["patient_id"].astype(str).to_numpy()
    out["monk_dorsal"] = [monk_band(v) for v in enc["monk_dorsal"]]
    out["monk_forehead"] = [monk_band(v) for v in enc["monk_forehead"]]
    out["fitzpatrick"] = [fitz_band(v) for v in enc["fitzpatrick"]]
    spec = pd.read_csv(P["spectro"], usecols=["encounter_id", "group", "lab_l", "lab_b"])
    spec = spec[(spec["group"] == "Dorsal (B)") & spec["encounter_id"].isin(recordings)]
    spec = spec.assign(ita=np.degrees(np.arctan2(spec["lab_l"] - 50.0, spec["lab_b"])))
    ita = spec.groupby("encounter_id")["ita"].median().reindex(recordings)
    # Amendment A1: two recordings have dorsal rows with empty L* and b*; they are dropped
    # from this grouping only.
    if int(ita.isna().sum()) != 2:
        raise RuntimeError(f"expected 2 recordings without a dorsal ITA, found {int(ita.isna().sum())}")
    out["ita_dorsal_deg"] = ita.to_numpy()
    # Lowest ITA third = darkest; thirds over the recordings that have a value.
    third = pd.qcut(ita.dropna(), 3, labels=["dark", "medium", "light"]).astype(str)
    out["ita_dorsal_third"] = [third.get(r) for r in recordings]
    return out


def stratified_bootstrap(y, probs, patient_index, n_patients, band, seed):
    """Patient bootstrap; every band and model sees the same resampled patients."""
    counts = lm.group_bootstrap_counts(patient_index, n_patients, N_BOOT, seed)
    rows, draws = [], {}
    for model, p in probs.items():
        pred = p >= THRESHOLD
        for name in ORDER:
            mask = band == name
            ym, pm, gm = y[mask], p[mask], patient_index[mask]
            ranked = lm.RankedScores(ym, pm)
            sums = {key: np.bincount(gm, weights=m.astype(float), minlength=n_patients)
                    for key, m in (("tp", (ym == 1) & pred[mask]), ("pos", ym == 1),
                                   ("tn", (ym == 0) & ~pred[mask]), ("neg", ym == 0))}
            with np.errstate(invalid="ignore", divide="ignore"):
                d = {"recall": (counts @ sums["tp"]) / (counts @ sums["pos"]),
                     "specificity": (counts @ sums["tn"]) / (counts @ sums["neg"])}
            d["auc"] = np.array([ranked.auc(counts[b][gm]) for b in range(N_BOOT)])
            point = {"auc": ranked.auc(), "recall": sums["tp"].sum() / sums["pos"].sum(),
                     "specificity": sums["tn"].sum() / sums["neg"].sum()}
            row = {"model": model, "band": name, "windows": int(mask.sum()),
                   "patients": int(np.unique(gm).size), "prevalence": float(ym.mean())}
            for metric in ("auc", "recall", "specificity"):
                lo, hi = np.nanpercentile(d[metric], [2.5, 97.5])
                row.update({metric: float(point[metric]), f"{metric}_lo": float(lo),
                            f"{metric}_hi": float(hi),
                            f"{metric}_nan_draws": int(np.isnan(d[metric]).sum())})
            rows.append(row)
            draws[(model, name)] = (point, d)
    return rows, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs/skin-tone-stratification-v1"))
    out_dir = Path(parser.parse_args().output)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; outputs are write-once")
    hashes_before = {k: sha256(v) for k, v in P.items()}

    kan = pd.read_csv(P["tinykan"]).sort_values("row_index").reset_index(drop=True)
    log = pd.read_csv(P["logreg"]).sort_values("row_index").reset_index(drop=True)
    feat = pd.read_csv(P["features"])
    for frame in (log, feat):
        if len(frame) != len(kan) or not (frame["recording_id"].astype(str).to_numpy()
                                         == kan["recording_id"].astype(str).to_numpy()).all():
            raise RuntimeError("tables are not row-aligned")
    if not (log["y_true"].to_numpy() == kan["y_true"].to_numpy()).all():
        raise RuntimeError("labels differ between prediction tables")
    y = kan["y_true"].to_numpy().astype(int)
    probs = {"TinyKAN": kan["y_prob"].to_numpy(float), "LogReg": log["y_prob"].to_numpy(float)}

    # M0.1 pooled metrics reproduce the paper to printed precision.
    pooled = lm.binary_metrics(y, probs["TinyKAN"], THRESHOLD)
    got = {"auc": f"{pooled['auc']:.3f}", "recall": f"{100 * pooled['recall']:.1f}",
           "specificity": f"{100 * pooled['specificity']:.1f}"}
    if got != PAPER:
        raise SystemExit(f"M0.1 failed: {got} != {PAPER}")

    # M0.2 join and counts.
    recordings = pd.unique(kan["recording_id"].astype(str))
    groups = recording_groupings(list(recordings))
    rec_patient = kan.groupby(kan["recording_id"].astype(str))["patient_id"].agg(lambda s: s.astype(str).unique())
    if any(len(v) != 1 for v in rec_patient):
        raise SystemExit("M0.2 failed: a recording has several patients")
    if not (groups["patient_id_encounter"].to_numpy()
            == np.array([rec_patient[r][0] for r in recordings])).all():
        raise SystemExit("M0.2 failed: patient IDs differ between predictions and encounter table")
    patients, patient_index = np.unique(kan["patient_id"].astype(str).to_numpy(), return_inverse=True)
    if (len(recordings), len(patients), len(kan)) != (154, 87, 48662):
        raise SystemExit(f"M0.2 failed: {(len(recordings), len(patients), len(kan))}")

    # M0.3 primary group sizes.
    dorsal_counts = groups["monk_dorsal"].value_counts().to_dict()
    if dorsal_counts != EXPECTED_DORSAL:
        raise SystemExit(f"M0.3 failed: {dorsal_counts}")

    window_band = {g: groups[g].reindex(kan["recording_id"].astype(str)).to_numpy()
                   for g in ("monk_dorsal", "monk_forehead", "fitzpatrick", "ita_dorsal_third")}
    strata, contrasts = [], []
    for cell, (grouping, band) in enumerate(window_band.items()):
        rows, draws = stratified_bootstrap(y, probs, patient_index, len(patients), band,
                                           BOOT_SEED + cell)
        rec_counts = groups[grouping].value_counts().to_dict()
        for row in rows:
            row.update({"grouping": grouping, "recordings": int(rec_counts.get(row["band"], 0))})
            strata.append(row)
        for model in probs:
            small = min(r["patients"] for r in rows
                        if r["model"] == model and r["band"] in ("dark", "light")) < MIN_PATIENTS
            for metric in ("recall", "specificity", "auc"):
                pd_, dd = draws[(model, "dark")]
                pl, dl = draws[(model, "light")]
                diff = dd[metric] - dl[metric]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                contrasts.append({
                    "grouping": grouping, "model": model, "metric": metric,
                    "dark_minus_light": float(pd_[metric] - pl[metric]),
                    "lo": None if small else float(lo), "hi": None if small else float(hi),
                    "nan_draws": int(np.isnan(diff).sum()),
                    "role": "PRIMARY" if (grouping, model, metric) == ("monk_dorsal", "TinyKAN", "recall")
                    else "secondary",
                    "read": "no contrast read (group under 10 patients)" if small
                    else "differs" if (lo > 0 or hi < 0) else "no difference detected"})

    # S3 signal context, primary grouping.
    signal = (feat.assign(band=window_band["monk_dorsal"])
              .groupby("band")[["ir_perfusion_index", "std"]].median()
              .rename(columns={"std": "ir_bandpassed_std"}).reindex(list(ORDER)).reset_index())

    hashes_after = {k: sha256(v) for k, v in P.items()}
    if hashes_after != hashes_before:
        raise SystemExit("an input changed during the run")
    out_dir.mkdir(parents=True)
    cols = ["grouping", "model", "band", "recordings", "patients", "windows", "prevalence"]
    strata = pd.DataFrame(strata)
    strata[cols + [c for c in strata.columns if c not in cols]].to_csv(out_dir / "strata.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out_dir / "contrasts.csv", index=False)
    signal.to_csv(out_dir / "signal_context.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": "docs/skin-tone-stratification-preregistration.md (commit 75e9fa5, amendment A1)",
        "threshold": THRESHOLD, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "m0": {"pooled": got, "recordings": 154, "patients": 87, "windows": 48662,
               "dorsal_recordings": dorsal_counts},
        "recordings_per_band": {g: groups[g].value_counts(dropna=False).to_dict()
                                for g in window_band},
        "ita_dorsal_third_edges_deg": [float(v) for v in
                                       groups["ita_dorsal_deg"].quantile([0, 1 / 3, 2 / 3, 1])],
        "inputs_sha256": hashes_before, "script_sha256": sha256(__file__),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
    }, indent=2, default=str), encoding="utf-8")
    print(strata[strata["grouping"] == "monk_dorsal"].round(4).to_string(index=False))
    print(pd.DataFrame(contrasts).round(4).to_string(index=False))
    print(signal.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

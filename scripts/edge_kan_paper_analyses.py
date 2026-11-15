"""Edge KAN: the IEEE draft's host analyses rerun (docs/edge-kan-paper-analyses-preregistration.md).

Host only, write-once (outputs/edge-kan-v1/paper-analyses/). Imports the original analysis code and
swaps the edge KAN (ekan16, edge_kan_weights_v2.h, threshold 0.50) in for the Embedded Spline Network.

  python scripts/edge_kan_paper_analyses.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import analyze_lamaakal_style as las  # noqa: E402
import analyze_physio_concept as apc  # noqa: E402
import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402
import analyze_panel_addenda_20261110 as addenda  # noqa: E402
import analyze_deletion_range_r4 as r4  # noqa: E402
import analyze_skin_tone_stratification as skin  # noqa: E402
import analyze_scalco2_vs_sao2 as sc  # noqa: E402
import train_edge_kan as tek  # noqa: E402
import train_tinykan_v2 as tk  # noqa: E402

OUT = ROOT / "outputs/edge-kan-v1/paper-analyses"
PREREG = ROOT / "docs/edge-kan-paper-analyses-preregistration.md"
P = {
    "features": ROOT / "outputs/openox-v2/window_features.csv",
    "edge_header": ROOT / "firmware/Arduino Nano 33/include/edge_kan_weights_v2.h",
    "spline_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "edge_oof": ROOT / "outputs/edge-kan-v1/predictions_ekan16_recording.csv",
    "spline_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "logreg_oof": ROOT / "outputs/openox-v2/predictions_c_v2_logreg16.csv",
    "artefact_rows": ROOT / "outputs/openox-v3/artefact_rows.csv",
    "s1_windows": ROOT / "outputs/scalco2-vs-sao2-v1/s1_windows.csv",
    "ptt_windows": ROOT / "outputs/openox-v2/ptt-subject-1/windows.csv",
}
V16 = list(las.V2_FEATURES)
THRESHOLD = 0.50
N_BOOT = 2000
BOOT_SEED = 20261111
SEEDS = (42, 1, 2, 3, 4, 5)
CLAMP = 3.0
ZERO = las.ZERO_DELTA


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EdgeModel(tek.EdgeHeaderKAN):
    """EdgeHeaderKAN plus the attributes the imported analyses use."""

    @classmethod
    def from_params(cls, params):
        self = cls.__new__(cls)
        self.p = {"EKAN_SCALER_MEAN": np.asarray(params["scaler_mean"], np.float32).astype(np.float64),
                  "EKAN_SCALER_STD": np.asarray(params["scaler_std"], np.float32).astype(np.float64),
                  "EKAN_CENTERS": np.asarray(params["centers"], np.float32).astype(np.float64),
                  "EKAN_STEP": np.float64(np.float32(params["step"])),
                  "EKAN1_THETA": np.asarray(params["theta1"], np.float32).astype(np.float64),
                  "EKAN2_THETA": np.asarray(params["theta2"], np.float32).astype(np.float64)}
        self.lo, self.hi = float(tk.SPLINE_RANGE[0]), float(tk.SPLINE_RANGE[1])
        self.n_inputs = self.p["EKAN_SCALER_MEAN"].shape[0]
        return self

    def hidden_preactivation(self, scaled):
        return self.hidden(np.asarray(scaled, np.float64))

    def all_hidden_clamped(self, scaled):
        return (np.abs(self.hidden_preactivation(scaled)) >= CLAMP).all(axis=1)


def fold_params(net, scaler):
    spline = net.layers[0].spline
    return {"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_,
            "centers": spline.centers.detach().numpy() if hasattr(spline, "centers") else np.linspace(-3, 3, 8),
            "step": float(spline.step) if hasattr(spline, "step") else 6 / 7,
            "theta1": net.layers[0].theta.detach().numpy(), "theta2": net.layers[1].theta.detach().numpy()}


def metrics_row(y, p):
    m = lm.binary_metrics(y, p, THRESHOLD)
    return {k: m[k] for k in ("auc", "recall", "specificity", "precision", "f1")} | {
        "median_p_negative": float(np.median(p[y == 0])), "median_p_positive": float(np.median(p[y == 1]))}


def safe_gain_rows(tag, reference_name, firmware, random_mean, masks, patient_index, n_patients, cell):
    """addenda.gain_rows, with an empty subset reported as 0 windows instead of bootstrapped."""
    rows = []
    for subset, mask in masks.items():
        if mask.sum() == 0:
            cell[0] += 1
            rows.append({"analysis": tag, "reference": reference_name, "subset": subset, "windows": 0,
                         "reading": "n/a (no windows)"})
        else:
            rows += addenda.gain_rows(tag, reference_name, firmware, random_mean, {subset: mask},
                                      patient_index, n_patients, cell)
    return rows


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    t0 = time.time()
    hashes_before = {k: sha256(v) for k, v in P.items()}
    v2 = pd.read_csv(P["features"])
    edge_oof = pd.read_csv(P["edge_oof"]).sort_values("row_index").reset_index(drop=True)
    spline_oof = pd.read_csv(P["spline_oof"]).sort_values("row_index").reset_index(drop=True)
    logreg_oof = pd.read_csv(P["logreg_oof"]).sort_values("row_index").reset_index(drop=True)
    for frame in (edge_oof, spline_oof, logreg_oof):
        if not (np.array_equal(frame["row_index"], np.arange(len(v2)))
                and np.array_equal(frame["recording_id"].astype(str), v2["recording_id"].astype(str))):
            raise SystemExit("prediction tables are not row-aligned with window_features.csv")
    if not np.array_equal(edge_oof["fold"], spline_oof["fold"]):
        raise SystemExit("edge and spline folds differ")
    y = v2["label"].to_numpy(np.int64)
    patients = v2["patient_id"].astype(str).to_numpy()
    _, pidx = np.unique(patients, return_inverse=True)
    n_pat = int(pidx.max()) + 1
    prob = {"edge_kan": edge_oof["y_prob"].to_numpy(float), "spline_network": spline_oof["y_prob"].to_numpy(float),
            "logreg": logreg_oof["y_prob"].to_numpy(float)}
    folds = edge_oof["fold"].to_numpy()
    raw = v2.loc[:, V16].to_numpy(np.float32)
    res = {}

    # H1
    res["H1_table_I"] = {name: metrics_row(y, p) for name, p in prob.items()}
    res["H1_table_I"]["edge_kan"]["parameters"] = tek.parameter_count(16)
    res["H1_table_I"]["spline_network"]["parameters"] = 321
    print("H1 done", flush=True)

    # H2
    point, draws = lm.group_bootstrap(y, prob, patients, N_BOOT, BOOT_SEED)
    h2 = {}
    for name in prob:
        h2[name] = {m: {"point": float(point[name][m]), "ci": list(lm.percentile_ci(draws[name][m]))}
                    for m in ("auc", "recall", "specificity")}
    for other in ("logreg", "spline_network"):
        h2[f"edge_minus_{other}"] = {}
        for m in ("auc", "recall", "specificity"):
            lo, hi = lm.percentile_ci(draws["edge_kan"][m] - draws[other][m])
            h2[f"edge_minus_{other}"][m] = {"difference": float(point["edge_kan"][m] - point[other][m]), "ci": [lo, hi],
                                            "reading": "differs" if lo > 0 or hi < 0 else "no difference detected"}
    res["H2_bootstrap"] = h2
    print("H2 done", flush=True)

    # H3
    res["H3_folds"] = {name: [{"fold": int(f), **{k: v for k, v in metrics_row(y[folds == f], p[folds == f]).items()
                                                   if k in ("auc", "recall", "specificity")}}
                              for f in sorted(np.unique(folds))] for name, p in prob.items() if name != "logreg"}
    print("H3 done", flush=True)

    # H4 seeds and H10 out-of-fold fold models (seed 42 also feeds A3)
    import torch  # noqa: F401
    from sklearn.preprocessing import StandardScaler
    seed_rows, fold_models = [], {}
    stored_edge = prob["edge_kan"]
    for seed in SEEDS:
        tk.SEED = seed
        p_seed = np.full(len(y), np.nan)
        for f in sorted(np.unique(folds)):
            test = folds == f
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(raw[~test]).astype(np.float32)
            x_te = scaler.transform(raw[test]).astype(np.float32)
            net, _ = tek.train_edge_kan(x_tr, y[~test])
            p_seed[test] = tek.torch_probabilities(net, x_te)[0]
            if seed == 42:
                fold_models[int(f)] = (EdgeModel.from_params(fold_params(net, scaler)), test)
        m = lm.binary_metrics(y, p_seed, THRESHOLD)
        seed_rows.append({"seed": seed, "auc": m["auc"], "recall": m["recall"], "specificity": m["specificity"],
                          "max_abs_diff_vs_stored_oof": float(np.max(np.abs(p_seed - stored_edge))),
                          "flag_differences_vs_stored_oof": int(np.sum((p_seed >= .5) != (stored_edge >= .5)))})
        print(f"H4 seed {seed}: AUC {m['auc']:.4f} recall {m['recall']:.4f} spec {m['specificity']:.4f} "
              f"(max |dP| vs stored {seed_rows[-1]['max_abs_diff_vs_stored_oof']:.2e}) {time.time() - t0:.0f}s", flush=True)
    tk.SEED = 42
    seeds = pd.DataFrame(seed_rows)
    res["H4_seeds"] = {"runs": seed_rows, **{f"{k}_{s}": float(getattr(seeds[k], s)()) for k in ("auc", "recall", "specificity")
                                             for s in ("mean", "std", "min", "max")}}

    # H5
    distance = v2["median_spo2"].to_numpy(float) - 90.0
    h5 = {}
    for name in ("edge_kan", "spline_network"):
        p = prob[name]
        pred = p >= THRESHOLD
        bins = []
        for lo, hi in las.DISTANCE_BINS:
            mask = (distance >= lo) & (distance < hi)
            errors = int(np.sum(pred[mask] != (y[mask] == 1)))
            bins.append({"bin": las.distance_label(lo, hi), "n": int(mask.sum()), "errors": errors,
                         "error_type": "FN" if hi <= 0 else "FP", "error_rate": errors / mask.sum()})
        per = pd.DataFrame({"rec": v2["recording_id"], "fp": (y == 0) & pred, "fn": (y == 1) & ~pred}).groupby("rec").sum()
        artefact = pd.read_csv(P["artefact_rows"])["v2_row_index"].to_numpy()
        on = np.zeros(len(y), bool)
        on[artefact] = True
        h5[name] = {"distance_bins": bins, "false_positives": int(per.fp.sum()),
                    "recordings_holding_50pct_fp": lm.groups_holding_share(per.fp.to_numpy(), 0.5),
                    "recordings_total": int(len(per)),
                    "frozen_loop_rows": int(on.sum()), "frozen_loop_recordings": int(v2.loc[on, "recording_id"].nunique()),
                    "auc_other_rows": float(lm.weighted_auc(y[~on], p[~on])), "auc_frozen_rows": float(lm.weighted_auc(y[on], p[on]))}
    res["H5_errors"] = h5
    print("H5 done", flush=True)

    # H6
    s1 = pd.read_csv(P["s1_windows"])
    m = s1.loc[s1["status"] == "matched"].copy()
    rows_idx = m["row_index"].astype(int).to_numpy()
    if not np.allclose(m["y_prob_tinykan"].to_numpy(float), prob["spline_network"][rows_idx], atol=1e-8):
        raise SystemExit("H6: s1_windows.csv does not match the stored spline OOF")
    truth = sc.below90(m["SaO2_draw"].to_numpy(float))
    y_true = m["y_true"].to_numpy(np.int64)
    stats = {}
    for name in ("edge_kan", "spline_network"):
        flag = (prob[name][rows_idx] >= THRESHOLD).astype(np.int64)
        stats[f"{name}_recall_vs_sao2"] = (lambda f: lambda w=None: sc.sens_spec(f, truth, w)[0])(flag)
        stats[f"{name}_specificity_vs_sao2"] = (lambda f: lambda w=None: sc.sens_spec(f, truth, w)[1])(flag)
        stats[f"{name}_recall_vs_label"] = (lambda f: lambda w=None: sc.sens_spec(f, y_true, w)[0])(flag)
        stats[f"{name}_specificity_vs_label"] = (lambda f: lambda w=None: sc.sens_spec(f, y_true, w)[1])(flag)
    cis, n_groups = sc.bootstrap(m["patient_id"].to_numpy(), stats)
    res["H6_sao2_draws"] = {"matched_draws": int(len(m)), "point": {k: fn() for k, fn in stats.items()}, "bootstrap": cis,
                            "patients": n_groups}
    print("H6 done", flush=True)

    # H7
    recording = v2["recording_id"].astype(str).to_numpy()
    start = v2["start_time_seconds"].to_numpy(float)
    lag_rows = []
    for k, lag in enumerate(addenda.LAGS):
        i, j = addenda.lag_pairs(recording, start, float(lag))
        p_i = prob["edge_kan"][i]
        point0, d0 = lm.group_bootstrap(y[i], {"p": p_i}, pidx[i], N_BOOT, BOOT_SEED + 100 + k)
        point1, d1 = lm.group_bootstrap(y[j], {"p": p_i}, pidx[i], N_BOOT, BOOT_SEED + 100 + k)
        row = {"lag_s": lag, "paired_windows": int(len(i))}
        for metric in ("auc", "recall", "specificity"):
            lo, hi = lm.percentile_ci(d1["p"][metric] - d0["p"][metric])
            row[f"{metric}_diff"] = float(point1["p"][metric] - point0["p"][metric])
            row[f"{metric}_ci"] = [lo, hi]
        lag_rows.append(row)
    res["H7_label_lag"] = lag_rows
    print("H7 done", flush=True)

    # H8
    recordings = list(pd.unique(edge_oof["recording_id"].astype(str)))
    groups = skin.recording_groupings(recordings)
    h8 = []
    sprob = {"EdgeKAN": prob["edge_kan"], "SplineNetwork": prob["spline_network"]}
    for cell, grouping in enumerate(("monk_dorsal", "monk_forehead", "fitzpatrick", "ita_dorsal_third")):
        band = groups[grouping].reindex(edge_oof["recording_id"].astype(str)).to_numpy()
        rows, bdraws = skin.stratified_bootstrap(y, sprob, pidx, n_pat, band, skin.BOOT_SEED + cell)
        for model in sprob:
            small = min(r["patients"] for r in rows if r["model"] == model and r["band"] in ("dark", "light")) < skin.MIN_PATIENTS
            for metric in ("recall", "specificity", "auc"):
                pd_, dd = bdraws[(model, "dark")]
                pl, dl = bdraws[(model, "light")]
                lo, hi = np.nanpercentile(dd[metric] - dl[metric], [2.5, 97.5])
                h8.append({"grouping": grouping, "model": model, "metric": metric,
                           "dark": float(pd_[metric]), "light": float(pl[metric]),
                           "dark_minus_light": float(pd_[metric] - pl[metric]),
                           "ci": None if small else [float(lo), float(hi)],
                           "role": "PRIMARY" if (grouping, metric) == ("monk_dorsal", "recall") else "secondary",
                           "read": "no contrast read (group under 10 patients)" if small
                           else "differs" if (lo > 0 or hi < 0) else "no difference detected"})
    res["H8_skin_tone"] = h8
    print("H8 done", flush=True)

    # H9 in-sample explanation behaviour, shipped edge header
    edge = EdgeModel(P["edge_header"])
    z = edge.standardize(raw)
    p_in, attr = edge.mean_perturbation(z)
    cand = p_in >= THRESHOLD
    order = lm.firmware_abs_order(attr)
    top1 = order[:, 0]
    top1_delta = attr[np.arange(len(attr)), top1]
    nonzero = np.sum(np.abs(attr) > ZERO, axis=1)
    max_abs = np.max(np.abs(attr), axis=1)
    clamped = edge.all_hidden_clamped(z)
    shares = {}
    for label in (1, 0):
        mask = y == label
        counts = np.bincount(top1[mask], minlength=16) / mask.sum()
        shares[f"top1_share_y{label}"] = {V16[i]: float(counts[i]) for i in np.argsort(-counts)[:4]}
    cases = {}
    for cell, mask in (("TP", (y == 1) & cand), ("FP", (y == 0) & cand), ("FN", (y == 1) & ~cand), ("TN", (y == 0) & ~cand)):
        idx = lm.median_member(p_in, mask)
        cases[cell] = {"row_index": int(idx), "probability": float(p_in[idx]), "top3": [V16[i] for i in order[idx, :3]],
                       "top3_deltas": [float(attr[idx, i]) for i in order[idx, :3]],
                       "nonzero_deltas": int(nonzero[idx]), "cell_size": int(mask.sum())}
    res["H9_in_sample_xai"] = {
        "in_sample_candidates": int(cand.sum()),
        "in_sample_confusion": {"TP": int(((y == 1) & cand).sum()), "FP": int(((y == 0) & cand).sum()),
                                "FN": int(((y == 1) & ~cand).sum()), "TN": int(((y == 0) & ~cand).sum())},
        **shares,
        "candidates_top3_has_zero_delta_slot": float(np.mean(nonzero[cand] < 3)),
        "candidates_all_deltas_zero": float(np.mean(nonzero[cand] == 0)),
        "candidates_top1_lowered_p": float(np.mean(top1_delta[cand] < -ZERO)),
        "candidates_max_abs_delta_below_0_01": float(np.mean(max_abs[cand] < 0.01)),
        "candidates_all_hidden_clamped": float(np.mean(clamped[cand])),
        "probability_range": [float(p_in.min()), float(p_in.max())],
        "distinct_probabilities_3dp_among_candidates": int(len(np.unique(np.round(p_in[cand], 3)))),
        "case_studies": cases}
    print("H9 done", flush=True)

    # H10 AOPC gain intervals: in sample (two references) and out of fold (fold models, two references)
    ref0 = np.median(z[y == 0], axis=0).astype(np.float32).astype(np.float64)
    _, attr_ref0 = apc.perturb_against(edge, z, ref0)
    subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": cand, "noncandidate_p_lt_0_50": ~cand,
               "label_1": y == 1, "label_0": y == 0}
    cell = [0]
    rows = []
    for rname, attribution, reference in (("training_mean", attr, np.zeros(16)), ("label0_fp32", attr_ref0, ref0)):
        fw, rnd = addenda.per_window_aopc(edge, z, attribution, reference)
        rows += safe_gain_rows("in_sample_shipped", rname, fw, rnd, subsets, pidx, n_pat, cell)
        rows += safe_gain_rows("in_sample_shipped", rname, fw, rnd,
                                  {"candidate_clamped": cand & clamped, "candidate_unclamped": cand & ~clamped}, pidx, n_pat, cell)
    print(f"H10 in-sample done {time.time() - t0:.0f}s", flush=True)
    oof_p = np.full(len(y), np.nan)
    oof_vals = {r: (np.full(len(y), np.nan), np.full(len(y), np.nan)) for r in ("training_mean", "label0_fp32")}
    for f, (fm, test) in sorted(fold_models.items()):
        z_tr, z_te = fm.standardize(raw[~test]), fm.standardize(raw[test])
        p_te, a_mean = fm.mean_perturbation(z_te)
        fref0 = np.median(z_tr[y[~test] == 0], axis=0).astype(np.float32).astype(np.float64)
        _, a_ref0 = apc.perturb_against(fm, z_te, fref0)
        oof_p[test] = p_te
        for rname, a, reference in (("training_mean", a_mean, np.zeros(16)), ("label0_fp32", a_ref0, fref0)):
            fw, rnd = addenda.per_window_aopc(fm, z_te, a, reference)
            oof_vals[rname][0][test], oof_vals[rname][1][test] = fw, rnd
    oof_cand = oof_p >= THRESHOLD
    oof_subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": oof_cand, "noncandidate_p_lt_0_50": ~oof_cand,
                   "label_1": y == 1, "label_0": y == 0}
    for rname, (fw, rnd) in oof_vals.items():
        rows += safe_gain_rows("out_of_fold_analysis_models", rname, fw, rnd, oof_subsets, pidx, n_pat, cell)
    res["H10_aopc_gains"] = rows
    res["H10_fold_models_vs_stored_oof"] = {"max_abs_probability_diff": float(np.max(np.abs(oof_p - stored_edge))),
                                            "candidate_flag_differences": int(np.sum(oof_cand != (stored_edge >= .5)))}
    print(f"H10 done {time.time() - t0:.0f}s", flush=True)

    # H11 deletion range (training-mean and label-0 references; targets P and logit)
    targets = {"P": edge.predict, "logit": edge.logit}
    r4_subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": cand,
                  "candidate_clamped": cand & clamped, "candidate_unclamped": cand & ~clamped}
    h11 = []
    for rname, attribution, reference in (("training_mean", attr, np.zeros(16)), ("label0_fp32", attr_ref0, ref0)):
        orders = {k: v for k, v in r4.ranked_orders(attribution, seed=las.AOPC_SEED, random_repeats=las.AOPC_RANDOM_REPEATS).items()
                  if k == "firmware_absolute" or k.startswith("random_")}
        for tname, target in targets.items():
            dele, ins, _ = r4.ranking_aopc(target, z, orders, reference)
            upper, lower = r4.greedy_bounds(target, z, reference)
            rng = upper - lower
            for sub, mask in r4_subsets.items():
                if mask.sum() == 0:
                    h11.append({"reference": rname, "target": tname, "subset": sub, "windows": 0, "reading": "n/a (no windows)"})
                    continue
                has = mask & (rng >= r4.MIN_RANGE)
                counts = lm.group_bootstrap_counts(pidx, n_pat, N_BOOT, r4.BOOT_SEED + 500 + len(h11))
                g, lo, hi = r4.bootstrap_mean(dele["firmware_absolute"] - dele["random_mean"], mask, pidx, counts)
                row = {"reference": rname, "target": tname, "subset": sub, "windows": int(mask.sum()),
                       "share_no_range": float(np.mean(rng[mask] < r4.MIN_RANGE)) if mask.any() else None,
                       "deletion_firmware": float(dele["firmware_absolute"][mask].mean()) if mask.any() else None,
                       "deletion_random": float(dele["random_mean"][mask].mean()) if mask.any() else None,
                       "deletion_gain": g, "deletion_gain_ci": [lo, hi], "reading": r4.reading(lo, hi),
                       "windows_with_range": int(has.sum())}
                if has.sum() >= 50:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        f_fw = (dele["firmware_absolute"] - lower) / rng
                        f_rd = (dele["random_mean"] - lower) / rng
                    g2, lo2, hi2 = r4.bootstrap_mean(np.where(has, f_fw - f_rd, 0.0), has, pidx, counts)
                    row.update(fraction_firmware=float(f_fw[has].mean()), fraction_random=float(f_rd[has].mean()),
                               fraction_gain=g2, fraction_gain_ci=[lo2, hi2], fraction_reading=r4.reading(lo2, hi2))
                h11.append(row)
    res["H11_deletion_range"] = h11
    print(f"H11 done {time.time() - t0:.0f}s", flush=True)

    # H12 PTT-PPG subject 1
    ptt = pd.read_csv(P["ptt_windows"])
    ptt["p_edge"] = edge.predict(edge.standardize(ptt.loc[:, V16].to_numpy(np.float32)))
    res["H12_ptt_ppg"] = {mapping: {act: {"windows": int(len(g)), "mean_p_edge": float(g.p_edge.mean()),
                                          "mean_p_spline_network": float(g.p_v2.mean()),
                                          "edge_candidates": int((g.p_edge >= .5).sum())}
                                    for act, g in block.groupby("activity")}
                          for mapping, block in ptt.groupby("mapping")}
    print("H12 done", flush=True)

    if {k: sha256(v) for k, v in P.items()} != hashes_before:
        raise SystemExit("an input changed during the run")
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "aopc_gain_intervals.csv", index=False)
    pd.DataFrame(h11).to_csv(OUT / "deletion_range.csv", index=False)
    pd.DataFrame(h8).to_csv(OUT / "skin_tone.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(OUT / "label_lag.csv", index=False)
    seeds.to_csv(OUT / "seeds.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(__file__), "inputs_sha256": hashes_before, "runtime_s": time.time() - t0, **res},
        indent=2, default=float) + "\n", encoding="utf-8")
    print(f"written {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

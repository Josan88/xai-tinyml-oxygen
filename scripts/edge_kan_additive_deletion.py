"""Edge KAN: deletion test of an additive (baseline Shapley) ranking
(docs/edge-kan-additive-deletion-preregistration.md).

Host only, write-once (outputs/edge-kan-v1/additive-deletion/). Reuses the H10 code of
scripts/edge_kan_paper_analyses.py unchanged and swaps the attribution matrix: mean-perturbation deltas
(shipped, recomputed as a check against H10) and baseline Shapley values from 32 fixed permutations.

  python scripts/edge_kan_additive_deletion.py            # the registered run
  python scripts/edge_kan_additive_deletion.py --self-test  # estimator checks on synthetic inputs only
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from datetime import datetime, timezone
import json
from math import factorial
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import edge_kan_paper_analyses as eka  # noqa: E402
import analyze_physio_concept as apc  # noqa: E402
import analyze_panel_addenda_20261110 as addenda  # noqa: E402
import lamaakal_metrics as lm  # noqa: E402
import train_edge_kan as tek  # noqa: E402
import train_tinykan_v2 as tk  # noqa: E402

ROOT = eka.ROOT
OUT = ROOT / "outputs/edge-kan-v1/additive-deletion"
PREREG = ROOT / "docs/edge-kan-additive-deletion-preregistration.md"
H10_SUMMARY = ROOT / "outputs/edge-kan-v1/paper-analyses/summary.json"
SEED = 0x9E3779B9          # kan_additive_attribution.h, KAN_ADDITIVE_SEED
PAIRS = 16                 # KAN_ADDITIVE_PAIRS: 32 orderings
CHECK_WINDOWS = 100
CHECK_SEED = 20261113
SUM_TOL = 1e-9


def xorshift32(s):
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s & 0xFFFFFFFF


def permutations(n, pairs=PAIRS, seed=SEED):
    """The device's orderings: each Fisher-Yates shuffle (from the identity) followed by its reverse."""
    s, out = seed, []
    for _ in range(pairs):
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            s = xorshift32(s)
            j = s % (i + 1)
            p[i], p[j] = p[j], p[i]
        out += [p, p[::-1]]
    return out


def shapley(model, z, reference, pairs=PAIRS):
    """Baseline Shapley values against reference from the fixed orderings: (phi, P(x), P(reference))."""
    z = np.asarray(z, np.float64)
    n, d = z.shape
    ref = np.broadcast_to(np.asarray(reference, np.float64), (n, d))
    base = model.predict(np.array(ref))
    score = model.predict(z)
    phi = np.zeros((n, d))
    perms = permutations(d, pairs)
    for p in perms:
        cur = np.array(ref)
        prev = base
        for step, i in enumerate(p):
            cur[:, i] = z[:, i]
            v = score if step == d - 1 else model.predict(cur)
            phi[:, i] += v - prev
            prev = v
    phi /= len(perms)
    return phi, score, base


def exact_shapley(model, z_row, reference):
    d = len(z_row)
    masks = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(bool)
    v = model.predict(np.where(masks, z_row, reference))
    size = masks.sum(1)
    w = np.array([factorial(s) * factorial(d - 1 - s) / factorial(d) for s in range(d)])
    phi = np.zeros(d)
    for i in range(d):
        on = masks[:, i]
        phi[i] = (v[on] * w[size[on] - 1]).sum() - (v[~on] * w[size[~on]]).sum()
    return phi


def self_test():
    """Synthetic inputs only: additivity, the device orderings, and convergence towards exact Shapley."""
    model = eka.EdgeModel(eka.P["edge_header"])
    z = np.random.default_rng(0).normal(0, 1.2, (6, 16))
    ref = np.zeros(16)
    phi, score, base = shapley(model, z, ref)
    gap = float(np.max(np.abs(phi.sum(1) - (score - base))))
    assert gap < SUM_TOL, gap
    perms = permutations(16, 2)
    assert perms[1] == perms[0][::-1] and sorted(perms[0]) == list(range(16))
    errs = {pairs: float(np.max(np.abs(shapley(model, z[:2], ref, pairs)[0] - np.array([exact_shapley(model, r, ref) for r in z[:2]]))))
            for pairs in (2, 16)}
    print(json.dumps({"sum_gap": gap, "max_error_vs_exact_by_pairs": errs}, indent=2))


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    t0 = time.time()
    inputs = {k: eka.sha256(v) for k, v in eka.P.items() if k in ("features", "edge_header", "edge_oof")}
    v2 = pd.read_csv(eka.P["features"])
    edge_oof = pd.read_csv(eka.P["edge_oof"]).sort_values("row_index").reset_index(drop=True)
    if not (np.array_equal(edge_oof["row_index"], np.arange(len(v2)))
            and np.array_equal(edge_oof["recording_id"].astype(str), v2["recording_id"].astype(str))):
        raise SystemExit("prediction table is not row-aligned with window_features.csv")
    y = v2["label"].to_numpy(np.int64)
    _, pidx = np.unique(v2["patient_id"].astype(str).to_numpy(), return_inverse=True)
    n_pat = int(pidx.max()) + 1
    folds = edge_oof["fold"].to_numpy()
    stored_edge = edge_oof["y_prob"].to_numpy(float)
    raw = v2.loc[:, eka.V16].to_numpy(np.float32)
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "preregistration_sha256": eka.sha256(PREREG), "script_sha256": eka.sha256(Path(__file__)),
           "inputs_sha256": inputs, "estimator": {"orderings": 2 * PAIRS, "seed": SEED}}

    # In sample, shipped header
    edge = eka.EdgeModel(eka.P["edge_header"])
    z = edge.standardize(raw)
    p_in, attr_mp = edge.mean_perturbation(z)
    cand = p_in >= eka.THRESHOLD
    ref0 = np.median(z[y == 0], axis=0).astype(np.float32).astype(np.float64)
    _, attr_mp0 = apc.perturb_against(edge, z, ref0)
    subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": cand, "noncandidate_p_lt_0_50": ~cand,
               "label_1": y == 1, "label_0": y == 0}
    cell = [0]
    rows, sums = [], {}
    for rname, mp, reference in (("training_mean", attr_mp, np.zeros(16)), ("label0_fp32", attr_mp0, ref0)):
        phi, score, base = shapley(edge, z, reference)
        sums[f"in_sample_{rname}"] = float(np.max(np.abs(phi.sum(1) - (score - base))))
        start = cell[0]
        fw, rnd = addenda.per_window_aopc(edge, z, mp, reference)
        rows += [dict(r, attribution="mean_perturbation") for r in eka.safe_gain_rows("in_sample_shipped", rname, fw, rnd, subsets, pidx, n_pat, cell)]
        cell[0] = start  # same bootstrap draws for both attributions of a cell
        fw, rnd = addenda.per_window_aopc(edge, z, phi, reference)
        rows += [dict(r, attribution="additive_shapley") for r in eka.safe_gain_rows("in_sample_shipped", rname, fw, rnd, subsets, pidx, n_pat, cell)]
        if rname == "training_mean":
            rng = np.random.default_rng(CHECK_SEED)
            pick = rng.choice(np.flatnonzero(cand), size=min(CHECK_WINDOWS, int(cand.sum())), replace=False)
            exact = np.array([exact_shapley(edge, z[i], reference) for i in pick])
            err = np.abs(phi[pick] - exact)
            top = lambda a: set(lm.firmware_abs_order(a[None])[0, :3])  # noqa: E731
            res["estimator_check"] = {"windows": int(len(pick)), "median_abs_error": float(np.median(err)),
                                      "max_abs_error": float(err.max()),
                                      "top3_set_agrees": float(np.mean([top(phi[i]) == top(e) for i, e in zip(pick, exact)]))}
        print(f"in sample {rname} done {time.time() - t0:.0f}s", flush=True)

    # Out of fold: seed-42 fold models, as H10
    import torch  # noqa: F401
    from sklearn.preprocessing import StandardScaler
    tk.SEED = 42
    oof_p = np.full(len(y), np.nan)
    vals = {(a, r): (np.full(len(y), np.nan), np.full(len(y), np.nan))
            for a in ("mean_perturbation", "additive_shapley") for r in ("training_mean", "label0_fp32")}
    for f in sorted(np.unique(folds)):
        test = folds == f
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(raw[~test]).astype(np.float32)
        net, _ = tek.train_edge_kan(x_tr, y[~test])
        fm = eka.EdgeModel.from_params(eka.fold_params(net, scaler))
        z_tr, z_te = fm.standardize(raw[~test]), fm.standardize(raw[test])
        p_te, a_mean = fm.mean_perturbation(z_te)
        fref0 = np.median(z_tr[y[~test] == 0], axis=0).astype(np.float32).astype(np.float64)
        _, a_ref0 = apc.perturb_against(fm, z_te, fref0)
        oof_p[test] = p_te
        for rname, mp, reference in (("training_mean", a_mean, np.zeros(16)), ("label0_fp32", a_ref0, fref0)):
            phi, score, base = shapley(fm, z_te, reference)
            sums[f"fold{f}_{rname}"] = float(np.max(np.abs(phi.sum(1) - (score - base))))
            for aname, a in (("mean_perturbation", mp), ("additive_shapley", phi)):
                fw, rnd = addenda.per_window_aopc(fm, z_te, a, reference)
                vals[(aname, rname)][0][test], vals[(aname, rname)][1][test] = fw, rnd
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    oof_cand = oof_p >= eka.THRESHOLD
    oof_subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": oof_cand, "noncandidate_p_lt_0_50": ~oof_cand,
                   "label_1": y == 1, "label_0": y == 0}
    for rname in ("training_mean", "label0_fp32"):
        start = cell[0]
        for aname in ("mean_perturbation", "additive_shapley"):
            cell[0] = start
            fw, rnd = vals[(aname, rname)]
            rows += [dict(r, attribution=aname) for r in eka.safe_gain_rows("out_of_fold_analysis_models", rname, fw, rnd, oof_subsets, pidx, n_pat, cell)]
    res["fold_models_vs_stored_oof"] = {"max_abs_probability_diff": float(np.max(np.abs(oof_p - stored_edge))),
                                        "candidate_flag_differences": int(np.sum(oof_cand != (stored_edge >= .5)))}
    res["additivity_max_gap"] = sums
    if max(sums.values()) > SUM_TOL:
        raise SystemExit(f"additivity check failed: {sums}")

    # Mean-perturbation rows must reproduce H10 (point gains; intervals depend on the bootstrap cell order)
    h10 = json.loads(H10_SUMMARY.read_text())["H10_aopc_gains"]
    key = lambda r: (r["analysis"], r["reference"], r["subset"])  # noqa: E731
    stored = {key(r): r for r in h10}
    mismatch = [key(r) for r in rows if r["attribution"] == "mean_perturbation" and key(r) in stored
                and abs(r["gain"] - stored[key(r)]["gain"]) > 1e-9]
    res["mean_perturbation_reproduces_h10"] = not mismatch
    if mismatch:
        raise SystemExit(f"mean-perturbation gains do not reproduce H10: {mismatch}")

    res["gains"] = rows
    primary = [r for r in rows if r["attribution"] == "additive_shapley" and r["analysis"] == "in_sample_shipped"
               and r["reference"] == "training_mean" and r["subset"] == "candidate_p_ge_0_50"][0]
    res["primary"] = primary
    res["runtime_s"] = time.time() - t0
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "aopc_gain_intervals.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"primary": primary, "estimator_check": res["estimator_check"],
                      "fold_models_vs_stored_oof": res["fold_models_vs_stored_oof"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        self_test()
    else:
        main()

"""Deletion-test range and target for R4 (host only).

Pre-registration: docs/deletion-range-preregistration.md (ef0cdfd). Shipped v2 header, in sample, 48,662 rows.
D1 all-reference value; D2 greedy oracle / reverse-greedy AOPC bounds; D3 each ranking as a fraction of the range;
D4 insertion; D5 the same on the unclamped output pre-activation. Writes a new write-once folder.

  py -3.14 scripts/analyze_deletion_range_r4.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from analyze_panel_addenda_20261110 import bootstrap_mean, reading  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs" / "deletion-range-r4-v1"
P = {"v2_features": ROOT / "outputs/openox-v2/window_features.csv",
     "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
     "label0_aopc": ROOT / "outputs/xai-reference-label0-v1/aopc_deletion.csv",
     "preregistration": ROOT / "docs/deletion-range-preregistration.md"}
V16 = list(las.V2_FEATURES)
THRESHOLD, CLAMP, N_BOOT, BOOT_SEED, MIN_RANGE, TOL = 0.50, 3.0, 2000, 20261112, 0.01, 1e-8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def greedy_bounds(target, scaled, reference):
    """Per-window deletion AOPC of the greedy oracle (max) and reverse greedy (min) orders on `target`."""
    rows, d = scaled.shape
    t0 = target(scaled)
    out = {}
    for name, pick in (("upper", np.argmin), ("lower", np.argmax)):
        current = np.array(scaled, dtype=np.float64, copy=True)
        remaining = np.ones((rows, d), bool)
        total = np.zeros(rows)
        for _ in range(d):
            trial = np.full((rows, d), np.nan)
            for f in range(d):
                changed = current.copy()
                changed[:, f] = reference[f]
                trial[:, f] = target(changed)
            masked = np.where(remaining, trial, np.inf if pick is np.argmin else -np.inf)
            chosen = pick(masked, axis=1)
            current[np.arange(rows), chosen] = reference[chosen]
            remaining[np.arange(rows), chosen] = False
            total += t0 - masked[np.arange(rows), chosen]
        out[name] = total / d
    return out["upper"], out["lower"]


def ranking_aopc(target, scaled, orders, reference):
    deletion, insertion, ins_curves = {}, {}, {}
    rows, d = scaled.shape
    ref_full = np.tile(reference, (rows, 1))
    for name, order in orders.items():
        curve = deletion_curves(target, scaled, order, reference)
        deletion[name] = np.mean(curve[:, :1] - curve[:, 1:], axis=1)
        inserted = ref_full.copy()
        ic = np.empty((rows, d + 1))
        ic[:, 0] = target(inserted)
        for step in range(d):
            f = order[:, step]
            inserted[np.arange(rows), f] = scaled[np.arange(rows), f]
            ic[:, step + 1] = target(inserted)
        insertion[name] = np.mean(ic[:, 1:] - ic[:, :1], axis=1)
        ins_curves[name] = ic
    rnd = [n for n in orders if n.startswith("random_")]
    for store in (deletion, insertion):
        store["random_mean"] = np.mean([store[n] for n in rnd], axis=0)
    ins_curves["random_mean"] = np.mean([ins_curves[n] for n in rnd], axis=0)
    return deletion, insertion, ins_curves


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; outputs are write-once")
    protected_before = apc.protected_hashes()
    v2 = pd.read_csv(P["v2_features"])
    if len(v2) != 48662:
        raise SystemExit("not the 48,662-window export")
    y = v2["label"].to_numpy(np.int64)
    _, pidx = np.unique(v2["patient_id"].astype(str).to_numpy(), return_inverse=True)
    n_pat = int(pidx.max()) + 1
    model = models.HeaderKAN(P["v2_header"])
    scaled = model.standardize(v2.loc[:, V16].to_numpy(np.float32))
    prob, attr_mean = model.mean_perturbation(scaled)
    ref0 = np.median(scaled[y == 0], axis=0).astype(np.float32).astype(np.float64)
    _, attr_ref0 = apc.perturb_against(model, scaled, ref0)
    candidate = prob >= THRESHOLD
    clamped = model.output_preactivation(scaled) >= CLAMP
    subsets = {"all": np.ones(len(y), bool), "candidate_p_ge_0_50": candidate,
               "candidate_clamped": candidate & clamped, "candidate_unclamped": candidate & ~clamped}
    refs = {"training_mean": (np.zeros(16), attr_mean), "label0_fp32": (ref0, attr_ref0)}
    targets = {"P": model.predict, "u2_preactivation": model.output_preactivation}

    # M0.1 / M0.2
    stored = pd.read_csv(P["label0_aopc"])
    stored = stored[stored.model == "v2"]
    m0 = {"m0_2": {"candidates": int(candidate.sum()), "candidates_clamped": int((candidate & clamped).sum())}}
    m0["m0_2"]["pass"] = m0["m0_2"]["candidates"] == 28661 and m0["m0_2"]["candidates_clamped"] == 11368
    worst = 0.0
    results = {}
    for rname, (reference, attribution) in refs.items():
        orders = {k: v for k, v in ranked_orders(attribution, seed=las.AOPC_SEED,
                                                 random_repeats=las.AOPC_RANDOM_REPEATS).items()
                  if k == "firmware_absolute" or k.startswith("random_")}
        for tname, target in targets.items():
            results[(rname, tname)] = (orders, ranking_aopc(target, scaled, orders, reference))
        dele = results[(rname, "P")][1][0]
        for sub in ("all", "candidate_p_ge_0_50"):
            for rk in ("firmware_absolute", "random_mean"):
                want = stored[(stored.reference == rname) & (stored.subset == sub) & (stored.ranking == rk)]
                worst = max(worst, abs(float(want.deletion_aopc_window_mean.iloc[0]) - float(dele[rk][subsets[sub]].mean())))
    m0["m0_1"] = {"max_abs_diff": worst, "tolerance": TOL, "pass": bool(worst <= TOL)}
    print(json.dumps(m0, indent=1), flush=True)
    if not all(c["pass"] for c in m0.values()):
        raise SystemExit("M0 failed; nothing written")

    cell = [0]

    def boot(values, mask):
        counts = lm.group_bootstrap_counts(pidx, n_pat, N_BOOT, BOOT_SEED + cell[0])
        cell[0] += 1
        return bootstrap_mean(values, mask, pidx, counts)

    d1, rows, curves = [], [], []
    for (rname, tname), (orders, (dele, ins, icurves)) in results.items():
        reference = refs[rname][0]
        target = targets[tname]
        d1.append({"reference": rname, "target": tname,
                   "value_at_all_reference_input": float(np.asarray(target(reference[None, :]))[0])})
        upper, lower = greedy_bounds(target, scaled, reference)
        rng = upper - lower
        for sub, mask in subsets.items():
            has = mask & (rng >= MIN_RANGE)
            row = {"reference": rname, "target": tname, "subset": sub, "windows": int(mask.sum()),
                   "share_no_range": float(np.mean(rng[mask] < MIN_RANGE)),
                   "range_median": float(np.median(rng[mask])), "upper_mean": float(upper[mask].mean()),
                   "lower_mean": float(lower[mask].mean())}
            for rk in ("firmware_absolute", "random_mean"):
                row[f"deletion_{rk}"] = float(dele[rk][mask].mean())
                row[f"insertion_{rk}"] = float(ins[rk][mask].mean())
            g, lo, hi = boot(dele["firmware_absolute"] - dele["random_mean"], mask)
            row.update(deletion_gain=g, deletion_gain_lo=lo, deletion_gain_hi=hi, deletion_gain_reading=reading(lo, hi))
            g, lo, hi = boot(ins["firmware_absolute"] - ins["random_mean"], mask)
            row.update(insertion_gain=g, insertion_gain_lo=lo, insertion_gain_hi=hi, insertion_gain_reading=reading(lo, hi))
            row["windows_with_range"] = int(has.sum())
            if has.sum() >= 50:
                with np.errstate(invalid="ignore", divide="ignore"):
                    f_fw = (dele["firmware_absolute"] - lower) / rng
                    f_rd = (dele["random_mean"] - lower) / rng
                row["fraction_firmware"] = float(f_fw[has].mean())
                row["fraction_random"] = float(f_rd[has].mean())
                g, lo, hi = boot(np.where(has, f_fw - f_rd, 0.0), has)
                row.update(fraction_gain=g, fraction_gain_lo=lo, fraction_gain_hi=hi, fraction_gain_reading=reading(lo, hi))
            rows.append(row)
            for rk in ("firmware_absolute", "random_mean"):
                for k, v in enumerate(icurves[rk][mask].mean(axis=0)):
                    curves.append({"reference": rname, "target": tname, "subset": sub, "ranking": rk, "k": k,
                                   "insertion_mean": float(v)})
    table = pd.DataFrame(rows)
    sel = lambda t: table[(table.reference == "training_mean") & (table.target == t) & (table.subset == "candidate_p_ge_0_50")].iloc[0]
    p_row, u_row = sel("P"), sel("u2_preactivation")
    readings = {
        "R-A": "cannot discriminate on most flagged windows" if p_row.share_no_range > 0.5 else "has range on flagged windows",
        "R-A_share_no_range": float(p_row.share_no_range),
        "R-B": p_row.get("fraction_gain_reading", "too few windows with range"),
        "R-C": u_row.deletion_gain_reading,
    }
    if apc.protected_hashes() != protected_before:
        raise SystemExit("a protected input changed during the run")
    OUT.mkdir(parents=True)
    table.to_csv(OUT / "range_and_gains.csv", index=False)
    pd.DataFrame(d1).to_csv(OUT / "all_reference_value.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "insertion_curves.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": "docs/deletion-range-preregistration.md (ef0cdfd)", "m0": m0, "readings": readings,
        "bootstrap": {"unit": "patient", "patients": n_pat, "resamples": N_BOOT, "seed_base": BOOT_SEED},
        "inputs_sha256": {k: sha256(v) for k, v in P.items()}, "script_sha256": sha256(__file__)}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 260)
    print(pd.DataFrame(d1).to_string(index=False))
    cols = ["reference", "target", "subset", "windows", "share_no_range", "range_median", "deletion_firmware_absolute",
            "deletion_random_mean", "deletion_gain", "deletion_gain_lo", "deletion_gain_hi", "fraction_firmware",
            "fraction_random", "fraction_gain", "fraction_gain_lo", "fraction_gain_hi", "insertion_gain",
            "insertion_gain_lo", "insertion_gain_hi"]
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))
    print(json.dumps(readings, indent=1))


if __name__ == "__main__":
    main()

"""Host-only sanity check: gradient x input versus the shipped mean-perturbation explainer.

Question: on the frozen shipped TinyKAN headers (v1 reported, v2 comparison), would
gradient x input have told the operator the same thing as the on-device explainer?

Both explainers use the same baseline. Standardized features have training mean 0, so
mean-perturbation is delta_i = P(x) - P(x with x_i = 0) and gradient x input is
g_i = x_i * dP/dx_i, the first-order Taylor estimate of delta_i. Disagreement between the
two is therefore a measure of how non-linear the learned splines are between x_i and 0.

Nothing here changes the shipped explainer, firmware, a header, or an existing output.
Gradients are analytic (chain rule through the clamped cubic B-spline basis of
kan_inference.h) and are checked against central finite differences. Deletion uses the
same mean-replacement curves as scripts/evaluate_xai_faithfulness.py, which builds a
first-step advantage into mean-perturbation; the gradient ranking does not get that
advantage, so the deletion comparison favours the shipped explainer by construction.

Run from the repository root:
  C:\\Python314\\python.exe scripts/compare_gradient_vs_perturbation.py
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
ROOT = SCRIPTS.parent

from evaluate_xai_faithfulness import curve_metrics, perturbation_curves, ranked_orders  # noqa: E402
from lamaakal_models import HeaderKAN  # noqa: E402

OUT_DEFAULT = ROOT / "outputs" / "xai-gradient-check-v1"
IR_FEATURES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
V2_FEATURES = IR_FEATURES + ("ratio_of_ratios", "ir_perfusion_index")
P = {
    "v1_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h",
    "v2_header": ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h",
    "v1_features": ROOT / "outputs/openox_canonical_window_features.csv",
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "aopc_v1_inputs": ROOT / "outputs/xai-faithfulness/openox-aopc-v1/inputs_attributions.npz",
    "aopc_v1_summary": ROOT / "outputs/xai-faithfulness/openox-aopc-v1/summary.json",
    "aopc_v2_new_run": ROOT / "outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv",
}
WRITE_ONCE = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/analysis-lamaakal-style-v1",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
)
THRESHOLD = 0.50
AOPC_SEED, AOPC_RANDOM_REPEATS = 20260911, 5
BOOTSTRAP_SEED, N_BOOT = 20261024, 2000
ZERO = 1e-12
FD_WINDOWS, FD_STEP = 2000, 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(path):
    path = Path(path)
    if path.is_file():
        return {path.relative_to(ROOT).as_posix(): sha256(path)}
    return {f.relative_to(ROOT).as_posix(): sha256(f) for f in sorted(path.rglob("*")) if f.is_file()}


# ----------------------------------------------------------------------------- gradient

def _basis_and_slope(model, u, layer):
    """Cubic B-spline basis of kan_inference.h and its derivative with respect to u.

    u is clamped to [-3, 3]; outside the open interval the clamp has zero slope, so the
    basis derivative is zero there (the firmware output is flat in u).
    """
    lo, hi = model.RANGE
    step = float(model.p[f"KAN{layer}_STEP"]) + 1e-10
    clipped = np.clip(u, lo, hi)
    signed = (clipped[..., None] - model.p[f"KAN{layer}_CENTERS"]) / step
    distance = np.abs(signed)
    basis = np.where(distance < 1.0, 2.0 / 3.0 - distance ** 2 + 0.5 * distance ** 3,
                     np.maximum(2.0 - distance, 0.0) ** 3 / 6.0)
    d_basis_d_distance = np.where(distance < 1.0, -2.0 * distance + 1.5 * distance ** 2,
                                  -0.5 * np.maximum(2.0 - distance, 0.0) ** 2)
    inside = ((u > lo) & (u < hi)).astype(np.float64)[..., None]
    slope = d_basis_d_distance * np.sign(signed) / step * inside
    return basis, slope


def probability_gradient(model, scaled):
    """dP/dx for standardized x, analytic through both spline layers."""
    x = np.asarray(scaled, dtype=np.float64)
    w1, b1, t1 = model.p["KAN1_W"], model.p["KAN1_B"], model.p["KAN1_THETA"]
    w2, b2, t2 = model.p["KAN2_W"][0], model.p["KAN2_B"][0], model.p["KAN2_THETA"][0]
    u1 = x @ w1.T + b1                                  # N x H
    basis1, slope1 = _basis_and_slope(model, u1, 1)     # N x H x K
    h = np.sum(basis1 * t1[None], axis=2)               # N x H
    dh_du1 = np.sum(slope1 * t1[None], axis=2)          # N x H
    u2 = h @ w2 + b2                                    # N
    basis2, slope2 = _basis_and_slope(model, u2, 2)     # N x K
    logit = basis2 @ t2
    dlogit_du2 = slope2 @ t2
    probability = 1.0 / (1.0 + np.exp(-logit))
    dp_du2 = probability * (1.0 - probability) * dlogit_du2
    dp_du1 = dp_du2[:, None] * w2[None, :] * dh_du1     # N x H
    return probability, dp_du1 @ w1                     # N x D


def finite_difference_check(model, scaled, rng):
    rows = rng.choice(len(scaled), size=min(FD_WINDOWS, len(scaled)), replace=False)
    x = scaled[rows]
    _, analytic = probability_gradient(model, x)
    numeric = np.empty_like(x)
    for feature in range(x.shape[1]):
        up, down = x.copy(), x.copy()
        up[:, feature] += FD_STEP
        down[:, feature] -= FD_STEP
        numeric[:, feature] = (model.predict(up) - model.predict(down)) / (2 * FD_STEP)
    error = np.abs(analytic - numeric)
    # A clamp edge or knot inside +/- h makes the central difference one-sided; report both.
    return {"windows": int(len(rows)), "step": FD_STEP,
            "max_abs_error": float(error.max()), "p99_abs_error": float(np.quantile(error, 0.99)),
            "max_abs_gradient": float(np.abs(analytic).max())}


# ----------------------------------------------------------------------------- agreement

def firmware_order(attribution):
    return np.argsort(-np.abs(attribution), axis=1, kind="stable")


def spearman_rows(a, b):
    """Per-row Spearman correlation of |a| and |b| (average ranks on ties)."""
    ra = pd.DataFrame(np.abs(a)).rank(axis=1, method="average").to_numpy()
    rb = pd.DataFrame(np.abs(b)).rank(axis=1, method="average").to_numpy()
    # pandas 3 copy-on-write returns read-only arrays, so centre without in-place writes.
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    denominator = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (ra * rb).sum(axis=1) / denominator, np.nan)


def agreement(delta, gxi, probability, features):
    d_order, g_order = firmware_order(delta), firmware_order(gxi)
    rows = np.arange(len(delta))
    top1_same = d_order[:, 0] == g_order[:, 0]
    top3_overlap = np.array([len(set(a) & set(b)) for a, b in zip(d_order[:, :3], g_order[:, :3])]) / 3.0
    top1_sign_same = np.sign(delta[rows, d_order[:, 0]]) == np.sign(gxi[rows, d_order[:, 0]])
    rho = spearman_rows(delta, gxi)
    delta_zero = np.all(np.abs(delta) <= ZERO, axis=1)
    gxi_zero = np.all(np.abs(gxi) <= ZERO, axis=1)
    per_window = pd.DataFrame({
        "probability": probability, "candidate": probability >= THRESHOLD,
        "mp_top1": np.asarray(features)[d_order[:, 0]], "gxi_top1": np.asarray(features)[g_order[:, 0]],
        "top1_same": top1_same, "top3_overlap": top3_overlap, "mp_top1_sign_same_in_gxi": top1_sign_same,
        "spearman_abs": rho, "all_delta_zero": delta_zero, "all_gxi_zero": gxi_zero,
    })
    return per_window


def summarise_agreement(per_window, delta, gxi, version):
    subsets = {"all": np.ones(len(per_window), bool), "candidate_p_ge_0_50": per_window["candidate"].to_numpy(),
               "noncandidate_p_lt_0_50": ~per_window["candidate"].to_numpy()}
    rows = []
    for name, mask in subsets.items():
        live = mask & ~per_window["all_delta_zero"].to_numpy() & ~per_window["all_gxi_zero"].to_numpy()
        flat_d, flat_g = delta[mask].ravel(), gxi[mask].ravel()
        rows.append({
            "version": version, "subset": name, "windows": int(mask.sum()),
            "windows_all_mp_delta_zero": int(per_window["all_delta_zero"][mask].sum()),
            "windows_all_gxi_zero": int(per_window["all_gxi_zero"][mask].sum()),
            "windows_both_nondegenerate": int(live.sum()),
            "top1_agreement": float(per_window["top1_same"][live].mean()) if live.any() else np.nan,
            "top3_overlap_mean": float(per_window["top3_overlap"][live].mean()) if live.any() else np.nan,
            "mp_top1_sign_agreement": float(per_window["mp_top1_sign_same_in_gxi"][live].mean()) if live.any() else np.nan,
            "spearman_abs_median": float(np.nanmedian(per_window["spearman_abs"][live])) if live.any() else np.nan,
            "pearson_delta_vs_gxi_cells": float(np.corrcoef(flat_d, flat_g)[0, 1]) if flat_d.std() > 0 and flat_g.std() > 0 else np.nan,
            "median_abs_delta_minus_gxi": float(np.median(np.abs(flat_d - flat_g))),
            "median_abs_delta": float(np.median(np.abs(flat_d))),
        })
    return rows


def global_top1(per_window, features, version):
    live = ~per_window["all_delta_zero"] & ~per_window["all_gxi_zero"]
    n = int(live.sum())
    mp = per_window.loc[live, "mp_top1"].value_counts()
    gx = per_window.loc[live, "gxi_top1"].value_counts()
    return [{"version": version, "feature": f, "windows": n,
             "mp_top1_share": float(mp.get(f, 0) / n), "gxi_top1_share": float(gx.get(f, 0) / n)}
            for f in features]


# ----------------------------------------------------------------------------- deletion

def deletion_comparison(model, scaled, probability, delta, gxi, recording_ids, version):
    orders = ranked_orders(delta, seed=AOPC_SEED, random_repeats=AOPC_RANDOM_REPEATS)
    orders = {k: v for k, v in orders.items() if k == "firmware_absolute" or k.startswith("random_")}
    orders["gradient_x_input_absolute"] = firmware_order(gxi)
    metrics = {}
    for name, order in orders.items():
        deletion, insertion = perturbation_curves(model.predict, scaled, order)
        metrics[name] = curve_metrics(deletion, insertion)
    random_names = [n for n in orders if n.startswith("random_")]
    metrics["random_mean"] = {k: np.mean([metrics[n][k] for n in random_names], axis=0) for k in metrics[random_names[0]]}
    subsets = {"all": np.ones(len(scaled), bool), "candidate_p_ge_0_50": probability >= THRESHOLD,
               "noncandidate_p_lt_0_50": probability < THRESHOLD}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    rows = []
    for subset, mask in subsets.items():
        for ranking in ("firmware_absolute", "gradient_x_input_absolute", "random_mean"):
            rows.append({"version": version, "subset": subset, "windows": int(mask.sum()), "ranking": ranking,
                         "deletion_aopc_window_mean": float(metrics[ranking]["deletion_aopc"][mask].mean()),
                         "insertion_aopc_window_mean": float(metrics[ranking]["insertion_aopc"][mask].mean())})
        # Paired difference gradient minus firmware, recording-cluster bootstrap.
        diff = metrics["gradient_x_input_absolute"]["deletion_aopc"][mask] - metrics["firmware_absolute"]["deletion_aopc"][mask]
        recs = recording_ids[mask]
        codes, inverse = np.unique(recs, return_inverse=True)
        sums, counts = np.bincount(inverse, weights=diff), np.bincount(inverse)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(codes), len(codes))
            boots.append(sums[pick].sum() / counts[pick].sum())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"version": version, "subset": subset, "windows": int(mask.sum()),
                     "ranking": "difference_gradient_minus_firmware",
                     "deletion_aopc_window_mean": float(diff.mean()), "insertion_aopc_window_mean": np.nan,
                     "ci95_low_recording_bootstrap": float(lo), "ci95_high_recording_bootstrap": float(hi),
                     "recordings": int(len(codes))})
    return rows, {k: float(metrics[k]["deletion_aopc"].mean()) for k in ("firmware_absolute", "random_mean")}


# ----------------------------------------------------------------------------- main

def run_version(version, header, raw, recording_ids, features, rng):
    model = HeaderKAN(header)
    scaled = model.standardize(raw)
    probability, delta = model.mean_perturbation(scaled)
    p_grad, gradient = probability_gradient(model, scaled)
    gxi = scaled * gradient
    checks = {"max_abs_p_gradient_pass_vs_predict": float(np.max(np.abs(p_grad - probability))),
              "finite_difference": finite_difference_check(model, scaled, rng)}
    per_window = agreement(delta, gxi, probability, features)
    agree_rows = summarise_agreement(per_window, delta, gxi, version)
    top_rows = global_top1(per_window, features, version)
    del_rows, recomputed = deletion_comparison(model, scaled, probability, delta, gxi, recording_ids, version)
    checks["recomputed_deletion_aopc_all"] = recomputed
    return dict(scaled=scaled, probability=probability, delta=delta, gxi=gxi, per_window=per_window,
                agree=agree_rows, top=top_rows, deletion=del_rows, checks=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    started = datetime.now(timezone.utc)
    write_once_before = {p.relative_to(ROOT).as_posix(): tree_hashes(p) for p in WRITE_ONCE}
    input_hashes = {k: sha256(v) for k, v in P.items()}
    rng = np.random.Generator(np.random.PCG64(20261024))

    v1 = pd.read_csv(P["v1_features"])
    v1_raw = v1.loc[:, IR_FEATURES].to_numpy(np.float32)
    v2 = pd.read_csv(P["v2_features"])
    v2_raw = v2.loc[:, V2_FEATURES].to_numpy(np.float32)
    results = {
        "v1": run_version("v1", P["v1_header"], v1_raw, v1["recording_id"].astype(str).to_numpy(), IR_FEATURES, rng),
        "v2": run_version("v2", P["v2_header"], v2_raw, v2["recording_id"].astype(str).to_numpy(), V2_FEATURES, rng),
    }

    # Reproduction checks against existing write-once numbers.
    aopc_inputs = np.load(P["aopc_v1_inputs"])
    cited_v1 = json.loads(P["aopc_v1_summary"].read_text(encoding="utf-8"))["summaries"]["all"]["methods"]
    cited_v2 = pd.read_csv(P["aopc_v2_new_run"]).query("subset == 'all'").set_index("ranking")["deletion_aopc_window_mean"]
    r1, r2 = results["v1"], results["v2"]
    checks = {
        "v1_p_vs_write_once_aopc_inputs_max_abs": float(np.max(np.abs(aopc_inputs["probability"] - r1["probability"]))),
        "v1_delta_vs_write_once_aopc_inputs_max_abs": float(np.max(np.abs(aopc_inputs["attribution"] - r1["delta"]))),
        "v1_scaled_vs_write_once_aopc_inputs_max_abs": float(np.max(np.abs(aopc_inputs["standardized_features"] - r1["scaled"]))),
        "v1_firmware_deletion_aopc_recomputed_vs_cited": [r1["checks"]["recomputed_deletion_aopc_all"]["firmware_absolute"],
                                                          cited_v1["firmware_absolute"]["deletion_aopc"]["window_mean"]],
        "v1_random_deletion_aopc_recomputed_vs_cited": [r1["checks"]["recomputed_deletion_aopc_all"]["random_mean"],
                                                        cited_v1["random_mean"]["deletion_aopc"]["window_mean"]],
        "v2_firmware_deletion_aopc_recomputed_vs_lamaakal_pack": [r2["checks"]["recomputed_deletion_aopc_all"]["firmware_absolute"],
                                                                  float(cited_v2["firmware_absolute"])],
        "v2_random_deletion_aopc_recomputed_vs_lamaakal_pack": [r2["checks"]["recomputed_deletion_aopc_all"]["random_mean"],
                                                                float(cited_v2["random_mean"])],
        "v1": r1["checks"], "v2": r2["checks"],
    }
    tolerance = 1e-9
    failures = [k for k in ("v1_p_vs_write_once_aopc_inputs_max_abs", "v1_delta_vs_write_once_aopc_inputs_max_abs")
                if checks[k] > tolerance]
    for k in ("v1_firmware_deletion_aopc_recomputed_vs_cited", "v1_random_deletion_aopc_recomputed_vs_cited",
              "v2_firmware_deletion_aopc_recomputed_vs_lamaakal_pack", "v2_random_deletion_aopc_recomputed_vs_lamaakal_pack"):
        if abs(checks[k][0] - checks[k][1]) > 1e-6:
            failures.append(k)
    for version in ("v1", "v2"):
        if checks[version]["finite_difference"]["p99_abs_error"] > 1e-6:
            failures.append(f"{version}_finite_difference")
    if failures:
        raise RuntimeError(f"Reproduction checks failed: {failures}\n{json.dumps(checks, indent=1, default=float)}")

    out.mkdir(parents=True)
    agree = pd.DataFrame(r1["agree"] + r2["agree"])
    top = pd.DataFrame(r1["top"] + r2["top"])
    deletion = pd.DataFrame(r1["deletion"] + r2["deletion"])
    fmt = dict(index=False, float_format="%.8g", lineterminator="\n")
    agree.to_csv(out / "agreement_summary.csv", **fmt)
    top.to_csv(out / "global_top1_share.csv", **fmt)
    deletion.to_csv(out / "deletion_aopc.csv", **fmt)
    for version in ("v1", "v2"):
        results[version]["per_window"].to_csv(out / f"per_window_{version}.csv.gz", compression="gzip", **fmt)
    source = out / "source"
    source.mkdir()
    for path in (Path(__file__), SCRIPTS / "evaluate_xai_faithfulness.py", SCRIPTS / "lamaakal_models.py"):
        shutil.copy2(path, source / path.name)

    write_once_after = {p.relative_to(ROOT).as_posix(): tree_hashes(p) for p in WRITE_ONCE}
    if write_once_after != write_once_before:
        raise RuntimeError("A write-once directory changed during the run")
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    summary = {
        "purpose": "host-only sanity check of gradient x input against the shipped mean-perturbation explainer; not a new on-device method",
        "created_utc": started.isoformat(), "git_head": head, "python": sys.version, "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(), "command": [sys.executable] + sys.argv,
        "baseline": "training mean (standardized 0) for both explainers",
        "gradient": "analytic dP/dx through clamped cubic B-splines; zero outside the open clamp interval",
        "ranking": "|value| descending, stable tie-break on feature index (firmware bubble sort)",
        "degenerate_rule": f"a window is degenerate for an explainer when every |value| <= {ZERO}",
        "aopc_seed": AOPC_SEED, "aopc_random_repeats": AOPC_RANDOM_REPEATS,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": N_BOOT, "unit": "recording"},
        "input_sha256": input_hashes, "write_once_unchanged": True, "checks": checks,
        "agreement": agree.to_dict(orient="records"), "deletion": deletion.to_dict(orient="records"),
        "limitations": [
            "In-sample: shipped headers evaluated on their own training tables, as in the write-once AOPC run.",
            "Deletion replaces features with the training mean, the same operation that defines mean-perturbation, so the shipped ranking has a built-in first-step advantage.",
            "Agreement is between two explanations of one model; neither is ground truth and neither is a clinical cause.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    (out / "README.md").write_text(
        "# xai-gradient-check-v1\n\nHost-only sanity check: gradient x input versus the shipped mean-perturbation explainer, "
        "on the frozen v1 (reported) and v2 (comparison) headers. Generated by `scripts/compare_gradient_vs_perturbation.py`. "
        "Nothing on the board, in `firmware/`, or in an existing output changed. The shipped explainer stays mean-perturbation.\n",
        encoding="utf-8")
    print(agree.to_string(index=False))
    print(top.to_string(index=False))
    print(deletion.to_string(index=False))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("v1", "v2")}, indent=1, default=float))
    print(json.dumps({"v1": checks["v1"], "v2": checks["v2"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

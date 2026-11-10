"""Device concept regressor for the P5 image: one all-window KAN regressor for estimated ScalcO2 (s_hat).

Host only, no board. Job F2 (30 Oct 2026). The P5 device image (ADR-0008) displays s_hat beside
the P5 risk score. s_hat does not feed the risk score or the explanation.

Recipe: the Stage 4 concept regressor of scripts/analyze_physio_concept.py, imported, not copied
(RECIPE: hidden 12, 8 knots, spline range [-3, 3], 100 epochs; seed 42; Adam lr 0.01, weight decay
1e-4, 1e-5 spline smoothness, grad-norm clip 10, full batch; MSE on the standardized clipped target,
linear output). The only difference from a Stage 4 job is the split: here it is trained once on all
48,662 windows (train = every row), as the P5 all-window classifier was. Inputs are the P5 beat
features from outputs/physio-concept-v1/physio_features.csv.gz; beat-invalid rows are imputed with
the all-window beat-valid mean before the scaler (analyze_physio_concept.impute), so they are 0
after scaling. Target: window-median ScalcO2 clipped to [50, 100] (TARGET_CLIP); ScalcO2 is
calculated from end-tidal O2, not a pulse-oximeter SpO2.

The metrics written here are resubstitution on the training rows, not an accuracy result. The
honest accuracy of this recipe is the cross-fitted Stage 4 table (s4_concept_quality.csv), which
is copied into metrics.json for reference.

Writes a fresh outputs/physio-device-v1/ (refuses an existing directory):
  sat_regressor_model.npz / .json   weights, scaler, imputation mean, target mean/std/clip
  resubstitution_predictions.csv.gz torch s_hat per row (row_index-aligned)
  metrics.json                      resubstitution metrics + the cross-fitted Stage 4 numbers
Run (the interpreter with torch that physio-concept-v1/summary.json records):
  C:\\Python314\\python.exe scripts/train_physio_device_regressor.py
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analyze_physio_concept as apc  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "physio-device-v1"
CONCEPT = ROOT / "outputs" / "physio-concept-v1"
P5_MODEL = CONCEPT / "p5_final_model.npz"
SPEC = {"name": "kanreg_all_windows_seed42", "features": list(apc.P5), **apc.RECIPE, "seed": 42,
        "loss": "MSE on the standardized target, linear output", "train": "all 48,662 windows"}


def regression_metrics(s, t):
    d = s - t
    sd = float(np.std(d, ddof=1))
    row = {"n": int(len(t)), "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d ** 2))),
           "bias": float(np.mean(d)), "bland_altman_low": float(np.mean(d) - 1.959964 * sd),
           "bland_altman_high": float(np.mean(d) + 1.959964 * sd), "pearson_r": float(np.corrcoef(s, t)[0, 1])}
    for name, lo, hi in apc.RANGES:
        m = (t >= lo) & (t < hi)
        row[f"rmse_{name}"] = float(np.sqrt(np.mean(d[m] ** 2)))
    return row


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} exists; refusing to overwrite")
    import torch
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    protected_before = apc.protected_hashes()
    concept_before = apc.las.tree_hashes(CONCEPT)
    v2, _, _, _, _ = apc.load_v2()
    table = apc.load_physio(CONCEPT)
    x5 = apc.feature_matrix(v2, table, apc.P5)                       # float32, NaN on beat-invalid rows
    target_raw = v2["median_spo2"].to_numpy(float)                   # concept target only, never an input
    t = np.clip(target_raw, *apc.TARGET_CLIP)
    y = v2["label"].to_numpy(np.int64)
    valid = table["beat_valid"].to_numpy(bool)
    train = np.ones(apc.N_ROWS, bool)

    # _kanreg_job (analyze_physio_concept.py) with train = every row, keeping the weights.
    apc.guard_inputs(SPEC["features"])
    xi, means = apc.impute(x5, train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(xi[train]).astype(np.float32)
    t_mean, t_std = float(t[train].mean()), float(t[train].std())
    model = apc.train_kan_regressor(x_train, ((t[train] - t_mean) / t_std).astype(np.float32), SPEC["hidden"],
                                    SPEC["knots"], SPEC["spline_range"], SPEC["epochs"], SPEC["seed"])
    with torch.no_grad():
        fit = model(torch.tensor(x_train, dtype=torch.float32)).numpy().ravel().astype(np.float64)
    s_torch = fit * t_std + t_mean
    params = apc.kan_params(model)

    # The P5 classifier's scaler comes from the same imputation and StandardScaler on the same rows.
    with np.load(P5_MODEL) as z:
        p5 = {k: z[k] for k in z.files}
    scaler_check = {"scaler_mean_equal_p5_bitwise": bool(np.array_equal(scaler.mean_, p5["scaler_mean"])),
                    "scaler_std_equal_p5_bitwise": bool(np.array_equal(scaler.scale_, p5["scaler_std"])),
                    "imputation_mean_equal_p5_bitwise": bool(np.array_equal(means, p5["imputation_mean"]))}

    # FP32 header-style replica (the device arithmetic class): ArrayKAN output pre-sigmoid.
    arrays = {"scaler_mean": scaler.mean_, "scaler_std": scaler.scale_, **params}
    replica = apc.ArrayKAN(arrays)
    raw32 = np.where(np.isfinite(x5), x5, replica.p["SCALER_MEAN"].astype(np.float32)[None, :]).astype(np.float32)
    scaled = replica.standardize(raw32)
    if not np.all(scaled[~valid] == 0.0):
        raise SystemExit("Imputed rows do not standardize to exactly 0")
    out_replica = replica._spline(replica.output_preactivation(scaled), 2) @ replica.p["KAN2_THETA"][0]
    s_replica = (out_replica * np.float64(np.float32(t_std)) + np.float64(np.float32(t_mean)))
    replica_vs_torch = float(np.max(np.abs(s_replica - s_torch)))

    resub = {"all": regression_metrics(s_torch, t), "beat_valid": regression_metrics(s_torch[valid], t[valid])}
    flag = s_torch < apc.CONCEPT_CUT
    resub["flag_s_hat_lt_90_vs_label"] = {
        "tp": int(np.sum(flag & (y == 1))), "fn": int(np.sum(~flag & (y == 1))),
        "tn": int(np.sum(~flag & (y == 0))), "fp": int(np.sum(flag & (y == 0))),
        "note": "descriptive only; the device risk score is the P5 TinyKAN, not this flag"}
    quality = pd.read_csv(CONCEPT / "s4_concept_quality.csv")
    honest = quality[(quality.concept_predictor == "kan_p5") & (quality.subset == "all")]
    honest = {row.grouping: {k: float(getattr(row, k)) for k in ("mae", "rmse", "bias", "bland_altman_low",
                                                                   "bland_altman_high", "pearson_r")}
              for row in honest.itertuples()}

    out.mkdir(parents=True)
    np.savez(out / "sat_regressor_model.npz", features=np.array(apc.P5), scaler_mean=scaler.mean_, scaler_std=scaler.scale_,
             imputation_mean=means, target_mean=np.float64(t_mean), target_std=np.float64(t_std),
             target_clip=np.array(apc.TARGET_CLIP), **{k: np.asarray(v) for k, v in params.items()})
    apc.las.write_gzip_csv(pd.DataFrame({"row_index": np.arange(apc.N_ROWS), "beat_valid": valid.astype(int),
                                         "target_clipped": t, "s_hat_torch": s_torch, "s_hat_fp32_replica": s_replica}),
                           out / "resubstitution_predictions.csv.gz")
    model_json = {
        "model": "Device concept regressor for estimated ScalcO2 (s_hat), P5 image only; displayed, never an input to the "
                 "risk score or the explanation. Not an oxygen reading; uncalibrated for the MAX30102.",
        "features_in_order": list(apc.P5), "recipe": SPEC, "training_rows": apc.N_ROWS,
        "imputation": "beat-invalid rows -> all-window beat-valid mean (imputation_mean), 0 after scaling",
        "target": "median_spo2 (window-median ScalcO2) clipped to [50, 100]; standardized with target_mean / target_std "
                  "(population std) for training; s_hat = output * target_std + target_mean, not clipped",
        "target_mean": t_mean, "target_std": t_std, "target_clip": list(apc.TARGET_CLIP),
        "scaler_mean": scaler.mean_.tolist(), "scaler_std": scaler.scale_.tolist(), "imputation_mean": means.tolist(),
        "parameter_count": int(params["parameter_count"]), "scaler_vs_p5_classifier": scaler_check,
        "fp32_replica_vs_torch_max_abs_s_hat": replica_vs_torch,
    }
    (out / "sat_regressor_model.json").write_text(json.dumps(model_json, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_head(), "environment": apc.environment(),
        "resubstitution": resub,
        "resubstitution_label": "resubstitution, not an accuracy result: in-sample on the 48,662 training rows",
        "cross_fitted_stage4_kan_p5": honest,
        "cross_fitted_source": "outputs/physio-concept-v1/s4_concept_quality.csv (kan_p5, subset all): the honest accuracy "
                               "of this recipe, out-of-fold, recording- and patient-grouped",
        "units": "ScalcO2 percentage points against the clipped window median",
        "seconds": time.perf_counter() - started,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): apc.sha256(p) for p in
                          (apc.P["v2_features"], CONCEPT / "physio_features.csv.gz", P5_MODEL, CONCEPT / "s4_concept_quality.csv",
                           Path(__file__), SCRIPTS / "analyze_physio_concept.py", SCRIPTS / "train_tinykan_v2.py")},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=apc.las._json_default) + "\n", encoding="utf-8")
    if apc.protected_hashes() != protected_before or apc.las.tree_hashes(CONCEPT) != concept_before:
        raise SystemExit("A protected input changed during the run")
    names = sorted(p.name for p in out.iterdir())
    (out / "SHA256SUMS").write_text("".join(f"{apc.sha256(out / n)} *{n}\n" for n in names), encoding="utf-8", newline="\n")
    print(json.dumps({"scaler_vs_p5_classifier": scaler_check, "target_mean": t_mean, "target_std": t_std,
                      "parameter_count": int(params["parameter_count"]), "fp32_replica_vs_torch_max_abs_s_hat": replica_vs_torch,
                      "resubstitution_all": resub["all"], "cross_fitted_stage4": honest,
                      "seconds": round(time.perf_counter() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

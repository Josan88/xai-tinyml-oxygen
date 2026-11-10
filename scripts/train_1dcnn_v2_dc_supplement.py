"""POST HOC supplement to the raw-waveform 1D-CNN baseline: v2 CNNs that also see the raw DC level.

Why this exists. The pre-registered recipe of scripts/train_1dcnn_baseline.py (recipe.json in
outputs/analysis-1dcnn-v1/) gives the CNN the band-passed IR/red windows plus each window's own
mean and std. Its "not_seen" entry already states the gap: the raw (unfiltered) DC level, which
v2 TinyKAN's ratio_of_ratios and ir_perfusion_index divide by. The v1 CNN sees everything v1
TinyKAN's 14 features are computed from; the v2 CNN does not. After the first run showed the v2
CNNs 0.11-0.12 AUC below v2 TinyKAN, this supplement asks how much of that gap is the missing
DC input. It was added AFTER seeing the first result and is labelled post hoc everywhere.

Change against the pre-registered recipe (nothing else differs): two extra side inputs per window,
log(ir_dc_mean) and log(red_dc_mean), taken from outputs/openox-raw-windows-v1/v2_ir_red_windows.npz
(mean of the 50 Hz resampled raw signal over the window, the quantity the v2 feature code divides
by). They are standardised with inner-training-split statistics like the other side inputs. Same
architectures, folds, inner split, seed 42, early stopping and bootstrap resamples as the v2 rows of
outputs/analysis-1dcnn-v1/.

Writes (write-once) outputs/analysis-1dcnn-v2dc-supplement/: metrics.csv, per_fold.csv,
bootstrap.csv (paired recording bootstrap against TinyKAN and against the no-DC CNN of the same
size), oof_v2dc.csv.gz, summary.json, README.md. Analysis models only; never deployed.

Run (repository root): C:\\Python314\\python.exe scripts/train_1dcnn_v2_dc_supplement.py --workers 10
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_1dcnn_baseline as base  # noqa: E402

lm = base.lm
ROOT = base.ROOT
OUT_DEFAULT = ROOT / "outputs" / "analysis-1dcnn-v2dc-supplement"
FIRST_RUN = ROOT / "outputs" / "analysis-1dcnn-v1"
CONFIGS = [{"name": f"v2dc_{s}_seed42", "version": "v2", "size": s, "seed": 42} for s in ("small", "large")]
FIRST_RUN_COLUMN = {"small": "cnn_small_seed42", "large": "cnn_large_seed42"}


def prepare_dc_cache(v2_folder: Path) -> Path:
    digest = base.sha256(base.P["v2_npz"])
    folder = base.CACHE_DIR / f"v2dc_{digest[:16]}"
    if (folder / "done").exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(base.P["v2_npz"], allow_pickle=False) as handle:
        ir_dc = handle["ir_dc_mean"].astype(np.float64)
        red_dc = handle["red_dc_mean"].astype(np.float64)
        if not np.array_equal(handle["recording_id"], np.load(v2_folder / "recording_id.npy")):
            raise RuntimeError("DC arrays do not align with the cached v2 rows")
    if not ((ir_dc > 0).all() and (red_dc > 0).all()):
        raise RuntimeError("A raw DC mean is not positive; log is undefined")
    side = np.concatenate([np.load(v2_folder / "side.npy"), np.log(ir_dc)[:, None], np.log(red_dc)[:, None]], axis=1)
    for name in ("z.npy", "label.npy", "recording_id.npy", "start_time_seconds.npy"):
        shutil.copyfile(v2_folder / name, folder / name)
    np.save(folder / "side.npy", side)
    (folder / "done").write_text(digest, encoding="utf-8")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; write-once.")
        return 2
    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    before = base.protected_hashes()

    ref = base.load_reference()
    paths = base.prepare_cache()
    checks = base.check_cache_join(paths, ref)
    folder = prepare_dc_cache(paths["v2"])
    folds = ref["folds"]["v2"]
    frame = ref["v2"]
    y = frame["label"].to_numpy(int)

    first = pd.read_csv(FIRST_RUN / "oof_v2.csv.gz")
    if not (np.array_equal(first["row_index"], np.arange(len(frame)))
            and np.array_equal(first["fold"].to_numpy(int), folds) and np.array_equal(first["y_true"], y)):
        raise RuntimeError("First-run v2 OOF does not align with the frozen v2 rows")

    out.mkdir(parents=True)
    units = [{**c, "fold": int(f), "folds": folds, "cache": str(folder)} for c in CONFIGS for f in sorted(np.unique(folds))]
    units.sort(key=lambda u: (u["size"] != "large", u["fold"]))
    print(f"Training {len(units)} (configuration, fold) jobs on {args.workers} workers ...", flush=True)
    results = base.run_units(units, args.workers)

    probabilities, per_fold = {}, []
    for config in CONFIGS:
        p = np.full(len(frame), np.nan)
        for r in (r for r in results if r["name"] == config["name"]):
            p[r["test_rows"]] = r["probabilities"]
            m = lm.binary_metrics(y[r["test_rows"]], r["probabilities"], base.THRESHOLD)
            per_fold.append({"configuration": r["name"], "fold": r["fold"], "auc": m["auc"], "recall": m["recall"],
                             "specificity": m["specificity"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                             "inner_val_auc": r["inner_val_auc"], "seconds": round(r["seconds"], 1)})
        if not np.isfinite(p).all():
            raise RuntimeError(f"{config['name']}: incomplete out-of-fold coverage")
        probabilities[config["name"]] = p
    base.write_csv(pd.DataFrame(per_fold).sort_values(["configuration", "fold"]), out / "per_fold.csv")
    oof = {"row_index": np.arange(len(frame)), "recording_id": frame["recording_id"].to_numpy(), "fold": folds, "y_true": y}
    oof.update({f"cnn_{c['size']}_dc_seed42": probabilities[c["name"]] for c in CONFIGS})
    base.write_gzip_csv(pd.DataFrame(oof), out / "oof_v2dc.csv.gz")

    import torch  # noqa: F401  (parameter counts)
    metric_rows = []
    for config in CONFIGS:
        model = base.build_model(config["size"], 2, 6)
        params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        m = lm.binary_metrics(y, probabilities[config["name"]], base.THRESHOLD)
        metric_rows.append({"version": "v2", "grouping": "recording", "model": base.MODEL_LABEL[config["size"]] + " + raw DC (post hoc)",
                            "configuration": config["name"], "trainable_params": params,
                            **{k: m[k] for k in ("n", "prevalence", "auc", "pr_auc", "balanced_accuracy", "f1", "recall",
                                                 "specificity", "precision", "candidate_rate", "tn", "fp", "fn", "tp")}})
    base.write_csv(pd.DataFrame(metric_rows), out / "metrics.csv")

    seed = base.BOOTSTRAP_SEED + base.BOOTSTRAP_OFFSET["v2"]
    probs = {"TinyKAN": ref["tinykan"]["v2"], **probabilities,
             **{f"v2_{s}_seed42": first[col].to_numpy(float) for s, col in FIRST_RUN_COLUMN.items()}}
    point, draws = lm.group_bootstrap(y, probs, frame["recording_id"].to_numpy(), base.N_BOOT, seed, base.THRESHOLD)
    first_boot = pd.read_csv(FIRST_RUN / "cnn_bootstrap_vs_tinykan.csv")
    boot_rows = []
    for config in CONFIGS:
        for reference in ("TinyKAN", f"v2_{config['size']}_seed42"):
            for metric in ("auc", "recall", "specificity"):
                delta = draws[config["name"]][metric] - draws[reference][metric]
                low, high = lm.percentile_ci(delta)
                boot_rows.append({"comparison": f"{config['name']} - {reference}", "metric": metric,
                                  "point": point[config["name"]][metric], "reference_point": point[reference][metric],
                                  "difference": point[config["name"]][metric] - point[reference][metric],
                                  "ci_low": low, "ci_high": high, "fraction_resamples_le_0": float(np.mean(delta <= 0)),
                                  "method": f"paired recording bootstrap ({base.N_BOOT} resamples, 95% percentile CI)",
                                  "seed": seed, "n_units": int(frame["recording_id"].nunique())})
        # the no-DC rows must reproduce the first run's bootstrap exactly (same seed, same resamples)
        row = first_boot[(first_boot.comparison == f"v2_{config['size']}_seed42 - TinyKAN") & (first_boot.metric == "auc")].iloc[0]
        d = draws[f"v2_{config['size']}_seed42"]["auc"] - draws["TinyKAN"]["auc"]
        lo, hi = lm.percentile_ci(d)
        checks[f"first_run_v2_{config['size']}_bootstrap_reproduced"] = bool(
            np.isclose(lo, row.ci_low, atol=1e-7, rtol=0) and np.isclose(hi, row.ci_high, atol=1e-7, rtol=0))
    base.write_csv(pd.DataFrame(boot_rows), out / "bootstrap.csv")

    if base.protected_hashes() != before:
        raise RuntimeError("A protected directory changed during the run")
    summary = {"created_utc": created, "duration_seconds": round(time.perf_counter() - started, 1),
               "script": "scripts/train_1dcnn_v2_dc_supplement.py", "post_hoc": True,
               "reason": "first-run v2 CNNs lacked the raw DC level that v2 TinyKAN's ratio_of_ratios and "
                         "ir_perfusion_index use (recipe.json not_seen); added after seeing the first result",
               "change_vs_recipe": "side inputs + log(ir_dc_mean), log(red_dc_mean); all else identical",
               "checks": checks,
               "metrics": {r["configuration"]: {k: r[k] for k in ("auc", "recall", "specificity", "trainable_params")} for r in metric_rows},
               "bootstrap": {f"{r['comparison']} {r['metric']}": [r["difference"], r["ci_low"], r["ci_high"]] for r in boot_rows}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    (out / "README.md").write_text(
        "# analysis-1dcnn-v2dc-supplement (POST HOC)\n\n"
        "v2 raw-waveform 1D-CNNs given the raw DC level as two extra side inputs. Added after the first run of\n"
        "`scripts/train_1dcnn_baseline.py` (`outputs/analysis-1dcnn-v1/`) to test how much of the v2 CNN-TinyKAN gap\n"
        "is the missing DC input. Not pre-registered. Analysis models only; never deployed. Write-up: `docs/analysis-1dcnn.md`.\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(summary["metrics"], indent=1))
    print(json.dumps(summary["checks"], indent=1))
    print(f"Done in {summary['duration_seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen TinyKAN feature-masking evaluation (ADR-0003); no training or board I/O.

Discrete AOPC and trapezoidal probability AUC are separate metrics. These are
exploratory responses of one exported model to mean replacement, not causal or
held-out physiological validation. Raw windows are never loaded or modified.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from check_probability_parity import NAMES, run_probe
from verify_features import numpy_kan_predict, parse_c_header

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h"
FEATURES = ROOT / "outputs/openox_canonical_window_features.csv"
METRICS = ("deletion_aopc", "insertion_aopc", "deletion_auc", "insertion_auc")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenKAN:
    """Vectorized float64 host model using constants quantized to header FP32.

    This matches the existing independent NumPy reference, not bitwise MCU
    arithmetic. Raw inputs are rounded to FP32 before host standardization.
    """

    def __init__(self, header=HEADER):
        self.parameters = {
            key: np.asarray(value, dtype=np.float32).astype(np.float64)
            for key, value in parse_c_header(header).items()
        }

    def standardize(self, raw):
        p = self.parameters
        return (np.asarray(raw, dtype=np.float32).astype(np.float64)
                - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)

    def predict(self, scaled):
        current = np.asarray(scaled, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 14 or not np.isfinite(current).all():
            raise ValueError("Expected finite N x 14 standardized features")
        p = self.parameters
        for layer in (1, 2):
            linear = current @ p[f"w{layer}"].T + p[f"b{layer}"]
            distance = np.abs((np.clip(linear, -3., 3.)[:, :, None]
                               - p[f"centers{layer}"]) / (p[f"step{layer}"] + 1e-10))
            inner = 2. / 3. - distance**2 + .5 * distance**3
            outer = np.maximum(2. - distance, 0.)**3 / 6.
            current = np.sum(np.where(distance < 1., inner, outer)
                             * p[f"theta{layer}"][None, :, :], axis=2)
        return 1. / (1. + np.exp(-current[:, 0]))


def mean_perturbation(predict, scaled):
    probability = predict(scaled)
    attribution = np.empty_like(scaled, dtype=np.float64)
    for feature in range(scaled.shape[1]):
        changed = scaled.copy()
        changed[:, feature] = 0.
        attribution[:, feature] = probability - predict(changed)
    return probability, attribution


def ranked_orders(attribution, seed=20260911, random_repeats=5):
    if attribution.ndim != 2 or not np.isfinite(attribution).all():
        raise ValueError("Expected finite attribution matrix")
    if random_repeats < 1 or seed < 0:
        raise ValueError("Use at least one random repeat and a nonnegative seed")
    signed = np.argsort(-attribution, axis=1, kind="stable")
    orders = {
        "signed_risk": signed,
        # Firmware bubble sort compares fabsf and preserves original order on ties.
        "firmware_absolute": np.argsort(-np.abs(attribution), axis=1, kind="stable"),
        "reverse_signed": signed[:, ::-1].copy(),
    }
    for repeat in range(random_repeats):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, repeat])))
        orders[f"random_{repeat:02d}"] = np.argsort(
            rng.random(attribution.shape), axis=1, kind="stable")
    return orders


def perturbation_curves(predict, scaled, order):
    """Return p-risk at k=0..d; order is fixed from the unperturbed input."""
    rows, dimensions = scaled.shape
    if order.shape != scaled.shape or not np.array_equal(
            np.sort(order, axis=1), np.tile(np.arange(dimensions), (rows, 1))):
        raise ValueError("Every ranking must be a permutation of all features")
    deleted = scaled.copy()
    inserted = np.zeros_like(scaled)
    deletion = np.empty((rows, dimensions + 1))
    insertion = np.empty_like(deletion)
    deletion[:, 0] = predict(deleted)
    insertion[:, 0] = predict(inserted)
    row = np.arange(rows)
    for step in range(dimensions):
        feature = order[:, step]
        deleted[row, feature] = 0.
        inserted[row, feature] = scaled[row, feature]
        deletion[:, step + 1] = predict(deleted)
        insertion[:, step + 1] = predict(inserted)
    return deletion, insertion


def curve_metrics(deletion, insertion):
    """AOPC averages k=1..d deltas; AUC trapezoid-integrates p over k/d."""
    if (deletion.ndim != 2 or deletion.shape != insertion.shape
            or deletion.shape[1] < 2
            or not np.isfinite(deletion).all() or not np.isfinite(insertion).all()):
        raise ValueError("Expected matching finite N x (d+1) curves")
    dimensions = deletion.shape[1] - 1
    return {
        "deletion_aopc": np.mean(deletion[:, :1] - deletion[:, 1:], axis=1),
        "insertion_aopc": np.mean(insertion[:, 1:] - insertion[:, :1], axis=1),
        "deletion_auc": np.trapezoid(deletion, dx=1. / dimensions, axis=1),
        "insertion_auc": np.trapezoid(insertion, dx=1. / dimensions, axis=1),
    }


def load_features(path, expected_windows):
    rows, metadata = [], []
    required = set(NAMES) | {"recording_id", "start_time_seconds", "end_time_seconds", "label"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV fields: {sorted(required - set(reader.fieldnames or []))}")
        for index, row in enumerate(reader):
            values = [float(row[name]) for name in NAMES]
            label = int(row["label"])
            if label not in (0, 1) or not np.isfinite(values).all():
                raise ValueError(f"Nonfinite features or nonbinary label at row {index}")
            if not row["recording_id"]:
                raise ValueError(f"Missing recording ID at row {index}")
            rows.append(values)
            metadata.append({"source_row": index, "recording_id": row["recording_id"],
                             "start_time_seconds": row["start_time_seconds"],
                             "end_time_seconds": row["end_time_seconds"], "label": label})
    if not rows or (expected_windows and len(rows) != expected_windows):
        raise ValueError(f"Expected {expected_windows} windows, found {len(rows)}")
    raw = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Features overflow FP32")
    return raw, metadata


def summarize(values, recording_ids):
    unique, inverse = np.unique(recording_ids, return_inverse=True)
    recording_means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return {"window_mean": float(np.mean(values)), "window_median": float(np.median(values)),
            "window_p05": float(np.quantile(values, .05)),
            "window_p95": float(np.quantile(values, .95)),
            "recording_macro_mean": float(np.mean(recording_means)),
            "recordings": len(unique)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--header", type=Path, default=HEADER)
    parser.add_argument("--output", type=Path, required=True,
                        help="New, empty local output directory; existing results are never overwritten")
    parser.add_argument("--expected-windows", type=int, default=52090,
                        help="Require this count (0 permits another cohort)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--probe", type=Path, help="Optional existing compiled firmware probability probe")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.random_repeats < 1 or args.seed < 0 or args.expected_windows < 0:
        parser.error("Batch size/repeats must be positive; seed/window count nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; use a new directory")
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    raw, metadata = load_features(args.features, args.expected_windows)
    model = FrozenKAN(args.header)
    scaled = model.standardize(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    # Snapshot all source code on which the numerical evaluation depends.
    source = args.output / "source"
    source.mkdir()
    source_paths = [Path(__file__), ROOT / "scripts/verify_features.py",
                    ROOT / "scripts/check_probability_parity.py", args.header,
                    ROOT / "firmware/Arduino Nano 33/include/kan_inference.h",
                    ROOT / "scripts/train_final_model.py"]
    provenance = {str(path.resolve()): sha256(path) for path in source_paths + [args.features]}
    for path in source_paths:
        shutil.copy2(path, source / path.name)
    probability, attribution = np.empty(len(raw)), np.empty_like(scaled)
    for start in range(0, len(raw), args.batch_size):
        stop = min(start + args.batch_size, len(raw))
        probability[start:stop], attribution[start:stop] = mean_perturbation(model.predict, scaled[start:stop])
    reference_indices = np.unique(np.linspace(0, len(raw) - 1, min(211, len(raw)), dtype=int))
    independent = np.array([numpy_kan_predict(raw[i].astype(np.float64), model.parameters)[1]
                            for i in reference_indices])
    scalar_error = float(np.max(np.abs(independent - probability[reference_indices])))
    if scalar_error >= 1e-12:
        raise RuntimeError(f"Vectorized/scalar probability mismatch: {scalar_error}")
    verification = {"scalar_reference_windows": len(reference_indices),
                    "scalar_max_probability_error": scalar_error,
                    "host_dtype": "float64 arithmetic; FP32 header constants and raw feature inputs"}
    if args.probe:
        native = run_probe(args.probe.resolve(), "features", raw[reference_indices])
        verification.update(native_probe_sha256=sha256(args.probe),
                            native_probe_path=str(args.probe.resolve()),
                            native_max_probability_error=float(np.max(np.abs(native[:, 0] - probability[reference_indices]))),
                            native_max_attribution_error=float(np.max(np.abs(native[:, 29:] - attribution[reference_indices]))))
        if max(verification["native_max_probability_error"], verification["native_max_attribution_error"]) >= 1e-5:
            raise RuntimeError(f"Native probe comparison failed: {verification}")
        np.savez_compressed(args.output / "native_check.npz", source_rows=reference_indices,
                            native=native, host_probability=probability[reference_indices],
                            host_attribution=attribution[reference_indices])
    orders = ranked_orders(attribution, args.seed, args.random_repeats)
    baseline = float(model.predict(np.zeros((1, len(NAMES))))[0])
    all_metrics, mean_curve_rows = {}, []
    for name, order in orders.items():
        print(f"Evaluating {name}: {len(raw)} windows", flush=True)
        deletion, insertion = np.empty((len(raw), 15)), np.empty((len(raw), 15))
        for start in range(0, len(raw), args.batch_size):
            stop = min(start + args.batch_size, len(raw))
            deletion[start:stop], insertion[start:stop] = perturbation_curves(
                model.predict, scaled[start:stop], order[start:stop])
        for actual, expected in ((deletion[:, 0], probability), (insertion[:, -1], probability),
                                 (deletion[:, -1], baseline), (insertion[:, 0], baseline)):
            if not np.allclose(actual, expected, rtol=0., atol=1e-12):
                raise RuntimeError(f"Endpoint invariant failed for {name}")
        metrics = curve_metrics(deletion, insertion)
        all_metrics[name] = metrics
        np.savez_compressed(args.output / f"curves_{name}.npz", order=order.astype(np.uint8),
                            deletion=deletion, insertion=insertion, **metrics)
        for k in range(15):
            mean_curve_rows.append({"method": name, "k": k, "fraction": k / 14,
                                    "deletion_mean": float(deletion[:, k].mean()),
                                    "insertion_mean": float(insertion[:, k].mean())})
    random_names = [name for name in all_metrics if name.startswith("random_")]
    all_metrics["random_mean"] = {metric: np.mean([all_metrics[name][metric] for name in random_names], axis=0)
                                  for metric in METRICS}
    np.savez_compressed(args.output / "inputs_attributions.npz", raw_features=raw,
                        standardized_features=scaled, probability=probability, attribution=attribution,
                        feature_names=np.array(NAMES))
    labels = np.array([row["label"] for row in metadata])
    recording_ids = np.array([row["recording_id"] for row in metadata])
    groups = {"all": np.ones(len(raw), dtype=bool), "candidate_p_ge_0_50": probability >= .5,
              "noncandidate_p_lt_0_50": probability < .5, "label_1": labels == 1, "label_0": labels == 0}
    summaries = {}
    paired = {}
    for group, mask in groups.items():
        if not mask.any():
            continue
        summaries[group] = {"windows": int(mask.sum()), "methods": {
            name: {metric: summarize(values[mask], recording_ids[mask]) for metric, values in metrics.items()}
            for name, metrics in all_metrics.items()}}
        paired[group] = {}
        for name in ("signed_risk", "firmware_absolute"):
            paired[group][name] = {}
            for metric in METRICS:
                # Positive gain always means better: lower deletion probability AUC,
                # higher insertion probability AUC and both signed AOPCs.
                direction = -1 if metric == "deletion_auc" else 1
                gain = direction * (all_metrics[name][metric] - all_metrics["random_mean"][metric])
                paired[group][name][metric] = {
                    **summarize(gain[mask], recording_ids[mask]),
                    "fraction_better_than_own_random_mean": float(np.mean(gain[mask] > 1e-12)),
                    "fraction_tied_within_1e_12": float(np.mean(np.abs(gain[mask]) <= 1e-12)),
                }
    for filename, rows in (("windows.csv", metadata), ("mean_curves.csv", mean_curve_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (args.output / "window_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_row", "method", *METRICS])
        for name, metrics in all_metrics.items():
            writer.writerows([index, name, *[float(metrics[metric][index]) for metric in METRICS]]
                             for index in range(len(raw)))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    report = {
        "schema_version": 1, "started_utc": timestamp, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started, "windows": len(raw),
        "recordings": len(np.unique(recording_ids)), "feature_names": NAMES,
        "target": "P(low-oxygen-risk), fixed for all windows regardless of label or predicted class",
        "baseline_probability": baseline, "candidate_threshold": .50,
        "masking": "Replace individual standardized coordinates with zero; fixed original-input ranking",
        "methods": {"signed_risk": "descending signed p(x)-p(x with feature i zero)",
                    "firmware_absolute": "descending absolute attribution; original feature index breaks ties",
                    "reverse_signed": "exact reverse of signed_risk, including ties",
                    "random": "independent per-window permutations; PCG64 SeedSequence([seed, repeat]); uniform-key argsort",
                    "random_mean": "per-window arithmetic mean of random replicate metrics"},
        "metric_formulas": {
            "deletion_aopc": "(1/d) sum(k=1..d) [D_0-D_k]; d=14",
            "insertion_aopc": "(1/d) sum(k=1..d) [I_k-I_0]; d=14",
            "deletion_auc": "(1/d) [D_0/2 + sum(k=1..d-1) D_k + D_d/2]",
            "insertion_auc": "(1/d) [I_0/2 + sum(k=1..d-1) I_k + I_d/2]",
        },
        "seed": args.seed, "random_repeats": args.random_repeats, "batch_size": args.batch_size,
        "verification": verification, "summaries": summaries, "paired_gains_vs_random_mean": paired,
        "provenance": {"sha256": provenance, "git_head": git.stdout.strip() if git.returncode == 0 else None,
                       "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                       "command": [sys.executable, *sys.argv]},
        "limitations": [
            "Final exported model is trained on the canonical cohort: in-sample exploratory evaluation, not out-of-fold results.",
            "Correlated derived features are masked independently; perturbed points may be physiologically impossible/out of distribution.",
            "Same mean-replacement operation defines attribution and deletion: first-step signed deletion advantage is built in.",
            "Host floating-point evaluation is not all-window firmware/board parity; native check, if present, is a sample.",
            "Firmware uses absolute ranking and reports only top three; signed-risk ranking is an offline comparator.",
            "Window summaries are descriptive, with overlapping windows and no independent-window confidence claims.",
            "Five random replicates by default are finite controls, not all 14! permutations.",
            "No causal physiology, clinical accuracy, live motion robustness, or held-out explanation validation is established.",
        ],
    }
    report["artifact_sha256"] = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"windows": len(raw), "recordings": report["recordings"],
                      "output": str(args.output.resolve()), "duration_seconds": report["duration_seconds"],
                      "all_window_means": {name: {key: float(np.mean(values)) for key, values in metrics.items()}
                                           for name, metrics in all_metrics.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

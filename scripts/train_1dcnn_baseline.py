"""Raw-waveform 1D-CNN baseline on the frozen window tables (host only, analysis comparator).

Closes the gap noted in docs/analysis-lamaakal-style.md Section A (Lamaakal et al., MAKE
8(3):55, Table 3 deep baselines): a 1D-CNN trained on the filtered PPG windows themselves,
on the same windows, labels and folds as the Section A baselines.

  v1  IR waveform (1 x 500), as-coded labels, folds = split_index recovered from
      outputs/openox_tinykan_predictions.csv (not a fresh GroupKFold).
  v2  IR + red waveforms (2 x 500), header-aligned labels, folds = the fold column of
      outputs/openox-v2/predictions_b_v2_tinykan16.csv (recording-grouped), which Section A
      used for every v2 baseline.

Inputs come from outputs/openox-raw-windows-v1/ (scripts/export_openox_raw_windows.py),
whose rows join 1:1 to the frozen CSV rows. The CNN is an analysis model only: it never
replaces TinyKAN, a weights header, or any frozen prediction, and it was never deployed.

Recipe (fixed before the run, written to recipe.json before any training starts):
  * Input encoding: each channel of each window is z-normalised by its own mean and std
    (per-window, no dataset statistics). The removed per-channel log(std) and mean/std are
    fed to the classifier head as side inputs, standardised with statistics of the inner
    training split only. Waveform plus side inputs is a lossless re-encoding of the
    exported window, so the CNN is not blind to amplitude, which TinyKAN's features carry.
  * Two sizes: "small" (about 2.3k parameters, MCU-scale) and "large" (about 37k).
  * Class-weighted BCE (pos_weight = negatives / positives in the inner training split),
    Adam lr 1e-3, weight decay 1e-4, batch 256, at most 40 epochs, early stopping with
    patience 6 on the inner-validation loss; the best-validation epoch is kept.
  * Inner validation = 15% of the training fold's recordings, drawn by recording with a
    fixed per-fold generator; the held-out fold is never seen during training or stopping.
  * Seed 42 for every configuration; seeds 0-4 also for the small model on v1.
  * CPU only, one thread per job, each (configuration, fold) in a freshly spawned process.

Outputs (write-once) in outputs/analysis-1dcnn-v1/: recipe.json, oof_v1.csv.gz, oof_v2.csv.gz,
cnn_metrics.csv (same columns as a_baselines.csv), cnn_per_fold.csv, cnn_training_log.csv,
cnn_seed_runs.csv, cnn_seed_summary.csv, cnn_bootstrap_vs_tinykan.csv, cnn_size_table.csv,
a_baselines_with_cnn.csv, paper_sentences.md, summary.json, README.md.

Run (from the repository root):
  C:\\Python314\\python.exe scripts/train_1dcnn_baseline.py --workers 8
  C:\\Python314\\python.exe scripts/train_1dcnn_baseline.py --only v1_small_seed42 --output build/1dcnn-rerun
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
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

import lamaakal_metrics as lm  # noqa: E402

ROOT = SCRIPTS.parent
OUT_DEFAULT = ROOT / "outputs" / "analysis-1dcnn-v1"
RAW_DIR = ROOT / "outputs" / "openox-raw-windows-v1"
CACHE_DIR = ROOT / "build" / "1dcnn-cache"
P = {
    "v1_npz": RAW_DIR / "v1_ir_windows.npz",
    "v2_npz": RAW_DIR / "v2_ir_red_windows.npz",
    "raw_export_report": RAW_DIR / "export_report.json",
    "v1_features": ROOT / "outputs/openox_canonical_window_features.csv",
    "v2_features": ROOT / "outputs/openox-v2/window_features.csv",
    "v1_tinykan_oof": ROOT / "outputs/openox_tinykan_predictions.csv",
    "v2_tinykan16_oof": ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    "a_baselines": ROOT / "outputs/analysis-lamaakal-style-v1/a_baselines.csv",
    "c_bootstrap_ci": ROOT / "outputs/analysis-lamaakal-style-v1/c_bootstrap_ci.csv",
    "lamaakal_summary": ROOT / "outputs/analysis-lamaakal-style-v1/summary.json",
}
PROTECTED = (
    ROOT / "outputs/xai-faithfulness/openox-aopc-v1",
    ROOT / "outputs/ptt-ppg/subject-1-v1",
    ROOT / "outputs/ptt_ppg_motion_spike.csv",
    ROOT / "outputs/parity/qualification/locked-board-replay-v1",
    ROOT / "outputs/analysis-lamaakal-style-v1",
    ROOT / "outputs/openox-v2",
    ROOT / "outputs/openox_canonical_window_features.csv",
    ROOT / "outputs/openox_tinykan_predictions.csv",
    ROOT / "outputs/openox-raw-windows-v1",
)

THRESHOLD = 0.50
BOOTSTRAP_SEED = 20261024     # Section C: v1 used +0, v2 (recording) used +1; same resamples here
BOOTSTRAP_OFFSET = {"v1": 0, "v2": 1}
N_BOOT = 2000
INNER_SPLIT_SEED = 20261025   # fixed per fold, independent of the model seed
WINDOW = 500

RECIPE = {
    "input_encoding": {
        "waveform": "per window and per channel: (x - mean(x)) / std(x); no dataset-level statistics",
        "side_inputs": "per channel: log(std(x)) and mean(x)/std(x), concatenated after global pooling; "
                       "standardised with mean/sd of the inner training split only",
        "lossless": "waveform z-score plus (mean, std) per channel reconstructs the exported window exactly",
        "not_seen": "the raw (unfiltered) DC level; v2 ratio_of_ratios and ir_perfusion_index divide by it",
        "v1_channels": ["filtered IR"],
        "v2_channels": ["filtered IR", "filtered red"],
    },
    "architectures": {
        "small": "Conv1d(C,8,7,pad 3)-ReLU-MaxPool(4) | Conv1d(8,16,7,pad 3)-ReLU-MaxPool(4) | "
                 "Conv1d(16,16,5,pad 2)-ReLU-GlobalAvgPool | concat 2C side inputs | Linear(16+2C,1)",
        "large": "Conv1d(C,16,9,pad 4)-BN-ReLU-MaxPool(2) | Conv1d(16,32,7,pad 3)-BN-ReLU-MaxPool(2) | "
                 "Conv1d(32,64,5,pad 2)-BN-ReLU-MaxPool(2) | Conv1d(64,64,5,pad 2)-BN-ReLU-GlobalAvgPool | "
                 "concat 2C side inputs | Linear(64+2C,32)-ReLU-Dropout(0.2) | Linear(32,1)",
    },
    "loss": "BCEWithLogits, pos_weight = negatives / positives in the inner training split",
    "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 1e-4},
    "batch_size": 256,
    "max_epochs": 40,
    "early_stopping": {"monitor": "inner-validation class-weighted BCE", "patience": 6, "min_delta": 1e-5,
                       "restore": "best-validation epoch"},
    "inner_validation": {"fraction_of_training_recordings": 0.15, "unit": "recording",
                         "generator": f"numpy default_rng({INNER_SPLIT_SEED} + fold), same for every model seed",
                         "refit_on_full_training_fold": False},
    "seeds": {"all_configurations": 42, "v1_small_extra": [0, 1, 2, 3, 4]},
    "seed_use": "torch.manual_seed(seed) before model construction; batch order from torch.Generator seeded with seed",
    "threads_per_job": 1,
    "deterministic_algorithms": True,
    "threshold": THRESHOLD,
    "tuning": "none; every value above was fixed before the first training run",
}
LR, WEIGHT_DECAY = RECIPE["optimizer"]["lr"], RECIPE["optimizer"]["weight_decay"]
BATCH, MAX_EPOCHS = RECIPE["batch_size"], RECIPE["max_epochs"]
# Development smoke test only (--smoke): one epoch per fold, never the default output. The
# variable is inherited by the spawned workers.
SMOKE = os.environ.get("CNN1D_SMOKE") == "1"
if SMOKE:
    MAX_EPOCHS = 1
PATIENCE, MIN_DELTA = RECIPE["early_stopping"]["patience"], RECIPE["early_stopping"]["min_delta"]
INNER_FRACTION = RECIPE["inner_validation"]["fraction_of_training_recordings"]

CONFIGS = [{"name": f"{v}_{s}_seed42", "version": v, "size": s, "seed": 42}
           for v in ("v1", "v2") for s in ("small", "large")]
CONFIGS += [{"name": f"v1_small_seed{seed}", "version": "v1", "size": "small", "seed": seed} for seed in range(5)]

MODEL_LABEL = {"small": "1D-CNN small (raw waveform)", "large": "1D-CNN large (raw waveform)"}
A_COLUMNS = ["version", "grouping", "model", "learner", "source", "size_count", "size_unit", "n_windows", "n_groups",
             "prevalence", "auc", "pr_auc", "balanced_accuracy", "f1", "recall", "specificity", "precision",
             "candidate_rate", "tn", "fp", "fn", "tp", "settings"]


# ----------------------------------------------------------------------------- utilities

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


def protected_hashes():
    return {p.relative_to(ROOT).as_posix(): tree_hashes(p) for p in PROTECTED if p.exists()}


def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.8g", lineterminator="\n")


def write_gzip_csv(frame, path):
    text = frame.to_csv(index=False, float_format="%.7g", lineterminator="\n").encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(text)


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def signed(value, digits=3):
    return f"{value:+.{digits}f}"


# ----------------------------------------------------------------------------- models

def build_model(size, channels, side=None):
    import torch
    import torch.nn as nn

    # side-input width; 2 per channel unless a caller adds more (the post hoc v2 DC supplement)
    side = 2 * channels if side is None else int(side)

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(channels, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(8, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.head = nn.Linear(16 + side, 1)

        def forward(self, x, s):
            return self.head(torch.cat([self.features(x), s], dim=1))

    class Large(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, cout, k, pool):
                layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU()]
                return layers + ([nn.MaxPool1d(2)] if pool else [])

            self.features = nn.Sequential(
                *block(channels, 16, 9, True), *block(16, 32, 7, True), *block(32, 64, 5, True),
                *block(64, 64, 5, False), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(64 + side, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, s):
            return self.head(torch.cat([self.features(x), s], dim=1))

    return Small() if size == "small" else Large()


def model_size(size, channels):
    """Trainable parameters, inference parameters (BN folded into the preceding conv),
    stored non-trainable buffers, and multiply-accumulates per window (conv and linear only;
    pooling, ReLU, BN and the z-normalisation are not counted)."""
    import torch
    import torch.nn as nn

    model = build_model(size, channels).eval()
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    bn_affine = int(sum(m.weight.numel() + m.bias.numel() for m in model.modules() if isinstance(m, nn.BatchNorm1d)))
    buffers = int(sum(b.numel() for name, b in model.named_buffers() if not name.endswith("num_batches_tracked")))
    macs = []

    def conv_hook(module, inputs, output):
        macs.append(output.shape[1] * output.shape[2] * module.in_channels * module.kernel_size[0])

    def linear_hook(module, inputs, output):
        macs.append(module.in_features * module.out_features)

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv1d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(torch.zeros(1, channels, WINDOW), torch.zeros(1, 2 * channels))
    for hook in hooks:
        hook.remove()
    inference = trainable - bn_affine
    return {"trainable_params": trainable, "inference_params_bn_folded": inference,
            "bn_running_stat_buffers": buffers, "fp32_weight_bytes": 4 * inference, "macs_per_window": int(sum(macs))}


# ----------------------------------------------------------------------------- data

def prepare_cache():
    """Per-window encoding is fold-independent, so it is computed once and cached as .npy.

    Returns paths of the cached arrays per version. The cache is keyed by the .npz hashes.
    """
    report = json.loads(P["raw_export_report"].read_text(encoding="utf-8"))
    paths = {}
    for version, key, channels in (("v1", "v1_npz", ("ir",)), ("v2", "v2_npz", ("ir", "red"))):
        digest = sha256(P[key])
        if digest != report["outputs_sha256"][P[key].name]:
            raise RuntimeError(f"{P[key].name} does not match its export_report.json hash")
        folder = CACHE_DIR / f"{version}_{digest[:16]}"
        paths[version] = folder
        if (folder / "done").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(P[key], allow_pickle=False) as handle:
            waves = np.stack([handle[c] for c in channels], axis=1).astype(np.float64)   # n x C x 500
            np.save(folder / "recording_id.npy", handle["recording_id"])
            np.save(folder / "label.npy", handle["label"].astype(np.int64))
            np.save(folder / "start_time_seconds.npy", handle["start_time_seconds"])
        mean = waves.mean(axis=2)
        std = waves.std(axis=2)
        if not (std > 0).all():
            raise RuntimeError(f"{version}: a window channel has zero variance")
        z = ((waves - mean[:, :, None]) / std[:, :, None]).astype(np.float32)
        side = np.concatenate([np.log(std), mean / std], axis=1)    # n x 2C: [log std_c..., mean_c/std_c...]
        np.save(folder / "z.npy", z)
        np.save(folder / "side.npy", side)
        (folder / "done").write_text(digest, encoding="utf-8")
    return paths


def inner_split(train_rows, recordings, fold):
    """Hold out 15% of the training fold's recordings for early stopping (by recording)."""
    unique = np.unique(recordings[train_rows])
    rng = np.random.default_rng(INNER_SPLIT_SEED + int(fold))
    n_val = max(1, int(round(INNER_FRACTION * len(unique))))
    val_recordings = set(rng.choice(unique, size=n_val, replace=False).tolist())
    is_val = np.array([r in val_recordings for r in recordings[train_rows]])
    return train_rows[~is_val], train_rows[is_val], len(unique) - n_val, n_val


def run_fold(unit):
    """One (configuration, fold) job. Runs in a freshly spawned process."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    folder = Path(unit["cache"])
    z = np.load(folder / "z.npy", mmap_mode="r")
    side_raw = np.load(folder / "side.npy")
    y = np.load(folder / "label.npy")
    recordings = np.load(folder / "recording_id.npy")
    folds = np.asarray(unit["folds"])
    fold = unit["fold"]
    test_rows = np.flatnonzero(folds == fold)
    train_all = np.flatnonzero(folds != fold)
    if len(set(recordings[test_rows]) & set(recordings[train_all])):
        raise RuntimeError("A recording is in both the training and the held-out fold")
    tr, va, n_tr_rec, n_va_rec = inner_split(train_all, recordings, fold)

    side_mean = side_raw[tr].mean(axis=0)
    side_sd = side_raw[tr].std(axis=0)
    side = ((side_raw - side_mean) / np.where(side_sd > 0, side_sd, 1.0)).astype(np.float32)

    def tensors(rows):
        return (torch.from_numpy(np.ascontiguousarray(z[rows])), torch.from_numpy(side[rows]),
                torch.from_numpy(y[rows].astype(np.float32)).unsqueeze(1))

    x_tr, s_tr, y_tr = tensors(tr)
    x_va, s_va, y_va = tensors(va)
    x_te, s_te, _ = tensors(test_rows)

    pos = float(y[tr].sum())
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    torch.manual_seed(unit["seed"])
    model = build_model(unit["size"], z.shape[1], side_raw.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(unit["seed"])

    def predict_logits(x, s):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                out.append(model(x[i:i + 2048], s[i:i + 2048]))
        return torch.cat(out)

    best_loss, best_epoch, best_state, bad = float("inf"), -1, None, 0
    log = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_tr), generator=generator)
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            optimizer.zero_grad()
            loss = criterion(model(x_tr[idx], s_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        val_loss = float(criterion(predict_logits(x_va, s_va), y_va))
        log.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss})
        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    p_test = torch.sigmoid(predict_logits(x_te, s_te)).numpy().ravel().astype(np.float64)
    p_val = torch.sigmoid(predict_logits(x_va, s_va)).numpy().ravel().astype(np.float64)
    return {
        "name": unit["name"], "fold": fold, "test_rows": test_rows, "probabilities": p_test,
        "best_epoch": best_epoch, "epochs_run": len(log), "best_val_loss": best_loss,
        "inner_val_auc": lm.weighted_auc(y[va], p_val),
        "n_inner_train_windows": int(len(tr)), "n_inner_val_windows": int(len(va)),
        "n_inner_train_recordings": int(n_tr_rec), "n_inner_val_recordings": int(n_va_rec),
        "pos_weight": float(pos_weight.item()), "log": log, "seconds": time.perf_counter() - started,
    }


# ----------------------------------------------------------------------------- reference data

def load_reference():
    v1 = pd.read_csv(P["v1_features"], usecols=["recording_id", "start_time_seconds", "label"])
    v1["recording_id"] = v1["recording_id"].astype(str)
    kan1 = pd.read_csv(P["v1_tinykan_oof"]).sort_values("row_index").reset_index(drop=True)
    v2 = pd.read_csv(P["v2_features"], usecols=["recording_id", "patient_id", "start_time_seconds", "label"])
    v2["recording_id"] = v2["recording_id"].astype(str)
    kan2 = pd.read_csv(P["v2_tinykan16_oof"]).sort_values("row_index").reset_index(drop=True)
    for name, frame, kan in (("v1", v1, kan1), ("v2", v2, kan2)):
        if not np.array_equal(kan["row_index"], np.arange(len(frame))):
            raise RuntimeError(f"{name} TinyKAN predictions do not cover every row once")
        if not (kan["recording_id"].astype(str).to_numpy() == frame["recording_id"].to_numpy()).all():
            raise RuntimeError(f"{name} TinyKAN recording IDs differ")
        if not (kan["y_true"].to_numpy() == frame["label"].to_numpy()).all():
            raise RuntimeError(f"{name} TinyKAN labels differ")
    folds = {"v1": kan1["split_index"].to_numpy(int), "v2": kan2["fold"].to_numpy(int)}
    for version, frame in (("v1", v1), ("v2", v2)):
        if pd.Series(folds[version]).groupby(frame["recording_id"].to_numpy()).nunique().max() != 1:
            raise RuntimeError(f"A {version} recording spans several folds")
    return {"v1": v1, "v2": v2, "folds": folds,
            "tinykan": {"v1": kan1["y_prob"].to_numpy(float), "v2": kan2["y_prob"].to_numpy(float)}}


def check_cache_join(paths, ref):
    checks = {}
    for version in ("v1", "v2"):
        folder = paths[version]
        frame = ref[version]
        same = (np.array_equal(np.load(folder / "recording_id.npy"), frame["recording_id"].to_numpy().astype("U64"))
                and np.array_equal(np.load(folder / "start_time_seconds.npy"), frame["start_time_seconds"].to_numpy(np.float64))
                and np.array_equal(np.load(folder / "label.npy"), frame["label"].to_numpy(np.int64)))
        if not same:
            raise RuntimeError(f"{version}: raw-window keys do not join 1:1 to the frozen CSV")
        checks[f"{version}_raw_windows_join_frozen_csv_in_order"] = True
    return checks


# ----------------------------------------------------------------------------- main

def run_units(units, workers):
    import multiprocessing

    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context, max_tasks_per_child=1) as pool:
        for result in pool.map(run_fold, units):
            results.append(result)
            print(f"  {result['name']} fold {result['fold']}: best epoch {result['best_epoch']}/{result['epochs_run']}, "
                  f"{result['seconds']:.0f} s", flush=True)
    return results


def settings_text(config, size_info):
    return (f"{RECIPE['architectures'][config['size']]}; Adam lr 1e-3 wd 1e-4, batch 256, <=40 epochs, "
            f"early stop patience 6 on inner-val (15% of training recordings), seed {config['seed']}; "
            f"per-window z-score + log(std), mean/std side inputs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="run only these configuration names (rerun checks)")
    parser.add_argument("--smoke", action="store_true", help="development only: 1 epoch per fold; refuses the default output")
    parser.add_argument("--reuse-results", type=Path, help="development only: skip training and reuse a results pickle "
                                                            "written by an earlier run (recorded in summary.json)")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        print(f"STOP: {out} already exists; this script never overwrites an output directory.")
        return 2
    if args.smoke:
        if out == OUT_DEFAULT:
            print("STOP: --smoke may not write the default output.")
            return 2
        os.environ["CNN1D_SMOKE"] = "1"
        global MAX_EPOCHS
        MAX_EPOCHS = 1
    configs = [c for c in CONFIGS if not args.only or c["name"] in args.only]
    if not configs:
        print("STOP: no configuration selected")
        return 2
    full_run = len(configs) == len(CONFIGS)

    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    before = protected_hashes()
    input_hashes = {key: sha256(path) for key, path in P.items()}
    ref = load_reference()
    paths = prepare_cache()
    checks = check_cache_join(paths, ref)

    sizes = {(size, version): model_size(size, 1 if version == "v1" else 2)
             for size in ("small", "large") for version in ("v1", "v2")}

    out.mkdir(parents=True)
    (out / "recipe.json").write_text(json.dumps({
        "written_before_training_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": RECIPE, "configurations": configs,
        "model_sizes": {f"{v}_{s}": info for (s, v), info in sizes.items()},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    units = []
    for config in configs:
        folds = ref["folds"][config["version"]]
        for fold in sorted(np.unique(folds)):
            units.append({**config, "fold": int(fold), "folds": folds, "cache": str(paths[config["version"]])})
    units.sort(key=lambda u: (u["size"] != "large", u["version"] != "v2", u["name"], u["fold"]))
    import pickle
    if args.reuse_results:
        results = pickle.loads(args.reuse_results.read_bytes())
        results = [r for r in results if r["name"] in {c["name"] for c in configs}]
        print(f"Reusing {len(results)} fold results from {args.reuse_results} (development)", flush=True)
    else:
        print(f"Training {len(units)} (configuration, fold) jobs on {args.workers} workers ...", flush=True)
        results = run_units(units, args.workers)
        # Saved before assembly so that a failure below does not cost the training run.
        cache = CACHE_DIR / f"results_{out.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.pkl"
        cache.write_bytes(pickle.dumps(results))
        print(f"Fold results saved to {cache}", flush=True)

    # Assemble out-of-fold probabilities per configuration.
    probabilities, per_fold_rows, log_rows = {}, [], []
    for config in configs:
        n = len(ref[config["version"]])
        p = np.full(n, np.nan)
        for r in (r for r in results if r["name"] == config["name"]):
            p[r["test_rows"]] = r["probabilities"]
        if not np.isfinite(p).all():
            raise RuntimeError(f"{config['name']}: incomplete out-of-fold coverage")
        probabilities[config["name"]] = p
    for r in sorted(results, key=lambda r: (r["name"], r["fold"])):
        config = next(c for c in configs if c["name"] == r["name"])
        version = config["version"]
        rows = r["test_rows"]
        y = ref[version]["label"].to_numpy(int)[rows]
        m = lm.binary_metrics(y, r["probabilities"], THRESHOLD)
        per_fold_rows.append({
            "configuration": r["name"], "version": version, "size": config["size"], "seed": config["seed"], "fold": r["fold"],
            "n_windows": m["n"], "n_groups": int(ref[version]["recording_id"].iloc[rows].nunique()),
            "prevalence": m["prevalence"], "auc": m["auc"], "pr_auc": m["pr_auc"], "recall": m["recall"],
            "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "best_inner_val_loss": r["best_val_loss"],
            "inner_val_auc": r["inner_val_auc"], "n_inner_train_windows": r["n_inner_train_windows"],
            "n_inner_val_windows": r["n_inner_val_windows"], "n_inner_train_recordings": r["n_inner_train_recordings"],
            "n_inner_val_recordings": r["n_inner_val_recordings"], "pos_weight": r["pos_weight"],
            "seconds": round(r["seconds"], 1)})
        for entry in r["log"]:
            log_rows.append({"configuration": r["name"], "fold": r["fold"], **entry})
    write_csv(pd.DataFrame(per_fold_rows), out / "cnn_per_fold.csv")
    write_csv(pd.DataFrame(log_rows), out / "cnn_training_log.csv")

    for version in ("v1", "v2"):
        names = [c["name"] for c in configs if c["version"] == version]
        if not names:
            continue
        frame = ref[version]
        columns = {"row_index": np.arange(len(frame)), "recording_id": frame["recording_id"].to_numpy(),
                   "start_time_seconds": frame["start_time_seconds"].to_numpy(), "fold": ref["folds"][version],
                   "y_true": frame["label"].to_numpy(int)}
        for name in names:
            columns[f"cnn_{name.split('_', 1)[1]}"] = probabilities[name]
        write_gzip_csv(pd.DataFrame(columns), out / f"oof_{version}.csv.gz")

    # Metrics rows with the a_baselines.csv columns.
    metric_rows = []
    for config in configs:
        version = config["version"]
        y = ref[version]["label"].to_numpy(int)
        m = lm.binary_metrics(y, probabilities[config["name"]], THRESHOLD)
        info = sizes[(config["size"], version)]
        metric_rows.append({
            "version": version, "grouping": "recording", "model": MODEL_LABEL[config["size"]],
            "learner": f"cnn1d_{config['size']}", "configuration": config["name"], "seed": config["seed"],
            "source": "trained here on the same folds from the raw-window export (analysis model, never deployed)",
            "size_count": info["trainable_params"], "size_unit": "trainable parameters",
            "n_windows": m["n"], "n_groups": int(ref[version]["recording_id"].nunique()), "prevalence": m["prevalence"],
            "auc": m["auc"], "pr_auc": m["pr_auc"], "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"],
            "recall": m["recall"], "specificity": m["specificity"], "precision": m["precision"],
            "candidate_rate": m["candidate_rate"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "settings": settings_text(config, info)})
    all_metrics = pd.DataFrame(metric_rows)
    main_rows = all_metrics[all_metrics["seed"] == 42]
    write_csv(main_rows.loc[:, A_COLUMNS], out / "cnn_metrics.csv")

    heads = {"checks": checks}
    seed_rows = all_metrics[(all_metrics["version"] == "v1") & (all_metrics["learner"] == "cnn1d_small")]
    if len(seed_rows) > 1:
        write_csv(seed_rows.loc[:, ["configuration", "version", "model", "seed", "auc", "pr_auc", "recall", "specificity",
                                    "balanced_accuracy", "candidate_rate"]], out / "cnn_seed_runs.csv")
        summary_rows = []
        for label, block in (("seeds 0-4", seed_rows[seed_rows["seed"] != 42]), ("seeds 42 and 0-4", seed_rows)):
            row = {"version": "v1", "model": MODEL_LABEL["small"], "seed_set": label, "n_seeds": len(block),
                   "seeds": " ".join(str(s) for s in block["seed"])}
            for metric in ("auc", "pr_auc", "recall", "specificity", "balanced_accuracy"):
                row[f"{metric}_mean"], row[f"{metric}_sd"] = lm.mean_sd(block[metric])
                row[f"{metric}_min"], row[f"{metric}_max"] = float(block[metric].min()), float(block[metric].max())
            summary_rows.append(row)
        write_csv(pd.DataFrame(summary_rows), out / "cnn_seed_summary.csv")
        heads["v1_small_seed_spread"] = summary_rows

    # Paired recording bootstrap, CNN - TinyKAN, with Section C's seeds (identical resamples).
    boot_rows, boot_checks = [], {}
    c_ci = pd.read_csv(P["c_bootstrap_ci"])
    for version in ("v1", "v2"):
        names = [c["name"] for c in configs if c["version"] == version and c["seed"] == 42]
        if not names:
            continue
        frame = ref[version]
        y = frame["label"].to_numpy(int)
        seed = BOOTSTRAP_SEED + BOOTSTRAP_OFFSET[version]
        probs = {"TinyKAN": ref["tinykan"][version], **{n: probabilities[n] for n in names}}
        point, draws = lm.group_bootstrap(y, probs, frame["recording_id"].to_numpy(), N_BOOT, seed, THRESHOLD)
        section_c = c_ci[(c_ci.version == version) & (c_ci.grouping == "recording") & (c_ci.model == "TinyKAN") & (c_ci.metric == "auc")].iloc[0]
        tk_low, tk_high = lm.percentile_ci(draws["TinyKAN"]["auc"])
        boot_checks[f"{version}_tinykan_auc_ci_equals_section_c"] = bool(
            np.isclose(tk_low, section_c.ci_low, rtol=0, atol=1e-7) and np.isclose(tk_high, section_c.ci_high, rtol=0, atol=1e-7))
        for name in names:
            for metric in ("auc", "recall", "specificity"):
                delta = draws[name][metric] - draws["TinyKAN"][metric]
                low, high = lm.percentile_ci(delta)
                c_low, c_high = lm.percentile_ci(draws[name][metric])
                t_low, t_high = lm.percentile_ci(draws["TinyKAN"][metric])
                boot_rows.append({
                    "version": version, "grouping": "recording", "comparison": f"{name} - TinyKAN", "metric": metric,
                    "cnn_point": point[name][metric], "cnn_ci_low": c_low, "cnn_ci_high": c_high,
                    "tinykan_point": point["TinyKAN"][metric], "tinykan_ci_low": t_low, "tinykan_ci_high": t_high,
                    "difference": point[name][metric] - point["TinyKAN"][metric], "ci_low": low, "ci_high": high,
                    "fraction_resamples_le_0": float(np.mean(delta <= 0)),
                    "method": f"paired recording bootstrap ({N_BOOT} resamples, 95% percentile CI)",
                    "n_boot": N_BOOT, "seed": seed, "n_units": int(frame["recording_id"].nunique())})
    if boot_rows:
        write_csv(pd.DataFrame(boot_rows), out / "cnn_bootstrap_vs_tinykan.csv")
    checks.update(boot_checks)

    # Size table next to TinyKAN.
    size_rows = []
    for version, params, hidden_in in (("v1", 297, 14), ("v2", 321, 16)):
        tk_macs = hidden_in * 12 + 12 * 8 + 12 + 8
        passes = hidden_in + 1
        size_rows.append({
            "version": version, "model": "TinyKAN (shipped)", "trainable_params": params, "inference_params_bn_folded": params,
            "fp32_weight_bytes": 4 * params, "macs_per_window": tk_macs,
            "macs_note": f"{hidden_in}x12 input mix + 12x8 hidden spline weights + 12 output mix + 8 output spline weights "
                         f"per forward pass; B-spline basis evaluation and the 14-feature extraction (Welch FFT) not counted",
            "forward_passes_per_decision": passes, "macs_per_decision": tk_macs * passes,
            "input": "14 IR features" if version == "v1" else "14 IR features + ratio_of_ratios + ir_perfusion_index",
            "board_latency": "249.7 ms whole decision block (median)" if version == "v1" else "444.3 ms whole decision block (median, r2)",
            "explainer": "mean-perturbation over named features (on device)"})
        for size in ("small", "large"):
            info = sizes[(size, version)]
            size_rows.append({
                "version": version, "model": MODEL_LABEL[size], **info,
                "macs_note": "Conv1d and Linear multiply-accumulates; ReLU, pooling, BN (folded) and z-normalisation not counted",
                "forward_passes_per_decision": 1, "macs_per_decision": info["macs_per_window"],
                "input": "filtered IR window, 1 x 500" if version == "v1" else "filtered IR + red windows, 2 x 500",
                "board_latency": "not measured (never deployed)",
                "explainer": "none on device; mean-perturbation over 500 samples is not an operator-readable explanation"})
    size_table = pd.DataFrame(size_rows)
    write_csv(size_table, out / "cnn_size_table.csv")

    # Combined Section A table: existing rows copied as text, CNN seed-42 rows appended, AUC rank added.
    combined = None
    if full_run:
        existing = pd.read_csv(P["a_baselines"], dtype=str, keep_default_na=False)
        cnn_text = pd.read_csv(out / "cnn_metrics.csv", dtype=str, keep_default_na=False)
        combined = pd.concat([existing, cnn_text], ignore_index=True)
        auc = combined["auc"].astype(float)
        ranks = np.zeros(len(combined), dtype=int)
        for _, block in combined.groupby(["version", "grouping"], sort=False):
            values = auc.loc[block.index]
            for i in block.index:
                ranks[i] = int((values > auc.loc[i]).sum() + 1)
        combined["auc_rank_in_version_grouping"] = ranks
        combined.to_csv(out / "a_baselines_with_cnn.csv", index=False, lineterminator="\n")
        existing_check = pd.read_csv(out / "a_baselines_with_cnn.csv", dtype=str, keep_default_na=False).iloc[:len(existing)]
        checks["a_baselines_rows_copied_verbatim"] = bool(existing_check.loc[:, existing.columns].equals(existing))
        rank_head = {}
        for version in ("v1", "v2"):
            block = combined[(combined.version == version) & (combined.grouping == "recording")]
            tk = block[block.learner == "tinykan"].iloc[0]
            best = block.loc[block["auc"].astype(float).idxmax()]
            rank_head[version] = {"tinykan_rank_by_auc": int(tk["auc_rank_in_version_grouping"]), "models": int(len(block)),
                                  "best_auc_model": best["model"], "best_auc": float(best["auc"]),
                                  "auc_range": [float(block["auc"].astype(float).min()), float(block["auc"].astype(float).max())]}
        heads["section_a_with_cnn"] = rank_head

    after = protected_hashes()
    if after != before:
        raise RuntimeError("A protected directory changed during the run")

    heads["cnn_metrics"] = {r["configuration"]: {k: r[k] for k in ("auc", "pr_auc", "recall", "specificity", "balanced_accuracy", "size_count")}
                            for r in metric_rows}
    heads["bootstrap"] = {f"{r['comparison']} {r['metric']}": [r["difference"], r["ci_low"], r["ci_high"]] for r in boot_rows}
    if full_run:
        (out / "paper_sentences.md").write_text(paper_sentences(out, heads), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(
        "# analysis-1dcnn-v1\n\n"
        "Raw-waveform 1D-CNN comparator on the frozen v1 (reported) and v2 (comparison) windows, same folds and labels as\n"
        "`outputs/analysis-lamaakal-style-v1/` Section A. Generated by `scripts/train_1dcnn_baseline.py` from\n"
        "`outputs/openox-raw-windows-v1/`. Analysis models only: never deployed, never a header, never a frozen prediction.\n"
        "Write-up and limits: `docs/analysis-1dcnn.md`. `recipe.json` was written before training started.\n",
        encoding="utf-8", newline="\n")

    import scipy
    import sklearn
    import torch
    summary = {
        "created_utc": created,
        "duration_seconds": round(time.perf_counter() - started, 1),
        "script": "scripts/train_1dcnn_baseline.py",
        "scope": "host only; analysis comparator; v1 reported, v2 comparison, v3 excluded; no board I/O; "
                 "no firmware, header or existing outputs/ writes; explainer of the shipped system is mean-perturbation",
        "configurations": [c["name"] for c in configs],
        "full_run": full_run,
        "smoke": bool(args.smoke),
        "results_reused_from": str(args.reuse_results) if args.reuse_results else None,
        "threshold": THRESHOLD,
        "recipe": RECIPE,
        "folds": {"v1": "split_index of outputs/openox_tinykan_predictions.csv (recording-grouped, the TinyKAN folds)",
                  "v2": "fold column of outputs/openox-v2/predictions_b_v2_tinykan16.csv (recording-grouped; Section A v2 folds)"},
        "bootstrap": {"seed_v1": BOOTSTRAP_SEED, "seed_v2": BOOTSTRAP_SEED + 1, "n_boot": N_BOOT,
                      "note": "same seeds and recording sets as Section C, so the resamples are identical"},
        "model_sizes": {f"{v}_{s}": info for (s, v), info in sizes.items()},
        "environment": {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "torch": torch.__version__, "device": "cpu", "platform": platform.platform(),
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "workers": args.workers},
        "inputs_sha256": {str(P[k].relative_to(ROOT).as_posix()): v for k, v in input_hashes.items()},
        "write_once_unchanged": True,
        "write_once_sha256": after,
        "checks": checks,
        "job_seconds": {f"{r['name']}_fold{r['fold']}": round(r["seconds"], 1) for r in sorted(results, key=lambda r: (r["name"], r["fold"]))},
        "headlines": heads,
        "outputs_sha256": {f.name: sha256(f) for f in sorted(out.iterdir()) if f.is_file()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summary["headlines"][k] for k in summary["headlines"] if k != "bootstrap"}, indent=2, default=_json_default))
    print(f"Done in {summary['duration_seconds']} s -> {out}")
    return 0


def paper_sentences(out, heads):
    metrics = pd.read_csv(out / "cnn_metrics.csv")
    boot = pd.read_csv(out / "cnn_bootstrap_vs_tinykan.csv")
    size = pd.read_csv(out / "cnn_size_table.csv")
    combined = pd.read_csv(out / "a_baselines_with_cnn.csv")
    seeds = pd.read_csv(out / "cnn_seed_summary.csv") if (out / "cnn_seed_summary.csv").exists() else None
    lines = ["# Paper-ready sentences (generated from the CSVs in this directory)", ""]

    def row(version, size_name):
        return metrics[(metrics.version == version) & (metrics.learner == f"cnn1d_{size_name}")].iloc[0]

    def diff(version, size_name, metric):
        return boot[(boot.version == version) & (boot.comparison == f"{version}_{size_name}_seed42 - TinyKAN") & (boot.metric == metric)].iloc[0]

    def params(version, model):
        return int(size[(size.version == version) & (size.model == model)].iloc[0]["trainable_params"])

    def macs(version, model):
        return int(size[(size.version == version) & (size.model == model)].iloc[0]["macs_per_window"])

    for version, label in (("v1", "the reported v1 windows (IR waveform, as-coded labels, 52,090 windows, 162 recordings)"),
                           ("v2", "the v2 windows (IR and red waveforms, header-aligned labels, 48,662 windows, 154 recordings)")):
        s, l = row(version, "small"), row(version, "large")
        ds, dl = diff(version, "small", "auc"), diff(version, "large", "auc")
        block = combined[(combined.version == version) & (combined.grouping == "recording")]
        tk_rank = int(block[block.learner == "tinykan"].iloc[0]["auc_rank_in_version_grouping"])
        lines.append(
            f"- On {label}, with the same recording-grouped folds, a raw-waveform 1D-CNN reached out-of-fold AUC "
            f"{fmt(s.auc)} with {params(version, MODEL_LABEL['small']):,} parameters (recall {fmt(s.recall)}, specificity {fmt(s.specificity)} at 0.50) "
            f"and {fmt(l.auc)} with {params(version, MODEL_LABEL['large']):,} parameters (recall {fmt(l.recall)}, specificity {fmt(l.specificity)}). "
            f"The paired recording-bootstrap AUC difference against TinyKAN is {signed(ds.difference)} (95% CI {signed(ds.ci_low)} to {signed(ds.ci_high)}) "
            f"for the small CNN and {signed(dl.difference)} ({signed(dl.ci_low)} to {signed(dl.ci_high)}) for the large CNN; "
            f"with both CNNs added, TinyKAN ranks {tk_rank} of {len(block)} by AUC.")
    tk1 = size[(size.version == "v1") & (size.model == "TinyKAN (shipped)")].iloc[0]
    lines.append(
        f"- Size gap (v1): the small CNN has {params('v1', MODEL_LABEL['small']):,} parameters and "
        f"{macs('v1', MODEL_LABEL['small']):,} multiply-accumulates per window, the large CNN {params('v1', MODEL_LABEL['large']):,} and "
        f"{macs('v1', MODEL_LABEL['large']):,}, against TinyKAN's {int(tk1.trainable_params)} parameters and {int(tk1.macs_per_window)} per forward pass "
        f"(feature extraction not counted). Neither CNN was deployed, so its board latency is not measured.")
    if seeds is not None:
        r = seeds[seeds.seed_set == "seeds 0-4"].iloc[0]
        lines.append(
            f"- Across seeds 0-4 the small v1 CNN gave AUC {fmt(r.auc_mean)} ± {fmt(r.auc_sd)}, recall {fmt(r.recall_mean)} ± {fmt(r.recall_sd)} "
            f"and specificity {fmt(r.specificity_mean)} ± {fmt(r.specificity_sd)} at 0.50 (mean ± sd). These are analysis models, not the shipped model.")
    lines.append("")
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not serialisable: {type(value)}")


if __name__ == "__main__":
    raise SystemExit(main())

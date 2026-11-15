"""Board input range check for R11 (host only). Pre-registration: docs/board-input-range-preregistration.md (eb986d6).

Compares the 16 inputs the firmware computed on the second module's rest sit (22 Oct 2026, revised live image)
with the 48,662 training rows. Reads only; writes a new write-once folder.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = ROOT / "outputs/live-v2/2026-10-22_f1a-r2_sit.txt"
SIDECAR_SHA = "37221255369768247e4bd325738a0b3d8de16eee23f00c6e2d06fd3ba21d0860"
TRAIN = ROOT / "outputs/openox-v2/window_features.csv"
HEADER = ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h"
OUT = ROOT / "outputs/board-input-range-v1"
FEATURES = ["mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis", "energy",
            "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz", "bandpower_2_5_hz",
            "bandpower_ratio_0_5_2_to_2_5", "ratio_of_ratios", "ir_perfusion_index"]
GROUP = {**{f: "linear" for f in ("mean", "std", "minimum", "maximum", "value_range", "rms", "derivative_std")},
         **{f: "quadratic" for f in ("energy", "bandpower_0_5_2_hz", "bandpower_2_5_hz")},
         **{f: "scale_free" for f in ("skewness", "kurtosis", "dominant_frequency_hz", "bandpower_ratio_0_5_2_to_2_5",
                                      "ratio_of_ratios", "ir_perfusion_index")}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header_array(text, name):
    body = re.search(name + r"\[16\]\s*=\s*\{([^}]*)\}", text).group(1)
    values = [float(v.strip().rstrip("f")) for v in body.replace("\n", " ").split(",") if v.strip()]
    assert len(values) == 16, (name, len(values))
    return np.array(values)


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; outputs are write-once")
    if sha256(CAPTURE) != SIDECAR_SHA:
        raise SystemExit("M0.3 failed: capture SHA-256 differs from the sidecar")
    rows = []
    for line in CAPTURE.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Feat:" not in line:
            continue
        pairs = dict(re.findall(r"(\w+)=(-?[0-9.]+(?:e[+-]?[0-9]+)?)", line.split("Feat:", 1)[1]))
        rows.append([float(pairs[f]) for f in FEATURES])
    board = np.array(rows)
    if board.shape != (35, 16) or not np.isfinite(board).all():
        raise SystemExit(f"M0.1 failed: board array {board.shape}")
    train = pd.read_csv(TRAIN, usecols=FEATURES)[FEATURES].to_numpy(float)
    if train.shape != (48662, 16):
        raise SystemExit(f"training rows {train.shape}")
    text = HEADER.read_text(encoding="utf-8")
    mu, sd = header_array(text, "SCALER_MEAN"), header_array(text, "SCALER_STD")
    rel = np.maximum(np.abs(mu - train.mean(0)) / np.maximum(np.abs(train.mean(0)), 1e-12),
                     np.abs(sd - train.std(0)) / train.std(0))
    if (rel > 1e-3).any():
        raise SystemExit(f"M0.2 failed: scaler differs from training rows, max rel {rel.max():.2e} at "
                         f"{FEATURES[int(rel.argmax())]}")
    lo, hi = train.min(0), train.max(0)
    p01, p99 = np.percentile(train, 1, axis=0), np.percentile(train, 99, axis=0)
    z = (board - mu) / sd
    out = []
    for j, f in enumerate(FEATURES):
        b = board[:, j]
        med = float(np.median(b))
        row = {"input": f, "group": GROUP[f], "board_median": med, "board_min": float(b.min()), "board_max": float(b.max()),
               "train_min": float(lo[j]), "train_p01": float(p01[j]), "train_p99": float(p99[j]), "train_max": float(hi[j]),
               "share_outside_train_range": float(((b < lo[j]) | (b > hi[j])).mean()),
               "z_median": float(np.median(z[:, j])), "z_min": float(z[:, j].min()), "z_max": float(z[:, j].max()),
               "share_abs_z_gt_3": float((np.abs(z[:, j]) > 3).mean()),
               "board_median_train_percentile": float((train[:, j] < med).mean() * 100)}
        row["out_of_distribution"] = (GROUP[f] != "scale_free") and (
            row["share_outside_train_range"] > 0.10 or not (p01[j] <= med <= p99[j]))
        out.append(row)
    table = pd.DataFrame(out)
    amp = table[table.group != "scale_free"]
    summary = {
        "preregistration": "docs/board-input-range-preregistration.md (eb986d6)",
        "decisions": 35, "training_rows": 48662,
        "share_of_all_board_values_outside_training_range": float(
            ((board < lo) | (board > hi)).mean()),
        "amplitude_inputs_out_of_distribution": amp.loc[amp.out_of_distribution, "input"].tolist(),
        "reading": ("narrow the device claim" if amp.out_of_distribution.any()
                    else "no amplitude-dependent input out of distribution; keep the unmeasured-effect caveat"),
        "inputs_sha256": {"capture": SIDECAR_SHA, "training": sha256(TRAIN), "header": sha256(HEADER)},
        "script_sha256": sha256(__file__),
    }
    OUT.mkdir(parents=True)
    table.to_csv(OUT / "input_range.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table[["input", "group", "share_outside_train_range", "z_median", "z_min", "z_max", "share_abs_z_gt_3",
                 "board_median_train_percentile", "out_of_distribution"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, indent=1))


if __name__ == "__main__":
    main()

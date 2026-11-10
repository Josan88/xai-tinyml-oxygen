"""Does the dashboard's explanation state say anything about whether the result is right?

For every OpenOximetry window, replicate what the operator page would show for the
shipped Baseline (v1) and Header-aligned (v2) headers with the training-mean
reference: the explanation gate (shown / withheld for 'zero' or 'opposes') and,
when shown, the driverSummary reading of the top three bars. Then compare how often
the 0.50 decision is right inside each state with the rate on that side overall.

In-sample: the shipped header scored on its own training table, because that is the
model whose explanation the dashboard shows. Fold models would explain differently.
Windows are correlated within a recording, so intervals use a recording-cluster
bootstrap. Host only: no board I/O, no header or firmware writes, no existing
outputs/ directory is written.

Run: py -3.14 scripts/dashboard_reading_reliability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import lamaakal_metrics as lm  # noqa: E402
import lamaakal_models as models  # noqa: E402

OUT = ROOT / "outputs" / "dashboard-reading-reliability-v1"
THRESHOLD = 0.50
EXPLANATION_ZERO = 0.0005  # dashboard_core.js: prints as 0.000 below this
BOOT = 1000
SEED = 20261105

IR_FEATURES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
V2_FEATURES = IR_FEATURES + ("ratio_of_ratios", "ir_perfusion_index")
SOURCES = {
    "v1": (ROOT / "outputs/openox_canonical_window_features.csv",
           ROOT / "firmware/Arduino Nano 33/include/kan_model_weights.h", IR_FEATURES),
    "v2": (ROOT / "outputs/openox-v2/window_features.csv",
           ROOT / "firmware/Arduino Nano 33/include/kan_model_weights_v2.h", V2_FEATURES),
}
COUNT = {1: "one", 2: "two", 3: "three"}


def reading(top3):
    """dashboard_core.js driverSummary, as a state name."""
    active = [d for d in top3 if abs(d) >= EXPLANATION_ZERO]
    n = len(active)
    up = sum(1 for d in active if d > 0)
    down = n - up
    if n == 0:
        return "none"
    if n == 1:
        return "only one feature moved"
    if n == 2:
        return "only two: " + ("both raised" if up == 2 else "both lowered" if down == 2 else "one raised, one lowered")
    if up == 3:
        return "all three raised"
    if down == 3:
        return "all three lowered"
    return f"pulled both ways: {COUNT[up]} raised, {COUNT[down]} lowered" if up and down else "?"


def screen_state(p, top3):
    """explanationGate (live: Candidate flag == P >= 0.50), then the reading."""
    flagged = p >= THRESHOLD
    lead = top3[0]
    if not abs(lead) >= EXPLANATION_ZERO:
        return "withheld: zero"
    if (lead < 0) if flagged else (lead > 0):
        return "withheld: opposes"
    return reading(top3)


def cluster_ci(frame, state, side_mask):
    """Recording-cluster bootstrap of (correct rate in state) - (correct rate on side)."""
    rng = np.random.default_rng(SEED)
    sub = frame[side_mask]
    in_state = sub["state"] == state
    per = pd.DataFrame({
        "rec": sub["recording_id"],
        "n_s": in_state.astype(int),
        "c_s": (in_state & sub["correct"]).astype(int),
        "n_a": 1,
        "c_a": sub["correct"].astype(int),
    }).groupby("rec").sum()
    arr = per.to_numpy(float)
    k = len(arr)
    diffs, rates = [], []
    for _ in range(BOOT):
        s = arr[rng.integers(0, k, k)].sum(axis=0)
        if s[0] == 0:
            continue
        rate = s[1] / s[0]
        rates.append(rate)
        diffs.append(rate - s[3] / s[2])
    lo, hi = np.percentile(rates, [2.5, 97.5])
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(dlo), float(dhi), int(len(diffs))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows, summary = [], {"threshold": THRESHOLD, "explanation_zero": EXPLANATION_ZERO,
                         "bootstrap": {"resamples": BOOT, "seed": SEED, "cluster": "recording_id"},
                         "scope": "in-sample: shipped header on its own training table; training-mean reference; "
                                  "deltas rounded to 4 decimals as in the BLE packet"}
    for version, (features_csv, header, features) in SOURCES.items():
        frame = pd.read_csv(features_csv)
        frame["recording_id"] = frame["recording_id"].astype(str)
        model = models.HeaderKAN(header)
        scaled = model.standardize(frame.loc[:, features].to_numpy(np.float32))
        probability, attribution = model.mean_perturbation(scaled)
        order = lm.firmware_abs_order(attribution)
        top3 = np.round(np.take_along_axis(attribution, order[:, :3], axis=1), 4)
        y = frame["label"].to_numpy(int)
        flagged = probability >= THRESHOLD
        frame["state"] = [screen_state(p, t) for p, t in zip(probability, top3)]
        frame["flagged"] = flagged
        frame["correct"] = np.where(flagged, y == 1, y == 0)
        summary[version] = {"windows": int(len(frame)), "recordings": int(frame["recording_id"].nunique())}
        for side, mask in (("flagged", flagged), ("not flagged", ~flagged)):
            side_rate = float(frame.loc[mask, "correct"].mean())
            summary[version][side] = {"n": int(mask.sum()), "correct_rate": side_rate}
            for state, group in frame[mask].groupby("state"):
                lo, hi, dlo, dhi, used = cluster_ci(frame, state, mask)
                rows.append({
                    "version": version, "side": side, "screen_state": state, "n": int(len(group)),
                    "share_of_side": len(group) / int(mask.sum()),
                    "correct_rate": float(group["correct"].mean()),
                    "correct_rate_ci_lo": lo, "correct_rate_ci_hi": hi,
                    "side_correct_rate": side_rate,
                    "difference": float(group["correct"].mean()) - side_rate,
                    "difference_ci_lo": dlo, "difference_ci_hi": dhi,
                    "recordings_in_state": int(group["recording_id"].nunique()),
                    "bootstrap_resamples_used": used,
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "reliability_by_screen_state.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["version", "side", "screen_state", "n", "share_of_side", "correct_rate",
                     "correct_rate_ci_lo", "correct_rate_ci_hi", "side_correct_rate",
                     "difference_ci_lo", "difference_ci_hi", "recordings_in_state"]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

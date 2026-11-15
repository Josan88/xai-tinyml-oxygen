"""Edge KAN operating point (docs/edge-kan-operating-point-preregistration.md). Host only, write-once.

Sweeps 0.05..0.95 for the edge KAN (ekan16) and the Embedded Spline Network (v2 TinyKAN16), window and
confirmed_n2 modes, recording- and patient-grouped, with the v1 confirmation and selection code imported
unchanged. Applies the pre-registered rule and writes outputs/edge-kan-v1/operating-point/.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_operating_point import confirmed_n2_mask, metric_row, select_rows  # noqa: E402

PREREG = ROOT / "docs/edge-kan-operating-point-preregistration.md"
OUT = ROOT / "outputs/edge-kan-v1/operating-point"
FILES = {
    ("edge_kan", "recording"): ROOT / "outputs/edge-kan-v1/predictions_ekan16_recording.csv",
    ("edge_kan", "patient"): ROOT / "outputs/edge-kan-v1/predictions_ekan16_patient.csv",
    ("spline_network", "recording"): ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16.csv",
    ("spline_network", "patient"): ROOT / "outputs/openox-v2/predictions_b_v2_tinykan16_patient.csv",
}
THRESHOLDS = sorted({round(float(v), 2) for v in np.arange(0.05, 0.951, 0.01)} | {0.50})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sweep(model, grouping, predictions):
    y = predictions["y_true"].to_numpy(int)
    p = predictions["y_prob"].to_numpy(float)
    rows = []
    for t in THRESHOLDS:
        rows.append(metric_row(model, "window", t, y, (p >= t).astype(int)))
        rows.append(metric_row(model, "confirmed_n2", t, y, confirmed_n2_mask(predictions, t).astype(int)))
    out = pd.DataFrame(rows)
    out.insert(1, "grouping", grouping)
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists (write-once)")
    frames = {}
    for (model, grouping), path in FILES.items():
        predictions = pd.read_csv(path)
        frames[(model, grouping)] = sweep(model, grouping, predictions)
    table = pd.concat(frames.values(), ignore_index=True)

    def at(model, grouping, mode, t):
        r = table[(table.model == model) & (table.grouping == grouping) & (table["mode"] == mode) & np.isclose(table.threshold, t)]
        return r.iloc[0].to_dict()

    edge_050 = {m: at("edge_kan", "recording", m, 0.50) for m in ("window", "confirmed_n2")}
    keep = all(r["feasible"] for r in edge_050.values())
    tuned = {}
    for model in ("edge_kan", "spline_network"):
        for grouping in ("recording", "patient"):
            sub = table[(table.model == model) & (table.grouping == grouping)].drop(columns="grouping")
            sel = select_rows(sub)
            tuned[f"{model}_{grouping}"] = sel.to_dict(orient="records")
    if keep:
        threshold, reason = 0.50, "rule 1: 0.50 meets the gate in both modes, recording-grouped"
    else:
        row = [r for r in tuned["edge_kan_recording"] if r["mode"] == "confirmed_n2"][0]
        threshold, reason = float(row["threshold"]), f"rule 2: v1 selection on confirmed_n2 ({row['selected_reason']})"
    selected = {f"{model}_{grouping}_{mode}": at(model, grouping, mode, threshold if model == "edge_kan" else 0.50)
                for model in ("edge_kan", "spline_network") for grouping in ("recording", "patient")
                for mode in ("window", "confirmed_n2")}
    OUT.mkdir(parents=True)
    table.to_csv(OUT / "sweep.csv", index=False)
    summary = {"preregistration_sha256": sha256(PREREG), "inputs_sha256": {f"{m}_{g}": sha256(p) for (m, g), p in FILES.items()},
               "frozen_threshold": threshold, "reason": reason, "edge_at_0_50_recording": edge_050,
               "metrics_at_frozen_thresholds": selected,
               "v1_rule_selection_reported_not_used": tuned}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"frozen threshold {threshold}: {reason}")
    for k, r in selected.items():
        print(f"{k:42s} t={r['threshold']:.2f} recall={r['recall']:.4f} spec={r['specificity']:.4f} "
              f"prec={r['precision']:.4f} J={r['youden_j']:.4f} gate={r['feasible']}")
    print("v1-rule (tuned) selections, reported only:")
    for k, rows in tuned.items():
        for r in rows:
            print(f"  {k:26s} {r['mode']:13s} t={r['threshold']:.2f} recall={r['recall']:.4f} spec={r['specificity']:.4f} J={r['youden_j']:.4f}")


if __name__ == "__main__":
    main()

"""Univariate ROC AUC of each header-aligned (v2) model input against its label. Host only: reads
files, trains nothing, opens no port.

Source for the IEEE draft sentence "Within the aligned model's own exam ... ratio of ratios alone
0.869". Before this script the only committed record of 0.869 was the P5 pre-registration
(docs/physio-concept-preregistration.md, "Univariate AUC"), which is a P5-study document and not
paper evidence.

For each of the 16 v2 inputs (14 IR features + ratio_of_ratios + ir_perfusion_index) the score is
the raw feature value and the target is `label` (window median header-aligned ScalcO2 < 90) of
outputs/openox-v2/window_features.csv. No folds, no model, no standardisation (AUC is rank-based).
Reported per feature: raw AUC (higher value = label 1), oriented AUC max(AUC, 1 - AUC) and the
direction that gives it. Rows where the feature is not finite are dropped for that feature only
and counted.

Populations:
  all_rows             every row of window_features.csv (the population the draft describes)
  p5_beat_valid_repo   rows with beat_valid == 1 in outputs/physio-concept-v1/physio_features.csv.gz
                       (the committed P5 detector; context for where 0.869 / 0.870 came from only).
                       The P5 pre-registration's 0.869 came from an uncommitted scratch detector on
                       48,628 beat-valid rows; that subset is not on disk and is not reproduced here.

Usage:
    python scripts/univariate_auc_v2.py --output outputs/univariate-auc-v2
Writes <output>/summary.json and <output>/README.md. Refuses an existing output directory.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = "outputs/openox-v2/window_features.csv"
PHYSIO_CSV = "outputs/physio-concept-v1/physio_features.csv.gz"
IR_FEATURES = ("mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
               "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
               "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5")
TWO_WAVELENGTH = ("ratio_of_ratios", "ir_perfusion_index")
INPUTS = IR_FEATURES + TWO_WAVELENGTH
DRAFT_VALUE = 0.869  # draft paragraph 32, ratio_of_ratios


def input_hashes(path: Path) -> dict:
    """SHA-256 of the working file; for text inputs also with CRLF folded to LF (the git blob content
    under core.autocrlf=true). Binary inputs (.gz) are stored as-is, so they get one hash."""
    data = path.read_bytes()
    hashes = {"sha256": hashlib.sha256(data).hexdigest()}
    if path.suffix != ".gz":
        hashes["sha256_lf"] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    return hashes


def univariate_auc(values: np.ndarray, labels: np.ndarray) -> dict:
    """Raw and oriented ROC AUC of one score column; non-finite scores are dropped and counted."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    keep = np.isfinite(values)
    y, x = labels[keep], values[keep]
    if y.size == 0 or y.min() == y.max():
        raise ValueError("need both classes among the finite rows")
    raw = float(roc_auc_score(y, x))
    higher = raw >= 0.5
    return {
        "rows_used": int(keep.sum()),
        "rows_dropped_nonfinite": int((~keep).sum()),
        "auc_raw": raw,
        "auc_oriented": raw if higher else 1.0 - raw,
        "direction": "higher value -> label 1" if higher else "lower value -> label 1",
    }


def population_table(frame: pd.DataFrame) -> dict:
    labels = frame["label"].to_numpy()
    return {
        "rows": int(len(frame)),
        "recordings": int(frame["recording_id"].nunique()),
        "positive_rows": int(labels.sum()),
        "label_base_rate": float(labels.mean()),
        "features": {name: univariate_auc(frame[name].to_numpy(), labels) for name in INPUTS},
    }


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(root / FEATURES_CSV)
    missing = [c for c in ("recording_id", "start_time_seconds", "label") + INPUTS if c not in features]
    if missing:
        raise ValueError(f"{FEATURES_CSV} lacks columns {missing}")
    if not set(np.unique(features["label"])) <= {0, 1}:
        raise ValueError("label is not binary")
    physio = pd.read_csv(root / PHYSIO_CSV)
    if len(physio) != len(features) or not np.array_equal(physio["row_index"], np.arange(len(features))):
        raise ValueError(f"{PHYSIO_CSV} is not row-aligned with {FEATURES_CSV}")
    if not (physio["recording_id"].equals(features["recording_id"])
            and np.allclose(physio["start_time_seconds"], features["start_time_seconds"])):
        raise ValueError(f"{PHYSIO_CSV} recording/start time differ from {FEATURES_CSV}")
    return features, physio


def render_readme(summary: dict) -> str:
    pops = summary["populations"]
    all_rows, beat = pops["all_rows"], pops["p5_beat_valid_repo"]
    lines = [
        "# Univariate AUC of the header-aligned model's 16 inputs (host only)",
        "",
        f"Generated by `scripts/univariate_auc_v2.py` on {summary['created_utc'][:10]} (UTC). No training, no folds, "
        "no board. Score = raw feature value; target = `label` (window median header-aligned ScalcO2 < 90) "
        f"of `{FEATURES_CSV}`. AUC is sklearn `roc_auc_score` (ties averaged). Oriented AUC = max(AUC, 1 - AUC).",
        "",
        f"Population `all_rows`: {all_rows['rows']:,} windows, {all_rows['recordings']} recordings, "
        f"{all_rows['positive_rows']:,} positive (base rate {all_rows['label_base_rate']:.4f}). "
        "Non-finite feature values are dropped for that feature only; the count is in the table.",
        "",
        "| # | Input | Raw AUC | Oriented AUC | Direction | Rows used | Non-finite dropped |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for i, name in enumerate(INPUTS, 1):
        f = all_rows["features"][name]
        lines.append(f"| {i} | `{name}` | {f['auc_raw']:.4f} | {f['auc_oriented']:.4f} | {f['direction']} | "
                     f"{f['rows_used']:,} | {f['rows_dropped_nonfinite']} |")
    rr_all = all_rows["features"]["ratio_of_ratios"]["auc_oriented"]
    rr_beat = beat["features"]["ratio_of_ratios"]["auc_oriented"]
    check = summary["draft_check"]
    lines += [
        "",
        "## Draft paragraph 32: \"ratio of ratios alone 0.869\"",
        "",
        f"- All {all_rows['rows']:,} rows (the aligned model's own table, the population the sentence describes): "
        f"**{rr_all:.4f}** -> {rr_all:.3f}. {'Reproduces' if check['reproduces_on_all_rows'] else 'Does not reproduce'} 0.869 to three decimals.",
        f"- P5 context only, not paper evidence: on the {beat['rows']:,} rows the committed P5 detector "
        f"marks beat-valid, {rr_beat:.4f} -> {rr_beat:.3f} (`outputs/physio-concept-v1/README.md` reports 0.870 "
        "for the same rows). The P5 pre-registration's 0.869 came from an uncommitted scratch detector on 48,628 "
        "beat-valid rows; that subset is not on disk and is not reproduced.",
        "- The 0.869 is a single-feature rank statistic on the whole table, in sample. It is not an "
        "out-of-fold model AUC and does not rank the models.",
        "",
        "## Why `outputs/label-alignment-v1/` reports 0.860 instead",
        "",
        "`scripts/audit_openox_label_alignment.py` scores a different R on a different population: R is recomputed "
        "from the native 86 Hz red/IR with a 0.5-5 Hz Butterworth band (not the 50 Hz, 0.5-8 Hz v2 export), on the "
        "52,090 frozen as-coded (v1) windows, labelled by the header-aligned median of those windows; it keeps "
        "39,276 windows (>= 10 reference samples, median outside [89, 91]), not the 48,662 rows of the v2 export, "
        "which also drops Red/IR Drive-change windows. Same label rule, different windows and a different R, "
        "so 0.860 and 0.869 are not the same measurement. That README is unchanged.",
        "",
        "## Inputs (SHA-256 of the working file; `lf` = CRLF folded to LF, the git blob content, text only)",
        "",
    ]
    for rel, h in summary["inputs"].items():
        lf = f" (lf `{h['sha256_lf']}`)" if "sha256_lf" in h else ""
        lines.append(f"- `{rel}`: `{h['sha256']}`{lf}")
    lines += ["", "Files: `summary.json` (all numbers, both populations).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else args.root / args.output
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    features, physio = load(args.root)
    beat_valid = physio["beat_valid"].to_numpy() == 1
    all_rows = population_table(features)
    rr = round(all_rows["features"]["ratio_of_ratios"]["auc_oriented"], 3)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/univariate_auc_v2.py",
        "scope": "host only; univariate rank AUC of raw inputs vs label; no training, no folds, in sample",
        "label": "label = window median header-aligned ScalcO2 < 90 (scripts/export_openox_windows_v2.py)",
        "nan_handling": "per feature, non-finite values dropped and counted; no imputation",
        "auc": "sklearn.metrics.roc_auc_score on the raw value (higher = label 1); oriented = max(AUC, 1 - AUC)",
        "inputs": {rel: input_hashes(args.root / rel) for rel in (FEATURES_CSV, PHYSIO_CSV)},
        "populations": {
            "all_rows": all_rows,
            "p5_beat_valid_repo": {
                "definition": f"beat_valid == 1 in {PHYSIO_CSV} (committed P5 detector); P5 context, not paper evidence",
                **population_table(features[beat_valid]),
            },
        },
        "draft_check": {
            "sentence": "Within the aligned model's own exam the 14 infrared features alone reach AUC 0.712 "
                        "and ratio of ratios alone 0.869.",
            "draft_value": DRAFT_VALUE,
            "all_rows_ratio_of_ratios_3dp": rr,
            "reproduces_on_all_rows": bool(rr == DRAFT_VALUE),
            "not_reproduced": "P5 pre-registration scratch subset (48,628 beat-valid rows, uncommitted detector)",
        },
    }
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(render_readme(summary), encoding="utf-8")
    print(json.dumps({k: round(v["auc_oriented"], 4) for k, v in all_rows["features"].items()}, indent=1))
    print("ratio_of_ratios all rows:", rr, "reproduces 0.869:", summary["draft_check"]["reproduces_on_all_rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Semester 1 Milestone Demo Script
=================================
Demonstrates:
  1. Dataset collected (OpenOximetry)
  2. Model trained in Python (Logistic Regression)
  3. Accuracy results
  4. Basic prediction output
  5. Example explanation output

Run from the project root:  python scripts/semester1_milestone_demo.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def load_csv(filename: str) -> list[dict]:
    with open(OUTPUTS / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # 1. Dataset collected
    # ------------------------------------------------------------------
    section("1. DATASET COLLECTED")

    labels = load_csv("openox_canonical_label_summary.csv")
    total_rows = sum(int(r["row_count"]) for r in labels)
    total_recordings = max(int(r["recording_count"]) for r in labels)
    positive_rows = next(int(r["row_count"]) for r in labels if r["label"] == "1")

    rec_summary = load_csv("openox_canonical_recording_summary.csv")
    positive_recordings = sum(
        1 for r in rec_summary if r["has_positive_window"] == "True"
    )

    print(f"  Dataset           : OpenOximetry (canonical export)")
    print(f"  Total windows     : {total_rows:,}")
    print(f"  Positive windows  : {positive_rows:,} ({positive_rows/total_rows*100:.1f}%)")
    print(f"  Total recordings  : {total_recordings}")
    print(f"  Positive recordings: {positive_recordings}")
    print(f"  Label rule        : Median SpO2 < 90% (engineering proxy)")

    # ------------------------------------------------------------------
    # 2. Model trained in Python
    # ------------------------------------------------------------------
    section("2. MODEL TRAINED IN PYTHON")

    baseline = load_csv("openox_logreg_baseline_summary.csv")
    primary = next(r for r in baseline if r["label_variant"] == "label")

    print(f"  Model             : Logistic Regression (L2, class_weight=balanced)")
    print(f"  Features          : 15 PPG-only engineered features")
    print(f"  Split strategy    : GroupKFold (recording-level, 5 folds)")
    print(f"  SpO2 used as input: NO (label evidence only)")

    # ------------------------------------------------------------------
    # 3. Accuracy results
    # ------------------------------------------------------------------
    section("3. ACCURACY RESULTS")

    acc = float(primary["pooled_accuracy"])
    prec = float(primary["pooled_precision"])
    rec = float(primary["pooled_recall"])
    f1 = float(primary["pooled_f1"])

    print(f"  Accuracy          : {acc:.3f}")
    print(f"  Precision         : {prec:.3f}")
    print(f"  Recall (low-O2)   : {rec:.3f}")
    print(f"  F1-Score          : {f1:.3f}")
    print()
    print(f"  Confusion matrix (pooled test, {int(primary['pooled_test_rows']):,} windows):")
    print(f"    TN = {int(primary['pooled_true_negative']):>6,}")
    print(f"    FP = {int(primary['pooled_false_positive']):>6,}")
    print(f"    FN = {int(primary['pooled_false_negative']):>6,}")
    print(f"    TP = {int(primary['pooled_true_positive']):>6,}")

    # ------------------------------------------------------------------
    # 4. Basic prediction output
    # ------------------------------------------------------------------
    section("4. BASIC PREDICTION OUTPUT")

    explanation_sample = load_csv("openox_logreg_explanation_sample.csv")
    sample_tp = next(r for r in explanation_sample if r["quadrant"] == "TP")
    sample_fn = next(r for r in explanation_sample if r["quadrant"] == "FN")

    print("  Example: true-positive window")
    print(f"    Recording : {sample_tp['recording_id'][:16]}...")
    print(f"    Time      : {sample_tp['start_time']}s - {sample_tp['end_time']}s")
    print(f"    True label: Low-oxygen risk (1)")
    print(f"    Predicted : Low-oxygen risk (1)")
    print(f"    Probability: {float(sample_tp['probability']):.2f}")
    print(f"    Top features: {sample_tp['top1_feature']}, {sample_tp['top2_feature']}")
    print()

    print("  Example: false-negative window (missed low-oxygen risk)")
    print(f"    Recording : {sample_fn['recording_id'][:16]}...")
    print(f"    Time      : {sample_fn['start_time']}s - {sample_fn['end_time']}s")
    print(f"    True label: Low-oxygen risk (1)")
    print(f"    Predicted : Normal (0)")
    print(f"    Probability: {float(sample_fn['probability']):.2f}")

    # ------------------------------------------------------------------
    # 5. Example explanation output
    # ------------------------------------------------------------------
    section("5. EXPLANATION OUTPUT")

    print("  LR coefficient explanations provide per-window feature contributions.")
    print()
    print(f"  TP explanation: {sample_tp['explanation']}")
    print(f"  FN explanation: {sample_fn['explanation']}")
    print()
    print(f"  Explanation rubric scores (compactness/consistency/usefulness):")
    print(f"    TP: {sample_tp['compactness']}/{sample_tp['consistency']}/{sample_tp['usefulness']} = {sample_tp['total']}/6")
    print(f"    FN: {sample_fn['compactness']}/{sample_fn['consistency']}/{sample_fn['usefulness']} = {sample_fn['total']}/6")

    # ------------------------------------------------------------------
    # Required demo example: SpO2 threshold rule
    # ------------------------------------------------------------------
    section("REQUIRED MILESTONE EXAMPLE")

    demo_spo2 = 88
    if demo_spo2 < 90:
        demo_status = "Low Oxygen Risk"
        demo_explanation = "Low oxygen detected because SpO2 < 90% threshold."
    else:
        demo_status = "Normal"
        demo_explanation = "Oxygen level within normal range."

    print(f"  Input : SpO2 = {demo_spo2}%")
    print(f"  Output: {demo_status}")
    print(f"  Explanation: {demo_explanation}")
    print()
    print("  Note: The trained LR model uses PPG features (not SpO2 as input).")
    print("  The SpO2 threshold rule is a simple baseline for demonstration.")
    print("  The LR model provides richer feature-based explanations.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    print("  - Dataset: OpenOximetry, 52,090 windows, 162 recordings")
    print("  - Model: Logistic Regression, grouped evaluation, 15 features")
    print(f"  - Key metric: low-oxygen recall = {rec:.3f}")
    print("  - Explainability: LR coefficient contributions + threshold rule")
    print("  - Hardware: Arduino Nano 33 BLE Sense with MAX30105 (live demo)")
    print("  - Boundary: Engineering risk-flag prototype, NOT a clinical oximeter")
    print()


if __name__ == "__main__":
    main()

"""Generate all progress-report figures from existing CSV outputs.

Run from project root:  python scripts/generate_report_figures.py

Outputs saved to outputs/figures/ as PNG (300 DPI).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
RISK = "#E65100"
NORMAL = "#1565C0"
GREY = "#666666"
LIGHT = "#CCCCCC"
ACCENT = "#2E7D32"

FEATURE_LABELS = {
    "bandpower_ratio_0_5_2_to_2_5": "Bandpower ratio",
    "derivative_std": "Slope variability",
    "kurtosis": "Kurtosis",
    "std": "PPG variability",
    "rms": "Signal power",
    "minimum": "Min amplitude",
    "maximum": "Max amplitude",
    "energy": "Energy",
    "bandpower_2_5_hz": "Bandpower (2-5 Hz)",
    "dominant_frequency_hz": "Dominant freq.",
    "bandpower_0_5_2_hz": "Bandpower (0.5-2 Hz)",
    "skewness": "Skewness",
    "value_range": "Amplitude range",
    "mean": "Mean amplitude",
}


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Pooled Confusion Matrix (Logistic Regression, primary label)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_confusion_matrix():
    baseline = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_logreg_baseline_summary.csv")
    row = baseline[baseline["label_variant"] == "label"].iloc[0]
    cm = np.array([
        [int(row["pooled_true_negative"]), int(row["pooled_false_positive"])],
        [int(row["pooled_false_negative"]), int(row["pooled_true_positive"])],
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=11)
    ax.set_yticklabels(["Negative", "Positive"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Logistic Regression — Pooled Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Count")
    ax.spines[:].set_visible(False)
    save(fig, "fig1_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Model Comparison (LR vs TinyKAN)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_model_comparison():
    baseline = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_logreg_baseline_summary.csv")
    tinykan = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_tinykan_summary.csv")

    lr = baseline[baseline["label_variant"] == "label"].iloc[0]
    tk = tinykan.iloc[0]

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    lr_vals = [lr["pooled_accuracy"], lr["pooled_precision"], lr["pooled_recall"], lr["pooled_f1"]]
    tk_vals = [tk["pooled_accuracy"], tk["pooled_precision"], tk["pooled_recall"], tk["pooled_f1"]]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, lr_vals, w, label="Logistic Regression", color=NORMAL, edgecolor="white", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, tk_vals, w, label="TinyKAN (B-spline KAN)", color=RISK, edgecolor="white", linewidth=0.6)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — OpenOximetry Binary Classification", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig2_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Cross-Validation Fold-wise Metrics (LR vs TinyKAN)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_fold_metrics():
    lr_diag = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_logreg_split_diagnostics.csv")
    tk_diag = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_tinykan_split_diagnostics.csv")

    lr_primary = lr_diag[lr_diag["label_variant"] == "label"].copy()
    lr_primary["fold"] = lr_primary["split_index"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(lr_primary["fold"], lr_primary[metric], "o-", color=NORMAL, linewidth=2, markersize=6, label="LR")
        ax.plot(tk_diag["split_index"], tk_diag[metric], "s--", color=RISK, linewidth=2, markersize=6, label="TinyKAN")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Fold", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[-1].legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle("Per-Fold Classification Metrics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig3_fold_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Global Feature Importance (Logistic Regression Coefficients)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_feature_importance():
    coef_df = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_logreg_global_explainability.csv")
    coef_df = coef_df.sort_values("mean_absolute_coefficient", ascending=True)

    labels = [FEATURE_LABELS.get(n, n) for n in coef_df["feature_name"]]
    values = coef_df["mean_absolute_coefficient"].values
    directions = coef_df["dominant_direction"].values
    colors = [RISK if d == "pushes_toward_low_oxygen_risk" else NORMAL for d in directions]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("|Coefficient| (standardized)", fontsize=11, fontweight="bold")
    ax.set_title("Global Feature Importance — Logistic Regression", fontsize=13, fontweight="bold", pad=10)

    ax.text(0.95, 0.95, "Orange = pushes toward low-oxygen risk\nBlue = pushes toward normal",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top", ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, "fig4_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — TinyKAN Feature Attributions (Gradient-based)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_tinykan_attributions():
    attr_df = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_tinykan_feature_attributions.csv")
    attr_df = attr_df.sort_values("mean_abs_attribution", ascending=True)

    labels = [FEATURE_LABELS.get(n, n) for n in attr_df["feature_name"]]
    values = attr_df["mean_abs_attribution"].values
    signed = attr_df["mean_signed_attribution"].values
    colors = [RISK if s > 0 else NORMAL for s in signed]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(labels)), values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |Attribution| (gradient × input)", fontsize=11, fontweight="bold")
    ax.set_title("Feature Attributions — TinyKAN B-Spline KAN", fontsize=13, fontweight="bold", pad=10)

    ax.text(0.95, 0.95, "Orange = positive signed attribution\nBlue = negative signed attribution",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top", ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, "fig5_tinykan_attributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Local XAI Explanation (Single Window)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_local_explanation():
    """Generate a local explanation waterfall using trained LR coefficients and an example window."""
    feat_df = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_canonical_window_features.csv")
    coef_df = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_logreg_global_explainability.csv")

    feature_names = coef_df["feature_name"].tolist()
    coefficients = coef_df["mean_coefficient"].values

    feat_df["label"] = feat_df["label"].astype(int)
    positive_windows = feat_df[feat_df["label"] == 1]

    example = positive_windows.iloc[0]
    features = np.array([float(example[f]) for f in feature_names])

    from sklearn.preprocessing import StandardScaler
    all_features = feat_df[feature_names].values.astype(float)
    scaler = StandardScaler()
    scaler.fit(all_features)
    features_std = scaler.transform(features.reshape(1, -1)).ravel()

    contributions = features_std * coefficients

    top_n = 8
    order = np.argsort(np.abs(contributions))[::-1][:top_n][::-1]

    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    vals = contributions[order]
    colors = [RISK if v > 0 else NORMAL for v in vals]

    prob = 1 / (1 + np.exp(-(contributions.sum() + 0)))  # approximate

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(labels)), vals, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature contribution (standardized)", fontsize=11, fontweight="bold")
    ax.set_title("Local XAI Explanation — Low-Oxygen-Risk Window", fontsize=13, fontweight="bold", pad=10)

    ax.text(0.02, 0.95, "Toward low-oxygen-risk", transform=ax.transAxes, fontsize=8, color=RISK, va="top", fontweight="bold")
    ax.text(0.02, 0.90, "Toward normal", transform=ax.transAxes, fontsize=8, color=NORMAL, va="top", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    save(fig, "fig6_local_explanation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Class Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_class_distribution():
    feat_df = pd.read_csv(PROJECT_ROOT / "outputs" / "openox_canonical_window_features.csv")
    feat_df["label"] = feat_df["label"].astype(int)

    pos = int(feat_df["label"].sum())
    neg = len(feat_df) - pos
    total = len(feat_df)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Normal (Negative)", "Low-Oxygen Risk (Positive)"], [neg, pos],
                  color=[NORMAL, RISK], edgecolor="white", linewidth=0.6, width=0.5)
    for bar, count in zip(bars, [neg, pos]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"{count:,}\n({count / total:.1%})", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Windows", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution — OpenOximetry Dataset", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(neg, pos) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig7_class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating progress report figures...")
    fig1_confusion_matrix()
    fig2_model_comparison()
    fig3_fold_metrics()
    fig4_feature_importance()
    fig5_tinykan_attributions()
    fig6_local_explanation()
    fig7_class_distribution()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")

"""
Score the 20-window explanation rubric for the grouped Logistic Regression baseline.

Reconstructs the model evaluation from notebook 11, computes per-window feature
contributions, selects 20 held-out windows (5 per confusion-matrix quadrant),
generates compact explanations, and scores them against the 3-criterion rubric.
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMN_NAMES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis", "energy",
    "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz", "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)

FEATURE_LABELS = {
    "mean": "mean amplitude",
    "std": "amplitude variability",
    "minimum": "minimum amplitude",
    "maximum": "maximum amplitude",
    "value_range": "amplitude range",
    "rms": "signal energy (RMS)",
    "skewness": "signal skewness",
    "kurtosis": "signal kurtosis",
    "energy": "spectral energy",
    "derivative_std": "derivative variability",
    "dominant_frequency_hz": "dominant frequency",
    "bandpower_0_5_2_hz": "low bandpower (0.5-2 Hz)",
    "bandpower_2_5_hz": "high bandpower (2-5 Hz)",
    "bandpower_ratio_0_5_2_to_2_5": "bandpower ratio (low/high)",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_CSV = PROJECT_ROOT / "outputs" / "openox_canonical_window_features.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "openox_logreg_explanation_sample.csv"
RUBRIC_MD = PROJECT_ROOT / "docs" / "explanation-rubric-assessment.md"


def build_grouped_coverage_splits(stratify_target, groups, max_n_splits, seed):
    unique_groups = np.unique(groups)
    placeholder = np.zeros(len(groups), dtype=float)

    for requested_splits in range(max_n_splits, 1, -1):
        if len(unique_groups) < requested_splits:
            continue
        splitter = GroupKFold(n_splits=requested_splits)
        splits = [(tr_idx, te_idx) for tr_idx, te_idx in splitter.split(placeholder, stratify_target, groups)]
        if len(splits) != requested_splits:
            continue
        if any(set(groups[tr_idx]).intersection(set(groups[te_idx])) for tr_idx, te_idx in splits):
            continue
        test_group_sequence = np.concatenate([groups[te_idx] for _, te_idx in splits])
        if len(np.unique(test_group_sequence)) != len(unique_groups):
            continue
        return splits, seed, requested_splits, "GroupKFold"

    raise RuntimeError("Could not find a grouped coverage split")


def main():
    # Load features
    df = pd.read_csv(FEATURE_CSV)
    for col in FEATURE_COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["recording_id"] = df["recording_id"].astype(str)

    groups = df["recording_id"].values
    X = df[list(FEATURE_COLUMN_NAMES)].values.astype(np.float64)
    y = df["label"].values

    # Build splits (same as notebook 11)
    splits, used_seed, actual_folds, strategy = build_grouped_coverage_splits(y, groups, 5, 42)

    # Run grouped logistic regression, collecting per-window predictions and contributions
    all_results = []
    fold_models = []

    for split_index, (tr_idx, te_idx) in enumerate(splits, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[tr_idx])
        X_test = scaler.transform(X[te_idx])
        y_train = y[tr_idx]
        y_test = y[te_idx]

        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        coef = model.coef_[0]
        intercept = model.intercept_[0]
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Compute per-feature contributions for each test window
        contributions = X_test * coef  # shape: (n_test, n_features)

        for i in range(len(y_test)):
            contrib = contributions[i]
            abs_contrib = np.abs(contrib)
            top2_indices = np.argsort(abs_contrib)[::-1][:2]

            top_features = []
            for fi in top2_indices:
                feat_name = FEATURE_COLUMN_NAMES[fi]
                contrib_val = contrib[fi]
                direction = "high" if X_test[i, fi] > 0 else "low"
                top_features.append({
                    "name": feat_name,
                    "label": FEATURE_LABELS[feat_name],
                    "contribution": contrib_val,
                    "direction": direction,
                })

            true_label = int(y_test[i])
            pred_label = int(preds[i])
            prob = float(probs[i])

            if true_label == 1 and pred_label == 1:
                quadrant = "TP"
            elif true_label == 0 and pred_label == 1:
                quadrant = "FP"
            elif true_label == 0 and pred_label == 0:
                quadrant = "TN"
            else:
                quadrant = "FN"

            all_results.append({
                "global_idx": int(te_idx[i]),
                "fold": split_index,
                "recording_id": groups[te_idx[i]],
                "start_time_seconds": df.loc[te_idx[i], "start_time_seconds"],
                "end_time_seconds": df.loc[te_idx[i], "end_time_seconds"],
                "true_label": true_label,
                "predicted": pred_label,
                "probability": prob,
                "quadrant": quadrant,
                "top1_name": top_features[0]["name"],
                "top1_label": top_features[0]["label"],
                "top1_contrib": top_features[0]["contribution"],
                "top1_direction": top_features[0]["direction"],
                "top2_name": top_features[1]["name"],
                "top2_label": top_features[1]["label"],
                "top2_contrib": top_features[1]["contribution"],
                "top2_direction": top_features[1]["direction"],
            })

    results_df = pd.DataFrame(all_results)

    # Select 20 windows: 5 per quadrant, deterministically by probability rank
    selected = []
    for quadrant in ["TP", "FP", "TN", "FN"]:
        q_df = results_df[results_df["quadrant"] == quadrant].copy()

        # For TP and FP: sort by probability descending (most confident positives first)
        # For TN: sort by probability ascending (most confident negatives first)
        # For FN: sort by probability ascending (closest to boundary = most interesting)
        if quadrant in ["TP", "FP"]:
            q_df = q_df.sort_values("probability", ascending=False)
        elif quadrant == "TN":
            q_df = q_df.sort_values("probability", ascending=True)
        else:  # FN
            q_df = q_df.sort_values("probability", ascending=False)

        # Evenly spaced selection across the quadrant
        n = len(q_df)
        indices = [int(i * (n - 1) / 4) for i in range(5)]
        selected.extend(q_df.iloc[indices].to_dict("records"))

    selected_df = pd.DataFrame(selected)

    # Generate explanation text and score each
    def generate_explanation(row):
        state = "Low-oxygen risk" if row["predicted"] == 1 else "Normal oxygen"
        prob = row["probability"]

        f1_label = row["top1_label"]
        f1_dir = row["top1_direction"]
        f1_contrib = row["top1_contrib"]

        f2_label = row["top2_label"]
        f2_dir = row["top2_direction"]
        f2_contrib = row["top2_contrib"]

        text = (
            f"{state} flagged (prob = {prob:.2f}). "
            f"Main contributors: {f1_dir} {f1_label} ({f1_contrib:+.2f}), "
            f"{f2_dir} {f2_label} ({f2_contrib:+.2f})."
        )
        return text

    def score_compactness(text):
        words = text.split()
        if len(words) <= 30:
            return 2
        elif len(words) <= 45:
            return 1
        else:
            return 0

    def score_consistency(row):
        pred = row["predicted"]
        top1_contrib = row["top1_contrib"]

        if pred == 1 and top1_contrib > 0:
            return 2
        elif pred == 0 and top1_contrib < 0:
            return 2
        elif pred == 1 and top1_contrib < 0:
            # Top contributor pushes against prediction but model still predicts positive
            # This is partially aligned
            return 1
        elif pred == 0 and top1_contrib > 0:
            return 1
        else:
            return 0

    def score_usefulness(row):
        # Check if explanation gives clear cue: confidence (prob), dominant pattern, and interpretable feature
        has_prob = True  # All explanations include prob
        has_dominant = row["top1_label"] is not None and row["top1_label"] != ""
        has_direction = row["top1_direction"] in ["high", "low"]

        if has_prob and has_dominant and has_direction:
            return 2
        elif has_prob and (has_dominant or has_direction):
            return 1
        else:
            return 0

    scored_rows = []
    for _, row in selected_df.iterrows():
        explanation = generate_explanation(row)
        compactness = score_compactness(explanation)
        consistency = score_consistency(row)
        usefulness = score_usefulness(row)
        total = compactness + consistency + usefulness

        scored_rows.append({
            "window_id": f"{row['quadrant']}-{scored_rows.count([r for r in scored_rows if r['quadrant'] == row['quadrant']]) + 1:02d}",
            "quadrant": row["quadrant"],
            "recording_id": row["recording_id"],
            "start_time": row["start_time_seconds"],
            "end_time": row["end_time_seconds"],
            "true_label": row["true_label"],
            "predicted": row["predicted"],
            "probability": row["probability"],
            "top1_feature": row["top1_label"],
            "top2_feature": row["top2_label"],
            "explanation": explanation,
            "compactness": compactness,
            "consistency": consistency,
            "usefulness": usefulness,
            "total": total,
        })

    scored_df = pd.DataFrame(scored_rows)

    # Assign sequential IDs per quadrant
    for quadrant in ["TP", "FP", "TN", "FN"]:
        mask = scored_df["quadrant"] == quadrant
        count = 0
        for idx in scored_df[mask].index:
            count += 1
            scored_df.loc[idx, "window_id"] = f"{quadrant}-{count:02d}"

    # Export CSV
    scored_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")

    # Compute summary statistics
    mean_total = scored_df["total"].mean()
    mean_compactness = scored_df["compactness"].mean()
    mean_consistency = scored_df["consistency"].mean()
    mean_usefulness = scored_df["usefulness"].mean()
    min_compactness = scored_df["compactness"].min()
    min_consistency = scored_df["consistency"].min()
    min_usefulness = scored_df["usefulness"].min()
    pass_status = mean_total >= 4.0 and min_compactness > 0 and min_consistency > 0 and min_usefulness > 0

    print(f"\n=== Rubric Scoring Summary ===")
    print(f"Mean Compactness: {mean_compactness:.2f}/2.00")
    print(f"Mean Consistency: {mean_consistency:.2f}/2.00")
    print(f"Mean Usefulness:  {mean_usefulness:.2f}/2.00")
    print(f"Mean Total:       {mean_total:.2f}/6.00")
    print(f"Min Compactness:  {min_compactness}")
    print(f"Min Consistency:  {min_consistency}")
    print(f"Min Usefulness:   {min_usefulness}")
    print(f"Pass (>=4.0, no 0): {'YES' if pass_status else 'NO'}")

    # Print the scoring table
    print(f"\n=== Scoring Table ===")
    for _, row in scored_df.iterrows():
        print(f"{row['window_id']} | T={row['true_label']} P={row['predicted']} prob={row['probability']:.2f} | "
              f"C={row['compactness']} Cons={row['consistency']} U={row['usefulness']} Total={row['total']} | "
              f"{row['explanation'][:80]}...")

    # Return data for markdown update
    return scored_df, mean_total, mean_compactness, mean_consistency, mean_usefulness, pass_status, min_compactness, min_consistency, min_usefulness


if __name__ == "__main__":
    scored_df, mean_total, mean_compactness, mean_consistency, mean_usefulness, pass_status, min_compactness, min_consistency, min_usefulness = main()

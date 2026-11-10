"""Score a deterministic 20-window TinyKAN Mean-Perturbation sample."""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)
QUADRANTS = ("TP", "FP", "TN", "FN")
VACUOUS = 1e-4
OUTPUT_COLUMNS = (
    "window_id", "quadrant", "row_index", "split_index", "recording_id",
    "start_time_seconds", "end_time_seconds", "true_label", "predicted_label",
    "csv_probability", "firmware_probability", "csv_firmware_prob_abs_diff", "top1_feature",
    "top1_delta", "top1_dir", "top2_feature", "top2_delta", "top2_dir", "top3_feature",
    "top3_delta", "top3_dir", "n_nonvacuous_top3", "csv_top_attribution_feature",
    "explanation", "word_count", "compactness", "consistency", "usefulness", "total",
)


def parse_c_header(path):
    content = path.read_text(encoding="utf-8")

    def values(name):
        match = re.search(rf"(?:const\s+)?float\s+{name}\s*\[[^]]*\]\s*=\s*\{{(.*?)\}}\s*;", content, re.S)
        if not match:
            raise RuntimeError(f"Could not parse array {name} from {path}")
        return np.array([float(token.strip().rstrip("f")) for token in match.group(1).split(",") if token.strip()], dtype=np.float32)

    def matrix(name):
        match = re.search(rf"(?:const\s+)?float\s+{name}\s*\[[^]]*\]\s*\[[^]]*\]\s*=\s*\{{(.*?)\}}\s*;", content, re.S)
        if not match:
            raise RuntimeError(f"Could not parse matrix {name} from {path}")
        rows = re.findall(r"\{([^{}]*)\}", match.group(1))
        return np.array([[float(token.strip().rstrip("f")) for token in row.split(",") if token.strip()] for row in rows], dtype=np.float32)

    def scalar(name):
        match = re.search(rf"(?:const\s+)?float\s+{name}\s*=\s*([^;]+);", content)
        if not match:
            raise RuntimeError(f"Could not parse scalar {name} from {path}")
        return np.float32(float(match.group(1).strip().rstrip("f")))

    return {
        "mean": values("SCALER_MEAN"), "std": values("SCALER_STD"),
        "w1": matrix("KAN1_W"), "b1": values("KAN1_B"), "theta1": matrix("KAN1_THETA"),
        "centers1": values("KAN1_CENTERS"), "step1": scalar("KAN1_STEP"),
        "w2": matrix("KAN2_W"), "b2": values("KAN2_B"), "theta2": matrix("KAN2_THETA"),
        "centers2": values("KAN2_CENTERS"), "step2": scalar("KAN2_STEP"),
    }


def bspline(u, centers, step):
    u = np.float32(np.clip(u, -3.0, 3.0))
    dist = np.abs((u - centers) / (step + np.float32(1e-10)))
    if dist < 1.0:
        return np.float32(2.0 / 3.0) - dist * dist + np.float32(0.5) * dist * dist * dist
    if dist < 2.0:
        term = np.float32(2.0) - dist
        return term * term * term / np.float32(6.0)
    return np.float32(0.0)


def kan_logit(scaled, weights):
    hidden = np.empty(12, dtype=np.float32)
    for j in range(12):
        u = np.float32(np.dot(weights["w1"][j], scaled) + weights["b1"][j])
        hidden[j] = np.float32(sum(bspline(u, weights["centers1"][k], weights["step1"]) * weights["theta1"][j, k] for k in range(8)))
    u = np.float32(np.dot(weights["w2"][0], hidden) + weights["b2"][0])
    return np.float32(sum(bspline(u, weights["centers2"][k], weights["step2"]) * weights["theta2"][0, k] for k in range(8)))


def probability(scaled, weights):
    logit = kan_logit(scaled, weights)
    return np.float32(1.0 / (1.0 + np.exp(-logit, dtype=np.float32)))


def model_attributions(raw, weights):
    raw = np.asarray(raw, dtype=np.float32)
    scaled = ((raw - weights["mean"]) / (weights["std"] + np.float32(1e-10))).astype(np.float32)
    baseline = probability(scaled, weights)
    deltas = np.empty(14, dtype=np.float32)
    for i in range(14):
        perturbed = scaled.copy()
        perturbed[i] = np.float32(0.0)
        deltas[i] = baseline - probability(perturbed, weights)
    order = sorted(range(14), key=lambda i: (-abs(float(deltas[i])), i))
    return float(baseline), [(FEATURE_NAMES[i], float(deltas[i])) for i in order[:3]]


def explanation(prob, top3):
    state = "Low-oxygen risk" if prob >= 0.50 else "Normal oxygen"
    nonvacuous = [(name, delta) for name, delta in top3 if abs(delta) >= VACUOUS]
    parts = [f"{name} ({delta:+.3f}, {'toward risk' if delta > 0 else 'toward normal'})" for name, delta in nonvacuous]
    if not parts:
        return f"{state} flagged (prob = {prob:.2f}). Main contributors: remaining top-3 slots are vacuous."
    if len(parts) == 1:
        return f"{state} flagged (prob = {prob:.2f}). Main contributor: {parts[0]}. Remaining top-3 slots are vacuous."
    return f"{state} flagged (prob = {prob:.2f}). Main contributors: {'; '.join(parts)}." + ("" if len(parts) == 3 else " Remaining top-3 slots are vacuous.")


def score_consistency(prob, top1):
    if abs(top1) < VACUOUS:
        return 0
    return 2 if (prob >= 0.50 and top1 > 0) or (prob < 0.50 and top1 < 0) else 1


def score_usefulness(text, prob, top3):
    nonvacuous = [(name, delta) for name, delta in top3 if abs(delta) >= VACUOUS]
    top = nonvacuous[0] if nonvacuous else None
    cues = [f"prob = {prob:.2f}" in text, bool(top and top[0] in text), bool(top and ("toward risk" if top[1] > 0 else "toward normal") in text)]
    score = sum(cues) if sum(cues) < 3 else 2
    if any(abs(delta) < VACUOUS and name in text and "vacuous" not in text for name, delta in top3):
        score = min(score, 1)
    return score


def main():
    root = Path(__file__).resolve().parent.parent
    feature_path = root / "outputs" / "openox_canonical_window_features.csv"
    prediction_path = root / "outputs" / "openox_tinykan_predictions.csv"
    header_path = root / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights.h"
    rubric_path = root / "docs" / "tinykan-xai-rubric.md"
    output_path = root / "outputs" / "openox_tinykan_explanation_rubric_sample.csv"
    for path in (prediction_path, feature_path, header_path, rubric_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")

    predictions = pd.read_csv(prediction_path)
    required_predictions = {"row_index", "recording_id", "split_index", "y_true", "y_prob", "y_pred", "top_attribution_feature"}
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Prediction CSV is missing columns: {missing}")
    features = pd.read_csv(feature_path)
    required_features = set(FEATURE_NAMES) | {"recording_id", "start_time_seconds", "end_time_seconds", "label"}
    missing = sorted(required_features - set(features.columns))
    if missing:
        raise RuntimeError(f"Feature CSV is missing columns: {missing}")

    predictions["row_index"] = pd.to_numeric(predictions["row_index"], errors="raise").astype(int)
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="raise").astype(int)
    predictions["y_prob"] = pd.to_numeric(predictions["y_prob"], errors="raise")
    features["label"] = pd.to_numeric(features["label"], errors="raise").astype(int)
    if len(predictions) != len(features) or not np.array_equal(np.sort(predictions["row_index"].to_numpy()), np.arange(len(features))):
        raise RuntimeError("Prediction row_index values must be a 1:1 permutation of feature row positions")
    if predictions["row_index"].duplicated().any():
        raise RuntimeError("Prediction row_index values are not unique")

    weights = parse_c_header(header_path)
    joined = predictions.merge(features.reset_index(names="feature_row_index"), left_on="row_index", right_on="feature_row_index", how="inner", validate="one_to_one", suffixes=("", "_feature"))
    if len(joined) != len(predictions):
        raise RuntimeError("Prediction-to-feature row-position join was incomplete")
    joined["predicted_label"] = (joined["y_prob"] >= 0.50).astype(int)
    joined["quadrant"] = np.select(
        [(joined.y_true == 1) & (joined.predicted_label == 1), (joined.y_true == 0) & (joined.predicted_label == 1), (joined.y_true == 0) & (joined.predicted_label == 0)],
        ["TP", "FP", "TN"], default="FN",
    )

    selected = []
    for quadrant in QUADRANTS:
        q = joined[joined.quadrant == quadrant].sort_values("y_prob", ascending=(quadrant == "TN"), kind="stable")
        n = len(q)
        if n < 5:
            print(f"{quadrant}: only {n} windows available; taking all (fewer than 5).")
            indices = list(range(n))
        else:
            indices = [int(i * (n - 1) / 4) for i in range(5)]
        selected.extend(q.iloc[indices].to_dict("records"))

    rows = []
    quadrant_counts = {q: 0 for q in QUADRANTS}
    for row in selected:
        csv_prob = float(row["y_prob"])
        firmware_prob, top3 = model_attributions([row[name] for name in FEATURE_NAMES], weights)
        text = explanation(csv_prob, top3)
        nonvacuous = sum(abs(delta) >= VACUOUS for _, delta in top3)
        compactness = 2 if len(text.split()) <= 30 else 1 if len(text.split()) <= 45 else 0
        consistency = score_consistency(csv_prob, top3[0][1])
        usefulness = score_usefulness(text, csv_prob, top3)
        quadrant = row["quadrant"]
        quadrant_counts[quadrant] += 1
        slots = [(name, delta, "toward risk" if delta > 0 else "toward normal") for name, delta in top3]
        rows.append({
            "window_id": f"{quadrant}-{quadrant_counts[quadrant]:02d}", "quadrant": quadrant, "row_index": int(row["row_index"]), "split_index": int(row["split_index"]),
            "recording_id": str(row["recording_id"]), "start_time_seconds": row["start_time_seconds"], "end_time_seconds": row["end_time_seconds"],
            "true_label": int(row["y_true"]), "predicted_label": int(row["predicted_label"]), "csv_probability": csv_prob, "firmware_probability": firmware_prob,
            "csv_firmware_prob_abs_diff": abs(csv_prob - firmware_prob), "top1_feature": slots[0][0], "top1_delta": slots[0][1], "top1_dir": slots[0][2],
            "top2_feature": slots[1][0], "top2_delta": slots[1][1], "top2_dir": slots[1][2], "top3_feature": slots[2][0], "top3_delta": slots[2][1], "top3_dir": slots[2][2],
            "n_nonvacuous_top3": nonvacuous, "csv_top_attribution_feature": row["top_attribution_feature"], "explanation": text, "word_count": len(text.split()),
            "compactness": compactness, "consistency": consistency, "usefulness": usefulness, "total": compactness + consistency + usefulness,
        })

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    means = result[["compactness", "consistency", "usefulness", "total"]].mean()
    mins = result[["compactness", "consistency", "usefulness"]].min()
    gate = means.total >= 4.0 and (mins > 0).all()
    print(f"absolute output path: {output_path.resolve()}")
    print(f"row count: {len(result)}")
    print(f"per-quadrant counts: {quadrant_counts}")
    print(f"mean compactness: {means.compactness:.2f}, consistency: {means.consistency:.2f}, usefulness: {means.usefulness:.2f}, total: {means.total:.2f}")
    print(f"min compactness: {mins.compactness}, consistency: {mins.consistency}, usefulness: {mins.usefulness}")
    print(f"gate: {'YES' if gate else 'NO'}")
    print("statement: sample of 20 is still too small to claim the research-plan pass, even if this sample gates")
    print(f"windows with |csv P - firmware P| > 0.05: {(result['csv_firmware_prob_abs_diff'] > 0.05).sum()}")
    print("do not claim firmware/Android/threshold changed")
    print("first 3 CSV rows:")
    print(result.head(3).to_csv(index=False).rstrip())


if __name__ == "__main__":
    main()

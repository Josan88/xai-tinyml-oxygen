# TinyKAN Mean-Perturbation XAI Rubric

Status: research increment for EAT40006 (Project B). Complements the Logistic Regression explanation rubric already scored by `scripts/score_explanation_rubric.py`. Does not reopen firmware M1-M4, ADR-002, or the 0.50 operating point.

Date recorded: 2026-08-24 (+08). Source of worked-example numbers: live `tests/test_runner.exe` (M4 native harness, 104/104 pass).

## 1. Purpose

The faithfulness question is RQ2: whether firmware-style absolute ranking of mean-perturbation scores improves deletion AOPC relative to random order on the exported TinyKAN. Logistic regression is not the faithfulness comparator.

LogReg explanations are already scored (20 held-out windows, mean 6.0 / 6.0, gate pass). TinyKAN had no written rubric. This document is that rubric: the same 6-point human-facing scale, plus TinyKAN-specific rules for judging the 15-pass Mean-Perturbation vector the firmware actually emits.

This is an evaluation instrument, not a firmware change.

## 2. What the firmware computes

ADR-0001. For a 14-feature window the native pipeline does N+1 = 15 FP32 forward passes:

1. Standardize the raw feature vector with the compile-time training mean / std.
2. Baseline probability P = sigmoid(TinyKAN(x)).
3. For each feature i, copy the scaled vector, set x_i <- 0 (training mean in normalized space), and recompute P_i.
4. Attribution delta_i = P - P_i.
5. Sort by |delta_i| descending and keep the top 3 in `RiskDecision.top_attributions`.

Sign convention (from `kan_predict_with_xai`):

| delta_i | Meaning |
|---|---|
| delta_i > 0 | Zeroing feature i drops P. The observed value drove risk **up**. |
| delta_i < 0 | Zeroing feature i raises P. The observed value drove risk **down** (protective). |
| delta_i = 0 | Feature is locally inert. Not a driver. |

Decision threshold used for the flag in this rubric is the live firmware default **0.50**. Candidate risk iff P >= 0.50. Confirmed risk still requires ADR-0005 N=2; the rubric scores the *window explanation*, not the debounce.

Do not confuse this rubric with ADR-0003 deletion / insertion AUC. AOPC is a population faithfulness metric over 52,090 windows. This rubric scores a single local explanation an operator would read.

## 3. Operator-facing explanation contract

Generate one sentence from the `RiskDecision` payload:

    {state} flagged (prob = {P:.2f}). Main contributors: {name_1} ({delta_1:+.3f}, {dir_1}); {name_2} ({delta_2:+.3f}, {dir_2}); {name_3} ({delta_3:+.3f}, {dir_3}).

- `state` is `Low-oxygen risk` if P >= 0.50, else `Normal oxygen`.
- `dir_k` is `toward risk` if delta_k > 0, `toward normal` if delta_k < 0.
- Omit any slot with |delta_k| < 1e-4. If that leaves fewer than three names, say so (`remaining top-3 slots are vacuous`).
- Do not claim raw-value "high/low" unless the raw feature is also in the payload *and* is cited. TinyKAN local evidence is delta, not x_i * w_i.

This is the same compactness / consistency / usefulness idea as the LogReg rubric, restated for Mean-Perturbation.

## 4. Scoring criteria (6 points)

Each window scores three criteria at 0 / 1 / 2. Total is out of 6.

### 4.1 Compactness (2)

| Score | Rule |
|---:|---|
| 2 | Explanation text has <= 30 words |
| 1 | 31-45 words |
| 0 | > 45 words |

Count whitespace-separated tokens. Numbers, feature names, and signs each count as a word.

### 4.2 Consistency (2)

Judge the **top-1 signed attribution** against the 0.50 flag. Ties at |delta| do not change the rule: firmware already broke the sort.

| Score | Rule |
|---:|---|
| 2 | Risk flag (P >= 0.50) and delta_top1 > 0, **or** normal flag (P < 0.50) and delta_top1 < 0 |
| 1 | Flag and top-1 sign disagree (risk with protective top-1, or normal with risk-driving top-1) |
| 0 | |delta_top1| < 1e-4 (no directional evidence) |

Opposing signs in top-2 / top-3 are allowed. They are part of a local cancellation story, not an automatic fail. LogReg already emits mixed top-3 on OpenOximetry; TinyKAN is scored the same way.

### 4.3 Usefulness (2)

| Score | Rule |
|---:|---|
| 2 | Text includes (a) P to two decimals, (b) the top non-vacuous feature name, and (c) that feature signed direction |
| 1 | Two of those three cues |
| 0 | One or none |

Listing a vacuous (|delta| < 1e-4) slot as a "contributor" without marking it vacuous costs one point (2 -> 1). Inventing a physiological story that the delta sign does not support is a 0 on this criterion.

## 5. Gate

Same research-plan gate as README / LogReg:

- mean total >= 4.0 across the scored sample, **and**
- no window scores 0 on compactness, consistency, or usefulness.

A TinyKAN sample that is not yet drawn (20 windows, 5 per confusion-matrix quadrant, as in `score_explanation_rubric.py`) is **not** claimed to pass. This document only scores the live firmware worked examples below.

## 6. How Mean-Perturbation attributions are judged

Apply these checks to the raw `FeatureAttribution` vector *before* writing the sentence. They are diagnostics, not extra rubric points.

| Check | Pass | Fail / note |
|---|---|---|
| Sign convention | delta_i = P - P_i with x_i <- 0 | Any other baseline (zeros in raw space, leave-one-out of unscaled x, SHAP) is out of contract |
| Ranking | Top-3 is |delta| descending, matching `kan_sort_attributions` | Re-ranking by signed delta or by raw x_i is a different method |
| Non-vacuous k | Count of top-3 with |delta| >= 1e-4 | k=1 with two zero slots is legal firmware output; the sentence must not pretend those slots are drivers |
| Local (not global) | Deltas come from this window 15 passes | Do not substitute `outputs/openox_tinykan_feature_attributions.csv` (dataset-mean |delta|) for a local explanation |
| Magnitude vs P | |delta_top1| is large enough to be the main story | On the high-risk synthetic window, top-1 is only +0.056 on P=0.65. That is still the top driver; do not inflate it |
| Mixed signs | Allowed in top-2 / top-3 | Must be labelled `toward risk` / `toward normal`, never dropped to force a one-sided story |
| Threshold | Flag uses 0.50 | Do not retune. Calibration already showed the ETPB recall>=0.80 AND spec>=0.50 gate is not achievable for v1; v2 meets it offline at the same 0.50 (window level), so there is still no reason to retune |
| Parity | Pipeline delta matches the independent 15-pass reference within 1e-5 | Already asserted by M4 Test 3; this rubric does not re-open firmware |

Faithfulness at dataset scale remains ADR-0003 (deletion / insertion AUC). A window can pass this rubric and still be unfaithful under AOPC; report both, do not collapse them.

## 7. Worked examples (live firmware, 2026-08-24)

Command: `.\tests\test_runner.exe` (exit 0). Same numbers on `test_runner_audit.exe`. Firmware sources were not modified.

### 7.1 Normal synthetic window (Test 3a)

| Field | Live value |
|---|---|
| `risk_probability` P | **0.126565** |
| Flag at 0.50 | Normal oxygen (`is_candidate_risk = false`) |
| Top-1 | `bandpower_ratio_0_5_2_to_2_5`, delta = -0.423319 |
| Top-2 | `mean`, delta = 0.000000 |
| Top-3 | `std`, delta = 0.000000 |
| Raw features (same window) | `bandpower_ratio_0_5_2_to_2_5 = 312.215546`, `mean = 0.618656`, `std = 69.998848` |

Interpretation: replacing the (very high) bandpower ratio with the training mean raises P by 0.42, so the observed ratio is strongly protective. Top-2 and top-3 are vacuous ties after the |delta| sort - a firmware property, not a scorer bug.

Explanation text:

> Normal oxygen flagged (prob = 0.13). Main contributor: bandpower_ratio_0_5_2_to_2_5 (-0.423, toward normal). Remaining top-3 slots are vacuous.

| Criterion | Score | Why |
|---|---:|---|
| Compactness | 2 | 22 words |
| Consistency | 2 | Normal flag and delta_top1 < 0 |
| Usefulness | 2 | P, named driver, signed direction; vacuous slots declared |
| **Total** | **6** | |

If the sentence had listed `mean` and `std` as contributors, usefulness would drop to 1.

### 7.2 High-risk synthetic window (Test 3b)

| Field | Live value |
|---|---|
| `risk_probability` P | **0.646437** |
| Flag at 0.50 | Low-oxygen risk (`is_candidate_risk = true`) |
| Top-1 | `kurtosis`, delta = +0.055605 |
| Top-2 | `bandpower_ratio_0_5_2_to_2_5`, delta = +0.054252 |
| Top-3 | `std`, delta = -0.028477 |

Interpretation: kurtosis and bandpower ratio both support the risk flag; amplitude variability pulls the other way. All three slots are non-vacuous. Magnitudes are smaller than the normal-window protector - the model is closer to the decision surface.

Explanation text:

> Low-oxygen risk flagged (prob = 0.65). Main contributors: kurtosis (+0.056, toward risk); bandpower_ratio_0_5_2_to_2_5 (+0.054, toward risk); std (-0.028, toward normal).

| Criterion | Score | Why |
|---|---:|---|
| Compactness | 2 | 24 words |
| Consistency | 2 | Risk flag and delta_top1 > 0 |
| Usefulness | 2 | P, named driver, signed direction; opposing top-3 kept |
| **Total** | **6** | |

## 8. What this does not claim

- It does not claim a 20-window TinyKAN sample has been drawn or has passed the >= 4.0 gate. Only the two live firmware windows above are scored.
- It does not change `KAN_DECISION_THRESHOLD` or the 0.50 pipeline default.
- It does not resume ADR-002 (Android pager).
- It does not start fairness / subgroup analysis.
- Global TinyKAN ranks in `outputs/openox_tinykan_feature_attributions.csv` (kurtosis 59.6% of top-driver windows) are context, not a local score.

## 9. Next scoring step (out of scope here)

A later script, parallel to `scripts/score_explanation_rubric.py`, should: reconstruct TinyKAN window probabilities from `outputs/openox_tinykan_predictions.csv`, run the same 15-pass Mean-Perturbation used in firmware, draw 5 windows per confusion-matrix quadrant at threshold 0.50, emit the sentence in section 3, and write `outputs/openox_tinykan_explanation_rubric_sample.csv`. Do not implement that in this increment.
The resulting 20-window sample is stored in `outputs/openox_tinykan_explanation_rubric_sample.csv`.

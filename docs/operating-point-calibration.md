# Operating-Point Calibration

## Purpose and Gate
This calibration evaluates existing EAT40006 probabilities under the ETPB gate: recall >= 0.80 and specificity >= 0.50. It does not retrain TinyKAN or change firmware.

## Published Window-Level Points
The script evaluates TinyKAN at threshold 0.30 and LogReg at threshold 0.50 using pooled held-out windows. Counts and metrics are recorded in `outputs/openox_operating_point_sweep.csv`.
Notebook 11 published LogReg @0.50 is TN=5615, FP=15892, FN=4093, TP=26490 (recall 0.8662 / specificity 0.2611). The live reconstruction differs by about 222 labels due to sklearn 1.9 drift; the gate conclusion is unchanged because the published specificity is also far below 0.50, and published recall collapses to 0.002 at 0.60.

| Model | Threshold | TN | FP | FN | TP | Recall | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyKAN | 0.30 | 1989 | 19518 | 884 | 29699 | 0.9711 | 0.0925 | 0.6083 |
| LogReg | 0.50 | 5672 | 15835 | 4315 | 26268 | 0.8589 | 0.2637 | 0.6132 |

## Selected Operating Points
| Model | Mode | Threshold | TN | FP | FN | TP | Recall | Specificity | Accuracy | Precision | F1 | NPV | Youden J | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LogReg | confirmed_n2 | 0.51 | 8425 | 13082 | 7238 | 23345 | 0.7633 | 0.3917 | 0.6099 | 0.6409 | 0.6968 | 0.5379 | 0.1551 | closest_infeasible |
| LogReg | window | 0.52 | 10575 | 10932 | 10679 | 19904 | 0.6508 | 0.4917 | 0.5851 | 0.6455 | 0.6481 | 0.4976 | 0.1425 | closest_infeasible |
| TinyKAN | confirmed_n2 | 0.49 | 8841 | 12666 | 7180 | 23403 | 0.7652 | 0.4111 | 0.6190 | 0.6488 | 0.7022 | 0.5518 | 0.1763 | closest_infeasible |
| TinyKAN | window | 0.50 | 8475 | 13032 | 7133 | 23450 | 0.7668 | 0.3941 | 0.6129 | 0.6428 | 0.6993 | 0.5430 | 0.1608 | closest_infeasible |

The ETPB gate is not achievable for any evaluated model/mode in this threshold sweep.

## Firmware Implication
Firmware was not changed in this task. For TinyKAN `confirmed_n2`, the selected threshold is 0.49; therefore the current default 0.50 is closer than the unused 0.30 threshold.

## Limitations
- No signal-quality-index (SQI) feature or gate was evaluated.
- No fairness or subgroup analysis was performed.
- `confirmed_n2` is a proxy for consecutive windows and uses the available recording/window ordering; it is not a full firmware timing simulation.
- LogReg reproduction status: `approximate`.
- Figure generated: `True`.

## Addendum, 28 Oct 2026 — scope
The sweep above evaluated **v1** probabilities only (frozen 14-IR TinyKAN `(14, 12, 1)` and LogReg, notebook-10 labels). "Not achievable" applies to v1. The v2 header-aligned model (16 inputs, ADR-0006) meets the ETPB gate at the same frozen 0.50, window level, out-of-fold: TinyKAN recall 0.893 (95% CI 0.859–0.922) / specificity 0.868 (0.838–0.894) recording-grouped (154 recordings), and 0.899 (0.870–0.925) / 0.864 (0.837–0.888) patient-grouped (87 patients) (`outputs/analysis-lamaakal-style-v1/c_bootstrap_ci.csv`). All 5 folds pass in both groupings; the lowest recording-grouped fold recall is 0.874 and the lowest fold specificity 0.826 (`c_per_fold.csv`). Every swept threshold 0.30–0.70 passes for v2 TinyKAN (`b_operating_point_sweep.csv`). v2 LogReg on the same 16 inputs passes at 0.50 (0.821 / 0.870). Limits: window level, not N = 2 confirmed; offline OpenOximetry header-aligned ScalcO2 labels (better than as-coded, not exact); v2 was built after v1 failed (same recipe and 0.50, but post hoc); live recall is unmeasured because no live window had low oxygen. The operating point stays 0.50.

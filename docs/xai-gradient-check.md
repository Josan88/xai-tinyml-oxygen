# Gradient×input vs the shipped mean-perturbation explainer (host-only check)

24 October 2026. This is a sanity check, not a new method. The on-device explainer stays mean-perturbation (training-mean replacement, N + 1 passes, ranked by |Δ|). There was no board I/O and no firmware, header or existing output was changed. The write-once directories were hashed before and after the run and are unchanged.

Code: `scripts/compare_gradient_vs_perturbation.py`. Tests: `tests/test_gradient_vs_perturbation.py`. Output: `outputs/xai-gradient-check-v1/` (`agreement_summary.csv`, `global_top1_share.csv`, `deletion_aopc.csv`, `per_window_v{1,2}.csv.gz`, `summary.json`).

**Why the comparison is like-for-like.** Standardized features have training mean 0, so both explainers use the same baseline. Mean-perturbation gives Δᵢ = P(x) − P(x with xᵢ = 0). Gradient×input gives gᵢ = xᵢ·∂P/∂xᵢ, which is the first-order Taylor estimate of Δᵢ. The two disagree only where the learned splines bend between xᵢ and 0, or where the clamp at u = ±3 is active. The gradient is computed analytically through the clamped cubic B-splines of `kan_inference.h`. It matches central finite differences to 2e-10 (v1) and 2e-9 (v2) on 2,000 windows each.

**Reproduction.** On v1, P and Δ match the write-once `openox-aopc-v1` arrays to 1e-16 and 3e-16. The recomputed deletion AOPC matches the cited 0.0443 / 0.0283 exactly. On v2, it matches the `analysis-lamaakal-style-v1` new run (−0.3092 / −0.1612).

## Results (in-sample, shipped headers on their own training tables)

| | v1 all | v1 candidates | v2 all | v2 candidates |
|---|---:|---:|---:|---:|
| Windows | 52,090 | 34,127 | 48,662 | 28,661 |
| Gradient all-zero, Δ not all-zero | 336 | 0 | 13,118 | **11,381 (39.7%)** |
| Top-1 agreement (both non-degenerate) | 0.923 | 0.942 | 0.688 | 0.379 |
| Top-3 overlap | 0.922 | 0.939 | 0.712 | 0.631 |
| Median Spearman of \|values\| | 0.991 | 0.991 | 0.941 | 0.921 |
| Deletion AOPC: mean-perturbation | 0.0443 | 0.0958 | −0.3092 | −0.0085 |
| Deletion AOPC: gradient×input | 0.0416 | 0.0912 | −0.2983 | −0.0265 |
| Gradient − mean-pert. (95% recording-bootstrap CI) | −0.0027 (−0.0041, −0.0015) | −0.0047 (−0.0057, −0.0039) | +0.0109 (−0.0016, +0.0276) | −0.0179 (−0.0279, −0.0102) |

**v1 (reported model).** The two explainers almost agree. The top-1 feature is the same in 92% of windows, and kurtosis is top-1 in 55% (mean-perturbation) and 53% (gradient) of windows. On deletion, mean-perturbation is slightly but significantly better. The 336 windows where the gradient is all zero are all non-candidates (P 0.13–0.41): every hidden pre-activation is clamped, so ∂P/∂x = 0, but replacing a feature with its mean still moves P.

**v2 (comparison).** The gradient is blind on 40% of candidate windows. These are the saturated alerts at P ≈ 0.89–0.905. There the output spline is clamped, ∂P/∂x is exactly 0, and gradient×input returns 14–16 zeros. Mean-perturbation still names a driver, `ratio_of_ratios` in 11,653 of the 13,118 blind windows. Where both explainers are non-degenerate, gradient×input ranks `ratio_of_ratios` top-1 in 77% of windows against 59% for mean-perturbation. It almost never ranks `ir_perfusion_index` top-1 (0.05% against 6.5%). Removing a feature moves u far enough that the local slope misses the effect. On candidates, mean-perturbation deletes risk better (difference −0.018, CI excludes 0). The non-candidate v2 difference favours the gradient (+0.052), but AOPC is negative there for every ranking: with a fixed P(risk) target, deleting features from a low-risk window raises P. It is not evidence that the gradient explains low-risk windows better.

## Limits

- In-sample, like the write-once AOPC run.
- Deletion uses mean replacement, the same operation that defines mean-perturbation. The shipped ranking therefore has a built-in first-step advantage on deletion. The agreement and blind-window counts do not depend on that.
- Agreement is between two explanations of one model. Neither is ground truth and neither is a clinical cause (ADR-0003).

## Paper sentence (methods, "methods not selected")

> Gradient×input was evaluated on the host as an alternative explainer with the same training-mean baseline. On v1 it agreed with mean-perturbation on the top-ranked feature in 92% of windows. On v2, the clamped output spline made the gradient exactly zero on 40% of alert windows (11,381 of 28,661), where it returns no explanation. Mean-perturbation was kept because it needs no derivative code on the microcontroller and still explains saturated outputs.

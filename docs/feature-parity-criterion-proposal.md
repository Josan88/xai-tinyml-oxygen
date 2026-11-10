# Feature parity criterion

Status: **adopted 2026-09-10**. This local-scale rule replaces the uniform absolute raw-feature tolerance of 1e-4 as the numerical feature-parity requirement. Adoption itself did not change firmware, weights, threshold 0.50, ADR-0005 N=2, the five-second context, or existing synthetic CTest gates. Original absolute errors remain visible in every report. The baseline failed 66 of 2,557 development-audit windows. The subsequently authorized FP32 repair passes these windows and all 2,464 windows locked before repair; its authorized 21-decision synthetic Nano replay and recorded development-target qualification also pass, including all 14 signed deltas. See [repair and validation evidence](fp32-filter-repair-validation.md). Parity is **not** complete; runtime latency/live evidence and the separate context boundary remain.

## Recommendation

Use a feature's local signal scale when measuring numerical agreement. Keep probability, all 14 signed Mean-Perturbation deltas, and decision behavior as separate required checks. Preserve the original absolute errors in every report.

The implementation at adoption did **not pass** this criterion. On the already examined 2,557-window audit, 66 windows failed at least one feature/probability/attribution check. That audit remains development evidence, not an independent validation set.

## Why the existing and first proposed rules are insufficient

The expanded audit's nearest representable FP32 energy value alone exceeds an absolute error of 1e-4 in **2,550 of 2,557 windows**. An implementation returning the best possible final FP32 representation would therefore still fail those comparisons. That does not explain away all observed error: intermediate filtering and feature arithmetic add further differences.

The previous diagnostic proposal used training standard deviations as the absolute budget scale. Those scales are useful for model sensitivity but do not directly describe the numerical scale of a particular PPG window. Large-amplitude recordings and nearly zero means can dominate or distort that comparison. The first proposal remains recorded as `diagnostic_only_not_adopted` in the expanded audit.

## Adopted required checks

### 1. Compare the same computation

Use identical FP32 input samples, the same 50 Hz sequence, target 500-sample window, available 250-sample history, 250-sample future context, and 250-sample stride. Use the canonical SciPy feature definitions and the existing exported model/scaler constants. Reject malformed or nonfinite reference and actual values; do not count an uncomparable window as a pass.

The numerical reference uses FP32-quantized input converted to FP64 for SciPy. Quantization of the original resampled input is reported separately. Full-recording versus finite-context filtering is also reported separately; it must not be absorbed into a numerical tolerance.

### 2. Feature values

For every continuous feature, require:

`abs(actual - reference) <= 1e-4 * local_scale + 8 * abs(spacing(float32(reference)))`

Choose `local_scale` using the reference window:

- Mean: RMS of the filtered target window. Its own mean can be near zero while the signal is large.
- Minimum and maximum: the larger of the feature's absolute value and target-window RMS.
- Standard deviation, value range, RMS, energy, derivative standard deviation, and each bandpower: the feature's absolute value.
- Skewness, kurtosis, and bandpower ratio: the larger of 1 and the feature's absolute value. These dimensionless features need a finite absolute budget near zero.

Dominant frequency must match the selected FFT bin exactly; no frequency tolerance is applied.

The relative budget is 0.01 percent, with a 1e-4 near-zero budget for dimensionless features. The eight-spacing term is an explicit output-rounding allowance. Neither is a mathematical bound on all intermediate FP32 errors or a clinical acceptance threshold. They are adopted engineering budgets. Their values are frozen and must not be raised in response to failures.

Unlike a fixed tolerance in optical units, the amplitude and power budgets scale consistently when signal units change. The review checks this for a factor-of-two amplitude rescaling, including the corresponding factor-of-four energy/bandpower rescaling.

### 3. Model, attribution, and decisions

- Absolute probability error must be at most **1e-4**. This retains the tighter existing synthetic regression budget; the earlier 0.01 comparison ceiling remains a separately reported historical check.
- Each of the **14 signed attribution deltas** must differ by at most **1e-5**. Testing only transmitted top-three deltas can hide other errors.
- Candidate classification at **0.50** must agree exactly. A small probability error does not excuse a threshold crossing.
- Full streaming tests must check window indices, target start/end, delay, reset, consecutive count, and N=2 confirmed state. Sparse sampled windows cannot establish consecutive-state agreement across a recording. Existing 21-decision synthetic board replay establishes only its documented scope.

For attribution ranking, first require the delta-value check. Require the order of any two reference attribution magnitudes whose gap exceeds **2e-5**, including comparisons across the top-three selection boundary. Gaps at or below 2e-5 are reported as ambiguous ordering. This follows from the two individual 1e-5 error budgets; it is not an exemption for a large delta error. Every observed top-three set/order difference remains visible in the report.

The previously investigated order swap has an unchanged top-three set, a reference magnitude gap about 4.56e-6, and maximum delta error about 6.59e-6. It is consistent with ambiguous ordering under this rule. No sort implementation change is proposed.

## Screen of the implementation at adoption

On the existing 2,557 complete windows from all 162 recordings:

- Feature failures: **19 windows**. Per-feature counts overlap: std 2, minimum 2, maximum 3, range 2, RMS 2, skewness 6, kurtosis 5, energy 3, lower-band power 8, upper-band power 7, ratio 17. Mean, derivative standard deviation, and dominant frequency have zero failures.
- Probability above 1e-4: **1 window**.
- Any signed attribution above 1e-5: **52 windows**.
- Candidate disagreements: **0**.
- Union of feature/probability/attribution failures: **66 windows**. These counts must not be added without accounting for overlap.

These results show that the proposal does not merely convert the present implementation into a pass. The broader audit is already used for development; passing it later would require additional evidence on windows not used to choose the criterion or repair.

## What the further investigation established

`scripts/review_feature_precision.py` selected the union of worst relative/standardized feature errors and worst probability, attribution, context, and ranking cases: **11 windows from 9 recordings**. It verified source and executable hashes, reconstructed each selected native feature vector exactly, and compared several stages.

Using the firmware's same rounded filter coefficients and initial states with FP64 arithmetic as an offline control reduced the selected-case maximum probability error from **1.17316e-4** to **6.26999e-6**, and maximum attribution error from **1.53927e-4** to **6.67016e-6**. Keeping the native filtered samples and calculating subsequent features/inference in Python retained nearly all the worst error: probability **1.17014e-4**, attribution **1.53685e-4**.

This localizes the dominant residual in those selected cases to filtering arithmetic. It does not prove every remaining feature failure has that cause. The worst probability case has a raw context range about **3,040 times** the filtered target-window RMS; small residual errors matter more when the target signal is small compared with the surrounding context. No sample was dropped or relabeled on that basis.

That earlier FP64 control was diagnostic only; its 11-case review made no firmware or model changes. The later authorized all-66 diagnosis and native FP32 repair are recorded separately in [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md).

## Context limitation remains separate

The current five-second-context pipeline has one sampled probability difference of about **0.01238** from full-recording filtering. Both candidates remain below 0.50. This proposal does not waive or reclassify that difference. See [training-context-decision.md](training-context-decision.md) for the reconstructed case and the scope of the already approved five-second delay.

## Decision and next action

**Approved.** The uniform absolute feature requirement is replaced by the local-scale feature rule and the separate probability, attribution, ranking, and streaming checks above.

Binding conditions of this adoption:

1. Budgets are frozen (`1e-4` local scale, `8` output spacings, probability `1e-4`, attribution `1e-5`, ranking gap `2e-5`, exact 0.50 candidate agreement, exact dominant-frequency bin). Do not raise them in response to the remaining 66 failures.
2. Keep original absolute-error results visible in every report.
3. The 2,557-window audit remains development evidence. Lock any independent validation windows **before** repair; later passing this same sample is not qualification.
4. Full-recording versus five-second-context filtering stays a separate check. Do not absorb it into these numerical budgets.
5. Existing synthetic C++ internal-consistency 1e-4 CTest gates stay unless a later explicit decision changes them. They compare a different pair (pipeline vs C++ reference), not SciPy vs native.
6. Filter-arithmetic localization applies to the 11 selected worst cases, not automatically to all 66 remaining windows. Bandpower-ratio and attribution failures still need per-window diagnosis.
7. If native FP32 cannot close a frozen check, report the residual and request a new decision. Do not fold that residual into this criterion.
8. Do not label parity complete on adoption. Board qualification follows only within separately authorized hardware scope.

Completed under this authorization: implemented frozen checks; locked 2,464 unseen validation windows before repair; diagnosed all 66 baseline failures; repaired native FP32 arithmetic; passed the complete development set; froze the candidate; then passed the locked validation set. Original absolute 1e-4 feature comparisons still fail on 2,556 development and all 2,464 validation windows, with every absolute error retained. Detailed results and remaining board/context boundaries are in [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md). Any later change to the budgets requires another explicit decision.

## Evidence and reproduction

- Expanded audit: `outputs/parity/feature_audit/20260910T025019351296Z/`.
- Precision review: `outputs/parity/precision_review/20260910T091743571035Z/report.json`.
- Existing review and context details: [probability-parity-investigation.md](probability-parity-investigation.md) and [training-context-decision.md](training-context-decision.md).

```powershell
python scripts/review_feature_precision.py --audit outputs/parity/feature_audit/20260910T025019351296Z --audit-probe build/parity/audit-release/feature_audit_probe.exe --probability-probe build/parity/audit-release/probability_probe.exe
```

This historical review command requires the original source/binary hashes; it is not the repaired implementation's qualification command. The script writes a fresh ignored output directory, checks relevant source/binary/data hashes against the prior audit, and does not modify that audit. Recorded identifiers and per-case feature values stay in local ignored outputs. There are no recorded waveforms in the new source file.

Validation for this review: high-precision filtering agreed with independent SciPy on synthetic 750- and 1,000-sample contexts; all 11 native feature vectors reconstructed exactly; near-zero-mean, exact-frequency, and unit-rescaling checks passed. Firmware and existing test gates were unchanged, so the previously recorded 8/8 CTest result was not rerun or presented as a new qualification result.

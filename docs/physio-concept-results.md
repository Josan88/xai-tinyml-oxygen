# Physiological inputs and a concept bottleneck: results (host only, 30 Oct 2026)

Study pre-registered in `docs/physio-concept-preregistration.md` before any feature was computed. Numbers here are copied from `outputs/physio-concept-v1/` (`README.md`, `summary.json`, stage files). Host only, laptop, no board. Results are for the final report; the IEEE paper is not edited.

Reader-facing names: Baseline model (repo v1), Header-aligned model (repo v2, 16 inputs). "P5" is the new five-input model: beat R (`r_beat`), beat perfusion index (`pi_beat`), beat-derived pulse rate (`hr_bpm`), template SQI and beat-interval max/min ratio.

## What was asked

The shipped explanations name waveform statistics (kurtosis on the Baseline model), so a bar cannot be checked against physiology. The study asked whether inputs with a physical meaning (a) keep prediction, (b) give explanations whose direction can be checked against physics, and (c) whether a concept bottleneck (inputs → estimated saturation → risk) gives a concept-level explanation, and at what cost.

## Results against the pre-registered rules

| Question | Result | Pre-registered reading |
|---|---|---|
| (a) Prediction | P5 TinyKAN AUC 0.917 [0.896, 0.935] vs Header-aligned 0.908; paired difference +0.009 [+0.003, +0.017] recording-grouped, +0.008 [+0.000, +0.015] patient-grouped. 189 parameters vs 321. Six seeds 0.915 ± 0.002. | Kept; higher on this table, recording-grouped |
| (b) Direction, `r_beat` (primary) | Sign agreement 0.702 at the device reference; ICE non-decreasing share 0.000 | Sign mixed; ICE contradicts |
| (b) Direction, `hr_bpm` (secondary, confounded) | 0.748; ICE 0.228 | Mixed; contradicts |
| (b) Direction, Header-aligned `ratio_of_ratios` (reference) | 0.756; ICE 0.000 | Mixed; contradicts |
| (c) Concept sign | Coefficient on estimated saturation −2.04 to −2.34 in all 12 head fits (5 folds + pooled, both groupings) | Holds |

## What the direction result means

The direction checks fail on the pre-registered reading, but not because R points the wrong way. Two things are going on, and the post hoc ICE descriptors (labelled post hoc in the outputs) show both:

1. **The model learned a step on R.** In every ICE curve, P jumps from about 0.07 to about 0.92 as R crosses roughly its training mean, then falls about 0.02 to 0.89 by z = +3. P at z = +3 is at or above P at z = −3 on 99.996% of windows. The strict "non-decreasing" test fails on that 0.02 dip. The dip is the output spline overshooting before the [−3, 3] clamp (peak P 0.921 at u = 2.47, 0.893 at the clamp). The Header-aligned model's `ratio_of_ratios` curves have the same shape.
2. **The device reference sits on the step.** The explainer replaces an input with its training mean. The training-mean R (1.007) is where P peaks, so for a flagged window with a high R, "set R to the mean" can raise P slightly. The delta is then negative, and the bar says R lowered the score. That is why only 47% of P5 candidates agree in sign, and why the top bar opposes the flag on 77% of P5 candidates (61% on the Header-aligned model). Against a "normal" reference (the label-0 median, R = 0.64; pre-registered as a host-only sensitivity check), R agrees in sign on 98.9% of windows, 100% of candidates, and only 0.3% of P5 candidates have an opposing top bar.

The physiological input did what the study wanted: it made the explanation checkable. The check then found a method artefact, the training-mean reference plus the output-spline overshoot. That artefact also affects the shipped Header-aligned explanation. Kurtosis on the Baseline model could not have exposed it, because kurtosis has no expected sign.

`hr_bpm` is weak (univariate AUC 0.665), its ICE curves are mostly flat, and its sign agreement stays mixed under both references. That fits its pre-registered role as a confounded context input.

## Concept bottleneck

- **The concept.** Estimated saturation `s_hat` comes from a 189-parameter KAN regressor on P5. MAE 3.8 points, RMSE 5.8, Bland-Altman limits −11.2 to +11.5, Pearson r 0.76 (recording-grouped; patient-grouped is the same within 0.1). Predictions never go much below about 75, so RMSE is worst below 80 (6.8). A linear ridge on the same inputs only reaches r 0.17, so the R-to-saturation mapping here is non-linear.
- **The task.** The head (logistic regression on `s_hat` and the four computed concepts) reaches AUC 0.906 [0.884, 0.924]. `s_hat` alone gives 0.907, and the pure bottleneck flag `s_hat < 90` also gives 0.907. That is level with the Header-aligned model (difference −0.002 [−0.013, +0.008]) and slightly below P5 TinyKAN recording-grouped (−0.011 [−0.024, −0.001]; patient-grouped −0.004 [−0.013, +0.006]). The four computed concepts on their own give 0.686, and adding any one of them to `s_hat` changes AUC by at most 0.002.
- **The explanation.** It is exact, because the head is linear. `s_hat` is the top concept on 95–99% of windows. The top contribution opposes the head decision on 3.9% of windows (7.1% of flagged ones), against 77% for P5 TinyKAN's device explanation of flagged windows. So the sentence a reader gets is "estimated saturation was low, which raised the risk", and it is direction-correct on about 95% of windows.
- **The catch.** The label is saturation < 90, so the bottleneck is close to "estimate saturation, then threshold it". Its explanation names an oxygen estimate with ±11-point limits of agreement. That estimate is uncalibrated for the MAX30102, and transfer from OpenOximetry is untested. It must not be shown as an oxygen reading. Doing so would contradict the device's "not an oxygen reading" wording and would invite comparison with pulse-oximeter accuracy, which this cannot meet.

## Other findings

- The beat features agree with the existing window features: Spearman `r_beat` vs `ratio_of_ratios` 0.989, and `pi_beat` vs `ir_perfusion_index` 0.997. HR is within 40–180 bpm on 99.95% of valid windows. 76 of 48,662 windows have fewer than 4 beats.
- The two SQIs rate the 2,006 ADR-0007 artefact rows (replayed 0.5 s sensor buffers) as near-perfect beats (AUC 0.002 in the quality direction). They do not detect that artefact, because a replayed buffer is periodic. Removing the SQIs (P3) costs 0.011 AUC [0.002, 0.021], so they carry some signal, but they are not artefact detectors.
- Deletion AOPC on P5 has the same pattern as the Header-aligned model: firmware-absolute is worse than random on all windows (−0.340 vs −0.207) and on candidates (−0.040 vs −0.025). With five inputs and a saturating output, the deletion curve is dominated by the step on R.

## Limits

Host only: no board time, flash, RAM or parity. Op counts in `scripts/physio_features.py` are estimates. The detector differs from the firmware's display-HR routine. Recording-grouped folds let a patient appear in train and test; patient-grouped results are reported beside them and agree. AOPC and sign checks are in-sample on the all-window model and functionally grounded only. One dataset (OpenOximetry); the label is ScalcO2 from end-tidal O2, not pulse-oximeter SpO2.

## Next steps (not started; each needs a decision from Jason)

1. **Report text.** A Results subsection and a Discussion paragraph, queued in the vault note `Final report - queued results` (section 5).
2. **Explainer reference point.** The data argue for a normal-reference baseline (for example, the label-0 median) instead of the training mean, or for clipping the output spline's overshoot. Either one changes the device explanation. It would be a firmware change with a parity re-run (a named board job), not a host tweak.
3. **On-device P5.** It needs a C port of the beat detector and the five features, a 189-parameter header, host parity, and board timing and RAM. That is a firmware and board job.
4. **Showing `s_hat`.** Any display of estimated saturation needs a wording decision first, because it is an uncalibrated oxygen estimate.

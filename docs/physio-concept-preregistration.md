# Pre-registration: physiological-input TinyKAN and concept-bottleneck prototype (30 Oct 2026)

Written and committed before any feature in this study is computed on the OpenOximetry windows, and before any model in this study is trained or explained. Host only, laptop, no board. Results go to the final report only. The IEEE paper is not edited. Named by Jason on 30 Oct 2026.

## Question

The shipped explanations name generic waveform statistics. On v1, `kurtosis` is top-1 on 55.6% of positive and 54.4% of negative windows. On v2, `ratio_of_ratios` is top-1 on 90% of negatives. So an explanation bar cannot be checked against physiology. This study asks three things:

- (a) Does a TinyKAN whose inputs have a physical meaning keep prediction?
- (b) Can the direction of its explanations be checked against physics?
- (c) Does a concept-bottleneck design (inputs → human-named concepts → risk) give a concept-level explanation, and at what cost?

The label is unchanged: window-median ScalcO2 < 90. ScalcO2 is a saturation calculated from end-tidal O2. It is not a pulse-oximeter SpO2, although the CSV column is named `median_spo2`. The device output stays a risk flag. The learned concept is called "estimated ScalcO2". It is an internal host quantity, not an oxygen reading.

## Already seen before this pre-registration (scratch probe, unreviewed)

Before this job, a quick, unreviewed scratch probe was run on the same v2 folds. It used a simple `find_peaks` detector, not repo code, on beat-valid windows (48,628 of 48,662). It produced these numbers and nothing else:

- Out-of-fold AUC:

  | Inputs | LogReg | HistGB |
  |---|---|---|
  | v2-16 | 0.869 | 0.932 |
  | 5 physiological inputs | 0.884 | 0.933 |
  | 16+5 | 0.889 | 0.938 |

- Univariate AUC: beat R 0.876, window `ratio_of_ratios` 0.869, beat HR 0.655, beat PI 0.478, template SQI 0.535, max/min beat-gap ratio 0.362.
- Spearman correlation of beat R with `ratio_of_ratios`: 0.989.
- LogReg on the 5 inputs had positive coefficients for R and HR.

No TinyKAN, XAI or concept-bottleneck result was seen. The detector below is not the scratch probe's detector. Its two free choices (prominence 1.0 × window std, and the polarity rule) were fixed on synthetic signals only (`build/physio/synth_probe*.py`, scratch, not committed). In those signals a clean pulse's systolic prominence was 3.0–3.8 × std. Dicrotic bumps stayed below 1 × std except at 40 bpm with a large notch.

## Data and folds (frozen inputs; SHA-256 recorded in summary.json)

- Rows and labels: `outputs/openox-v2/window_features.csv`. 48,662 windows, 154 recordings, 87 patients, prevalence 0.592.
- Waveforms: `outputs/openox-raw-windows-v1/v2_ir_red_windows.npz`. Its SHA-256 is checked against `SHA256SUMS` first, and the run stops if it differs. Row i of the npz is row i of the table. Its IR and red are filtered (Butterworth 4, 0.5–8 Hz, sosfiltfilt, 500 samples at 50 Hz). `ir_dc_mean` and `red_dc_mean` are the means of the raw (unfiltered) resampled windows.
- Recording-grouped folds: the `fold` column of `predictions_b_v2_tinykan16.csv`. Patient-grouped folds: the `fold` column of `predictions_b_v2_tinykan16_patient.csv`. Both are aligned by `row_index`, and both alignment and no-group-spans-folds are asserted. "Recording-grouped" means a patient can appear in both folds. It is the primary grouping because it is the grouping of every reported v2 number. Patient-grouped is reported beside it.
- Artefact audit: `outputs/openox-v3/artefact_rows.csv` (ADR-0007), used for SQI validation only.

## Stage 1: feature definitions (`scripts/physio_features.py`)

Pure NumPy/SciPy and deterministic. Every step is a loop a microcontroller could run, and nothing is learned. Op counts are estimates. Board cost is not measured. Per 500-sample window, from filtered IR `x`, filtered red `r`, and the DC means `dc_ir`, `dc_red`:

1. **Flat or unusable window.** The window is beat-invalid if `std(x) == 0` or `dc_ir <= 0` or `dc_red <= 0`.
2. **Polarity.** Let `d = diff(x)`. Orientation `s = +1` if `P99(d) >= -P1(d)`, else `-1` (linear-interpolated percentiles). The steepest slope of a pulse is the systolic upstroke, so `s·x` has systolic peaks pointing up. The chosen `s` is recorded per window. Inverting `x` and `r` together flips `s` and leaves every feature unchanged, except where `P99(d) == -P1(d)` exactly.
3. **Beats.** `scipy.signal.find_peaks(s·x, distance=17, prominence=1.0·std(x))`. 17 samples is 0.34 s at 50 Hz, the firmware's `HR_MIN_PEAK_GAP`, so the maximum rate is about 176 bpm. Peaks are refined to sub-sample positions by a 3-point parabola for the interval computation only. `n_beats` is the number of detected peaks.
4. **Beat-valid** iff step 1 passes and `n_beats >= 4` (at least 3 intervals, the firmware's `HR_MIN_INTERVALS`). If a window is beat-invalid, all five inputs below are missing.
5. **IBI** = differences of the refined peak positions / 50 Hz. **`hr_bpm`** = 60 / median(IBI). **`sqi_ibi_ratio`** = max(IBI) / min(IBI). The firmware's display-HR rule rejects values above 1.4. Here the ratio is reported as an input, not used as a rejection rule.
6. **Beat segments** are the samples `[p_k, p_{k+1}]` (inclusive) between consecutive integer IR peaks. For each segment, `AC_ir = max(x) − min(x)` and `AC_red = max(r) − min(r)` over the same samples. A segment with `AC_ir <= 0` is skipped.
   - **`r_beat`** = median over segments of `(AC_red / dc_red) / (AC_ir / dc_ir)`.
   - **`pi_beat`** = median over segments of `AC_ir / dc_ir`.
7. **`sqi_template`.** Each segment is linearly resampled to the median segment length L. The mean beat is the average of the resampled segments. The SQI is the mean over segments of the Pearson r between each resampled segment and the mean beat.

The firmware's `estimateHeartRateBpm()` differs from this detector. It runs on the raw MAX30102 trace (a 1.5 s baseline minus a 180 ms smooth, which inverts the trace) over an 8 s window. It thresholds peaks at 60% of the range rather than by prominence, uses the mean gap rather than the median, rejects HR when the max/min gap exceeds 1.4, and limits HR to 40–180 bpm. The firmware is not changed.

**A limitation fixed in advance.** On band-limited Gaussian noise, a std-relative prominence still finds 17–19 peaks per window. So noise is not beat-invalid under the brief's "< 4 beats" rule. On synthetic signals, noise gives `sqi_template` ≈ 0.61–0.70 (peak-to-peak segmentation imposes a shared U-shape) and `sqi_ibi_ratio` ≈ 2.3–2.7. A clean pulse gives about 0.97–1.0 and about 1.0. The unit test therefore asserts that a flat line is beat-invalid. For noise, it asserts that noise is separated by the SQIs, and it documents that noise is not beat-invalid. No extra validity gate is added. That would change the input definition the scratch probe and this study share.

**Imputation.** For a beat-invalid window, each of the five inputs is replaced with the training-fold mean of the beat-valid training rows. That value is 0 after the per-fold StandardScaler. The counts and label prevalence of invalid windows are reported. Every Stage 2 metric is also reported restricted to beat-valid windows (same out-of-fold predictions, subset scored).

**Stage 1 checks:**
- Share of `hr_bpm` within 40–180.
- Spearman correlation of `hr_bpm` with 60 × `dominant_frequency_hz`.
- Spearman correlation of `r_beat` with `ratio_of_ratios`.
- `ratio_of_ratios` and `ir_perfusion_index` recomputed from the npz (`std(filtered)/dc`) against the CSV: max abs and max rel error, to prove row alignment.
- AUC of each SQI for flagging the ADR-0007 artefact rows, plus the beat-invalid share inside those rows.
- Chosen polarity per window and per recording.

**Unit tests** (`tests/test_physio_features.py`):
- Synthetic asymmetric pulses at 60, 90 and 120 bpm with known red/IR AC/DC: HR within 1 bpm, R within 2%.
- Flat line: beat-invalid.
- Noise: separated by the SQIs, documented as not beat-invalid.
- Joint inversion of IR and red: same HR and R.

## Stage 2: models (analysis models only)

Feature sets:
- **P5** = `r_beat, pi_beat, hr_bpm, sqi_template, sqi_ibi_ratio` (main).
- **P3** = `r_beat, pi_beat, hr_bpm` (SQI ablation).
- **V16+P5** = the 16 v2 inputs plus P5 (21).

Learners, on the same folds with a StandardScaler fitted per training fold after imputation:
- **TinyKAN.** The v2 recipe: hidden 12, 8 knots, [−3, 3], cubic, Adam lr 0.01, weight decay 1e-4, 100 epochs, pos_weight BCE, smoothness 1e-5, grad clip 10, full batch.
- **LogReg.** `BASELINE_SETTINGS["logreg"]`.
- **HistGradientBoosting.** `BASELINE_SETTINGS["hist_gb"]`, the ceiling.

Every TinyKAN job runs in a freshly spawned process, as `analyze_lamaakal_style.run_jobs` does.

Runs:
- Recording-grouped: P5 / P3 / V16+P5 × TinyKAN (seed 42), LogReg, HistGB.
- P5 TinyKAN at seeds 42, 1, 2, 3, 4, 5 (mean ± sd).
- Patient-grouped: P5 × TinyKAN (seed 42), LogReg, HistGB.
- Comparators: existing v2-16 out-of-fold predictions (TinyKAN recording and patient, LogReg recording). v2-16 HistGB (recording) and v2-16 LogReg (patient) are retrained here because no stored out-of-fold vectors exist.

**Reproduction check first** (the run stops if it fails): v2-16 TinyKAN seed 42 is retrained through `train_tinykan_v2.out_of_fold` and compared with `predictions_b_v2_tinykan16.csv` (AUC 0.90753). If the probabilities are not bitwise identical, the AUC must be within 0.002, and that is said.

Metrics at 0.50:
- AUC, PR-AUC, recall, specificity, balanced accuracy, candidate rate.
- Per class: P median and IQR, and the share with P ≥ 0.50.
- Recording-bootstrap 95% CI for AUC: 2,000 resamples, `lamaakal_metrics.group_bootstrap`. The seed follows the section C convention (`BOOTSTRAP_SEED` 20261024 + comparison index): index 1 (20261025) for all v2 recording-grouped models and index 2 (20261026) for all patient-grouped models. Models inside a comparison are paired.
- Parameter counts, and forwards per explained decision (n + 1).

## Stage 3: explanation checks (P5 and the shipped v2 header)

Models:
- The P5 all-window final model. Same recipe, seed 42, all 48,662 windows, imputation with the all-window beat-valid mean. Its weights are saved as npz/json under `outputs/physio-concept-v1/` only.
- The shipped v2 header model, loaded with `train_tinykan_v2.parse_header` (host replica `lamaakal_models.HeaderKAN`) and not retrained.

The explainer is the device method. Each standardized input is set to 0 (the training mean), `delta = P − P_perturbed`, and the ranking is firmware absolute (descending |delta|, ties by feature index).

1. **AOPC.** Deletion and insertion through `analyze_lamaakal_style.aopc_run` (seed 20260911, 5 random repeats). Subsets: all, candidate, non-candidate, label_1, label_0. Rankings: firmware_absolute, random_mean, signed_risk. Same-code check first: the shipped v2 header must reproduce `d_aopc_v2_new_run.csv` (firmware_absolute −0.3092 vs random −0.1612, all windows), or the run stops.
2. **Class separation.**
   - Top-1 and top-3 share per feature by class.
   - Total variation distance between the label-1 and label-0 top-1 distributions, for P5 and v2, and for v1 read from `d_global_attribution.csv`.
   - P5 has 5 inputs and v2 has 16, so top-3 covers 60% vs 19% of the inputs. TVD is reported, not thresholded.
3. **Degeneracy.**
   - The `d_explanation_degeneracy.csv` columns.
   - The dashboard withhold rule as coded in `explanationGate` (replay mode). The rule withholds as "zero" if lead |delta| < 0.0005. Otherwise it withholds as "opposes" if P ≥ 0.50 and the lead delta is < 0, or P < 0.50 and the lead delta is > 0.
4. **Sign agreement** (pre-registered directions below).
   - Among windows with |delta| > 0.0005, the share where sign(delta) == sign(z). This is the share where an above-mean value raises the score. It is reported overall, by class, and by candidate vs non-candidate.
   - ICE: the input's z is swept over −3…3 (61 points) with the window's other inputs fixed. The share of windows whose curve is non-decreasing (tolerance 1e-6) is reported.
   - Run for P5 `r_beat` (primary), P5 `hr_bpm` (secondary), and v2 `ratio_of_ratios` (reference).
5. **Baseline-saturation sensitivity** (host only, not the device method). Checks 3 and 4 are repeated with the reference point set to the per-feature median (in z) of label-0 windows instead of 0. For the sign test this uses sign(z − ref). The v2 training-mean R (1.048) sits in the saturated output-spline region. ICE does not depend on the reference and is not repeated.

## Expected directions (fixed now)

| Input | Expected sign | Role |
|---|---|---|
| `r_beat` | + (a higher red/IR ratio means lower saturation, so higher risk) | **Primary** sign test |
| `hr_bpm` | + (weak, confounded: hypoxaemia can raise pulse rate) | Secondary sign test |
| `pi_beat`, `sqi_template`, `sqi_ibi_ratio` | none (reliability and context only) | Excluded from the sign test |
| Concept head weight on estimated ScalcO2 | − (lower saturation, higher risk) | Concept sign test |

## Stage 4: concept bottleneck (Koh et al. 2020, sequential; host only)

**Learned concept: estimated ScalcO2 (`s_hat`).**
- The target is `median_spo2` clipped to [50, 100], and the clipped-row count is reported. `median_spo2` is used only as this target and never as an input.
- The main predictor is a TinyKAN-architecture regressor on P5. It uses `KANBinaryClassifier((5, 12, 1))` with no sigmoid, MSE on the target standardized per training fold, the same optimizer, epochs, smoothness and clip, and seed 42.
- Comparators: Ridge(alpha=1.0) on P5 and on V16+P5.

**Given (computed) concepts:** `hr_bpm`, `pi_beat`, `sqi_template`, `sqi_ibi_ratio`.

**Label head.** LogReg (`BASELINE_SETTINGS["logreg"]`) on standardized `[s_hat, hr_bpm, pi_beat, sqi_template, sqi_ibi_ratio]`. Training is cross-fitted:
- Inside each outer training fold, GroupKFold(5) produces inner out-of-fold `s_hat` for fitting the head. The inner grouping is by recording for recording-grouped outer folds and by patient for patient-grouped outer folds.
- The outer-test `s_hat` comes from a regressor fitted on the whole outer training fold.
- The head therefore never sees in-sample concept predictions.

**Pure bottleneck:** score = 90 − `s_hat`, flag = `s_hat` < 90.

**Reported:**
- Concept quality, recording- and patient-grouped: MAE, RMSE, bias, Bland-Altman limits (bias ± 1.96 SD), Pearson, and RMSE by reference range (< 80, 80–90, ≥ 90). The export dropped medians in [89, 91], so 80–90 means [80, 89) and ≥ 90 means ≥ 91.
- Task AUC, recall, specificity and PR-AUC at 0.50, beside TinyKAN-P5 and v2-16.
- Head coefficients and their signs, per fold and pooled.
- Information check: heads with `s_hat` only, `s_hat` plus each given concept, all five, and the given concepts only.
- Oracle check: the true clipped target in place of `s_hat`. It will be near-perfect trivially, because the label is the median < 90.
- Concept-level explanation: logit contributions `w_k · z_k` per window (exact for a linear head), and the top-concept share by class.

## Reading rules (fixed now)

- **Prediction kept.** P5 TinyKAN keeps prediction if the paired recording-bootstrap 95% CI of AUC(P5 TinyKAN) − AUC(v2-16 TinyKAN) has a lower bound above −0.02. No model is called better unless the lower bound is above 0. Even then the claim is limited to "recording-grouped, on this table". Patient-grouped is the secondary reading.
- **Direction checks.** For `r_beat` (and, as the reference, v2 `ratio_of_ratios`), the sign-agreement share and the ICE non-decreasing share are each read as:
  - ≥ 0.90 "consistent with the expected direction";
  - 0.50–0.90 "mixed";
  - < 0.50 "contradicts".
  `hr_bpm` uses the same bands, labelled secondary and confounded.
- **Concept sign.** The concept direction holds if the `s_hat` coefficient is negative in the pooled head and in every outer fold of both groupings.
- **Wording.**
  - AOPC is host, in-sample, on the exported model, and functionally grounded only. No explanation is called "faithful" without that qualification.
  - Mean-perturbation deltas are not effect sizes.
  - `s_hat` is uncalibrated for the MAX30102: R-to-saturation is sensor-specific, and transfer from OpenOximetry to the MAX30102 is untested.

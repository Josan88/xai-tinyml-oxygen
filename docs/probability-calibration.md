# Probability calibration of the TinyKAN score (host only)

3 November 2026. Question: is the TinyKAN risk probability calibrated on OpenOximetry? Does P = 0.3 mean that about 30% of such windows are low-oxygen? This is a measurement only. Nothing in the model, the 0.50 operating point, ADR-0005 N = 2 or the firmware was changed, and nothing was retrained or recalibrated.

**Scope.** v1 is the reported system: TinyKAN `(14, 12, 1)`, notebook-10 as-coded labels. v2 is the comparison: 16 inputs, header-aligned ScalcO2 labels (ADR-0006). Host only: no COM8, no flashing, no build, no board script. No header, firmware source or existing `outputs/` file was written.

**Inputs.** The existing out-of-fold files, loaded and fold-checked as in `scripts/analyze_lamaakal_style.py` (`load_inputs`): `outputs/openox_tinykan_predictions.csv` (`y_prob`, folds `split_index`), `outputs/openox_logreg_local_explanations.csv` (`probability_low_oxygen_risk`, aligned on `source_index`), and `outputs/openox-v2/predictions_{b_v2_tinykan16,b_v2_tinykan16_patient,c_v2_logreg16}.csv` (`y_prob`, `fold`). No `y_pred` or `predicted_label` column was read (the v1 `y_pred` is the 0.30 cut). Every flag was recomputed from the probability at 0.50.

**Sanity checks** (all pass; `summary.json` → `checks`): v1 52,090 windows / 162 recordings, v2 48,662 / 154. At 0.50, v1 TinyKAN reproduces recall 0.767 / specificity 0.394, v2 recording-grouped 0.893 / 0.868, and v2 patient-grouped 0.899 / 0.864. The v1 LogReg row is the notebook-11 same-fold row (0.866 / 0.261), not the README's 0.859 / 0.264 reconstruction. The Newton-fitted calibration slope and intercept match sklearn's unpenalised `LogisticRegression` to within 4e-9.

Outputs: `outputs/calibration-v1/` (`calibration_metrics.csv`, `reliability_bins.csv`, `per_fold.csv`, `bootstrap_ci.csv`, `paired_differences.csv`, `fig_reliability.png`, `paper_sentences.md`, `summary.json`). Code: `scripts/analyze_calibration.py`. Tests: `tests/test_calibration_metrics.py` (10 synthetic-data tests).

---

## What was measured

For each series:

- **Base rate and calibration-in-the-large:** mean y, mean P, and mean P minus base rate.
- **Brier score and Brier skill score:** BSS = 1 − Brier / Brier_ref, where Brier_ref is the constant base-rate predictor.
- **Murphy decomposition:** reliability, resolution and uncertainty on 10 equal-width bins, plus the binning residual.
- **ECE and MCE:** ECE on 10 equal-width bins and on 10 equal-mass bins, weighted by bin count. MCE on the equal-width bins.
- **Calibration intercept and slope:** unpenalised logistic regression of y on logit(P), with P clipped to [1e-6, 1 − 1e-6]. Also the intercept with the slope fixed at 1 (logit P as offset). Slope < 1 means overconfident; slope > 1 means compressed/underconfident.
- **Near the operating point:** the observed rate in [0.45, 0.50) and [0.50, 0.55), and PPV and NPV at 0.50.
- **Per fold:** Brier, ECE and slope.

95% CIs come from a cluster bootstrap over recordings: 2,000 resamples, seeds 20261103 (v1) and 20261104 (v2). The v2 patient-grouped series also has a patient bootstrap (87 patients, seed 20261105). TinyKAN − LogReg differences are paired, because both models use the same resamples. Equal-mass bins are 10 runs of the stably sorted P with equal total weight. The v2 TinyKAN output clamps near 0.89, so tied values can straddle a bin edge.

## Results

| Series | Brier | BSS | ECE (equal-width) | ECE (equal-mass) | Slope | Intercept (slope 1) |
|---|---:|---:|---:|---:|---:|---:|
| v1 TinyKAN | 0.240 (0.237–0.244) | 0.009 (−0.008–0.025) | 0.086 (0.073–0.100) | 0.085 (0.072–0.099) | 0.98 (0.77–1.24) | 0.357 (0.299–0.414) |
| v1 LogReg | 0.244 (0.242–0.247) | −0.007 (−0.021–0.005) | 0.091 (0.080–0.104) | 0.088 (0.077–0.102) | 0.55 (0.18–1.24) | 0.352 (0.302–0.400) |
| v2 TinyKAN, recording-grouped | 0.098 (0.083–0.114) | 0.596 (0.530–0.656) | 0.037 (0.026–0.055) | 0.042 (0.030–0.059) | 1.04 (0.93–1.17) | 0.353 (0.193–0.514) |
| v2 TinyKAN, patient-grouped (patient bootstrap) | 0.095 (0.083–0.108) | 0.606 (0.552–0.657) | 0.032 (0.025–0.050) | 0.035 (0.028–0.053) | 1.08 (0.98–1.19) | 0.295 (0.157–0.447) |
| v2 LogReg, recording-grouped | 0.164 (0.153–0.175) | 0.323 (0.274–0.367) | 0.179 (0.154–0.202) | 0.163 (0.142–0.189) | 0.70 (0.25–1.45) | 0.380 (0.276–0.482) |

95% recording-bootstrap CIs in brackets, except where a patient bootstrap is stated. The two-parameter fit gives intercepts within 0.004 of the slope-1 column for TinyKAN and within 0.011 for LogReg (`calibration_metrics.csv`). Calibration-in-the-large (mean P − base rate) is −0.085 for v1 TinyKAN (base rate 0.587), −0.037 for v2 TinyKAN and −0.031 for the patient-grouped series (base rate 0.592).

**Murphy decomposition (equal-width).** v1 TinyKAN: reliability 0.0078, resolution 0.0092, uncertainty 0.2424. v2 TinyKAN: 0.0022, 0.1469 and 0.2416. The binning residual is at most 0.0008 in absolute value.

**Paired TinyKAN − LogReg** (`paired_differences.csv`). v1: Brier −0.004 (−0.006 to −0.002), ECE −0.005 (−0.018 to +0.005). v2 recording-grouped: Brier −0.066 (−0.073 to −0.058), ECE −0.142 (−0.168 to −0.108). On v1, TinyKAN's Brier advantage over LogReg is resolved but tiny, and the ECE difference is not resolved. On v2, TinyKAN is better calibrated than LogReg on both measures.

**Near the operating point** (window level). For v1 TinyKAN, windows with P in [0.45, 0.50) are low-oxygen 53.4% of the time (n = 6,380), and windows in [0.50, 0.55) 61.6% (n = 17,975). PPV at 0.50 is 0.643 and NPV is 0.543. For v2 TinyKAN, the same bands are 60.2% (n = 274) and 68.4% (n = 266), with PPV 0.907 and NPV 0.848. On both, a window just below 0.50 is more often low-oxygen than not. This is descriptive only. The cut stays at 0.50.

**Per fold** (`per_fold.csv`). v1 TinyKAN Brier ranges 0.236–0.246 across folds and slope 0.68–1.37. v2 TinyKAN Brier ranges 0.075–0.110 and slope 0.86–1.29. With n = 5 folds, these ranges show spread; they are not a test. LogReg slopes swing from 0.12 to 2.43 across folds because a few extreme-P windows drive them, so ECE is the better LogReg summary. The same cause gives the LogReg slope CIs in the table their width.

![Reliability curves](../outputs/calibration-v1/fig_reliability.png)

*Reliability curves on 10 equal-mass bins, with shaded recording-bootstrap bands for the observed rate (bin membership fixed at the full-data bins). Panels: v1, v2 recording-grouped, v2 patient-grouped (TinyKAN only; no patient-grouped LogReg out-of-fold file was an input). Under each panel is the histogram of predicted P. The dotted line is perfect calibration; the dash-dot line is the 0.50 operating point.*

## What it means

**v1: no, a 0.3 does not mean about 30%.** Windows scored 0.2–0.3 (mean P 0.24) were low-oxygen 33% of the time, and windows scored 0.3–0.4 (mean 0.36) 43% of the time. The lowest equal-mass bin (mean P 0.27) observed 37% (95% CI 29–42%). The more important point is that a v1 score of 0.3 is rare. Only 5.5% of windows score below 0.3, and 82% fall between 0.45 and 0.60. The slope of 0.98 does not mean v1 is well calibrated. The scores are nearly constant, so the slope has little to fit. The Brier skill of 0.009, with a CI that includes 0, means the v1 probability is no better than always saying the base rate of 0.587.

**v2: roughly, at the ends; it reads low in the middle.** Windows scored 0.2–0.3 (mean 0.24) were low-oxygen 27% of the time. Windows scored 0.3–0.4 (mean 0.34) were low-oxygen 46% of the time. So a v2 0.3 means roughly 30–45%. Only 4.8% of windows fall in 0.2–0.4. Most v2 scores sit near 0.07–0.13 (observed 8–18%) or near 0.89 (observed 92–94%). Slope 1.04 and ECE 0.037 mean v2 ranks well and is nearly calibrated.

**Both TinyKANs read low.** On average, both underpredict the low-oxygen rate: by 8.5 points for v1 and 3.7 points for v2. The slope-1 intercepts (0.357 and 0.353) are close to logit(base rate) (0.352 and 0.371). That is the pattern expected from training with `pos_weight` BCE, which weights the classes as if prevalence were 50%. This is consistent with the observation; it is not proven by it.

## Limits

1. **CV recipe, not the shipped weights.** These out-of-fold probabilities come from the cross-validation fold models. The shipped headers are all-rows fits. This measures the calibration of the recipe, not of the exact shipped weights.
2. **Effective n is the recording count.** All numbers are window-level, and windows overlap (10 s window, 5 s hop). The effective n is closer to the number of recordings (162 for v1, 154 for v2), which is why the CIs come from a recording bootstrap.
3. **Only v2 is patient-grouped.** v1 recording-grouped folds let a patient appear in both a training fold and the held-out fold. Only v2 has a patient-grouped series.
4. **Labels.** v1 uses the as-coded notebook-10 labels. v2 uses header-aligned ScalcO2, which is estimated from ETO2. It is not a pulse-oximeter SpO2 and is not exact.
5. **OpenOximetry only.** Calibration on OpenOximetry does not transfer to the MAX30102, because the R-to-saturation relation is sensor-specific. Nothing here says anything about live captures, including the Subject-1 PTT data.
6. **A risk probability, not a saturation.** Even a calibrated risk probability is not an SpO2 value and not a clinical probability for a patient.
7. **Nothing was recalibrated.** Platt or isotonic scaling would produce a new model and would reset parity, timing and on-device XAI evidence. It is not proposed as a change to the shipped model.

## Reproduction

`C:\Python314\python.exe scripts/analyze_calibration.py` (about 5 minutes; single-threaded BLAS). The output directory is write-once: the script stops if `outputs/calibration-v1/` exists, and it writes nothing if a sanity check fails. A dry run to a scratch directory before the committed run gave identical numbers. `summary.json` records input SHA-256s, interpreter and library versions, seeds, checks and every headline number. Tests: `C:\Python314\python.exe -m pytest tests/test_calibration_metrics.py`.

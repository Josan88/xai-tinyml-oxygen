# Label-0-median explainer reference: host analysis pre-registration (30 Oct 2026)

Written and committed **before** any number in this document's "New measurement" section was computed. Host only, laptop, no board. Every result here is host, in-sample, functionally grounded only. Results go to the final report; the IEEE paper is not edited.

## Decision being evaluated

The device explainer (Mean-Perturbation Attribution) computes, for each standardized input i,
delta_i = P(x) - P(x with z_i set to r_i), ranks |delta_i| descending (ties by index) and sends the top 3.

- Shipped v1 and v2 images: r_i = 0, the training mean.
- New device images (Jason, 30 Oct 2026, job F1): r_i = the **label-0 median**, the median standardized input over the windows whose label is 0 (window-median ScalcO2 >= 90). The shipped images are unchanged.

For the Header-aligned model (repo v2) the reference is the vector `KAN_XAI_REF_Z` in `firmware/Arduino Nano 33/include/kan_xai_reference_v2.h`, written by `scripts/make_xai_reference_header.py` from `outputs/openox-v2/window_features.csv` and the shipped `kan_model_weights_v2.h` scaler. It holds the float64 medians rounded to FP32, as the device holds them. For P5 the reference is `outputs/physio-concept-v1/s3.json` -> `headline.references.P5.*.label0_median_z`, rounded to FP32 the same way. P5 has no header yet; a later job ports it.

## Already known from Stage 3 (not re-tested here)

These come from `outputs/physio-concept-v1/` (`s3.json`, `s3_degeneracy.csv`, `s3_sign_agreement.csv`, `s3_aopc.csv`). Stage 3 used the float64 label-0 median as a host-only sensitivity check. The withhold-rule cells already exist there, as the `dashboard_withheld*` columns.

| Quantity (all 48,662 v2 windows, in-sample) | v2 header, training mean | v2 header, label-0 median | P5, training mean | P5, label-0 median |
|---|---|---|---|---|
| Candidates whose top bar opposes the flag | 60.8% | 0.0% | 77.0% | 0.3% |
| Sign agreement of the R input (v2 `ratio_of_ratios`, P5 `r_beat`), all / candidates | 0.756 / 0.585 | 0.963 / 1.000 | 0.702 / 0.472 | 0.989 / 1.000 |
| Withhold-rule share, all / candidates / non-candidates | 0.362 / 0.608 / 0.008 | 0.148 / 0.000 / 0.361 | 0.457 / 0.774 / 0.040 | 0.225 / 0.003 / 0.518 |
| Deletion AOPC, all windows: firmware-absolute vs random (deletion to 0) | -0.3092 vs -0.1612 | not computed | -0.3404 vs -0.2074 | not computed |
| Deletion AOPC, candidates: firmware-absolute vs random (deletion to 0) | -0.0085 vs -0.0004 | not computed | -0.0398 vs -0.0255 | not computed |

Known consequence, stated now so it is not read later as a finding: under the label-0 reference, the withhold rule's "opposes" branch moves from candidates to non-candidates. An unflagged window whose lead input is above its label-0 value gets a positive lead delta, and the dashboard rule treats a score-raising lead on an unflagged window as opposing the result. That rule was written for the training-mean reference. Whether its wording suits label-0 images is a dashboard decision. This study does not make it.

## New measurement (fixed now)

Script: `scripts/analyze_xai_reference_label0.py`. It is new, and it imports from `scripts/analyze_lamaakal_style.py` and `scripts/analyze_physio_concept.py` without editing either. Output: `outputs/xai-reference-label0-v1/` (new directory) with `README.md`, `summary.json` (input and code SHA-256) and CSV tables.

Models and rows:
- **v2**: the shipped header `kan_model_weights_v2.h` through `lamaakal_models.HeaderKAN`, on all 48,662 rows of `outputs/openox-v2/window_features.csv`.
- **P5**: the all-window analysis model `outputs/physio-concept-v1/p5_final_model.npz` through `analyze_physio_concept.ArrayKAN`, on the same rows (`physio_features.csv.gz`). Beat-invalid inputs take the training mean (z = 0), as in Stage 3.

**M0, same-code checks. They run first; the script stops if one fails.**
1. Training-mean cells for v2 (deletion to 0, firmware-absolute of training-mean deltas, random 5 repeats, seed 20260911) must reproduce `outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv`, every subset and ranking, within 1e-8. Expected headline: firmware_absolute -0.3092 vs random -0.1612, all windows.
2. Training-mean cells for P5 must reproduce `outputs/physio-concept-v1/s3_aopc.csv` within 1e-8.
3. Withhold-rule and opposing-bar shares under the float64 label-0 median must reproduce `s3_degeneracy.csv` within 1e-12, for both models and both references.
4. The v2 reference read from `kan_xai_reference_v2.h` must equal the float64 Stage 3 medians rounded to FP32, bit for bit.

**M1, deletion AOPC under the label-0 reference (the new number).** Deleted inputs are set to r_i (FP32 label-0 median), not 0. Rankings:
- firmware-absolute: descending |delta|, ties by index, of the **label-0-reference** deltas. This is the ranking the new device image sends.
- random: 5 repeats, seed 20260911 (`evaluate_xai_faithfulness.ranked_orders`, unchanged).
- signed_risk: secondary, as in the existing tables.

AOPC is the mean over k = 1..d of P(x) - P(x with the top-k inputs deleted). The target is P(risk). Subsets: all, candidate (P >= 0.50), non-candidate (P < 0.50), label_1, label_0. P is the unperturbed probability, which is the same under both references. Insertion is not part of this measurement.

**M2, withhold-rule share under both references.** This is the dashboard rule (`dashboard/dashboard_core.js`; `analyze_physio_concept.degeneracy_rows`). Withheld if |lead delta| < 0.0005, or if the lead sign opposes the flag: lead < 0 at P >= 0.50, or lead > 0 at P < 0.50. Reported with the zero and opposes parts separately, and the share of candidates whose top bar opposes the flag. Subsets: all, candidate, non-candidate, label_1, label_0. Both models, with the FP32 reference as the device holds it. The float64 cells are only for M0.3.

## Readings, fixed in advance

- **R1, per model and subset.** "Firmware-absolute beats random under the label-0 reference" if and only if the label-0-reference AOPC of firmware-absolute is strictly greater than the label-0-reference random mean. Also reported: the fraction of windows where it beats its own random mean. No significance test: in-sample, one dataset, functionally grounded.
  - Primary cell: **v2 candidates**. The flagged windows are the ones where an operator sees a top bar.
  - Secondary cell: v2 all windows.
  - P5 cells and the other subsets are descriptive.
- **R2, cross-reference comparison.** Descriptive only. Each reference defines its own deletion target, so a training-mean AOPC and a label-0 AOPC are not the same experiment. No "better" or "more faithful" reading is drawn from comparing them. The comparable statement is R1 under each reference: does firmware-absolute beat random? If R1 holds under the label-0 reference where it failed under the training mean, that is reported as such, with this caveat.
- **R3, withhold share.** Reported per subset and reference. The candidate-side change is already known from Stage 3 (above). Here it is reproduced with the FP32 reference. A difference of more than 0.001 from the float64 cells is reported as an FP32-rounding effect.

## Limits stated now

- In-sample on the rows the v2 header and the P5 model were trained on.
- One dataset (OpenOximetry). The label is ScalcO2 from end-tidal O2, not pulse-oximeter SpO2.
- Deletion AOPC is a functionally grounded proxy, not a human evaluation.
- No result here is called faithful or better without "host, in-sample, functionally grounded".
- Mean-perturbation deltas are not effect sizes.
- Nothing here is board evidence. The ref0 images' board parity, live idle and sit are separate, in `docs/xai-reference-label0-runbook.md`.

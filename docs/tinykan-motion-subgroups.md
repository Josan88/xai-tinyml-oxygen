# TinyKAN motion / signal-quality subgroups

- Date: 2026-08-26
- Project: EAT40006
- Dataset: OpenOximetry canonical windows
- Operating point: 0.50 (fixed)

## Method

`outputs/openox_tinykan_predictions.csv` was aligned one-to-one with `outputs/openox_logreg_local_explanations.csv` on TinyKAN `row_index == source_index`. Equality of `recording_id` and `y_true == true_label` was asserted after alignment. TinyKAN predictions were recomputed as `y_prob >= 0.50`, ignoring stored `y_pred`; Logistic Regression predictions were recomputed as `probability_low_oxygen_risk >= 0.50`.

`recording_id` was joined to `encounter_id` through `outputs/openox_canonical_export_audit.csv`; the mapping is one-to-one in this export. `pulseox_pi_mean` was parsed as a float from `outputs/openox_sample_reference_pairs.csv`, with blanks treated as missing, and the mean of non-null sample PI values was computed per encounter. This encounter mean is the only quality axis.

The tertile cutpoints were computed from the 160 available encounter-mean PI values among the 162 prediction encounters, using equal-count tertiles on encounters rather than windows. Each prediction window inherited its encounter's label: `T1_low`, `T2_mid`, or `T3_high`; encounters without PI were assigned `missing`. Intervals follow right-closed `qcut` assignment: `T1_low` includes the minimum through the first cutpoint, `T2_mid` is above the first through the second cutpoint, and `T3_high` is above the second cutpoint through the maximum.

Encounter-mean PI tertile edges [min, q1, q2, max] are 0.579583, 10.0436, 29.608, 215.164; encounter counts are T1_low=54, T2_mid=53, T3_high=53, missing=2.

Predictions contain no `sample_num`, so window-local PI cannot be joined and was not invented. Eligibility required `n >= 50`, `n_pos >= 5`, and `n_neg >= 5`; overall was always included.

## Columns Searched

All `outputs/*.csv` headers were searched.

- No motion/imu/accel/gyro/artifact/sqi/quality/confidence/spo2_quality column exists in any `outputs/*.csv`.
- The usable existing column is `pulseox_pi_mean` in `outputs/openox_sample_reference_pairs.csv` (perfusion index).
- Source `pulseoximeter.csv` also has `pi`, but this analysis uses the already-exported `pulseox_pi_mean`; no new scrape was performed.
- `encounter.csv` has no motion column.

## Results

Positive means low-oxygen risk (`y_true == 1`). Gaps are signed stratum-minus-overall differences at the same fixed threshold.

| Stratum | Model | n | Prevalence | Recall | Recall gap vs overall | Spec | Spec gap vs overall | PPV | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | TinyKAN | 52090 | 0.587118 | 0.766766 | +0.000000 | 0.394058 | +0.000000 | 0.642783 | 0.605942 |
| Overall | Logistic Regression | 52090 | 0.587118 | 0.866167 | +0.000000 | 0.261078 | +0.000000 | 0.625029 | 0.738922 |
| T1_low | TinyKAN | 18411 | 0.585357 | 0.774241 | +0.007476 | 0.382237 | -0.011820 | 0.638897 | 0.617763 |
| T1_low | Logistic Regression | 18411 | 0.585357 | 0.837060 | -0.029107 | 0.275216 | +0.014138 | 0.619830 | 0.724784 |
| T2_mid | TinyKAN | 16898 | 0.587939 | 0.710317 | -0.056449 | 0.453827 | +0.059770 | 0.649816 | 0.546173 |
| T2_mid | Logistic Regression | 16898 | 0.587939 | 0.829592 | -0.036575 | 0.301881 | +0.040804 | 0.629016 | 0.698119 |
| T3_high | TinyKAN | 16199 | 0.587382 | 0.830058 | +0.063292 | 0.336326 | -0.057732 | 0.640344 | 0.663674 |
| T3_high | Logistic Regression | 16199 | 0.587382 | 0.942722 | +0.076555 | 0.194345 | -0.066733 | 0.624869 | 0.805655 |
| missing | TinyKAN | 582 | 0.611684 | 0.424157 | -0.342609 | 0.659292 | +0.265234 | 0.662281 | 0.340708 |
| missing | Logistic Regression | 582 | 0.611684 | 0.721910 | -0.144257 | 0.500000 | +0.238922 | 0.694595 | 0.500000 |

The largest absolute TinyKAN recall gap versus overall among eligible PI-axis strata is 0.342609 for `missing` (recall 0.424157; signed gap -0.342609 versus overall 0.766766).

The largest absolute TinyKAN specificity gap versus overall among eligible PI-axis strata is 0.265234 for `missing` (specificity 0.659292; signed gap +0.265234 versus overall 0.394058).

Subject/recording heterogeneity is in `docs/tinykan-fairness-subgroups.md`; this note does not rewrite that table.

## Honest Limits

- This is NOT motion. No IMU/accel/gyro labels exist. `README.md` still lists motion-artifact stress testing as unchecked future work.
- PI is a perfusion / signal-quality proxy, joinable only as an encounter-level mean (7-29 non-null samples per encounter here). It is not window-local quality.
- The exported sample-level `pulseox_pi_mean` scale (`n=3942`) has median 18.425 and maximum 418.5; do not treat these encounter-mean tertiles as clinical PI cutoffs such as 0.3 or 1.0.
- Two prediction encounters lack PI.
- Windows within an encounter are not independent.
- The threshold was not retuned; models were not retrained; firmware was untouched.
- This is not a fifth assessment and not a firmware milestone.

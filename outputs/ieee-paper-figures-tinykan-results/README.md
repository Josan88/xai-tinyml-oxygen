# ieee-paper-figures-tinykan-results

Three one-column (235 pt = 3.264 in) Results figures for the IEEE A4 draft, for the reported TinyKAN (repo v2,
16 inputs, header `kan_model_weights_v2.h`; ADR-0006) only. Drawn by `scripts/make_ieee_paper_results_figs_tinykan.py`.
Plotting only: every plotted value is read from a committed CSV and nothing is recomputed from a model. Every font is
8 pt, 300 dpi, no figure title and no model-version id on any image. Identity is carried by marker shape or by solid
vs hatched fill as well as colour, so each figure reads in greyscale. `manifest.json` lists every plotted value with
its source file, row keys and column, and the SHA-256 and git blob id of every source.

- `figR1_fold_spread.png`: 3.264 x 2.45 in, 979 x 735 px, SHA-256 `1e52de3042288d6898123ce3f0548bc2444a87948c666c7e7749ec3f379e6a16`
- `figR2_error_by_reference_distance.png`: 3.264 x 2.10 in, 979 x 630 px, SHA-256 `a7fff9b66972b949722e2bea030852b5c40e915f5b5fb9fb218b8396b1eeeda2`
- `figR3_deletion_aopc.png`: 3.264 x 2.30 in, 979 x 690 px, SHA-256 `809c0e1d520b4f45083687a6bfb4e7157c887b06b3ae7d4313f411ac916b7b70`

## figR1_fold_spread.png (RQ1)

Per-fold spread for TinyKAN and logistic regression under recording-grouped 5-fold cross-validation, in three
mini-panels with their own x axes: AUC, recall (%), specificity (%). Markers are the five fold values (TinyKAN filled
circles, logistic regression open squares), offset vertically by fold index (fold 1 highest, fold 5 lowest) so all
five can be counted; the short ink bar is the pooled out-of-fold value. The dashed line at 80 in the recall panel is
the recall gate; the 50% specificity gate is not drawn (far left of the data).

Sources: `outputs/analysis-lamaakal-style-v1/c_per_fold.csv` rows version == v2, grouping == recording, model in (TinyKAN, LogReg), folds 1-5, columns
auc, recall, specificity; pooled values from `outputs/analysis-lamaakal-style-v1/a_baselines.csv` rows version == v2, grouping == recording, learner in
(tinykan, logreg), same columns.

Observation (not a check): logistic-regression folds with recall below 80%: fold 2 recall 78.53%. TinyKAN meets the gate in
every fold.

## figR2_error_by_reference_distance.png

Error rate by distance of the window's reference median ScalcO2 from the cut (median - 90, in points). Rows above the
cut line are negative windows (y = 0) and the hatched bar is the false-flag rate FP / n; rows below are positive
windows (y = 1) and the solid bar is the missed-flag rate FN / n. Bins are half-open as in the CSV: "+1 to +2" is
[+1, +2), "-1 to -2" is [-2, -1), "+10 or more" is [+10, inf), "more than 10 below" is [-inf, -10).

Source: `outputs/analysis-lamaakal-style-v1/g_error_by_spo2_distance.csv` rows version == v2 (eight bins), columns n, error_rate (errors / n checked).

Gap at the cut, verified on `outputs/openox-v2/window_features.csv` (column `median_spo2`, the window's reference median ScalcO2; the
`label` column equals median_spo2 < 90 on every window): {"[89, 91)": 0, "[89, 91]": 0, "[89, 90)": 0, "[89, 90]": 0, "[90, 91)": 0, "[90, 91]": 0, "[88, 92)": 2528, "[88, 92]": 2528}. Largest median below 90:
88.999382; smallest median at or above 90: 91.000240. Windows in
the excluded band [-1, +1) of distance: 0. So no window has a reference
median in [89, 91], as the analysis doc says, and the separator is labelled "cut (90)".

Sum of n over the eight v2 rows: 48662 (window table: 48662). They are equal: because the band [89, 91)
is empty, the eight bins cover every window.

## figR3_deletion_aopc.png (RQ2)

Deletion AOPC of the firmware ranking (mean-perturbation |delta|, filled circle) against random order (mean of five
repeats, open diamond, drawn on top and slightly larger so it stays visible where the two nearly coincide), joined by a
thin grey line; vertical line at zero. Panel (a): shipped training-mean reference; panel (b): label-0 median reference.
Rows: candidates (P >= 0.50), non-candidates, all windows, with n read from the CSV. signed_risk and reverse_signed are
not plotted.

Sources: (a) `outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv` rows version == v2, rankings firmware_absolute and random_mean, column
deletion_aopc_window_mean; (b) `outputs/xai-reference-label0-v1/aopc_deletion.csv` rows model == v2, reference == label0_fp32, same rankings and
column. Panel (b) non-candidates (not printed in the paper): firmware 0.031671,
random 0.013945.

## Consistency checks run (all must pass or the script exits before writing)

- PASS: R1 TinyKAN: folds 1-5 present once (folds [1, 2, 3, 4, 5])
- PASS: R1 TinyKAN: fold windows sum to the pooled n (sum 48662, pooled 48662)
- PASS: R1 Logistic regression: folds 1-5 present once (folds [1, 2, 3, 4, 5])
- PASS: R1 Logistic regression: fold windows sum to the pooled n (sum 48662, pooled 48662)
- PASS: R1 TinyKAN: pooled AUC / recall / specificity = Table II (got ('0.908', '89.3', '86.8'), paper ('0.908', '89.3', '86.8'))
- PASS: R1 Logistic regression: pooled AUC / recall / specificity = Table II (got ('0.863', '82.1', '87.0'), paper ('0.863', '82.1', '87.0'))
- PASS: R1 TinyKAN fold AUC mean 0.911, s.d. (ddof=1) 0.019 (mean 0.911028, s.d. 0.018657)
- PASS: R1 TinyKAN fold AUC mean and s.d. equal c_fold_summary.csv (csv mean 0.91102791, s.d. 0.01865675)
- PASS: R1 TinyKAN meets the gate in every fold (recall >= 80%, specificity >= 50%) (min recall 87.43%, min specificity 82.64%)
- PASS: R2 eight v2 bins, matching the drawn rows (bins ['[-inf, -10)', '[-10, -5)', '[-5, -2)', '[-2, -1)', '[+1, +2)', '[+2, +5)', '[+5, +10)', '[+10, inf)'])
- PASS: R2 [+10, inf): error type FP, errors / n = error_rate (FP, 1 / 115 vs 0.0086956522)
- PASS: R2 [+5, +10): error type FP, errors / n = error_rate (FP, 1697 / 11967 vs 0.14180663)
- PASS: R2 [+2, +5): error type FP, errors / n = error_rate (FP, 659 / 6287 vs 0.10481947)
- PASS: R2 [+1, +2): error type FP, errors / n = error_rate (FP, 265 / 1498 vs 0.17690254)
- PASS: R2 [-2, -1): error type FN, errors / n = error_rate (FN, 430 / 1030 vs 0.41747573)
- PASS: R2 [-5, -2): error type FN, errors / n = error_rate (FN, 1179 / 6094 vs 0.19346899)
- PASS: R2 [-10, -5): error type FN, errors / n = error_rate (FN, 672 / 8684 vs 0.077383694)
- PASS: R2 [-inf, -10): error type FN, errors / n = error_rate (FN, 809 / 12987 vs 0.062293062)
- PASS: R2 FN rate in [-2, -1) = 0.417 (paper) (0.417476)
- PASS: R2 FN rate in [-inf, -10) = 0.062 (paper) (0.062293)
- PASS: R2 sum of n over the eight bins = 48662 windows (sum 48662, window table rows 48662)
- PASS: R2 window label = (median_spo2 < 90) on every window (28795 positives, 0 missing medians)
- PASS: R3 windows equal for all / firmware_absolute (48662 vs 48662)
- PASS: R3 windows equal for all / random_mean (48662 vs 48662)
- PASS: R3 windows equal for all / signed_risk (48662 vs 48662)
- PASS: R3 windows equal for all / reverse_signed (48662 vs 48662)
- PASS: R3 windows equal for candidate_p_ge_0_50 / firmware_absolute (28661 vs 28661)
- PASS: R3 windows equal for candidate_p_ge_0_50 / random_mean (28661 vs 28661)
- PASS: R3 windows equal for candidate_p_ge_0_50 / signed_risk (28661 vs 28661)
- PASS: R3 windows equal for candidate_p_ge_0_50 / reverse_signed (28661 vs 28661)
- PASS: R3 windows equal for noncandidate_p_lt_0_50 / firmware_absolute (20001 vs 20001)
- PASS: R3 windows equal for noncandidate_p_lt_0_50 / random_mean (20001 vs 20001)
- PASS: R3 windows equal for noncandidate_p_lt_0_50 / signed_risk (20001 vs 20001)
- PASS: R3 windows equal for noncandidate_p_lt_0_50 / reverse_signed (20001 vs 20001)
- PASS: R3 windows equal for label_1 / firmware_absolute (28795 vs 28795)
- PASS: R3 windows equal for label_1 / random_mean (28795 vs 28795)
- PASS: R3 windows equal for label_1 / signed_risk (28795 vs 28795)
- PASS: R3 windows equal for label_1 / reverse_signed (28795 vs 28795)
- PASS: R3 windows equal for label_0 / firmware_absolute (19867 vs 19867)
- PASS: R3 windows equal for label_0 / random_mean (19867 vs 19867)
- PASS: R3 windows equal for label_0 / signed_risk (19867 vs 19867)
- PASS: R3 windows equal for label_0 / reverse_signed (19867 vs 19867)
- PASS: R3 label-0 file's training_mean rows equal d_aopc_v2_new_run.csv (to 1e-9) (20 subset x ranking rows compared, max |difference| 0.000e+00)
- PASS: R3 (a) candidate_p_ge_0_50: one n for both rankings (n 28661)
- PASS: R3 (a) noncandidate_p_lt_0_50: one n for both rankings (n 20001)
- PASS: R3 (a) all: one n for both rankings (n 48662)
- PASS: R3 (b) candidate_p_ge_0_50: one n for both rankings (n 28661)
- PASS: R3 (b) noncandidate_p_lt_0_50: one n for both rankings (n 20001)
- PASS: R3 (b) all: one n for both rankings (n 48662)
- PASS: R3 (a) candidate_p_ge_0_50: firmware / random = paper (got ('-0.0085', '-0.0004'), paper ('-0.0085', '-0.0004'))
- PASS: R3 (a) noncandidate_p_lt_0_50: firmware / random = paper (got ('-0.7402', '-0.3917'), paper ('-0.7402', '-0.3917'))
- PASS: R3 (a) all: firmware / random = paper (got ('-0.3092', '-0.1612'), paper ('-0.3092', '-0.1612'))
- PASS: R3 (b) candidate_p_ge_0_50: firmware / random = paper (got ('0.775', '0.414'), paper ('0.775', '0.414'))
- PASS: R3 (b) all: firmware / random = paper (got ('0.469', '0.249'), paper ('0.469', '0.249'))
- PASS: R3 subset n: 28661 + 20001 = 48662 ([28661, 20001, 48662])

## Sources

- `outputs/analysis-lamaakal-style-v1/c_per_fold.csv`: SHA-256 `a3f21b262b1aeb5c38dad24e42c4afe4b197a2c9a128deea8a852653af340df3`, git blob `e064446bc93df5a917016c9d1a6a649c3e2f40cc` (equals HEAD blob: True)
- `outputs/analysis-lamaakal-style-v1/c_fold_summary.csv`: SHA-256 `1611d9da9211517db04f9b4bd613c477a68bcd82c74307a261096a53d3a3e139`, git blob `50d4f026705ccb5ae165fd68860de2093c2b6153` (equals HEAD blob: True)
- `outputs/analysis-lamaakal-style-v1/a_baselines.csv`: SHA-256 `304e59aa09c0569af600f06c4ace812d1a4d2aea54c789dda1b7ba105e766f81`, git blob `3ad9e243d4ff029192d64e4bbefd1d8e817d64fb` (equals HEAD blob: True)
- `outputs/analysis-lamaakal-style-v1/g_error_by_spo2_distance.csv`: SHA-256 `79b1ce1d4bc0177eee5ddec8ce6ba50af552f77148e09f1e2ec0ad72160d0c89`, git blob `92ddb005fe9c65d59069762bb88c92821ca3c84c` (equals HEAD blob: True)
- `outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv`: SHA-256 `ad106fd8df92bf27abdeb437d47b1d0c3138899673c286393c422a3b02b47583`, git blob `fb14e81d6fe35ea8b9174f2a2ff37b172c1221a2` (equals HEAD blob: True)
- `outputs/xai-reference-label0-v1/aopc_deletion.csv`: SHA-256 `8bfe47939540072696add62732ff7b2bc4bfa30480851f59d9b11e63bc906a52`, git blob `6dd9eecfe40be5730b12d1c0741794d93faa6723` (equals HEAD blob: True)
- `outputs/openox-v2/window_features.csv`: SHA-256 `d96d93989f1ffa85f72689a2e084c898dacf18d4eafce50712fb0dcf3ad1b3d6`, git blob `7ace27b35ecebdc7ff4997e17ac57e814c275657` (equals HEAD blob: True)

## Run

```
C:\Python314\python.exe scripts/make_ieee_paper_results_figs_tinykan.py            # refuses if this directory exists
C:\Python314\python.exe scripts/make_ieee_paper_results_figs_tinykan.py --compare-with <earlier clean output dir>
```

Determinism: this directory was drawn after an earlier clean run into a scratch directory; the PNG SHA-256s are identical.

- `figR1_fold_spread.png`: earlier `1e52de3042288d6898123ce3f0548bc2444a87948c666c7e7749ec3f379e6a16`, this run `1e52de3042288d6898123ce3f0548bc2444a87948c666c7e7749ec3f379e6a16`
- `figR2_error_by_reference_distance.png`: earlier `a7fff9b66972b949722e2bea030852b5c40e915f5b5fb9fb218b8396b1eeeda2`, this run `a7fff9b66972b949722e2bea030852b5c40e915f5b5fb9fb218b8396b1eeeda2`
- `figR3_deletion_aopc.png`: earlier `809c0e1d520b4f45083687a6bfb4e7157c887b06b3ae7d4313f411ac916b7b70`, this run `809c0e1d520b4f45083687a6bfb4e7157c887b06b3ae7d4313f411ac916b7b70`

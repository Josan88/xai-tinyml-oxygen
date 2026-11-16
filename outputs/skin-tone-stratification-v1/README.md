# Skin-tone stratification of RQ1 (11 Nov 2026)

**Host only, exploratory, added after review.** Stored out-of-fold probabilities (recording-grouped folds) at the 0.50 threshold; nothing trained or retuned. Nothing shipped changes.

- Pre-registration: `docs/skin-tone-stratification-preregistration.md`, commit 75e9fa5 (before the script), amendment A1 commit 2db40be (before any stratified number).
- Script: `scripts/analyze_skin_tone_stratification.py` (first committed unrun as f73555f). Input and script SHA-256 in `summary.json`.
- Bootstrap: 2,000 resamples of whole patients (87), 95 % percentile intervals. A patient brings all their recordings; each recording keeps its own group, so patient counts across groups sum to more than 87.
- One stopped run: two recordings have dorsal spectrophotometer rows with empty L* and b*. Amendment A1 drops them from the ITA grouping only.

## S1 primary: Monk rating at the dorsal site (light 1-4, medium 5-7, dark 8-10)

| Model | Group | Recordings | Patients | Windows | Prevalence | AUC [95 % CI] | Recall % [95 % CI] | Specificity % [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| TinyKAN | light | 47 | 28 | 15390 | 58.7 % | 0.912 [0.887, 0.938] | 92.3 [88.0, 95.3] | 86.0 [82.6, 89.5] |
| TinyKAN | medium | 63 | 45 | 19863 | 60.6 % | 0.932 [0.911, 0.948] | 88.8 [83.4, 93.0] | 91.0 [88.5, 92.9] |
| TinyKAN | dark | 44 | 20 | 13409 | 57.5 % | 0.867 [0.809, 0.911] | 86.4 [77.9, 93.5] | 81.9 [73.4, 88.5] |
| LogReg | light | 47 | 28 | 15390 | 58.7 % | 0.853 [0.813, 0.890] | 82.5 [75.7, 88.6] | 83.4 [78.8, 87.5] |
| LogReg | medium | 63 | 45 | 19863 | 60.6 % | 0.889 [0.848, 0.923] | 83.8 [77.6, 88.5] | 90.2 [87.1, 92.7] |
| LogReg | dark | 44 | 20 | 13409 | 57.5 % | 0.836 [0.778, 0.885] | 79.2 [69.4, 86.5] | 86.6 [80.1, 91.8] |

**Primary contrast (TinyKAN recall, dark minus light): -5.9 points [-14.9, +2.1].**

**Reading (pre-registered).** No difference was detected; the interval does not rule out a difference of up to 14.9 points. This is not evidence of equal performance.

## S2 secondary contrasts, dark minus light (descriptive; no reading was pre-registered)

| Grouping | Model | Metric | Difference [95 % CI] | Interval excludes zero |
|---|---|---|---|---|
| monk_dorsal | TinyKAN | recall | -5.9 [-14.9, +2.1] | no |
| monk_dorsal | TinyKAN | specificity | -4.1 [-13.5, +3.4] | no |
| monk_dorsal | TinyKAN | auc | -0.045 [-0.107, +0.008] | no |
| monk_dorsal | LogReg | recall | -3.4 [-14.2, +6.9] | no |
| monk_dorsal | LogReg | specificity | +3.2 [-4.6, +10.0] | no |
| monk_dorsal | LogReg | auc | -0.017 [-0.083, +0.047] | no |
| monk_forehead | TinyKAN | recall | -6.0 [-14.7, +1.9] | no |
| monk_forehead | TinyKAN | specificity | -5.3 [-14.8, +1.5] | no |
| monk_forehead | TinyKAN | auc | -0.048 [-0.107, +0.004] | no |
| monk_forehead | LogReg | recall | -3.1 [-14.6, +7.1] | no |
| monk_forehead | LogReg | specificity | +2.1 [-5.2, +8.4] | no |
| monk_forehead | LogReg | auc | -0.022 [-0.087, +0.042] | no |
| fitzpatrick | TinyKAN | recall | -7.0 [-14.2, -1.1] | yes |
| fitzpatrick | TinyKAN | specificity | -2.9 [-10.3, +3.2] | no |
| fitzpatrick | TinyKAN | auc | -0.041 [-0.092, +0.001] | no |
| fitzpatrick | LogReg | recall | -3.9 [-12.6, +5.1] | no |
| fitzpatrick | LogReg | specificity | +2.4 [-3.9, +8.3] | no |
| fitzpatrick | LogReg | auc | -0.029 [-0.085, +0.018] | no |
| ita_dorsal_third | TinyKAN | recall | -3.8 [-12.1, +3.8] | no |
| ita_dorsal_third | TinyKAN | specificity | -2.9 [-12.2, +4.0] | no |
| ita_dorsal_third | TinyKAN | auc | -0.031 [-0.087, +0.014] | no |
| ita_dorsal_third | LogReg | recall | -2.1 [-12.5, +6.8] | no |
| ita_dorsal_third | LogReg | specificity | +3.2 [-4.3, +10.4] | no |
| ita_dorsal_third | LogReg | auc | -0.002 [-0.060, +0.051] | no |

Recall and specificity differences are in points; AUC differences in AUC units.

- Every TinyKAN point estimate is lower in the dark group than in the light group, under all four groupings. Only one interval excludes zero: TinyKAN recall under the Fitzpatrick grouping. It is one of 24 secondary cells and carries no pre-registered reading.
- Logistic regression shows the same sign for recall and AUC and the opposite sign for specificity; no interval excludes zero.

## S3 signal context (primary grouping, medians over windows)

| Group | IR perfusion index | Standard deviation of the band-passed IR signal |
|---|---|---|
| light | 0.0101 | 858 |
| medium | 0.0103 | 898 |
| dark | 0.0062 | 594 |

The dark group's median perfusion index is about 40 % lower than the other two. This is a description, not a tested cause.

## Limits

- 87 patients from one laboratory; not powered for a subgroup difference. The dark group has 20 patients.
- Groups differ in more than skin tone (which patients, prevalence, perfusion). Descriptive, not causal.
- The label is ScalcO2 from end-tidal oxygen, not a pulse-oximeter reading; this is not a replay of the occult-hypoxaemia finding.
- Monk ratings are an observer's judgement and vary between a patient's encounters (19 of 87 patients change dorsal level).
- Recording-grouped folds; 32 of 87 patients span folds. Windows with a median from 89 to 91 percent are excluded.
- Host scores only; nothing here speaks to the MAX30102 on the board.

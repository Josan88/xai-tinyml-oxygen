# TinyKAN fairness / subgroup table

- Date: 2026-08-26
- Project: EAT40006
- Dataset: OpenOximetry canonical windows
- Operating point: 0.50 (fixed; not retuned)

## Method

This table loads `outputs/openox_tinykan_predictions.csv` and `outputs/openox_logreg_local_explanations.csv`, aligned one-to-one on TinyKAN `row_index` equal to Logistic Regression `source_index`. Recording IDs and labels were checked for equality after alignment. TinyKAN labels were recomputed as `y_prob >= 0.50`; the stored `y_pred` was ignored because it represents another threshold. Logistic Regression labels were recomputed as `probability_low_oxygen_risk >= 0.50` from the existing comparison probabilities; the result matches the stored `predicted_label`.

Patient IDs were joined from TinyKAN `recording_id` to `outputs/openox_canonical_export_audit.csv` `recording_id`, then through `encounter_id` to `outputs/openox_encounter_coverage.csv` `encounter_id`. In this export, `recording_id` is one-to-one with `encounter_id`, and no analyzed window has a missing `patient_id`.

A stratum is eligible only when `n >= 50`, `n_pos >= 5`, and `n_neg >= 5`. The overall row is always included. The 0.50 threshold was not retuned, and no model was retrained.

## Axes

Available axes:

- `patient_id`: subject-level strata.
- `recording_id`: recording/encounter-level strata; `recording_id` is one-to-one with `encounter_id` in this export.

Skipped axes because no labeled columns exist in the prediction or repository metadata CSVs used here:

- Pigmentation, Monk, Fitzpatrick, or other skin-tone labels.
- Device identity.
- Site.
- Sex, gender, age, race, or ethnicity.

No values were invented. OpenOximetry has a patient table with 237 rows and 5 columns according to `outputs/openox_table_row_summary.csv`, but those columns were not exported into repository metadata, so they were not used.

## Overall Results

Positive means low-oxygen risk (`y_true == 1`).

| Model | n | Prevalence | TP | FP | TN | FN | Recall | Specificity | PPV | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyKAN @ 0.50 | 52090 | 0.587118 | 23450 | 13032 | 8475 | 7133 | 0.766766 | 0.394058 | 0.642783 | 0.605942 |
| Logistic Regression @ 0.50 | 52090 | 0.587118 | 26490 | 15892 | 5615 | 4093 | 0.866167 | 0.261078 | 0.625029 | 0.738922 |

## Subject Results

There are 91 patients; 87 are eligible. Four fail `n >= 50` and/or `n_pos >= 5`; the `n_neg >= 5` criterion was also applied.

### TinyKAN Subject Distribution

| Statistic | Recall | Specificity |
|---|---:|---:|
| Min | 0.000000 | 0.006135 |
| P25 | 0.546826 | 0.188931 |
| Median | 0.840278 | 0.345528 |
| P75 | 0.982095 | 0.634261 |
| Max | 1.000000 | 1.000000 |

The largest absolute TinyKAN recall gap from overall is 0.766766 for patient `47fe963ca125f321164421dbac360d655f2bbff382cd43b32338841ee0605fc6` (subject recall 0.000000 versus overall 0.766766).

The largest absolute TinyKAN specificity gap from overall is 0.605942 for patient `898d5cec03e22881b32dcdbc0fe52c1d73dde114834b3b4d0271fcb784323d2f` (subject specificity 1.000000 versus overall 0.394058).

### Eligible Patients

Patient IDs are truncated to 12 characters for compact display; the CSV contains each full hash.

| Patient ID | n | Prevalence | TinyKAN recall | TinyKAN spec | TinyKAN PPV | TinyKAN FPR | LogReg recall | LogReg spec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 016339afef15 | 292 | 0.609589 | 0.500000 | 0.798246 | 0.794643 | 0.201754 | 0.584270 | 0.763158 |
| 03407c788533 | 705 | 0.612766 | 0.840278 | 0.304029 | 0.656420 | 0.695971 | 0.972222 | 0.080586 |
| 035135a7d427 | 435 | 0.459770 | 0.980000 | 0.089362 | 0.478049 | 0.910638 | 1.000000 | 0.042553 |
| 08e4c37178f4 | 648 | 0.611111 | 0.613636 | 0.718254 | 0.773885 | 0.281746 | 0.883838 | 0.448413 |
| 0ba2c34c5c09 | 907 | 0.615215 | 0.969534 | 0.255014 | 0.675406 | 0.744986 | 0.973118 | 0.197708 |
| 0c16265fcc7d | 653 | 0.637060 | 0.396635 | 0.784810 | 0.763889 | 0.215190 | 0.968750 | 0.253165 |
| 0c552fa00d2b | 334 | 0.631737 | 0.440758 | 0.902439 | 0.885714 | 0.097561 | 0.511848 | 0.560976 |
| 0d32db4d393f | 392 | 0.650510 | 0.996078 | 0.036496 | 0.658031 | 0.963504 | 0.992157 | 0.036496 |
| 0dd3bc09a488 | 442 | 0.622172 | 0.360000 | 0.904192 | 0.860870 | 0.095808 | 0.643636 | 0.437126 |
| 1756ffbf8b12 | 767 | 0.603651 | 0.792657 | 0.082237 | 0.568111 | 0.917763 | 0.792657 | 0.072368 |
| 19316b15f9d4 | 1211 | 0.612717 | 0.838275 | 0.251599 | 0.639260 | 0.748401 | 0.858491 | 0.228145 |
| 211843577b7f | 178 | 0.679775 | 0.909091 | 0.736842 | 0.880000 | 0.263158 | 0.942149 | 0.684211 |
| 244e495a1195 | 923 | 0.605634 | 0.540250 | 0.445055 | 0.599206 | 0.554945 | 0.640429 | 0.296703 |
| 2492a0d2b09d | 365 | 0.605479 | 1.000000 | 0.048611 | 0.617318 | 0.951389 | 1.000000 | 0.041667 |
| 2a6a349b3d7c | 291 | 0.439863 | 1.000000 | 0.006135 | 0.441379 | 0.993865 | 1.000000 | 0.006135 |
| 2d58949e03a8 | 390 | 0.458974 | 0.988827 | 0.393365 | 0.580328 | 0.606635 | 0.994413 | 0.379147 |
| 3094c7542818 | 370 | 0.654054 | 0.636364 | 0.367188 | 0.655319 | 0.632812 | 0.917355 | 0.367188 |
| 3601a3644bf7 | 1828 | 0.589168 | 0.720520 | 0.544607 | 0.694097 | 0.455393 | 0.975859 | 0.142477 |
| 36875d239b9e | 88 | 0.420455 | 0.918919 | 0.117647 | 0.430380 | 0.882353 | 0.972973 | 0.098039 |
| 371fafce9498 | 343 | 0.606414 | 0.932692 | 0.348148 | 0.687943 | 0.651852 | 0.951923 | 0.348148 |
| 375d9f4e0fbd | 1531 | 0.620509 | 0.745263 | 0.325301 | 0.643636 | 0.674699 | 0.910526 | 0.208262 |
| 417fc9ee7185 | 736 | 0.542120 | 0.997494 | 0.166172 | 0.586156 | 0.833828 | 0.997494 | 0.118694 |
| 438f089affeb | 357 | 0.605042 | 0.930556 | 0.368794 | 0.693103 | 0.631206 | 0.981481 | 0.014184 |
| 467159a4953a | 929 | 0.617869 | 0.679443 | 0.546479 | 0.707804 | 0.453521 | 0.818815 | 0.366197 |
| 47fe963ca125 | 421 | 0.456057 | 0.000000 | 0.991266 | 0.000000 | 0.008734 | 0.000000 | 0.986900 |
| 4ce39480db49 | 614 | 0.597720 | 0.961853 | 0.283401 | 0.666038 | 0.716599 | 0.964578 | 0.279352 |
| 53845746f000 | 514 | 0.659533 | 0.979351 | 0.154286 | 0.691667 | 0.845714 | 0.979351 | 0.154286 |
| 5471da0b60a1 | 981 | 0.568807 | 0.700717 | 0.394799 | 0.604328 | 0.605201 | 0.700717 | 0.385343 |
| 55475c97e020 | 300 | 0.616667 | 0.951351 | 0.495652 | 0.752137 | 0.504348 | 0.989189 | 0.434783 |
| 564f04f0c98f | 316 | 0.658228 | 1.000000 | 0.046296 | 0.668810 | 0.953704 | 1.000000 | 0.037037 |
| 57acecd444b1 | 593 | 0.625632 | 0.501348 | 0.545045 | 0.648084 | 0.454955 | 0.501348 | 0.504505 |
| 5c5cd4a7a0ce | 1806 | 0.542082 | 0.724208 | 0.406288 | 0.590833 | 0.593712 | 0.782431 | 0.268440 |
| 5ecd258265ab | 466 | 0.596567 | 0.992806 | 0.271277 | 0.668281 | 0.728723 | 0.996403 | 0.250000 |
| 61ac1a3b22be | 323 | 0.563467 | 0.406593 | 0.957447 | 0.925000 | 0.042553 | 0.780220 | 0.780142 |
| 65f53879388b | 648 | 0.560185 | 0.914601 | 0.263158 | 0.612546 | 0.736842 | 0.988981 | 0.203509 |
| 678f021f35fb | 1179 | 0.625106 | 0.830393 | 0.233032 | 0.643533 | 0.766968 | 0.989145 | 0.140271 |
| 681eef9b4930 | 772 | 0.589378 | 0.936264 | 0.324921 | 0.665625 | 0.675079 | 0.993407 | 0.088328 |
| 6a1f5d54f432 | 776 | 0.612113 | 0.995789 | 0.172757 | 0.655125 | 0.827243 | 0.993684 | 0.149502 |
| 6a354526b461 | 1392 | 0.556034 | 0.965116 | 0.184466 | 0.597122 | 0.815534 | 0.997416 | 0.097087 |
| 6c3858399808 | 407 | 0.538084 | 0.315068 | 0.898936 | 0.784091 | 0.101064 | 0.579909 | 0.781915 |
| 6c6cfc6b3f87 | 299 | 0.642140 | 1.000000 | 0.037383 | 0.650847 | 0.962617 | 1.000000 | 0.037383 |
| 78749fc2701f | 382 | 0.675393 | 1.000000 | 0.056452 | 0.688000 | 0.943548 | 1.000000 | 0.056452 |
| 79e33750260b | 274 | 0.602190 | 0.993939 | 0.211009 | 0.656000 | 0.788991 | 1.000000 | 0.211009 |
| 7a9caeaedf0d | 311 | 0.585209 | 0.664835 | 0.263566 | 0.560185 | 0.736434 | 0.912088 | 0.193798 |
| 850f264f00c8 | 1439 | 0.605281 | 0.523536 | 0.628521 | 0.683658 | 0.371479 | 0.936854 | 0.255282 |
| 881726c9eefc | 311 | 0.594855 | 0.745946 | 0.769841 | 0.826347 | 0.230159 | 1.000000 | 0.142857 |
| 8836e369dd47 | 294 | 0.557823 | 0.548780 | 0.838462 | 0.810811 | 0.161538 | 0.573171 | 0.761538 |
| 898d5cec03e2 | 279 | 0.591398 | 0.000000 | 1.000000 |  | 0.000000 | 0.000000 | 1.000000 |
| 8d87deadfc8c | 1107 | 0.432701 | 0.755741 | 0.335987 | 0.464698 | 0.664013 | 0.755741 | 0.307325 |
| 9270f65cc1ae | 578 | 0.653979 | 0.600529 | 0.620000 | 0.749175 | 0.380000 | 0.608466 | 0.620000 |
| 94061699f08e | 376 | 0.638298 | 0.925000 | 0.308824 | 0.702532 | 0.691176 | 0.995833 | 0.147059 |
| 940c4e957854 | 349 | 0.584527 | 0.990196 | 0.220690 | 0.641270 | 0.779310 | 0.985294 | 0.172414 |
| 953b408ad866 | 262 | 0.629771 | 1.000000 | 0.010309 | 0.632184 | 0.989691 | 0.939394 | 0.020619 |
| 9b655ba81fe2 | 379 | 0.638522 | 0.942149 | 0.284672 | 0.699387 | 0.715328 | 0.942149 | 0.109489 |
| 9d03b7f6cf59 | 442 | 0.529412 | 1.000000 | 0.389423 | 0.648199 | 0.610577 | 1.000000 | 0.240385 |
| 9d57f117c614 | 871 | 0.662457 | 0.937608 | 0.078231 | 0.666256 | 0.921769 | 0.942808 | 0.040816 |
| 9e6cd4a96a03 | 880 | 0.584091 | 0.924125 | 0.311475 | 0.653370 | 0.688525 | 0.998054 | 0.155738 |
| 9ec81da36d03 | 601 | 0.653910 | 0.984733 | 0.115385 | 0.677758 | 0.884615 | 0.697201 | 0.139423 |
| 9fe169f43ce9 | 394 | 0.642132 | 0.984190 | 0.326241 | 0.723837 | 0.673759 | 0.988142 | 0.127660 |
| a299eab7b30b | 338 | 0.588757 | 0.221106 | 0.791367 | 0.602740 | 0.208633 | 0.708543 | 0.230216 |
| a3402929569b | 290 | 0.568966 | 0.690909 | 0.096000 | 0.502203 | 0.904000 | 0.690909 | 0.096000 |
| a3cf6fd88056 | 307 | 0.475570 | 0.253425 | 0.962733 | 0.860465 | 0.037267 | 0.993151 | 0.223602 |
| a44d4afb781e | 1039 | 0.571704 | 0.942761 | 0.233708 | 0.621532 | 0.766292 | 0.981481 | 0.123596 |
| a4f9ae639dfe | 1428 | 0.607843 | 0.690092 | 0.441071 | 0.656798 | 0.558929 | 0.830645 | 0.294643 |
| a7aac8218c20 | 981 | 0.592253 | 1.000000 | 0.035000 | 0.600827 | 0.965000 | 0.996558 | 0.035000 |
| b01779ab7380 | 357 | 0.585434 | 0.019139 | 0.817568 | 0.129032 | 0.182432 | 0.119617 | 0.581081 |
| b150d044b396 | 347 | 0.688761 | 0.489540 | 0.129630 | 0.554502 | 0.870370 | 0.665272 | 0.046296 |
| b339cd40b224 | 503 | 0.542744 | 0.476190 | 0.769565 | 0.710383 | 0.230435 | 0.824176 | 0.534783 |
| b489b1a5785a | 551 | 0.626134 | 0.559420 | 0.500000 | 0.652027 | 0.500000 | 0.559420 | 0.495146 |
| b5a3bd8e6aa5 | 298 | 0.607383 | 0.966851 | 0.529915 | 0.760870 | 0.470085 | 0.966851 | 0.538462 |
| b74602793967 | 1200 | 0.522500 | 0.789474 | 0.568935 | 0.667116 | 0.431065 | 0.923445 | 0.364747 |
| ba38ee8d1dcc | 405 | 0.592593 | 0.966667 | 0.042424 | 0.594872 | 0.957576 | 0.966667 | 0.042424 |
| c16756597f48 | 605 | 0.515702 | 0.528846 | 0.675768 | 0.634615 | 0.324232 | 0.528846 | 0.645051 |
| c8e741845e21 | 312 | 0.538462 | 1.000000 | 0.006944 | 0.540193 | 0.993056 | 1.000000 | 0.006944 |
| cbc584babd2c | 429 | 0.505828 | 1.000000 | 0.193396 | 0.559278 | 0.806604 | 1.000000 | 0.174528 |
| d647774dc40c | 616 | 0.629870 | 0.940722 | 0.350877 | 0.711501 | 0.649123 | 0.951031 | 0.258772 |
| d80e089b2c92 | 258 | 0.565891 | 0.000000 | 0.839286 | 0.000000 | 0.160714 | 0.136986 | 0.508929 |
| e57edb5300ce | 362 | 0.596685 | 0.532407 | 0.726027 | 0.741935 | 0.273973 | 0.787037 | 0.356164 |
| ede5b74f91ab | 1260 | 0.619841 | 0.924456 | 0.194154 | 0.651625 | 0.805846 | 0.925736 | 0.187891 |
| eec335ecb0d0 | 449 | 0.452116 | 0.985222 | 0.345528 | 0.554017 | 0.654472 | 1.000000 | 0.288618 |
| f0feb4ab2e48 | 402 | 0.539801 | 0.995392 | 0.113514 | 0.568421 | 0.886486 | 1.000000 | 0.032432 |
| f298d5aa52c9 | 1019 | 0.628067 | 0.814063 | 0.345646 | 0.677503 | 0.654354 | 0.998437 | 0.171504 |
| f46f5af889e8 | 322 | 0.518634 | 0.131737 | 0.877419 | 0.536585 | 0.122581 | 0.988024 | 0.232258 |
| f59ffa3be7f1 | 249 | 0.626506 | 0.544872 | 0.376344 | 0.594406 | 0.623656 | 0.929487 | 0.236559 |
| f9d13b9c4162 | 675 | 0.589630 | 0.582915 | 0.718412 | 0.748387 | 0.281588 | 0.748744 | 0.447653 |
| ff9b6a312ed7 | 264 | 0.621212 | 0.481707 | 0.640000 | 0.686957 | 0.360000 | 0.542683 | 0.560000 |
| ffcfe0f5b1bd | 306 | 0.532680 | 0.993865 | 0.195804 | 0.584838 | 0.804196 | 0.993865 | 0.181818 |

## Recording Results

There are 162 recordings; 155 are eligible.

| Statistic | TinyKAN recall | TinyKAN specificity |
|---|---:|---:|
| Min | 0.000000 | 0.006135 |
| Median | 0.915254 | 0.330645 |
| Max | 1.000000 | 1.000000 |

The 155 eligible recording rows, including full recording hashes and both models' confusion matrices and rates, are in `outputs/openox_tinykan_fairness_subgroups.csv`.

## Honest Limits

- This analysis measures subject and recording heterogeneity. It is not a pigmentation fairness audit.
- Windows within a patient are not independent observations.
- The 91-subject sample is small for equity claims.
- No causal claim is made.
- The fixed 0.50 operating point has high FPR by design; see `docs/operating-point-calibration.md`.
- Recall of 0 or specificity near 0 for some subjects can reflect score shift or prevalence, not bias.
- The Logistic Regression comparison uses existing probabilities and is not a new training run.
- The XAI rubric and sample were not expanded.

## Paths

`outputs/openox_tinykan_fairness_subgroups.csv` (regenerated locally; not redistributed)

`docs/tinykan-fairness-subgroups.md`

# Panel 10 Nov 2026 addenda: host analyses A1 to A4 (10 Nov 2026)

**Host only.** A1 and A2 are in-sample on the shipped header. A3 uses five fresh recording-fold analysis models. A4 scores the stored out-of-fold probabilities. Nothing shipped changes.

- Pre-registration: `docs/panel-2026-11-10-addenda-preregistration.md`, with amendments A1 and A2. It was written before the run but **not committed before it**; its ordering rests on the file, not on a commit.
- Script: `scripts/analyze_panel_addenda_20261110.py` (`py -3.14`). Input and code SHA-256 in `summary.json`. Protected inputs hashed before and after, unchanged.
- Bootstrap: 2,000 resamples of whole patients (87), 95 % percentile intervals, seed 20261110 + cell index.
- Answers roadmap items R3, R4 and R8 of `docs/ieee-draft-panel-review-2026-11-10/`.

## Two stopped runs (nothing written by either)

1. M0.2 expected 11,381 candidates at the output clamp. 11,368 are; 11,381 is the gradient-all-zero count (those plus 13 with every hidden unit clamped). Amendment A1 fixed the check, not the stratum.
2. M0.3 expected the retrained fold models to match the stored out-of-fold table within 1e-5. Retraining this model is documented as not bit-reproducible (`docs/analysis-lamaakal-style.md`, Provenance item 5). Amendment A2 replaced the tolerance with that documented envelope. This run: maximum |dP| 0.049, 41 candidate flags differ, AUC difference -1.1e-4.

## A1 and A3: deletion-AOPC gain over random, with intervals

Gain = firmware-absolute minus the mean of five random orders, per window, averaged. Candidates are the primary subset.

| Model | Reference | Candidates | Firmware-absolute | Random | Gain [95 % CI] | Reading |
|---|---|---|---|---|---|---|
| Shipped, in-sample | training mean (shipped) | 28,661 | -0.0085 | -0.0004 | -0.0081 [-0.0127, -0.0014] | worse than random |
| Fold models, out-of-fold | training mean | 28,356 | -0.0077 | -0.0013 | -0.0064 [-0.0123, +0.0011] | not distinguishable from random |
| Shipped, in-sample | label-0 median | 28,661 | 0.7745 | 0.4138 | +0.3607 [+0.3566, +0.3643] | beats random |
| Fold models, out-of-fold | label-0 median | 28,356 | 0.7783 | 0.4128 | +0.3655 [+0.3607, +0.3694] | beats random |

On all windows the training-mean ranking is worse than random in sample (-0.148 [-0.157, -0.139]) and out of fold (-0.148 [-0.158, -0.137]); the label-0 ranking beats random both ways (+0.220, +0.218). Every cell is in `aopc_gain_intervals.csv`.

**Reading (pre-registered).** Under the shipped reference the ranking does not beat random on the windows it flags, in sample or out of fold. In sample the interval lies below zero, so "worse than random" is the stricter wording there; out of fold it includes zero. The size is small either way (under 0.013 in probability). The label-0 result is not an in-sample artefact: it holds out of fold with the reference taken from the training folds.

## A2: how much of the flagged subset the test can discriminate on (shipped model, in-sample)

| Stratum | Reference | Windows | Firmware-absolute | Random | Gain [95 % CI] | Reading |
|---|---|---|---|---|---|---|
| Candidates at the output clamp | training mean | 11,368 | +0.0103 | +0.0023 | +0.0080 [-0.0002, +0.0218] | not distinguishable |
| Candidates not at the clamp | training mean | 17,293 | -0.0209 | -0.0022 | -0.0187 [-0.0224, -0.0144] | worse than random |
| Candidates at the output clamp | label-0 median | 11,368 | 0.7950 | 0.4268 | +0.3682 [+0.3639, +0.3721] | beats random |
| Candidates not at the clamp | label-0 median | 17,293 | 0.7611 | 0.4053 | +0.3558 [+0.3498, +0.3607] | beats random |

- Under the training mean, 48.3 % of candidates have a largest |delta| below 0.01 (24.3 % of the unclamped ones). No candidate has all sixteen deltas exactly zero.
- **Reading (pre-registered).** The "does not beat random" verdict is not only a clamp artefact. It is on the unclamped candidates that the shipped ranking is worse than random; on the clamped ones the test has almost no range and cannot tell the two apart.

## A4: RQ1 metrics with the label taken L seconds away

Stored out-of-fold probability of each window scored against the label of the window of the same recording that starts L seconds later (negative L: earlier). "Lag 0" columns are the ordinary metrics on the same paired windows.

| L (s) | Pairs | Labels changed | AUC lag 0 -> lagged | Difference [95 % CI] | Recall difference (points) | Specificity difference (points) |
|---|---|---|---|---|---|---|
| -30 | 44,738 | 5.6 % | 0.906 -> 0.929 | +0.022 [+0.017, +0.028] | +0.6 | +3.5 |
| **-20** | 45,541 | 3.6 % | 0.908 -> 0.931 | +0.023 [+0.020, +0.027] | +0.8 | +3.7 |
| -10 | 46,610 | 1.6 % | 0.910 -> 0.925 | +0.014 [+0.013, +0.016] | +0.5 | +2.3 |
| +10 | 46,610 | 1.6 % | 0.912 -> 0.896 | -0.015 [-0.017, -0.014] | -0.5 | -2.3 |
| **+20** | 45,541 | 3.6 % | 0.915 -> 0.881 | -0.034 [-0.038, -0.030] | -1.6 | -4.8 |
| +30 | 44,738 | 5.6 % | 0.920 -> 0.866 | -0.054 [-0.060, -0.047] | -3.0 | -7.3 |

Full table (ten lags, all intervals): `label_lag_sensitivity.csv`.

**Reading (pre-registered).** Not insensitive. A 20 s shift moves AUC by +0.023 or -0.034, specificity by +3.7 or -4.8 points and recall by +0.8 or -1.6 points, every interval excluding zero. The scores agree better with the reference from 15 to 20 s **earlier** than with the header-aligned one, the direction a lung-to-finger delay would give, and the size of the 20 s peak in `outputs/label-alignment-v1/`. No headline number is replaced: the residual offset per recording is not known, and this is a sensitivity bound, not a corrected label.

## Limits

- A1 and A2 are in-sample; the label-0 median is taken from the same rows.
- A3's fold models are not the shipped model and are not the models behind the stored out-of-fold table. 32 of 87 patients span recording folds.
- A4 moves labels by whole strides between kept windows. Windows with a median in 89 to 91 were dropped at export, so pairs near the cut are under-represented.
- Deletion AOPC is a functionally grounded test on host arithmetic. It is not board XAI and not a human-grounded test.
- The pre-registration was not committed before the run.

# Panel 10 Nov 2026 addenda: host analysis pre-registration (10 Nov 2026)

Written **before** any number in "New measurements" was computed. Host only, laptop, no board, no firmware or header writes. Nothing shipped changes: the TinyKAN header, its 0.50 threshold, N = 2 and the training-mean reference stay as they are.

**Honest limit of this pre-registration.** It was written by Claude in the session that then ran the script, and it is not committed before the run (Jason had not asked for a commit). The file's timestamp is the only ordering evidence. Treat the readings as fixed-in-advance by intent, not by a commit hash.

## Why

The 10 Nov 2026 simulated panel (`docs/ieee-draft-panel-review-2026-11-10/`) scored methodology as a repairable block. Four of its must-fix items ask for numbers the repo does not hold:

| Roadmap | Ask | Measurement here |
|---|---|---|
| R3 (Meth W2) | uncertainty and an out-of-fold counterpart for the RQ2 deletion-AOPC figures | A1, A3 |
| R4 (Meth W3, Persp W2) | how much of the flagged subset the deletion test can discriminate on | A2 |
| R8 (Meth W6) | propagate residual label-timing error into the RQ1 metrics | A4 |

R5, R6 (patient-level intervals) and R10 (label against arterial SaO2) are already answered by `outputs/patient-bootstrap-v1/` and `outputs/scalco2-vs-sao2-v1/` and are not re-run.

## New measurements (fixed now)

Script `scripts/analyze_panel_addenda_20261110.py`, new, importing from `analyze_lamaakal_style`, `analyze_physio_concept`, `analyze_xai_reference_label0`, `lamaakal_models`, `lamaakal_metrics`, `train_tinykan_v2` and editing none. Output `outputs/panel-2026-11-10-addenda-v1/` (new, write-once). Rows: all 48,662 windows of `outputs/openox-v2/window_features.csv`. Threshold 0.50. Bootstrap everywhere: 2,000 resamples of whole patients (87), 95 % percentile intervals, seed 20261110 + cell index.

"Gain" is the per-window deletion AOPC of the firmware-absolute ranking minus the mean of the five random orders (seed 20260911), the same per-window quantities as `analyze_xai_reference_label0.aopc_deletion`. Two references, each with its own deletion target and its own ranking: the shipped training mean (z = 0) and the label-0 median (FP32).

**M0 same-code checks, run first; the script stops before writing a number if one fails.**
1. In-sample subset means from this script's per-window arrays reproduce `outputs/xai-reference-label0-v1/aopc_deletion.csv` (v2 rows, both references, firmware_absolute and random_mean, five subsets) within 1e-8.
2. Candidates with output pre-activation at or above +3 number 11,381 of 28,661 (the 39.7 % in `docs/xai-gradient-check.md`).
3. A3 only: the five retrained fold models reproduce the stored out-of-fold probabilities in `outputs/openox-v2/predictions_b_v2_tinykan16.csv` within 1e-5 (torch arithmetic), using that file's fold column.

**A1. In-sample uncertainty (shipped model).** Patient-bootstrap interval of the mean gain on candidates (primary) and on all windows (secondary), per reference.

**A2. Dynamic range on candidates (shipped model).** Candidates split into clamped (output pre-activation >= +3) and unclamped. Per stratum and reference: firmware-absolute and random AOPC, mean gain with its interval, and the fraction of windows better than their own random mean. Also, for the training-mean reference: the share of candidates whose 16 deltas are all exact zeros (|delta| <= 1e-12), and the share whose largest |delta| is below 0.01.

**A3. Out-of-fold counterpart.** For each of the five recording folds, an analysis model is fitted on the training folds with the shipped recipe (`train_tinykan_v2.train_kan`, seed 42, scaler on the training folds). Held-out windows are explained by that fold's model with (a) the fold's training mean and (b) the FP32 label-0 median of the fold's training rows. Candidates are held-out windows with fold-model P >= 0.50 (host FP32-constant arithmetic, `ArrayKAN`). Per-window gains are pooled over the five folds; same subsets and intervals as A1. These are analysis models, not the shipped model.

**A4. Label-timing sensitivity of RQ1.** For lag L in -30, -20, -15, -10, -5, +5, +10, +15, +20, +30 s, each window is paired with the window of the same recording whose start time is L later (stride 5 s). The stored out-of-fold probability of the first is scored against the label of the second. Reported per L, on the paired windows only: n, AUC, recall and specificity at 0.50, the same three at L = 0 on those same windows, and the difference with a patient-bootstrap interval. The lag scan in `outputs/label-alignment-v1/` peaked 20 s from header alignment, so +/-20 s are the primary cells.

## Readings (fixed now)

- **R-A1 / R-A3 (per reference, candidates):** "beats random" if the gain interval lies above zero; "worse than random" if it lies below zero; otherwise "not distinguishable from random". The paper's RQ2 answer for the shipped reference is restated with whichever of the three the in-sample interval gives, and the out-of-fold cell is reported beside it whether or not it agrees.
- **R-A2:** descriptive. If the unclamped-candidate gain under the training mean is also not above zero, the "does not beat random" verdict is not only a clamp artefact; if it is above zero, the paper must say the verdict is driven by the clamped share.
- **R-A4:** descriptive. A shift of the primary cells' AUC by less than 0.01 and of recall and specificity by less than 1 point is reported as "insensitive at this resolution"; anything larger is reported with its size and direction. No headline number is replaced.

## Limits known in advance

- A1 and A2 are in-sample: the shipped model was fitted on these rows, and the label-0 median is taken from them.
- A3's fold models are not the shipped model. Patient overlap across recording folds (32 of 87 patients) remains.
- A4 moves labels by whole strides between kept windows. Windows whose median lay in 89 to 91 were dropped at export, so pairs near the cut are under-represented, and the unknown per-recording offset is not estimated. It bounds sensitivity; it does not correct the labels.
- A patient bootstrap corrects the interval for clustering, nothing else.

## Amendment A1 (10 Nov 2026, after the first run stopped; before any A1 to A4 number)

The first run stopped at M0.2 and wrote nothing: 11,368 candidates have an output pre-activation at or above +3, not 11,381. A diagnostic on the check itself showed why. The 11,381 in `docs/xai-gradient-check.md` counts candidates whose gradient is exactly zero: the 11,368 at the output clamp plus 13 more whose twelve hidden pre-activations are all clamped (14 such candidates, one of them also at the output clamp). M0.2 as written compared two different quantities.

- M0.2 is replaced by: candidates number 28,661, and candidates with (output pre-activation >= +3, or <= -3, or every hidden pre-activation clamped) number 11,381.
- The A2 stratum is unchanged: clamped means output pre-activation >= +3, as pre-registered. It holds 11,368 candidates (39.7 % of 28,661, the same rounded share).
- No reading, subset, seed or other rule changed. The only numbers seen before this amendment are the four counts above.

## Amendment A2 (10 Nov 2026, after the second run stopped at M0.3; before any A1 to A4 number)

The second run passed M0.1 and the amended M0.2, trained the five fold models, and stopped at M0.3 without writing anything: the retrained fold models differ from the stored out-of-fold probabilities by up to 0.031, against the pre-registered 1e-5. That tolerance was wrong on a fact already in the repo. `docs/analysis-lamaakal-style.md` (Provenance item 5) records that retraining the Header-aligned model reproduces the stored predictions bit for bit on some folds only, with a maximum |dP| of 0.041, 17 differing candidate flags and an AUC change of -1.3e-4, from float-level sensitivity of training to process state. I had not read that before writing M0.3.

- M0.3 is replaced by the documented envelope: maximum |dP| at most 0.05, at most 50 differing candidate flags, and |AUC difference| at most 0.001 against the stored out-of-fold table.
- A3 is unchanged otherwise. Its models are fresh analysis models of the same recipe and folds, not the models behind the stored table; the output states the three differences.
- The only numbers seen before this amendment are the per-fold maximum |dP| values printed by the stopped run (0.025 to 0.031).

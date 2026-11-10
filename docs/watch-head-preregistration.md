# Pre-registration: below-95 Watch head beside the shipped below-90 head (7 Nov 2026)

Written and committed before any model in this study is trained. Host only, laptop, no board. Results go to the final report only. The IEEE paper is not edited. No firmware is built unless the gate below passes. Named by Jason on 7 Nov 2026.

## What this is and is not

- It **is** a second binary TinyKAN head on the same 16 v2 inputs, label `median_spo2 < 95`. On device it raises a **Watch** state only when the shipped below-90 head does not flag.
- It is **not**:
  - a saturation reading;
  - a diagnosis of "mild hypoxaemia";
  - a change to the shipped head, its 0.50 threshold, N=2, inputs, recipe or explanation.
- The label is **ScalcO2** (SaO2 estimated from end-tidal O2). The CSV column is named `median_spo2`, but it is not pulse-oximeter SpO2. WHO's thresholds (below) are SpO2 thresholds; cite them as clinical context only.
- **The IEEE paper is unaffected.** Do not edit the vault docx. r2 board evidence (parity 1,243/1,243, 444.3 ms, the peak-RAM bracket 86,576–87,348 B, the sensor-2 sit) stays attached to r2. It does not describe any new image.
- **No device explanation for Watch.** The 17-pass mean-perturbation explainer stays on the below-90 head only. A Watch state shows no bars. That keeps the timing cost to one extra forward pass and avoids shipping a second, untested explanation.

## Question

Can a second TinyKAN head on the v2 inputs flag median 91–94.9 (M) against ≥95 (N) well enough to ship a Watch state, without touching the shipped below-90 head?

## Wording rules (fixed now)

- The state is called **Watch**, or the "below-95 pattern".
- Never "mild hypoxaemia". Never "SpO2" or "oxygen level" for the label. The label is window-median ScalcO2.
- "Recording-grouped" means a patient can appear in both training and test folds. It is the primary grouping because it is the grouping of every reported v2 number. Patient-grouped is the check on it (G3).
- Head A = the shipped below-90 head, read from its stored out-of-fold vectors. Head B = the new below-95 head.

## Clinical context (WHO; checked 7 Nov, quotes verified verbatim against the fetched PDFs)

- [1] WHO, *Pulse Oximetry Training Manual* (2011), https://cdn.who.int/media/docs/default-source/patient-safety/pulse-oximetry/who-ps-pulse-oxymetry-training-manual-en.pdf
  - "A healthy individual with normal lungs, breathing air at sea level, will have an arterial oxygen saturation of 95% – 100%."
  - "If the oxygen saturation is 94% or lower, the patient is hypoxic and needs to be treated quickly. A saturation of less than 90% is a clinical emergency."
- [2] WHO, *Oxygen therapy for children* (2016), https://iris.who.int/handle/10665/204584
  - "The normal range of SpO2 at sea level is 97–99%, with a lower limit (mean minus 2 standard deviations) of 94%."
  - "In practice, the threshold at which oxygen is given is often SpO2 < 90% …"
  - "Small reductions in SpO2 below 90% may represent a dangerous fall in PaO2 (steep part of the curve)."
- [3] WHO, *Clinical management of COVID-19: living guideline*, 18 Aug 2023, https://iris.who.int/handle/10665/372288
  - "The oxygen saturation threshold of 90% to define severe COVID-19 is arbitrary and should be interpreted cautiously."
  - "… a saturation between 90–94% on room air may be abnormal (in patient with normal lungs) and can be an early sign of severe disease, mainly if patient is on a downward trend."

These justify why a 91–94.9 band is worth a separate state. They do not make the Watch state hypoxaemia: the label is ScalcO2, and the band is defined by a window median.

## Already seen before this pre-registration (scratch probe, unreviewed)

The probe was run 7 Nov on the same folds. A copy of the script is at `.hermes/plans/2026-11-07-watch-head-probe.py`. It used only existing out-of-fold vectors plus sklearn LogReg/HistGB retrained on the 16 inputs. **No TinyKAN was trained for this question.** The bootstrap was 500 resamples, seed 20261107, not the convention.

Shipped head (P_A, out-of-fold) flag rate at 0.50 by band:

| Band | n | Flagged | Median P |
|---|---|---|---|
| [91, 92) | 1,498 | 0.177 | 0.175 |
| [92, 93) | 2,142 | 0.094 | 0.130 |
| [93, 94) | 1,910 | 0.069 | 0.110 |
| [94, 95) | 2,235 | 0.145 | 0.108 |
| [95, 96) | 3,123 | 0.073 | 0.084 |
| [96, 97) | 2,519 | 0.054 | 0.080 |
| [97, 101) | 6,440 | 0.207 | 0.083 |

Task M vs N, on windows with median ≥ 91 (n = 19,867, M share 0.392):

| Score | Grouping | AUC [CI] | Recall / Spec @0.50 |
|---|---|---|---|
| P_A used as a score | recording | 0.673 [0.635, 0.707] | — |
| P_A used as a score | patient | 0.678 | — |
| LogReg-16 retrained | recording | 0.541 [0.510, 0.574] | 0.026 / 0.984 |
| HistGB-16 retrained | recording | 0.798 [0.767, 0.826] | 0.614 / 0.847 |
| HistGB-16 retrained | patient | 0.790 | 0.586 / 0.849 |
| `ratio_of_ratios` alone | — | 0.691 | — |

On all windows, <95 vs ≥95: HistGB-16 reached AUC 0.879, and P_A used as a score reached 0.855.

**Expectation, stated before training:** even the 12,200-node ceiling misses recall 0.80 at 0.50. TinyKAN usually lands below HistGB (0.908 vs 0.931 on the main task). **A gate fail is the likely outcome**, and a fail is a reportable result.

## Data, labels and folds (frozen inputs; SHA-256 recorded in summary.json)

Re-checked on disk 7 Nov 2026 (`py -3.14`, sklearn 1.9.0). Each item is asserted at the start of the run, and the run stops if one fails.

- Rows: `outputs/openox-v2/window_features.csv`. 48,662 rows, 154 recordings, 87 patients. `label == (median_spo2 < 90)` holds on every row. `median_spo2` ranges from 3.96 to 100.0.
- Bands by `median_spo2`:

  | Band | Rule | Windows | Recordings | Patients |
  |---|---|---|---|---|
  | L | < 89 | 28,795 | 149 | 85 |
  | (dropped at export) | [89, 91) | 0 | — | — |
  | M | [91, 95) | 7,785 | 151 | 86 |
  | N | ≥ 95 | 12,082 | 154 | 87 |

- New label: `y95 = (median_spo2 < 95)` as int64, on all 48,662 rows, no drop band. `y95 == 1` on 36,580 of 48,662 (0.7517).
- S = windows with `median_spo2 >= 91` = M ∪ N, n = 19,867. On S, `y95 == M`.
- Windows with `median_spo2` in [94, 96] (closed): 5,358. With them removed, S keeps 5,550 M and 8,959 N.
- Head A out-of-fold vectors (never retrained): `outputs/openox-v2/predictions_b_v2_tinykan16.csv` (recording-grouped) and `predictions_b_v2_tinykan16_patient.csv` (patient-grouped). Columns row_index, recording_id, patient_id, start_time_seconds, fold, y_true, y_prob. After sorting by row_index, row_index == 0..48661, `recording_id` equals the table, and `y_true == label`. P_A_rec = `y_prob` of the recording file; P_A_pat = `y_prob` of the patient file.
- Folds: recording-grouped = `fold` of the recording file; patient-grouped = `fold` of the patient file. No recording spans two recording folds, and no patient spans two patient folds. `GroupKFold(5)` on `recording_id` (read as str) and on `patient_id` regenerates both stored fold columns exactly (checked 7 Nov, no training).
- Every fold restricted to S holds both M and N. Recording folds 1–5 (M, N): (1,448, 2,611), (1,735, 2,252), (1,455, 2,495), (1,511, 2,380), (1,636, 2,344). Patient folds: (1,730, 2,304), (1,549, 2,494), (1,354, 2,521), (1,488, 2,520), (1,664, 2,243).
- Head A at 0.50 on all windows against `label` must reproduce, from the stored vectors: recording recall 0.89269, specificity 0.86802, AUC 0.907528; patient recall 0.89932, specificity 0.86354, AUC 0.906936. Compared to 5 decimals (AUC to 6). The run stops if they differ.
- Shipped header, for reference only: `firmware/Arduino Nano 33/include/kan_model_weights_v2.h` defines un-prefixed `SCALER_MEAN`, `KAN1_W`, …, `KAN_DECISION_THRESHOLD = 0.50f`. It is read, never written.

## Training mechanisms (fixed now)

The v2 recipe throughout: 16 inputs `train_tinykan_v2.V2_FEATURES`, hidden 12, 8 knots, [−3, 3], cubic, Adam lr 0.01, weight decay 1e-4, 100 epochs, pos_weight BCE, smoothness 1e-5, grad clip 10, full batch, StandardScaler fitted on the training fold only.

Every TinyKAN job runs in a freshly spawned process (`ProcessPoolExecutor(max_tasks_per_child=1)`, as `analyze_physio_concept.run_jobs_cached`, `scripts/analyze_physio_concept.py:517-543`). Finished jobs are cached under `build/watch-head/jobs/`, keyed by job kind, spec and input arrays.

1. **Reproduction check first (stage `repro`; the run stops if it fails).**
   - Job: read `train_tinykan_v2.V2_CSV`, cast `recording_id` to str, call `train_tinykan_v2.out_of_fold(frame, V2_FEATURES, "recording_id", "tinykan")` on the original `label`. This is `analyze_physio_concept._repro_job` (`scripts/analyze_physio_concept.py:451-458`).
   - Pass: the returned folds equal the stored recording `fold` column, and the AUC against `label` is within 0.002 of the stored vector's 0.907528.
   - Bitwise identity with the stored `y_prob` is reported, not required. Also reported: max abs probability difference overall and per fold, and the number of candidate flags at 0.50 that differ.
   - Expected from precedent: not bitwise. The physio study's same check gave AUC 0.907399, max abs difference 0.0406 (folds 1–2 only; folds 3–5 ≤ 1.2e-16) and 17 flag flips (`outputs/physio-concept-v1/s2_reproduction.json`).
2. **Primary head B (stage `cv`).**
   - `out_of_fold` reads only `frame["label"]` for the target, plus the feature and group columns (`scripts/train_tinykan_v2.py:170-187`). `train_kan` hard-codes `torch.manual_seed(SEED)` with `SEED = 42` (`scripts/train_tinykan_v2.py:44`, `:140`), so head B is seed 42 by construction. `torch.set_num_threads(1)` is set when the module is imported (`scripts/train_tinykan_v2.py:64`).
   - Job: read `V2_CSV` inside the job, cast `recording_id` to str, make `copy = frame.copy()`, set `copy["label"] = (copy["median_spo2"] < 95).astype(np.int64)`.
   - Assert, before training: `copy.drop(columns="label").equals(frame.drop(columns="label"))`; same column order and index; `copy["label"]` sums to 36,580; `frame["label"] == (median_spo2 < 90)` still holds on every row.
   - Two calls, each its own job: `out_of_fold(copy, V2_FEATURES, "recording_id", "tinykan")` → P_B_rec; `out_of_fold(copy, V2_FEATURES, "patient_id", "tinykan")` → P_B_pat. `patient_id` is used as read (str), as in `train_tinykan_v2.run_cv` (`:211-222`).
   - Assert the returned fold vectors equal the stored `fold` columns (recording file for P_B_rec, patient file for P_B_pat). The run stops if not.
3. **Seeds 42, 1, 2, 3, 4, 5 (stage `seeds`; recording-grouped only).**
   - `out_of_fold` cannot take a seed. The six seeds are trained through `lamaakal_models.run_cv_job(spec, x, y95, fold_rec)` (`scripts/lamaakal_models.py:131-161`), with `spec = {learner: "tinykan", hidden: 12, knots: 8, spline_range: [-3, 3], epochs: 100, seed: s}`. That calls `_tinykan_fold` → `train_tinykan(..., seed)` (`:51-84`), which sets `torch.manual_seed(seed)` before model construction. This is the path of `analyze_lamaakal_style` section C (`scripts/analyze_lamaakal_style.py:262`) and of the physio study's seed runs (`analyze_physio_concept._cv_job`, `:421-448`, there with imputation, which this study does not need).
   - `x = frame[V2_FEATURES].to_numpy(np.float32)` (as `out_of_fold` line 171). `fold_rec` = the stored recording fold column.
   - **Same-path check:** seed 42 through this path is compared with the primary P_B_rec. Reported: bitwise flag, max abs difference, and AUC-on-S difference. Tolerance: max abs difference ≤ 1e-5 (`REPRODUCTION_TOLERANCE`, `scripts/analyze_lamaakal_style.py:121`). Precedent: on the original label, this path and the `out_of_fold` path gave the same per-fold differences from the stored vector (0.0405773, 0.0258923, ≤ 1.2e-16 × 3; `outputs/analysis-lamaakal-style-v1/summary.json` `checks.seed_42_reproduces_frozen_oof.v2` vs `s2_reproduction.json`), so agreement is expected.
   - If the tolerance fails, the run does not stop (seeds are not gated). The seed table is still written, labelled "not the same path as the primary head", and the spread is not read as head B's spread.
4. **Secondary head B_cond (stage `secondary`; reported, never gated).**
   - Trained only on S, label M (= `y95` on S). Seed 42, same recipe, through `lamaakal_models.run_cv_job(spec, x[S], y95[S], fold[S])`.
   - Folds are the **stored** fold columns restricted to S. GroupKFold is not re-run on the subset. The scaler and pos_weight are fitted on the S training rows of each fold.
   - Because the stored folds are grouped, no S window is scored by a model that saw its recording (recording-grouped, `fold_rec[S]`) or its patient (patient-grouped, `fold_pat[S]`). Both are run: B_cond_rec and B_cond_pat. Outside S they are NaN.
5. **Comparators LogReg-16 and HistGB-16 on `y95` (stage `secondary`; reported, never gated).**
   - `lamaakal_models.run_cv_job` with learner `"logreg"` (`BASELINE_SETTINGS["logreg"]`) and `"hist_gb"` (`BASELINE_SETTINGS["hist_gb"]`), all 48,662 windows, `y95`, stored recording folds. The scaler is fitted per training fold on float64, as `run_cv_job` does before `_baseline_fold` (`scripts/lamaakal_models.py:153-155`, `:87-128`).
   - Each runs in its own spawned process, like the TinyKAN jobs.
6. **Outputs are write-once.** Output dir `outputs/watch-head-v1/`. Every file goes through a `new_file`-style guard that refuses an existing file (`scripts/analyze_physio_concept.py:124-128`). `summary.json` records the SHA-256 of `window_features.csv`, both stored prediction files, the scripts, the tests and this pre-registration, plus versions, seeds and every check.

## State rule (per window, on out-of-fold P)

- `LOW` if P_A ≥ 0.50; else `WATCH` if P_B ≥ 0.50; else `NONE`.
- Recording-grouped: P_A_rec with P_B_rec. Patient-grouped: P_A_pat with P_B_pat. P_A is always the stored vector, never retrained.
- alerted = state ∈ {WATCH, LOW}. Equivalently, alerted ⟺ max(P_A, P_B) ≥ 0.50, so `alert = max(P_A, P_B)` is the score used for bootstrap CIs of the state rule.
- The N=2 confirmation is a stream property. It is not scored at window level (same as the paper).

## Ship gate (fixed now; 0.50 only)

On S (n = 19,867), truth = M:

- recall = #(M and alerted) / #M
- specificity = #(N and NONE) / #N

Gate quantities are computed in float64 and compared unrounded. ≥ and > are as written.

- **G1 (recording-grouped):** recall ≥ 0.80 **and** specificity ≥ 0.50, using P_A_rec and P_B_rec. This is the project's ETPB dual gate.
- **G2:** paired recording bootstrap. `lamaakal_metrics.group_bootstrap(y=M[S], {"P_B_rec": P_B_rec[S], "P_A_rec": P_A_rec[S], "alert_rec": max(P_A_rec, P_B_rec)[S]}, groups=recording_id[S], n_boot=2000, seed=20261025, threshold=0.50)`. delta_b = AUC_b(P_B_rec) − AUC_b(P_A_rec) per resample. The 95% percentile CI (`lamaakal_metrics.percentile_ci`, 2.5th and 97.5th) of delta must have **lower bound > 0**. The point difference and the share of resamples with delta ≤ 0 are reported. Non-finite draws, if any, are counted and reported. The new head must rank M against N better than reusing the shipped score (probe: P_A 0.673).
- **G3 (patient-grouped):** G1's two point estimates also hold using P_A_pat and P_B_pat: recall ≥ 0.80 and specificity ≥ 0.50.
- Patient CIs (reported, not gated): the same `group_bootstrap` call on S with P_B_pat, P_A_pat, `alert_pat`, groups = `patient_id[S]`, seed 20261026.

**Pass = G1 ∧ G2 ∧ G3.** Anything else is a fail. Threshold sweeps are descriptive only. On a fail there is no final model and no firmware, and the negative result goes to the final report.

## Reported, not gated

- **3×3 table.** Rows true band L / M / N, columns state LOW / WATCH / NONE, on all 48,662 windows (L + M + N = 48,662). Counts and row shares. Recording-grouped and patient-grouped.
- **False-Watch rate on N** = #(N and WATCH) / #N, with the LOW share #(N and LOW) / #N beside it. Both groupings. (Specificity on N = 1 − the two.)
- **Head A on all windows** against `label` at 0.50, from the stored vectors: must reproduce 0.893 / 0.868 (recording; 0.89269 / 0.86802) and 0.89932 / 0.86354 (patient). Stop if not (checked in stage `repro`).
- **P_B AUC on S** (M vs N) with its bootstrap CI, both groupings (seeds 20261025 / 20261026). Also P_B AUC on all windows against `y95`.
- **Six seeds** (42, 1, 2, 3, 4, 5; recording-grouped): mean ± sd (`lamaakal_metrics.mean_sd`) of AUC on S, G1 recall and G1 specificity (state rule with P_A_rec and that seed's P_B).
- **LogReg-16 and HistGB-16 on `y95`**: AUC on S with CI (same recording bootstrap call, so paired), and G1 recall and specificity with the same state rule, the comparator's P substituted for P_B. AUC on all windows against `y95`.
- **B_cond** (recording and patient): AUC on S, and recall / specificity on S with B_cond substituted for P_B in the state rule.
- **Sensitivity.** Every gate quantity (G1 recall and specificity, G2 point and CI with the same seed, G3 recall and specificity) recomputed with windows whose `median_spo2` is in [94, 96] (closed) removed. Re-scoring of the same out-of-fold vectors only; no retraining. It mirrors the [89, 91] gap at the 90 cut.

## Limits fixed in advance

- ScalcO2 label noise against a 4-point band. The Stage 4 concept regressor's RMSE on windows ≥ 90 is 6.0407 with bias −2.8786 (`outputs/physio-concept-v1/s4_concept_quality.csv`, row `kan_p5` / recording / all; n_ge_90 = 19,867, which is S, since the export dropped [89, 91)).
- No drop band at 95. The 90 cut has a dropped [89, 91) band; the 95 cut has none. The [94, 96] sensitivity is the only check on this.
- Recording-grouped folds let a patient span folds (hence G3).
- Oxygen cannot be varied live, so a live Watch at rest is a false alarm by assumption.
- The probe above was seen before this was written.
- A pass would still be window-level, on retrospective ScalcO2, not a live result.

## Planned outputs (`outputs/watch-head-v1/`, write-once)

Script `scripts/analyze_watch_head.py`, stages `repro`, `cv`, `seeds`, `secondary`, `summary`, run in that order. `repro` must pass before any other stage runs.

- `oof_watch.csv.gz`: row_index, recording_id, patient_id, median_spo2, band, P_A_rec, P_A_pat, P_B_rec, P_B_pat, B_cond_rec, B_cond_pat (NaN outside S), fold_rec, fold_pat.
- `gate.json`: G1, G2, G3 values and the verdict.
- `confusion_3x3.csv`, `metrics.csv`, `seeds.csv`, `sensitivity_94_96.csv`.
- `summary.json`: input SHA-256s, versions, seeds, checks (including the reproduction and same-path checks).
- `README.md` with the readings table filled in, as in `outputs/physio-concept-v1/README.md`.
- **Only if the gate passes:** head B on all 48,662 windows, seed 42, trained as `train_tinykan_v2.run_final` trains the shipped head (scaler on all rows, `train_kan`) but with `y95`. Saved as `watch_final_model.npz` + `watch_final_model.json` in the same dir. On a fail these two files are not written.

## Tests written first (`tests/test_watch_head.py`, written and failing before the script)

- Band and `y95` counts on the real table (as above).
- The state-rule truth table: P_A ≥ 0.5 gives LOW whatever P_B is; WATCH needs P_A < 0.5 and P_B ≥ 0.5; NONE otherwise; boundary values at exactly 0.50.
- The gate function on synthetic vectors: pass, recall-fail, spec-fail, G2-fail, G3-fail.
- The label-swap copy changes only `label`.

## Amendments

This file is not edited after any result in `outputs/watch-head-v1/` is read. Any change after that is a dated amendment appended at the end, saying what changed, why, and which results had been seen. Its commit must predate every file in `outputs/watch-head-v1/`.

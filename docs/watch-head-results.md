# Below-95 Watch head: results (7 Nov 2026, host only)

**Verdict: FAIL.** The pre-registered ship gate failed on recall, in both groupings. No final model was trained, and no firmware is built (plan Part B does not run). The IEEE paper is not edited. This result goes to the final report only.

- Pre-registration: `docs/watch-head-preregistration.md`, commit `c2a6e8c` (7 Nov 2026 21:26 +08:00), before any model in this study was trained.
- Script and tests: `scripts/analyze_watch_head.py`, `tests/test_watch_head.py` (25 passed), commit `2c4d17e`.
- Outputs: `outputs/watch-head-v1/`, write-once, generated 7 Nov 2026 23:05 from `2c4d17e`. Numbers below are from `gate.json`, `README.md`, `metrics.csv`, `confusion_3x3.csv`, `seeds.csv` and `sensitivity_94_96.csv` there.
- Plan: `.hermes/plans/2026-11-07_210839-watch-head-below-95.md`.

## What was tested

A second TinyKAN head (head B) on the same 16 inputs as the shipped Header-aligned head (head A), trained on window-median ScalcO2 < 95. ScalcO2 is SaO2 estimated from end-tidal O2. The CSV column is called `median_spo2`, but it is not a pulse-oximeter reading. Head A was not retrained: its stored out-of-fold vectors were used.

Rule per window, at 0.50: LOW if head A flags; otherwise **Watch** if head B flags; otherwise none. The gate is scored on S, the 19,867 windows with median ≥ 91. In S, the 7,785 windows in [91, 95) (M) are the ones Watch should catch, and the 12,082 windows ≥ 95 (N) are the ones that should stay clear. Recall is the share of M windows given Watch or LOW. Specificity is the share of N windows given no state.

## Gate (pre-registered; 0.50 only)

| Condition | Rule | Result | Reading |
|---|---|---|---|
| G1, recording-grouped | recall ≥ 0.80 and specificity ≥ 0.50 | recall 0.440, specificity 0.825 | fail |
| G2, paired recording bootstrap (2,000, seed 20261025) | lower bound of AUC(head B) − AUC(head A) on S > 0 | +0.051 [+0.030, +0.074]; AUC 0.724 vs 0.673 | pass |
| G3, patient-grouped | G1's two point estimates | recall 0.453, specificity 0.809 | fail |
| Verdict | G1 and G2 and G3 | | **fail** |

Hermes recomputed G1, G2 and G3 from `oof_watch.csv.gz` with a separate script that does not import the study code: recall 0.4401 / 0.4528, specificity 0.8249 / 0.8086, G2 +0.0509 [0.0298, 0.0736]. All three match `gate.json`. Head A's stored vectors in that file match the stored prediction files to within 1.1e-16 (a CSV parse rounding), with identical 0.50 flags.

## States by true band (all 48,662 windows)

Recording-grouped (`confusion_3x3.csv`):

| True band | n | LOW | Watch | none |
|---|---|---|---|---|
| L (< 89) | 28,795 | 25,705 (0.893) | 1,081 (0.038) | 2,009 (0.070) |
| M [91, 95) | 7,785 | 924 (0.119) | 2,502 (0.321) | 4,359 (0.560) |
| N (≥ 95) | 12,082 | 1,698 (0.141) | 418 (0.035) | 9,966 (0.825) |

Patient-grouped: M gets LOW 0.125, Watch 0.327, none 0.547. N gets LOW 0.144, Watch 0.048, none 0.809.

- False-Watch rate on N: 0.035 recording-grouped and 0.048 patient-grouped. Head A's own LOW share on N is 0.141 and 0.144.
- Most of the specificity loss on N comes from the shipped head, not from Watch.
- More than half of the M windows get no state at all.

## Reported, not gated

- **Head B as a score.** AUC on S: 0.724 [0.694, 0.750] recording-grouped and 0.721 [0.690, 0.749] patient-grouped. Against y95 on all windows: 0.859 and 0.862.
- **Six seeds** (42, 1–5; recording-grouped): AUC on S 0.724 ± 0.005, G1 recall 0.436 ± 0.014, G1 specificity 0.822 ± 0.005. These were trained through `lamaakal_models.run_cv_job`. Its seed-42 run differs from the primary head by up to 0.072 in probability, in fold 1 only (folds 2–5 are identical; tolerance 1e-5). Its AUC on S differs by only −6e-5. As pre-registered, the table is labelled "not the same path", and its spread is not read as head B's spread. The reproduction check on the original label had the same pattern: AUC 0.907399 against the stored 0.907528, passing the 0.002 tolerance, with a maximum difference of 0.041 confined to folds 1–2.
- **Comparators on y95** (recording-grouped; the comparator replaces head B in the rule):

  | Comparator | AUC on S | Rule recall / spec. |
  |---|---|---|
  | HistGB-16 | 0.755 | 0.613 / 0.759 |
  | LogReg-16 | 0.609 | 0.297 / 0.810 |

  HistGB minus head B is +0.031 [+0.013, +0.049]. Even the tree ensemble does not reach recall 0.80 at 0.50, as the probe predicted.
- **B_cond** (trained on S only, label M; reported, never gated). AUC on S: 0.759 recording-grouped and 0.761 patient-grouped. Rule recall / specificity: 0.784 / 0.633 and 0.779 / 0.628. It is closer to the recall bar, but still below it. It is not a pass, and adopting it now would be a choice made after seeing the results.
- **Sensitivity**, with windows in [94, 96] removed (re-scoring only): G1 0.487 / 0.814, G2 +0.068 [+0.041, +0.096], G3 0.494 / 0.798. Verdict unchanged.

## What it means

The 16 inputs carry some information about whether a window sits just below 95 or above it. Head B ranks those windows better than reusing the shipped score (G2). At the fixed 0.50 cut, though, it still misses more than half of the 91–94.9 windows. A device Watch state would therefore be silent on most of the band it is named for.

The limits were fixed in advance:
- ScalcO2 is noisy against a 4-point band. The concept regressor's RMSE on windows ≥ 90 was 6.04, with bias −2.88.
- There is no drop band at 95.
- The study is window-level on retrospective data.

The negative result stands as measured. 0.50 is not retuned, and no other head is adopted from this run.

## Not done (by the plan, because the gate failed)

- No `watch_final_model.*`. `--stage final` refused (exit 4) and wrote nothing.
- No `-DOXY_WATCH_HEAD` firmware, header, BLE field, ADR-0011, runbook, CONTEXT entry or dashboard change.
- No board session.

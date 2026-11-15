# Edge KAN: the paper's host analyses rerun (pre-registration)

Written 12 Nov 2026, before any number below was computed for the edge KAN. Jason asked for the full
swap on 12 Nov ("lets try to do everything"). Model: ekan16 as exported (`edge_kan_weights_v2.h`),
threshold 0.50 frozen by `docs/edge-kan-operating-point-preregistration.md`. Nothing is retrained
except where an analysis itself retrains (folds, seeds), exactly as it did for the Embedded Spline Network.

**Principle.** Each analysis is the one the IEEE draft (extract `draft-2026-11-12a.txt`) reports for
the Embedded Spline Network, with the same inputs, folds, seeds, bootstrap unit and reading rule, and the
edge KAN swapped in. Where the original code is importable it is imported, not copied. The Spline
Network is recomputed by the same code beside it where cheap, as a check against its stored numbers.
No analysis is added or dropped after seeing a result; every one listed is reported.

Script: `scripts/edge_kan_paper_analyses.py`, writes `outputs/edge-kan-v1/paper-analyses/` (write-once).

## Analyses

| ID | Paper item | Edge KAN version |
|---|---|---|
| H1 | Table I: AUC, recall, specificity, precision, F1 at 0.50; median P of negative / positive windows; parameters | `predictions_ekan16_recording.csv`; parameters 1,632 |
| H2 | Patient-bootstrap 95 % intervals (2000, seed 20261111); paired difference against logistic regression | `lm.group_bootstrap`, same seed; edge minus LogReg and edge minus Spline Network |
| H3 | Spread over five folds | per-fold AUC, recall, specificity |
| H4 | Six training seeds (42, 1–5), same folds | edge retrained per seed and fold (`train_edge_kan` with `tk.SEED` set); mean, SD, min, max |
| H5 | Error by distance from the cut; recordings holding half of the false positives; frozen sensor-loop rows (2,006) | same bins as `g_error_by_spo2_distance.csv`; `lm.groups_holding_share`; AUC on and off the rows of `outputs/openox-v3/artefact_rows.csv` |
| H6 | Flags against arterial SaO2 below 90 at matched draws | `outputs/scalco2-vs-sao2-v1/s1_windows.csv` joined by `row_index`; `sens_spec` and patient bootstrap of that script |
| H7 | Label shifted ±5–30 s | `lag_pairs`, `rq1_metrics` and bootstrap of the 10 Nov addenda script |
| H8 | Skin-tone stratification (primary: Monk dorsal, recall, dark minus light) | `recording_groupings`, `stratified_bootstrap` of that script |
| H9 | In-sample explanation behaviour of the shipped model: top-1 shares by class; zero-delta slot in the top three; top-ranked feature lowering P on candidates; largest of 16 deltas below 0.01; median-P TP and FP case windows; clamped share | edge header, training-mean reference. "Clamped" for the edge KAN: all 12 hidden activations at or beyond ±3 (the output is then locally constant) |
| H10 | Deletion AOPC gain over random with patient-bootstrap interval: in sample, training-mean and label-0-median references; out of fold with five recording-fold analysis models | `per_window_aopc`, `gain_rows` of the addenda script; fold models retrained with seed 42 and checked against the stored edge out-of-fold predictions |
| H11 | Deletion-test range (R4): share of flagged windows with no range, firmware fraction of the range against random | `greedy_bounds`, `ranking_aopc` of the R4 script; targets P and the logit (the edge analogue of the unclamped pre-activation) |
| H12 | PTT-PPG subject 1, mean P at sit / walk / run, both channel mappings | the windows of `outputs/ptt-ppg/subject-1-v1/windows.csv` scored by the edge header |

Readings are the originals': an interval excluding zero is a difference, else "no difference detected";
an AOPC gain interval above zero "beats random", below "worse than random", else "not distinguishable".

## Not rerun here (stated so it is not mistaken for done)

- Host stream parity over 71,462 decisions (C++ against Python). The pipeline features are unchanged;
  the edge scorer's C++ and Python agreement is covered by the 48,662-row E3 parity and the board parity.
- Gradient×input (not selected for the shipped explainer; edge has no single output clamp).
- The live-capture numbers (sits, input range, module-1 reference shift): they need new captures.
- The six further learners of Table I: they do not depend on the shipped model and are unchanged.

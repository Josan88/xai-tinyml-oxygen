# Edge KAN operating point (pre-registration)

Written 12 Nov 2026, before any threshold sweep, window-level or N = 2 confirmed metric of the edge KAN
other than the 0.50 window-level row already in `outputs/edge-kan-v1/cv_summary.json`
(recording-grouped recall 0.8671 / specificity 0.8892). Jason asked for the full swap on 12 Nov
("lets try to do everything"). Model: ekan16 (`docs/edge-kan-preregistration.md`); nothing is retrained.

## Inputs

Out-of-fold predictions, unchanged: `outputs/edge-kan-v1/predictions_ekan16_recording.csv` (primary)
and `predictions_ekan16_patient.csv` (check). Comparator, same rule and same code:
`outputs/openox-v2/predictions_b_v2_tinykan16.csv` and `_patient.csv` (the Embedded Spline Network).

## Modes

- **window**: P >= t.
- **confirmed_n2**: the window and the previous window of the same recording (ordered by
  `start_time_seconds`, then `row_index`) are both >= t. This is `confirmed_n2_mask` from
  `scripts/calibrate_operating_point.py`, imported unchanged (the v1 rule; time gaps are not checked).

Gate (ETPB): recall >= 0.80 and specificity >= 0.50.

## Rule (decided before the sweep)

1. **Keep 0.50** if, recording-grouped, the edge KAN meets the gate at 0.50 in **both** modes. This is
   the rule the Embedded Spline Network's 0.50 was kept under (`docs/operating-point-calibration.md`,
   addendum 28 Oct), so the two models are compared at the same threshold and neither is tuned.
2. Otherwise, select on recording-grouped **confirmed_n2** with the v1 selection rule
   (`select_rows`: among feasible thresholds the highest Youden J, then F1, then specificity, then the
   lowest threshold; if none is feasible, the smallest gate violation). Grid: 0.05 to 0.95 in 0.01 steps.
3. The patient-grouped result is reported at the selected threshold. It does not change the choice.

Reported, not used to choose: the full sweep for both models and both groupings, and the v1-rule
selection for both models (what each would get if tuned), so a reader can see what the tie to 0.50 costs.

## Consequences fixed in advance

- If 0.50 is kept, the 12 Nov board parity (1,243 decisions at 0.50) is the parity of the live threshold.
- If another threshold is selected, the edge live image uses it, and a board parity replay at that
  threshold is owed before the live image is used for any claim.
- Whichever threshold results is frozen for the live image, the sits and the paper. It is not revisited
  after any live capture.

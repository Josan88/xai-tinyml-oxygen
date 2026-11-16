# Deletion-test range and target, R4 (11 Nov 2026)

**Host only, in sample, shipped v2 header, 48,662 windows.** Pre-registration `docs/deletion-range-preregistration.md` (ef0cdfd); script `scripts/analyze_deletion_range_r4.py` (committed unrun as 5617b35). M0 passed: stored AOPC reproduced to 4.8e-9; 28,661 candidates, 11,368 at the output clamp. Patient bootstrap, 87 patients, 2,000 resamples, 95 % percentile intervals.

## Pre-registered readings (training-mean reference, candidates)

| Reading | Result |
|---|---|
| R-A: can the test discriminate on P? | **Has range.** 6.1 % of candidates can move by less than 0.01 (rule: more than half). The median range is only 0.012, so half the candidates sit just above the cut. |
| R-B: within that range, is the ranking better than random? | **Worse than random.** Mean fraction of the attainable span 0.486 (firmware) against 0.544 (random); difference −0.057 [−0.072, −0.041]. |
| R-C: on the unclamped output pre-activation, does the ranking order inputs by influence? | **Beats random** on candidates: deletion gain +0.386 [+0.278, +0.505]. |

**R-C is driven by the clamped candidates**: gain +1.24 [+1.04, +1.47] on the 11,368 clamped, −0.17 [−0.22, −0.12] on the 17,293 unclamped. On unclamped candidates the shipped ranking is below random on both targets (P: −0.019; fraction −0.028 [−0.036, −0.020]).

## D1: the target at the all-reference input

| Reference | P | Output pre-activation |
|---|---|---|
| Training mean (shipped) | 0.891 | +1.88 |
| Label-0 median | 0.081 | −1.58 |

The shipped reference lies inside the high-risk region; the label-0 reference does not.

## Label-0 reference (descriptive)

On candidates its ranking reaches 0.984 of the span against 0.508 for random (+0.476 [+0.471, +0.481]); it beats random on both targets in every subset. Insertion agrees in sign with deletion in every cell.

Full table: `range_and_gains.csv`; insertion curves: `insertion_curves.csv`; D1: `all_reference_value.csv`.

## Limits

In sample on the shipped model only. The greedy bounds are not the exact optimum over 16! orders. The pre-activation target is internal to the model. Host arithmetic; not board XAI, not human-grounded.

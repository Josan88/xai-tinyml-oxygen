# Deletion-test range and target (R4): host analysis pre-registration (11 Nov 2026)

**Status: approved by Jason on 11 Nov 2026 ("Approve and run") and committed before the script was written. No number below had been computed.** Host only, laptop, no board, no firmware or header writes. Nothing shipped changes: the TinyKAN header, its 0.50 threshold, N = 2 and the training-mean reference stay as they are.

**Seen before fixing.** Only the code paths named below (`HeaderKAN`, `deletion_curves`, `ranked_orders`, `aopc_deletion`) and the already-reported numbers in `outputs/panel-2026-11-10-addenda-v1/` and `outputs/xai-reference-label0-v1/`. No oracle ordering, insertion curve or pre-activation target has been computed for the v2 model.

## Why

Roadmap item R4 of `docs/ieee-draft-panel-review-2026-11-10/` asks the paper to show how much of the flagged subset the deletion test can discriminate on, and to separate ranking quality from reference visibility. The re-review of 11 Nov (`re-review-2026-11-11/`) graded the remaining gap `must_fix`: P at the all-reference input, the attainable AOPC range from a greedy oracle and its reverse with each ranking as a fraction of it, an insertion curve, a non-saturating target, and an RQ2 answer that says whether the test cannot discriminate or the ranking is uninformative.

## Model, rows, references, rankings (fixed now)

- **Model:** the shipped header through `lamaakal_models.HeaderKAN` (host FP32-constant arithmetic). In sample, all 48,662 rows of `outputs/openox-v2/window_features.csv`. No retraining, no out-of-fold models.
- **References**, each with its own ranking, as in `analyze_xai_reference_label0.aopc_deletion`: (a) the training mean (standardised 0, shipped); (b) the FP32 label-0 median.
- **Rankings:** the firmware-absolute order of each reference's own deltas, and the mean of the five random orders (seed 20260911), unchanged from the reported runs.
- **Subsets:** all windows; candidates (P ≥ 0.50, primary); candidates at the output clamp (output pre-activation ≥ +3) and the others.

## M0 checks (stop before writing a number if one fails)

1. Firmware-absolute and random-mean deletion AOPC on P reproduce `outputs/xai-reference-label0-v1/aopc_deletion.csv` (v2 rows, both references, five subsets) within 1e-8.
2. Candidates number 28,661, and 11,368 of them are at the output clamp.

## Measurements

Per window, with d = 16, target T(x), and x^(S) = x with the inputs in S set to the reference:

- **D1 All-reference value.** T at the all-reference input, for each reference and target (one number each).
- **D2 Attainable range.** Deletion AOPC of a ranking = (1/d) Σ_{k=1..d} [T(x) − T(x with its top-k deleted)]. The **greedy oracle** deletes, at each step, the not-yet-deleted input whose deletion lowers T most; the **reverse greedy** deletes the one that raises T most. Their AOPCs are the window's upper bound U and lower bound L, and R = U − L is its attainable range.
- **D3 Fraction of range.** For each ranking, f = (AOPC − L) / R on windows with R ≥ 0.01; the share of windows with R < 0.01 is reported separately as "no range".
- **D4 Insertion.** Insertion AOPC = (1/d) Σ_{k=1..d} [T(reference with the top-k observed values inserted) − T(all-reference)], for the firmware and random orders, with the mean insertion curve per subset at k = 0..16.
- **D5 Non-saturating target.** D2 to D4 repeated with T = the output pre-activation u2 (`HeaderKAN.output_preactivation`, the input to the clamped output spline), which does not saturate at the clamp. The rankings are not recomputed: they stay the shipped deltas on P, so this asks whether the shipped ranking orders inputs by their effect on the model's internal score.

Every subset mean gets a 95 % percentile interval from 2,000 resamples of whole patients (87), seed 20261112 + cell index. Gains (firmware minus random) are paired per window.

## Readings (fixed now; training-mean reference, candidates, unless stated)

- **R-A (can the test discriminate?)** If more than half of the candidates have R < 0.01 on P, the paper says the deletion test on P **cannot discriminate on most flagged windows**. Otherwise it says the test has range there.
- **R-B (is the ranking informative?)** On the windows with R ≥ 0.01 on P: if the interval of the mean f gain (firmware f minus random f) lies above zero, the ranking is **better than random within the range the test can see**; below zero, **worse than random**; otherwise **not distinguishable**.
- **R-C (does the ranking order inputs by influence?)** On the pre-activation target (D5), the same rule applied to the firmware-minus-random deletion AOPC gain on candidates. Above zero: the shipped ranking orders inputs by their effect on the model's score, and the P-target failure is a visibility problem of the reference and the clamp. Not above zero: the ranking is uninformative on flagged windows whatever the target.
- **RQ2 wording.** The paper's RQ2 answer is restated from R-A, R-B and R-C in at most three sentences, whichever way they fall. Insertion (D4) and the label-0 reference are reported descriptively beside it, with no reading attached.
- No headline number is replaced, and the shipped explainer, reference and threshold are not changed whatever the result.

## Limits known in advance

- In sample on the shipped model only; R3's out-of-fold check is not repeated here.
- The greedy oracle is greedy, not the exact optimum over 16! orders, so U and L bound what a single-step-greedy search finds; the true range can be wider.
- The pre-activation target is internal to this model; it is a diagnostic, not a target any user sees.
- Deletion and insertion are functionally grounded tests on host arithmetic, not board XAI and not a human-grounded test.

## Cost

About 2 × 16 × 17 / 2 ≈ 272 forward passes per window per reference per target for the two greedy searches, on 48,662 windows: about 53 million row evaluations in total, vectorised. Minutes on the laptop.

# Edge KAN: deletion test of an additive (Shapley) ranking (pre-registration)

Written 13 Nov 2026, before any additive attribution was computed for the edge KAN. Jason asked on
13 Nov to set up "the additive deletion test for the edge-KAN" after the IEEE draft's conclusion
("By the criterion of Section I, the flag and its explanation are still not shown to be one system")
was discussed. Model: ekan16 as exported (`edge_kan_weights_v2.h`), threshold 0.50, unchanged.

## Why, and what this can and cannot change

- Section I of the draft (extract `draft-2026-11-13b.txt`, ¶9) says the two outputs could be believed
  as one system only if the flag meets the gate of Section III-C **and the shipped ranking** beats random
  on the windows it flags. The shipped ranking is mean-perturbation with the training-mean reference.
  Its in-sample candidate gain is +0.002 (−0.005 to +0.009), and out of fold −0.002 (−0.011 to +0.005)
  (`outputs/edge-kan-v1/paper-analyses/`, H10).
- This analysis asks one question: does a **different** local attribution, baseline Shapley values
  against the same reference, rank the inputs better than random under the same deletion test? Its
  values add up exactly to P(x) − P(reference), which mean-perturbation deltas do not.
- **Whatever it finds, it does not change the sentence about the shipped system.** The shipped image
  sends mean-perturbation deltas. A "beats random" result here can only be reported as a host result for
  an alternative attribution that is not deployed. Moving the shipped system to it would need its own
  firmware, board replay and registration. The criterion of Section I is not changed by this analysis.

## Attribution

- **Baseline Shapley values** against reference r: φ_i is the average over orderings of the change in P
  when input i moves from r_i to x_i, with the inputs before it in the ordering already at x.
- **Estimator (fixed before running):** 32 permutations, as on the board image of the public snapshot
  (`fyrp-xai-tinyml-public`, `include/kan_additive_attribution.h`, ADR-0012). There are 16 xorshift32
  Fisher–Yates shuffles from seed 0x9E3779B9, each used together with its reverse, and the same 32
  orderings for every window. Each ordering telescopes, so sum(φ) = P(x) − P(r) for every window
  (checked, tolerance 1e-9).
- **Estimator check (reported, not a gate):** on 100 candidate windows drawn with seed 20261113, exact
  baseline Shapley over all 2^16 subsets for the training-mean reference. The report gives the median
  and maximum |φ_estimate − φ_exact|, and the share of windows whose top-3 set agrees.
- The ranking is by |φ|, descending, ties by input index. This is `lm.firmware_abs_order` applied to the
  φ matrix, exactly as the shipped deltas are ranked.

## Deletion test (unchanged from H10)

- `analyze_panel_addenda_20261110.per_window_aopc(model, z, φ, reference)` gives the per-window AOPC of the
  |φ| ranking and of the mean of the five random orders (`las.AOPC_SEED`, `las.AOPC_RANDOM_REPEATS`).
  Deleted values are set to the same reference.
- `gain_rows`: gain = ranking − random, patient bootstrap (2000 resamples, seed 20261111 + cell, as
  H10), with H10's reading rule: interval above zero "beats random", below "worse than random", else
  "not distinguishable from random".
- **Cells:**
  - in sample on the shipped header, two references: training mean (z = 0) and label-0 median
    (median standardized input over label-0 rows, FP32-rounded, as H10);
  - out of fold with the five recording-fold analysis models retrained with seed 42, as H10, with
    each fold's own label-0 median. The retrained fold models are checked against the stored
    out-of-fold predictions (H10 found 5e-7).
- **Subsets:** all, candidate (P ≥ 0.50), non-candidate, label 1, label 0. These are H10's, with each
  cell's candidate set from its own model.
- **Side by side:** the shipped mean-perturbation gains are recomputed by the same code in the same run
  and listed next to the additive ones. Their point gains must reproduce H10 within 1e-9, or the run stops.
  Their intervals can differ slightly from H10's, because this run numbers its bootstrap cells
  differently. Within a cell, both attributions use the same bootstrap draws.
- **Additivity gate:** if any window's |sum(φ) − (P(x) − P(r))| exceeds 1e-9, the run stops.

## Primary result and reading, fixed now

- **Primary:** in sample, training-mean reference, candidate windows. The additive gain and its interval
  are read with the rule above.
- **Secondary:** out of fold, training mean, candidates. Label-0 reference, both cells. The other subsets.
- No cell is added, dropped or re-cut after seeing a result. Every cell is reported.
- **Known bias, stated now:** with the training-mean reference, φ sums to P(x) − P(r), and deletion moves
  inputs towards r. An additive ranking is therefore built to suit the fixed-P deletion target, more
  than a one-at-a-time ranking is. A positive gain is partly built into the metric, as the draft already
  says of the label-0 result. It is evidence about consistency with the deletion test, not about
  physiology.

Script: `scripts/edge_kan_additive_deletion.py`, writes `outputs/edge-kan-v1/additive-deletion/`
(write-once; records this file's sha256, the script's and the inputs').

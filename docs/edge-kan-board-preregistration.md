# Edge KAN on the board: parity and timing (pre-registration)

Written 12 Nov 2026, before either edge-KAN board image was built or flashed. It extends
`docs/edge-kan-preregistration.md` (SHA-256 `cb70b259…`), which listed board parity and timing under
"Not done here". Jason named the job on 12 Nov 2026 ("Yes, run both"). The model is ekan16 as
exported (`edge_kan_weights_v2.h`); nothing is retrained, and 0.50 is still not a frozen threshold for it.

## Images (diagnostic only, never the shipped image)

Both images run the shipped v2 r2 feature pipeline (`OxygenRiskPipeline`, `-DOXY_MODEL_V2`) unchanged on
replayed IR/red blocks, then evaluate the edge KAN (`edge_kan_inference.h`) on the decision's
`raw_features`. TinyKAN still runs inside the pipeline and its outputs are reported as before.

- `nano33ble_ekan_context_parity` (`src/context_parity_replay_ekan.cpp`). Identity
  `CONTEXT_PARITY_EKAN WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16`.
  RESULT = the 32 v2 RESULT words, then edge P, the 16 edge deltas (incremental method) and the 3 edge top-3 indices.
- `nano33ble_ekan_context_timing` (`src/context_timing_replay_ekan.cpp`). Identity
  `TIMING_REPLAY_EKAN WINDOW=500 CONTEXT=250 STRIDE=250 N=2 CLOCK_HZ=64000000 GPIO=D2 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16`.
  Per block: DWT cycles of the full v2 pipeline (D2 high around it, as in the v2 timing image). Per
  decision, on the same `raw_features`, 5 repetitions each of: TinyKAN forward, TinyKAN 17-pass XAI (a
  verbatim copy of `evaluate_kan_and_xai`), edge forward, edge incremental XAI. Min and max cycles reported.

## Inputs

The v2 r2 prepared set, unchanged (`outputs/parity/v2-board-prepared-r2/`): 3 synthetic paired fixtures
(21 decisions) and the 611 locked OpenOximetry contexts (1,222 decisions). Its `expected_host.npz` gives
the C++ host v2 features, TinyKAN probability and states for every decision.

## Board parity gate (all 1,243 decisions; every failure reported, none excluded)

Edge host expectation: `tests/edge_kan_probe.cpp` (zig c++, `-ffp-contract=off`, the E3 build) run on the
C++ host v2 features of each decision.

1. **Pipeline unchanged.** Board features pass `parity_criteria_v2.feature_check` against the C++ host
   features; TinyKAN probability within 1e-4 of the host; metadata, Candidate/Confirmed/streak and
   `two_wavelength_valid` equal to the host row.
2. **Edge KAN.** `parity_criteria_v2.check_window(host features, board features, host edge P, board edge P,
   host edge deltas, board edge deltas)`: the frozen v2 budgets (probability 1e-4, each delta 1e-5,
   ranking gap 2e-5, same side of 0.50).
3. **Edge payload.** The board's three top indices equal the stable |delta| argsort of the board's own 16 deltas.

Reported, not gated: board edge vs the host engine evaluated on the *board's* features (arithmetic
alone: max |dP|, max |d delta|, top-3 agreement); top-3 set/order changes against the host.

Pass = every decision passes 1–3. A partial `--locked` run is a first look, not the result.

## Timing (reported, no pass/fail)

Fixture streams (21 decisions) × 10 repetitions, as the v2 timing run, plus 8 empty-bracket
calibrations (`C`) and a memory snapshot (`M`). Each timed decision is checked against the C++ host
TinyKAN probability (1e-4) and the host edge probability (1e-4) so a timing row is from a correct run.
Reported in µs at 64 MHz: median and max of each kernel, and of the block. The edge decision path is
**derived**, not measured: block − TinyKAN XAI + edge XAI, using the same decision's min cycles. The
640 ms FIFO budget is the reference line.

## Afterwards

Reflash the image that was on the board before this job: `nano33ble_v2_f1a_spo2`
(`outputs/firmware-v2-f1a-spo2maxim`, `f4a78cf6…`, last receipt 6 Nov 2026), with a receipt.

## Not done here

No live sit, no threshold freeze, no peak-RAM probe, no change to the shipped images or to the paper.

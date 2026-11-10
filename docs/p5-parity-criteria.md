# P5 parity criteria (pre-registered)

**Fixed 30 Oct 2026, before any C++-vs-Python comparison on OpenOximetry rows.** Job F2 (ADR-0008). Committed alone. Before this commit the only C++-vs-Python comparison run was the synthetic set of `tests/test_physio_features.py` (`outputs/parity/p5-features-unit-20261030/synthetic.json`: 46 of 46 windows with identical peaks; features bit-equal except `sqi_template`, at most 2.2e-16). No budget below is changed after a result is seen. A result outside a budget is reported as a residual, never absorbed by widening.

## What is compared

- **C++ side.** The P5 deployed pipeline: `BasicOxygenRiskPipeline<250>` built with `-DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0`.
  - Host probe `tests/p5_stream_probe.cpp`, zig c++ (clang) `-std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu`, as the v2 host parity.
  - Per decision: the paired IR/red context filter in FP32 (as v2), then the raw-window FP32 DC means (as v2's ratio inputs), then `include/physio_features.h`, imputation, standardization, TinyKAN P5, five Mean-Perturbation passes with the label-0-median reference, top-3, N=2, and the `s_hat` regressor.
- **Python side.** `scripts/p5_deployed_reference.py`:
  - the same deployed filter path as `scripts/v2_deployed_reference.py`: FP32-quantized streams, then `sosfiltfilt` in float64 on `[max(0, start-250), start+750)`;
  - DC = `np.mean` of the raw FP32 target window in float64;
  - then `scripts/physio_features.extract` in float64;
  - then the FP32-constant model replica (lamaakal_models.HeaderKAN arithmetic on `kan_model_weights_p5.h`, `kan_sat_regressor_p5.h`, `kan_xai_reference_p5.h`).
- **Rows.** The 154 recordings and streams of the v2 host parity: raw red/IR resampled to 50 Hz as `scripts/export_openox_windows_v2.py` does, FP32, one reset per recording. Every decision of every recording; nothing excluded. Output: a fresh `outputs/parity/p5-host-<date>/`.

The extractor itself is not the source of disagreement: given the same window, the port reproduces Python to double rounding. The two sides see different filtered windows, the FP32 cascade versus float64 `sosfiltfilt`. Every budget below is about that difference.

## Classes and budgets

A decision **fails** if it fails any class. `parity_complete` is true only if no decision fails. Each class is reported separately, with every failing decision listed.

### (a) Beat detection: integer peak-list agreement (exact, its own class)

- **Rule.** Polarity, the full integer IR peak list from `find_peaks`, `n_beats` and `beat_valid` must all be identical. There is no tolerance: these are integers and flags.
- **Reported.** The share of decisions with identical peaks, with a Wilson 95% interval. Every mismatch is listed with recording, window index and cause.
- **Cause.** Fixed now. The probe dumps the C++ FP32 filtered IR/red window of every mismatched decision. Python then runs the extractor stages on both windows, the C++ one and its own float64 one. The cause is the first stage whose output differs:
  1. usable-window check;
  2. polarity (`P99(diff) >= -P1(diff)`);
  3. local-maximum set;
  4. distance selection;
  5. prominence selection;
  6. `beat_valid` with identical peaks.
- **Margin.** Each cause is reported with its margin relative to the window std `σ`:
  - polarity: `|P99 + P1| / σ`;
  - local maxima: the flipped adjacent difference `/ σ`;
  - distance selection: the height difference of the competing peaks `/ σ`, and the C++ `distance_tie` flag (an exact FP32 height tie);
  - prominence selection: `(prominence - σ) / σ`.
- **Port check.** It runs on the same dumped windows, plus a SHA-256-selected sample of 200 matched decisions. Python `extract` on the C++ FP32 window must reproduce the C++ discrete fields exactly, and its features to `1e-12 x max(1, |f|)`. A port-check failure is a code bug. It is reported as such and fixed. It is not a parity residual.

### (b) Features, on decisions that pass (a) and are beat-valid on both sides

These are the frozen contract's budgets (`tests/fixtures/numerical_parity_contract.json`, through `scripts/parity_criteria.py`), unchanged:

`|f_cpp - f_py| <= 1e-4 x scale + 8 x spacing_FP32(f_py)`

`f_cpp` is the FP32 model input the device holds, not the double. As for v2, the only new decision is the local-scale class of each new input:

| input | class | scale | reason |
|---|---|---|---|
| `r_beat` | `absolute_or_one` | `max(1, \|r\|)` | dimensionless ratio near 1; the class of `ratio_of_ratios` |
| `pi_beat` | `absolute` | `\|pi\|` | positive amplitude ratio; the class of `ir_perfusion_index` |
| `hr_bpm` | `absolute` | `\|hr\|` | positive rate, 40–180 |
| `sqi_template` | `absolute_or_one` | `max(1, \|sqi\|)` | correlation in [-1, 1]; a pure relative scale would be undefined near 0 |
| `sqi_ibi_ratio` | `absolute_or_one` | `max(1, \|q\|)` | ratio >= 1 |

**Why 1e-4 holds for these inputs: FP32 arithmetic, not data.** The FP32 filter's sample error, relative to the window's AC scale, is of order `G·k·u`:

- `u = 2^-24 ≈ 6e-8`;
- `k` is about 10 FP32 operations per section, 4 sections, 2 passes;
- `G` is about 10, the noise gain of the 0.5 Hz high-pass section (pole radius ≈ 0.94).

That gives `ε` between 1e-6 and 1e-5. It is the error class the frozen 1e-4 was adopted for, and it held for all 16 v2 inputs (`outputs/parity/v2-host/20261021T163919Z/`). Propagation:

- `r_beat`, `pi_beat`. Each segment's `AC = max - min` moves by at most `2ε` of the AC scale. The FP32 two-pass DC mean differs from the float64 mean by a few `u`. Each ratio's relative error is about `2ε + 4u`, so 2e-5 or less. A median is 1-Lipschitz in the max norm, so the median moves no more than the worst segment, even when two near-equal segments swap order.
- `hr_bpm`. The parabola offset is `0.5(a-c)/(a-2b+c)`. Its error is about `ε·A / |a-2b+c|`. At a typical systolic peak, `|a-2b+c|` is 0.02 A or more, so the offset error is 5e-4 samples or less and the relative IBI error is 2e-5 or less. **Not covered:** flat-topped peaks with near-zero curvature, where the offset is ill-conditioned and clipped at ±0.5. Those decisions are expected residuals and will be reported as such.
- `sqi_template`. A Pearson r moves by `O(ε)` under a sample perturbation of `ε` of the AC scale.
- `sqi_ibi_ratio`. Twice the IBI relative error.

Beat-invalid on both sides: every input must be exactly `KAN_P5_IMPUTATION_MEAN[i]`, bit for bit. That value is `SCALER_MEAN[i]`, so `z = 0`.

### (c) Probability, explanation, flag: frozen contract numbers applied to P5, all decisions

- Probability: `|ΔP| <= 1e-4`.
- Each of the **five** signed Mean-Perturbation deltas (label-0-median reference `KAN_XAI_REF_Z`, `kan_xai_reference_p5.h`): `|Δδ_i| <= 1e-5`.
- Ranking: for every pair whose reference magnitudes differ by more than 2e-5, the C++ order must be the same. Pairs within 2e-5 are ambiguous and not judged.
- Candidate: `(P_py >= 0.50) == (P_cpp >= 0.50)`, exact.

These are the contract's numbers. They are not re-derived for P5, and this job may not change them. FP32 rounding inside the network is smaller than for v2: 5 terms per hidden pre-activation instead of 16, with the same 12 hidden units and 8 knots. The remaining risk is input-error propagation, dominated by `r_beat` (top-1 on 81–92% of windows, Stage 3). Its standardized error is `|Δr|/σ_r ≈ 2e-5 / 1.24`. (c) is judged on every decision. The headline is (c) on decisions that pass (a); decisions that fail (a) are already failures, and their (c) checks are listed beside the cause.

### (d) `s_hat`, all decisions (beat-valid and not)

`|Δs_hat| <= 4e-4 x KANSAT_TARGET_STD + 8 x spacing_FP32(s_hat_py)`

With `KANSAT_TARGET_STD = 8.828` that is **0.00353 + about 6e-5 percentage points**.

Derivation:

- `s_hat = TARGET_STD · out + TARGET_MEAN`, where `out` is the same 5 -> 12 -> 1 KAN shape fed the same standardized input as the risk logit.
- The contract's 1e-4 probability budget at the decision point, where `dP/dlogit = 1/4`, accepts a logit error of 4e-4. `out` gets the same 4e-4.
- That needs `out` to be no more sensitive to its inputs than the logit. This is a model property, computed from the two headers, not from parity data. Dense sampling of every spline's slope gives worst-case input-Lipschitz bounds for `out` of 0.27 to 0.78 times the logit's, per input; summed, 22.1 against 52.4.
- FP32 rounding of the forward pass itself is about 1e-5 in `out`.
- The final multiply-add at `s_hat ≈ 90` adds at most 1.5 FP32 spacings, covered by the 8-spacing term.

This is the least direct of the four derivations: an analogy to the probability budget, supported by the Lipschitz ratio. It is stated as such.

**Display, not gated.** BLE field 10 prints one decimal. The count of decisions whose `%.1f` text differs is reported. Any nonzero budget allows a flip at a rounding boundary.

### (e) Exact fields, all decisions

All of these must match exactly:

- metadata: window index, total samples, start, end, delay;
- Candidate, Confirmed and streak over the continuous stream (ADR-0005, N=2, P >= 0.50);
- the top-3 payload: the indices are the stable argsort of the C++ build's own five `|δ|`, and the values equal those deltas;
- `beat_valid`.

## Real-window unit test of the port (separate from parity)

`scripts/check_physio_features_cpp.py --real 400` gives the same FP32 window to both sides. The 400 rows of `outputs/openox-raw-windows-v1/v2_ir_red_windows.npz` are chosen by SHA-256 order of `p5-features-unit-2026-10-30:<row>`; no label, feature or outcome is used.

- **Rule.** Discrete fields identical on every window, and every feature within `1e-12 x max(1, |f|)`.
- That is double rounding, about eight orders of magnitude inside the parity budgets.
- A failure is a bug in the port. It is fixed and the test rerun, and the fix is reported.

`--tie-scan` reports how many of the 48,662 offline windows contain two local maxima closer than 17 samples with exactly equal height. Only in those windows does `find_peaks` depend on `np.argsort`'s tie order. That order is CPU-dependent on NumPy 2.x and is not reproduced; the port uses position order.

## Deployed versus offline (descriptive, not gated)

This is the comparison ADR-0006 reports for v2: 8 of 48,625 Candidate flips, 0.016%.

1. **Features.** Deployed path (C++), from 20 s context buffers, against `outputs/physio-concept-v1/physio_features.csv.gz`, from the whole-recording offline filter. At export windows the deployed path can decide (`start + 750 <=` recording length). Reported:
   - `beat_valid` and `n_beats` agreement;
   - `|Δ|` quantiles per input.
2. **P5 Candidate flip share.** Offline P is the P5 replica on the offline features, beat-invalid imputed; it is compared with the deployed C++ P at those windows. Flips are reported with a Wilson 95% interval, by direction and by label. Resubstitution: the P5 model was fitted on all rows, so the shift is sized here, not generalization.

There is no threshold. It is the buffer-edge gap, now through a discrete beat detector, and is reported as found.

## Board parity (for the board session; criteria fixed here)

A prepared set is built as for v2: the same 3 synthetic paired fixtures and the same locked-context selection rule and seed. Each board decision is judged against **both** the Python reference and the C++ host, with classes (a)–(e) above. The board image returns, per decision:

- the five FP32 inputs;
- `n_beats`, `beat_valid`, the probability, all five deltas, the top-3 and `s_hat`.

It does not return peak positions. Class (a) on the board is therefore `n_beats` and `beat_valid` identical, plus (b) on the features, plus exact equality with the C++ host's features. The host is the same code, so the board and host peak lists agree whenever the board and host features are bit-equal.

The pass criterion is zero failing decisions against both. A locked case that already fails host-vs-Python counts as a board failure. It is reported; it is not excluded.

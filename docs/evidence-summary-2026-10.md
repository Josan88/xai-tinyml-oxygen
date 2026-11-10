# Evidence summary — what is measured as of 7 October 2026

Engineering record for EAT40006. Every number here comes from a run on disk; each section names the
evidence. Nothing in this file is a new experiment — it is an inventory of what already exists, because
the remaining-work list had drifted from the repository.

Frozen throughout: exported TinyKAN weights (`firmware/Arduino Nano 33/include/kan_model_weights.h`,
SHA-256 `123a305f...`), operating point **0.50**, ADR-0005 N=2. Nothing was retrained.

---

## 1. Model inference parity — PASS

Same 14 features into the C++ port and the NumPy reference:

| Check | Result |
|---|---|
| Rows | 208 (every 251st canonical row) |
| Max probability error | **1.99e-7** |
| Max attribution error | **2.61e-7** |
| Candidate disagreements | **0** |

Evidence: `outputs/parity/report.json` → `same_feature_inference`. Gate was 1e-5. The C++ TinyKAN port
is numerically sound.

## 2. Preprocessing gap — bounded, and it is not what it looked like

Two facts that changed the picture:

- **The firmware filter is not causal.** `ppg_bandpass_filter` runs the same zero-phase forward-and-backward
  SOS as SciPy `sosfiltfilt`, with the same 27-sample odd-extension padding. The gap is *whole-recording vs
  short-buffer* — edge transients at the buffer boundary, not phase distortion.
- **There are two firmware paths.** The device (`main.cpp`, `OxygenRiskPipeline` = `<250>`) filters
  **5 s history + 10 s window + 5 s future** (1000 samples). The window-local `<0>` path used by the host
  probes filters the 500-sample window alone and **does not ship**.

Measured on **640 stratified windows, all 162 recordings** (320 near-threshold, 320 clear-cut, both labels),
frozen model, threshold 0.50:

| | Deployed `<250>` (the device) | Window-local `<0>` (probe only) |
|---|---|---|
| Alert flips | **0 / 640** | 29 / 640 |
| Flip rate | 0%, Wilson upper bound ≈ 1.2% | sample 4.53% ± 0.82%; population-weighted **7.11% ± 1.40%** |
| Median \|Δp\| | **0.000011** | 0.0044 |
| p99 \|Δp\| | 0.00050 | 0.3402 |
| Max \|Δp\| | **0.0026** | 0.4123 |

On the window-local path the population-weighted recall shift is about **−3.4 pts** and specificity
**+3.0 pts** (neither significant at 95% on its own). On the deployed path the shift is **zero** — no sampled
window changed its Candidate flag.

**Conclusion:** the deployed device agrees with the offline preprocessing to within 0.0026 in probability and
never flips an alert on this sample. A limitation paragraph is sufficient. No metric re-derivation is required
on this ground.

Evidence: `outputs/parity/filter-impact/` (`windows.csv`, `report.md`, `summary.json`, `error_convolution.json`).
Reproduce: `python outputs/parity/filter-impact/measure_filter_impact.py`.

## 3. Explanation faithfulness (AOPC) — run, with an uncomfortable result

ADR-0003 deletion/insertion AOPC over all **52,090** windows, masking standardized coordinates to the
training mean, five PCG64 random repeats, seed 20260911. Tests `7/7`; vectorised vs scalar NumPy max error
5.0e-16; host vs `probability_probe.exe` max P error 1.53e-7.

Window-mean results:

| Ranking | Deletion AOPC | Insertion AOPC | Deletion AUC | Insertion AUC |
|---|---:|---:|---:|---:|
| signed_risk | **0.1246** | **0.1283** | 0.3796 | 0.5788 |
| **firmware_absolute** (what the device uses) | 0.0443 | 0.0444 | 0.4599 | 0.4949 |
| random_mean | 0.0283 | 0.0254 | 0.4759 | 0.4759 |
| reverse_signed | −0.0746 | −0.0709 | 0.5788 | 0.3796 |

- `signed_risk` beats its own random mean on **99.74%** of windows (mean gain 0.0963).
- `firmware_absolute` beats its own random mean on **77.58%** of windows (mean gain 0.0160).
- On **candidate** windows (exported-model P ≥ 0.50, n=34,127) firmware-absolute is better than random on
  **98.33%** (gain 0.0382).
- On **non-candidate** windows (n=17,963) firmware-absolute mean deletion AOPC is **−0.0536** — **worse than
  random** on most of them.

**This is the honest answer to "how explainable is the model".** Ranking by absolute perturbation size, which
is what the firmware emits as top-3, is only modestly above random on the pooled set and actively misleading
when the model outputs Normal. It works well when the model is flagging risk.

Reported limitations (from the run itself, not to be dropped): in-sample masking of the exported model;
curves use the full 14-feature order, not the top-3 payload; masked points can be physically impossible
because the derived PPG features are correlated; signed deletion's first-step advantage is partly tautological
because the same mean replacement defines both Δ and the first deletion step; five random repeats are not 14!.

Evidence: `docs/xai-faithfulness-aopc.md`, `outputs/xai-faithfulness/openox-aopc-v1/`.
Reproduce into a **new empty** directory: `python scripts/evaluate_xai_faithfulness.py --output <dir> --probe build/parity/audit-release/probability_probe.exe`.

## 4. On-device latency and board replay

Protocol `TIMING_REPLAY_V1`: WINDOW=500, CONTEXT=250, STRIDE=250, N=2, THRESHOLD=0.50, CLOCK_HZ=64000000,
GPIO=D2, ATTR14=0, 10 repetitions, COM8.

DWT cycle timing over 250-sample blocks (5 s of signal at 50 Hz), including the decision and XAI when due:

| Phase | n | median | p95 | max |
|---|---:|---:|---:|---:|
| Warmup block (no decision due) | 60 | 164.6 µs | 191.8 µs | 214.0 µs |
| First decision after reset | 30 | 204.2 ms | 204.7 ms | 204.8 ms |
| Subsequent decision blocks | 180 | 249.7 ms | 250.5 ms | 250.8 ms |

A block containing a decision plus the 15-pass Mean-Perturbation explanation costs about **250 ms of CPU**.
Against a 5 s window stride that is roughly **5% duty**, so the loop keeps up with the sensor with headroom.

Scope caveats recorded with the run: accelerated synthetic USB replay; raw timings include instrumentation
overhead; **no sensor or BLE latency**; parsing, printing and memory queries are outside the interval;
interrupts remain enabled; the stack watermark and newlib heap snapshots are **not** total peak RAM.

Board replay of the locked set: **2,464 windows, 0 failed, 0 candidate disagreements, 0 timing-state
failures.** Max probability absolute error **8.25e-7**; max attribution absolute error **1.58e-6**.

Evidence: `outputs/parity/timing-board-v1/report.json`,
`outputs/parity/qualification/locked-board-replay-v1/report.json`.

## 5. B-spline saturation — a real limitation

On windows with extreme amplitude (`std` ≈ 3.7e5, `energy` ≈ 7e13) all **12 hidden pre-activations exceed the
spline grid at ±3** and clamp there. The B-spline basis does **not** vanish at the clamp — at `u = 3` it evaluates
to `[0, 0, 0, 0, 0, 0, 1/6, 2/3]` — but every saturated input clamps to the *same* point and therefore the *same*
basis vector, the same hidden output, and the same probability. The model returns a **constant regardless of
input** because it has been driven to one end of its curve, not because the basis is zero.

Observed on **3,380 of 52,090 rows (6.5%)** whose `y_prob` is shared with a row that has different features.
Worst case: **403 consecutive windows** of one recording all returning `0.4597584307193756`. Rows 29764 and
29876 differ in `std` at the fourth significant figure yet both return the same probability.

This is a property of the shipped node-spline net on unbounded features, not a bug in the port. It should be
reported as a limitation: beyond the trained spline range the TinyKAN is blind, and its explanation on those
windows is not informative.

## 6. Two traps fixed or documented

- **`KAN_DECISION_THRESHOLD`.** `scripts/train_final_model.py` wrote `0.30` into the generated header. The
  firmware correctly uses its own `decision_threshold = 0.50f` and never reads that constant. The **source** is
  now `0.50` so a future regeneration is correct. The header was deliberately **not** regenerated: retraining
  would change the weights and break comparability with every result above.
- **`y_pred` in `outputs/openox_tinykan_predictions.csv` is `y_prob >= 0.30`, not 0.50** (100% agreement with
  0.30, 75.6% with 0.50). Anyone reading that column raw will get the wrong operating point. The calibration
  script correctly ignores it and recomputes at 0.50.

## 7. What this evidence does not claim

- Not clinical SpO₂ accuracy. TinyKAN outputs risk probability, not saturation.
- Not a motion classifier. OpenOximetry has no IMU or artefact labels.
- Not out-of-fold metrics. The frozen exported model was trained on all windows; the published GroupKFold
  numbers are separate and were not recomputed here.
- Not on-device faithfulness on live signals. AOPC is in-sample masking of canonical features.
- Not peak RAM. Not BLE or sensor latency.

## 8. Live bench capture — 7 October 2026 (this session)

First live capture with the instrumented build. Four findings, one of which is a defect that
affected every earlier live reading.

### 8.1 A sample-rate defect made every earlier live capture measure the wrong system

The board was draining the sensor FIFO at **3.12 Hz, not 50 Hz** — a factor of 16 low. Two
independent factors multiplied:

- `sampleAverage = 4` divides the FIFO output rate, so `sampleRate = 50` yields 12.5 records/s.
- `getRed()`, `getIR()` and `getGreen()` each call `safeCheck(250)`, so each **blocks waiting for a
  new record**. One loop pass therefore consumed 4 records while incrementing `samplesTaken` once.

12.5 / 4 = **3.125 Hz**. The print condition fires every 25 samples, so the model predicts a print
gap of 25 x 4 x 80 ms = **8.000 s** against a measured **8.005 s** (ten consecutive intervals, jitter
under 3 ms). A second consequence: Red and IR were read from **different records 80 ms apart**, so
`estimateSpO2(red, ir)` was pairing mismatched samples.

At 3.12 Hz a "500-sample window" is **160 s** of real signal rather than 10 s, so the features did not
describe the window the model was validated on. Fixed in `main.cpp`: `sampleRate` 50 -> 200 (keeping
`sampleAverage = 4`, giving 50 records/s), Red and IR taken from one record, and the hardware FIFO
drained directly instead of the library's 4-slot software ring. Verified after flashing: `Hz: 49.97-49.99`,
a status line every **0.496-0.503 s**, and samples advancing in exact 250-sample steps.

**Every live number published before this fix should be regarded as measured on a decimated stream.**

### 8.2 Still capture: not a sustained confirmed alert

120 s, thumb, enclosure closed, finger verified still (Red 59,116-63,669; IR 60,051-62,920).
The pipeline behaved exactly to contract — first decision at 750 samples (15 s), then one every 250
samples (5 s), `Samples` 0 -> 6750, `Win` 0 -> 24.

25 decisions, not a sustained 0.52-0.60 alert. Window 0 is 0.13 with both flags clear. Windows 1-6
are candidates (0.51-0.59); Confirmed starts at window 2. Windows 7-9 are 0.38, 0.38 and 0.46 with
Candidate NO and Confirmed NO. The log shows no disturbance there. Windows 10-24 return to about
0.52-0.60. 21 of 25 decisions were candidates and 19 of 25 were confirmed. Top attribution is often
`kurtosis`; window 2's largest bar is `maximum` (0.0413) beside `kurtosis` (0.0402).

This is not a perfusion result. No perfusion index was recorded. Oxygen was not varied. The firmware
running to contract does not explain the score.

### 8.3 Wiggle capture: in this log, movement dropped the score

In this capture, movement did not raise the score. It dropped it. Do not generalise that sentence
past this log.

| Phase | Windows | `KAN_Prob` | State |
|---|---|---|---|
| Still | 3-10 | 0.54-0.58 | `Confirmed: YES` |
| Wiggle (2-3 s) | 11-13 | **0.13** | `Candidate: NO`, `Confirmed: NO` |
| Recover | 14-19 | 0.56-0.57 | `Confirmed: YES` |
| Later disturbance | 20-24 | 0.17-0.31 | `Candidate: NO` |
| Later disturbance | 25-26 | **0.13** | `Candidate: NO` |

The sensor clearly registered the motion (`IR` jumped 63,000 -> 65,096 at the wiggle). Those windows
never became candidates, so the N=2 rule did not suppress a motion alert. The movement drove the score
below threshold. On a wearer who was at risk this would be a missed alert, and N=2 played no part.
There was no reference oximeter reading, so this is not established as a false negative.

The value 0.13 also occurs on still window 0, after finger placement, and on later-disturbance windows 25-26.
The logs record no feature values, so no mechanism is claimed for the repeated value. Together with
the PTT-PPG stress test (sit/walk/run mean P 0.552/0.576/0.538, candidate fraction ~0.99 on a recording
whose header SpO2 stayed >= 96%), this is domain shift. It is not evidence of hypoxaemia, and it is
not evidence that the model tracks waveform regularity rather than oxygenation. Oxygen was not varied.

### 8.4 Ambient light is a null result at the deployed LED current

The enclosure was built to reject ambient light. Tested properly, there is no light problem to reject.

With the **lid off** (sensor fully exposed) and nothing moving near the board, the room-light switch
was flipped at t = 30 s:

| | Before (15 s) | After (15 s) | Shift | Significance |
|---|---|---|---|---|
| Red | 62,822 +/- 422 | 62,574 +/- 420 | -248 counts (-0.40%) | 0.59 sigma |
| IR | 62,843 +/- 222 | 62,589 +/- 201 | -254 counts (-0.40%) | 1.15 sigma |

n = 33 per side. The table rounds both channels to 0.40%. Unrounded, red is 0.395% and infrared is 0.404%, so infrared is not under 0.4%. The shift is in the negative direction and indistinguishable from zero.
At `ledBrightness = 0x3F` the LED dominates the photodiode and the finger occludes it, so ambient
light does not reach the detector in any measurable amount.

**Consequence:** the enclosure cannot be validated as light rejection and that claim should be dropped.
What it does provide is mechanical protection. Repeatable finger placement was not measured. Seven of
the 14 features are amplitude-based, so placement can move them; that effect was not measured here. An earlier draft explained why ambient light *can*
disturb a photodiode; that is correct physics but does not occur here, and the measurement supersedes
the reasoning.

Two earlier light attempts are **discarded as confounded** and are not evidence: a lamp moved by hand
(showed a 9% fall in both channels, wrong sign for added light, consistent with contact change) and a
10 s flash timed against 2 s buckets.

## 9. Open, and small

| Item | State | Why it matters |
|---|---|---|
| `parity_complete` | **false** | The qualification scripts write this as `false` unconditionally; it is not computed from either feature criterion (see the note below). |
| `physical_gpio_verified` | **false** | GPIO D2 appears in the protocol string only; never probed on hardware. |
| `all_14_attributions_observed` | **false** | The timing image reports top-3 only. |
| Live still / wiggle / walk | **still and wiggle captured; walk not on disk** | Still and wiggle logs are at `outputs/live/2026-10-07_still.txt` and `outputs/live/2026-10-07_wiggle.txt`. No walk log is on disk. Oxygen was not varied; these logs are not live validation. See `live-capture-runbook.md`. |

**Note: reading `original_absolute_1e_4_failed_windows`.** The active feature gate is the scale-aware FP32
budget (`"pass"` in `scripts/parity_criteria.py`, frozen contract `native-fp32-local-scale-v1`). It passes
with **0** failed windows on the 2,464-window board replay (`locked-board-replay-v1`: `failed_windows: 0`,
candidate disagreements 0, max probability error 8.25e-7). `original_absolute_1e_4_failed_windows` (2,464 in that
report) reports the original uniform absolute 1e-4 raw-feature comparison, which is retained in the
report. It cannot be met in FP32: the nearest representable energy value already exceeds 1e-4.
The local-scale contract was adopted on 10 September 2026, before the repair. The 1e-4 comparison
is not the live gate, and it was not retired so the replay would pass.

Everything else in the remaining-work list is already measured.

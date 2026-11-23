# Edge KAN (spline on every edge, no W): host results (11 Nov 2026) and board parity and timing (12 Nov 2026)

Pre-registration: `docs/edge-kan-preregistration.md`, SHA-256 `cb70b259aa265b815fa9d1052c447d922793d1689fa561a75a0065450e5ba2f1`. That hash is recorded in every output below. The file was written before any edge KAN number existed. It is not committed, so git cannot prove the order. The host stages (E0–E4) ran on the laptop only. The board stages (12 Nov) are in their own section below. The shipped TinyKAN, its headers, `kan_inference.h`, `main.cpp` and `platformio.ini` were unchanged through the host stages: every stage hashes them before and after. On 12 Nov, two diagnostic envs were added to `platformio.ini` for the board job. No existing env changed. The IEEE paper is not edited.

Script `scripts/train_edge_kan.py`. Outputs in `outputs/edge-kan-v1/` (gitignored, like the other run directories). No hyperparameter was searched. Every stage ran once.

## What was built

| File | What it is |
|---|---|
| `scripts/train_edge_kan.py` | Edge KAN in PyTorch: `theta` has shape (d_out, d_in, 8) and `out_j = sum_i sum_k B_k(x_i) theta[j,i,k]`, with no W and no bias. Stages `cv`, `final`, `parity`, `aopc`, `size`. |
| `firmware/Arduino Nano 33/include/edge_kan_inference.h` | FP32 C engine with the TinyKAN basis expressions. Mean-perturbation XAI is computed incrementally (below). Included by no firmware translation unit. |
| `firmware/Arduino Nano 33/include/edge_kan_weights_v2.h` | `ekan16` export, fit on all 48,662 rows, 16 v2 inputs. Default for the engine. |
| `firmware/Arduino Nano 33/include/edge_kan_weights_ir14.h` | `ekan14` export, 14 IR inputs (`-DEKAN_WEIGHTS_IR14`). |
| `tests/edge_kan_probe.cpp` | Host probe: runs the incremental and full-pass XAI on every row. |

**Why 16 inputs, not 14.** The reported TinyKAN (repo `v2`) takes 16 inputs: the 14 IR features plus `ratio_of_ratios` and `ir_perfusion_index`. A like-for-like replacement therefore has 16 x 8 basis values, 1,632 multiply-adds and 6,528 B of theta. The 14-input model matches the 14 x 8 sizing in the request (1,440 multiply-adds, 5,760 B). It is reported as the secondary model.

**Incremental XAI.** Layer 1 is a sum of per-edge functions. Setting input i to its reference therefore moves hidden unit j by `sum_k (B_k(ref_i) - B_k(z_i)) theta[j,i,k]`, and only layer 2 has to be recomputed. The engine computes the 16 mean-perturbation deltas this way instead of running 16 extra full passes. The quantity is the same as the shipped firmware's: `P(x) - P(x with z_i = 0)`. E3.4 checks it against the full-pass method.

## Cost per decision (analytic; `final_model_report.json`, `size/summary.json`)

A "basis evaluation" is one scalar expanded into its 8 bump values, each with a divide, a `fabsf` and a cubic. The scaler, sigmoid and sort are not counted.

| 16 inputs | TinyKAN (shipped v2) | Edge KAN `ekan16` |
|---|---:|---:|
| Model floats | 321 | 1,632 (theta only) |
| Forward multiply-adds / basis evaluations | 308 / 13 | 1,632 / 28 |
| Forward + 16-delta XAI, multiply-adds | 5,236 (17 full passes) | 4,704 (incremental) |
| Forward + 16-delta XAI, basis evaluations | 221 | 252 |
| Cortex-M4 `-Os` object: code / constants (B) | 900 / 1,707 | 832 / 6,955 |

The object sizes come from PlatformIO `arm-none-eabi-g++` 7.2.1 (`-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os`). They are object-file sizes, not a linked image. TinyKAN's code figure includes `kan_predict_probability`, which is off the XAI path. The edge constants are 6,144 B (layer 1) + 384 B (layer 2) of theta, plus scaler, centres and feature-name strings.

One forward pass costs about 5x TinyKAN's. With the incremental XAI, the whole decision (P plus 16 deltas) costs about the same arithmetic as TinyKAN's 17 passes. Board timing (12 Nov, below) measured 3.2x and 1.08x. The 444.3 ms board decision block includes filtering and features, and it does not isolate the inference and XAI share.

## E0: same splits (passed)

GroupKFold(5) fold assignment and labels equal the TinyKAN comparator files row for row (`b_v2_tinykan16`, `b_v2_tinykan16_patient`, `a_v2_tinykan14`). The comparator metrics recomputed from those files reproduce `outputs/openox-v2/summary.json` exactly.

## E1: out-of-fold discrimination at threshold 0.50 (`cv_summary.json`)

Window level, pooled held-out windows, not N = 2 confirmed. 0.50 is TinyKAN's frozen cut, carried over unchanged and not calibrated for the edge KAN.

| Model | Grouping (groups) | AUC | Recall | Specificity | Precision | F1 | Gate (R >= 0.80, S >= 0.50) |
|---|---|---:|---:|---:|---:|---:|---|
| **Edge KAN `ekan16`** | recording (154) | **0.9229** | 0.8671 | 0.8892 | 0.9190 | 0.8923 | met |
| TinyKAN `b_v2_tinykan16` | recording (154) | 0.9075 | 0.8927 | 0.8680 | 0.9074 | 0.9000 | met |
| Logistic regression `c_v2_logreg16` | recording (154) | 0.8631 | 0.8214 | 0.8701 | 0.9016 | 0.8597 | met |
| Edge KAN `ekan16` | patient (87) | 0.9199 | 0.8621 | 0.8846 | 0.9155 | 0.8880 | met |
| TinyKAN `b_v2_tinykan16_patient` | patient (87) | 0.9069 | 0.8993 | 0.8635 | 0.9052 | 0.9023 | met |
| Edge KAN `ekan14` | recording (154) | 0.7042 | 0.6291 | 0.6594 | 0.7280 | 0.6750 | not met |
| TinyKAN `a_v2_tinykan14` | recording (154) | 0.7117 | 0.6104 | 0.6805 | 0.7347 | 0.6668 | not met |

`ekan16` per-fold AUC (recording): 0.908, 0.914, 0.941, 0.931, 0.929. Every fold's training loss fell (about 0.57 to 0.24–0.29) and stayed finite.

**R1:** `ekan16` meets the gate at 0.50, recording-grouped (primary) and patient-grouped.

## E2: paired patient-cluster bootstrap, edge minus TinyKAN (recording-grouped)

87 patients, 2,000 resamples, seed 20261111, percentile 95% intervals.

| Pair | Quantity | Difference | 95% interval | R2 reading |
|---|---|---:|---|---|
| `ekan16` vs `b_v2_tinykan16` | AUC | +0.0154 | +0.0061 to +0.0242 | edge KAN higher |
| | Recall at 0.50 | −0.0256 | −0.0457 to −0.0028 | edge KAN lower |
| | Specificity at 0.50 | +0.0211 | +0.0025 to +0.0394 | edge KAN higher |
| `ekan14` vs `a_v2_tinykan14` | AUC | −0.0076 | −0.0116 to −0.0036 | edge KAN lower |
| | Recall at 0.50 | +0.0187 | −0.0059 to +0.0381 | no difference detected |
| | Specificity at 0.50 | −0.0211 | −0.0347 to −0.0046 | edge KAN lower |

On the 16 inputs the edge KAN separates the classes slightly better. At the carried-over 0.50 cut it trades about 2.6 points of recall for about 2.1 points of specificity. Moving along a ROC curve trades recall against specificity, so the 0.50 split is a statement about this cut, not about which model is better. The interval resamples patients. It is not a subject-generalisation test, because a patient can still sit in both a training fold and the held-out fold under recording grouping. On the 14 IR inputs alone, neither architecture meets the gate and the edge KAN is slightly worse.

## E3: export and host parity (all passed; `final_model_report.json`, `parity/report.json`)

All 48,662 rows. Reference: float64 forward from the parsed header (FP32 constants, raw inputs rounded to FP32). Host compiler: zig 0.16.0 c++ (clang) `-std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu`.

| Check | `ekan16` | `ekan14` | Tolerance |
|---|---:|---:|---:|
| E3.1 torch vs header, max \|dP\| | 4.95e-7 | 1.57e-7 | 1e-5 |
| E3.2 C (incremental) vs header, max \|dP\| / max \|d delta\| | 3.99e-7 / 4.02e-7 | 1.47e-7 / 1.97e-7 | 1e-5 |
| E3.3 top-3 index-and-order mismatches | 0 | 0 | near-tie rule unused |
| E3.4 incremental vs full 17-pass, max \|dP\| / max \|d delta\| | 0 / 2.98e-7 | 0 / 1.79e-7 | 1e-6 |
| Decision flips at 0.50 (any comparison) | 0 | 0 | — |

**R4:** pass. This says the C engine evaluates the exported function on this laptop. It is not board parity.

Export notes: 0.94% of standardized inputs and 0.32% of hidden activations fall outside the [−3, 3] grid and are clamped. The resubstitution fit on the training rows (AUC 0.931) is not a generalisation estimate.

## E4: deletion AOPC, RQ2 protocol (`aopc/aopc_deletion.csv`, `aopc/summary.json`)

The test is in sample, on the all-row export, with the training-mean reference (deletion to 0), seed 20260911 and 5 random repeats. Before the edge KAN run, the same code reproduced `outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv` for the shipped TinyKAN within 5.0e-9. Gain intervals: patient-cluster bootstrap, 2,000 resamples, seed 20261111.

| Model | Subset (windows) | Firmware-absolute | Random mean | Gain, 95% interval | Windows beating own random | R3 |
|---|---|---:|---:|---|---:|---|
| **Edge KAN** | **candidates (27,586)** | **0.00964** | **0.00726** | +0.0024 (−0.0047 to +0.0088) | 65.4% | beats random (point) |
| TinyKAN (shipped) | candidates (28,661) | −0.00852 | −0.00042 | −0.0081 (−0.0129 to −0.0016) | 31.7% | does not |
| Edge KAN | all (48,662) | −0.2786 | −0.1434 | −0.135 (−0.143 to −0.127) | 37.2% | does not |
| TinyKAN (shipped) | all (48,662) | −0.3092 | −0.1612 | −0.148 (−0.157 to −0.139) | 19.2% | does not |

**R3, primary cell (candidates):** by the pre-registered rule, the edge KAN's shipped-style ranking beats random, 0.00964 against 0.00726, where TinyKAN's does not. The gain's 95% interval includes zero, though, so read it as "no longer below random", not as "better than random". TinyKAN's interval lies below zero. On non-candidates and all windows both models are below random, the same pattern as before (the deletion target is P(risk), and on unflagged windows deletion toward the mean raises it). `signed_risk` ordering beats random on candidates for both models (edge 0.135, TinyKAN 0.103). It is not what the firmware sends.

## Board parity and timing (12 Nov 2026, Nano 33 BLE on COM8)

Pre-registration: `docs/edge-kan-board-preregistration.md`, written before either image was built. Jason named the job on 12 Nov. Both images are diagnostics that run the shipped v2 r2 feature pipeline unchanged and evaluate ekan16 on each decision's features. Neither is a live image, and `main.cpp` has no path for the edge KAN.

| Image | Env | `firmware.bin` SHA-256 | Receipt |
|---|---|---|---|
| Parity | `nano33ble_ekan_context_parity` | `455e57b0…` | `outputs/firmware-v2-uploads/nano33ble_ekan_context_parity-20261112-upload-receipt.json` |
| Timing | `nano33ble_ekan_context_timing` | `cc5e724d…` | `outputs/firmware-v2-uploads/nano33ble_ekan_context_timing-20261112-upload-receipt.json` |
| Restored afterwards | `nano33ble_v2_f1a_spo2` | `f4a78cf6…` | `outputs/firmware-v2-uploads/nano33ble_v2_f1a_spo2-restore-after-ekan-20261112-upload-receipt.json` |

The images and their SHA256SUMS are in `outputs/firmware-ekan/`. The driver is `scripts/replay_edge_kan_board.py`.

**Board parity: passed, 1,243 of 1,243 decisions** (`outputs/edge-kan-v1/board-parity-20261112/report.json`). These are the v2 r2 prepared set's 21 fixture and 1,222 locked decisions, judged by the frozen v2 budgets.

| Check | Result |
|---|---|
| Gate 1, pipeline unchanged (features, TinyKAN P, metadata and states against the C++ host) | 0 failed; TinyKAN max \|dP\| 8.3e-7, as on 22 Oct |
| Gate 2, edge KAN against the host engine on host features | 0 failed; max \|dP\| 1.8e-7, max \|d delta\| 3.6e-7 |
| Gate 3, edge top-3 payload | 0 failed |
| Edge decisions at 0.50 that differ from the host | 0; candidates 642 on board and 642 on host (TinyKAN 673) |
| Edge top-3 set or order changed against the host | 0 / 0 |
| Arithmetic only (host engine on the board's own features) | max \|dP\| 1.2e-7, max \|d delta\| 1.2e-7, top-3 disagreements 0 |

A 5-case first look ran before the full set (`board-parity-firstlook-20261112/`; 31 decisions, 0 failed).

**Timing** (`outputs/edge-kan-v1/board-timing-20261112/report.json`). 21 fixture decisions × 10 repetitions = 210 timed decisions. Each kernel ran 5 times per decision, and the table uses the minimum. Every timed decision matched the host: TinyKAN P and edge P within 1e-4, and the TinyKAN copy bit-equal to the pipeline's own P. The empty-bracket calibration is 6 cycles. The kernels ran with interrupts enabled, on the same features.

| Per decision, 64 MHz | Median | Max |
|---|---:|---:|
| TinyKAN forward | 1.37 ms | 1.51 ms |
| Edge forward | 4.32 ms | 4.37 ms |
| TinyKAN XAI (17 passes, as shipped) | 22.60 ms | 24.76 ms |
| Edge XAI (incremental, P + 16 deltas) | 24.52 ms | 24.58 ms |
| v2 decision block, measured (filter, features, TinyKAN XAI) | 450.2 ms | 452.7 ms |
| Edge decision block, **derived** (block − TinyKAN XAI + edge XAI) | 451.3 ms | 454.6 ms |

- **Speed.** The edge forward pass is 3.2× TinyKAN's, not the 5× the operation count suggested. The edge XAI is 1.08× TinyKAN's (+1.9 ms).
- **Where the time goes.** Inference and XAI are about 5% of the decision block. The rest is filtering and features.
- **FIFO headroom.** The derived edge decision block (max 454.6 ms) stays under the 640 ms FIFO.
- **Same pipeline, different image.** The v2 block reads 450.2 ms here against 444.3 ms on the 22 Oct r2 timing image. The pipeline code is the same, so the gap is image-dependent. That is why the edge block is derived from same-image kernels and not measured in a live loop.
- **Stack.** The loop-thread stack watermark rose from 1,416 to 5,072 B of 32,768 B over the run. That figure covers this diagnostic's own kernels and reporting. It is not the shipped image's peak RAM.

**Restore.** The previous live image was reflashed: verify successful, receipt as above. Two serial reads straight after the flash returned no bytes. After a reset, without reflashing, it streamed `NO FINGER` lines at `Hz:` 50.1–50.9 (`build/ekan/boot-after-restore-20261112.txt`), the same as before the job (`build/ekan/ident-before-20261112.txt`).

## Full swap, 12 Nov 2026 (Jason: "lets try to do everything")

### Operating point: 0.50 kept (`operating-point/summary.json`)

Rule pre-registered in `docs/edge-kan-operating-point-preregistration.md`: keep 0.50 if it meets the gate in both modes, recording-grouped (the rule the Spline Network's 0.50 was kept under), else select by the v1 Youden rule. The edge KAN meets it, so **0.50 is frozen** and the 12 Nov board parity is the parity of the live threshold.

| Recording-grouped, 0.50 | Recall | Specificity |
|---|---:|---:|
| Edge KAN, window | 0.867 | 0.889 |
| Edge KAN, N = 2 confirmed | 0.848 | 0.901 |
| Spline Network, window | 0.893 | 0.868 |
| Spline Network, N = 2 confirmed | 0.876 | 0.883 |

Reported only: tuned by the v1 rule, the edge KAN would pick 0.40 (confirmed) and the Spline Network 0.42.

### Firmware

`-DOXY_MODEL_EKAN` replaces only `evaluate_kan_and_xai` in `oxygen_risk_pipeline.h` (same window, filter, features, 0.50, N = 2, training-mean reference). Without the flag nothing changes: the v2 images rebuild byte-identical (`cdefac1d` live, `9d08702e` timing, `8bcf8e8f` parity, `f4a78cf6` SpO2, `bd3a494d` peak-RAM sampler; `build/ekan/builds-step2*.txt`).

| Edge image (`outputs/firmware-ekan/`) | `firmware.bin` | Flash (linker) | Static RAM | v2 counterpart |
|---|---|---:|---:|---:|
| `nano33ble_ekan_f1a` (live) | `6a80fcd8…` | 348,884 B | 94,032 B | 343,620 B / 94,032 B |
| `nano33ble_ekan_pipeline_timing` | `d1cccebe…` | 342,772 B | 92,320 B | 337,524 B / 92,320 B |
| `nano33ble_ekan_looptiming3` | `a3a0114d…` | 348,988 B | 96,344 B | (rebuild) 348,972 B |
| `nano33ble_ekan_f1a_peakram_sampler` | `fcf99e3b…` | 355,916 B | 95,472 B | 350,660 B / 95,472 B |

### Board: integrated image, parity and timing (`board-pipeline-20261112/report.json`)

The edge KAN running inside the shipped pipeline (`nano33ble_ekan_pipeline_timing`, the v2 timing image plus the flag), replaying the v2 r2 prepared set.

- **Functional:** 1,243 of 1,243 decisions pass. Metadata, features (v2 budgets), P (1e-4, same side of 0.50), Candidate/Confirmed/streak, top-3 order and deltas, and `two_wavelength_valid` all match. Max \|dP\| is 1.8e-7, and the top-3 is identical to the host on every decision. There are 642 Candidates and 308 Confirmed.
- **Timing:** fixtures × 10, as the v2 timing run. Subsequent decision block: **median 433.9 ms, p95 435.4, max 435.7**, against the v2 r2 image's 444.3 / 445.9 (22 Oct). The first decision after reset is 350.2 ms, the FIFO headroom ratio is 1.47 and calibration is 6 cycles.
- **Reading:** the kernel run above timed the edge XAI at 1.9 ms *more* than TinyKAN's, so the 10 ms lower block is not read as the edge KAN being faster. The two blocks come from different images on different days, and image-to-image variation of about 6 ms was seen above. A same-day v2 control (reflashing `9d08702e…`) was planned, but the permission check blocked that flash, so it was not run.
- **Hardware FPU follow-up (19 Nov 2026):** enabling the Cortex-M4F FPU via the opt-in `*_fpu` envs passes all 1,243 replayed decisions identically (0 flag, state, or top-3 differences vs soft float) and reduces median decision-block latency 22.0× from 433.9 ms to **19.71 ms** (p95 19.77 ms; 0.49 ms forward, 3.00 ms incremental XAI). See `docs/edge-kan-fpu-results.md` and `outputs/edge-kan-fpu/summary.json`.

**L0 idle on the edge live image** (`outputs/live-ekan/2026-11-12/L0_ekan_f1a_idle_60s.txt`, receipt `nano33ble_ekan_f1a-20261112`): 120 `Hz:` lines at 50.09–50.12, all `NO FINGER`, so there is no rate collapse. The boot banner was missed after the flash; the receipt binds the image.

### Host analyses (`paper-analyses/summary.json`)

Pre-registered in `docs/edge-kan-paper-analyses-preregistration.md`. The same code, seeds and readings as the paper's, with the edge KAN swapped in. The Spline Network is recomputed beside it and reproduces the paper: SaO2-draw recall 92.5 / specificity 89.3, 2,622 FP from 20 recordings, frozen-loop AUC 0.919 / 0.559, a miss rate of 0.417 at 1–2 points below the cut, and dark-minus-light recall −5.9 (−14.9 to +2.1). Fitzpatrick −7.0 (−14.2 to −1.1) also matches.

| Item | Edge KAN | Spline Network (paper) |
|---|---|---|
| AUC / recall / specificity (95% patient CI) | 0.923 (0.909–0.935) / 86.7 (83.3–89.7) / 88.9 (86.4–91.0) | 0.908 / 89.3 / 86.8 |
| Edge minus LogReg | AUC +0.060, recall +4.6 (both differ); specificity +1.9 (no difference detected) | — |
| Parameters | 1,632 | 321 |
| Median P, negative / positive windows | 0.117 / 0.878 | 0.092 / 0.890 |
| Five folds, AUC range | 0.909–0.941 | — |
| Six seeds: AUC, recall, specificity (mean ± SD) | 0.922 ± 0.001, 86.8 ± 0.3, 88.4 ± 0.4 | 0.908 ± 0.002, 89.5 ± 0.3, 86.7 ± 0.3 |
| Miss rate 1–2 points below the cut | 0.502 | 0.417 |
| False positives; recordings holding half | 2,202; 20 | 2,622; 20 |
| AUC off / on the 2,006 frozen-loop rows | 0.929 / 0.601 | 0.919 / 0.559 |
| Flags vs arterial SaO2 < 90 at 3,203 draws (recall / specificity) | 90.0 (86.4–93.3) / 92.5 (88.7–95.6) | 92.5 / 89.3 |
| Label 20 s earlier / later, AUC change | +0.028 (0.024–0.032) / −0.043 (−0.047 to −0.039) | +0.023 / −0.034 |
| Skin tone, recall dark − light: Monk dorsal (primary) / Fitzpatrick | −2.0 (−11.7 to +6.0) / −3.7 (−10.7 to +2.1); no difference detected | −5.9 (−14.9 to +2.1) / −7.0 (−14.2 to −1.1) |
| PTT-PPG subject 1, mean P sit / walk / run | 0.04 / 0.02 / 0.06 (red and IR swapped: 0.97 / 0.77 / 0.87) | 0.07 / 0.10 / 0.09 (0.89 swapped) |

**Explanation behaviour, in sample, shipped header, training-mean reference.**

| On candidates (edge 27,586; Spline Network 28,661) | Edge KAN | Spline Network |
|---|---:|---:|
| Output clamped (edge: all 12 hidden units at ±3) | 0% | 39.7% |
| Zero-delta slot in the transmitted top three | 0% | 36.2% |
| Largest of 16 deltas below 0.01 | 0% | 48.3% |
| Top-ranked feature lowered P | 25.3% | 60.8% |
| Ratio of ratios top-ranked, positive / negative windows | 68.4% / 90.3% | 50.8% / 90.4% |

The median-P TP (P 0.902) and FP (P 0.817) windows both have 16 non-zero deltas. Both have ratio of ratios first (+0.073 and +0.036), and their second and third inputs differ: kurtosis and skewness for the TP, the band-power ratio and kurtosis for the FP.

**Deletion test (RQ2), gain over random with patient-bootstrap 95% interval** (`paper-analyses/aopc_gain_intervals.csv`, `deletion_range.csv`)

| Candidates | Edge KAN | Spline Network (paper) |
|---|---|---|
| In sample, training mean | +0.0024 (−0.0046 to +0.0092), not distinguishable | below random |
| Out of fold, training mean | −0.0023 (−0.0113 to +0.0052), not distinguishable | does not beat random |
| In sample / out of fold, label-0 reference | +0.344 / +0.350, beats random | beats random |
| Windows with no deletion range | 0% | — |
| Firmware share of each window's range (training mean, target P) | 0.525 against random 0.395; gain +0.130 (+0.110 to +0.153), **beats random** | worse than random within the range |
| Target logit, training mean | gain +0.210 (0.162 to 0.258), beats random | — |

**Reading for RQ2.** For the edge KAN, the shipped training-mean ranking is no longer worse than random. Within each window's range it ranks better than random. Its raw deletion gain on P is still not distinguishable from zero, in sample or out of fold. On all windows and on non-candidates it stays below random: the deletion target is P(risk), and on unflagged windows replacing toward the mean raises it.

### Live session with Jason (`outputs/live-ekan/2026-11-12/SESSION.md`)

Module 2, seated rest. One wearer, oxygen not varied: these are device-behaviour results, not oxygen results.

| | Edge KAN (12 Nov) | Spline Network (paper) |
|---|---|---|
| 180 s rest sit, live image | 36 decisions, P 0.02–0.45, none Candidate; ratio of ratios top on 35; `Hz:` 50.05–50.09 | 35 decisions, P 0.07–0.10, none Candidate; ratio of ratios top on 29; 49.94–50.09 Hz |
| Sit inputs outside training range | 0 of 576 | 0 of 560 |
| Loop timing, dashboard connected: intervals losing ≥ 1 record | 0 of 34; in-loop decision step median 447.7 ms | 0 of 34; 445.1 ms (15 Oct, pre-r2 image) |
| Peak RAM, probe image, finger on, dashboard connected | 86,640–87,392 B (valid) | 86,576–87,348 B |

The board was restored to `nano33ble_v2_f1a_spo2` (`f4a78cf6…`, verified, 50.08–50.15 Hz).

### Still owed

1. **Same-day v2 timing control**, if wanted. It needs `9d08702e…` flashed, and that flash was blocked.
2. **Not rerun:** host stream parity over 71,462 decisions, gradient×input, and the module-1 reference-shift figure (it needs module-1 captures).
3. **Paper.** Swapped to the edge KAN on 12 Nov (Option A, `docs/ieee-draft-edge-kan-swap-proposal.md`; evidence map, seventeenth pass; extract `draft-2026-11-12c.txt`; 7 pages).
4. **Terminology.** CONTEXT.md says TinyKAN is not an edge KAN, which stays true; the edge KAN is a separate model. Naming it in deliverables is Jason's call.

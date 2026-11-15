# Edge KAN with the label-0 reference on the board (pre-registration)

Written 14 Nov 2026. No label-0 edge-KAN image had been built or flashed, and no firmware or host-probe source had changed for it. Jason chose this job on 14 Nov 2026 ("Flash the label-0 reference"). It is option (a) of R1 in the simulated review (`docs/ieee-draft-gemini-package-audit-2026-11-14.md`). The model is ekan16 as exported (`edge_kan_weights_v2.h`). Nothing is retrained, and the threshold stays 0.50.

## Why, and what is registered

The draft's Section I criterion has two parts:

- the flag meets the gate of Section III-C;
- the shipped ranking beats random on the windows it flags.

The training-mean ranking fails the second part. On the host, the label-0 median reference makes the same mean-perturbation ranking beat random on candidates: 0.7207 against 0.3771, gain +0.344 (+0.336 to +0.351), and +0.350 out of fold (`outputs/edge-kan-v1/paper-analyses/`, H10; `outputs/edge-kan-v1/additive-deletion/`). That result is already known, so it is **not** registered here.

This job asks only whether the board computes the same function with that reference. It registers two things:

- board parity on the 1,243 prepared decisions;
- the decision-block timing, which is reported with no pass or fail.

The deletion test also deletes to the same reference, which partly favours a ranking built on that reference. The draft already says so, and that caveat stays.

## Reference

`EKAN_XAI_REF_Z[16]` is the in-sample label-0 median used on the host:

- inputs standardized by the ekan16 header's FP32 scaler, z = (x − EKAN_SCALER_MEAN) / (EKAN_SCALER_STD + 1e-10), as `edge_kan_paper_analyses.EdgeModel.standardize`;
- the median taken over the 19,867 label-0 rows of `outputs/openox-v2/window_features.csv`;
- the result rounded to FP32.

These are the code and values of `scripts/edge_kan_additive_deletion.py` (`ref0`). Model input order, as shortest round-trip FP32 literals:

```
mean 0.0021604758221656084            std -0.10124751925468445
minimum 0.09709542244672775           maximum -0.10771775245666504
value_range -0.101768359541893        rms -0.10124099254608154
skewness 0.13236039876937866          kurtosis -0.22007694840431213
energy -0.043084800243377686          derivative_std -0.0960390642285347
dominant_frequency_hz -0.2361954301595688
bandpower_0_5_2_hz -0.03288060426712036
bandpower_2_5_hz -0.03570161759853363
bandpower_ratio_0_5_2_to_2_5 -0.004364212043583393
ratio_of_ratios -0.24717028439044952  ir_perfusion_index -0.1249193623661995
```

P(reference) = 0.1126 (float64 host model).

**Check, done before this file was written** (host only; scratch script, nothing written to the repo). With exactly these 16 floats, the host reproduced the registered in-sample candidate values of `additive-deletion/aopc_gain_intervals.csv` (`label0_fp32`, `mean_perturbation`):

| | Host | Registered | Difference |
|---|---|---|---|
| Candidates | 27,586 | 27,586 | 0 |
| Firmware-absolute AOPC | 0.7206783857 | 0.7206783857 | 0 |
| Random mean | 0.3771180351 | 0.3771180351 | 0 |

## Firmware change (to be made after this file is committed)

1. **A new flag, `-DEKAN_XAI_REF_LABEL0`.** The existing `OXY_XAI_REF_LABEL0` is not reused. It switches TinyKAN's reference in `kan_inference.h`, and the context-parity image runs TinyKAN inside the v2 pipeline, so reusing it would change the v2 RESULT words and break gate 1 below. The existing `#error`s that forbid `OXY_MODEL_EKAN` with `OXY_XAI_REF_LABEL0` stay.
2. **A new header, `include/edge_kan_xai_reference.h`.** It holds `EKAN_XAI_REF_Z[16]` (the literals above) and defines `EKAN_XAI_REFERENCE(i)` as `EKAN_XAI_REF_Z[(i)]`. `edge_kan_inference.h` includes it under the flag, before its `#ifndef EKAN_XAI_REFERENCE` fallback to 0.
   - Under the flag, a missing header or an undefined macro is an `#error`, so the image can't silently run the training mean.
3. **Identity strings gain ` XAI_REF=LABEL0` under the flag:**
   - `CONTEXT_PARITY_EKAN … MODEL=EKAN16 XAI_REF=LABEL0`;
   - `TIMING_REPLAY_V2 … PAIRS=IR,RED MODEL=EKAN16 XAI_REF=LABEL0`.
4. **New envs** (each extends the archived env and adds only the flag):
   - `nano33ble_ekan_context_parity_ref0`;
   - `nano33ble_ekan_pipeline_timing_ref0`;
   - `nano33ble_ekan_f1a_ref0` (live; see "Live image").
5. **Floating point is unchanged.** The images build with software float, as every image in the paper; the FPU change on the public branch is not part of this job.
6. **The host probe `tests/edge_kan_probe.cpp`** is built a second time with the flag (zig c++, `-ffp-contract=off`, as in `train_edge_kan.py`), giving `edge_kan_probe_ekan16_ref0.exe`.

## Gates before any flash (each must hold, or the job stops)

- **G1, flag off.** Clean builds of the three archived envs it extends match their archives: `nano33ble_ekan_context_parity` `455e57b0…`, `nano33ble_ekan_pipeline_timing` `d1cccebe…`, `nano33ble_ekan_f1a` `6a80fcd8…`. This was checked at HEAD `47ad8f2` before this file was written, and all three match. It is checked again after the source change, and the same three must still match.
- **G2, the reference header.** The 16 header literals parse back to the FP32 values above bit for bit. G2's host reproduction is the check above.
- **G3, host engine against Python.** The flag-on C++ probe (incremental method) is run on all 48,662 windows' features and compared with the Python float64 brute-force label-0 deltas (`analyze_physio_concept.perturb_against(EdgeModel, z, ref0)`). Pass:
  - probability within 1e-4 and each delta within 1e-5 (the frozen v2 budgets);
  - every top-3 equal to the stable |delta| order of the Python deltas, except where the 3rd and 4th |delta| differ by less than 2e-5.

  The maxima are reported. The incremental path has only ever run with a reference of 0, which is why this gate exists.
- **G4, images.** The three ref0 images are built and archived with `scripts/archive_firmware_env.py` (`firmware.bin`, `SHA256SUMS`, build receipt). Only archived files are flashed.

## Board (all 1,243 decisions of `outputs/parity/v2-board-prepared-r2`: 21 fixture + 1,222 locked; every failure reported, none excluded)

**(A) All-delta parity, `nano33ble_ekan_context_parity_ref0`, driven by `scripts/replay_edge_kan_board.py`.**

The checks of `docs/edge-kan-board-preregistration.md` are unchanged:

1. pipeline unchanged against the C++ host v2 rows;
2. edge KAN within the frozen v2 budgets against the flag-on host probe;
3. payload top-3 equal to the stable |delta| order of the board's own deltas.

Added for this job:

4. **Board deltas against the Python float64 label-0 deltas,** within the same budgets (probability 1e-4, each delta 1e-5). This catches a reference that stayed 0 in both C++ builds.
5. **The `H` identity must contain ` XAI_REF=LABEL0`,** or the driver stops before sending samples.

Pass = every decision passes 1–5.

**(P) Deployed pipeline, `nano33ble_ekan_pipeline_timing_ref0`,** run as `board-pipeline-20261112`.

- Functional: all 1,243 decisions. The checks are those of that report: meta, features, probability, states, top-3 order, top-3 deltas, valid. The expected top-3 deltas are the flag-on host's, and they are also checked against the Python label-0 deltas (1e-5).
- Pass = 0 failed decisions.
- Timing is reported, not gated: the median, p95 and max of subsequent decision blocks, against 433.9 ms for the training-mean image. The op count per decision is unchanged (1 forward and 16 incremental replacements), because each replacement reads a flash constant instead of writing 0.

**Ending the session.** Before the first flash, the image on the board is identified over serial and recorded. It is reflashed at the end with a receipt, whatever the results.

## Live image

`nano33ble_ekan_f1a_ref0` is built and archived (G4) so that the "sensor image that would ship" exists with this reference, and its flash size is reported.

- **BLE packet:** unchanged.
- **New serial banner,** printed under the flag at boot and on each DTR rise, as the v2 ref0 image does: `XAI_REF=LABEL0: edge KAN Mean-Perturbation reference is the label-0 median of each standardized input, not the training mean`.

Jason chose on 14 Nov 2026, before any build ("Also flash it, 60 s idle"), to flash it for a **live idle check of 60 s** with no finger on the sensor. Pass = both of:

- every `Hz:` line of the capture is ≥ 49.5;
- the capture contains the banner above.

This shows only that the image boots, samples at rate and names its reference. It says nothing about oxygen.

## What changes in the paper, and what doesn't

**If (A) and (P) pass:**

- Table I gains a row: board replay with the label-0 image, 1,243 of 1,243, and its decision block.
- §IV and §VI say the label-0 image reproduces the host on the board, and that with it the ranking beats random on the windows the KAN flags (host deletion test, numbers above).
- The title can stay.
- The training-mean result stays in the paper as the finding about the shipped reference.

**What stays measured on the training-mean images only,** and is stated as such:

- peak RAM (the change adds 64 bytes of flash constants and no RAM);
- in-loop timing with Bluetooth;
- the 180 s rest sit;
- the in-sample rankings of Figs. 4 and 5.

**If any gate or board check fails:** the failures are reported, nothing is excluded, and the paper takes the retitle (option b of R1).

## Not done here

No retraining, no threshold change, no FPU build, no change to the archived training-mean images, no live sit, no peak-RAM probe.

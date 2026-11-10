# 0008 - P5: a third device model (five beat inputs) with a model saturation estimate, beside v1 and v2

**Status:** Recorded 30 Oct 2026 (job F2, named by Jason). Images built, archived, host-parity checked; **not flashed, no board result**. The IEEE paper reports v1 and is not edited. The shipped images are unchanged: v1 `outputs/firmware-v1-rollback/` (`1f2f95db…`), v2 `outputs/firmware-v2-f1a-r2/` (`cdefac1d…`). Results go to the final report only.

## Context

The host study (`docs/physio-concept-preregistration.md`, `docs/physio-concept-results.md`, `outputs/physio-concept-v1/`) trained a TinyKAN on five beat-level inputs: `r_beat, pi_beat, hr_bpm, sqi_template, sqi_ibi_ratio`.

- **P5.** Out-of-fold AUC 0.917 against the v2-16 TinyKAN's 0.908 on the same recording-grouped folds (Stage 2). 189 parameters against 321.
- **Concept regressor.** A KAN regressor for estimated ScalcO2 (`s_hat`) on the same inputs. Its cross-fitted error is MAE 3.78, RMSE 5.79 percentage points (Stage 4).
- **Explainer reference.** The device explainer's training-mean reference gave a top bar that opposes the flag on 77% of P5 candidates. Under the label-0-median reference that falls to 0.3% (Stage 3).
- **F1.** Job F1 (`docs/xai-reference-label0-preregistration.md`, `outputs/firmware-v2-ref0/`) built that reference as a firmware flag for v2.

## Decision

1. **A third model beside v1 and v2, never replacing them.** It is built only with `-DOXY_MODEL_P5`, in three new envs: `nano33ble_p5_f1a`, `nano33ble_p5_context_parity` and `nano33ble_p5_context_timing`. With the flag off, all 16 existing envs rebuild byte-identical (`outputs/firmware-p5/README.md`). The repo IDs v1/v2 and their reader-facing names stay as they are.
2. **Risk score.** The P5 TinyKAN with the **exact weights** of `outputs/physio-concept-v1/p5_final_model.npz`, not retrained (`kan_model_weights_p5.h`).
   - Beat-invalid windows take the training-time imputation, which standardizes to 0.
   - The inputs come from a C++ port of the frozen extractor (`include/physio_features.h`). The port runs on the same paired IR/red context filter and raw-window DC means that v2 ships.
   - The 14-feature/FFT path is not compiled on P5 images.
3. **Explanation.** Mean-perturbation over the five inputs with the **label-0-median reference** (`kan_xai_reference_p5.h`, F1's generator; medians equal `s3.json` P5).
   - Ranking, top-3, 0.50 and N=2 are unchanged.
   - The label-0 reference is used on the new images only. The shipped images keep the training mean.
4. **Concept.** `s_hat` comes from one all-window KAN regressor (`outputs/physio-device-v1/`, Stage 4 recipe, seed 42).
   - It is **displayed only**: BLE field 10 with one decimal, or `-`, and the serial `Feat:` line. It never feeds the risk score or the explanation.
   - Wording: "estimated ScalcO2" or "model saturation estimate", never SpO2 or an oxygen reading.
   - It is uncalibrated for the MAX30102. The honest accuracy is the cross-fitted Stage 4 numbers; the device model's resubstitution numbers are not an accuracy result.
5. **BLE.** Fields 0–9 keep their meaning: field 0 is still the toy Red/IR estimate, field 9 the display HR. Field 10 is `s_hat`, field 11 `label0`. The worst case is 192 of 219 characters.

## Evidence (30 Oct 2026)

- **Port.** Identical peaks on 46/46 synthetic and 400/400 real windows. Four features bit-equal, `sqi_template` within 5e-16 (`outputs/parity/p5-features-unit-20261030/`).
- **Host parity, pre-registered budgets** (`docs/p5-parity-criteria.md`, committed before any OpenOximetry comparison). **18 of 71,462 decisions fail. `parity_complete` is false.**
  - Peak lists identical on 71,460.
  - 0 Candidate disagreements.
  - The failures sit at near-flat systolic peaks (`hr_bpm` refinement, one-sample peak shifts) and on the 1e-5 delta budget.
  - Detail: `outputs/parity/p5-host-20261030/`.
- **Deployed vs offline.** 3 of 48,625 P5 Candidate flips (0.0062%). `n_beats` agrees on 98.5% of windows.
- **Static RAM.** Live image 99,576 B (+5,544 B against v2 r2). Whole-device peak RAM was not measured.
- **Board session, 30 Oct 2026** (named by Jason; criteria fixed before any flash in `docs/xai-reference-label0-runbook.md`, section P5, plus the sit amendment). Detail: `outputs/live-v2/2026-10-30_p5-session.sidecar.md`.
  - Board parity: **pass**, 0 of 1,243.
  - Timing: **fail as written**, stopped at the first decision on the exact-equality check (probability 1 ulp off the x86 host).
  - Timing re-run under the amended check (parity budgets; amendment `d9b1f4e`, committed before the re-flash): **pass**. 210/210 decisions within budget. Subsequent-decision block median 425.2 ms, p95 426.6 ms, max 426.8 ms, against the 640 ms bound.
  - Live idle: **fail as written**. Rate 50.09–50.30 Hz holds, but the `MODEL=P5` banner is absent; banner delivery after a DTR rise is unreliable (1 of 4 probe opens).
  - Sit: **fail as written** on the banner clause only. Rate 50.03–50.09 Hz. 36 decisions, 0 Confirmed, 1 Candidate (Win 0, beat-invalid, P 0.91).
  - Restore r2: **pass**.
  - The board-gated D2/GPIO and peak-RAM items stay open for every image.

## Consequences

- Three models now coexist in one source tree, separated only by compile flags. Every later firmware change must re-run the 16-env (now 19-env) byte-identity table.
- P5 claims nothing about hypoxia detection on the MAX30102. The Subject-1 PTT finding (domain shift) applies to any model on this sensor until shown otherwise.
- `s_hat` is a concept display. A dashboard tile labelled "Model sat. est" must not be read as a saturation measurement.
- AOPC or faithfulness statements about the P5 explanation remain host, in-sample and functionally grounded only (Stage 3, `outputs/xai-reference-label0-v1/`).

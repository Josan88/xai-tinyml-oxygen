# 0010 - SpO2 est from the Maxim reference algorithm, on a new v2 live image

**Status:** Decided by Jason, 5 Nov 2026. Board session 6 Nov 2026 (`outputs/live-v2/2026-11-06_f1a-spo2_sit.sidecar.md`). He then said leave the new image on. The live image is `outputs/firmware-v2-f1a-spo2maxim/` (`f4a78cf6…6af0`), keep receipt `outputs/firmware-v2-uploads/nano33ble_v2_f1a_spo2-keep-20261106-upload-receipt.json`. r2 (`cdefac1d…`) remains the paper's image and the restore target only if he names that flash.

## Context

- The dashboard tile and the serial `SpO2_est:` field show a demo formula, `estimateSpO2()` in `main.cpp`: `95 + 10·(IR/(Red+IR) − 0.45)`, clamped to 0–100.
  - It uses raw DC levels only. It has no AC component, no pulse and no ratio of ratios, so it is not an oximetry method.
  - It cannot go below 90.5, its value at IR = 0, whatever the signal. On the r2 sit it read 95.7.
  - It is labelled `SpO2 est` and kept out of every decision and sentence, but a number made up from a formula is still a weak thing to put on screen.
- Maxim's MAXREFDES117 reference algorithm ships in the SparkFun MAX3010x library the firmware already uses. It is the standard pulse-based red/IR method: per-beat R = (AC_red/DC_red)/(AC_IR/DC_IR), the median over a 4 s window, then the curve SpO2 ≈ −45.060·R² + 30.354·R + 94.845.
- Reading the reference code showed one line that reads the IR value at the **red** peak index (`an_x[n_y_dc_max_idx]`) where the red line uses its own index.

## Decision

Jason, 5 Nov 2026: "I want to do it, it's better than using a made-up formula." He was told the costs first: a new image, board re-checks, still uncalibrated, and values below 90 can now appear.

1. **New image, new env.** `nano33ble_v2_f1a_spo2` = `nano33ble_v2_f1a` + `-DOXY_SPO2_MAXIM`. Every change sits under the flag. All nine archived envs (r2, v1, the r2 parity/timing builds, ref0 ×2, P5 ×3) rebuild byte-identical. The shipped r2 image (`cdefac1d…`) is unchanged and stays the restore target.
2. **Vendored, with one fix.** `src/spo2_rd117.cpp` is the library file with renamed symbols (`rd117_*`) and the IR line reading the IR peak index (`n_x_dc_max_idx`). `-DRD117_UPSTREAM_IR_INDEX` restores the upstream line, for the host test only. Maxim's notice is kept.
3. **Module** (`include/spo2_maxim.h`):
   - 50 Hz → 25 Hz by pair average; 4 s ring; first result after 4 s of contact, then every 1 s;
   - runs outside the `LOOP_PRIORITY_V2` bracket; reset on finger loss.
4. **Display rule.**
   - Shown only if valid and 70–100. 70–100 is the range over which ISO 80601-2-61 specifies oximeter accuracy; the curve itself reaches ~0 % at R 1.83.
   - Held through 2 invalid results, dash on the 3rd. No value before the first valid one.
   - Field 0 and `SpO2_est:` keep the `"%.1f"` format (e.g. `97.0`), or `---`.
5. **Kept out of everything else.** SpO2 est is not a TinyKAN input. It is not used by any decision, alert, explanation sentence, replay row or survey stimulus. The dead `printRiskStatus()` ("Low oxygen detected because SpO2 … < 90%") is not compiled on this image.
6. **Not extended.** v1, ref0 and P5 images keep the demo formula. Combining the flag with P5 or ref0 is a compile error.

## Comparison algorithm (serial only)

- **D8.** Robert Fraczkiewicz's algorithm (`MAX30102_by_RF`, commit `553ba619…`, MIT-style licence, notice kept) runs on the **same** 4 s buffers right after Maxim. It goes on a separate `SPO2CMP:` serial line, never on BLE or the tile. Fraczkiewicz's method:
  - RMS AC on detrended signals;
  - a red/IR Pearson-correlation gate of 0.8;
  - an autocorrelation periodicity gate of 0.5 (HR 40–180);
  - the same Maxim curve, in floating point.
- **D9: the only changes** to it, so the comparison stays a comparison:
  - (a) its `an_x/an_y` (`float[100]` ×2) are file-scope static, keeping 800 B off the loop-thread stack;
  - (b) its function-static `n_last_peak_interval` is at file scope, and `rfalg_reset()` restores it on finger loss.

  No threshold, range or curve is changed.
- **D10: what the comparison may conclude.** Only *which algorithm gives a value more often and moves less at rest on this sensor; not which is more accurate.* There is no reference oximeter and oxygen is not varied. `scripts/analyze_spo2_compare.py` puts that sentence at the top of its output.
- Switching the tile to the other algorithm would be a later decision by Jason, and a new image.

## Consequences

- **Uncalibrated.** The curve is Maxim's, fitted for their module. It is not fitted for this sensor, enclosure, LED current (~12.5 mA) or 16-bit pulse width. The number is a display value computed on the oximetry principle, not a blood-oxygen reading. The label stays `SpO2 est`. Never write "SpO2" alone or "oxygen level" for it.
- **Values below 90 can appear**, including next to an Alert, where an uncalibrated 86 can read as confirmation. Mitigations:
  - the tile hint "Red/IR estimate · uncalibrated · not the model";
  - the 70–100 gate;
  - the rule that SpO2 est never enters a sentence.

  Say this aloud in any demo.
- **A mostly-dash result is possible.** The peak threshold is clamped to 30–60 counts, tuned for Maxim's 18-bit settings. The host test cannot show the valid fraction, because its AC is chosen; the board sit (runbook criterion 4) answers it.
- **Paper unaffected.** The IEEE draft reports r2 and never mentions SpO2 est. r2's board evidence (parity, timing, sit, peak-RAM bracket) does not transfer to this image.
- **RAM +3,256 B static** (pio RAM line 94,032 → 97,288 B): the module state (1,648 B) plus `an_x/an_y` for both algorithms (1,600 B). The plan estimated +2.4 KB, but its own items add up to 3.2 KB. The peak-RAM bracket is r2's. Flash +3,072 B.
- **Host evidence** (`tests/test_spo2_maxim.cpp`, 13/13; numbers in the archive README):
  - The plumbing reproduces the curve: R 0.5/0.7/0.9/1.1 → 99/94/86/74.
  - On uncorrelated red/IR, Maxim reports valid 99–100 and Fraczkiewicz's algorithm rejects every window. This is the behaviour the comparison is about.
  - With red lagging IR by 2 raw samples, upstream gives 95–96 and fixed gives 96, so the IR-index fix did not move that synthetic case toward the curve. The fix rests on the code reading, not on a measured gain.
- **Cost on the board is unmeasured.** Fraczkiewicz's autocorrelation and `sqrt` run in soft float once a second. If the rate collapses (runbook criterion 2), the comparison is the first suspect, and the fix is a new image without it, which is Jason's call.

## Alternatives rejected

- **Keep the demo formula.** It is not an oximetry method, and its ~90.5 floor hides any low reading.
- **Use the upstream library unmodified.** That knowingly ships the IR-index line. Vendoring also makes the code host-testable (no Arduino dependency) and avoids symbol clashes.
- **Display Fraczkiewicz's algorithm instead.** It is a single-author method published as an Instructable. It is kept as the serial comparison, so the board sit can show whether it behaves better on this sensor before anyone switches.
- **Calibrate the module.** That needs a controlled-desaturation study against a reference, which is out of scope for this project and its ethics basis (ADR-0009 does not cover sensor sessions on other people).

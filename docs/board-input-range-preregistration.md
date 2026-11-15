# Board input range check (R11): host analysis pre-registration (11 Nov 2026)

**Status: written before any board value was compared with the training data. Host only, laptop, no board, no flash, no firmware or header writes.** Nothing shipped changes. Jason asked for this item on 11 Nov ("Do R11"). Commit this file before the script is written.

**Seen before fixing.** The capture's line structure only: 35 `Feat:` lines (one per decision) with all 16 inputs, the sidecar, and the shipped header's input order. No feature value was compared with any training statistic.

## Why

Roadmap item R11 of `docs/ieee-draft-panel-review-2026-11-10/` asks for the share of the second module's standardised inputs that fall outside the training range, because the device standardises MAX30102 counts with the dataset's training means and standard deviations (no unit mapping). The re-review of 11 Nov (`re-review-2026-11-11/`) graded this remainder `must_fix`.

## Data (fixed now)

- **Board values:** `outputs/live-v2/2026-10-22_f1a-r2_sit.txt` (SHA-256 `37221255…0860` in its sidecar), the second module at rest on the revised live image, the sitting the paper reports in Section IV. The 35 `Feat:` lines are parsed; each gives the 16 inputs of one decision as the firmware computed them, before standardisation.
- **Training values:** all 48,662 rows of `outputs/openox-v2/window_features.csv`, the rows the shipped model was fitted on.
- **Scaler:** `SCALER_MEAN` and `SCALER_STD` of `firmware/Arduino Nano 33/include/kan_model_weights_v2.h`.

## M0 checks (stop before writing a number if one fails)

1. 35 `Feat:` lines, each with exactly the 16 header inputs in header order, all finite.
2. The header scaler equals the mean and population standard deviation of the 48,662 training rows per input, within 1e-3 relative.
3. The capture file's SHA-256 equals the sidecar's.

## Measurements

Per input (16) and in summary:

- **Out-of-range share (primary):** the share of the 35 board values outside the training [minimum, maximum]. Because standardisation is monotone, this equals the share of standardised board values outside the standardised training range.
- Standardised board values z = (x − SCALER_MEAN) / SCALER_STD: median, minimum, maximum; share with |z| > 3.
- Percentile of the board median within the training distribution.

Inputs are grouped by how they respond when the signal is multiplied by a constant c: **linear in amplitude** (mean, std, minimum, maximum, value_range, rms, derivative_std: scale by c), **quadratic** (energy, bandpower_0_5_2_hz, bandpower_2_5_hz: scale by c²), **scale-free** (skewness, kurtosis, dominant_frequency_hz, bandpower_ratio_0_5_2_to_2_5, ratio_of_ratios, ir_perfusion_index).

## Readings (fixed now)

- **An amplitude-dependent input (linear or quadratic) is "out of distribution on the board"** if more than 10 % of its 35 values are outside the training range, or its board median lies outside the training 1st to 99th percentile.
- **If any amplitude-dependent input is out of distribution**, the paper narrows the device claim: the board score is not evidence that the classifier, as trained, transfers to the MAX30102, and the rest-sit result is reported as a behaviour of the device, not as a transfer test. If none is, the paper says so and keeps the existing caveat that the effect on the score is unmeasured.
- Scale-free inputs are reported descriptively; no reading attaches to them.
- The paper text says "seven are amplitude-based". Under the grouping above ten inputs depend on amplitude (seven linearly, three quadratically). The paper wording is corrected whatever the result.

## Limits known in advance

- One sitting, one wearer, one module, 35 decisions, oxygen not varied. A range check, not an accuracy test.
- "Outside the training range" is necessary evidence of a units problem, not proof that the score is wrong; inside the range is not proof that it is right.
- The scale-free ablation of R11 is not part of this check; the existing two-input ablation (`outputs/analysis-lamaakal-style-v1/e_ablation.csv`) is cited as it stands.

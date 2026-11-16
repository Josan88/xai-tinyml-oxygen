# Board input range check, R11 (11 Nov 2026)

**Host only.** Pre-registration `docs/board-input-range-preregistration.md` (eb986d6); script `scripts/analyze_board_input_range.py` (committed unrun as 1f7e12b). Board values: the 35 `Feat:` lines of `outputs/live-v2/2026-10-22_f1a-r2_sit.txt` (second module, rest sit, revised live image). Training values: the 48,662 rows of `outputs/openox-v2/window_features.csv`. M0 checks passed (35 × 16 finite values; header scaler equals the training mean and population SD; capture hash equals its sidecar).

## Result

- **0 of 560 board values lie outside the training range.** No input has |z| > 3 except three scale-free ones on 2 to 3 decisions each (skewness, kurtosis, band-power ratio).
- **Reading (pre-registered): no amplitude-dependent input is out of distribution**, so the device claim is not narrowed and the "unmeasured effect" caveat stays.
- **Descriptive:** the board sits at the low-amplitude end of the training data. For nine of the ten amplitude-dependent inputs the board median lies in the lowest 6 % of training amplitude (std, maximum, value_range, rms, energy, both band powers 4.1 to 4.3 percentile; derivative_std 2.5; minimum at the 94.8th percentile of a negative quantity, i.e. 5.2 % from the small end). The window mean is near zero on both sources.

Per-input table: `input_range.csv`. Summary and hashes: `summary.json`.

## Limits

One sitting, one wearer, one module, 35 decisions, oxygen not varied. Inside the range is not evidence the score is right. The scale-free ablation is not part of this check.

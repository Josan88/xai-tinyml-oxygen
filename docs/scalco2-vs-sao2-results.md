# Results: ScalcO2 (the label) against arterial blood SaO2 (8 Nov 2026, host only)

Pre-registration `docs/scalco2-vs-sao2-preregistration.md` (commit 456b006, amendment A1 in 6d2b8a0). Script `scripts/analyze_scalco2_vs_sao2.py` (6d2b8a0). Outputs `outputs/scalco2-vs-sao2-v1/` (f74af0d). Final report only; the IEEE paper, the shipped TinyKAN, its 0.50 threshold and N = 2 are unchanged.

## Question

Every reported number for the model is graded against window-median ScalcO2 < 90. ScalcO2 is SaO2 calculated from end-tidal O2 (Severinghaus 1979, per Fong et al. 2025). OpenOximetry also stores arterial co-oximetry SaO2 at each blood draw, with ScalcO2 on the same row, and the SpO2 of every pulse oximeter tested at that draw. At those draws, does the label agree with arterial blood at least as closely as the median pulse oximeter tested in the same lab?

## Verdict: PASS

| Gate part (all 16,709 draws, 218 patients) | ScalcO2 | Median of 54 benchmark pulse oximeters | ScalcO2 rank of 55 |
|---|---|---|---|
| P1 ARMS against SaO2, draws with SaO2 70–100 (lower is better) | 2.39 [2.22, 2.56] | 2.84 | 18 |
| P2 kappa, below 90 against SaO2 below 90 (higher is better) | 0.949 [0.940, 0.958] | 0.886 | 8 |

Brackets are patient-grouped bootstrap 95 % CIs (B 2000, seed 20261108). The gate uses point estimates. Benchmark devices are those with at least 100 valid paired readings.

## Readings

- **Below-90 decision at draws:** ScalcO2 < 90 caught 98.7 % of draws with SaO2 < 90 (sensitivity 0.987) and cleared 96.0 % of draws with SaO2 ≥ 90 (specificity 0.960). 386 of 16,709 draws disagree; 281 of those have SaO2 within 2 points of 90.
- **Bias:** +0.13 points overall (CI −0.09 to +0.37), 95 % limits of agreement −4.5 to +4.8. By SaO2 band: 70–80 bias +1.39, ARMS 3.39; 80–90 bias −0.23, ARMS 2.18; 90–100 bias −0.58, ARMS 1.33. ScalcO2 reads high at the lowest plateaus and slightly low near normal.
- **Recordings the model was trained on (154 recordings, 3,458 draws, 87 patients):** ARMS 2.25 [1.99, 2.53], kappa 0.967 [0.950, 0.982]; ScalcO2 ranks 14 of 31 on ARMS and 3 of 31 on kappa against the 30 devices with enough readings there.
- **Sensitivities (never change the verdict):** devices pooled by model number (39 benchmark models) give medians 2.87 / 0.878, ScalcO2 ranks 11 and 2. The plan's looser pulse-oximeter rule (0 < SpO2 ≤ 100) gives 2.88 / 0.884. Each blood-gas row as its own unit: ARMS 2.33, kappa 0.953.

## S1: the stored model flags against arterial blood (descriptive)

3,203 of the 3,458 draws in the 154 recordings were mapped to a 10 s window (255 fell in no kept window). Single-window flag at 0.50, stored out-of-fold scores, no retraining.

| Model | Recall vs SaO2 < 90 | Specificity vs SaO2 < 90 | Recall vs label | Specificity vs label |
|---|---|---|---|---|
| TinyKAN | 0.925 [0.889, 0.955] | 0.893 [0.853, 0.929] | 0.919 | 0.905 |
| Logistic regression | 0.860 [0.820, 0.896] | 0.899 [0.862, 0.932] | 0.850 | 0.901 |

On these windows the window label agrees with arterial blood (kappa 0.975, sensitivity 0.999, specificity 0.970). Swapping the reference from the label to arterial blood moves TinyKAN's recall by +0.006 and specificity by −0.011. These windows sit on stable plateaus, so their rates are higher than the full out-of-fold figures (recall 0.893, specificity 0.868 on 48,662 windows); they are a reference check, not a better headline.

## What this means

- The label holds up: at blood draws, ScalcO2 tracks arterial SaO2 more closely than the median pulse oximeter tested in the same desaturation lab, as a number and as the below-90 decision.
- The project's accuracy figures were already graded against a reference with oxygen varied (70–100 % SaO2 plateaus). This check shows that reference agrees with arterial blood.
- For the final report, the progress report's certified-oximeter item can be recast as this check (supervisor's call: keep, recast or defer). It still leaves the live board ungraded: oxygen was never varied on the MAX30102, and a reference oximeter at rest would test false alarms only.

## Limits

- Draws are taken on stable plateaus with the participant motionless. Agreement at draws does not describe transitions, where end-tidal O2 leads arterial saturation; many training windows sit in transitions.
- Windows with median 89–91 were dropped at export, so S1 under-represents draws near 90.
- The PPG is not time-synced to the 2 Hz stream (PhysioNet); header alignment is closer, not exact. S1 inherits it.
- If ScalcO2 were fitted to the blood samples, agreement would be optimistic; the data paper does not say it is.
- The benchmark devices are devices under test, of mixed quality and finger sites; the median device is a benchmark, not a standard. No ISO 80601-2-61 limit was used.
- Device identity is inferred from label spelling; the model-number sensitivity is the only check.
- Nothing here tests the MAX30102 or the board.

## Checks

- An independent recomputation (no import of the study script or `lamaakal_metrics`) reproduced ARMS_S, kappa_S, both device medians and counts, the P1/P2 decisions, the ARMS and kappa CIs, and the S1 rates.
- A second run into `build/scalco2-vs-sao2-v1/rerun-1/` gave byte-identical `draws.csv`, `devices.csv`, `bands.csv`, `s1_windows.csv` and `gate.json`.
- 49 synthetic tests pass (`tests/test_scalco2_vs_sao2.py`). Every pre-registered real-file count was asserted before the output directory was created.

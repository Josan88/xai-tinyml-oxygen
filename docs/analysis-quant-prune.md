# int8 / LUT / pruning ablation of v1 TinyKAN (host only, never deployed)

24 October 2026. Jason asked for host-only ablation rows: quantised and pruned copies of v1, with the AUC and size change reported and labelled "never deployed". The shipped FP32 model stays untouched. These are the Lamaakal et al. Table 7 / Table 10 int8 rows that [analysis-lamaakal-style.md](analysis-lamaakal-style.md) had listed as not ported. **No variant was written to a header, flashed, or timed.** Latency, peak RAM and energy are "not measured (never deployed)" for every row. Deploying any variant would invalidate board parity (2,464/2,464), the 249.7 ms timing, the on-device mean-perturbation payload, and every host replica check of `kan_model_weights.h` (ADR-0004).

Code: `scripts/ablate_quant_prune_v1.py`. Output: `outputs/analysis-quant-prune-v1/` (`ablation_quant_prune.csv`, `lut_error_header.csv`, `prune_mask_by_tensor.csv`, `fig_auc_vs_bytes.png`, `summary.json`).

## Method

- **(a) Out-of-fold.** The 5 v1 fold models were retrained with the notebook-12 recipe (80 epochs, seed 42, `split_index` folds). Each variant was applied to each fold model and scored on its held-out fold. The paired recording bootstrap (Section C, 2,000 resamples) gives the CI of AUC(variant) − AUC(FP32).
- **(b) Shipped header, in-sample.** Each variant was applied to the `kan_model_weights.h` constants and run on the header's own training table, as the AOPC run was. This measures |ΔP|, flips of the candidate flag at 0.50, and top-1 / top-3 agreement of the mean-perturbation explainer.
- **Quantisation** is simulated: symmetric int8 with zero-point 0, dequantised to float64. Weights use one scale per tensor or one scale per hidden unit. Activations are saturating (input scale from the 99.9th percentile). The LUT variants replace each of the 13 learned splines by a 64- or 256-entry int8 table.
- **Pruning** is global magnitude pruning over all 297 parameters (29, 74 or 148 zeroed), with or without a 20-epoch masked fine-tune on the training fold. Storage stays dense: unstructured zeros save no bytes without a sparse format.

**Reproduction guard.** The retrained FP32 fold models must match the frozen out-of-fold P within 1e-4, with zero flag flips. The recorded run reached a max of 8.4e-5, zero flips, and an AUC difference of −3.5e-7. The FP32 retrain is not bit-reproducible across runs on this machine. An earlier attempt of the same command, started while 14 CNN training processes loaded the CPU, reached 2.2e-4 on fold 1 and stopped at the guard (`build/quant-prune-full-attempt1-guardfail.log`). A separate check run (`build/quant-prune-check/`) produced numerically identical tables to the recorded run. Every reported Δ is paired against the run's own retrained FP32, so the retrain noise does not enter the deltas.

## Results (v1, 52,090 windows, 162 recordings)

| Variant | Header bytes | AUC | ΔAUC (95% CI) | Recall / Spec @0.50 | Header: flag flips | Header: top-1 agree |
|---|---:|---:|---|---|---:|---:|
| FP32 (shipped recipe) | 1,376 | 0.606 | — | 0.767 / 0.394 | 0 | 1.000 |
| int8 weights, per-tensor | 509 | 0.606 | −0.001 (−0.001 to +0.000) | 0.774 / 0.388 | 0.6% | 0.984 |
| int8 weights, per-channel | 597 | 0.606 | +0.000 (−0.000 to +0.000) | 0.771 / 0.389 | 0.7% | 0.983 |
| int8 weights + int8 activations | 705 | 0.604 | −0.002 (−0.006 to +0.001) | 0.762 / 0.400 | 3.9% | 0.818 |
| int8 weights + LUT-64 splines | 1,261 | 0.606 | −0.001 (−0.002 to −0.000) | 0.771 / 0.388 | 2.0% | 0.960 |
| int8 weights + LUT-256 splines | 3,757 | 0.606 | −0.000 (−0.001 to +0.000) | 0.770 / 0.391 | 1.7% | 0.958 |
| int8 weights + activations + LUT-256 | 3,865 | 0.603 | −0.003 (−0.007 to +0.000) | 0.759 / 0.401 | 3.9% | 0.809 |
| prune 10%, no fine-tune | 1,376 | 0.602 | −0.005 (−0.010 to +0.002) | 0.860 / 0.285 | 3.3% | 0.972 |
| prune 25%, no fine-tune | 1,376 | 0.604 | −0.002 (−0.011 to +0.008) | 0.901 / 0.216 | 20.3% | 0.884 |
| prune 50%, no fine-tune | 1,376 | 0.571 | −0.035 (−0.051 to −0.021) | 0.919 / 0.180 | 23.0% | 0.476 |
| prune 10% + fine-tune | 1,376 | 0.606 | +0.000 (−0.003 to +0.003) | 0.772 / 0.389 | — | — |
| prune 25% + fine-tune | 1,376 | 0.605 | −0.002 (−0.007 to +0.003) | 0.876 / 0.254 | — | — |
| prune 50% + fine-tune | 1,376 | 0.572 | −0.034 (−0.049 to −0.020) | 0.897 / 0.213 | — | — |

"Header bytes" are all header constants: weights, scaler, centres, steps, threshold, plus int8 scales and LUTs. The FP32 parameters alone are 1,188 B; the int8 parameters alone are 297 B. Header columns are in-sample (shipped header on its own training table). Fine-tune rows exist only out-of-fold.

![AUC vs bytes](../outputs/analysis-quant-prune-v1/fig_auc_vs_bytes.png)

## What it means

- **int8 weights cost nothing measurable in discrimination.** AUC is unchanged to three decimals, and the header shrinks from 1,376 B to 509–597 B. Mean-perturbation top-1 still agrees on about 98% of windows. Whether that holds on the board is unknown: no int8 kernel exists, and the shipped build uses software float.
- **LUT splines are not a size win at this scale.** 13 functions × 64 or 256 entries cost more bytes than the 297 parameters they replace. LUT-64 already changes the top-1 feature on 4% of windows.
- **Activation quantisation is where the explanation breaks.** AUC barely moves (−0.002), but top-1 agreement falls to 0.82 and ordered top-3 agreement to 0.33. Out-of-fold, a few saturated windows move P by up to 0.71. For an explainable model, "AUC unchanged" is not "model unchanged".
- **Pruning mostly moves the operating point, not the AUC.** At 10–25% the AUC CIs include zero, but recall at 0.50 jumps to 0.86–0.90 and specificity falls to 0.22–0.29. The cause is that v1 out-of-fold P is compressed around 0.5 (Section B), so a small shift crosses the cut. Fine-tuning restores the 0.50 point at 10% only. At 50% the AUC drops by 0.035 (CI excludes zero), and top-1 explanation agreement falls to 0.48.
- None of this changes the shipped system. The FP32 v1 is 1.2 kB of parameters already, so size was never the binding constraint. The decision block is 249.7 ms, and how that splits between inference and explanation is not measured.

## Limits

v1 only. Simulated quantisation, not an int8 kernel. No board latency, RAM or energy for any variant. Header agreement is in-sample. Pruning is unstructured, so it saves no bytes or cycles without a sparse format. Window-level metrics; folds n = 5.

## Reproduction

```text
C:\Python314\python.exe scripts/ablate_quant_prune_v1.py --jobs 4      # outputs/analysis-quant-prune-v1/ (refuses if it exists), about 4 min
```

The figure was redrawn once from the written CSV after a layout fix (clipped axis labels). The data did not change.

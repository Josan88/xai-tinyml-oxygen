# Host Mean-Perturbation deletion/insertion evaluation (ADR-0003)

11 September 2026, 11:03 MYT. Jason authorized choosing the next host-only job. This run is **AOPC on existing canonical window features**. No upload, COM8, GPIO, live capture, PTT, training, retune, or raw-waveform rewrite. Weights, threshold **0.50**, ADR-0005 N=2, and the numerical contract were not changed.

**This is in-sample feature-masking of the exported TinyKAN, not clinical or causal validation, not on-device XAI, and not a reason to call firmware explanations newly faithful.**

## What was measured

`scripts/evaluate_xai_faithfulness.py` on all **52,090** OpenOximetry windows (**162** recordings) from `outputs/openox_canonical_window_features.csv`. Each window is already the 14 extracted features. Ranking is frozen from the unperturbed Mean-Perturbation vector. Masking replaces one standardized coordinate at a time with **0.0** (training mean). Target is always P(low-oxygen risk).

Four metrics are stored separately (they are not interchangeable, even when paired gains match because every ranking shares the same endpoints):

| Metric | Formula in this run |
|---|---|
| Deletion AOPC | `(1/14) Σ_{k=1..14} (D_0 − D_k)` |
| Insertion AOPC | `(1/14) Σ_{k=1..14} (I_k − I_0)` |
| Deletion AUC | trapezoid of `D_k` over `k/14` |
| Insertion AUC | trapezoid of `I_k` over `k/14` |

Rankings: `signed_risk` (descending signed Δ), `firmware_absolute` (descending \|Δ\|, original index on ties — firmware sort), `reverse_signed`, five PCG64 random permutations, and their per-window mean.

Host model: float64 arithmetic on FP32 header constants (`kan_model_weights.h`). Optional native check used the existing host `probability_probe.exe` on **211** spaced windows (not the Nano).

## Window-mean results (all 52,090)

Mean unperturbed P = **0.50239**. All-zero (training-mean) P = **0.45230**.

| Ranking | Deletion AOPC | Insertion AOPC | Deletion AUC | Insertion AUC |
|---|---:|---:|---:|---:|
| signed_risk | 0.1246 | 0.1283 | 0.3796 | 0.5788 |
| firmware_absolute | 0.0443 | 0.0444 | 0.4599 | 0.4949 |
| random_mean | 0.0283 | 0.0254 | 0.4759 | 0.4759 |
| reverse_signed | −0.0746 | −0.0709 | 0.5788 | 0.3796 |

Mean curves at first step (k=1): signed deletion P **0.4428**; firmware-absolute **0.4801**; random_00 **0.4979**; reverse **0.5574**.

Versus each window’s own random-mean:

- signed_risk deletion AOPC is better on **99.74%** of windows (mean gain **0.0963**).
- firmware_absolute deletion AOPC is better on **77.58%** (mean gain **0.0160**).

`firmware_absolute` is the ranking the firmware actually uses. It is only modestly above random on the pooled set.

## Candidate windows (exported-model P ≥ 0.50)

**34,127** windows. This count is the **exported header model**, in-sample. It is not the GroupKFold sweep candidate count (36,482 at TinyKAN window 0.50). Do not substitute one for the other.

| Ranking | Deletion AOPC | Insertion AOPC | Deletion AUC | Insertion AUC |
|---|---:|---:|---:|---:|
| signed_risk | 0.1454 | 0.1417 | 0.4204 | 0.5900 |
| firmware_absolute | 0.0958 | 0.0981 | 0.4699 | 0.5465 |
| random_mean | 0.0576 | 0.0599 | 0.5082 | 0.5082 |

Firmware-absolute deletion AOPC beats that window’s random mean on **98.33%** of candidate windows (mean gain **0.0382**). Non-candidate windows (n=17,963): firmware-absolute mean deletion AOPC **−0.0536**, worse than random on most windows. That follows from ranking by \|Δ\|, which can mask protective features first; it is not a live still/wiggle/walk result.

## Checks that passed

- `python tests/test_xai_faithfulness.py`: **7/7**.
- Vectorized vs scalar NumPy on 211 windows: max P error **5.0e-16**.
- Host vs `probability_probe.exe` on those 211: max P error **1.53e-7**, max attribution error **2.40e-7** (gate 1e-5).
- For every ranking: D_0 = unperturbed P, I_14 = unperturbed P, D_14 = I_0 = mean-input P.

## What this does not show

- Not board Mean-Perturbation (timing image still reports only top-three; locked-set board replay has since been run (2464/2464 adopted-contract pass; not live-sensor faithfulness; parity_complete remains false)).
- Not faithfulness of the **top-3 payload**; curves use the full 14-feature order.
- Not out-of-fold TinyKAN (final exported model saw these windows in training).
- Not independent feature effects (derived PPG features are correlated; masked points can be impossible).
- Signed deletion’s first-step advantage is partly tautological: the same mean-replacement defines Δ and the first deletion step.
- Five random repeats are not 14! permutations.
- ADR-0003’s “true low-oxygen causal drivers” wording remains an overclaim relative to this output.

## Evidence

Directory (gitignored, write-once): `outputs/xai-faithfulness/openox-aopc-v1/`

- `summary.json` SHA-256 `3524d43ae47ed82216f3b8cefbd5f8dca37dabdc01722deb52d5a067dd7bf74c`
- `mean_curves.csv`, `window_metrics.csv`, `windows.csv`
- `curves_*.npz`, `inputs_attributions.npz`, `native_check.npz`
- Copied sources under `source/`

Header SHA-256 `123a305f0bb91a0bff79c298f7d35df3c113c660eede9afebdcd30ba96a3fe55`. Features CSV SHA-256 `e9494090659783947abebd7a8e3512d1d8d3271681d286c5ee1ea04c8eacc3f0`. Seed **20260911**, five random repeats. Duration **65.9 s**. Started 2026-09-11 03:01:29 UTC.

Reproduce into a **new empty** directory only:

```powershell
python tests/test_xai_faithfulness.py
python scripts/evaluate_xai_faithfulness.py --output outputs/xai-faithfulness/openox-aopc-repeat --probe build/parity/audit-release/probability_probe.exe
```

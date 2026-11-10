# 0001 - On-Device Mean-Perturbation Feature Attribution

We use Mean-Perturbation attribution ($N+1$ forward passes with zeroed normalized features) for TinyKAN on-device local explanations instead of post-hoc approximation libraries (KernelSHAP/LIME) or surrogate trees. This delivers deterministic, exact local sensitivity attributions directly on Cortex-M4 microcontrollers within ~15ms without allocating dynamic heap memory or storing background training distributions.

## Correction (2026-10-07)

The ~15 ms figure above is not supported by measurement. DWT cycle timing on the nRF52840 at 64 MHz puts a full decision block, including all 15 forward passes of the explanation, at about 250 ms CPU (median 249.7 ms, max 250.8 ms over 180 post-warm-up decisions), roughly 5% of the 5 s stride. The explanation passes were not timed separately. Timing used accelerated synthetic USB replay on a diagnostic image without live sensing or BLE. Evidence: `outputs/parity/timing-board-v1/report.json` (`latency_summary.subsequent_decisions`). The decision recorded above is unchanged.

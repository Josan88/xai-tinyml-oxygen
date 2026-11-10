# Numerical validation lock v1

Created before arithmetic repair on 10 September 2026. No native results from this validation set may be used to select a repair. Freeze a development candidate and its hashes before evaluating the locked windows; if it fails, preserve those results and do not call that set independent after using its failures for another repair.

- Contract: `tests/fixtures/numerical_parity_contract.json` (local-scale v1, adopted 2026-09-10).
- Contract SHA-256: `9e616224232854c9dcf3a42289ad0d27560f065462e9b4c41ee0a007b4914c49`.
- Locked manifest: `outputs/parity/validation_lock/numerical-validation-v1.json`.
- Manifest SHA-256: `341909bffbcaa396beb174b5e2403ea6aa864a8dd0aee0eb85207edbca5b2339`.
- Locked windows: **2,464** from **155** recordings.

Selection uses recording ID, start sample, and a fixed SHA-256 ordering seed; no feature values, labels, or model outcomes enter selection. Each window has all required future samples. Its full filter-input interval does not overlap any of the 2,557 development contexts, historical every-251st feature-check contexts, the first eight historical manifest replay segments, or another selected validation interval. Up to 16 windows per recording were selected. Recordings without enough unexposed context are not forced into the set.

The write-once manifest stores every key, selection rule, contract values/hash, data/metadata provenance, and firmware source hashes before repair. `scripts/lock_parity_validation.py` refuses to overwrite it. Recorded IDs remain under ignored outputs.

These are previously unevaluated windows from recordings already used in development. This is a numerical implementation check, not independent-subject, training/test, or clinical validation. Whole-recording Fourier resampling is still the reference preprocessing operation.

The 2,557 existing windows remain development data. The five-second versus full-recording probability discrepancy remains separate. Neither adoption of the contract nor a development pass constitutes completed parity qualification.

First native use occurred on 10 September at **11:28:08 UTC**, after the passing development candidate was frozen at **11:27:14 UTC**. The unchanged candidate passed all 2,464 locked windows. Original absolute 1e-4 feature comparisons still fail on all 2,464 windows and remain recorded. Candidate SHA-256: `1cfd4ed28b5aad114d1064a1a7323673b8725e434700e7b6303b5009ace74d0b`. The first-use receipt is `outputs/parity/validation_lock/numerical-validation-v1.use.json`; details are in [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md). This set is now evaluated evidence, not a fresh holdout for later repairs. Parity is not complete.

# 0011 - The edge KAN becomes the paper's reported model

**Status:** Decided by Jason, 12 Nov 2026 ("Option A: full swap"), applied to the IEEE draft the same day (evidence map, seventeenth pass; extract `docs/ieee-draft-extracts/draft-2026-11-12c.txt`; commit `7052f26`). **Open:** which image is the shipped live image (see Consequences).

## Context

- The reported model until 12 Nov was TinyKAN v2 (ADR-0006): a dense linear mix, one cubic B-spline per hidden node and one on the output, 321 parameters. CONTEXT.md already said it is not a KAN in Liu et al.'s sense.
- Its explanation failed the paper's RQ2 test. Under the training-mean reference, the firmware ranking was worse than random on flagged windows. The mechanism was the clamped output spline: it held 39.7% of candidates at a constant probability.
- On 11 Nov Jason asked for a real edge KAN: theta stored as (d_out, d_in, n_knots), out_j = Σ_i Σ_k B_k(x_i)·θ_jik, no W. It was built as ekan16 (16 → 12 → 1, 1632 parameters) with the TinyKAN v2 recipe, inputs, folds and threshold unchanged (`docs/edge-kan-results.md`).
- It was built **after** the RQ2 result was known. Every later step was pre-registered before its numbers existed:
  - `docs/edge-kan-preregistration.md`
  - `docs/edge-kan-board-preregistration.md`
  - `docs/edge-kan-operating-point-preregistration.md`
  - `docs/edge-kan-paper-analyses-preregistration.md`

## Decision

1. **The paper reports the edge KAN.** The Spline Network (TinyKAN v2) stays in the paper as "the earlier spline-activation network", the comparator that shows the clamp mechanism. The post-hoc build is disclosed in ¶9 and in the limitations.
2. **The threshold stays 0.50.** It was kept by the pre-registered rule: keep 0.50 if the gate holds at window level and for N = 2. Window level: 86.7 / 88.9. N = 2: 84.8 / 90.1.
3. **The firmware gets a switch, not a fork.** `-DOXY_MODEL_EKAN` replaces only `evaluate_kan_and_xai` in `oxygen_risk_pipeline.h`. The window, filter, features, debounce, BLE payload and training-mean reference are unchanged. Without the flag, every v2 image rebuilds byte-identical (`cdefac1d`, `9d08702e`, `8bcf8e8f`, `f4a78cf6`, `bd3a494d`).
4. **New envs, never the default:**
   - `nano33ble_ekan_f1a` (live)
   - `nano33ble_ekan_pipeline_timing`
   - `nano33ble_ekan_looptiming3`
   - `nano33ble_ekan_f1a_peakram_sampler`

   Archived images are in `outputs/firmware-ekan/`.

## Evidence (12 Nov 2026)

- **Host:** AUC 0.923; every paper analysis was rerun and the Spline Network numbers reproduce exactly (`outputs/edge-kan-v1/paper-analyses/`).
- **Board:** the integrated image passes 1,243/1,243 decisions, with a decision block of 433.9 ms.
- **Live:** a 180 s sit; 0 of 34 loss intervals with Bluetooth connected; peak RAM 86,640–87,392 B (`outputs/live-ekan/2026-11-12/SESSION.md`).
- **RQ2:** the ranking on flagged windows is not distinguishable from random (it was worse than random for the Spline Network). It beats random within each window's range, and it beats random under the label-0 reference. By the paper's own criterion, flag and explanation are still not shown to be one system.

## Consequences

- **Which image ships is still open.** The board runs `nano33ble_v2_f1a_spo2` (`f4a78cf6…`, Spline Network scorer, ADR-0010) by Jason's 6 Nov decision. The paper describes `nano33ble_ekan_f1a` (`6a80fcd8…`).
  - Option one: the edge image becomes the live default, with an SpO2-est variant built as `-DOXY_MODEL_EKAN -DOXY_SPO2_MAXIM`.
  - Option two: the paper calls the edge image a diagnostic build.

  Until Jason decides, no flash changes the board.
- **Not rerun for the edge KAN:**
  - host stream parity over 71,462 decisions
  - the 8-flag host-versus-device filtering check
  - the module-1 sit
  - a same-day v2 timing control (the flash was blocked)

  The paper says so.
- **Terminology.**
  - "TinyKAN" still names the Spline Network models (v1, v2) in the repo.
  - The paper says "KAN" for ekan16 and "earlier spline-activation network" for TinyKAN v2.
  - The repo ID is `ekan16`; the env and file prefix is `ekan`.

# 0005 - Consecutive Window Confirmation for Risk State Transitions

We require 2 consecutive positive classification windows ($N=2$, corresponding to 3.0–4.0s confirmation time) before triggering the `LOW_OXYGEN_RISK` state and visual alert. This eliminates transient false alarms caused by sensor movement or fingertip re-positioning while keeping alert latency well within safe clinical reaction windows.

## Correction (2026-10-07)

The 3.0–4.0 s confirmation time above does not match the deployed pipeline. The frozen design advances decisions with a 5 s stride (250 samples at 50 Hz) and filters each 10 s window inside a 20 s buffer with 5 s of past and 5 s of future context, so each decision is emitted 5 s after its window ends. N = 2 therefore spans two consecutive Candidate windows one stride apart, and a Confirmed alert arrives no earlier than 10 s of signal after the first Candidate window ends. Evidence: `firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h` (`PPG_STEP_SIZE`, `ContextSamples`); `docs/paper-sections-draft.md` §A.4. The N = 2 rule is unchanged. Its suppression of motion-induced false alarms has not yet been tested on live captures.

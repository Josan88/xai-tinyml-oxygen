# Edge KAN live session (12 Nov 2026): what needs Jason at the board

The three captures the paper reports for the shipped model, repeated on the edge-KAN images. Same
procedures, durations and validity checks as the v2 runs; nothing is retuned. Sensor module 2, lid on,
seated rest. Images are archived under `outputs/firmware-ekan/<env>/` and flashed with bossac from the
archive (never `pio run -t upload`). Captures go to `outputs/live-ekan/2026-11-12/`.

| Step | Image (env) | Jason | Capture | Judged by | v2 counterpart in the paper |
|---|---|---|---|---|---|
| L0 | `nano33ble_ekan_f1a` | not needed | 60 s, no finger | every `Hz:` ≥ 49.5; banner `Model: Edge KAN ekan16` | 60 s idle on f1a-r2 |
| L1 | `nano33ble_ekan_f1a` | finger on, still, 180 s | `cap.py COM8 180` | `Hz:` range; decisions, P range, Candidate count, top feature; input range (`analyze_board_input_range.py`) | 35 decisions at 0.07–0.10, none Candidate, ratio of ratios top on 29, 49.94–50.09 Hz |
| L2 | `nano33ble_ekan_looptiming3` | finger on, dashboard connected, 180 s | `cap.py COM8 180` | `analyze_loop_timing.py`: intervals with loss ≥ 1 record | 0 of 34 intervals (v2 looptiming3, dash) |
| L3 | `nano33ble_ekan_f1a_peakram_sampler` | identify (no finger), then finger on and dashboard connected, 180 s | `cap.py COM8 180` | `analyze_peak_ram.py --sampler` (checks pre-registered 30 Oct, unchanged) | 86,576–87,348 B |
| R | `nano33ble_v2_f1a_spo2` (`f4a78cf6…`) | not needed | 30 s | receipt verified, 0 `PR` lines, `Hz:` ≥ 49.5 | — |

One rerun per step is allowed if a check fails for a removable cause (for example the dashboard dropped);
the failed capture is kept. A sit is not an oxygen result: one wearer, oxygen not varied, no reference oximeter.

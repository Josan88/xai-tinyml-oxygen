# Edge KAN with the label-0 reference on the board: results (14–15 Nov 2026)

These are the results of `docs/edge-kan-label0-board-preregistration.md`.

- The pre-registration was committed as `9685e4d` before any image was built.
- The source change was committed as `86009af`.
- All three images are archived in `outputs/firmware-ekan-ref0/`.
- Every check fixed in the pre-registration is reported here, and nothing is excluded.

## Verdict

| Check | Result |
|---|---|
| G1 flag off | **pass** |
| G2 reference header | **pass** |
| G3 host engine against Python | **pass** |
| G4 images built and archived | **done** |
| (A) all-delta board parity | **pass**, 1,243 of 1,243 |
| (P) deployed pipeline | **pass**, 1,243 of 1,243; timing reported |
| Live idle 60 s | **fail as written,** on the banner clause only |
| Restore | **done** |

## Gates before flashing

**G1, flag off** (`outputs/edge-kan-label0/g1-flag-off-builds.tsv`). With the flag wiring in place, these rebuild byte-identical to their archives:

- the six archived edge-KAN envs (including the three the new envs extend: `455e57b0…`, `d1cccebe…`, `6a80fcd8…`);
- the three P5 envs;
- the two v2 label-0 envs;
- both peak-RAM envs.

The other envs I compared match a build of the pre-change HEAD made in a separate worktree. Those are v1, `v2_f1a`, `v2_f1a_spo2`, v2 context parity and timing, `v2_looptiming3`, and the two peak-RAM sampler envs. The sampler archives already differ from the HEAD build, so that difference predates this change.

**G2, reference header** (`outputs/xai-reference-headers/edge_kan_xai_reference.json`).

- The 16 literals round-trip to FP32 exactly.
- With them the host reproduces the registered candidate values exactly: 27,586 candidates, AOPC 0.7206783857, random mean 0.3771180351.
- The values equal `kan_xai_reference_v2.h`, because the two models share one scaler.

**G3, host engine against Python** (`outputs/edge-kan-label0/host-parity/summary.json`). The flag-on C++ probe was run on all 48,662 windows and compared with the Python float64 label-0 deltas:

| | Max \|dP\| | Max \|d delta\| | Top-3 |
|---|---|---|---|
| Incremental engine | 4.0e-7 | 3.9e-7 | identical on every window |
| Full-pass engine | 4.0e-7 | 4.0e-7 | identical on every window |

- **The reference is in effect.** All 48,662 rows differ from the training-mean deltas by more than 1e-5.
- **The flag-off probe matches the archived probe** in its `.text`, `.rdata` and `.data` sections. Only `.buildid` differs, because it holds the output path.

**G4, images.**

| env | `firmware.bin` SHA-256 | Flash (B) | Static RAM (B) |
|---|---|---|---|
| `nano33ble_ekan_context_parity_ref0` | `73ec66b4d04352d7852f33b18b0d6a2fa5b535851faeed5bf07d5b63e9f01c1a` | 344,092 (+64) | 92,272 (±0) |
| `nano33ble_ekan_pipeline_timing_ref0` | `26dcd3bc0774545d7521175de769a84b7063a52484cfbb875845fdbd7a4632a1` | 342,868 (+96) | 92,320 (±0) |
| `nano33ble_ekan_f1a_ref0` | `df3890205d4da1d25982dab4a5149ec5fa48e2d8501c9ee447eabd844dbc0eb8` | 349,164 (+280) | 94,040 (+8) |

Differences are against the training-mean image of the same role. The live image's extra 8 bytes of static RAM hold the banner state. The reference itself is a flash constant, as the pre-registration stated.

## Board (COM8; upload receipts in `outputs/firmware-v2-uploads/*-ekanref0-*`)

**(A) All-delta parity** (`outputs/edge-kan-label0/board-parity-20261114/report.json`).

- 1,243 of 1,243 decisions pass all five checks (pipeline, edge KAN against the flag-on host, payload, the Python label-0 deltas, identity).

| Board against | Max probability error | Max delta error |
|---|---|---|
| Python float64 label-0 | 2.9e-7 | 2.6e-7 |
| Flag-on C++ host | 1.8e-7 | 3.0e-7 |
| Host engine on the board's own features | 1.2e-7 | 1.2e-7 |

- Top-3 set and order are identical to the host on every decision.
- 642 edge candidates; no decision disagrees with the host at 0.50.

**(P) Deployed pipeline** (`outputs/edge-kan-label0/board-pipeline-20261114/report.json`).

- 1,243 of 1,243 functional decisions pass every check, including the added Python label-0 check.
- Max probability error 1.8e-7; top-3 identical to the host on every decision; 642 candidates and 308 confirmed.

Timing, reported and not gated:

| Decision block | Label-0 image | Training-mean image (`board-pipeline-20261112`) |
|---|---|---|
| Subsequent decisions, median (n = 180) | **434.3 ms** | 433.9 ms |
| p95 | 435.7 ms | 435.4 ms |
| Max | 436.0 ms | 435.7 ms |
| First decision after reset, median | 350.5 ms | 350.2 ms |

- FIFO headroom is 1.47 against the 640 ms the sensor FIFO holds.
- Every timed decision is functionally correct.

**Live idle 60 s, `nano33ble_ekan_f1a_ref0`** (`outputs/live-ekan/2026-11-15/ekan-f1a-ref0_idle.txt`).

- **Fail as written.** The rate clause holds: 121 `Hz:` lines, 50.09–50.19. The banner clause fails: the `XAI_REF=LABEL0` banner is absent.
- Three supplementary 8 s reopens (`…_reopen1-3.txt`) also show no banner. They are reported, and they don't change the verdict.
- The banner string is in the image. ADR-0008 records the same failure for the P5 live image: delivery of a banner after a DTR rise is unreliable on this core (1 of 4 opens there). Had I read it first, a different identity check would have been registered.
- The image on the board is established by bossac's verify against the archived binary (349,164 bytes) and by the receipt, not by the banner.

**Restore.** The image found on the board before the job was the public repository's FPU additive image (`nano33ble_v2_f1a_additive`, r3, `1f63b52f…`, ADR-0012). It was reflashed with bossac and verified, and a 30 s capture shows `Hz:` lines (`outputs/live-ekan/2026-11-15/restore-additive-r3_30s.txt`). That archive is not one the receipt tool knows, so its receipt was written by hand: `outputs/firmware-v2-uploads/nano33ble_v2_f1a_additive-20261115-restore-after-ekan-ref0-upload-receipt.json`.

## What this changes in the paper (as pre-registered)

**Both board checks pass.** So the paper can say that a board image with the label-0 reference reproduces the host on all 1,243 replayed decisions, and that with this reference the ranking beats random on the windows the KAN flags (host deletion test: +0.344, +0.336 to +0.351; out of fold +0.350). Its decision block is 434.3 ms. The training-mean result stays as the finding about the reference first shipped.

**Still measured on training-mean images only:**

- peak RAM;
- in-loop timing with Bluetooth;
- the 180 s rest sit;
- the in-sample rankings of Figs. 4 and 5.

**The live label-0 image was checked idle only,** and its banner check failed as written.

The deletion test deletes to the same reference, which partly favours a ranking built on it; that caveat stays.

## Ruling, 15 Nov 2026

The pre-registration says both "If (A) and (P) pass … the title can stay" and "If any gate or board check fails … the paper takes the retitle". The live idle check failed as written, on the banner clause only, so the two sentences point in different directions. Jason ruled on 15 Nov 2026 ("Keep title"): the retitle fallback concerns the two board checks that answer R1, (A) and (P), both of which passed; the idle fail is the known DTR-banner fault, reported as ADR-0008 reported P5's. The committed pre-registration is not edited. The draft took the keep-title wording (`scripts/apply_ieee_label0_board.py`; evidence map, 15 Nov section).

## Banner diagnostic and amended live idle check, 15 Nov 2026

Pre-registered in `docs/edge-kan-label0-banner-amendment.md` (`e914287`) before any of these runs. Jason asked: "Fix the banner check". The archived `nano33ble_ekan_f1a_ref0` (`df389020…`) was reflashed with a receipt (`outputs/firmware-v2-uploads/nano33ble_ekan_f1a_ref0-20261115-banner-diag-upload-receipt.json`). No finger was on the sensor.

**Cause found.** It is on the capture side, not in the firmware. pyserial 3.5's Windows `open()` raises DTR and then purges the receive buffer, and the image prints its banner on the first loop pass after the DTR rise, so the banner falls into the purge. In 20 alternating 8 s opens (`outputs/live-ekan/2026-11-15/banner-diagnostic/summary.json`):

| Method | Opens | With banner | `Hz:` lines per open |
|---|---|---|---|
| `cap` (as `build/v2/cap.py`) | 10 | 0 | 16–17 |
| `dtr-toggle` (`scripts/capture_serial.py`: drop DTR 1 s after opening, then raise it) | 10 | 10 | 16–17 |

The registered rule was "hypothesis supported if `dtr-toggle` shows the banner on all 10". It did.

**Amended live idle check: pass** (`outputs/live-ekan/2026-11-15/ekan-f1a-ref0_idle_amended-dtr-toggle.txt`). The 60 s capture with `dtr-toggle` has 121 `Hz:` lines, all at 50.09, and the banner. This is reported next to the original result, which stays **fail as written**.

**The same mechanism fits ADR-0008's P5 result** (banner on 1 of 4 opens). The P5 and v2 ref0 live images re-announce on each DTR rise in the same way, so `--method dtr-toggle` should apply to them too. Neither was tested here.

**Restore.** The additive r3 image (`1f63b52f…`) was reflashed and verified, with a receipt written by hand (`…-20261115-restore-after-banner-diag-upload-receipt.json`). A 30 s `dtr-toggle` capture shows 60 `Hz:` lines and no label-0 banner, as expected for that image.

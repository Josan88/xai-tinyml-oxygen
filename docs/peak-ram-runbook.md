# Peak-RAM runbook

Board job named by Jason on 25 Oct 2026: "Peak RAM: build the image, flash it, measure with no finger, then restore the current live image." The method is in [peak-ram-design.md](peak-ram-design.md). **Status, 25 Oct 2026.** Both probe images are built from commit `fcaa9a0` and archived under `outputs/firmware-peakram-prepared/`. **They have not been flashed.** Flash the archived files below. Do not rebuild on the day, and do not use `pio run -t upload`, which rebuilds and flashes an image nobody hashed.

| Step | Archive (`$ARCH`) | Env (`$JOB`) | `firmware.bin` SHA-256 | Identity on serial |
|---|---|---|---|---|
| (1) v2 probe | `outputs/firmware-peakram-prepared/nano33ble_v2_f1a_peakram` | `nano33ble_v2_f1a_peakram` | `8be6244f6621751dd30677ea2170074256786a06b22e3ef0c2f33f80cdb837cf` | `PR_INFO: v=1 img=nano33ble_v2_f1a_peakram model=v2 force=1 …`, then `PR: v=1 …` every 5 s |
| (2) v1 probe | `outputs/firmware-peakram-prepared/nano33ble_peakram` | `nano33ble_peakram` | `d5051a82b407bb9456af4bdacd53771421244bc35734f00bbc6deb5175757793` | `PR_INFO: v=1 img=nano33ble_peakram model=v1 force=1 …`, then `PR: v=1 …` every 5 s |
| (3) Restore | `outputs/firmware-v2-f1a-r2` | `nano33ble_v2_f1a` | `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2` | No `PR` lines; `Status: NO FINGER` with no finger; `Hz:` ≥ 49.5 |

Run (1), (2), (3) in that order. (1) comes first because v2 f1a-r2 is the image on the board now. If the session has to stop early, skip (2) but still run (3). Never leave a probe image on the board: its finger gate is held open and its `KAN_Prob`/`Candidate`/`Confirmed` values are dark-floor nonsense.

Each probe archive's `README.md` gives the image's region table, the probe-vs-live section deltas and the build proof. Clean rebuilds of `nano33ble` and `nano33ble_v2_f1a` on that tree reproduce `1f2f95db…` and `cdefac1d…`. Commands are bash (Git Bash), run from the repo root. The session takes about 15 minutes.

## 0. Pre-flight

```bash
cd /d/fyrp-xai-tinyml
PY=/c/Python314/python.exe                  # numpy, pandas, pyserial
BOSSAC=~/.platformio/packages/tool-bossac-nordicnrf52/bossac.exe
DAY=$(date +%Y%m%d)
RUN=outputs/peak-ram-board-$DAY             # fresh directory for this session
[ -e "$RUN" ] && echo "STOP: $RUN exists" || mkdir -p "$RUN"
"$PY" -c "import serial"
"$PY" -m serial.tools.list_ports -v         # application port COM8, VID:PID=2341:805A
for ARCH in outputs/firmware-peakram-prepared/nano33ble_v2_f1a_peakram outputs/firmware-peakram-prepared/nano33ble_peakram outputs/firmware-v2-f1a-r2; do (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"; done
```

1. Close every serial monitor, PlatformIO monitor, Arduino IDE and **every Chrome Web Bluetooth dashboard tab**. A connected central changes the workload: the `no_central` check fails and the run is void.
2. Every `sha256sum -c` line must say `OK`. If one does not, stop.
3. **No finger.** Nothing touches the sensor during any capture. Enclosure and room light stay as they normally are. Note both in the sidecar (step 4).

## 1. Identify the image on the board (before any flash)

```bash
"$PY" -u build/v2/cap.py COM8 20 "$RUN/00_identify_before.txt"
"$PY" -u -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; print('PR lines', t.count(' PR: '), '| NO FINGER', t.count('NO FINGER'), '| Hz', (min(v), max(v)) if v else None, '| v2 banner', 'Model: TinyKAN v2' in t)" "$RUN/00_identify_before.txt"
```

Expect 0 `PR` lines, `Status: NO FINGER` lines and `Hz:` lines. `cap.py` opens the port without a reset, so the boot banner is normally missed. Record what you see. Do not infer an identity the capture does not show.

## Flashing an archived image (same for every step)

This is the bossac procedure from [v2-flash-runbook.md](v2-flash-runbook.md). Set `JOB` and `ARCH` from the table, then:

```bash
"$PY" -u build/v2/touch1200.py COM8          # 1200-baud touch: board enters the bootloader (or double-tap reset)
"$PY" -m serial.tools.list_ports -v          # the bootloader appears as VID:PID=2341:005A, recently COM7
BOOT=COM7                                    # set to the port just listed
LOG=$RUN/$JOB-bossac-upload.log
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/firmware.bin" 2>&1 | tee "$LOG"
```

- The log must end with `Verify successful`, and `Write N bytes to flash` must equal the BIN size: 346,468 B for the v1 probe, 347,812 B for the v2 probe, 343,620 B for f1a-r2. If it does not, stop. Do not retry with other tools.
- `scripts/record_v2_upload_receipt.py` knows only the r2 archives, so it refuses the probe images. For (1) and (2), `scripts/analyze_peak_ram.py --image … --bossac-log …` binds the capture to the verified write instead. It checks the log and the image against the archive's `SHA256SUMS`. Use the receipt script for (3).
- The application port comes back as COM8 after reset.

## (1) and (2): probe capture, 180 s, no finger

```bash
JOB=nano33ble_v2_f1a_peakram; ARCH=outputs/firmware-peakram-prepared/$JOB    # (2): JOB=nano33ble_peakram
# flash as above, then start the capture straight away:
CAP=$RUN/${JOB}_nofinger_180s.txt
"$PY" -u build/v2/cap.py COM8 180 "$CAP"
"$PY" -u scripts/analyze_peak_ram.py "$CAP" --expect-image $JOB \
    --image "$ARCH/firmware.bin" --bossac-log "$LOG" --output "$RUN/${JOB}_report"
```

**Re-identify.** The capture must contain `PR_INFO: v=1 img=$JOB …`, which repeats every 60 s, so a capture that misses boot still has one. It must also contain `PR: v=1 …` lines every 5 s. The analyser checks both. The existing live lines (`Red: … Hz: …`, `Feat:`, `Status:`) carry on unchanged.

**Validity criteria.** These were fixed on 25 Oct 2026, before any probe flash, and are coded in `scripts/analyze_peak_ram.py`. A run counts only if every check passes:

| Check | Criterion |
|---|---|
| `identity` | ≥ 1 `PR_INFO:` line, all naming `img=$JOB` |
| `painted` | `wm=1 force=1 heap_painted=1 msp_painted=1 main_painted=1 setup_painted=1`, `gap=0`, RAM 0x2000_0000 + 262,144 |
| `parse` | no malformed `PR:`/`PR_INFO:` line |
| `pr_lines` | ≥ 30 `PR:` lines (36 expected in 180 s) |
| `gaps` | no stretch > 15 s without a `PR:` line: capture start to first, between lines (host and board clocks), last to capture end |
| `no_reboot` | board `t_ms` and `rep` rise on every line |
| `decisions` | final `dec` ≥ 25 forced decisions since boot, and ≥ 20 during the capture |
| `regions` | the same heap/MSP/thread set on every line; reserved = peak + headroom for every region; `whole_hw + untouched = 262144`; `ovl = 0`; the sum of headrooms equals `untouched` |
| `monotonic` | no high-water falls between lines; `main_pre`, `main_setup`, `static` stay constant |
| `no_finger` | the live gate passed ≤ 1% of samples (`lf / (lf + lnf)`) |
| `no_central` | no `BLE connected to:` line |
| `upload` | bossac `Verify successful`, bytes written = BIN size, BIN matches the archive `SHA256SUMS` |

The analyser exits 0 and writes `paper_figure` only when every check passes. If a run fails, keep it and note why. One rerun is allowed after the cause is removed (for example a dashboard tab that connected), into a new capture file named `…_rerun1.txt` and a new report directory. Report both runs. Do not move a criterion after a run.

## (3) Restore f1a-r2

```bash
JOB=nano33ble_v2_f1a; ARCH=outputs/firmware-v2-f1a-r2
(cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP"
# flash as above (LOG=$RUN/nano33ble_v2_f1a-bossac-upload.log), then:
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --archive $ARCH --bossac-log "$LOG" \
    --bootloader-port $BOOT --output "$RUN/nano33ble_v2_f1a-restore-upload-receipt.json"
"$PY" -u build/v2/cap.py COM8 30 "$RUN/99_identify_after_restore.txt"
"$PY" -u -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; print('PR lines', t.count(' PR: '), '| NO FINGER', t.count('NO FINGER'), '| Hz', (min(v), max(v)) if v else None, '| v2 banner', 'Model: TinyKAN v2' in t)" "$RUN/99_identify_after_restore.txt"
```

The restore is done when the receipt is written (`uploaded_and_verified: true`, image `cdefac1d…`), the capture has **0** `PR` lines and only `Status: NO FINGER` status lines, and every `Hz:` line is ≥ 49.5 (the 22 Oct rate criterion). If `PR` lines are still present, the probe is still on the board: stop and report.

## 4. Sidecar

Write `$RUN/SESSION.md`. Include, for each step:
- image hash, bossac log and report directory;
- which checks passed;
- `whole_hw` and the `T.main` row;
- room light and enclosure state;
- that nothing touched the sensor;
- that no central connected.

## What goes in the paper

Only from a run whose report says `valid: true`, per image:

- `whole_hw` from the final `PR:` line (`paper_figure.whole_hw_bytes`), in bytes and as a share of 262,144 B. It is the RAM high-water, not the peak. Since 30 Oct the wording is the bracket from `scripts/peak_ram_bracket.py`: *"whole-device peak RAM is not measured; for the 25 Oct probe run (probe image, no finger, forced decisions, no central) a derived bracket puts it between X and Y; the upper end is the RAM high-water"* ([peak-ram-design.md §7](peak-ram-design.md#7-peak-versus-high-water-added-30-oct-2026); the analyser's `paper_figure.wording` string is the 25 Oct one and is superseded);
- next to it, the main-thread stack peak and headroom (`T.main`), the loop-phase peak (`main_loop`) as the decision-path stack figure, and the live image's static RAM from the linker (v1 88,008 B, v2 94,032 B);
- the probe's own cost (+944 B static RAM for v1, +952 B for v2; archive README).

Do not add the per-region peaks together. Do not call `whole_hw` an instantaneous measurement. Do not call it the live image's peak without the word "probe". Do not quote any `KAN_Prob`, `Candidate` or `Confirmed` value from a probe capture. The connected-central path (notifications, connection events) is not covered.

## Do not

- Do not edit firmware, rebuild, or flash anything other than the three archived BINs above.
- Do not use `pio run -t upload`, and do not flash from `.pio/build/`.
- Do not write into existing `outputs/` directories. This session writes only to the fresh `$RUN`.
- Do not leave a probe image on the board.
- Do not open the dashboard during the session.

## Sampler run, finger on, dashboard connected: validity checks (pre-registered 30 Oct 2026)

Written and committed on 30 Oct 2026, before the sampler firmware exists, before any build of it and before any run. Jason named the build job on 30 Oct. The board session is his later named job. Method: [peak-ram-design.md §8](peak-ram-design.md) (written with the firmware). The images will be `nano33ble_peakram_sampler` (v1) and `nano33ble_v2_f1a_peakram_sampler` (v2 f1a). They are the 25 Oct probe images with two more flags:
- `-DPEAK_RAM_LIVE_GATE`: the finger gate and the BLE payload behave as shipped (no forced decisions; `PR_INFO: … force=0`);
- `-DPEAK_RAM_SAMPLER`: a TIMER4 interrupt at 4 kHz sums, at each tick, the static/fixed constant, the heap in use, every thread's stack from its SP to its top and the MSP. It reports `samp_max`, the largest instantaneous total seen (a measured lower bound on the peak), and `s_win`, the largest total since the previous `PR:` line.

**Fields these checks read.** `PR_INFO:` adds `smp smp_timer smp_hz smp_prio prio_svc prio_pendsv prio_systick vtor hlock s_const s_nthr`. `PR:` adds `ble sub ble_conn ble_disc hl hl_st s_n s_ok s_skk s_skl s_skh s_skt s_set samp_max s_win sm_heap sm_msp sm_stk smx_heap smx_msp s_cyc_max s_cyc_sum` and one `S.<thread>@<id>=<depth at samp_max>/<largest sampled depth>` per thread. `s_n` counts timer ticks serviced, `s_ok` valid samples, and `s_skk`/`s_skl`/`s_skh`/`s_skt` the skips (kernel exception active, heap lock held, free list implausible, thread table mismatch). The old fields and the old line format are unchanged.

"During the capture" means from the first to the last `PR:` line in the capture file. Δx is the last line's x minus the first line's x. A run counts only if every check passes. The checks will be coded in `scripts/analyze_peak_ram.py --sampler` with these exact constants:

| Check | Criterion |
|---|---|
| `identity` | ≥ 1 `PR_INFO:` line, all naming the expected sampler image |
| `config` | `wm=1 force=0 smp=1`, all four paint flags 1, `gap=0`, RAM 0x2000_0000 + 262,144; `smp_prio ≤ prio_pendsv` and `smp_prio ≤ prio_systick` (no asynchronous kernel exception can preempt the sampler); `hlock` ≠ 0; `vtor=0x20000000` |
| `parse` | no malformed `PR:`/`PR_INFO:` line |
| `pr_lines` | ≥ 30 `PR:` lines |
| `gaps` | no stretch > 15 s without a `PR:` line (as for the 25 Oct runs) |
| `no_reboot` | board `t_ms` and `rep` rise on every line |
| `regions` | as for the 25 Oct runs (arithmetic re-derives, one thread set), plus `s_set=1` on every line (the sampler's thread table matches the enumeration) |
| `monotonic` | as for the 25 Oct runs, plus `samp_max`, `s_n`, `s_ok`, `ble_conn`, `ble_disc` never fall |
| `finger` | finger present on ≥ 95% of live-gate samples during the capture: Δ`lf` / (Δ`lf` + Δ`lnf`) ≥ 0.95 |
| `central` | a Bluetooth central connected before capture start and stayed: `ble=1` and `sub=1` on every `PR:` line; `ble_conn` and `ble_disc` constant during the capture; no `BLE connected to:` or `BLE disconnected` line in the capture |
| `rate` | every `Hz:` value in the capture ≥ 49.5 |
| `decisions` | Δ`dec` ≥ 25 during the capture |
| `samples` | Δ`s_n` ≥ 0.90 × 4,000 × Δ`t_ms` / 1,000 |
| `skips` | (Δ`s_skk` + Δ`s_skl` + Δ`s_skh` + Δ`s_skt`) / Δ`s_n` ≤ 0.05 |
| `bound_order` | `s_win ≤ samp_max ≤ whole_hw` on every line |
| `heap_source` | on every line with `hl_st=1`, `hl` = `hs_cur + hs_ovh`; `hl_st=1` on ≥ half the lines |
| `upload` | flash log `Verify successful`, bytes written = BIN size, BIN matches the archive `SHA256SUMS` (as for the 25 Oct runs) |

**Why these thresholds.**
- `samples` 0.90: a tick is lost only when the sampler stays pending for more than one 250 µs period. That happens when handler-mode code or an interrupt-masked section runs longer than that. The sampler sits at the kernel exceptions' level, so every device interrupt preempts it. Ordinary operation should service well above 99% of ticks. 0.90 still leaves ≥ 3,600 samples per second, and a lower share would mean long masked or handler-mode stretches the design did not expect.
- `skips` 0.05: kernel skips are impossible by construction (the sampler cannot preempt PendSV or SysTick, and SVC is synchronous), so they are expected to be 0. Heap-lock skips happen only while a thread holds the newlib malloc mutex. The 25 Oct runs show at most one allocation after `setup()` in 180 s, so the expected share is far below 0.1%. Above 5%, heap-active instants would be under-sampled enough to bias the lower end.
- `finger` 0.95 and `decisions` 25 were set when Jason named the job (30 Oct): the run is finger-on, and 25 of the ~35 possible strides in 180 s must produce decisions.
- `heap_source`: the sampler reads the heap as break − `__end__` − free-list bytes. That equals the allocated chunk bytes only when the arena has no alignment gaps. Lines read with no allocation or free in between (`hl_st=1`) must show it.

**Figures, only from a run that passes every check.**
- Lower end: the largest `s_win` over the capture's `PR:` lines after the first. Each of those windows lies inside the capture, so this is the largest measured instantaneous total during the finger-on, connected capture.
- Upper end: `whole_hw` on the final line. It covers the whole run from the hook, so it is conservative for the capture window.
- Also reported: `samp_max` (the whole run since the end of `setup()`), its breakdown, the component maxima and the sampler's cycle cost.

Wording: *"peak RAM between [lower] and [whole_hw] on the probe image, finger on, dashboard connected"*. Still not covered: the unmodified live image (the probe adds static RAM, its own stack frames and the sampler's interrupt frame), other workloads, and instants the sampler cannot see (inside interrupt handlers, and between ticks).

**One rerun** is allowed per image, after the cause is removed, into new file names (`…_rerun1`). Report both runs. Do not move a criterion after a run.

## Sampler run, finger on, dashboard connected: steps (Jason's later named job)

**Status, 2 Nov 2026: run, both images valid** ([SESSION.md](../outputs/peak-ram-sampler-board-20261102/SESSION.md)). r2 images flashed and verified; both started the sampler (`smp=1`, `lk_step=0`, `lk_stale=1`, `hlock` as in the archive README). v2 f1a: valid on the first capture, peak RAM between 86,576 B and 87,348 B on the probe image, finger on, dashboard connected. v1: the first capture failed `central` (`ble=0` on every line; the flash had dropped the v2 link and the v1 Connect did not take), kept on file; the one allowed rerun is valid, between 80,552 B and 81,392 B. Sensor module 2, lid on. f1a-r2 restored (receipt `uploaded_and_verified: true`, 0 `PR`, 50.08–50.40 Hz). No criterion moved. Before a judged capture, read `ble=1 sub=1` from a short separate capture (as the rerun did); a flash resets the board and drops any central, and the Ready answer alone did not show the connection.

**Status, 1 Nov 2026 (r2).** Repaired images built from `8eea953` and archived under `outputs/firmware-peakram-sampler-prepared-r2/` (cause and repair in its README). **Not flashed, not run.** The cause of the stop: RTX in this core leaves `owner_thread` set after the last release. The lookup's `owner_thread == nullptr` test therefore could never pass, and the tick's owner test would have skipped every tick. The repaired lookup and tick test use the lock count, which is RTX's own test ([peak-ram-design.md §8, 1 Nov note](peak-ram-design.md)). `PR_INFO:` also prints `st_*`/`lk_*` start diagnostics, which no check reads. The checks below, their constants and the analyser's checks are unchanged. The step table and archive paths below now point at r2. The 31 Oct images stay in `outputs/firmware-peakram-sampler-prepared/` as the failed images; do not flash them again. The r2 images add 1,440 B of static RAM over the live images, not the 1,376 B quoted under "Sidecar and figures". The 1 Nov session took no capture of the kind the checks judge; its `SESSION.md` records the one rerun per image as unused. Not dry-run, as on 31 Oct (below). Host tests bind both archives to the 1 Nov flash logs through the analyser's binding function.

**Status, 1 Nov 2026.** Session run and stopped at identify (`outputs/peak-ram-sampler-board-20261101/SESSION.md`). Both archived sampler images flashed and verified, but on both `PR_INFO:` reads `smp=0 hlock=0x00000000` and every `PR:` line `s_n=0`: `prs_find_malloc_mutex()` returned null, so `prs_start()` never started TIMER4. No finger capture, no dashboard connect, no analyser report. f1a-r2 restored (receipt `uploaded_and_verified: true`, 0 `PR`, 50.1–50.3 Hz). The one rerun per image is unused and needs repaired images in a new archive. The criteria below do not change.

**Status, 31 Oct 2026.** Images built from `d7c9d54` and archived under `outputs/firmware-peakram-sampler-prepared/`. **Not flashed, not run.** The checks above were committed in `f782745`, before the firmware. Flash the archived files only. Do not rebuild on the day, and do not use `pio run -t upload`. Run (s1), (s2), (s3) in that order: (s1) comes first because v2 f1a-r2 is the image on the board now. If the session has to stop early, skip (s2) but still run (s3). The session takes about 30 minutes, with about 4 minutes of finger time per image.

| Step | Archive (`$ARCH`) | Env (`$JOB`) | `firmware.bin` SHA-256 | Bytes |
|---|---|---|---|---:|
| (s1) v2 sampler r2 | `outputs/firmware-peakram-sampler-prepared-r2/nano33ble_v2_f1a_peakram_sampler` | `nano33ble_v2_f1a_peakram_sampler` | `bd3a494d23108d1a065473da2ca6e7e4675e646400c2124b53bba66321abd465` | 350,660 |
| (s2) v1 sampler r2 | `outputs/firmware-peakram-sampler-prepared-r2/nano33ble_peakram_sampler` | `nano33ble_peakram_sampler` | `908940113497dea93344c80c3ceba4efddf027e6f929e66e0a916ba63c1c2920` | 349,308 |
| (s3) Restore | `outputs/firmware-v2-f1a-r2` | `nano33ble_v2_f1a` | `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2` | 343,620 |

### 0. Pre-flight

Same as section 0 above, with `RUN=outputs/peak-ram-sampler-board-$DAY` and these archives:

```bash
for ARCH in outputs/firmware-peakram-sampler-prepared-r2/nano33ble_v2_f1a_peakram_sampler outputs/firmware-peakram-sampler-prepared-r2/nano33ble_peakram_sampler outputs/firmware-v2-f1a-r2; do (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"; done
# cue: the pop-up of the local build/v2/walk_cues.sh (gitignored), without its walk timing
cue() { powershell.exe -NoProfile -Command "[System.Media.SystemSounds]::Exclamation.Play(); Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('$1','FYRP peak RAM','OK','Information') | Out-Null" >/dev/null 2>&1 & }
```

Close every serial monitor and Arduino IDE. **Close every dashboard tab for now.** It is opened at step 3, after the flash. Write the sensor module, enclosure state and room light into the sidecar.

### 1. Identify the installed image (before any flash)

Exactly section 1 above (`$RUN/00_identify_before.txt`, 20 s). Expect 0 `PR` lines, `Status: NO FINGER`, `Hz:` lines.

### (s1)/(s2) For each sampler image

```bash
JOB=nano33ble_v2_f1a_peakram_sampler; ARCH=outputs/firmware-peakram-sampler-prepared-r2/$JOB    # (s2): JOB=nano33ble_peakram_sampler
# 2. flash as in "Flashing an archived image" above (LOG=$RUN/$JOB-bossac-upload.log);
#    the log must say Verify successful and write the byte count in the table.
# 3. re-identify, no finger yet:
"$PY" -u build/v2/cap.py COM8 70 "$RUN/${JOB}_identify.txt"
grep -c "PR_INFO: v=1 img=$JOB .*force=0.*smp=1" "$RUN/${JOB}_identify.txt"   # >= 1 (PR_INFO repeats every 60 s)
grep -o "smp=[0-9]*\|hlock=0x[0-9A-F]*\|st_[a-z]*=[0-9]*\|lk_[a-z]*=[0-9A-Fx]*" "$RUN/${JOB}_identify.txt" | sort | uniq -c
#    start diagnostics (added 1 Nov): expect smp=1, st_hlock=1 st_set=1 st_vtor=1, lk_step=0, and hlock equal to
#    the value in $ARCH/README.md. If smp is not 1: STOP here, before the finger cue. No finger, no capture;
#    go to (s3) and record the st_*/lk_* values in the sidecar (they name the failed precondition or lookup step).
# 4. finger on, then connect:
cue "Put your finger on the sensor now and keep it still. Then connect the dashboard."
#    open dashboard/index.html in Chrome, Connect to OxygenRiskDemo, wait until values update
#    and at least one decision has arrived (about 20 s after the finger went on).
# 5. capture, finger on and dashboard connected throughout:
CAP=$RUN/${JOB}_finger_dash_180s.txt
"$PY" -u build/v2/cap.py COM8 180 "$CAP"
cue "Capture done. Lift your finger now. Disconnect the dashboard."
# 6. analyse with the pre-registered checks:
"$PY" -u scripts/analyze_peak_ram.py "$CAP" --sampler --expect-image $JOB \
    --image "$ARCH/firmware.bin" --bossac-log "$LOG" --output "$RUN/${JOB}_report"
```

- The capture must start **after** the dashboard is connected (check `central`: no `BLE connected to:` line may fall inside the capture), and the finger stays on until the capture ends.
- If any check fails, keep the run and note why. One rerun is allowed per image after the cause is removed, into `…_rerun1.txt` and a new report directory. Do not move a criterion.

### (s3) Restore f1a-r2 and re-identify

Exactly step (3) of the 25 Oct procedure above: flash `outputs/firmware-v2-f1a-r2`, write the receipt with `scripts/record_v2_upload_receipt.py` into `$RUN/nano33ble_v2_f1a-restore-upload-receipt.json`, capture 30 s into `$RUN/99_identify_after_restore.txt`. Done when the receipt says `uploaded_and_verified: true` for `cdefac1d…`, the capture has **0** `PR` lines, and every `Hz:` ≥ 49.5. Restore even if (s1) or (s2) failed. Never leave a sampler image on the board.

### Sidecar and figures

Write `$RUN/SESSION.md`: image hashes, flash logs, report directories, which checks passed, sensor module, enclosure and room light, and when the dashboard connected and disconnected. From a report with `valid: true` only:
- `paper_figure.wording`: *"peak RAM between [lower] and [whole_hw] on the probe image, finger on, dashboard connected"*, with lower = the largest `s_win` inside the capture, and upper = the final `whole_hw`;
- next to it, `samp_max`, the breakdown at `samp_max`, the sampler's cycle cost, and the static RAM of the live image from the linker (v1 88,008 B, v2 94,032 B);
- still not covered: the unmodified live image (the sampler image adds 1,376 B of static RAM, its own frames and an interrupt frame), other workloads, and instants the sampler cannot see (between ticks, inside interrupt handlers).

**Not dry-run when this was written (31 Oct).** No serial port was opened. The flasher, `pio run -t upload`, `build/v2/touch1200.py`, `build/v2/cap.py`, `build/v2/ident.py` and the analyser's `--bossac-log` binding were not run: a permission rule denies any command containing the flasher's name. `--sampler` was run only on synthetic captures (`tests/test_peak_ram_sampler.py`).

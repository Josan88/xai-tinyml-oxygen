# Whole-device peak-RAM method (design)

Written 25 Oct 2026, before any probe code. Board job named by Jason the same day: "Peak RAM: build the image, flash it, measure with no finger, then restore the current live image." This document is the method. Commands are in [peak-ram-runbook.md](peak-ram-runbook.md); the built images are under `outputs/firmware-peakram-prepared/`.

**Why a new method.** The existing memory evidence is not whole-device peak RAM ([technical-remaining-review.md §5](technical-remaining-review.md#5-runtime-memory), [on-device-timing.md](on-device-timing.md)). The `M` command of the timing images reports one RTX main-thread watermark (31,308 / 32,768 B, already at that level before replay) and a newlib `mallinfo` snapshot (1,040 B). Those come from the timing images, not the live images. Interrupt (MSP) and other-thread stacks were never measured, and heap and stack were never combined defensibly. This method measures every RAM region of the **live** v1 and v2 images, on probe builds of those images, during a no-finger run that still runs the full decision path.

**Scope.** Target is the Nano 33 BLE Sense Rev2: nRF52840, 256 KB SRAM (262,144 B at 0x2000_0000), Arduino mbed core 4.5.0 (Mbed OS with RTX5), toolchain GCC 7.2.1. The live images are `nano33ble` (v1, `1f2f95db…`) and `nano33ble_v2_f1a` (v2 f1a-r2, `cdefac1d…`). The probe images are `nano33ble_peakram` and `nano33ble_v2_f1a_peakram`. They are those envs plus `-DPEAK_RAM_PROBE`, which adds the paint/scan, the `PR:` report and the forced-decision mode. Existing envs compile byte-identical; the build proof is in the archive README.

## 1. Regions

Read from the core's linker script (`variants/ARDUINO_NANO33BLE/linker_script.ld`), `arm-none-eabi-size -A` and `nm -S` on the live ELFs (`ba2673dd…` v1, `ba76b9a6…` v2), and the core and library sources. Sizes in bytes.

| Region | Address (v1 / v2) | v1 reserved | v2 reserved | What it is |
|---|---|---:|---:|---|
| `.nvictable` | 0x2000_0000 | 256 | 256 | RAM vector table (`nrf_dispatch_vector`) |
| `.crash_data_ram` | 0x2000_0100 | 256 | 256 | Mbed crash-report RAM |
| `.data` | 0x2000_0200 | 1,944 | 1,944 | initialised statics |
| `.bss` | 0x2000_0998 | 86,064 | 92,088 | zeroed statics, including three thread stacks (below) |
| `.heap` | 0x2001_59C8 / 0x2001_7150 | 172,600 | 166,576 | `__end__` up to `__HeapLimit` = 0x2003_FC00 |
| `.stack` (MSP) | 0x2003_FC00 | 1,024 | 1,024 | boot stack, then the interrupt/exception stack |
| **Total** | | **262,144** | **262,144** | the sections tile RAM exactly |

`.noinit` is empty in both images. Static RAM (`.data` + `.bss`) is **88,008 B** (v1) and **94,032 B** (v2), the PlatformIO "RAM used" figure. The v2 increase is the larger `pipeline` object (15,336 → 21,352 B: red ring buffer, red filtered window) plus alignment.

**Large objects in `.bss`** (both images unless stated):

| Object | Bytes | Note |
|---|---:|---|
| `_main_stack` | 32,768 | RTX main thread = Arduino `setup()`/`loop()` thread (`MBED_CONF_RTOS_MAIN_THREAD_STACK_SIZE`) |
| `pipeline` (OxygenRiskPipeline) | 15,336 / 21,352 | ring buffers, filter scratch, Hann window; no dynamic allocation |
| `ble::NRFCordioHCIDriver::…::buffer` | 4,904 | Cordio WSF buffer pool (BLE) |
| Cordio link-layer tables (seven largest: `lctrMstExtInitTbl`, `lctrMstPerScanTbl`, `lctrMstExtScanOneMCtx`, `lctrMstExtScanCodedCtx`, `lctrMstScan`, `lctrMstInit`, `lctrSlvAdv`) | 8,264 | BLE controller state |
| `hrRing`, `hrWork` | 1,600 each | display heart-rate ring and work buffer |
| `PluggableUSBD()::obj`, `_SerialUSB`, `Wire`, `Wire1` | 1,424 / 696 / 572 / 572 | USB CDC and I2C drivers |
| `os_timer_thread_stack` | 768 | RTX timer thread stack |
| `os_idle_thread_stack` | 512 | RTX idle thread stack |
| `loop()::frames` | 256 | FIFO drain buffer |

**Heap.** `_sbrk` (Mbed, weak) moves a break from `__end__` and refuses any request past 0x2003_FC00 (`__HeapLimit`, disassembled). The allocator is newlib-nano (`__malloc_free_list`, `__malloc_sbrk_start`). `_malloc_r` and friends are wrapped by Mbed heap statistics (`MBED_ALL_STATS_ENABLED 1`, `--wrap,_malloc_r`). Nothing else writes above the break.

**Thread stacks.** RTX watermarking is compiled in: `osRtxConfig.flags` = 0x5 (privileged mode | stack watermark; no stack check). Every stack is filled with 0xCCCCCCCC at thread creation, and `osThreadGetStackSpace` returns the never-touched bytes from the bottom.

| Thread | Stack | Where | Created |
|---|---:|---|---|
| main (`setup`/`loop`, decisions) | 32,768 | `.bss` `_main_stack` | kernel start |
| RTX idle | 512 | `.bss` | kernel start |
| RTX timer | 768 | `.bss` | kernel start, if RTX creates it |
| `USBevt` (core `USBSerial::begin`) | 256 | heap (`new rtos::Thread(…, 256, …)`) | in core `main()`, before `setup()` |
| ArduinoBLE `bleLoop` | 4,096 | heap (`new rtos::Thread()`, `MBED_CONF_RTOS_THREAD_STACK_SIZE`) | `BLE.begin()` in `setup()` |

The probe does not rely on this list. It enumerates threads at runtime (`osThreadEnumerate`) and prints each one with its stack base and whether the base is in `.bss` or the heap.

**MSP.** Before `osKernelStart` the boot code runs on the 1,024 B MSP region. After it, threads run on their own stacks (PSP) and only exceptions use MSP: SVC, PendSV and SysTick for RTX, plus USB, RADIO/RTC/TIMER (Cordio), TWIM (I2C) and GPIOTE handlers. The hardware frame of an exception taken from a thread goes on that thread's stack.

## 2. Paint and scan

**Hook.** A C++ constructor with `init_priority(101)` in `main.cpp` (probe builds only). The disassembly shows the order `software_init_hook` → `mbed_rtos_start` → `osKernelStart` → main thread `mbed_start` → `mbed_toolchain_init` → `__libc_init_array` (all constructors) → `mbed_main` → core `main()` → `init()`, USB, `setup()`. The hook therefore runs in thread mode on the main thread's PSP, with the kernel running. It runs before every unprioritised constructor (the linker script sorts `.init_array.*` first), before USB start, before `BLE.begin`, and before any thread other than main, idle and timer exists. `mbed_sdk_init` is defined by the nRF52 target and cannot be overridden. `mbed_main` runs later than the constructor. Earlier than this, the code runs on MSP before the kernel starts, and no hook exists there.

At the hook, the probe paints only memory that is provably unused:

1. **Heap above the break.** Under `osKernelLock()` it reads `_sbrk(0)` (the current break, no side effect) and fills [break, `__HeapLimit`) with pattern 0x9E3779B9. Only `_sbrk` hands out that memory, and while the kernel is locked no thread can call it. ISRs do not allocate: Mbed's malloc lock is an RTOS mutex. No allocator assumes that fresh memory is zero (`calloc` clears its own block).
2. **MSP below the current MSP.** It checks thread mode (IPSR = 0), PSP in use (CONTROL.SPSEL = 1) and privileged mode (CONTROL.nPRIV = 0, needed to read MSP). It then fills [`__StackLimit`, MSP) with the same pattern, with interrupts masked for the ~1 KB fill. In thread mode no exception is active, so everything below MSP is a dead frame. The boot frames above MSP are left alone and count as used.
3. **Main-thread stack below its own SP.** It first records the RTX lifetime watermark (`main_pre`). A leaf function then reads PSP and refills [stack base + 4, PSP − 256) with the RTX fill 0xCCCCCCCC, which restarts RTX's watermark at the hook. Nothing live is below SP: ARM has no red zone, and an exception frame pushed there during the fill is popped before the fill resumes. The 256 B margin covers the fill function's own frame. The magic word at the stack base is not touched.

All other threads use RTX's own watermark. The probe does not repaint them.

**Threads created after painting.** `USBevt` and `bleLoop` get their stacks from `new`, so the memory comes out of the painted heap. RTX then fills them with 0xCC at creation, so their watermark covers their whole life. The heap scan sees those bytes as touched heap (0xCC is not the heap pattern), which is right: the heap has handed them out.

**Second mark at the end of `setup()`.** The probe records the main-thread watermark and the MSP low-water there, then repaints both in the same way, so the main thread and MSP each get a boot+setup peak (`main_setup`) and a loop-only peak (`main_loop`, `msp_loop`). The lifetime figure for each region is the maximum over the phases. That figure is what goes into the whole-device sum. The split exists to explain the 31,308 B pre-replay watermark seen on the timing images, and to give the decision-path stack peak on its own.

**Scan.** Every 5 s, from `loop()` on the main thread:
- heap high-water = the highest word in [paint break, `__HeapLimit`) that is not the pattern (scanning down from the limit), raised to the current break if that is higher. Allocated-but-unwritten heap counts as used;
- MSP low-water = the lowest word in [`__StackLimit`, paint MSP) that is not the pattern (scanning up);
- each enumerated thread: `osThreadGetStackSpace` (main: the minimum over its phases);
- Mbed heap statistics (`current_size`, `max_size`, `overhead_size`, `alloc_fail_cnt`) as a cross-check.

High-water marks persist in memory, so the scan time does not matter. A write that races a scan is caught by the next scan. The final `PR:` line, taken at least 5 s after the last decision, holds the run's peaks.

## 3. Whole-device peak

Per region the probe reports **reserved / peak / headroom**:
- **heap**: reserved = `__HeapLimit` − `__end__`; peak = heap high-water − `__end__`. This includes the heap-allocated thread stacks in full and allocator headers.
- **MSP**: reserved 1,024; peak = `__StackTop` − lowest touched word.
- **each thread**: reserved = stack size; peak = size − never-touched bytes. This is RTX's definition, the same as the `M` command and Mbed stack stats. Threads are tagged `bss` or `heap`.
- **static**: `.data` + `.bss` from the linker symbols. **Fixed**: the 512 B NVIC + crash-data RAM.

**Whole-device high-water** = 262,144 − the bytes never touched in any painted or watermarked region. The never-touched set is the union of these address intervals:

- heap: [heap high-water, `__HeapLimit`);
- MSP: [`__StackLimit`, lowest touched word);
- each thread: [stack base, stack base + never-touched bytes), lifetime minimum for main.

Everything else counts as used: all of `.data`, all of `.bss` outside the never-touched bottoms of the three static stacks, the heap below its high-water, the MSP above its low-water, and the NVIC and crash-data RAM.

**No double counting.** Every byte of RAM is in exactly one section, and the never-touched intervals are address ranges, so they cannot be counted twice. The firmware sorts and merges them before summing. It also prints `ovl`, the bytes listed in more than one interval, which must be 0. A heap-allocated thread stack lies wholly below the heap high-water, so its never-touched bottom is disjoint from the heap's never-touched top. A `.bss` thread stack is inside `.bss`, which is otherwise counted as used. The per-region peaks are **not** added together: the thread rows overlap the heap row (heap stacks) and the static row (`.bss` stacks). The firmware prints the arithmetic, and the analyser re-derives `whole_hw` from the parts.

**Meaning.** Stack and heap peaks need not happen at the same moment, so `whole_hw` is an **upper bound on simultaneous RAM use**. It is not a measured instantaneous total. It is also conservative because memory touched before the hook counts as used, as do static buffers that are never written and allocated heap that is never written.

## 4. No-finger workload

The live loop does not run decisions without a finger. Under `PEAK_RAM_PROBE` only:
- `isFingerPresent()` counts the live gate's verdict (`lf`/`lnf`: samples where red and IR are both > 3000, or not), then returns true. `FINGER_PRESENT_THRESHOLD` is 0 in the probe, so every `irVal < FINGER_PRESENT_THRESHOLD` gate in `loop()` and `updateBLE()` stays open. The dark-floor samples (IR ~150) flow into the unchanged `pipeline.push_sample` path: PPG Sliding Window, filter, features, TinyKAN, Mean-Perturbation Attribution, Temporal Debounce, then the `Feat:`/status lines and LED logic.
- `updateBLE()` runs on every print with or without a central. The ASCII Telemetry Payload (`snprintf` with `%.2f`/`%.4f`) is built and written into the characteristic. A notification goes out only to a subscribed central.

The pipeline headers, weights, threshold 0.50, N = 2 and the window/stride are unchanged. The output line formats are unchanged. The values on those lines are dark-floor nonsense: a probe-image `KAN_Prob`, `Candidate` or `Confirmed` is not a result and must not be quoted.

**Data dependence.**
- The pipeline does not allocate (no `new`/`malloc` in `oxygen_risk_pipeline.h`, `feature_extraction.h` or `kan_inference.h`). Its buffers are members of the static `pipeline`, and its stack locals have fixed size: the Welch PSD, segment and FFT arrays take ~3.6 KB, and the scaled/perturbed input arrays are small. Stack depth along the decision path does not depend on the data, apart from small branch differences.
- `Serial.print(float)` and `printFloatSafe` do not allocate.
- **Data-dependent.** newlib `snprintf` with `%f` goes through `_dtoa_r`, which takes `Bigint` blocks from the heap (`_Balloc` → `_calloc_r`) and keeps them on a per-reent free list. Their size depends on the magnitude of the printed value. Dark-floor probabilities and attributions are not finger values, so this path is exercised but not at finger magnitudes.
- **Not exercised without a connection.** The connect path (`central.address()` String on the heap), ATT notification, HCI ACL buffers and connection-event radio interrupts. BLE keeps advertising throughout. A connected-central run would be a separate job. Every capture records that no central was connected (no `BLE connected` line).

## 5. Serial report

At the end of `setup()`, and again every 60 s so that a capture that misses the boot banner still holds one, the probe prints:

```text
PR_INFO: v=1 img=<env> model=v1|v2 force=1 hook=ctor101 ram=0x20000000+262144 fixed=0x20000000+512 data=<addr>+<n> bss=<addr>+<n> heap=<addr>+<n> msp=<addr>+<n> gap=0 pattern=0x9E3779B9 rtx_fill=0xCCCCCCCC wm=1 period_ms=5000 heap_painted=1 msp_painted=1 main_painted=1 setup_painted=1 heap_paint=<addr>+<n> msp_paint=<addr>+<n> main_paint=<addr>+<n> brk_ctor=<addr> brk_setup=<addr> nthr=<n> T.<name>@<id>=<base>+<size>/<bss|heap|other> …
```

Every 5 s:

```text
PR: v=1 t_ms=<uptime> rep=<n> dec=<decisions> lf=<n> lnf=<n> heap=<R>/<P>/<H> msp=<R>/<P>/<H> msp_loop=<P> main_pre=<P> main_setup=<P> main_loop=<P> nthr=<n> T.<name>@<id>=<R>/<P>/<H>/<loc> … brk=<addr> heap_hw=<addr> hs_cur=<n> hs_max=<n> hs_ovh=<n> hs_fail=<n> static=<n> fixed=<n> untouched=<n> ovl=<n> whole_hw=<n> ram=262144 scan_us=<n> rpt_alloc=<n>
```

`R/P/H` = reserved/peak/headroom in bytes. `T.<name>@<id>` names each thread by RTX name and control-block address; unnamed threads print as `anon`. The report is printed with `Serial.print` of integers and strings only, so it does not use `snprintf`, floats or the heap, and its buffers are static. `rpt_alloc` is the change in Mbed's cumulative allocated bytes across the report. It should be 0. Anything else is still inside the measured heap high-water. The existing live lines (`Red: … Hz: …`, `Feat:`, `Status:`) are unchanged.

The probe's own RAM cost is its `.bss` and `.data` additions and its stack frames on the main thread, which are far shallower than a decision. The archive README gives the exact probe-vs-live section deltas from the ELFs.

## 6. Limits

- **Pre-hook use counts as used**, or is lost. Heap allocated before the hook is below the paint break and counts as used. Boot frames above the hook-time MSP count as used. Exceptions taken before the hook may have gone deeper into the MSP than the hook-time MSP, and that depth is not seen. Only boot and a few timer ticks happen before the hook. Main and the static threads keep RTX's lifetime watermark from creation, so nothing is lost for them.
- **Pattern collisions.** A touched word that happens to equal the pattern at the edge of a region would be read as untouched, which slightly underestimates use. The heap pattern 0x9E3779B9 is not a small integer, a common float, an address in RAM or flash, or a byte-repeat. The RTX pattern 0xCCCCCCCC carries the same risk in every RTX tool.
- **Unenumerated threads.** Suspended threads would be missed. Missing a `.bss` stack makes the figure more conservative, because it then counts as fully used. A missing heap stack is inside the heap high-water already.
- **Hardware.** EasyDMA buffers live inside allocated objects, so they are counted. RAM retained by peripherals outside any object is not visible to this method, and none is known.
- **One run, one workload.** No finger, forced decisions on the dark floor, BLE advertising and not connected, USB serial open. It is the peak of this run, not a proven worst case. Nested-interrupt worst cases and the connected-central path may go deeper.
- **Probe image ≠ live image.** The probe adds code and a little static RAM, and the forced mode runs `updateBLE` without a connection. The number is the whole-device RAM high-water on the probe image. The live images' static RAM (88,008 / 94,032 B) comes from the linker and is a separate, exact figure.

## What goes into the paper

Per image, after a capture that passes the runbook's validity criteria: `whole_hw` from the final `PR:` line, as a byte count and as a share of 262,144 B, with this wording: *"whole-device RAM high-water on the probe image, no finger, forced decisions; conservative upper bound"*. Add the main-thread stack peak and headroom (`T.main`, and `main_loop` for the decision loop) and the live image's static RAM from the linker. Do not add per-region peaks together, and do not present `whole_hw` as an instantaneous measurement.

## 7. Peak versus high-water (added 30 Oct 2026)

**Correction.** Whole-device peak RAM is not measured. The 25 Oct run measured `whole_hw`, the RAM high-water: RAM minus the address ranges never touched at any time during the run. That is a union over time, an upper bound on simultaneous use, not the peak. Commit `5f0e008`'s message ("whole-device peak RAM measured") overstates this. The history stays. The docs say instead: *"whole-device peak RAM is not measured; for the 25 Oct probe run (probe image, no finger, forced decisions, no central) a derived bracket puts it between X and Y; the upper end is the RAM high-water"*. Numbers: `outputs/peak-ram-bracket-v1/` (`scripts/peak_ram_bracket.py`, tests `tests/test_peak_ram_bracket.py`).

**In use at an instant.** For an instant `t` after the hook, with the same regions as section 3:

```text
in_use(t) = fixed                                   NVIC + crash RAM (512 B)
          + static − Σ_bss  (SP_i(t) − base_i)        .data + .bss; a .bss stack counts from its SP up to its top
          + heap_live(t) − Σ_heap (SP_i(t) − base_i)  allocated chunks; a heap stack counts from its SP up to its top
          + (__StackTop − MSP(t))                    MSP from its current value up to its top
```

`heap_live` is the sum of allocated newlib-nano chunks: request, the 8 B Mbed stats header, the 4 B chunk header, alignment and rounding. Every allocation goes through Mbed's wrapper (`malloc_wrapper`/`free_wrapper` in the ELF are the only callers of the real `_malloc_r`/`_free_r`). The wrapper adds `chunk size − request` to `overhead_size`, so `heap_live = hs_cur + hs_ovh`. `SP_i` is the stack pointer a thread would resume from: the saved `thread->sp` for a thread that is not running (RTX pushes r4–r11 below the 32 B hardware frame), and PSP for the running one. The peak is the maximum of `in_use(t)` over the run.

`whole_hw` is the same sum with `SP_i` replaced by `base_i + space_i` (RTX never-touched bytes; lifetime minimum for main), `heap_live` by `heap_hw − __end__`, and `MSP(t)` by the lowest touched MSP word. With `P_i = size_i − space_i` and `depth_i(t) = top_i − SP_i(t)`, for every `t`:

```text
whole_hw − in_use(t) = (heap_peak − heap_live(t)) + Σ_i (P_i − depth_i(t)) + (msp_peak − msp_depth(t))
```

**Touched depth versus SP depth.** `P_i` is the depth of the deepest word ever written (other than with the fill pattern). `depth_i(t)` is where the stack pointer is. They differ in two ways:
- *A frame allocated but its bottom not yet written* (`sub sp, #N` with the lowest locals or outgoing-argument slots unwritten). SP is then deeper than any written word: `depth_i(t) > P_i` is possible. This can push `in_use(t)` above `whole_hw` by the unwritten bottom of the deepest live frame. It cannot hurt the lower bound, because a word is only written at or above SP (no red zone on ARM), so `P_i` is always reached by SP at the instant of the write.
- *Exception frames.* Hardware stacking writes all eight words of the basic frame, and RTX's context save writes r4–r11 below it, so the lowest word of an exception or context frame is always written. With lazy floating-point stacking the extended frame reserves S0–S15/FPSCR above the basic frame and may leave that middle part unwritten; that space lies above a written word and does not change the deepest written word. (The application is soft-float. The RTX port in the core is the FP-aware variant and still writes r4–r11 at the bottom.)

**Upper bound.** For every `t` from the hook to the last `PR:` line, `in_use(t) ≤ whole_hw`, because each bracket term above is ≥ 0:
- heap: `heap_live(t) ≤ brk(t) − __end__ ≤ brk_max − __end__ ≤ heap_peak`. Allocated chunks lie below the break, and the probe raises `heap_hw` to the highest break seen. This term needs no assumption; a heap pattern collision cannot break it.
- stacks: `depth_i(t) ≤ P_i` holds if (U1) no stack pointer ran below its stack's deepest written word, and (U2) the word at a watermark edge never equals the RTX fill 0xCCCCCCCC.
- MSP: `msp_depth(t) ≤ msp_peak` holds under the same two conditions for the MSP (pattern 0x9E3779B9), and only from the hook on. MSP depth before the hook (boot code, exceptions before the kernel started) was painted over and is not seen (section 6), so instants before the hook are outside the claim.

(U1) and (U2) are the limits every stack-watermark tool has. The 25 Oct data cannot test them.

**Lower bound.** At some instant `t*` the main thread wrote its deepest touched word, so `depth_main(t*) ≥ P_main`. Every other stack and the MSP are ≥ 0 at every instant. The heap term at `t*` is not observed: the reports every 5 s only bound it. On all 72 `PR:` lines of both captures, `hs_cur + hs_ovh = brk − __end__` and `heap_hw = brk`, so the arena below the break was fully allocated at every report. The break never falls: in both probe ELFs `_sbrk` is called only by `_malloc_r` (through `_sbrk_r`, to grow the arena) and by the probe's own zero-increment reads; `_free_r` does not call it. Let report `a` be the last before the main peak first appears and `b` the first at it. **Assumption: no heap block was freed and re-allocated between two consecutive reports.** Then any free in (a, b] would leave a free chunk at `b`, which `b`'s tiling rules out. So no block was freed in that interval, `heap_live` does not fall in it, and `heap_live(t*) ≥ heap_live(a)`. Hence

```text
lower = fixed + static − Σ_bss size_i + heap_live(a) − Σ_heap size_i + P_main
      = whole_hw − (heap_peak − heap_live(a)) − Σ_{i≠main} P_i − msp_peak
```

The reports support the assumption (`hs_cur == hs_max` on every line, the arena tiled at every report, `rpt_alloc = 0`) but do not prove it: they print no allocation count, so a free followed by a re-allocation of the same chunk between two reports would leave no trace.

v1's heap steps once, by 88 B of chunk (56 B requested + 32 B overhead), between rep 5 (`t_ms` 20,603) and rep 6 (25,610). Its main-thread peak 4,584 B first appears at rep 4 (15,602 ms; rep 3 at 10,210 ms shows 1,760 B), before the step, so `heap_live(a)` is the pre-step 23,508 B. Subtracting only the stack and MSP peaks from `whole_hw` would keep the post-step heap and overstate v1's lower end by 88 B (79,244 B). The v2 heap is 23,508 B on every line; its main peak 4,616 B first appears at rep 28 (135,700 ms).

| Probe run, 25 Oct | Lower (derived) | Upper = `whole_hw` (RAM high-water) | Width |
|---|---:|---:|---:|
| v1 `nano33ble_peakram` | 79,156 B (30.2%) | 80,660 B (30.8%) | 1,504 B |
| v2 `nano33ble_v2_f1a_peakram` | 85,220 B (32.5%) | 86,644 B (33.1%) | 1,424 B |

The lower end leaves out two floors that also hold: each of the four threads that are not running keeps at least a 64 B saved context on its stack, and the thread-mode MSP sits 64 B below `__StackTop` (PR_INFO `msp_paint`). Counting them would raise the lower end by 320 B. Both ends are for the probe image under that workload: no finger, forced decisions on the dark floor, no central. The unmodified live images, finger-on decisions and a connected central are not covered.

## 8. Sampler: a measured lower bound (added 31 Oct 2026)

**What it adds.** The sampler images `nano33ble_peakram_sampler` and `nano33ble_v2_f1a_peakram_sampler` are the 25 Oct probe envs plus two flags. `-DPEAK_RAM_LIVE_GATE` restores the shipped finger gate and BLE payload (no forced decisions, `force=0`). `-DPEAK_RAM_SAMPLER` adds `include/peak_ram_sampler.h`. Each run then reports both ends from the same capture: `whole_hw` (section 3, the RAM high-water and upper end) and `samp_max`, the largest instantaneous total actually observed (a measured lower end). `s_win` is the largest total since the previous `PR:` line, so the analyser can take the maximum inside the capture window. Validity checks: [peak-ram-runbook.md](peak-ram-runbook.md), sampler section, pre-registered in `f782745` before this code existed.

**Timer.** TIMER4, as in the `nano33ble_v2_looptiming2` sampler (`outputs/firmware-v2-looptiming2/`, which ran it at 4 kHz on this board). In both probe ELFs nothing references TIMER3 or TIMER4 (no literal in 0x4001_A000–0x4001_BFFF; their IRQ handlers are the weak defaults). The occupied peripherals are TIMER0 (Cordio radio timing), TIMER1 (us_ticker), TIMER2 (Cordio), RTC1 (lp ticker, the RTOS tick through SysTimer) and TWIM (I2C, sensor). TIMER4 runs at 1 MHz (16 MHz / 2⁴) with CC[0] = 250 and clear-on-compare: **4 kHz**. Its vector goes into the RAM vector table at 0x2000_0000 through `NVIC_SetVector`; the sampler does not start if `SCB->VTOR` is anywhere else.

**Priority.** From the ELF: `svcRtxKernelStart` writes 0xFF to the PendSV priority byte and 0xC0 to SVC (`0xFE << 5`). Mbed's `OS_Tick_Setup` writes 0xE0 to the SysTick byte, and Mbed's SysTimer pends SysTick from the RTC1 interrupt. With 3 priority bits that makes SVC level 6, and PendSV and SysTick level 7. The sampler reads the three bytes at the end of `setup()`, prints them (`prio_svc prio_pendsv prio_systick`), and sets TIMER4 to the more urgent of PendSV and SysTick (`smp_prio`, 7 here). Consequences:
1. PendSV and SysTick cannot preempt the sampler (same level), and the sampler cannot preempt them. SVC is synchronous: only an `svc` instruction raises it, no interrupt handler executes one, and a level-7 handler can never be active at the same time as SVC. `SVC_Context` is the only code that writes a thread's saved `sp`, PSP (`msr PSP`) and `osRtxInfo.thread.run.curr`, and it runs only inside SVC, PendSV and SysTick. So no context switch is half done while the sampler runs, and none can start until it returns. As a guard, a tick is still skipped if SHCSR shows any of the three active (`s_skk`; expected 0).
2. Every device interrupt (radio, USB, I2C, RTC) preempts the sampler, so it adds no latency to them.
3. The sampler only ever interrupts thread mode, directly or by tail-chaining. PSP is then the running thread's SP with its hardware frame, and MSP is at its thread-mode value.

**Each sample is a simultaneous total.** The quantities and the only code that changes them:
- thread SPs and `run.curr`: thread execution and `SVC_Context`. Neither can run while the sampler runs (point 1).
- heap break and free list: `_malloc_r`/`_free_r`, under the newlib malloc mutex. In both ELFs `_sbrk` is called only from `_malloc_r` and the probe's zero-increment reads. The mutex is an RTOS mutex, which an interrupt cannot take, so allocation happens in threads only. This is the same premise as section 2. If the mutex has an owner or a lock count at the tick, the tick is skipped (`s_skl`). Otherwise no thread is inside the allocator, and none can enter it before the sampler returns.
- MSP: higher-priority interrupts that preempt the sampler push and pop their frames and leave MSP where it was. The sampler reads MSP once, in its entry stub.

All terms therefore describe one instant, the sampler's entry. To find the malloc mutex, the probe takes `__malloc_lock` once at the end of `setup()` and reads the new head of its own RTX `mutex_list`; it checks the head changed, has the mutex id and is owned by itself. The thread table (control block, stack base, top) is built once at the end of `setup()`, when all five threads exist. Every report checks it against `osThreadEnumerate` (`s_set`). A tick is skipped (`s_skt`) if a table entry is no longer an RTX thread, its SP is outside its stack, or the running thread is not in the table.

**The total is in_use(t).** At each valid tick:

```text
total   = s_const + heap + Σ_i (top_i − SP_i) + (__StackTop − MSP_stub)
s_const = fixed + static − Σ_i size_i          (all five stacks, .bss and heap)
heap    = break − __end__ − Σ free newlib-nano chunks
```

Section 7's `in_use(t) = fixed + static − Σ_bss (SP_i − base_i) + heap_live − Σ_heap (SP_i − base_i) + msp` rearranges to exactly this, with `SP_i − base_i = size_i − (top_i − SP_i)`. `heap` equals section 7's `heap_live` (allocated chunks) when the arena has no alignment gaps. Each `PR:` line checks that: `hl` (the same walk, done under the mutex) must equal `hs_cur + hs_ovh` on lines where the heap statistics did not change across the walk (`hl_st=1`; pre-registered check `heap_source`).

**samp_max ≤ whole_hw must hold.** Term by term, with section 7's identity:
- heap: `heap ≤ break − __end__ ≤ heap_peak`, because the probe raises `heap_hw` to the highest break seen.
- stacks: every sampled SP points at a word that has been written. For a thread that is not running, that word is the r4 slot `SVC_ContextSave` pushed, or the 16-word frame RTX writes at creation. For the running thread, it is the r0 slot of the hardware frame stacked when the sampler (or the interrupt it tail-chained from) was entered. A written word is touched, so `top_i − SP_i ≤ P_i`. Section 7's condition (U1), a frame allocated but not written, cannot break this: the frame stacked at entry is written below any such frame.
- MSP: the entry stub pushes r4 and LR, then reads MSP, which points at the pushed r4: `__StackTop − MSP_stub ≤ msp_peak`. The body's own frame below it is not counted. In the ELF it is 36 B of pushes plus a 20 B local area, part of which is never written, and counting it could have let a sample exceed `whole_hw`. That is why MSP is read in the stub, not in the body.

What remains is (U2), a written word that happens to equal a fill pattern at a watermark edge. The analyser checks `s_win ≤ samp_max ≤ whole_hw` on every line (`bound_order`).

**Lower bound, not the peak.** Every valid sample is a true simultaneous total, so `samp_max ≤ peak`. It can be below the peak. The sampler sees nothing between ticks (250 µs) or inside interrupt handlers: device-interrupt MSP depth, and anything that exists only in handler mode. It starts at the end of `setup()`, and a tick may be lost when the sampler stays pending for more than one period.

**Cost.**
- Static RAM: the sampler's variables are 419 B (`nm`: table, counters, maxima, snapshot, scratch). `.bss` grows by 432 B (v1) and 424 B (v2) over the 25 Oct probe images, alignment included (89,384 / 95,408 B static against 88,952 / 94,984 B). Archive README: section deltas against the live images.
- Frames: 8 B stub push and a 56 B body frame on the MSP (not in samples; in `msp_peak`); the 32 B hardware frame on the interrupted thread's stack, as for any interrupt.
- CPU: one valid tick runs about 200 body instructions with five threads and an empty free list, plus the 6-instruction stub and exception entry and exit. That is roughly 300–400 cycles, or 5–6 µs at 64 MHz: 2.0–2.5% of the CPU at 4 kHz, all taken from thread mode. The v2 decision block (~440 ms) grows by about 11 ms, far inside the 640 ms FIFO budget, so `Hz:` ≥ 49.5 is not at risk and the rate stays at 4 kHz. The board run measures the body's cycles (`s_cyc_max`, `s_cyc_sum`; entry, exit and stub not included).

**Paper wording after a valid run:** *"peak RAM between [lower] and [whole_hw] on the probe image, finger on, dashboard connected"*, with lower = the largest `s_win` inside the capture. Still not covered:
- the unmodified live image: the probe adds static RAM (+944 / +952 B for the 25 Oct probe, +1,376 B for the sampler images over the live images), its own stack frames and the sampler's interrupt frame;
- other workloads;
- instants the sampler cannot see.

**Note, 1 Nov 2026: the malloc-mutex lookup and the heap-lock skip, corrected.** The 1 Nov board session stopped at identify: `PR_INFO: … smp=0 … hlock=0x00000000` on both images, `s_n=0` on every `PR:` line (`outputs/peak-ram-sampler-board-20261101/SESSION.md`). The cause is in RTX as compiled into this core. The package ships no RTX source, so this is read from the ELF. `svcRtxMutexRelease` lowers `lock`, and when it reaches 0 it unlinks the mutex from the owner's `mutex_list`. It writes `owner_thread` only when it hands the mutex to a waiting thread. With no waiter, `owner_thread` keeps naming the old owner after the last release. `svcRtxMutexAcquire` decides free or held by `lock` alone. The lookup above ended with "`owner_thread == nullptr` and `lock == 0` after `__malloc_unlock`", so it could never succeed, and `prs_start()` returned before starting TIMER4. The skip test in "Each sample is a simultaneous total" ("an owner or a lock count") would have failed the same way. After the first lock the stale owner would have skipped every tick (`s_skl = s_n`), and the pre-registered `skips` check would have failed. Changes, from the r2 images in `outputs/firmware-peakram-sampler-prepared-r2/` on:
- *Skip test.* A tick is skipped when the mutex's `lock` ≠ 0, which is RTX's own test. A hand-over to a waiter happens inside SVC: `lock` drops to 0 and is set back to 1 before SVC returns. The sampler cannot run inside SVC (point 1 of "Priority"), so it never sees `lock = 0` while a thread is inside the allocator. The argument "Otherwise no thread is inside the allocator" stands, with "lock count 0" in place of "no owner and no lock count".
- *Lookup.* The method is unchanged: take `__malloc_lock` once at the end of `setup()` and read the new head of the caller's `mutex_list`. While the lock is held, the head must have changed, carry the mutex id and be owned by the caller with `lock = 1` (a fresh acquire). After `__malloc_unlock` the mutex must be off the caller's list again. The stale owner and the lock count after the release are printed but not tested. In both r2 ELFs the mutex the lookup should find is mbed's `malloc_mutex_obj` (`mbed_toolchain_init` passes it to `osMutexNew` as `cb_mem`, `cb_size` 28 = `sizeof(osRtxMutex_t)`, attributes recursive, priority-inherit, robust).
- *Diagnostics.* `PR_INFO:` appends `st_hlock st_set st_vtor` (the three start preconditions as evaluated) and `lk_step lk_ipsr lk_mask lk_self lk_pre lk_head lk_id lk_own lk_cnt lk_post lk_rel lk_stale` (each lookup step; `lk_step` 0 = found). No check reads them. The old fields and the line format are unchanged.
- *Cost.* Static RAM grows by 64 B over the 31 Oct images (v1 89,448 B, v2 95,472 B). The r2 images therefore add 1,440 B over the live images, not 1,376 B. The tick body does one load fewer.

No pre-registered check changes: `hlock` is still the mutex address, and `config` still needs `smp=1` and `hlock` ≠ 0. Everything else in this section stands.

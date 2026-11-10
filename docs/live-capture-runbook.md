# Live capture runbook — still / wiggle / walk

Operational checklist. Do this at the board when you get home. Estimated 30–40 minutes.

This is the one remaining evidence gap: everything else is already measured on disk
(`docs/evidence-summary-2026-10.md`).

---

## What this produces

Three serial logs showing how the deployed pipeline behaves when the finger is still, moved, and
re-settled. The claim it supports is **hardware robustness under motion with N=2 debounce** — not a motion
classifier, not OpenOximetry motion, not clinical anything.

Outcomes to read: `KAN_Prob`, `TopFeature`, `Candidate`, `Confirmed`. **Not** `SpO2_est` — that is the toy
Red/IR formula and is not TinyKAN.

---

## Before you start

### Toolchain prerequisite — check this first

Items 3 and 4 below (all-14 attributions on hardware, GPIO D2 check) need a **firmware rebuild**. As of
7 October 2026 **no working C++ toolchain is installed on this machine**: both
`C:/Program Files/Microsoft Visual Studio/18` and `.../2022` are empty stubs, and `cl.exe`, `vcvars64.bat`
and `cmake` are all absent. The 10 September build used MSVC 19.51 from `VS/18/Community/VC/Tools/MSVC/14.51.36231/`,
which is no longer present.

**Do this before you sit down at the board**, or you will get halfway and stop:

- [ ] Either reinstall **Visual Studio Build Tools** with the *Desktop development with C++* workload and
      CMake, **or** build on whichever machine still has the toolchain.
- [ ] Confirm with `cmake --version` and a test compile.
- [ ] Then: `cmake --build build/parity/audit-release` and run `test_runner.exe` and `stress_runner.exe`.
      The sources were edited on 7 Oct and are **not yet compile-verified**.

If you cannot get a toolchain tonight, do only the serial captures (still / wiggle / recover / walk / light).
Those need no compiler. Leave items 3 and 4 for a second session.

### Board and kit

- [ ] Board on **COM8**, Nano 33 BLE (`USB VID:PID=2341:805A`). Ignore COM3/COM4 — those are Bluetooth.
- [ ] **Close every serial monitor / PlatformIO monitor holding COM8.** If upload or open fails with
      `PermissionError (13)`, something still has the port.
- [ ] Enclosure: the printed `simple-box-v0` box and the tighter lid (`LIP_GAP` 0.15). Black PLA.
- [ ] Black foam or a cut-off glove finger for the finger-hole collar, if you have made one.
- [ ] Phone or desk lamp nearby — you will need it for a light check.
- [ ] Do **not** reflash. The board should already have the current build (finger gate 3000, LED 0x3F,
      sampleRate 200 / sampleAverage 4, FIFO drain, RGB LED off, serial every 25 samples). Only flash if
      serial looks like an old build.

**Sanity that the build is current:** idle `NO FINGER` with IR ~150 is normal. IR ~14500 with `NO FINGER`
means the **old 50000** finger gate is still on the board. `Hz:` is samples counted divided by seconds
since boot — the real ingest rate, not the print rate. It should read about **50**. `Hz: 3.12` means the
old build is on the board (blocking reads at sampleRate 50 gave 3.125 samples/s); flash before capturing.

---

## Capture

Command, from `firmware/Arduino Nano 33` in the repository root:

```
pio device monitor --port COM8 --baud 115200 | tee ../../../outputs/live/2026-10-07_<condition>.txt
```

One file per condition. `<condition>` is `still`, `wiggle`, `walk`, `recover`, `light`.

### What to do per condition

Why these durations: the first decision needs 15 s of unbroken finger contact, then one decision arrives
every 5 s, and each decision looks at up to 20 s of signal (5 s past context, 10 s window, 5 s future
context). A 2–3 s movement therefore shows up across several consecutive decisions spread over roughly
25 s, and you need baseline windows before it and clean windows after it to see onset and recovery.
Lifting the finger resets the pipeline and restarts the 15 s wait.

| File | Action | Duration | What you want to see |
|---|---|---|---|
| `2026-10-07_still.txt` | Finger on the sensor, pad down, **absolutely still** | 120 s | `Signal: OK`, first decision at `Samples: 750`, then real `KAN_Prob` every 250 samples, `Candidate: NO` unless it genuinely flags |
| `2026-10-07_wiggle.txt` | Still for 30 s, **wiggle once** — tap the desk or rock the finger for 2–3 s — then still for at least 60 s | 90–120 s | `Candidate: YES` is fine and expected. **`Confirmed: YES` would mean N=2 did not hold** — record it honestly if it happens |
| `2026-10-07_recover.txt` | Hands off, back to **still** | 90 s | `Candidate` should drop back to `NO` |
| `2026-10-07_walk.txt` | Still for 30 s, then standing, walk on the spot for 30 s, then still for 60 s (only if the cable is long enough) | 120 s | Same read as wiggle. If the cable will not reach, say so and skip it |
| `2026-10-07_light.txt` | Still for 30 s, then shine the lamp or phone torch at the box for 10 s, then 30 s without; closed vs open | 90 s per state | This is the enclosure's own test. IR level and `KAN_Prob` should hold steady when closed |

### Read the log

Each completed window prints a `Samples:` count (750 for the first, then +250 each window), a non-zero
`KAN_Prob`, and a real `TopFeature`. The very first lines are the buffer filling: `KAN_Prob: 0.00` /
`Win: 0` / `Samples: 0` is not a result.

`Win` is 0-based, so the first completed window is `Win: 0`. The first decision needs **750 samples**
(10 s window + 5 s future context), which is **15 s** of unbroken finger contact at 50 Hz; after that a
decision arrives every 250 samples (5 s). `Hz:` is samples counted / seconds since boot and should read
about 50; lines print every 25 samples (~0.5 s) and on every completed window.

A `WARNING: sensor FIFO overflow` line means samples were lost. The firmware resets the pipeline so no
window spans the gap, and the 15 s wait starts again. Note the time — it is a gap in the evidence.

---

## Record alongside the log

Write these next to each file — the log alone will not tell you what happened physically.

- Start time, condition, and how long the movement actually lasted.
- Whether the finger was the thumb or index. **Thumb fills the opening better**; index leaves more gap.
- Collar on or off.
- Enclosure lid on or off, and if the light test was open or closed.
- Anything odd: finger lifted, wire pulled, board moved.

A gap in the log is a gap in the evidence. Do not fill it from memory.

---

## After

- [ ] Five files in `outputs/live/`.
- [ ] For each: does `Candidate` spike on motion and return to `NO` on still?
- [ ] Did `Confirmed` ever fire during a single wiggle? **If yes, that is a finding, not a failure.** It means
      N=2 was not enough on this hardware and the report says so.
- [ ] Light test: did `KAN_Prob` move when the lamp hit the open sensor versus the closed box?

Copy the numbers into the Week's workbook entry. The honest summary is what the device actually did.

---

---

## Sunlight test — CANCELLED (decided 7 October 2026)

**Not doing this.** The plan below is kept as a record of what was considered, not as instructions.
Do not propose this test again. Reasons:

- **Ambient light rejection in pulse oximetry is established engineering, not open science.** Commercial
  oximeters publish rejection ranges and the MAX3010x datasheet characterises it. It is not a
  contribution anyone would credit.
- **It is not part of the research question.** The contribution is the integration of grouped
  oxygen-risk classification, on-device TinyKAN, and a microcontroller-budget explanation. The
  enclosure is build hygiene, not a research claim.
- **A thumb in a printed box in sunlight could not be published as a light-rejection characterisation.**
  That needs calibrated irradiance, known spectral content, and multiple angles. A desktop result would
  be anecdote presented as evidence, which is worse than not testing.
- **Time is better spent on the paper** (due 22 November).

What we already have is sufficient and correctly labelled as an observation, not a claim: at deployed
LED current (`0x3F`), flipping the room lights moved the detector output by less than 0.4%, within
noise (`docs/evidence-summary-2026-10.md` section 8.4). Any mention of light in the write-up must carry
that framing and a citation to standard pulse-oximetry practice.

**How the enclosure is described instead:** mechanical protection. Repeatable finger placement was not
measured. Seven of the 14 features are amplitude-based, so placement can move them; do not say the
opening removes that variance.

### Original plan (superseded, not to be run)

The room-light test on 7 October was a **null at indoor intensity** (<0.4%, within noise). That bounds
nothing about strong light. Direct sunlight is roughly 100-1000x brighter and is the condition a
wearable oximeter would actually face, so this is the test that can validate the enclosure rather than
test a non-existent effect.

### Design

The comparison is **shade vs direct sun**, not "lamp on / lamp off". Four conditions, in this order:

| # | Condition | What it establishes |
|---|---|---|
| 1 | **Open box, shade** | Baseline. Must match 7 Oct levels (Red ~63k, IR ~63k). |
| 2 | **Open box, direct sun** | Does strong light reach the detector at all? |
| 3 | **Closed box, direct sun** | Does the enclosure block it? |
| 4 | **Closed box, shade** | Recovery, and closes the loop. |

**The claim is made or broken on 2 vs 3.** If 2 moves and 3 does not, the enclosure works outdoors.
If 2 and 3 move equally, it does not protect. If neither moves, the sensor is immune.

### Rules — this is where the earlier tests failed

- **Nothing moves near the board when the condition changes.** Move the whole rig, wait for the finger
  to settle, then capture. Do not move a lamp by hand — that confounded the 7 Oct attempt.
- Sun conditions: **at least 30 s in the new light state before the capture starts**, so thermal and
  settling drift has passed. The 7 Oct captures all showed slow downward drift over the first 20 s.
- Same finger, same position, enclosure closed unless the condition says otherwise.
- Record the time, whether shade was a hand/box/desk, and how direct the sun was (full disc vs hazy).
- Capture **90 s** per condition.

### What to measure

Same as 7 Oct: 15 s before vs 15 s after does not apply here — instead compare the **steady plateau**
of conditions 1 and 2, then 2 and 3. Report the shift in counts and as a percentage, with the standard
deviation of each plateau, so the shift can be stated in sigma.

Report even a null. A null in sunlight is a stronger result than a null indoors.

### Note on thermal effects

Direct sun heats the board and the finger. A slow monotonic drift is **not** a light effect — look for a
**step between plateaus**, not a slope. If in doubt, capture longer and compare the first and last 20 s
of the sun condition for internal consistency.



- Do not interpret `SpO2_est` as model output.
- Do not reflash to chase a number.
- Do not treat a single `Candidate` as a confirmed low-oxygen event — that is exactly what N=2 exists to stop.
- Do not add IMU. The board IMU is optional logging and is not a TinyKAN input.
- Do not record anyone but yourself.

# simple-box-v1

A plain rectangular box around the breadboard with a **sliding lid** that clicks shut. Two holes only: one for the USB wire and one for the finger. No screws, nut pockets, side pocket, deck or sensor bay.

v1 changes only the lid and the top of the long side walls. The box is otherwise the same as `../simple-box-v0`, which Jason has already printed and holds shut with a rubber band.

**Fit is not verified.** v1 has not been printed or test-fitted. How hard the detent clicks depends on the printer and the filament.

## Files

| File | What |
| --- | --- |
| `src/params.py` | All dimensions (mm). Edit here. |
| `src/box.py`, `src/lid.py` | Parametric build123d sources (cadgen). |
| `STL/box.stl` | Print open side up, as exported. |
| `STL/lid.stl` | Print flat on its face, as exported. |
| `STEP/box.step`, `STEP/lid.step` | Exact geometry for measuring and editing. |

To regenerate after editing `params.py` (from this folder):

```
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.14 python src/box.py
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.14 python src/lid.py
```

## Orientation

The USB plug is at the left end (-X). The finger presses the sensor pad-down from the right end (+X). Power rails are removed.

## Sliding lid

- The lid is a flat plate that **slides along X**. It enters from the **-X (USB) end** and closes toward +X.
- It is captured vertically in grooves cut into the **thickened tops of the two long side walls**, so it cannot fall into the box or lift off.
- The **+X end wall stops the lid** when it is fully closed.
- Sliding clearance is **0.4 mm** on each side (Y and Z).
- The top band of the side walls is thickened **0.5 mm inward and 1.5 mm outward** so the groove and the detent pockets still leave **1.5 mm of solid material** behind them. The interior opening stays 36 mm wide, which is still clear of the 35 mm board.
- The lid has a **tab** at its +X end that fills the top of the finger notch when closed, so the notch is not left open.
- The lid **can** be removed with the USB cable plugged in. It rides in the grooves at Z 37–39.4 while the cable exits the end wall at Z 12–24, so the lid slides out above the cable. Only a cable that loops up above the rim would catch on it.

## Detent

- Each long edge of the lid is **slotted near its -X end to form a flexible arm**. A 45-degree bump on the arm's outer edge snaps into a **pocket in the back wall of the groove**.
- Both arms flex sideways, in the print plane, so **both bumps engage at once** and the lid is held square rather than cocked.
- The bump and pocket centres sit at **-42.5 mm** with the lid closed, near the lid's trailing edge, so the detent resists the lid sliding back out.
- The bump stands **0.8 mm** proud of the lid edge with a 0.6 mm flat crest. **0.1 mm** clearance between the seated bump and the pocket back, and **0.2 mm** along X between the flanks.
- The detent holds the lid against tipping and being carried in a bag, but it is **hand-openable**: grip the recess and pull. It is a click, not a lock.
- The arms must not be brittle. If a bump snaps off on your printer, raise `DETENT_DEPTH` to lengthen the arm and make it more flexible, or drop the detent and use a band.

## Finger grip

- A shallow **half-round recess in the lid top at the -X edge**, radius 10, **0.5 deep**, leaving **1.5 mm** of material under it.
- It is a grip to pull the lid open, not a hole through the lid.

## Dimensions

Measured inputs:

- Breadboard: 85 x 35.
- Tallest content: 35. This is the interior clear height.
- USB plug: 10 x 10.

Derived:

- Clearance: 1 per side, so the interior is **87 x 37 x 35**.
- Walls and floor: 2. Minimum solid material anywhere: **1.5**.
- Box exterior: **91 long x 44 wide x 40.9 high**. Wider and taller than v0 because the top band carries the groove and the detent pockets.
- Lid: **91 x 40.8 x 2** flat plate, with the +X tab and the two detent arms.
- Assembled: **91 x 44 x 40.9**.

## Holes (both PROVISIONAL)

- **USB-wire hole**, left end wall: 14 wide x 12 high. It is bigger than the 10 x 10 plug so the plug can pass through. The bottom edge is 10 above the interior floor, centred across the width. This position is provisional; check it against the real plug height on the board.
- **Finger opening**, right end wall: a U-shaped notch 24 wide, centred, open at the top through the rim (no bridge), with a rounded bottom (radius 12) at interior mid-height (17.5 above the floor). The sliding lid's +X tab closes the top of this notch. Provisional, not based on a measured finger; check that a finger can reach the sensor.

## Not checked

- Fit of the breadboard, plug and wiring inside the box.
- Whether the lid slides freely and how hard the detent clicks on this printer. The 0.4 mm sliding clearance is the number to open first if it binds.
- Whether the detent arms survive repeated opening.
- Whether a finger can reach the sensor through the slot.
- No slicer was run.

# simple-box-v0

A plain rectangular box around the breadboard with a separate flat lid.
It has two holes: one for the USB wire and one for the finger. There are no screws, nut pockets, side pocket, deck or sensor bay.

**Fit is not verified.** The box has been printed once with the original lid (0.3 lip clearance), which was loose. The tightened, single-orientation lid below has not been printed or test-fitted yet.

**Lid change (box unchanged):** `LIP_GAP` is now **0.15** (was 0.3). The lid is now **single-orientation**: the lip notch is at the finger end only, and the USB end of the lip is solid to seal out light. Only `STL/lid.stl` and `STEP/lid.step` were regenerated. `STL/box.stl` and `STEP/box.step` were not rebuilt, so the already-printed box is still correct. A fit tighter than 0.3 may need light sanding of the lip, depending on the printer.

## Files

| File | What |
| --- | --- |
| `src/params.py` | All dimensions (mm). Edit here. |
| `src/box.py`, `src/lid.py` | Parametric build123d sources (cadgen). |
| `STL/box.stl` | Print open side up, as exported. |
| `STL/lid.stl` | Print outside face down (lip up), as exported. |
| `STEP/box.step`, `STEP/lid.step` | Exact geometry for measuring and editing. |

To regenerate after editing `params.py` (from this folder). Only rebuild the box if you mean to change it. `LIP_GAP` affects the lid only.

```
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.14 python src/box.py
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.14 python src/lid.py
```

## Orientation

The USB plug is at the left end (-X). The finger presses the sensor pad-down from the right end (+X). Power rails are removed.

### Lid orientation (one way only)

| Lid lip end | Goes over | Lip there |
| --- | --- | --- |
| **Notched end** (26-wide gap in the lip) | **Finger end** (+X, U-shaped finger opening) | Gap, so the lid clears the top of the finger U-notch |
| **Solid end** (unbroken lip) | **USB end** (-X, USB-wire hole) | Continuous, so it seals light |

To fit the lid, look at the lip: **put the notched end over the finger opening.** If the lid is turned the other way, the solid lip end lands on the open top of the finger U-notch and the notch end leaves a light gap at the USB end. That is wrong.

In the model files (print orientation, lip up), the notch is at +X.

## Measured inputs

- Breadboard: 85 x 35.
- Tallest content: 35. This is the interior clear height.
- USB plug: 10 x 10.

## Derived dimensions

- Clearance: 1 per side, so the interior is **87 x 37 x 35**.
- Walls and floor: 2.
- Box exterior: **91 x 41 x 37** (open top).
- Lid: 91 x 41 plate, 2 thick. It has a 1.5-thick locating lip that is 3 deep and has **0.15 clearance per side** to the box's inner wall (`LIP_GAP`, was 0.3). The lip outside is 86.7 x 36.7. Overall height is 5.
- Assembled: **91 x 41 x 39**.
- The lid plate rests on the box rim, and the lip drops inside. There are no fasteners.
- The lip is notched 26 wide at the **finger end only** so it clears the top of the finger slot. The USB end of the lip is solid. The lid fits one way round only (see *Lid orientation*).

## Holes (both PROVISIONAL)

- **USB-wire hole**, left end wall: 14 wide x 12 high. It is bigger than the 10 x 10 plug so the plug can pass through. The bottom edge is 10 above the interior floor, centred across the width. This position is provisional; check it against the real plug height on the board.
- **Finger opening**, right end wall: a U-shaped notch 24 wide, centred, open at the top through the rim (no bridge; the lid closes the top), with a rounded bottom (radius 12) at interior mid-height (17.5 above the floor). Provisional, not based on a measured finger; check that a finger can reach the sensor.

## Not checked

- Fit of the breadboard, plug and wiring inside the box.
- Lid snugness: the 0.15 lip clearance has not been printed. At 0.3 the lid was loose. At 0.15 it may be snug, or it may need light sanding of the lip, depending on the printer.
- The light seal at the USB end with the now-solid lip.
- Whether a finger can reach the sensor through the slot.
- No slicer was run.

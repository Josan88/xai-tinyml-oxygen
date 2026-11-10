# Covered finger-pocket enclosure — concept v0

This first concept keeps a 400-point breadboard and moves the optical sensor into
a covered finger pocket beside it. The finger enters horizontally, pad down. A
removable deck separates the finger from the sensor PCB and wiring below. The
controller remains on the breadboard. The roof opens with four screws.

**This is a dimensional concept, not a verified fit for your hardware.** The STL
files are valid closed meshes, but the sensor module, USB plug, assembled wiring
height and finger fit have not been measured or physically tested. Prototype the
small finger sample first. The enclosure does not establish sensor accuracy.

## Dimensions used

- Body and lid assembled: 107 × 90 × 40.4 mm.
- Breadboard allocation: 85 × 56 mm, with another 1 mm clearance per side.
- Electronics height allowance: 32 mm above the base floor, plus 3 mm headroom.
- Finger pocket: about 51.7 mm usable depth, 25.4 mm minimum clear width and 22 mm
  clear height. The optical contact is 32 mm from the outer entrance.
- Provisional sensor PCB: 20 × 16 × 1.6 mm; package height 1.5 mm above the PCB.
- Optical opening: 8 × 6 mm, with a small bevel. Confirm that both the emitter
  and receiver are completely exposed on your actual board.
- USB opening: 20 × 16 mm on the opposite end from the finger entrance; its
  position is provisional. Orient the controller toward this opening.
- Walls 2 mm; base and lid 2.4 mm. Finger deck 2 mm, with 0.3 mm side clearance.

The user identified a half-size, 400-point breadboard. Reference products vary:
[Adafruit lists 82.6 × 55 mm](https://www.adafruit.com/product/64), while
[SparkFun lists 83.5 × 54.5 mm](https://www.sparkfun.com/breadboard-translucent-self-adhesive-red.html).
The allocation above is an assumption, not a measurement of the user's board.
Check interlocking tabs, power rails and any metal backing plate too.

Project notes call the sensor MAX30105; the conversation mentioned MAX30102.
The installed library name does not identify the physical module conclusively.
The sensor dimensions above are placeholders and must be adjusted to the actual
breakout board, connector orientation, optical package position and underside
components. The preview's controller block is schematic, not an exact board model.

## Files and printing

- `base.stl`: full base, already oriented open-side up.
- `lid.stl`: full roof, already oriented outside-face down; underside supports
  grow upwards from the plate.
- `finger-deck.stl`: removable finger floor, already oriented contact-face down.
- `finger-fit-sample.stl` and `finger-fit-roof.stl`: shortened base and roof for
  trying the finger pocket. Use with the same `finger-deck.stl`. The small roof
  is held by hand for the fit trial; it does not include the full case fasteners.
- `assembly.scad`: exploded assembly preview in OpenSCAD. Individual `.scad`
  files contain the corresponding printable geometry.
- `dimensions.json`: adjustable dimensions, all in millimetres.
- `build.py`: rebuilds the CAD and STL files after dimension changes.
- `verification.json`: mesh and provisional assembly-interference results.

For an initial indoor fit trial, use opaque PLA, a 0.4 mm nozzle, 0.2 mm layers,
four perimeters and approximately 15–20% infill as starting settings. Use the
printer's normal material profile. Inspect the slicer preview, especially the USB
opening and bottom nut pockets: these short bridges may need local support on
your printer. The deck and roof are oriented to avoid support inside the finger
contact area. Deburr every finger-contact edge after printing.

Four 3.4 mm clearance bores and bottom hex nut pockets are provided for M3
hardware. Nut pockets are 6.2 mm across flats and 2.8 mm deep. Approximately
40 mm screw length is a starting point for this version; choose the actual
length so it engages the nuts without protruding below the base. Changing the
case height also changes the required screw length.

## Assembly and fit trial

1. Print the small base, small roof and finger deck first. Try the pocket with
   the roof held loosely in place. Check comfortable insertion and withdrawal;
   no spring clamp or interference grip is intended.
2. Verify the optical opening and module dimensions before mounting the sensor.
   Four provisional support pads contact the sensor PCB margins. Reposition or
   resize these to avoid underside components and connectors. Adjust their
   height through the sensor PCB and package-height parameters so the optical
   surface sits flush with the finger floor.
3. Route the sensor wires below the deck through the divider opening. Secure the
   module with removable nonconductive tape at its margins/back if required,
   keeping the optics and connector clear. Do not pull wires tight.
4. Fit the deck over the two locating keys on the side ledges. The keys limit
   forward sliding; the full lid's rear feet limit lift at the rear. The underside
   recess locates the sensor PCB. Check that the lid closes freely and the deck
   stays seated during finger insertion and removal.
5. After measurements and the finger trial, print the full base and lid. Secure
   the breadboard on the flat base with removable mounting tape. Its allowed
   envelope includes insertion clearance; do not rely on a press fit.
6. Insert the nuts and screws, then tighten lightly. Confirm that the USB plug
   fully seats and the roof does not press on jumper wires.
7. Compare raw PPG under room lighting, shading and movement. Outside-light
   blocking and sensor coupling need experimental verification; dark-looking
   plastic should not be assumed to block the sensor's infrared wavelength.

The opaque roof and walls are intended to reduce stray light while leaving a
direct finger-to-sensor optical path. This approach follows the mechanical
coupling considerations discussed by
[Analog Devices](https://ez.analog.com/ez-blogs/b/engineerzone-spotlight/posts/getting-to-reliable-sensor-data-interaction-between-electronics-optics-and-mechanical-design).
Printing clearances need iteration, as described in
[Prusa's design guidance](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164135).

## Regeneration

Install Python with `numpy`, `manifold3d` and `trimesh`. Change `dimensions.json`
and run `python build.py` in this folder. The generator validates closed meshes,
positive volume, connected parts, print-bed position, and interference between
the printed components and the provisional breadboard envelope. This checks
geometry, not printer tolerances or physical hardware fit. Generated `.scad`
files are viewable and editable, but regenerating replaces direct edits to them.

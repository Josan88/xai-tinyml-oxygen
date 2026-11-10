"""simple-box-v0 box: open-top rectangular box around the breadboard.

Two openings only: a closed USB-wire hole in the left (-X) end wall and a
U-shaped finger notch, open at the rim, in the right (+X) end wall. Printed open side up as modelled.
"""

from cadgen import build123d as bd
from cadgen import step, stl

from params import (
    FINGER_R, FINGER_W, FINGER_Y, FINGER_Z0,
    INNER_H, INNER_L, INNER_W,
    OUTER_H, OUTER_L, OUTER_W,
    USB_H, USB_W, USB_Y, USB_Z0,
    WALL,
)


@step(out="../STEP/box.step")
@stl(out="../STL/box.stl", mesh_tolerance=0.01)
def box():
    shell = bd.Box(OUTER_L, OUTER_W, OUTER_H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    cavity = bd.Box(INNER_L, INNER_W, INNER_H + 1, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    body = shell - bd.Pos(0, 0, WALL) * cavity

    # USB-wire hole: rectangular cut through the -X end wall.
    usb_cut = bd.Box(WALL * 3, USB_W, USB_H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    body -= bd.Pos(-OUTER_L / 2, USB_Y, WALL + USB_Z0) * usb_cut

    # Finger opening: U-shaped notch through the +X end wall, open at the rim.
    # Semicircular bottom (radius FINGER_R) plus straight sides up past the top.
    above_rim = OUTER_H - (WALL + FINGER_Z0 + FINGER_R) + 1
    u_profile = bd.Circle(FINGER_R) + bd.Rectangle(
        FINGER_W, above_rim, align=(bd.Align.CENTER, bd.Align.MIN)
    )
    notch = bd.extrude(bd.Plane.YZ * u_profile, amount=WALL * 3, both=True)
    body -= bd.Pos(OUTER_L / 2, FINGER_Y, WALL + FINGER_Z0 + FINGER_R) * notch

    body.label = "box"
    return body


if __name__ == "__main__":
    box()

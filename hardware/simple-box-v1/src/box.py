"""simple-box-v1 box: open-top rectangular box with a sliding-lid groove.

Two openings only: a closed USB-wire hole in the left (-X) end wall and a
U-shaped finger notch, open at the rim, in the right (+X) end wall.

The tops of the two long side walls are thickened (0.5 in, 1.5 out) and carry
a groove the lid slides in. The grooves are open at the -X end, where the end
wall stops at the lid underside, and closed at the +X end, which is the stop.
Just inside the -X entrance each groove has a shallow 45 deg detent pocket
in its back wall that the lid's bumps snap into when closed.
Printed open side up as modelled: the groove and pocket roofs are 45 deg
slopes plus a 1.1 flat ledge, and both wall steps are 45 deg chamfers, so no
supports. The inward ledge under each groove floor has a 1.5 square lip
above its chamfer, so it does not end in a knife edge.
"""

from cadgen import build123d as bd
from cadgen import step, stl

from params import (
    BOX_H, CEILING_Z,
    DETENT_BUMP, DETENT_CLEAR_X, DETENT_CREST, DETENT_DEPTH, DETENT_X,
    FINGER_R, FINGER_W, FINGER_Y, FINGER_Z0,
    GROOVE_BACK_Y, GROOVE_FLOOR_Z,
    INNER_H, INNER_L, INNER_W,
    LID_BEVEL, LID_HALF_W, MIN_MAT, OPEN_Y,
    OUTER_L, OUTER_W,
    SLIDE_GAP, TOP_DROP, TOP_IN, TOP_OUT, TOP_OUTER_Y,
    USB_H, USB_W, USB_Y, USB_Z0,
    WALL, WALL_TOP_Z,
)

# Groove roof follows the lid's 45 deg top bevel, offset by SLIDE_GAP:
# the sloped part lies on the plane y + z = ROOF_YZ.
SLOPE_TOP_Y = GROOVE_BACK_Y - SLIDE_GAP - LID_BEVEL       # 19.1
ROOF_YZ = SLOPE_TOP_Y + CEILING_Z                         # 58.5


def side_rail(sy):
    """Thickened top of one long side wall, with its groove, along full X."""
    wall_in = INNER_W / 2
    wall_out = OUTER_W / 2
    slope_bot_z = ROOF_YZ - GROOVE_BACK_Y                     # 38.5
    step_z = WALL_TOP_Z - TOP_DROP - TOP_OUT                  # 34.5
    yz = [
        (wall_in, step_z),                                   # overlaps the wall
        (wall_out, step_z),
        (TOP_OUTER_Y, step_z + TOP_OUT),                     # 45 deg outward
        (TOP_OUTER_Y, BOX_H),
        (OPEN_Y, BOX_H),
        (OPEN_Y, CEILING_Z),                                 # 1.1 flat roof
        (SLOPE_TOP_Y, CEILING_Z),
        (GROOVE_BACK_Y, slope_bot_z),                        # 45 deg roof
        (GROOVE_BACK_Y, GROOVE_FLOOR_Z),
        (OPEN_Y, GROOVE_FLOOR_Z),                            # groove floor
        (OPEN_Y, GROOVE_FLOOR_Z - MIN_MAT),                  # square ledge lip
        (wall_in, GROOVE_FLOOR_Z - MIN_MAT - TOP_IN),        # 45 deg inward
    ]
    profile = bd.Polyline(*[(sy * y, z) for y, z in yz], close=True)
    face = bd.make_face(bd.Plane.YZ * profile)
    return bd.extrude(face, amount=OUTER_L / 2, both=True)


def detent_pocket(sy):
    """Cutter for the detent pocket in the back wall of one groove.

    The groove section pushed DETENT_DEPTH further out, under the same 45 deg
    roof, limited along X by 45 deg flanks that match the lid bump plus
    DETENT_CLEAR_X. The -X flank is the catch the bump must ride over to open.
    """
    y0 = GROOVE_BACK_Y - 0.2                   # starts inside the (empty) groove
    y1 = GROOVE_BACK_Y + DETENT_DEPTH          # 20.5, pocket back
    yz = [
        (y0, GROOVE_FLOOR_Z),
        (y1, GROOVE_FLOOR_Z),
        (y1, ROOF_YZ - y1),
        (y0, ROOF_YZ - y0),
    ]
    face = bd.make_face(bd.Plane.YZ * bd.Polyline(*[(sy * y, z) for y, z in yz], close=True))
    section = bd.Pos(DETENT_X, 0, 0) * bd.extrude(face, amount=5, both=True)

    crest_y = LID_HALF_W + DETENT_BUMP         # 20.4, bump crest with lid centred

    def half(y):
        return DETENT_CREST / 2 + DETENT_CLEAR_X + (crest_y - y)

    xy = [
        (DETENT_X - half(y0), sy * y0),
        (DETENT_X + half(y0), sy * y0),
        (DETENT_X + half(y1), sy * y1),
        (DETENT_X - half(y1), sy * y1),
    ]
    plan = bd.extrude(bd.make_face(bd.Polyline(*xy, close=True)), amount=CEILING_Z - GROOVE_FLOOR_Z)
    return section & bd.Pos(0, 0, GROOVE_FLOOR_Z) * plan


@step(out="../STEP/box.step")
@stl(out="../STL/box.stl", mesh_tolerance=0.01)
def box():
    bottom = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
    shell = bd.Box(OUTER_L, OUTER_W, WALL_TOP_Z, align=bottom)
    cavity = bd.Box(INNER_L, INNER_W, INNER_H + 1, align=bottom)
    body = shell - bd.Pos(0, 0, WALL) * cavity

    body += side_rail(1) + side_rail(-1)

    # +X end wall carried up to the rim between the rails: closes the grooves
    # and stops the lid. Sits directly on the end wall, so no overhang.
    end_cap = bd.Box(WALL, 2 * GROOVE_BACK_Y, BOX_H - WALL_TOP_Z, align=bottom)
    body += bd.Pos(INNER_L / 2 + WALL / 2, 0, WALL_TOP_Z) * end_cap

    # Detent pockets, one per groove, just inside the -X entrance.
    body -= detent_pocket(1) + detent_pocket(-1)

    # USB-wire hole: rectangular cut through the -X end wall.
    usb_cut = bd.Box(WALL * 3, USB_W, USB_H, align=bottom)
    body -= bd.Pos(-OUTER_L / 2, USB_Y, WALL + USB_Z0) * usb_cut

    # Finger opening: U-shaped notch through the +X end wall, open at the rim.
    # Semicircular bottom (radius FINGER_R) plus straight sides up past the top.
    above_rim = BOX_H - (WALL + FINGER_Z0 + FINGER_R) + 1
    u_profile = bd.Circle(FINGER_R) + bd.Rectangle(
        FINGER_W, above_rim, align=(bd.Align.CENTER, bd.Align.MIN)
    )
    notch = bd.extrude(bd.Plane.YZ * u_profile, amount=WALL * 3, both=True)
    body -= bd.Pos(OUTER_L / 2, FINGER_Y, WALL + FINGER_Z0 + FINGER_R) * notch

    body.label = "box"
    return body


if __name__ == "__main__":
    box()

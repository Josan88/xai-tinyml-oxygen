"""simple-box-v1 lid: flat sliding plate, no fasteners.

Modelled in box XY (centred) and print orientation, which is also the use
orientation: underside down on Z=0. In use the underside sits at Z=37 on the
box. The long edges have a 45 deg top bevel that matches the groove roof.
A tab at +X fills the finger notch through the end wall thickness.

Detent: near the -X end a slot cuts each long edge free into a flexible arm.
A 45 deg bump on the arm's outer edge squeezes past the groove entrance and
snaps into the pocket in the groove back wall when the lid is fully closed.
The arms bend in the print plane, along the extrusion lines.

Grip: a shallow half-round recess in the top at the -X edge.
"""

from cadgen import build123d as bd
from cadgen import step, stl

from params import (
    ARM_L, ARM_SLOT_W, ARM_W,
    DETENT_BUMP, DETENT_CREST, DETENT_X,
    GRIP_DEPTH, GRIP_R,
    LID_EDGE_H, LID_HALF_W, LID_T, LID_X0, LID_X1,
    OUTER_L, TAB_HALF_W,
)


def bump(sy):
    """45 deg trapezoid bump on one long edge, full lid height before bevel."""
    y0 = LID_HALF_W - 0.2                      # overlaps the plate
    y1 = LID_HALF_W + DETENT_BUMP              # 20.4, crest

    def half(y):
        return DETENT_CREST / 2 + (y1 - y)

    xy = [
        (DETENT_X - half(y0), sy * y0),
        (DETENT_X + half(y0), sy * y0),
        (DETENT_X + half(y1), sy * y1),
        (DETENT_X - half(y1), sy * y1),
    ]
    return bd.extrude(bd.make_face(bd.Polyline(*xy, close=True)), amount=LID_T)


def arm_slot(sy):
    """Through slot, open at the -X edge, with a round end at the arm root."""
    y_mid = LID_HALF_W - ARM_W - ARM_SLOT_W / 2          # 17.1
    x_end = LID_X0 + ARM_L                               # -27.5
    cut = bd.Pos((LID_X0 - 1 + x_end) / 2, sy * y_mid) * bd.Rectangle(x_end - LID_X0 + 1, ARM_SLOT_W)
    cut += bd.Pos(x_end, sy * y_mid) * bd.Circle(ARM_SLOT_W / 2)
    return bd.extrude(cut, amount=LID_T + 2).moved(bd.Location((0, 0, -1)))


@step(out="../STEP/lid.step")
@stl(out="../STL/lid.stl", mesh_tolerance=0.01)
def lid():
    # Envelope with the 45 deg bevel along both long edges. It is wide enough
    # for the bumps, so they get the same bevel and clear the groove roof.
    bevel_yz = LID_HALF_W + LID_EDGE_H                   # bevel on y + z = 21.1
    w = LID_HALF_W + DETENT_BUMP + 1
    yz = [
        (-w, 0),
        (w, 0),
        (w, bevel_yz - w),
        (bevel_yz - LID_T, LID_T),
        (-(bevel_yz - LID_T), LID_T),
        (-w, bevel_yz - w),
    ]
    envelope = bd.make_face(bd.Plane.YZ * bd.Polyline(*yz, close=True))
    envelope = bd.Pos(LID_X0, 0, 0) * bd.extrude(envelope, amount=LID_X1 - LID_X0)

    plate = bd.Box(
        LID_X1 - LID_X0, 2 * LID_HALF_W, LID_T,
        align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN),
    )
    plate = bd.Pos(LID_X0, 0, 0) * plate
    body = (plate + bump(1) + bump(-1)) & envelope
    body -= arm_slot(1) + arm_slot(-1)

    # Finger grip: half-round recess centred on the -X edge.
    grip = bd.Cylinder(GRIP_R, GRIP_DEPTH + 1, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    body -= bd.Pos(LID_X0, 0, LID_T - GRIP_DEPTH) * grip

    tab_l = OUTER_L / 2 - LID_X1
    tab = bd.Box(tab_l, 2 * TAB_HALF_W, LID_T, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN))
    body += bd.Pos(LID_X1, 0, 0) * tab

    body.label = "lid"
    return body


if __name__ == "__main__":
    lid()

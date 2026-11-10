"""simple-box-v0 lid: flat cover with a locating lip, no fasteners.

Modelled in print orientation: outside face down on Z=0, lip pointing +Z.
Flip it over to fit; the plate rests on the box rim and the lip drops inside.
"""

from cadgen import build123d as bd
from cadgen import step, stl

from params import (
    INNER_L, INNER_W, LID_T,
    LIP_DEPTH, LIP_GAP, LIP_NOTCH_W, LIP_T,
    OUTER_L, OUTER_W,
)


@step(out="../STEP/lid.step")
@stl(out="../STL/lid.stl", mesh_tolerance=0.01)
def lid():
    bottom = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
    plate = bd.Box(OUTER_L, OUTER_W, LID_T, align=bottom)

    lip_l = INNER_L - 2 * LIP_GAP
    lip_w = INNER_W - 2 * LIP_GAP
    ring = bd.Box(lip_l, lip_w, LIP_DEPTH, align=bottom) - bd.Box(
        lip_l - 2 * LIP_T, lip_w - 2 * LIP_T, LIP_DEPTH, align=bottom
    )
    # Notch the finger (+X) end only so the lip clears the open finger notch.
    # The USB (-X) end stays solid for the light seal, so the lid fits one way round.
    notch = bd.Box(LIP_T * 3, LIP_NOTCH_W, LIP_DEPTH, align=bottom)
    ring -= bd.Pos(lip_l / 2 - LIP_T / 2, 0, 0) * notch

    body = plate + bd.Pos(0, 0, LID_T) * ring
    body.label = "lid"
    return body


if __name__ == "__main__":
    lid()

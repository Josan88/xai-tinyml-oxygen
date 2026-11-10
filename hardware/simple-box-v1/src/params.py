"""Shared dimensions for simple-box-v1. Millimetres, +Z up.

Box frame: exterior bottom face at Z=0, interior floor at Z=WALL,
X along the breadboard length (USB end at -X, finger end at +X),
Y across the breadboard width, origin at the box centre in XY.

v1 changes only the lid and the top of the box walls: the lid is a flat
plate that slides along X in grooves cut into thickened tops of the two
long side walls. It enters from the -X (USB) end and stops against the
+X end wall. A detent near the -X end holds it shut.
"""

# Measured inputs
BOARD_L = 85.0          # breadboard length (power rails removed)
BOARD_W = 35.0          # breadboard width
CONTENT_H = 35.0        # tallest content = interior clear height

# Construction
CLEARANCE = 1.0         # each side of the breadboard
WALL = 2.0              # side walls and floor
MIN_MAT = 1.5           # minimum solid material anywhere in box or lid

INNER_L = BOARD_L + 2 * CLEARANCE   # 87
INNER_W = BOARD_W + 2 * CLEARANCE   # 37
INNER_H = CONTENT_H                 # 35

OUTER_L = INNER_L + 2 * WALL        # 91
OUTER_W = INNER_W + 2 * WALL        # 41, below the thickened top band
WALL_TOP_Z = INNER_H + WALL         # 37, top of the v0 walls = lid underside

# USB-wire hole, left (-X) end wall. PROVISIONAL position.
USB_W = 14.0            # plug is 10 x 10; opening is larger so it can pass
USB_H = 12.0
USB_Z0 = 10.0           # bottom of opening above interior floor (provisional)
USB_Y = 0.0             # centred across the width (provisional)

# Finger opening, right (+X) end wall. PROVISIONAL U-shaped notch:
# semicircular bottom, straight sides, open through the top rim (no bridge).
FINGER_W = 24.0
FINGER_R = FINGER_W / 2           # radius of the rounded bottom (12)
FINGER_Z0 = INNER_H / 2           # bottom edge at interior mid-height (17.5)
FINGER_Y = 0.0                    # centred on the end wall

# Sliding lid
SLIDE_GAP = 0.4         # sliding clearance, each side (Y and Z)
LID_T = 2.0             # lid plate thickness
LID_EDGE_H = 1.5        # height of the vertical lid edge below its 45 deg top bevel
LID_BEVEL = LID_T - LID_EDGE_H    # 0.5

# Detent pocket depth into the back wall of each groove. The top band is
# thickened outward by this much extra so MIN_MAT still stands behind it.
DETENT_DEPTH = 0.5

# Thickened top band of the two long side walls, which carries the groove.
TOP_IN = 0.5                      # inward thickening (opening stays 36 > 35 board)
TOP_OUT = 1.0 + DETENT_DEPTH      # outward thickening (1.5)
TOP_DROP = 1.0          # outer face of the band runs this far below the groove
                        # floor before its 45 deg step, so the pocket corner
                        # keeps MIN_MAT to the step

OPEN_Y = INNER_W / 2 - TOP_IN                       # 18.0, inner face of ledge and ceiling
TOP_OUTER_Y = OUTER_W / 2 + TOP_OUT                 # 22.0, outer face of the top band
GROOVE_BACK_Y = TOP_OUTER_Y - MIN_MAT - DETENT_DEPTH  # 20.0, back of the groove

LID_HALF_W = GROOVE_BACK_Y - SLIDE_GAP    # 19.6
LID_W = 2 * LID_HALF_W                    # 39.2

GROOVE_FLOOR_Z = WALL_TOP_Z               # 37, lid underside rests here
CEILING_Z = GROOVE_FLOOR_Z + LID_T + SLIDE_GAP   # 39.4
BOX_H = CEILING_Z + MIN_MAT               # 40.9, top of the side and +X walls

# Lid length: from the -X outer face to the +X end wall inner face (the stop),
# plus a tab that fills the finger notch through the +X wall thickness.
LID_X0 = -OUTER_L / 2                     # -45.5, flush with the USB end
LID_X1 = INNER_L / 2                      # 43.5, stops on the +X end wall
TAB_HALF_W = FINGER_W / 2 - SLIDE_GAP     # 11.6

# Detent. Each long edge of the lid, near its -X end, is cut free by a slot
# into a flexible arm. A 45 deg bump on the arm's outer edge snaps into a
# matching pocket in the back of the groove just inside the -X entrance.
# The arms flex sideways (in the print plane), so both bumps engage at once
# and the lid's Y play cannot let it slip out.
DETENT_CLEAR = 0.1      # Y gap between seated bump and pocket back
DETENT_BUMP = SLIDE_GAP + DETENT_DEPTH - DETENT_CLEAR   # 0.8 out from the lid edge
DETENT_CREST = 0.6      # flat length along X at the bump crest
DETENT_CLEAR_X = 0.2    # X gap between bump flanks and pocket flanks
DETENT_X = -42.5        # bump and pocket centre with the lid closed
ARM_W = 2.0             # flexible arm width (Y)
ARM_SLOT_W = 1.0        # slot between arm and lid body
ARM_L = 18.0            # slot length from the lid's -X edge (arm root at X=-27.5)

# Finger grip: shallow half-round recess in the lid top at the -X edge.
GRIP_R = 10.0
GRIP_DEPTH = LID_T - MIN_MAT              # 0.5, leaves 1.5 under it

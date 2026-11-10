"""Shared dimensions for simple-box-v0. Millimetres, +Z up.

Box frame: exterior bottom face at Z=0, interior floor at Z=WALL,
X along the breadboard length (USB end at -X, finger end at +X),
Y across the breadboard width, origin at the box centre in XY.
"""

# Measured inputs
BOARD_L = 85.0          # breadboard length (power rails removed)
BOARD_W = 35.0          # breadboard width
CONTENT_H = 35.0        # tallest content = interior clear height

# Construction
CLEARANCE = 1.0         # each side of the breadboard
WALL = 2.0              # side walls and floor
LID_T = 2.0             # lid plate thickness

INNER_L = BOARD_L + 2 * CLEARANCE   # 87
INNER_W = BOARD_W + 2 * CLEARANCE   # 37
INNER_H = CONTENT_H                 # 35

OUTER_L = INNER_L + 2 * WALL        # 91
OUTER_W = INNER_W + 2 * WALL        # 41
OUTER_H = INNER_H + WALL            # 37 (box only, open top)

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

# Lid lip: a thin ring that drops inside the box opening to locate the lid.
LIP_GAP = 0.15          # clearance between lip and box inner wall, each side
LIP_T = 1.5             # lip wall thickness
LIP_DEPTH = 3.0         # how far the lip drops into the box
# The lip is notched at the finger (+X) end only, so it clears the open
# finger notch. The USB (-X) end is solid to seal light. Single orientation.
LIP_NOTCH_W = FINGER_W + 2.0

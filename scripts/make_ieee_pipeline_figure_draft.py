"""Draft system-pipeline figure for the IEEE A4 paper (proposal, not yet in the draft).

Same conventions as make_ieee_paper_results_figs_tinykan.py: one column (235 pt = 3.264 in), every font 8 pt,
300 dpi, no figure title. Every label restates a fact already in the 2026-11-11l draft text; nothing is computed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLUMN_IN = 3.264
FONT_PT = 8
OUT = Path(__file__).resolve().parents[1] / "outputs" / "ieee-paper-figures-proposal" / "fig_pipeline_draft.png"

# (label, filled) top to bottom; filled = runs on the microcontroller for every decision
STAGES = [
    ("MAX30102 red + IR, 50 Hz", False),
    ("10 s window, 5 s stride\nButterworth band-pass 0.5–8 Hz", True),
    ("16 features: 14 IR, ratio of ratios,\nIR perfusion index; standardised", True),
    ("TinyKAN 16 → 12 → 1 (321 parameters)\nP ≥ 0.50 Candidate; 2 in a row Confirmed", True),
    ("Mean perturbation: 1 + 16 forwards\ntop three |Δi| named", True),
    ("Bluetooth LE payload → dashboard", False),
]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT})
    height = 2.72
    fig = plt.figure(figsize=(COLUMN_IN, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, COLUMN_IN)
    ax.set_ylim(0, height)
    ax.axis("off")

    box_w, box_h, gap = 2.42, 0.33, 0.115
    x0 = 0.04
    top = height - 0.04
    centres = []
    for i, (label, on_mcu) in enumerate(STAGES):
        y = top - box_h - i * (box_h + gap)
        ax.add_patch(FancyBboxPatch((x0, y), box_w, box_h, boxstyle="round,pad=0,rounding_size=0.04",
                                    linewidth=0.7, edgecolor="black",
                                    facecolor="#DDE6F0" if on_mcu else "white"))
        ax.text(x0 + box_w / 2, y + box_h / 2, label, ha="center", va="center", linespacing=1.15)
        centres.append(y)
    for upper, lower in zip(centres, centres[1:]):
        ax.add_patch(FancyArrowPatch((x0 + box_w / 2, upper), (x0 + box_w / 2, lower + box_h),
                                     arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color="black"))

    # bracket: the measured decision block (filled boxes)
    y_hi = centres[1] + box_h
    y_lo = centres[4]
    xb = x0 + box_w + 0.05
    ax.plot([xb, xb + 0.06, xb + 0.06, xb], [y_hi, y_hi, y_lo, y_lo], color="black", linewidth=0.7)
    ax.text(xb + 0.1, (y_hi + y_lo) / 2, "Cortex-M4\n64 MHz\nFP32\n\nmedian\n444.3 ms\nper 5 s\nstride",
            ha="left", va="center", linespacing=1.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white", metadata={"Software": None})
    print(OUT)


if __name__ == "__main__":
    main()

"""One-model IEEE Fig. 1: top-1 attribution share by class for the reported TinyKAN (read-only; nothing is recomputed).

Reads outputs/analysis-lamaakal-style-v1/d_global_attribution.csv, keeps the repo v2 rows (the model the paper
reports as TinyKAN, header kan_model_weights_v2.h; ADR-0006) and draws, at print size for one column of the
IEEE A4 template (235 pt = 3.264 in), with every font at 8 pt:

  fig1_global_top1_share.png  share of windows in which each feature is top-ranked by |delta|, by class, in-sample

Same encoding as the v2 panel of figure_global() in scripts/make_ieee_paper_figures.py, copied here so this
script does not depend on that file. One panel, no title and no model name; the paper caption says what it is.
Explainer: mean-perturbation (training-mean replacement). Fill (solid vs hatched) carries the class, so the
figure reads in greyscale.

    C:\\Python314\\python.exe scripts/make_ieee_paper_fig1_tinykan.py   # refuses if the output directory exists
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/analysis-lamaakal-style-v1/d_global_attribution.csv"
OUTPUT = ROOT / "outputs/ieee-paper-figures-tinykan"
VERSION = "v2"  # repo ID of the reported TinyKAN; never drawn on the figure
COLUMN_IN = 235 / 72  # A4 template column width
HEIGHT_IN = 1.7  # measured: six rows, x axis and legend fit at 8 pt without clipping
FONT_PT = 8
BLUE, ORANGE = "#2a78d6", "#eb6834"  # same palette as analyze_lamaakal_style.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
TOP_K = 6

SHORT = {
    "mean": "mean", "std": "std", "minimum": "minimum", "maximum": "maximum", "value_range": "value range",
    "rms": "rms", "skewness": "skewness", "kurtosis": "kurtosis", "energy": "energy",
    "derivative_std": "derivative std", "dominant_frequency_hz": "dominant frequency",
    "bandpower_0_5_2_hz": "bandpower 0.5-2 Hz", "bandpower_2_5_hz": "bandpower 2-5 Hz",
    "bandpower_ratio_0_5_2_to_2_5": "bandpower ratio", "ratio_of_ratios": "ratio of ratios",
    "ir_perfusion_index": "IR perfusion index",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "hatch.linewidth": 0.6, "hatch.color": "white", "savefig.dpi": 300, "figure.dpi": 100,
    })


def save(fig, path):
    fig.savefig(path, dpi=300, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def draw(table, height_in=HEIGHT_IN):
    """Top-1 share by class; the TOP_K features most often top-ranked in either class."""
    block = table[table.version == VERSION].copy()
    block["share_max"] = block[["top1_share_y1", "top1_share_y0"]].max(axis=1)
    block = block.sort_values(["share_max", "feature"], ascending=[False, True])
    top, rest = block.head(TOP_K), block.iloc[TOP_K:]
    shown = {"features": list(top.feature), "rest_max_top1_share": float(rest.share_max.max()),
             "top1_share_y1": [float(v) for v in top.top1_share_y1],
             "top1_share_y0": [float(v) for v in top.top1_share_y0]}
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height_in), layout="constrained")
    y = range(len(top))[::-1]
    # White edges carry the white hatch; fill (solid vs hatched) carries the class in greyscale.
    ax.barh([v + 0.2 for v in y], 100 * top.top1_share_y1, height=0.4, color=BLUE, edgecolor="white",
            linewidth=0.4, label="y = 1 (ScalcO2 < 90)")
    ax.barh([v - 0.2 for v in y], 100 * top.top1_share_y0, height=0.4, color=ORANGE, edgecolor="white",
            linewidth=0.4, hatch="////", label="y = 0")
    ax.set_yticks(list(y), [SHORT[f] for f in top.feature])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of windows top-ranked by |\u0394| (%)", loc="right")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=1.6, columnspacing=1.2)
    return fig, shown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; figures are write-once, use a new directory")
    style()
    fig, shown = draw(pd.read_csv(SOURCE))
    args.output.mkdir(parents=True)
    png = args.output / "fig1_global_top1_share.png"
    save(fig, png)
    manifest = {
        "script": "scripts/make_ieee_paper_fig1_tinykan.py",
        "input_sha256": {"outputs/analysis-lamaakal-style-v1/d_global_attribution.csv": sha256(SOURCE)},
        "print_width_in": COLUMN_IN, "height_in": HEIGHT_IN, "min_font_pt": FONT_PT, "dpi": 300,
        "fig1_global_top1_share": shown,
        "outputs_sha256": {png.name: sha256(png)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nOne-column (235 pt) Fig. 1 for the IEEE A4 draft: the share of windows in which "
        "each feature is top-ranked by |\u0394|, by class, for the reported TinyKAN (repo v2) only. Drawn by "
        "`scripts/make_ieee_paper_fig1_tinykan.py` from `outputs/analysis-lamaakal-style-v1/d_global_attribution.csv`; "
        "nothing is recomputed. Explainer: mean-perturbation (training-mean replacement), in-sample on the shipped "
        "header. It replaces the two-panel Fig. 1 in `outputs/ieee-paper-figures-named` after the 6 Nov 2026 decision "
        "to remove the Baseline from the paper (ADR-0006). `manifest.json` records the input hash and the plotted "
        "values.\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

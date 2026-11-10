"""IEEE-sized paper figures from the Lamaakal-style analysis tables (read-only; nothing is recomputed).

Reads two CSVs from outputs/analysis-lamaakal-style-v1/ and draws, at print size for one column of the
IEEE A4 template (235 pt = 3.264 in), with every font at least 8 pt:

  fig1_global_top1_share.png  share of windows in which each feature is top-ranked by |delta|, by class,
                              for the shipped v1 (reported) and v2 (comparison) headers, in-sample
  fig2_v2_case_studies.png    signed mean-perturbation deltas of the median-probability v2 TP and FP windows

Explainer: mean-perturbation (training-mean replacement). The figures describe the
architecture and the explanation behaviour. Fill (solid vs hatched) carries every
class or direction, so the figures read in greyscale.

    C:\\Python314\\python.exe scripts/make_ieee_paper_figures.py   # refuses if the output directory exists
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/analysis-lamaakal-style-v1"
OUTPUT = ROOT / "outputs/ieee-paper-figures-v1"
COLUMN_IN = 235 / 72  # A4 template column width
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
# Reader-facing names (25 Oct 2026): v1 = "Baseline model", v2 = "Header-aligned model". Repo IDs stay v1/v2.
TITLES = {"v1": "Baseline model: 14 IR inputs", "v2": "Header-aligned model: 16 inputs"}


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


def figure_global(table, path):
    """Top-1 share by class; the TOP_K features most often top-ranked in either class, per model."""
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 2.6), sharex=True, layout="constrained")
    shown = {}
    for ax, version in zip(axes, ("v1", "v2")):
        block = table[table.version == version].copy()
        block["share_max"] = block[["top1_share_y1", "top1_share_y0"]].max(axis=1)
        block = block.sort_values(["share_max", "feature"], ascending=[False, True])
        top, rest = block.head(TOP_K), block.iloc[TOP_K:]
        shown[version] = {"features": list(top.feature), "rest_max_top1_share": float(rest.share_max.max()),
                          "top1_share_y1": [float(v) for v in top.top1_share_y1],
                          "top1_share_y0": [float(v) for v in top.top1_share_y0]}
        y = range(len(top))[::-1]
        # White edges carry the white hatch; fill (solid vs hatched) carries the class in greyscale.
        ax.barh([v + 0.2 for v in y], 100 * top.top1_share_y1, height=0.4, color=BLUE, edgecolor="white",
                linewidth=0.4, label="y = 1 (ScalcO2 < 90)")
        ax.barh([v - 0.2 for v in y], 100 * top.top1_share_y0, height=0.4, color=ORANGE, edgecolor="white",
                linewidth=0.4, hatch="////", label="y = 0")
        ax.set_yticks(list(y), [SHORT[f] for f in top.feature])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_title(TITLES[version], loc="left", pad=3)
    axes[1].set_xlabel("Share of windows top-ranked by |\u0394| (%)", loc="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=1.6, columnspacing=1.2)
    save(fig, path)
    return shown


def figure_cases(cases, path):
    """Signed deltas of the median-probability v2 TP and FP windows; payload slots marked #1-#3."""
    v2 = cases[cases.version == "v2"]
    features = list(dict.fromkeys(v2.feature))
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.6), sharey=True, layout="constrained")
    meta = {}
    for ax, cell in zip(axes, ("TP", "FP")):
        rows = v2[v2.cell == cell].set_index("feature").loc[features]
        delta = rows.delta.to_numpy()
        nonzero = int((delta != 0).sum())
        probability = float(rows.probability.iloc[0])
        y = list(range(len(features)))[::-1]
        for yy, d, direction in zip(y, delta, rows.direction):
            raised = direction == "toward risk"
            ax.barh(yy, 1e3 * d, height=0.62, color=BLUE if raised else ORANGE,
                    edgecolor="white", linewidth=0.4, hatch=None if raised else "////")
        low = 1e3 * min(delta.min(), 0)
        ax.set_xlim(low * 1.5, -low * 0.45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.axvline(0, color=INK2, linewidth=0.6)
        for yy, rank, in_payload, d in zip(y, rows.abs_rank, rows.in_top3_payload, delta):
            if in_payload:
                if d == 0:
                    ax.text(-low * 0.04, yy, f"#{rank}", va="center", ha="left")
                else:
                    ax.text(1e3 * d - low * 0.03, yy, f"#{rank}", va="center", ha="right")
        ax.set_title(f"{cell}, P = {probability:.3f}\n{nonzero}/{len(features)} \u0394 \u2260 0", pad=3)
        ax.tick_params(axis="y", length=0)
        meta[cell] = {"probability": probability, "y_true": int(rows.y_true.iloc[0]), "nonzero_deltas": nonzero,
                      "payload": list(rows[rows.in_top3_payload].sort_values("abs_rank").index),
                      "payload_delta": [float(v) for v in rows[rows.in_top3_payload].sort_values("abs_rank").delta]}
    axes[0].set_yticks(y, [SHORT[f] for f in features])
    # Only lowered-P (hatched) bars occur in these two windows; the caption says so, so there is no legend.
    fig.supxlabel("\u0394 = P(x) \u2212 P(feature at training mean), \u00d710\u207b\u00b3", fontsize=FONT_PT)
    save(fig, path)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; figures are write-once, use a new directory")
    inputs = {name: SOURCE / name for name in ("d_global_attribution.csv", "d_case_studies.csv")}
    style()
    args.output.mkdir(parents=True)
    shown = figure_global(pd.read_csv(inputs["d_global_attribution.csv"]), args.output / "fig1_global_top1_share.png")
    cases = figure_cases(pd.read_csv(inputs["d_case_studies.csv"]), args.output / "fig2_v2_case_studies.png")
    manifest = {
        "script": "scripts/make_ieee_paper_figures.py",
        "inputs_sha256": {f"outputs/analysis-lamaakal-style-v1/{k}": sha256(v) for k, v in inputs.items()},
        "print_width_in": COLUMN_IN, "min_font_pt": FONT_PT, "dpi": 300,
        "fig1_global_top1_share": shown, "fig2_v2_case_studies": cases,
        "outputs_sha256": {p.name: sha256(p) for p in sorted(args.output.glob("*.png"))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nOne-column (235 pt) versions of two analysis figures for the IEEE A4 draft, "
        "drawn by `scripts/make_ieee_paper_figures.py` from `outputs/analysis-lamaakal-style-v1/` "
        "(`d_global_attribution.csv`, `d_case_studies.csv`). Nothing is recomputed. Explainer: mean-perturbation "
        "(training-mean replacement), in-sample on the shipped headers. `manifest.json` records input hashes and the plotted selections.\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

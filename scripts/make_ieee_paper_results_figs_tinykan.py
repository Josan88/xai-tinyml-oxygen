"""Three one-column IEEE Results figures for the reported TinyKAN (read-only; nothing is recomputed from a model).

Every plotted value is read from a committed CSV. Rows are the repo v2 model (the model the paper reports as
TinyKAN, header kan_model_weights_v2.h; ADR-0006); the repo id is never drawn. Print size is one column of the
IEEE A4 template (235 pt = 3.264 in), every font 8 pt, 300 dpi, no figure title and no model-version id:

  figR1_fold_spread.png                    per-fold AUC, recall and specificity, TinyKAN and logistic regression,
                                           recording-grouped 5-fold, with the pooled out-of-fold value (RQ1)
  figR2_error_by_reference_distance.png    error rate by distance of the window's reference median ScalcO2 from 90
  figR3_deletion_aopc.png                  deletion AOPC of the firmware |delta| ranking against random order,
                                           training-mean (shipped) and label-0 median references (RQ2)

Style block, palette, sha256 helper, write-once rule, manifest and README follow
scripts/make_ieee_paper_fig1_tinykan.py (copied, not imported). Identity is carried by marker shape or by solid vs
hatched fill as well as colour, so every figure reads in greyscale. The script asserts the paper's printed numbers
before it draws anything and fails loudly if a source disagrees.

    C:\\Python314\\python.exe scripts/make_ieee_paper_results_figs_tinykan.py   # refuses if the output directory exists
    C:\\Python314\\python.exe scripts/make_ieee_paper_results_figs_tinykan.py --compare-with <earlier output dir>
"""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = "outputs/analysis-lamaakal-style-v1"
SRC = {
    "per_fold": f"{ANALYSIS}/c_per_fold.csv",
    "fold_summary": f"{ANALYSIS}/c_fold_summary.csv",
    "pooled": f"{ANALYSIS}/a_baselines.csv",
    "distance": f"{ANALYSIS}/g_error_by_spo2_distance.csv",
    "aopc_shipped": f"{ANALYSIS}/d_aopc_v2_new_run.csv",
    "aopc_label0": "outputs/xai-reference-label0-v1/aopc_deletion.csv",
    "windows": "outputs/openox-v2/window_features.csv",
}
OUTPUT = ROOT / "outputs/ieee-paper-figures-tinykan-results"
VERSION = "v2"  # repo ID of the reported TinyKAN; never drawn on a figure
COLUMN_IN = 235 / 72  # A4 template column width
HEIGHT_IN = {"R1": 2.45, "R2": 2.1, "R3": 2.3}
FONT_PT = 8
BLUE, ORANGE = "#2a78d6", "#eb6834"  # same palette as analyze_lamaakal_style.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
JOIN = "#a9a8a3"  # thin grey line joining the two markers of a row in R3
MINUS = "\u2212"
CUT = 90.0
N_WINDOWS = 48662


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def png_size(path):
    with open(path, "rb") as handle:
        head = handle.read(24)
    return list(struct.unpack(">II", head[16:24]))


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


class Checks:
    """Named assertions; every one is recorded, and any failure stops the run before a file is written."""

    def __init__(self):
        self.items = []

    def __call__(self, name, ok, detail):
        self.items.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise SystemExit(f"CHECK FAILED: {name}: {detail}")


def one(frame, **keys):
    block = frame
    for column, value in keys.items():
        block = block[block[column] == value]
    if len(block) != 1:
        raise SystemExit(f"expected one row for {keys}, found {len(block)}")
    return block.iloc[0]


def value(store, figure, panel, label, number, source, row, column):
    store.append({"figure": figure, "panel": panel, "label": label, "value": float(number), "source": SRC[source],
                  "row": row, "column": column})
    return float(number)


# ----------------------------------------------------------------------------- R1

R1_MODELS = (  # (c_per_fold model, a_baselines learner, label, colour, marker, filled)
    ("TinyKAN", "tinykan", "TinyKAN", BLUE, "o", True),
    ("LogReg", "logreg", "Logistic regression", ORANGE, "s", False),
)
R1_METRICS = (("auc", "AUC", 1.0), ("recall", "Recall (%)", 100.0), ("specificity", "Specificity (%)", 100.0))
FOLD_OFFSET = {1: 0.28, 2: 0.14, 3: 0.0, 4: -0.14, 5: -0.28}  # deterministic, by fold index


def r1_data(per_fold, summary, pooled, check, store):
    folds = per_fold[(per_fold.version == VERSION) & (per_fold.grouping == "recording")]
    data = {}
    for model, learner, label, *_ in R1_MODELS:
        rows = folds[folds.model == model].sort_values("fold")
        check(f"R1 {label}: folds 1-5 present once", list(rows.fold) == [1, 2, 3, 4, 5], f"folds {list(rows.fold)}")
        base = one(pooled, version=VERSION, grouping="recording", learner=learner)
        data[label] = {}
        for metric, _, scale in R1_METRICS:
            fold_values = [value(store, "R1", metric, f"{label} fold {int(r.fold)}", scale * r[metric], "per_fold",
                                 {"version": VERSION, "grouping": "recording", "model": model, "fold": int(r.fold)},
                                 metric) for _, r in rows.iterrows()]
            pooled_value = value(store, "R1", metric, f"{label} pooled out-of-fold", scale * base[metric], "pooled",
                                 {"version": VERSION, "grouping": "recording", "learner": learner}, metric)
            data[label][metric] = {"folds": fold_values, "pooled": pooled_value}
        check(f"R1 {label}: fold windows sum to the pooled n", int(rows.n_windows.sum()) == int(base.n_windows) == N_WINDOWS,
              f"sum {int(rows.n_windows.sum())}, pooled {int(base.n_windows)}")
    # Paper Table II values.
    paper = {"TinyKAN": ("0.908", "89.3", "86.8"), "Logistic regression": ("0.863", "82.1", "87.0")}
    for label, (auc, rec, spec) in paper.items():
        got = (f"{data[label]['auc']['pooled']:.3f}", f"{data[label]['recall']['pooled']:.1f}",
               f"{data[label]['specificity']['pooled']:.1f}")
        check(f"R1 {label}: pooled AUC / recall / specificity = Table II", got == (auc, rec, spec),
              f"got {got}, paper {(auc, rec, spec)}")
    auc = np.array(data["TinyKAN"]["auc"]["folds"])
    mean, sd = auc.mean(), auc.std(ddof=1)
    check("R1 TinyKAN fold AUC mean 0.911, s.d. (ddof=1) 0.019", (f"{mean:.3f}", f"{sd:.3f}") == ("0.911", "0.019"),
          f"mean {mean:.6f}, s.d. {sd:.6f}")
    s = one(summary, version=VERSION, grouping="recording", model="TinyKAN")
    check("R1 TinyKAN fold AUC mean and s.d. equal c_fold_summary.csv", abs(mean - s.auc_mean) < 1e-6 and abs(sd - s.auc_sd) < 1e-6,
          f"csv mean {s.auc_mean}, s.d. {s.auc_sd}")
    rec, spec = data["TinyKAN"]["recall"]["folds"], data["TinyKAN"]["specificity"]["folds"]
    check("R1 TinyKAN meets the gate in every fold (recall >= 80%, specificity >= 50%)",
          min(rec) >= 80.0 and min(spec) >= 50.0, f"min recall {min(rec):.2f}%, min specificity {min(spec):.2f}%")
    below = [{"fold": i + 1, "recall_pct": round(v, 2)} for i, v in enumerate(data["Logistic regression"]["recall"]["folds"])
             if v < 80.0]
    return data, below


def r1_draw(data):
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_IN, HEIGHT_IN["R1"]), layout="constrained")
    rows = {"TinyKAN": 1, "Logistic regression": 0}
    for ax, (metric, xlabel, _) in zip(axes, R1_METRICS):
        values = []
        for _, _, label, colour, marker, filled in R1_MODELS:
            y0 = rows[label]
            folds = data[label][metric]["folds"]
            values += folds + [data[label][metric]["pooled"]]
            # Pooled bar drawn over the fold markers and taller than their spread, so it stays visible on a fold.
            ax.plot([data[label][metric]["pooled"]] * 2, [y0 - 0.4, y0 + 0.4], color=INK, lw=1.1, zorder=4,
                    solid_capstyle="butt")
            for fold, v in enumerate(folds, start=1):
                ax.plot(v, y0 + FOLD_OFFSET[fold], linestyle="none", marker=marker, markersize=3.2, zorder=3,
                        markerfacecolor=colour if filled else "white", markeredgecolor="white" if filled else colour,
                        markeredgewidth=0.4 if filled else 0.9)
        if metric == "recall":
            values.append(80.0)
            ax.axvline(80.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(80.3, 1.0, "gate", ha="left", va="center", color=INK2)  # TinyKAN row is empty left of 87
        lo, hi = min(values), max(values)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([1, 0], ["TinyKAN", "Logistic regression"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel, labelpad=1)
    fold_handle = (Line2D([], [], ls="none", marker="o", markersize=3.2, markerfacecolor=BLUE, markeredgecolor="white",
                          markeredgewidth=0.4),
                   Line2D([], [], ls="none", marker="s", markersize=3.2, markerfacecolor="white", markeredgecolor=ORANGE,
                          markeredgewidth=0.9))
    pooled_handle = Line2D([], [], ls="none", marker="|", markersize=7, markeredgewidth=1.2, color=INK)
    fig.legend([fold_handle, pooled_handle], ["fold (1 to 5)", "pooled out-of-fold"], loc="outside lower center", ncol=2,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=2.0, columnspacing=1.6)
    return fig


# ----------------------------------------------------------------------------- R2

R2_ROWS = (  # top to bottom; (distance_bin in the CSV, label on the image)
    ("[+10, inf)", "+10 or more"), ("[+5, +10)", "+5 to +10"), ("[+2, +5)", "+2 to +5"), ("[+1, +2)", "+1 to +2"),
    ("[-2, -1)", f"{MINUS}1 to {MINUS}2"), ("[-5, -2)", f"{MINUS}2 to {MINUS}5"), ("[-10, -5)", f"{MINUS}5 to {MINUS}10"),
    ("[-inf, -10)", "more than 10 below"),
)
R2_Y = (7.5, 6.5, 5.5, 4.5, 3.0, 2.0, 1.0, 0.0)
R2_CUT_Y = 3.75


def r2_data(distance, windows, check, store):
    block = distance[distance.version == VERSION]
    check("R2 eight v2 bins, matching the drawn rows", sorted(block.distance_bin) == sorted(b for b, _ in R2_ROWS),
          f"bins {list(block.distance_bin)}")
    data = []
    for bin_name, label in R2_ROWS:
        r = one(block, distance_bin=bin_name)
        expected = "FN" if r.bin_high <= 0 else "FP"
        check(f"R2 {bin_name}: error type {expected}, errors / n = error_rate",
              r.error_type == expected and abs(r.errors / r.n - r.error_rate) < 1e-7,  # CSV keeps 8 significant digits
              f"{r.error_type}, {int(r.errors)} / {int(r.n)} vs {r.error_rate}")
        key = {"version": VERSION, "distance_bin": bin_name}
        data.append({"bin": bin_name, "label": label, "type": r.error_type, "n": int(r.n), "errors": int(r.errors),
                     "rate_pct": value(store, "R2", "", f"{label}: {r.error_type} rate (%)", 100 * r.error_rate,
                                       "distance", key, "error_rate"),
                     "n_value": value(store, "R2", "", f"{label}: n", r.n, "distance", key, "n")})
    rate = {d["bin"]: d["rate_pct"] / 100 for d in data}
    check("R2 FN rate in [-2, -1) = 0.417 (paper)", f"{rate['[-2, -1)']:.3f}" == "0.417", f"{rate['[-2, -1)']:.6f}")
    check("R2 FN rate in [-inf, -10) = 0.062 (paper)", f"{rate['[-inf, -10)']:.3f}" == "0.062", f"{rate['[-inf, -10)']:.6f}")
    n_sum = sum(d["n"] for d in data)
    check("R2 sum of n over the eight bins = 48662 windows", n_sum == N_WINDOWS == len(windows),
          f"sum {n_sum}, window table rows {len(windows)}")
    # Gap at the cut, verified on the window table the label was made from.
    median = windows["median_spo2"].to_numpy(float)
    label_ok = bool(((median < CUT) == (windows["label"].to_numpy(int) == 1)).all())
    check("R2 window label = (median_spo2 < 90) on every window", label_ok and not np.isnan(median).any(),
          f"{int((windows['label'] == 1).sum())} positives, {int(np.isnan(median).sum())} missing medians")
    bands = {}
    for lo, hi in ((89, 91), (89, 90), (90, 91), (88, 92)):
        bands[f"[{lo}, {hi})"] = int(((median >= lo) & (median < hi)).sum())
        bands[f"[{lo}, {hi}]"] = int(((median >= lo) & (median <= hi)).sum())
    gap = {"band_counts": bands, "max_median_below_90": float(median[median < CUT].max()),
           "min_median_at_or_above_90": float(median[median >= CUT].min()),
           "windows_in_excluded_band_[-1, +1)": int(((median - CUT >= -1) & (median - CUT < 1)).sum())}
    return data, n_sum, gap


def r2_draw(data, gap_empty):
    fig, ax = plt.subplots(figsize=(COLUMN_IN, HEIGHT_IN["R2"]), layout="constrained")
    ax.grid(False)
    for d, y in zip(data, R2_Y):
        fp = d["type"] == "FP"
        ax.barh(y, d["rate_pct"], height=0.68, color=BLUE, edgecolor="white", linewidth=0.4,
                hatch="////" if fp else None)
        ax.text(d["rate_pct"] + 1.0, y, f"{d['rate_pct']:.1f}% (n = {d['n']})", va="center", ha="left")
    ax.axhline(R2_CUT_Y, color=INK2, lw=0.6)
    ticks, labels = list(R2_Y) + [R2_CUT_Y], [d["label"] for d in data] + ["cut (90)" if gap_empty else "90"]
    ax.set_yticks(ticks, labels)
    ax.get_yticklabels()[-1].set_color(INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 8.1)
    ax.set_xlim(0, 100)  # room for "41.7% (n = 1030)" right of the longest bar
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Error rate (%)", loc="left")
    handles = [Patch(facecolor=BLUE, edgecolor="white", linewidth=0.4, hatch="////"),
               Patch(facecolor=BLUE, edgecolor="white", linewidth=0.4)]
    fig.legend(handles, ["false flags (negative windows)", "missed flags (positive windows)"], loc="outside lower center",
               ncol=1, handlelength=1.6)
    return fig


# ----------------------------------------------------------------------------- R3

R3_SUBSETS = (("candidate_p_ge_0_50", "candidates (P \u2265 0.50)"), ("noncandidate_p_lt_0_50", "non-candidates"),
              ("all", "all windows"))
R3_RANKINGS = ("firmware_absolute", "random_mean")
R3_PANELS = (("a", "(a) training-mean reference (shipped)"), ("b", "(b) label-0 median reference"))


def r3_data(shipped, label0, check, store):
    shipped = shipped[shipped.version == VERSION]
    tm = label0[(label0.model == VERSION) & (label0.reference == "training_mean")]
    worst, compared = 0.0, 0
    for _, r in shipped.iterrows():
        other = one(tm, subset=r.subset, ranking=r.ranking)
        worst = max(worst, abs(other.deletion_aopc_window_mean - r.deletion_aopc_window_mean))
        compared += 1
        check(f"R3 windows equal for {r.subset} / {r.ranking}", int(other.windows) == int(r.windows), f"{other.windows} vs {r.windows}")
    check("R3 label-0 file's training_mean rows equal d_aopc_v2_new_run.csv (to 1e-9)", compared == 20 and worst <= 1e-9,
          f"{compared} subset x ranking rows compared, max |difference| {worst:.3e}")
    cross = {"rows_compared": compared, "max_abs_difference": worst, "tolerance": 1e-9, "column": "deletion_aopc_window_mean"}
    data = {"a": [], "b": []}
    for panel, frame, source, keys in (("a", shipped, "aopc_shipped", {"version": VERSION}),
                                       ("b", label0[(label0.model == VERSION) & (label0.reference == "label0_fp32")],
                                        "aopc_label0", {"model": VERSION, "reference": "label0_fp32"})):
        for subset, label in R3_SUBSETS:
            rows = {k: one(frame, subset=subset, ranking=k) for k in R3_RANKINGS}
            n = int(rows["firmware_absolute"].windows)
            check(f"R3 ({panel}) {subset}: one n for both rankings", n == int(rows["random_mean"].windows), f"n {n}")
            pair = [value(store, "R3", panel, f"{label}: {k}", rows[k].deletion_aopc_window_mean, source,
                          {**keys, "subset": subset, "ranking": k}, "deletion_aopc_window_mean") for k in R3_RANKINGS]
            data[panel].append({"subset": subset, "label": f"{label}, n = {n}", "n": n, "firmware": pair[0], "random": pair[1]})
    get = {(p, d["subset"]): (d["firmware"], d["random"]) for p in data for d in data[p]}
    paper = {("a", "candidate_p_ge_0_50", 4): ("-0.0085", "-0.0004"), ("a", "noncandidate_p_lt_0_50", 4): ("-0.7402", "-0.3917"),
             ("a", "all", 4): ("-0.3092", "-0.1612"), ("b", "candidate_p_ge_0_50", 3): ("0.775", "0.414"),
             ("b", "all", 3): ("0.469", "0.249")}
    for (panel, subset, dp), expected in paper.items():
        got = tuple(f"{v:.{dp}f}" for v in get[(panel, subset)])
        check(f"R3 ({panel}) {subset}: firmware / random = paper", got == expected, f"got {got}, paper {expected}")
    check("R3 subset n: 28661 + 20001 = 48662", [d["n"] for d in data["a"]] == [28661, 20001, N_WINDOWS]
          == [d["n"] for d in data["b"]], f"{[d['n'] for d in data['a']]}")
    b_non = get[("b", "noncandidate_p_lt_0_50")]
    return data, cross, {"firmware_absolute": b_non[0], "random_mean": b_non[1]}


def r3_draw(data):
    # Subfigures so each panel label starts at the figure's left edge, above the long row labels.
    fig = plt.figure(figsize=(COLUMN_IN, HEIGHT_IN["R3"]), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1.0, 1.22], hspace=0.0)
    for sub, (panel, title) in zip(subfigs, R3_PANELS):
        ax = sub.subplots()
        sub.suptitle(title, x=0.0, ha="left", fontsize=FONT_PT)
        rows = data[panel]
        ys = [2, 1, 0]
        ax.axvline(0.0, color=INK2, lw=0.6, zorder=1)
        for d, y in zip(rows, ys):
            ax.plot([d["firmware"], d["random"]], [y, y], color=JOIN, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(d["firmware"], y, ls="none", marker="o", markersize=4.4, markerfacecolor=BLUE,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
            ax.plot(d["random"], y, ls="none", marker="D", markersize=5.2, markerfacecolor="none",
                    markeredgecolor=INK2, markeredgewidth=0.9, zorder=4)
        values = [v for d in rows for v in (d["firmware"], d["random"])] + [0.0]
        lo, hi = min(values), max(values)
        pad = 0.07 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(ys, [d["label"] for d in rows])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_xlabel("Deletion AOPC", labelpad=1)
    handles = [Line2D([], [], ls="none", marker="o", markersize=4.4, markerfacecolor=BLUE, markeredgecolor="white",
                      markeredgewidth=0.4),
               Line2D([], [], ls="none", marker="D", markersize=5.2, markerfacecolor="none", markeredgecolor=INK2,
                      markeredgewidth=0.9)]
    subfigs[1].legend(handles, ["firmware ranking", "random order (mean of 5)"], loc="outside lower center", ncol=2,
                      handlelength=1.0, columnspacing=1.4)
    return fig


# ----------------------------------------------------------------------------- main

FIGURES = {"R1": "figR1_fold_spread.png", "R2": "figR2_error_by_reference_distance.png", "R3": "figR3_deletion_aopc.png"}


def fmt_checks(items):
    return "\n".join(f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']} ({c['detail']})" for c in items)


def readme(output, manifest, compare):
    m = manifest
    r2 = m["figures"]["figR2_error_by_reference_distance"]
    gap = r2["gap_at_cut"]
    lr_low = m["figures"]["figR1_fold_spread"]["logistic_regression_folds_below_80_recall"]
    lr_text = ("; ".join(f"fold {f['fold']} recall {f['recall_pct']:.2f}%" for f in lr_low) if lr_low else "none")
    sizes = "\n".join(f"- `{name}`: {f['size_in'][0]:.3f} x {f['size_in'][1]:.2f} in, {f['size_px'][0]} x {f['size_px'][1]} px, "
                      f"SHA-256 `{m['outputs_sha256'][name]}`" for name, f in
                      ((FIGURES[k], m["figures"][Path(FIGURES[k]).stem]) for k in FIGURES))
    if compare:
        det = (f"Determinism: this directory was drawn after an earlier clean run into `{compare['directory']}`; the PNG "
               f"SHA-256s are {'identical' if compare['identical'] else 'NOT identical'}.\n\n" +
               "\n".join(f"- `{n}`: earlier `{compare['earlier'][n]}`, this run `{m['outputs_sha256'][n]}`" for n in FIGURES.values()))
    else:
        det = "Determinism: not compared in this run (no `--compare-with`)."
    sources = "\n".join(f"- `{p}`: SHA-256 `{s['sha256']}`, git blob `{s['git_blob']}` (equals HEAD blob: {s['equals_head_blob']})"
                        for p, s in m["sources"].items())
    return f"""# {output.name}

Three one-column (235 pt = 3.264 in) Results figures for the IEEE A4 draft, for the reported TinyKAN (repo v2,
16 inputs, header `kan_model_weights_v2.h`; ADR-0006) only. Drawn by `scripts/make_ieee_paper_results_figs_tinykan.py`.
Plotting only: every plotted value is read from a committed CSV and nothing is recomputed from a model. Every font is
8 pt, 300 dpi, no figure title and no model-version id on any image. Identity is carried by marker shape or by solid
vs hatched fill as well as colour, so each figure reads in greyscale. `manifest.json` lists every plotted value with
its source file, row keys and column, and the SHA-256 and git blob id of every source.

{sizes}

## figR1_fold_spread.png (RQ1)

Per-fold spread for TinyKAN and logistic regression under recording-grouped 5-fold cross-validation, in three
mini-panels with their own x axes: AUC, recall (%), specificity (%). Markers are the five fold values (TinyKAN filled
circles, logistic regression open squares), offset vertically by fold index (fold 1 highest, fold 5 lowest) so all
five can be counted; the short ink bar is the pooled out-of-fold value. The dashed line at 80 in the recall panel is
the recall gate; the 50% specificity gate is not drawn (far left of the data).

Sources: `{SRC['per_fold']}` rows version == v2, grouping == recording, model in (TinyKAN, LogReg), folds 1-5, columns
auc, recall, specificity; pooled values from `{SRC['pooled']}` rows version == v2, grouping == recording, learner in
(tinykan, logreg), same columns.

Observation (not a check): logistic-regression folds with recall below 80%: {lr_text}. TinyKAN meets the gate in
every fold.

## figR2_error_by_reference_distance.png

Error rate by distance of the window's reference median ScalcO2 from the cut (median - 90, in points). Rows above the
cut line are negative windows (y = 0) and the hatched bar is the false-flag rate FP / n; rows below are positive
windows (y = 1) and the solid bar is the missed-flag rate FN / n. Bins are half-open as in the CSV: "+1 to +2" is
[+1, +2), "-1 to -2" is [-2, -1), "+10 or more" is [+10, inf), "more than 10 below" is [-inf, -10).

Source: `{SRC['distance']}` rows version == v2 (eight bins), columns n, error_rate (errors / n checked).

Gap at the cut, verified on `{SRC['windows']}` (column `median_spo2`, the window's reference median ScalcO2; the
`label` column equals median_spo2 < 90 on every window): {json.dumps(gap['band_counts'])}. Largest median below 90:
{gap['max_median_below_90']:.6f}; smallest median at or above 90: {gap['min_median_at_or_above_90']:.6f}. Windows in
the excluded band [-1, +1) of distance: {gap['windows_in_excluded_band_[-1, +1)']}. So no window has a reference
median in [89, 91], as the analysis doc says, and the separator is labelled "cut (90)".

Sum of n over the eight v2 rows: {r2['n_sum']} (window table: {N_WINDOWS}). They are equal: because the band [89, 91)
is empty, the eight bins cover every window.

## figR3_deletion_aopc.png (RQ2)

Deletion AOPC of the firmware ranking (mean-perturbation |delta|, filled circle) against random order (mean of five
repeats, open diamond, drawn on top and slightly larger so it stays visible where the two nearly coincide), joined by a
thin grey line; vertical line at zero. Panel (a): shipped training-mean reference; panel (b): label-0 median reference.
Rows: candidates (P >= 0.50), non-candidates, all windows, with n read from the CSV. signed_risk and reverse_signed are
not plotted.

Sources: (a) `{SRC['aopc_shipped']}` rows version == v2, rankings firmware_absolute and random_mean, column
deletion_aopc_window_mean; (b) `{SRC['aopc_label0']}` rows model == v2, reference == label0_fp32, same rankings and
column. Panel (b) non-candidates (not printed in the paper): firmware {m['figures']['figR3_deletion_aopc']['panel_b_noncandidates']['firmware_absolute']:.6f},
random {m['figures']['figR3_deletion_aopc']['panel_b_noncandidates']['random_mean']:.6f}.

## Consistency checks run (all must pass or the script exits before writing)

{fmt_checks(m['checks'])}

## Sources

{sources}

## Run

```
C:\\Python314\\python.exe scripts/make_ieee_paper_results_figs_tinykan.py            # refuses if this directory exists
C:\\Python314\\python.exe scripts/make_ieee_paper_results_figs_tinykan.py --compare-with <earlier clean output dir>
```

{det}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="earlier output directory whose PNG SHA-256s must equal this run's (determinism check)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; figures are write-once, use a new directory")
    style()
    check, store = Checks(), []
    read = {k: pd.read_csv(ROOT / p) for k, p in SRC.items() if k != "windows"}
    windows = pd.read_csv(ROOT / SRC["windows"], usecols=["median_spo2", "label"])
    r1, lr_low = r1_data(read["per_fold"], read["fold_summary"], read["pooled"], check, store)
    r2, n_sum, gap = r2_data(read["distance"], windows, check, store)
    r3, cross, b_non = r3_data(read["aopc_shipped"], read["aopc_label0"], check, store)
    gap_empty = gap["band_counts"]["[89, 91]"] == 0
    figs = {"R1": r1_draw(r1), "R2": r2_draw(r2, gap_empty), "R3": r3_draw(r3)}
    args.output.mkdir(parents=True)
    out_sha, figures = {}, {}
    for key, fig in figs.items():
        size_in = [float(v) for v in fig.get_size_inches()]
        png = args.output / FIGURES[key]
        save(fig, png)
        out_sha[png.name] = sha256(png)
        figures[png.stem] = {"size_in": size_in, "size_px": png_size(png), "dpi": 300,
                             "values": [v for v in store if v["figure"] == key]}
    figures["figR1_fold_spread"]["logistic_regression_folds_below_80_recall"] = lr_low
    figures["figR2_error_by_reference_distance"].update({"n_sum": n_sum, "n_expected": N_WINDOWS, "gap_at_cut": gap})
    figures["figR3_deletion_aopc"].update({"training_mean_cross_check": cross, "panel_b_noncandidates": b_non})
    sources = {p: {"sha256": sha256(ROOT / p), "git_blob": git("hash-object", p),
                   "equals_head_blob": git("hash-object", p) == git("rev-parse", f"HEAD:{p}")} for p in SRC.values()}
    compare = None
    if args.compare_with is not None:
        earlier = {n: sha256(args.compare_with / n) for n in FIGURES.values()}
        compare = {"directory": str(args.compare_with), "earlier": earlier, "identical": earlier == out_sha}
    manifest = {
        "script": "scripts/make_ieee_paper_results_figs_tinykan.py",
        "model": "repo v2 (reported as TinyKAN; header kan_model_weights_v2.h; ADR-0006)",
        "print_width_in": COLUMN_IN, "min_font_pt": FONT_PT, "dpi": 300,
        "sources": sources, "checks": check.items, "figures": figures,
        "outputs_sha256": out_sha, "determinism": compare,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(readme(args.output, manifest, compare), encoding="utf-8")
    print(json.dumps({"outputs_sha256": out_sha, "sizes": {k: (v["size_in"], v["size_px"]) for k, v in figures.items()},
                      "checks_passed": len(check.items), "determinism": compare}, indent=2))
    if compare is not None and not compare["identical"]:
        raise SystemExit("determinism check failed: PNG SHA-256s differ from the earlier run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

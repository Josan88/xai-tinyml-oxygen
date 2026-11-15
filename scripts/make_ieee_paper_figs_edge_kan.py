"""IEEE draft Figs. 2-6 for the edge KAN (Option A swap, docs/ieee-draft-edge-kan-swap-proposal.md).

Same style, column width (235 pt), 8 pt fonts and 300 dpi as make_ieee_paper_results_figs_tinykan.py and
make_ieee_paper_figures.py (style block copied). Write-once: outputs/ieee-paper-figures-edge-kan/.

  fig2_fold_spread.png        per-fold and pooled AUC / recall / specificity, KAN and logistic regression
  fig3_error_by_distance.png  out-of-fold error rate by distance of the reference median ScalcO2 from 90
  fig4_global_top1_share.png  in-sample top-1 share by class, edge header, training-mean reference
  fig5_case_studies.png       signed deltas of the median-probability TP and FP windows (in sample)
  fig6_deletion_aopc.png      in-sample deletion AOPC, firmware ranking vs random, both references
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lamaakal_metrics as lm  # noqa: E402

OUT = ROOT / "outputs/ieee-paper-figures-edge-kan"
SRC = {
    "summary": ROOT / "outputs/edge-kan-v1/paper-analyses/summary.json",
    "edge_oof": ROOT / "outputs/edge-kan-v1/predictions_ekan16_recording.csv",
    "logreg_folds": ROOT / "outputs/analysis-lamaakal-style-v1/c_per_fold.csv",
    "logreg_pooled": ROOT / "outputs/analysis-lamaakal-style-v1/a_baselines.csv",
    "features": ROOT / "outputs/openox-v2/window_features.csv",
    "edge_header": ROOT / "firmware/Arduino Nano 33/include/edge_kan_weights_v2.h",
}
COLUMN_IN = 235 / 72
FONT_PT = 8
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
JOIN = "#a9a8a3"
MINUS = "−"
FEATURES = ["mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis", "energy",
            "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz", "bandpower_2_5_hz",
            "bandpower_ratio_0_5_2_to_2_5", "ratio_of_ratios", "ir_perfusion_index"]
SHORT = {"mean": "mean", "std": "std", "minimum": "minimum", "maximum": "maximum", "value_range": "value range",
         "rms": "rms", "skewness": "skewness", "kurtosis": "kurtosis", "energy": "energy",
         "derivative_std": "derivative std", "dominant_frequency_hz": "dominant frequency",
         "bandpower_0_5_2_hz": "bandpower 0.5-2 Hz", "bandpower_2_5_hz": "bandpower 2-5 Hz",
         "bandpower_ratio_0_5_2_to_2_5": "bandpower ratio", "ratio_of_ratios": "ratio of ratios",
         "ir_perfusion_index": "IR perfusion index"}
KAN = "KAN"
LOGREG = "Logistic regression"


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


def edge_model():
    import train_edge_kan as tek
    return tek.EdgeHeaderKAN(SRC["edge_header"])


# ---------------------------------------------------------------- Fig. 2
def fig2(summary, edge_oof, path):
    y = edge_oof["y_true"].to_numpy(int)
    p = edge_oof["y_prob"].to_numpy(float)
    pooled_edge = lm.binary_metrics(y, p, 0.5)
    folds_edge = summary["H3_folds"]["edge_kan"]
    lf = pd.read_csv(SRC["logreg_folds"])
    lf = lf[(lf.version == "v2") & (lf.grouping == "recording") & (lf.model == "LogReg")].sort_values("fold")
    lp = pd.read_csv(SRC["logreg_pooled"])
    lp = lp[(lp.version == "v2") & (lp.grouping == "recording") & (lp.learner == "logreg")].iloc[0]
    assert list(lf.fold) == [1, 2, 3, 4, 5] and len(folds_edge) == 5
    data = {KAN: {m: {"folds": [s * f[m] for f in folds_edge], "pooled": s * pooled_edge[m]}
                  for m, s in (("auc", 1), ("recall", 100), ("specificity", 100))},
            LOGREG: {m: {"folds": [s * v for v in lf[m]], "pooled": s * lp[m]}
                     for m, s in (("auc", 1), ("recall", 100), ("specificity", 100))}}
    assert f"{data[LOGREG]['recall']['pooled']:.1f}" == "82.1" and f"{data[KAN]['auc']['pooled']:.3f}" == "0.923"
    offs = {1: 0.12, 2: -0.12, 3: 0.04, 4: -0.04, 5: 0.0}
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_IN, 2.45), layout="constrained")
    models = ((KAN, 1, BLUE, "o", True), (LOGREG, 0, ORANGE, "s", False))
    for ax, (metric, xlabel) in zip(axes, (("auc", "AUC"), ("recall", "Recall (%)"), ("specificity", "Specificity (%)"))):
        values = []
        for label, y0, colour, marker, filled in models:
            d = data[label][metric]
            values += d["folds"] + [d["pooled"]]
            ax.plot([d["pooled"]] * 2, [y0 - 0.4, y0 + 0.4], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
            for k, v in enumerate(d["folds"], start=1):
                ax.plot(v, y0 + offs[k], ls="none", marker=marker, markersize=3.2, zorder=3,
                        markerfacecolor=colour if filled else "white", markeredgecolor="white" if filled else colour,
                        markeredgewidth=0.4 if filled else 0.9)
        if metric == "recall":
            values.append(80.0)
            ax.axvline(80.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(80.3, 0.55, "gate", ha="left", va="center", color=INK2)
        lo, hi = min(values), max(values)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([1, 0], [KAN, LOGREG])
        ax.tick_params(axis="y", length=0)
        if metric != "auc":
            ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(xlabel, labelpad=1)
    fold_handle = (Line2D([], [], ls="none", marker="o", markersize=3.2, markerfacecolor=BLUE, markeredgecolor="white",
                          markeredgewidth=0.4),
                   Line2D([], [], ls="none", marker="s", markersize=3.2, markerfacecolor="white", markeredgecolor=ORANGE,
                          markeredgewidth=0.9))
    pooled_handle = Line2D([], [], ls="none", marker="|", markersize=7, markeredgewidth=1.2, color=INK)
    fig.legend([fold_handle, pooled_handle], ["fold (1 to 5)", "pooled out-of-fold"], loc="outside lower center", ncol=2,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=2.0, columnspacing=1.6)
    save(fig, path)
    return {k: {m: v for m, v in d.items()} for k, d in data.items()}


# ---------------------------------------------------------------- Fig. 3
ROWS3 = (("[+10, inf)", "+10 or more"), ("[+5, +10)", "+5 to +10"), ("[+2, +5)", "+2 to +5"), ("[+1, +2)", "+1 to +2"),
         ("[-2, -1)", f"{MINUS}1 to {MINUS}2"), ("[-5, -2)", f"{MINUS}2 to {MINUS}5"),
         ("[-10, -5)", f"{MINUS}5 to {MINUS}10"), ("[-inf, -10)", "more than 10 below"))
Y3 = (7.5, 6.5, 5.5, 4.5, 3.0, 2.0, 1.0, 0.0)


def fig3(summary, path):
    bins = {b["bin"]: b for b in summary["H5_errors"]["edge_kan"]["distance_bins"]}
    assert sum(b["n"] for b in bins.values()) == 48662
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.1), layout="constrained")
    ax.grid(False)
    shown = []
    for (key, label), y in zip(ROWS3, Y3):
        b = bins[key]
        rate = 100 * b["error_rate"]
        ax.barh(y, rate, height=0.68, color=BLUE, edgecolor="white", linewidth=0.4,
                hatch="////" if b["error_type"] == "FP" else None)
        ax.text(rate + 1.0, y, f"{rate:.1f}% (n = {b['n']})", va="center", ha="left")
        shown.append({"bin": key, "rate_pct": rate, "n": b["n"], "type": b["error_type"]})
    ax.axhline(3.75, color=INK2, lw=0.6)
    ax.set_yticks(list(Y3) + [3.75], [l for _, l in ROWS3] + ["cut (90)"])
    ax.get_yticklabels()[-1].set_color(INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 8.1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Error rate (%)", loc="left")
    handles = [Patch(facecolor=BLUE, edgecolor="white", linewidth=0.4, hatch="////"),
               Patch(facecolor=BLUE, edgecolor="white", linewidth=0.4)]
    fig.legend(handles, ["false flags (negative windows)", "missed flags (positive windows)"], loc="outside lower center",
               ncol=1, handlelength=1.6)
    save(fig, path)
    return shown


# ---------------------------------------------------------------- Figs. 4 and 5 (in sample, edge header)
def explain():
    v2 = pd.read_csv(SRC["features"])
    model = edge_model()
    z = model.standardize(v2.loc[:, FEATURES].to_numpy(np.float32))
    p, a = model.mean_perturbation(z)
    return v2, p, a


def fig4(v2, p, a, summary, path, top_k=6):
    y = v2["label"].to_numpy(int)
    top1 = lm.firmware_abs_order(a)[:, 0]
    share = {f: (float(np.mean(top1[y == 1] == i)), float(np.mean(top1[y == 0] == i))) for i, f in enumerate(FEATURES)}
    want = summary["H9_in_sample_xai"]["top1_share_y1"]["ratio_of_ratios"]
    assert abs(share["ratio_of_ratios"][0] - want) < 1e-12
    order = sorted(FEATURES, key=lambda f: (-max(share[f]), f))
    top, rest = order[:top_k], order[top_k:]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.70), layout="constrained")
    ys = list(range(len(top)))[::-1]
    ax.barh([v + 0.2 for v in ys], [100 * share[f][0] for f in top], height=0.4, color=BLUE, edgecolor="white",
            linewidth=0.4, label="y = 1 (ScalcO2 < 90)")
    ax.barh([v - 0.2 for v in ys], [100 * share[f][1] for f in top], height=0.4, color=ORANGE, edgecolor="white",
            linewidth=0.4, hatch="////", label="y = 0")
    ax.set_yticks(ys, [SHORT[f] for f in top])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of windows top-ranked by |Δ| (%)", loc="right")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=1.6, columnspacing=1.2)
    save(fig, path)
    return {"shown": {f: share[f] for f in top}, "rest_max_share": max(max(share[f]) for f in rest)}


def fig5(p, a, summary, path):
    cases = summary["H9_in_sample_xai"]["case_studies"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.60), sharey=True, layout="constrained")
    meta = {}
    span = max(abs(a[cases[c]["row_index"]]).max() for c in ("TP", "FP")) * 1e3
    for ax, cell in zip(axes, ("TP", "FP")):
        idx = cases[cell]["row_index"]
        delta = a[idx]
        assert abs(p[idx] - cases[cell]["probability"]) < 1e-12
        order = lm.firmware_abs_order(a[idx:idx + 1])[0]
        rank = {int(f): r + 1 for r, f in enumerate(order)}
        ys = list(range(len(FEATURES)))[::-1]
        for yy, d in zip(ys, delta):
            raised = d > 0
            ax.barh(yy, 1e3 * d, height=0.62, color=BLUE if raised else ORANGE, edgecolor="white", linewidth=0.4,
                    hatch=None if raised else "////")
        ax.axvline(0, color=INK2, linewidth=0.6)
        ax.set_xlim(-1.3 * span, 1.3 * span)
        ax.set_xticks([-75, 0, 75])
        for i, (yy, d) in enumerate(zip(ys, delta)):
            if rank[i] <= 3:
                x = 1e3 * d
                ax.text(x + (0.04 if d >= 0 else -0.04) * span, yy, f"#{rank[i]}", va="center",
                        ha="left" if d >= 0 else "right")
        nonzero = int(np.sum(np.abs(delta) > 1e-12))
        ax.set_title(f"{cell}, P = {p[idx]:.3f}\n{nonzero}/16 Δ ≠ 0", pad=3)
        ax.tick_params(axis="y", length=0)
        meta[cell] = {"row_index": idx, "probability": float(p[idx]), "top3": [FEATURES[i] for i in order[:3]],
                      "top3_delta": [float(delta[i]) for i in order[:3]], "nonzero": nonzero}
    axes[0].set_yticks(ys, [SHORT[f] for f in FEATURES])
    fig.supxlabel("Δ = P(x) − P(feature at training mean), ×10⁻³", fontsize=FONT_PT)
    handles = [Patch(facecolor=BLUE, edgecolor="white", linewidth=0.4),
               Patch(facecolor=ORANGE, edgecolor="white", linewidth=0.4, hatch="////")]
    fig.legend(handles, ["raised P", "lowered P"], loc="outside upper center", ncol=2, handlelength=1.6)
    save(fig, path)
    return meta


# ---------------------------------------------------------------- Fig. 6
SUB6 = (("candidate_p_ge_0_50", "candidates (P ≥ 0.50)"), ("noncandidate_p_lt_0_50", "non-candidates"),
        ("all", "all windows"))


def fig6(summary, path):
    rows = pd.DataFrame(summary["H10_aopc_gains"])
    rows = rows[rows.analysis == "in_sample_shipped"]
    data = {}
    for panel, ref in (("a", "training_mean"), ("b", "label0_fp32")):
        data[panel] = []
        for sub, label in SUB6:
            r = rows[(rows.reference == ref) & (rows.subset == sub)].iloc[0]
            data[panel].append({"label": f"{label}, n = {int(r.windows)}", "firmware": float(r.firmware_absolute),
                                "random": float(r.random_mean), "n": int(r.windows)})
    assert data["a"][0]["n"] + data["a"][1]["n"] == 48662
    fig = plt.figure(figsize=(COLUMN_IN, 2.3), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1.0, 1.22], hspace=0.0)
    for sub, (panel, title) in zip(subfigs, (("a", "(a) training-mean reference (shipped)"),
                                             ("b", "(b) label-0 median reference"))):
        ax = sub.subplots()
        sub.suptitle(title, x=0.0, ha="left", fontsize=FONT_PT)
        ys = [2, 1, 0]
        ax.axvline(0.0, color=INK2, lw=0.6, zorder=1)
        for d, y in zip(data[panel], ys):
            ax.plot([d["firmware"], d["random"]], [y, y], color=JOIN, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(d["firmware"], y, ls="none", marker="o", markersize=4.4, markerfacecolor=BLUE,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
            ax.plot(d["random"], y, ls="none", marker="D", markersize=5.2, markerfacecolor="none",
                    markeredgecolor=INK2, markeredgewidth=0.9, zorder=4)
        values = [v for d in data[panel] for v in (d["firmware"], d["random"])] + [0.0]
        lo, hi = min(values), max(values)
        pad = 0.07 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(ys, [d["label"] for d in data[panel]])
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
    save(fig, path)
    return data


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    style()
    summary = json.loads(SRC["summary"].read_text())
    edge_oof = pd.read_csv(SRC["edge_oof"]).sort_values("row_index")
    OUT.mkdir(parents=True)
    manifest = {"sources_sha256": {k: sha256(v) for k, v in SRC.items()}, "script_sha256": sha256(__file__)}
    manifest["fig2"] = fig2(summary, edge_oof, OUT / "fig2_fold_spread.png")
    manifest["fig3"] = fig3(summary, OUT / "fig3_error_by_distance.png")
    v2, p, a = explain()
    manifest["fig4"] = fig4(v2, p, a, summary, OUT / "fig4_global_top1_share.png")
    manifest["fig5"] = fig5(p, a, summary, OUT / "fig5_case_studies.png")
    manifest["fig6"] = fig6(summary, OUT / "fig6_deletion_aopc.png")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(json.dumps({k: manifest[k] for k in ("fig4", "fig5")}, indent=1, default=float))


if __name__ == "__main__":
    main()

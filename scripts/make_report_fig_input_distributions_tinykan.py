"""Report figure: input distributions of the reported TinyKAN by out-of-fold outcome (Zhong et al. Fig. 11 style, adapted).

Reads only; nothing is retrained, re-scored or re-attributed. Draws, at full A4 text width (6.3 in), a 4 x 4 grid
with one panel per input of the repo-v2 TinyKAN (16 inputs, header kan_model_weights_v2.h; reader-facing name
"Header-aligned model", ADR-0006). Each panel is a 25-bin histogram of the input for three out-of-fold outcome
groups at the fixed 0.50 operating point, bars normalised to sum to 1 within each group (as in Zhong et al.,
Artif. Intell. Med. 145:102685, 2023, Fig. 11, where the groups are K-means clusters instead):

  true flags      y_true == 1 and y_prob >= 0.50
  false flags     y_true == 0 and y_prob >= 0.50
  true negatives  y_true == 0 and y_prob <  0.50
  (missed flags, y_true == 1 and y_prob < 0.50, are counted but not drawn)

Panels are ordered by v2 mean_abs_delta_all (mean |delta| of Mean-Perturbation attribution), largest first.

x-axis rule per input: transform, clip to the POOLED 1st-99th percentile of the transformed values, scale 0-1, so
the two end bins hold the clipped tails. Transform: log10 for the strictly positive heavy-tailed inputs in LOG10
(marked *), log10(-x) for `minimum` (all values negative; marked †), linear otherwise. KS statistics
(scipy.stats.ks_2samp) are on the raw values.

Sources (read only):
  outputs/openox-v2/predictions_b_v2_tinykan16.csv     recording-grouped 5-fold out-of-fold y_prob
  outputs/openox-v2/window_features.csv                same row order; label and the 16 inputs
  outputs/openox-v2/summary.json                       variant b_v2_tinykan16 tp/fp/tn/fn
  outputs/analysis-lamaakal-style-v1/d_global_attribution.csv   v2 rows: feature, gloss, mean_abs_delta_all

Writes (write-once directory): fig_input_distributions_by_outcome.png / .pdf, ks_by_input.csv,
tail_share_by_group.csv, false_flag_tail_recordings.csv, manifest.json, README.md.

    C:\\Python314\\python.exe scripts/make_report_fig_input_distributions_tinykan.py   # refuses if the output directory exists
    C:\\Python314\\python.exe scripts/make_report_fig_input_distributions_tinykan.py --output DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = "outputs/openox-v2/predictions_b_v2_tinykan16.csv"
FEATURES = "outputs/openox-v2/window_features.csv"
SUMMARY = "outputs/openox-v2/summary.json"
ATTRIBUTION = "outputs/analysis-lamaakal-style-v1/d_global_attribution.csv"
SOURCES = [PREDICTIONS, FEATURES, SUMMARY, ATTRIBUTION]
OUTPUT = ROOT / "outputs/report-figure-input-distributions-tinykan"
PNG, PDF = "fig_input_distributions_by_outcome.png", "fig_input_distributions_by_outcome.pdf"
VERSION, VARIANT = "v2", "b_v2_tinykan16"  # repo IDs; never drawn on the figure
THRESHOLD = 0.50
N_ROWS = 48662
WINDOW_SAMPLES = 500  # energy = N * rms^2 check
WIDTH_IN, HEIGHT_IN = 6.3, 5.8  # A4 text width
FONT_PT, TICK_PT = 7, 6.5
N_BINS = 25
TIE_TOL = 1e-9  # transformed units; "inclusive" tail boundary for windows tied at a percentile
VECTOR_DIGITS, VECTOR_ZERO = 8, 1e-6  # identical-input-vector key

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"  # same palette as analyze_lamaakal_style.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
GROUPS = ["True flags", "False flags", "True negatives"]
STYLE = {  # line style carries identity in greyscale
    "True flags": dict(color=BLUE, linestyle="solid", linewidth=1.0),
    "False flags": dict(color=ORANGE, linestyle=(0, (3.2, 1.6)), linewidth=1.0),
    "True negatives": dict(color=GREEN, linestyle=(0, (1, 1.1)), linewidth=1.2),
}
FILL_ALPHA = 0.10

META = ["recording_id", "patient_id", "start_time_seconds", "end_time_seconds", "sample_rate_hz",
        "median_spo2", "min_spo2", "label", "label_min_spo2"]
LOG10 = ["std", "maximum", "value_range", "rms", "energy", "derivative_std", "bandpower_0_5_2_hz",
         "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5", "ratio_of_ratios", "ir_perfusion_index"]
LOG10_NEG = ["minimum"]
LINEAR = ["kurtosis", "skewness", "mean", "dominant_frequency_hz"]
MARK = {"log10": "*", "log10(-x)": "\u2020", "linear": ""}
SHORT = {  # the three "waveform extremes" rows need distinct names; keep titles one panel wide
    "ratio_of_ratios": "red/infrared pulse ratio",
    "minimum": "lowest point",
    "maximum": "highest point",
    "value_range": "peak-to-trough range",
    "ir_perfusion_index": "infrared pulse strength",
    "bandpower_ratio_0_5_2_to_2_5": "low vs high freq. power",
    "bandpower_0_5_2_hz": "low-frequency power",
    "bandpower_2_5_hz": "higher-frequency power",
}
AMPLITUDE_POWER = ["std", "rms", "energy", "maximum", "minimum", "value_range", "derivative_std",
                   "bandpower_0_5_2_hz", "bandpower_2_5_hz", "ir_perfusion_index"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({
        "font.family": "Arial",  # narrower than DejaVu Sans; titles must fit one panel
        "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": FONT_PT,
        "figure.labelsize": FONT_PT, "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.0, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    })


def transform_of(feature):
    if feature in LOG10:
        return "log10"
    if feature in LOG10_NEG:
        return "log10(-x)"
    assert feature in LINEAR, feature
    return "linear"


def transformed(x, transform, feature):
    if transform == "log10":
        assert (x > 0).all(), f"{feature}: log10 input not strictly positive (min {x.min()})"
        return np.log10(x)
    if transform == "log10(-x)":
        assert (x < 0).all(), f"{feature}: expected all values negative (max {x.max()})"
        return np.log10(-x)
    return x


def load():
    pred = pd.read_csv(ROOT / PREDICTIONS)
    feat = pd.read_csv(ROOT / FEATURES)
    summary = json.loads((ROOT / SUMMARY).read_text(encoding="utf-8"))
    attr = pd.read_csv(ROOT / ATTRIBUTION)
    checks = {}

    # 1. row alignment
    assert len(pred) == len(feat) == N_ROWS, (len(pred), len(feat))
    assert (pred.recording_id.values == feat.recording_id.values).all(), "recording_id differs"
    assert (pred.start_time_seconds.values == feat.start_time_seconds.values).all(), "start_time_seconds differs"
    assert (pred.y_true.values == feat.label.values).all(), "y_true != label"
    checks["1_rows_aligned"] = f"{N_ROWS} rows; recording_id, start_time_seconds, y_true==label equal row by row"

    # 2. outcome counts at 0.50
    y, p = pred.y_true.values, pred.y_prob.values
    flag = p >= THRESHOLD
    masks = {"True flags": (y == 1) & flag, "False flags": (y == 0) & flag, "True negatives": (y == 0) & ~flag}
    missed = (y == 1) & ~flag
    counts = {g: int(m.sum()) for g, m in masks.items()}
    n_missed = int(missed.sum())
    assert counts == {"True flags": 25705, "False flags": 2622, "True negatives": 17245}, counts
    assert n_missed == 3090, n_missed
    variant = next(v for v in summary["variants"] if v["variant"] == VARIANT)
    assert (variant["tp"], variant["fp"], variant["tn"], variant["fn"]) == (
        counts["True flags"], counts["False flags"], counts["True negatives"], n_missed), variant
    assert summary["threshold"] == THRESHOLD, summary["threshold"]
    recall = counts["True flags"] / (counts["True flags"] + n_missed)
    specificity = counts["True negatives"] / (counts["True negatives"] + counts["False flags"])
    assert round(recall, 3) == 0.893 and round(specificity, 3) == 0.868, (recall, specificity)
    checks["2_counts"] = (f"TF {counts['True flags']}, FF {counts['False flags']}, TN {counts['True negatives']}, "
                          f"missed {n_missed} == summary.json {VARIANT} tp/fp/tn/fn; recall {recall:.3f}, "
                          f"specificity {specificity:.3f}")

    # 3. attribution block matches the input columns
    inputs = [c for c in feat.columns if c not in META]
    block = attr[attr.version == VERSION].sort_values("mean_abs_delta_all", ascending=False, kind="stable")
    assert len(block) == 16 and len(inputs) == 16, (len(block), len(inputs))
    assert sorted(block.feature) == sorted(inputs), (sorted(block.feature), sorted(inputs))
    assert sorted(LOG10 + LOG10_NEG + LINEAR) == sorted(inputs)
    assert not feat[inputs].isna().any().any(), "NaN in inputs"
    checks["3_attribution_block"] = "v2 block has 16 features == the 16 input columns of window_features.csv"
    return pred, feat, block, masks, counts, n_missed, recall, specificity, checks


def compute(pred, feat, block, masks, counts):
    y = pred.y_true.values
    ff = masks["False flags"]
    ks_rows, tail_rows, rec_rows, panels = [], [], [], []
    for rank, r in enumerate(block.itertuples(index=False), start=1):
        f = r.feature
        x = feat[f].values.astype(float)
        tr = transform_of(f)
        xt = transformed(x, tr, f)
        lo, hi = np.percentile(xt, [1, 99])
        xs = (np.clip(xt, lo, hi) - lo) / (hi - lo)
        ks = {
            "ks_y1_vs_y0": ks_2samp(x[y == 1], x[y == 0]).statistic,
            "ks_tf_vs_ff": ks_2samp(x[masks["True flags"]], x[ff]).statistic,
            "ks_ff_vs_tn": ks_2samp(x[ff], x[masks["True negatives"]]).statistic,
        }
        ks_rows.append(dict(rank=rank, feature=f, gloss=r.gloss, transform=tr,
                            mean_abs_delta_all=r.mean_abs_delta_all, p1=lo, p99=hi,
                            **{k: float(v) for k, v in ks.items()},
                            n_true_flags=counts["True flags"], n_false_flags=counts["False flags"],
                            n_true_negatives=counts["True negatives"]))
        strict = {"below": xt < lo, "above": xt > hi}
        incl = {"below": xt <= lo + TIE_TOL, "above": xt >= hi - TIE_TOL}
        for g, m in masks.items():
            tail_rows.append(dict(feature=f, group=g, n=int(m.sum()),
                                  below_p1=float(strict["below"][m].mean()),
                                  above_p99=float(strict["above"][m].mean()),
                                  outside=float((strict["below"] | strict["above"])[m].mean()),
                                  at_or_below_p1=float(incl["below"][m].mean()),
                                  at_or_above_p99=float(incl["above"][m].mean()),
                                  at_or_outside=float((incl["below"] | incl["above"])[m].mean())))
        for boundary, sides in (("strict", strict), ("inclusive", incl)):
            out = ff & (sides["below"] | sides["above"])
            sel = pred.loc[out]
            per_rec = sel.groupby("recording_id").size().sort_values(ascending=False, kind="stable")
            row = dict(feature=f, boundary=boundary, ff_windows=int(out.sum()),
                       ff_below=int((ff & sides["below"]).sum()), ff_above=int((ff & sides["above"]).sum()),
                       recordings=int(sel.recording_id.nunique()), patients=int(sel.patient_id.nunique()))
            for k in range(3):
                row[f"top{k + 1}_windows"] = int(per_rec.iloc[k]) if len(per_rec) > k else 0
                row[f"top{k + 1}_recording"] = per_rec.index[k][:12] if len(per_rec) > k else ""
            row["top3_share"] = float(per_rec.iloc[:3].sum() / max(out.sum(), 1))
            rec_rows.append(row)
        panels.append(dict(feature=f, rank=rank, transform=tr, xs=xs, ks=ks,
                           label=SHORT.get(f, r.gloss) + MARK[tr]))
    return pd.DataFrame(ks_rows), pd.DataFrame(tail_rows), pd.DataFrame(rec_rows), panels


def spot_checks(ks, recs):
    k = ks.set_index("feature")
    rr = k.loc["ratio_of_ratios"]
    got = (round(rr.ks_y1_vs_y0, 3), round(rr.ks_tf_vs_ff, 3), round(rr.ks_ff_vs_tn, 3))
    assert got == (0.750, 0.173, 0.880), got
    assert k.ks_tf_vs_ff.idxmax() == "kurtosis" and round(k.ks_tf_vs_ff.max(), 3) == 0.268, k.ks_tf_vs_ff
    e = recs[(recs.feature == "energy") & (recs.boundary == "strict")].iloc[0]
    got_e = (e.ff_windows, e.recordings, e.patients, e.top1_windows, e.top2_windows, e.top3_windows)
    assert got_e == (190, 13, 12, 80, 48, 21), got_e
    return {"4_ks_spot_check": f"ratio_of_ratios KS y1-y0 {got[0]:.3f}, TF-FF {got[1]:.3f}, FF-TN {got[2]:.3f}; "
                               f"max TF-FF {k.ks_tf_vs_ff.max():.3f} (kurtosis)",
            "5_energy_tail_recordings": "strict boundary: 190 false-flag windows, 13 recordings, 12 patients, "
                                        "top-3 80/48/21 (prototype reproduced)"}


def near_duplicates(feat, panels):
    lstd, lrms, leng = (np.log10(feat[c].values) for c in ("std", "rms", "energy"))
    ratio = np.abs(feat["mean"].values) / feat["std"].values
    hist = {p["feature"]: np.histogram(p["xs"], bins=np.linspace(0, 1, N_BINS + 1))[0] for p in panels}
    return {
        "pearson_r_log10_std_vs_log10_rms": float(np.corrcoef(lstd, lrms)[0, 1]),
        "pearson_r_log10_energy_vs_log10_rms": float(np.corrcoef(leng, lrms)[0, 1]),
        "max_abs_mean_over_std": float(ratio.max()),
        "p99_abs_mean_over_std": float(np.percentile(ratio, 99)),
        "median_abs_mean_over_std": float(np.median(ratio)),
        "max_rms_over_std_minus_1": float((feat["rms"].values / feat["std"].values - 1).max()),
        f"max_rel_dev_energy_vs_{WINDOW_SAMPLES}_rms_sq": float(
            np.abs(feat["energy"].values / (WINDOW_SAMPLES * feat["rms"].values ** 2) - 1).max()),
        "pooled_bin_counts_windows_differing_std_vs_rms": int(np.abs(hist["std"] - hist["rms"]).sum() // 2),
        "pooled_bin_counts_windows_differing_energy_vs_rms": int(np.abs(hist["energy"] - hist["rms"]).sum() // 2),
    }


def identical_vectors(feat, pred, masks, inputs):
    """Windows whose whole 16-input vector repeats; these sit as point masses in every panel.

    Key: each input to VECTOR_DIGITS significant digits, |v| < VECTOR_ZERO set to 0 (the filtered window mean is
    float noise of order 1e-11 in repeated segments, and the ratio inputs differ there at about 1e-10 relative).
    """
    vals = feat[inputs].mask(feat[inputs].abs() < VECTOR_ZERO, 0.0)
    key = vals.apply(lambda c: c.map(lambda v: f"{v:.{VECTOR_DIGITS}g}")).agg("|".join, axis=1)
    size = key.map(key.value_counts())
    rep = size >= 2
    blocks = (pd.DataFrame({"key": key[rep], "rec": feat.recording_id[rep], "ff": masks["False flags"][rep],
                            "y": feat.label[rep], "p": pred.y_prob[rep]})
              .groupby("key").agg(n=("rec", "size"), recordings=("rec", "nunique"), ff=("ff", "sum"),
                                  y1=("y", "sum"), p_min=("p", "min"), p_max=("p", "max"),
                                  recording=("rec", "first"))
              .sort_values(["n", "recording"], ascending=[False, True], kind="stable"))
    lrms = np.log10(feat["rms"].values)
    tied = np.abs(lrms - np.percentile(lrms, 99)) <= TIE_TOL
    tied_recs = feat.recording_id[tied].unique()
    return {
        "rms_p99_tie": dict(windows=int(tied.sum()), recordings=[r[:12] for r in tied_recs],
                            false_flags=int((tied & masks["False flags"]).sum()), y1=int(feat.label[tied].sum()),
                            distinct_vectors=int(key[tied].nunique()),
                            y_prob=sorted({round(float(v), 4) for v in pred.y_prob[tied]})),
        "windows_in_repeated_vectors": int(rep.sum()),
        "repeated_vector_groups": int(len(blocks)),
        "groups_spanning_more_than_one_recording": int((blocks.recordings > 1).sum()),
        "groups_with_10_or_more_windows": int((blocks.n >= 10).sum()),
        "windows_in_groups_with_10_or_more": int(blocks.n[blocks.n >= 10].sum()),
        "false_flags_in_repeated_vectors": int(blocks.ff.sum()),
        "largest_groups": [dict(windows=int(b.n), recording=b.recording[:12], false_flags=int(b.ff),
                                y1=int(b.y1), y_prob=round(float(b.p_min), 4),
                                y_prob_constant=bool(b.p_max - b.p_min < 1e-9))
                           for b in blocks.head(6).itertuples()],
    }


def draw(panels, masks, counts):
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH_IN, HEIGHT_IN), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.01, hspace=0.06, wspace=0.02)
    bins = np.linspace(0, 1, N_BINS + 1)
    drawn = {}
    for ax, p in zip(axes.ravel(), panels):
        props = {}
        for g in GROUPS:
            v = p["xs"][masks[g]]
            props[g] = np.histogram(v, bins=bins, weights=np.full(len(v), 1 / len(v)))[0]
            assert abs(props[g].sum() - 1) < 1e-9
        for g in GROUPS:  # fills first, outlines on top
            ax.stairs(props[g], bins, fill=True, color=to_rgba(STYLE[g]["color"], FILL_ALPHA), linewidth=0)
        for g in ["True negatives", "True flags", "False flags"]:
            ax.stairs(props[g], bins, baseline=None, **STYLE[g])
        drawn[p["feature"]] = {g: [round(float(v), 6) for v in props[g]] for g in GROUPS}
        ks = p["ks"]
        ax.set_title(f"({p['rank']}) {p['label']}\nKS TF\u2013FF {ks['ks_tf_vs_ff']:.2f}\u2009\u00b7\u2009"
                     f"FF\u2013TN {ks['ks_ff_vs_tn']:.2f}", pad=2.5, linespacing=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], min_n_ticks=2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="x", which="both", grid_alpha=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Proportion")
    fig.supxlabel("Input value, clipped to the pooled 1st\u201399th percentile and scaled 0\u20131 "
                  "(end bins hold the clipped tails)\n* log10 before clipping    "
                  "\u2020 log10 of |value| before clipping", linespacing=1.3)
    handles = [Patch(facecolor=to_rgba(STYLE[g]["color"], 0.18), edgecolor=STYLE[g]["color"],
                     linestyle=STYLE[g]["linestyle"], linewidth=STYLE[g]["linewidth"],
                     label=f"{g} (n={counts[g]:,})") for g in GROUPS]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=2.6, columnspacing=2.0)
    return fig, axes, drawn


def check_layout(fig, axes):
    """Titles must stay inside their panel's width; no two panels' decorations may overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    worst = 0.0
    for ax in axes.ravel():
        t = ax.title.get_window_extent(r)
        a = ax.get_window_extent(r)
        worst = max(worst, t.width / a.width)
        assert t.x0 >= a.x0 - 0.5 and t.x1 <= a.x1 + 0.5, (ax.get_title(), t, a)
    boxes = [ax.get_tightbbox(r) for ax in axes.ravel()]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ov = boxes[i].intersection(boxes[i], boxes[j])
            assert ov is None or ov.width < 1 or ov.height < 1, (i, j, ov)
    fb = fig.bbox
    for b in boxes:
        assert b.x0 >= fb.x0 and b.x1 <= fb.x1 and b.y0 >= fb.y0 and b.y1 <= fb.y1, b
    return {"max_title_width_over_panel_width": round(worst, 3), "panel_overlap": "none", "inside_figure": True}


def pct(v):
    return f"{100 * v:.1f}%"


def readme(out, ks, tails, recs, near, ident, counts, n_missed, recall, specificity):
    k = ks.set_index("feature")
    ff_tn = k.ks_ff_vs_tn.sort_values(ascending=False)
    assert ff_tn.index[0] == "ratio_of_ratios" and ff_tn.iloc[1] < 0.5 < ff_tn.iloc[0], ff_tn
    second = ff_tn.index[1]
    tf_ff_max, tf_ff_arg = k.ks_tf_vs_ff.max(), k.ks_tf_vs_ff.idxmax()
    rr_y = k.loc["ratio_of_ratios", "ks_y1_vs_y0"]
    assert k.ks_y1_vs_y0.idxmax() == "ratio_of_ratios", k.ks_y1_vs_y0
    assert ident["groups_spanning_more_than_one_recording"] == 0, ident
    gl = lambda f: SHORT.get(f, k.loc[f, "gloss"])  # noqa: E731
    t = tails.set_index(["feature", "group"])
    rs = recs[recs.boundary == "strict"].set_index("feature")
    ri = recs[recs.boundary == "inclusive"].set_index("feature")
    rows = []
    for f in AMPLITUDE_POWER:
        rows.append(f"| {gl(f)} | {pct(t.loc[(f, 'False flags'), 'outside'])} | "
                    f"{pct(t.loc[(f, 'True flags'), 'outside'])} | {pct(t.loc[(f, 'True negatives'), 'outside'])} | "
                    f"{rs.loc[f, 'ff_windows']} | {rs.loc[f, 'recordings']} | {rs.loc[f, 'patients']} | "
                    f"{rs.loc[f, 'top1_windows']}/{rs.loc[f, 'top2_windows']}/{rs.loc[f, 'top3_windows']} "
                    f"({pct(rs.loc[f, 'top3_share'])}) |")
    ff_out = [t.loc[(f, "False flags"), "outside"] for f in AMPLITUDE_POWER]
    other_out = [max(t.loc[(f, "True flags"), "outside"], t.loc[(f, "True negatives"), "outside"])
                 for f in AMPLITUDE_POWER]
    assert all(a > b for a, b in zip(ff_out, other_out)), "false flags not the heaviest-tailed group everywhere"
    top3 = rs.loc[AMPLITUDE_POWER, "top3_share"]
    tie = ident["rms_p99_tie"]
    assert len(tie["recordings"]) == 1 and tie["distinct_vectors"] == 1 and len(tie["y_prob"]) == 1, tie
    e_s, e_i = rs.loc["energy"], ri.loc["energy"]
    ties_ff = int(e_i.ff_above - e_s.ff_above)
    n_all = sum(counts.values()) + n_missed
    text = f"""# {out.name}

Drawn by `scripts/make_report_fig_input_distributions_tinykan.py` (read only; nothing is retrained, re-scored or
re-attributed). For the final report and possibly the slides, not the IEEE paper.

## What the figure is

One small histogram per model input (16 panels) for the Header-aligned model (repo `v2`, 16 inputs,
`kan_model_weights_v2.h`). Each panel compares three groups of windows, split by what the model said
out-of-fold at the fixed 0.50 operating point: windows it correctly flagged as low-oxygen risk (true flags),
windows it flagged that were not low-oxygen risk (false flags), and windows it correctly left unflagged (true
negatives). Each group's bars sum to 1, so groups of very different size can be compared by shape. If the
false-flag outline sits on top of the true-flag outline and away from the true-negative outline, the model saw
those windows as positive-looking on that input. Panels run from the input with the largest mean |Δ| to the
smallest.

The panel order is mean |Δ| of Mean-Perturbation attribution (`mean_abs_delta_all`, v2 rows of
`d_global_attribution.csv`). The IEEE paper's Fig. 1 orders by top-1 share instead, so the two orders differ on
purpose.

Files: `{PNG}` (300 dpi) and `{PDF}` (same figure), `ks_by_input.csv`, `tail_share_by_group.csv`,
`false_flag_tail_recordings.csv`, `manifest.json` (hashes, counts, panel order, transforms, layout checks).

## Draft report caption

Distribution of the 16 inputs of the Header-aligned model for three outcome groups at the fixed 0.50
operating point, using out-of-fold probabilities from recording-grouped 5-fold cross-validation
({n_all:,} windows): true flags (low-oxygen risk, P ≥ 0.50; n = {counts['True flags']:,}), false flags (not
low-oxygen risk, P ≥ 0.50; n = {counts['False flags']:,}) and true negatives (not low-oxygen risk, P < 0.50;
n = {counts['True negatives']:,}). The {n_missed:,} missed flags (low-oxygen risk, P < 0.50) are not drawn.
Each input is log10-transformed where marked (* log10; † log10 of |value|), clipped to the pooled 1st–99th
percentile of all windows and scaled to 0–1, so the two end bins hold the clipped tails; bars are 25 equal bins
normalised to sum to 1 within each group. Panels are ordered by mean |Δ| of Mean-Perturbation attribution,
largest first. KS is the two-sample Kolmogorov–Smirnov statistic between groups, computed on the raw values.

## What it shows

Numbers below come from the CSVs in this directory. Recall at 0.50 is {recall:.3f} and specificity
{specificity:.3f}.

The red/infrared pulse ratio is the only input that separates false flags from true negatives: KS FF–TN
{k.loc['ratio_of_ratios', 'ks_ff_vs_tn']:.3f}, against {ff_tn.iloc[1]:.3f} for the next input ({gl(second)}).
It is also the input that best separates the two labels overall (KS y=1 vs y=0 {rr_y:.3f}). On that input
the false flags sit with the true flags (KS TF–FF {k.loc['ratio_of_ratios', 'ks_tf_vs_ff']:.3f}).

False flags overlap true flags on every input. The largest TF–FF KS over all 16 inputs is
{tf_ff_max:.3f} ({gl(tf_ff_arg)}). No input on its own tells a false flag from a true flag.

In the amplitude and power panels the false flags have spikes in the end bins. A larger share of false-flag
windows falls outside the pooled 1st–99th band than of either other group, on all {len(AMPLITUDE_POWER)} of
these inputs. Those out-of-band false flags come from a few recordings: on these inputs the top three
recordings hold {pct(top3.min())}–{pct(top3.max())} of them. The spikes reflect a handful of recordings
rather than false flags as a whole.

| input | FF outside | TF outside | TN outside | FF windows outside | recordings | patients | top-3 recordings (share) |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

"Outside" is strictly below p1 or above p99 (`tail_share_by_group.csv`, `false_flag_tail_recordings.csv`,
boundary `strict`). The upper-end figure for the amplitude inputs is fragile. The pooled 99th percentile of
rms falls on {tie['windows']} windows from one recording (`{tie['recordings'][0]}`…) that share an identical
16-input vector ({tie['false_flags']} false flags, {tie['y1']} true flags, P = {tie['y_prob'][0]:.3f} for all).
Whether those windows count as "above p99" then depends on float rounding.
Counting windows at the boundary as well (boundary `inclusive`, tolerance {TIE_TOL:g}) raises the energy count
from {e_s.ff_windows} to {e_i.ff_windows} false-flag windows ({ties_ff} more at the top end). In the figure these
windows are drawn in the top end bin under either rule.

Across the dataset, {ident['windows_in_repeated_vectors']:,} windows repeat an identical 16-input vector, in
{ident['repeated_vector_groups']} groups ({ident['groups_with_10_or_more_windows']} groups of 10 or more holding
{ident['windows_in_groups_with_10_or_more']:,} windows). Every group lies inside one recording, and the 16 inputs repeat to
{VECTOR_DIGITS} significant digits along each stretch. The raw waveforms were not opened for this figure, so
whether the cause is a held sensor value or an export artefact is not established here. The model gives every window in a
group the same P while the label can change along the stretch. {ident['false_flags_in_repeated_vectors']} of the
{counts['False flags']:,} false flags ({pct(ident['false_flags_in_repeated_vectors'] / counts['False flags'])})
sit in these groups, {tie['false_flags']} of them in the one block above. They appear as point masses in every
panel. Whether to keep such stretches is a data-quality question for the dataset; this figure does not change
them.

The std, rms and energy panels look the same, and the numbers confirm it. The filtered window mean is
small next to the spread: |mean|/std is at most {near['max_abs_mean_over_std']:.3f} (median
{near['median_abs_mean_over_std']:.4f}), so rms exceeds std by at most
{pct(near['max_rms_over_std_minus_1'])}. Pearson r is {near['pearson_r_log10_std_vs_log10_rms']:.7f} between
log10(std) and log10(rms), and 1 − {1 - near['pearson_r_log10_energy_vs_log10_rms']:.0e} between log10(energy)
and log10(rms); energy equals {WINDOW_SAMPLES}·rms² to a relative error of
{near[f'max_rel_dev_energy_vs_{WINDOW_SAMPLES}_rms_sq']:.1e}. After the clip and scale, the std and rms panels
differ in the bin of {near['pooled_bin_counts_windows_differing_std_vs_rms']} of {n_all:,} windows, and the
energy and rms panels in {near['pooled_bin_counts_windows_differing_energy_vs_rms']}. The three inputs carry
one piece of information.

## Limits

Counts are windows. Windows overlap by 50% and many come from the same recording, so they are correlated, and
the n in the legend overstates the independent evidence. The figure describes one model at one operating
point. It is descriptive: it shows where each outcome group sits on each input, but it does not say which input
caused a decision. It is not SHAP and not an explanation method. The panel ranking is mean |Δ| of
Mean-Perturbation attribution, computed in-sample on the shipped header, while the groups use out-of-fold
probabilities.

## Relation to Zhong et al. Fig. 11

Zhong et al. (Artif. Intell. Med. 145:102685, 2023) draw, for each top SHAP feature, histograms of min-max
scaled values for two K-means clusters, normalised within each cluster and ordered by global importance. This
figure keeps the layout and the within-group normalisation but uses model outcome groups instead of clusters,
and a log transform plus a pooled 1st–99th percentile clip instead of plain min-max, because min-max on these
heavy-tailed inputs puts almost every window in one bin.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; figures are write-once, use a new directory")
    pred, feat, block, masks, counts, n_missed, recall, specificity, checks = load()
    ks, tails, recs, panels = compute(pred, feat, block, masks, counts)
    checks.update(spot_checks(ks, recs))
    near = near_duplicates(feat, panels)
    ident = identical_vectors(feat, pred, masks, list(block.feature))
    style()
    fig, axes, drawn = draw(panels, masks, counts)
    layout = check_layout(fig, axes)

    out = args.output
    out.mkdir(parents=True)
    fig.savefig(out / PNG, dpi=300, facecolor="white", metadata={"Software": None})
    fig.savefig(out / PDF, facecolor="white", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    ks.to_csv(out / "ks_by_input.csv", index=False)
    tails.to_csv(out / "tail_share_by_group.csv", index=False)
    recs.to_csv(out / "false_flag_tail_recordings.csv", index=False)
    readme(out, ks, tails, recs, near, ident, counts, n_missed, recall, specificity)

    outputs = [PNG, PDF, "ks_by_input.csv", "tail_share_by_group.csv", "false_flag_tail_recordings.csv", "README.md"]
    manifest = {
        "script": "scripts/make_report_fig_input_distributions_tinykan.py",
        "model": f"repo {VERSION} TinyKAN (Header-aligned model), predictions variant {VARIANT}; ids not on the figure",
        "sources_sha256": {s: sha256(ROOT / s) for s in SOURCES},
        "asserts": checks,
        "threshold": THRESHOLD,
        "counts": {**counts, "Missed flags (not drawn)": n_missed},
        "recall": recall, "specificity": specificity,
        "panel_order": list(block.feature),
        "panel_order_rule": "v2 mean_abs_delta_all descending (d_global_attribution.csv)",
        "transform": {f: transform_of(f) for f in block.feature},
        "x_rule": "transform, clip to pooled 1st-99th percentile of transformed values, scale 0-1; 25 equal bins",
        "ks_on": "raw values, scipy.stats.ks_2samp",
        "tail_boundaries": {"strict": "< p1 or > p99", "inclusive": f"<= p1 + {TIE_TOL:g} or >= p99 - {TIE_TOL:g}"},
        "near_duplicates": near,
        "identical_input_vectors": {"key": f"16 inputs to {VECTOR_DIGITS} significant digits, |v| < {VECTOR_ZERO:g} "
                                           "set to 0", **ident},
        "layout": {"width_in": WIDTH_IN, "height_in": HEIGHT_IN, "font_pt": FONT_PT, "tick_pt": TICK_PT,
                   "dpi": 300, **layout},
        "plotted_proportions": drawn,
        "versions": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "numpy": np.__version__},
        "outputs_sha256": {o: sha256(out / o) for o in outputs},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("asserts", "counts", "near_duplicates", "identical_input_vectors",
                                               "layout", "outputs_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

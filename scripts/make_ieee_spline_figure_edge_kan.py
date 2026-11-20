"""Learned splines of the shipped edge KAN: the data behind paper Fig. 6 and the full 204-spline grid.

Reads the shipped weight header (firmware/Arduino Nano 33/include/edge_kan_weights_v2.h) and, unless
--weights-only is given, the window feature table outputs/openox-v2/window_features.csv (regenerate it with
the OpenOximetry export scripts; it is not redistributed). Writes to --out:

  spline_curves.csv            every spline on 241 standardised grid points in [-3, 3]: 192 input-hidden
                               splines phi_{j,i} (with raw-unit input values) and 12 hidden-output splines psi_j
  all_splines_grid.png / .pdf  all 204 splines; solid over the 1st-99th percentile of the data   (needs data)
  label_share_by_ratio_bin.csv windows, patients, share labelled positive and mean P per ratio-of-ratios bin;
                               the bins were chosen after looking at the data, so the table is descriptive
                                                                                                (needs data)

With --fig6 STEM it also re-draws Fig. 6 (STEM_600dpi.png and STEM.pdf): (a) the 12 input splines on ratio of
ratios, (b) the 12 output splines, (c) partial dependence of P on ratio of ratios. The figure in the paper was
rendered from these same arrays before this script existed; the re-draw matches it in content, not pixel for
pixel, so the published fig6_kan_splines_ratio files are left alone by default.

Checks (with data): summing the splines edge by edge reproduces the header's forward pass on every window;
the ratio-of-ratios partial dependence equals outputs/module8-review-v1/A9/partial_dependence.csv when that
file is present; the three highlighted hidden units are the three whose output spline changes most when
ratio of ratios is set to its training mean.

Run:  py -3.14 scripts/make_ieee_spline_figure_edge_kan.py --out outputs/edge-kan-v1/spline-curves
      py -3.14 scripts/make_ieee_spline_figure_edge_kan.py --weights-only --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_edge_kan as tek  # noqa: E402

HEADER = ROOT / "firmware" / "Arduino Nano 33" / "include" / "edge_kan_weights_v2.h"
FEATURES = ROOT / "outputs" / "openox-v2" / "window_features.csv"
A9_PD = ROOT / "outputs" / "module8-review-v1" / "A9" / "partial_dependence.csv"

V16 = ["mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis", "energy",
       "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz", "bandpower_2_5_hz",
       "bandpower_ratio_0_5_2_to_2_5", "ratio_of_ratios", "ir_perfusion_index"]
SHORT = ["IR mean", "IR std", "IR min", "IR max", "IR range", "IR rms", "IR skewness", "IR kurtosis",
         "IR energy", "IR deriv. std", "Dom. frequency", "BP 0.5–2 Hz", "BP 2–5 Hz", "BP ratio",
         "Ratio of ratios", "IR perfusion idx"]
RATIO = V16.index("ratio_of_ratios")
GRID = np.linspace(-3.0, 3.0, 241)
BINS = [0.0, 0.6, 0.8, 0.9, 1.0, 1.2, 1.45, 1.684, 2.5, 3.4, np.inf]

BLUE, ORANGE, PURPLE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#7b4fa6", "#b4b3ad", "#0b0b0b", "#52514e"


def load_model(header: Path) -> tek.EdgeHeaderKAN:
    names = re.search(r"EKAN_FEATURE_NAMES\[[^\]]*\]\s*=\s*\{([^}]*)\}", header.read_text(encoding="utf-8"))
    if names is None or re.findall(r'"([^"]+)"', names.group(1)) != V16:
        raise SystemExit(f"{header}: feature names are not the 16 inputs this script expects")
    model = tek.EdgeHeaderKAN(header)
    assert model.p["EKAN1_THETA"].shape == (12, 16, 8) and model.p["EKAN2_THETA"].shape == (1, 12, 8)
    return model


def input_splines(model, u):
    """phi[g, j, i]: input-hidden spline j <- i at standardised value u[g]."""
    return np.einsum("gk,jik->gji", model.basis(np.asarray(u, float)[:, None])[:, 0, :], model.p["EKAN1_THETA"])


def output_splines(model, h):
    """psi[g, j]: hidden-output spline j at hidden value h[g] (clamped like the firmware)."""
    return model.basis(np.asarray(h, float)[:, None])[:, 0, :] @ model.p["EKAN2_THETA"][0].T


def curves_table(model) -> pd.DataFrame:
    phi, psi = input_splines(model, GRID), output_splines(model, GRID)
    mean, std = model.p["EKAN_SCALER_MEAN"], model.p["EKAN_SCALER_STD"]
    parts = []
    for j in range(12):
        for i in range(16):
            parts.append(pd.DataFrame({"layer": "input_hidden", "hidden_j": j + 1, "input_i": i + 1,
                                       "feature": V16[i], "u_standardised": GRID,
                                       "raw_value": GRID * (std[i] + 1e-10) + mean[i], "value": phi[:, j, i]}))
        parts.append(pd.DataFrame({"layer": "hidden_output", "hidden_j": j + 1, "input_i": pd.NA, "feature": pd.NA,
                                   "u_standardised": GRID, "raw_value": np.nan, "value": psi[:, j]}))
    table = pd.concat(parts, ignore_index=True)
    table["input_i"] = table["input_i"].astype("Int64")
    return table


def forward_parts(model, z):
    """Edge-by-edge forward pass: phi (n, 12, 16), h (n, 12), P (n,)."""
    phi = np.einsum("nik,jik->nji", model.basis(z), model.p["EKAN1_THETA"])
    h = phi.sum(axis=2)
    return phi, h


def prob_from_h(model, h):
    logit = np.einsum("njk,jk->n", model.basis(h), model.p["EKAN2_THETA"][0])
    return 1.0 / (1.0 + np.exp(-logit))


def ratio_partial_dependence(model, raw, phi, h):
    lo, hi = np.percentile(raw[:, RATIO], [1, 99])
    grid = np.linspace(lo, hi, 61)
    zg = (np.float32(grid).astype(np.float64) - model.p["EKAN_SCALER_MEAN"][RATIO]) / (
        model.p["EKAN_SCALER_STD"][RATIO] + 1e-10)
    phig = input_splines(model, zg)[:, :, RATIO]                     # (61, 12)
    base_h = h - phi[:, :, RATIO]
    pdv, q10, q90 = [], [], []
    for g in range(len(grid)):
        p = prob_from_h(model, base_h + phig[g][None, :])
        pdv.append(p.mean()); q10.append(np.quantile(p, 0.1)); q90.append(np.quantile(p, 0.9))
    return grid, np.array(pdv), np.array(q10), np.array(q90)


def highlighted_units(model, phi, h):
    """Hidden units ranked by mean |change in psi_j(h_j)| when ratio of ratios is set to its training mean."""
    phi0 = input_splines(model, np.zeros(1))[0, :, RATIO]           # (12,)
    h0 = h - phi[:, :, RATIO] + phi0[None, :]
    change = np.abs(output_splines_n(model, h) - output_splines_n(model, h0)).mean(axis=0)
    order = np.argsort(-change)
    return order, change


def output_splines_n(model, h):
    return np.einsum("njk,jk->nj", model.basis(h), model.p["EKAN2_THETA"][0])


def label_shares(model, raw, y, patient, p) -> pd.DataFrame:
    ratio = raw[:, RATIO].astype(np.float64)
    bins = pd.cut(ratio, BINS)
    frame = pd.DataFrame({"bin": bins, "y": y, "patient": patient, "p": p})
    out = frame.groupby("bin", observed=False).agg(n=("y", "size"), n_patients=("patient", "nunique"),
                                                  pos=("y", "mean"), meanP=("p", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def draw_grid(model, curves, z, h, path_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(3, 6, figsize=(10.74, 6.3), dpi=200, sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k > 16:
            ax.axis("off")
            continue
        out = k == 16
        lo, hi = None, None
        for j in range(12):
            if out:
                v = curves.query("layer == 'hidden_output' and hidden_j == @j + 1")["value"].to_numpy()
                lo, hi = np.percentile(h[:, j], [1, 99])
            else:
                v = curves.query("layer == 'input_hidden' and hidden_j == @j + 1 and input_i == @k + 1")["value"].to_numpy()
                lo, hi = np.percentile(z[:, k], [1, 99])
            inside = (GRID >= lo) & (GRID <= hi)
            ax.plot(GRID, np.where(inside, np.nan, v), color=colors[j], lw=0.8, ls=":", alpha=0.6)
            ax.plot(GRID, np.where(inside, v, np.nan), color=colors[j], lw=1.6, label=f"$j$={j + 1}")
        ax.axhline(0, color=INK2, lw=0.6, zorder=0)
        ax.set_title("Output $\\psi_j(h_j)$" if out else SHORT[k], fontsize=11, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(-3, 3)
    fig.supylabel("Spline value", fontsize=12)
    fig.suptitle("Every learned spline of the shipped edge KAN: 16×12 input edges and 12 output edges "
                 "(solid: 1st–99th pct of data)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.text(0.5, 0.065, "Standardised input $x_i$ (output panel: hidden value $h_j$), clamped to ±3",
             ha="center", fontsize=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=12, frameon=False,
               fontsize=10, handlelength=1.2, columnspacing=0.8)
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem.with_suffix(ext), metadata={"Creator": None, "Producer": None} if ext == ".pdf"
                    else {"Software": None})
    plt.close(fig)


def draw_fig6(model, raw, z, phi, h, pd_parts, units, path_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, pdv, q10, q90 = pd_parts
    mean, std = model.p["EKAN_SCALER_MEAN"][RATIO], model.p["EKAN_SCALER_STD"][RATIO]
    r1, r95, r99 = np.percentile(raw[:, RATIO], [1, 95, 99])
    r = np.linspace(r1, r99, 241)
    phir = input_splines(model, (np.float32(r).astype(np.float64) - mean) / (std + 1e-10))[:, :, RATIO]
    hgrid = np.linspace(-3.3, 3.3, 265)
    psi = output_splines(model, hgrid)
    tint = {units[0]: BLUE, units[1]: ORANGE, units[2]: PURPLE}
    plt.rcParams.update({"font.size": 9})
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(3.264, 3.55), dpi=600,
                                  gridspec_kw={"height_ratios": [1, 1.05, 1], "hspace": 1.25})
    for j in range(12):
        if j not in tint:
            a.plot(r, phir[:, j], color=GRAY, lw=0.7)
    ends = sorted(tint, key=lambda j: -phir[-1, j])                 # label the line ends top to bottom
    nudge = {j: dy for j, dy in zip(ends, (4, -4, 0))}
    for j, col in tint.items():
        a.plot(r, phir[:, j], color=col, lw=1.4)
        a.annotate(f"$j$={j + 1}", (r[-1], phir[-1, j]), xytext=(2, nudge[j]), textcoords="offset points",
                   color=col, fontsize=8, va="center")
    centres = model.p["EKAN_CENTERS"] * (std + 1e-10) + mean
    shown = centres[(centres >= r1) & (centres <= r99)]
    a.plot(shown, np.full_like(shown, a.get_ylim()[0]), "^", color=INK2, ms=4, clip_on=False)
    a.axvline(r95, color=INK2, lw=0.6, ls="--")
    a.set_title("(a) Input splines on the 12 edges from $R$", fontsize=9, loc="left")
    a.set_ylabel("$\\varphi_{j,R}(R)$")
    for j in range(12):
        lo, hi = np.percentile(h[:, j], [1, 99])
        inside = (hgrid >= lo) & (hgrid <= hi)
        col = tint.get(j, GRAY)
        b.plot(hgrid, np.where(inside, np.nan, psi[:, j]), color=col if j in tint else "#d0cfca", lw=0.6, ls=":")
        b.plot(hgrid, np.where(inside, psi[:, j], np.nan), color=col, lw=1.4 if j in tint else 0.7)
    b.axhline(0, color=INK2, lw=0.5)
    b.set_title("(b) Output splines (solid: 1st–99th pct of $h_j$)", fontsize=9, loc="left")
    b.set_ylabel("$\\psi_j(h_j)$, logit")
    b.set_xlabel("Hidden value $h_j$")
    b.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    c.fill_between(grid, q10, q90, color=BLUE, alpha=0.16, lw=0)
    c.plot(grid, pdv, color=BLUE, lw=1.4)
    ip = int(np.argmax(pdv)); it = ip + int(np.argmin(pdv[ip:]))
    for i, dy in ((ip, -0.16), (it, -0.16)):
        c.plot(grid[i], pdv[i], "o", color=BLUE, mec="white", mew=0.6, ms=4)
        c.annotate(f"{pdv[i]:.2f}", (grid[i], pdv[i] + dy), ha="center", va="top", fontsize=8)
    c.axvline(r95, color=INK2, lw=0.6, ls="--")
    c.text(r95 + 0.05, 0.08, "95th pct of $R$", color=INK2, fontsize=7.5)
    c.set_ylim(0, 1)
    c.set_yticks([0, 1])
    c.set_title("(c) Composed response (partial dependence)", fontsize=9, loc="left")
    c.set_ylabel("Mean P")
    c.set_xlabel("Ratio of ratios $R$")
    for ax in (a, b, c):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a.set_xlim(c.get_xlim())
    fig.savefig(path_stem.parent / (path_stem.name + "_600dpi.png"), bbox_inches="tight", metadata={"Software": None})
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"Creator": None, "Producer": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--header", type=Path, default=HEADER)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--out", type=Path, required=True, help="directory for the CSVs and the spline grid")
    ap.add_argument("--weights-only", action="store_true", help="write spline_curves.csv only; no data needed")
    ap.add_argument("--fig6", type=Path, help="also re-draw Fig. 6 to this path stem (no extension)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.header)
    curves = curves_table(model)
    curves.to_csv(args.out / "spline_curves.csv", index=False)
    print(f"wrote {args.out / 'spline_curves.csv'} ({len(curves)} rows)")
    if args.weights_only:
        return

    v2 = pd.read_csv(args.features)
    raw = v2.loc[:, V16].to_numpy(np.float32)
    y = v2["label"].to_numpy(np.int64)
    patient = v2["patient_id"].astype(str).to_numpy()
    z = model.standardize(raw)
    phi, h = forward_parts(model, z)
    p = prob_from_h(model, h)

    gap = float(np.abs(np.log(p / (1 - p)) - model.logit(z)).max())
    print(f"check: edge-by-edge logit vs header forward pass, {len(p)} windows, max |diff| {gap:.1e}")
    assert gap < 1e-9, "edge-by-edge sum does not reproduce the header forward pass"

    pd_parts = ratio_partial_dependence(model, raw, phi, h)
    grid, pdv = pd_parts[0], pd_parts[1]
    if A9_PD.exists():
        a9 = pd.read_csv(A9_PD).query("feature == 'ratio_of_ratios'")
        err = max(np.abs(a9["raw_value"].to_numpy() - grid).max(), np.abs(a9["pd_mean_p"].to_numpy() - pdv).max(),
                  np.abs(a9["ice_p10"].to_numpy() - pd_parts[2]).max(), np.abs(a9["ice_p90"].to_numpy() - pd_parts[3]).max())
        print(f"check: ratio-of-ratios partial dependence vs {A9_PD.relative_to(ROOT)}, max |diff| {err:.1e}")
        assert err < 1e-9, "partial dependence differs from the A9 analysis"
    ip = int(np.argmax(pdv)); it = ip + int(np.argmin(pdv[ip:]))
    print(f"partial dependence: peak {pdv[ip]:.3f} at R = {grid[ip]:.3f}; trough {pdv[it]:.3f} at R = {grid[it]:.3f}")

    order, change = highlighted_units(model, phi, h)
    print("hidden units by mean |change in psi_j| with R at its training mean: "
          + ", ".join(f"j={j + 1} ({change[j]:.2f})" for j in order[:5]))

    shares = label_shares(model, raw, y, patient, p)
    shares.to_csv(args.out / "label_share_by_ratio_bin.csv", index=False)
    above = raw[:, RATIO] > 1.684
    print(f"wrote {args.out / 'label_share_by_ratio_bin.csv'}; above R 1.684: {int(above.sum())} windows "
          f"({above.mean():.1%}) from {len(set(patient[above]))} patients; smallest bin "
          f"{int(shares['n'].min())} windows, {int(shares['n_patients'].min())} patients")

    draw_grid(model, curves, z, h, args.out / "all_splines_grid")
    print(f"wrote {args.out / 'all_splines_grid'}.png/.pdf")
    if args.fig6:
        args.fig6.parent.mkdir(parents=True, exist_ok=True)
        draw_fig6(model, raw, z, phi, h, pd_parts, list(order[:3]), args.fig6)
        print(f"wrote {args.fig6}_600dpi.png and {args.fig6}.pdf")


if __name__ == "__main__":
    main()

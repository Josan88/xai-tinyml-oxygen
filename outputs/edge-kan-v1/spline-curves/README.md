# Learned splines of the shipped edge KAN

These files back paper Fig. 6 and its caption's line "All 204 splines are plotted in the code repository".

| File | What it is |
|---|---|
| `spline_curves.csv` | Every spline on 241 standardised grid points in [−3, 3]. There are 192 input–hidden splines φ<sub>j,i</sub> (`layer = input_hidden`, with `raw_value` giving the grid in the input's own units) and 12 hidden–output splines ψ<sub>j</sub> (`layer = hidden_output`). Arguments are clamped to ±3, as on the device. |
| `all_splines_grid.png` / `.pdf` | All 204 splines, one panel per input plus one for the output splines. Each line is solid over the 1st–99th percentile of that input (or of h<sub>j</sub>) in the 48,662 training windows. |
| `label_share_by_ratio_bin.csv` | For each ratio-of-ratios bin: windows, patients, the share labelled positive, and mean P. The bins were chosen after looking at the data, so the table is descriptive. |

Fig. 6 itself is `../../ieee-paper-figures-edge-kan/fig6_kan_splines_ratio_600dpi.png` (vector copy `fig6_kan_splines_ratio.pdf`). It was drawn from these same arrays.

## Provenance and checks

- **Weights:** `firmware/Arduino Nano 33/include/edge_kan_weights_v2.h`, the shipped header (16 inputs, 12 hidden units, 8 centred cubic B-splines per edge, 1632 parameters).
- **Script:** `scripts/make_ieee_spline_figure_edge_kan.py`. With `--weights-only` it writes `spline_curves.csv` from the header alone. The other files need `outputs/openox-v2/window_features.csv`, which is regenerated from OpenOximetry by the export scripts and not redistributed.
- **Checks the script runs:**
  - Summing the splines edge by edge reproduces the header's forward pass on all 48,662 windows (max |Δ logit| about 1e-14).
  - The ratio-of-ratios partial dependence equals the post hoc A9 analysis: a peak of 0.937 at R = 1.684 and a trough of 0.479 at R = 3.403.
  - Hidden units 7, 1 and 6 are the three whose output spline changes most when ratio of ratios is set to its training mean, which is why Fig. 6 colours them.
- **Status:** all of this is in sample and post hoc (paper Section V-A).

```
py -3.14 scripts/make_ieee_spline_figure_edge_kan.py --out outputs/edge-kan-v1/spline-curves
py -3.14 scripts/make_ieee_spline_figure_edge_kan.py --weights-only --out <dir>
```

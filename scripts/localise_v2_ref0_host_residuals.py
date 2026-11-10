"""Localise the v2 ref0 host-parity attribution failures that the r2 (training-mean) run did not have.

Host only. For each case, every delta with |C++ - Python| > 5e-6 is reported under both references
(label-0 probe build/ref0/v2_stream_probe_ref0.exe vs the r2 probe build/v3/host/v2_stream_probe.exe),
with the perturbed-pass error split into (a) the float64 model at the C++ FP32 features minus at the
Python features, and (b) the rest (FP32 arithmetic), plus the gradient norm of P over the other
inputs at the perturbed point. Writes residual_localisation.csv into the ref0 parity directory
(refuses to overwrite). Cases: the two decisions failing only under label-0.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import check_v2_host_parity as host, v2_deployed_reference as ref

CASES = [("8da5eec4", 275), ("f4c657d6", 0)]
export = pd.read_csv(ROOT / "outputs/openox-v2/window_features.csv", low_memory=False)
export["recording_id"] = export["recording_id"].astype(str)
eligible = pd.read_csv(ROOT / "outputs/openox_ppg_eligible_encounters.csv").set_index("encounter_id")
rids = {p: next(r for r in dict.fromkeys(export.recording_id) if r.startswith(p)) for p, _ in CASES}
p = ref.load_parameters(); r0 = ref.load_reference("label0")

def probs(scaled):
    return 1 / (1 + np.exp(-ref.logits_from_scaled(np.atleast_2d(scaled), p)))

out = []
for prefix, w in CASES:
    ir, red, _ = host.recording_streams(eligible.loc[rids[prefix]].to_dict())
    ir32, red32 = ir.astype(np.float32), red.astype(np.float32)
    seg = ROOT / "build/ref0/localise.f32"; seg.unlink(missing_ok=True)
    host.write_segments(seg, [{"samples": len(ir32), "ir": ir32, "red": red32}])
    rows = {}
    for tag, probe in (("ref0", "build/ref0/v2_stream_probe_ref0.exe"), ("r2", "build/v3/host/v2_stream_probe.exe")):
        cpp, _ = host.run_cpp(ROOT / probe, seg); rows[tag] = cpp[cpp[:, 1] == w][0]
    seg.unlink()
    py0 = ref.stream(ir32, red32, p, r0); pym = ref.stream(ir32, red32, p)
    fpy = py0["features"][w]; fcpp = rows["ref0"][host.F0:host.F1].astype(np.float32).astype(np.float64)
    zpy = (fpy - p["scaler_mean"]) / (p["scaler_std"] + 1e-10); zcpp = (fcpp - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)
    for tag, reference, py, row in (("label0", r0, py0, rows["ref0"]), ("mean", np.zeros(16), pym, rows["r2"])):
        dcpp = row[host.D0:host.D0 + 16]; dpy = py["deltas"][w]; err = np.abs(dcpp - dpy)
        for i in np.flatnonzero(err > 5e-6):
            a, b = zpy.copy(), zcpp.copy(); a[i] = b[i] = reference[i]
            ppert_py, ppert_pyfeat_cpp = probs(a)[0], probs(b)[0]
            # numerical gradient norm of P at the perturbed point (other inputs)
            g = np.array([(probs(a + 1e-6 * np.eye(16)[k])[0] - probs(a - 1e-6 * np.eye(16)[k])[0]) / 2e-6 for k in range(16)]); g[i] = 0
            u2 = (np.sum(ref._bspline(a @ p["w1"].T + p["b1"], p["centers1"], p["step1"]) * p["theta1"], -1) @ p["w2"].T + p["b2"])[0]
            out.append({"case": f"{prefix} w{w}", "reference": tag, "input": ref.FEATURE_NAMES[i], "abs_delta_error": float(err[i]),
                        "budget": 1e-5, "fails": bool(err[i] > 1e-5), "delta_py": float(dpy[i]), "delta_cpp": float(dcpp[i]),
                        "P_py": float(py["probability"][w]), "P_cpp": float(row[6]),
                        "Ppert_py": float(ppert_py), "Ppert_cpp": float(row[6] - dcpp[i]),
                        "Ppert_float64_at_cpp_features": float(ppert_pyfeat_cpp),
                        "Ppert_error_from_fp32_features": float(ppert_pyfeat_cpp - ppert_py),
                        "Ppert_error_remaining_fp32_arithmetic": float((row[6] - dcpp[i]) - ppert_pyfeat_cpp),
                        "grad_norm_at_perturbed_point": float(np.linalg.norm(g)), "u2_at_perturbed_point": float(u2),
                        "z_input": float(zpy[i]), "reference_value": float(reference[i])})
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
df = pd.DataFrame(out); print(df.T.to_string())
with (ROOT / "outputs/parity/v2-host-ref0-20261030/residual_localisation.csv").open("x", newline="") as stream:
    df.to_csv(stream, index=False)

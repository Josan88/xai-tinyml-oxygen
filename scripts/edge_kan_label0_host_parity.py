"""Gate G3 of docs/edge-kan-label0-board-preregistration.md: the edge-KAN C++ engine with the label-0
reference against the Python float64 label-0 deltas, on all 48,662 windows.

Builds tests/edge_kan_probe.cpp twice with zig c++ exactly as scripts/train_edge_kan.py parity does
(-std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu):
  - flag off: must be byte-identical to outputs/edge-kan-v1/parity/edge_kan_probe_ekan16.exe
    (reported; zig output is deterministic for the same source and flags);
  - -DEKAN_XAI_REF_LABEL0: edge_kan_probe_ekan16_ref0.exe, used as the host expectation on the board.
Both the incremental (shipped) and the full-pass engine outputs are compared with the Python
reference: edge_kan_paper_analyses.EdgeModel, inputs standardized in float64 from FP32 constants,
deltas by analyze_physio_concept.perturb_against(model, z, ref0) with ref0 from
firmware/.../edge_kan_xai_reference.h (parsed, FP32).

Pass (fixed in the pre-registration): probability within 1e-4, each delta within 1e-5, and every
top-3 equal to the stable |delta| order of the Python deltas, except where the 3rd and 4th |delta|
of the Python deltas differ by less than 2e-5. Also checked: the label-0 deltas are not the
training-mean deltas (the reference is in effect).

Writes outputs/edge-kan-label0/host-parity/ (write-once): summary.json, the two probe .exe, SHA256SUMS.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import edge_kan_paper_analyses as eka  # noqa: E402
import analyze_physio_concept as apc  # noqa: E402

INCLUDE = ROOT / "firmware/Arduino Nano 33/include"
PROBE_SRC = ROOT / "tests/edge_kan_probe.cpp"
REF_HEADER = INCLUDE / "edge_kan_xai_reference.h"
ARCHIVED_PROBE = ROOT / "outputs/edge-kan-v1/parity/edge_kan_probe_ekan16.exe"
PREREG = ROOT / "docs/edge-kan-label0-board-preregistration.md"
OUT = ROOT / "outputs/edge-kan-label0/host-parity"
ZIG = ["uvx", "--from", "ziglang==0.16.0", "python", "-m", "ziglang"]
ZIG_FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-mcpu=baseline", "-target", "x86_64-windows-gnu"]
N = 16
TOL_P, TOL_D, TOL_TIE = 1e-4, 1e-5, 2e-5


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_reference(path):
    text = Path(path).read_text(encoding="utf-8")
    body = text[text.index("EKAN_XAI_REF_Z[16] = {"):]
    body = body[:body.index("};")]
    vals = [float(m) for m in re.findall(r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)f", body)]
    if len(vals) != N:
        raise SystemExit(f"parsed {len(vals)} reference values, expected 16")
    return np.array(vals, dtype=np.float32)


def build(exe, define=None):
    cmd = ZIG + ["c++", *ZIG_FLAGS, "-I", str(INCLUDE)]
    if define:
        cmd.append(f"-D{define}")
    cmd += [str(PROBE_SRC), "-o", str(exe)]
    subprocess.run(cmd, check=True, timeout=1200, cwd=ROOT)


def run_probe(exe, raw):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        with src.open("wb") as f:
            np.array([len(raw), N], dtype="<u4").tofile(f)
            raw.astype("<f4").tofile(f)
        subprocess.run([str(exe), str(src), str(dst)], check=True, capture_output=True, timeout=1200)
        out = np.fromfile(dst, dtype="<f4").astype(np.float64).reshape(len(raw), 2 * (1 + N + 3))
    half = 1 + N + 3
    return {"inc": (out[:, 0], out[:, 1:1 + N], out[:, 1 + N:half].astype(int)),
            "full": (out[:, half], out[:, half + 1:half + 1 + N], out[:, half + 1 + N:].astype(int))}


def stable_top3(deltas):
    return np.argsort(-np.abs(deltas), axis=1, kind="stable")[:, :3]


def compare(name, engine, p_ref, d_ref, top_ref, tie_ok):
    p, d, top = engine
    dp, dd = np.abs(p - p_ref), np.abs(d - d_ref)
    top_diff = np.any(top != top_ref, axis=1)
    unexcused = top_diff & ~tie_ok
    res = {"max_abs_dp": float(dp.max()), "max_abs_ddelta": float(dd.max()),
           "rows_dp_over_1e-4": int((dp > TOL_P).sum()), "rows_ddelta_over_1e-5": int((dd > TOL_D).any(axis=1).sum()),
           "top3_differs": int(top_diff.sum()), "top3_differs_not_tie": int(unexcused.sum())}
    res["pass"] = res["rows_dp_over_1e-4"] == 0 and res["rows_ddelta_over_1e-5"] == 0 and res["top3_differs_not_tie"] == 0
    print(name, json.dumps(res))
    return res


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    if shutil.which("uvx") is None:
        raise SystemExit("uvx not found; the probe needs zig 0.16.0 via uv")
    OUT.mkdir(parents=True)
    exe_off, exe_on = OUT / "edge_kan_probe_ekan16_flagoff.exe", OUT / "edge_kan_probe_ekan16_ref0.exe"
    build(exe_off)
    build(exe_on, "EKAN_XAI_REF_LABEL0")
    for pdb in OUT.glob("*.pdb"):
        pdb.unlink()

    table = pd.read_csv(eka.P["features"])
    raw = table.loc[:, eka.V16].to_numpy(np.float32)
    edge = eka.EdgeModel(eka.P["edge_header"])
    z = edge.standardize(raw)
    ref0 = parse_reference(REF_HEADER).astype(np.float64)
    p_py, d_py = apc.perturb_against(edge, z, ref0)
    p_py, d_py = np.asarray(p_py, float), np.asarray(d_py, float)
    _, d_mean = edge.mean_perturbation(z)
    top_py = stable_top3(d_py)
    sorted_abs = -np.sort(-np.abs(d_py), axis=1)
    tie_ok = (sorted_abs[:, 2] - sorted_abs[:, 3]) < TOL_TIE

    on = run_probe(exe_on, raw)
    off = run_probe(exe_off, raw)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "reference_header_sha256": sha256(REF_HEADER), "probe_source_sha256": sha256(PROBE_SRC),
        "table_sha256": sha256(eka.P["features"]), "edge_header_sha256": sha256(eka.P["edge_header"]),
        "zig": " ".join(ZIG), "zig_flags": ZIG_FLAGS, "windows": int(len(raw)),
        "flag_off_probe_identical_to_archived": sha256(exe_off) == sha256(ARCHIVED_PROBE),
        "archived_probe_sha256": sha256(ARCHIVED_PROBE),
        "probe_sha256": {"flag_off": sha256(exe_off), "ref0": sha256(exe_on)},
        "python_reference": "EdgeModel float64; perturb_against(model, z, ref0)",
        "rows_with_3rd_4th_gap_below_2e-5": int(tie_ok.sum()),
        "ref0_incremental_vs_python": compare("ref0 incremental", on["inc"], p_py, d_py, top_py, tie_ok),
        "ref0_fullpass_vs_python": compare("ref0 full-pass", on["full"], p_py, d_py, top_py, tie_ok),
        "ref0_incremental_vs_fullpass_max_abs_ddelta": float(np.abs(on["inc"][1] - on["full"][1]).max()),
        "reference_in_effect": {
            "max_abs_ref0_minus_training_mean_delta_python": float(np.abs(d_py - d_mean).max()),
            "ref0_probe_vs_training_mean_python_rows_ddelta_over_1e-5": int((np.abs(on["inc"][1] - d_mean) > TOL_D).any(axis=1).sum()),
            "flagoff_probe_vs_training_mean_python_max_abs_ddelta": float(np.abs(off["inc"][1] - d_mean).max()),
        },
    }
    summary["g3_pass"] = bool(summary["ref0_incremental_vs_python"]["pass"]
                              and summary["reference_in_effect"]["ref0_probe_vs_training_mean_python_rows_ddelta_over_1e-5"] > 0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as f:
        for p in sorted(OUT.iterdir()):
            if p.name != "SHA256SUMS":
                f.write(f"{sha256(p)}  {p.name}\n")
    print(json.dumps({k: summary[k] for k in ("flag_off_probe_identical_to_archived", "g3_pass", "reference_in_effect")}, indent=2))


if __name__ == "__main__":
    main()

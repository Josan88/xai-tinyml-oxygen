"""C++ beat-feature port (include/physio_features.h) vs the frozen Python extractor. Host only.

Both sides get the same window: the filtered IR/red and the two DC means rounded to FP32 (what
the device holds). Python runs scripts/physio_features.extract on their float64 values; C++ runs
tests/physio_features_probe.cpp on the FP32 arrays. Reported per window: beat_valid, n_beats,
polarity and the integer peak list (exact agreement required), and the five features (absolute
and relative differences; NaN must match NaN).

Height-tie check: a tie between two local maxima of s*x closer than 17 samples is the only case in
which find_peaks' result depends on np.argsort's tie order (not portable; see physio_features.h).
Every window is scanned for such a pair with SciPy's own _local_maxima_1d.

  --synthetic   the cases of tests/test_physio_features.py (no repository data)
  --real N      N windows of outputs/openox-raw-windows-v1/v2_ir_red_windows.npz, chosen by
                SHA-256 order of "p5-features-unit-2026-10-30:<row>" (no labels or features used);
                the npz arrays are indexed once into memory
  --tie-scan    every npz window (no C++): count windows with a height tie inside 17 samples
Writes <out>/<mode>.json (refuses an existing file).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.signal._peak_finding_utils import _local_maxima_1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import physio_features as pf  # noqa: E402

NPZ = ROOT / "outputs/openox-raw-windows-v1/v2_ir_red_windows.npz"
SEED = "p5-features-unit-2026-10-30"
PEAK_SLOTS = 30


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def synthetic_windows():
    """Every window the pytest cases of tests/test_physio_features.py build, plus their labels."""
    import test_physio_features as t

    cases = []
    for hr in (60.0, 90.0, 120.0):
        for ratio in (0.5, 1.0, 1.6):
            cases.append((f"recovers_hr{hr:g}_ratio{ratio:g}", t.window(hr, ac_ir=0.02, ac_red=ratio * 0.02)))
    cases.append(("perfusion_low", t.window(75.0, ac_ir=0.01, ac_red=0.01)))
    cases.append(("perfusion_high", t.window(75.0, ac_ir=0.03, ac_red=0.03)))
    for hr in (60.0, 90.0, 120.0):
        ir, red, dc_ir, dc_red = t.window(hr, ac_red=0.013)
        cases.append((f"polarity_hr{hr:g}_up", (ir, red, dc_ir, dc_red)))
        cases.append((f"polarity_hr{hr:g}_inverted", (-ir, -red, dc_ir, dc_red)))
    cases.append(("intensity_true", t.window(80.0, intensity=True)))
    cases.append(("intensity_false", t.window(80.0, intensity=False)))
    for name, flat in (("flat_zero", np.zeros(500)), ("flat_three", np.full(500, 3.0))):
        cases.append((name, (flat, flat, 1.0e5, 6.0e4)))
    ir, red, _, _ = t.window(70.0)
    cases.append(("dc_ir_zero", (ir, red, 0.0, 6.0e4)))
    cases.append(("dc_red_negative", (ir, red, 1.0e5, -1.0)))
    ir, red, dc_ir, dc_red = t.window(60.0, intensity=False)
    keep = np.zeros(500)
    keep[60:200] = 1.0
    cases.append(("three_pulses", (ir * keep, red * keep, dc_ir, dc_red)))
    rng = np.random.default_rng(20261030)
    for k in range(20):
        noise = signal.sosfiltfilt(t.SOS, rng.standard_normal(2000))[750:1250]
        cases.append((f"noise_{k}", (noise, 0.5 * noise, 1.0e5, 6.0e4)))
    for hr in (60.0, 95.0):
        cases.append((f"extract_many_hr{hr:g}", t.window(hr)))
    return cases


def fp32_window(ir, red, dc_ir, dc_red):
    return (np.asarray(ir, np.float32), np.asarray(red, np.float32), np.float32(dc_ir), np.float32(dc_red))


def run_probe(probe, windows, scratch):
    with scratch.open("wb") as stream:
        for i, (ir, red, dc_ir, dc_red) in enumerate(windows):
            stream.write(np.uint32(i).tobytes() + ir.astype("<f4").tobytes() + red.astype("<f4").tobytes()
                         + np.array([dc_ir, dc_red], dtype="<f4").tobytes())
    out = subprocess.run([str(probe), str(scratch)], capture_output=True, text=True, check=True, timeout=3600)
    scratch.unlink()
    rows = [line.split() for line in out.stdout.splitlines()]
    if len(rows) != len(windows) or any(len(r) != 10 + PEAK_SLOTS for r in rows):
        raise SystemExit("Malformed probe output")
    return rows


def tie_pairs(x, s):
    """Pairs of local maxima of s*x closer than 17 samples with exactly equal height."""
    y = s * np.asarray(x, dtype=np.float64)
    peaks = _local_maxima_1d(np.ascontiguousarray(y))[0]
    count = 0
    for a in range(len(peaks)):
        b = a + 1
        while b < len(peaks) and peaks[b] - peaks[a] < pf.MIN_PEAK_DISTANCE:
            count += int(y[peaks[a]] == y[peaks[b]])
            b += 1
    return count


def compare(names, windows, rows):
    out = []
    for name, (ir, red, dc_ir, dc_red), row in zip(names, windows, rows):
        x64, r64 = ir.astype(np.float64), red.astype(np.float64)
        py = pf.extract(x64, r64, float(dc_ir), float(dc_red))
        usable = bool(np.std(x64) > 0 and np.isfinite(x64).all() and np.isfinite(r64).all() and dc_ir > 0 and dc_red > 0)
        py_peaks = pf.detect_beats(x64)[1].tolist() if usable else []
        cpp_valid, cpp_n, cpp_s, cpp_tie = (int(v) for v in row[1:5])
        cpp_f = np.array([float(v) for v in row[5:10]])
        cpp_peaks = [int(v) for v in row[10:] if int(v) >= 0]
        py_f = np.array([py[n] for n in pf.FEATURES])
        both_nan = np.isnan(py_f) & np.isnan(cpp_f)
        abs_diff = np.where(both_nan, 0.0, np.abs(cpp_f - py_f))
        rel_diff = np.where(both_nan | (py_f == 0), 0.0, abs_diff / np.abs(np.where(py_f == 0, 1.0, py_f)))
        peaks_equal = py_peaks[:PEAK_SLOTS] == cpp_peaks and (len(py_peaks) == cpp_n if usable else cpp_n == 0)
        out.append({"window": name, "python_beat_valid": bool(py["beat_valid"]), "cpp_beat_valid": bool(cpp_valid),
                    "python_n_beats": int(py["n_beats"]), "cpp_n_beats": cpp_n, "python_polarity": int(py["polarity"]),
                    "cpp_polarity": cpp_s, "cpp_distance_tie": bool(cpp_tie),
                    "tie_pairs_within_17": tie_pairs(x64, py["polarity"]) if py["polarity"] else 0,
                    "peaks_equal": bool(peaks_equal),
                    "discrete_equal": bool(peaks_equal and bool(cpp_valid) == py["beat_valid"] and cpp_n == py["n_beats"]
                                           and cpp_s == py["polarity"]),
                    "nan_pattern_equal": bool(np.array_equal(np.isnan(py_f), np.isnan(cpp_f))),
                    "max_abs_diff": dict(zip(pf.FEATURES, abs_diff.tolist())),
                    "max_rel_diff": dict(zip(pf.FEATURES, rel_diff.tolist())),
                    "bit_equal": dict(zip(pf.FEATURES, [bool(b) for b in (both_nan | (cpp_f == py_f))]))})
    return out


def summarize(results):
    feats = pf.FEATURES
    return {"windows": len(results),
            "discrete_equal": sum(r["discrete_equal"] for r in results),
            "peaks_equal": sum(r["peaks_equal"] for r in results),
            "nan_pattern_equal": sum(r["nan_pattern_equal"] for r in results),
            "beat_valid_windows": sum(r["python_beat_valid"] for r in results),
            "cpp_distance_tie_windows": sum(r["cpp_distance_tie"] for r in results),
            "windows_with_tie_pairs_within_17": sum(r["tie_pairs_within_17"] > 0 for r in results),
            "max_abs_diff": {f: max(r["max_abs_diff"][f] for r in results) for f in feats},
            "max_rel_diff": {f: max(r["max_rel_diff"][f] for r in results) for f in feats},
            "bit_equal_windows": {f: sum(r["bit_equal"][f] for r in results) for f in feats},
            "failures": [r["window"] for r in results if not (r["discrete_equal"] and r["nan_pattern_equal"])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--real", type=int, metavar="N")
    mode.add_argument("--tie-scan", action="store_true")
    parser.add_argument("--probe", type=Path, default=ROOT / "build/p5/host/physio_features_probe.exe")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    name = "synthetic" if args.synthetic else ("tie_scan" if args.tie_scan else f"real_{args.real}")
    target = args.out / f"{name}.json"
    if target.exists():
        parser.error(f"{target} exists; refusing to overwrite")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": name, "board_run": False,
              "scope": "Host only. Same FP32 window to both sides; C++ include/physio_features.h vs scripts/physio_features.py."}
    if args.tie_scan:
        with np.load(NPZ) as z:
            ir, red = z["ir"], z["red"]
            dc_ir, dc_red = z["ir_dc_mean"], z["red_dc_mean"]
        ties = []
        for i in range(len(ir)):
            x = ir[i].astype(np.float64)
            if not (np.std(x) > 0 and dc_ir[i] > 0 and dc_red[i] > 0):
                continue
            n = tie_pairs(x, pf.polarity(x))
            if n:
                ties.append({"row_index": i, "pairs": n})
        report.update(windows=int(len(ir)), windows_with_tie_pairs_within_17=len(ties), tie_windows=ties,
                      npz_sha256=sha256(NPZ),
                      note="Scanned on the stored FP32 offline windows (float64 of FP32), as physio_features.csv.gz was computed.")
        print(json.dumps({k: report[k] for k in ("windows", "windows_with_tie_pairs_within_17")}, indent=2))
    else:
        if args.synthetic:
            cases = synthetic_windows()
            names = [c[0] for c in cases]
            windows = [fp32_window(*c[1]) for c in cases]
            report["source"] = "tests/test_physio_features.py window builders, rounded to FP32"
        else:
            with np.load(NPZ) as z:
                total = len(z["row_index"])
                rows = sorted(range(total), key=lambda r: hashlib.sha256(f"{SEED}:{r}".encode()).hexdigest())[:args.real]
                rows = np.array(sorted(rows))
                ir, red = z["ir"][rows], z["red"][rows]
                dc_ir, dc_red = z["ir_dc_mean"][rows], z["red_dc_mean"][rows]
            names = [int(r) for r in rows]
            windows = [fp32_window(ir[k], red[k], dc_ir[k], dc_red[k]) for k in range(len(rows))]
            report.update(source=f"{NPZ.relative_to(ROOT).as_posix()} rows by SHA-256 order of '{SEED}:<row>'",
                          npz_sha256=sha256(NPZ), selection_seed=SEED)
        args.out.mkdir(parents=True, exist_ok=True)
        rows_out = run_probe(args.probe.resolve(), windows, args.out / f"{name}.scratch.bin")
        results = compare(names, windows, rows_out)
        report.update(probe=str(args.probe), probe_sha256=sha256(args.probe),
                      header_sha256=sha256(ROOT / "firmware/Arduino Nano 33/include/physio_features.h"),
                      extractor_sha256=sha256(ROOT / "scripts/physio_features.py"),
                      summary=summarize(results), windows=results)
        print(json.dumps(report["summary"], indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare the two SpO2 est algorithms from SPO2CMP: lines of a -DOXY_SPO2_MAXIM capture (ADR-0010). No serial port.

usage: analyze_spo2_compare.py <capture.txt> [--json out.json] [--md out.md]

Input: a build/v2/cap.py capture of the nano33ble_v2_f1a_spo2 image. One line per computation (1 Hz of
finger contact, the first after 4 s):
  SPO2CMP: n=<k> maxim=<int|-999> maxim_valid=<0|1> rf=<x.x|-999.0> rf_valid=<0|1> rf_corr=<x.xxx|nan> shown=<int|->
Other lines are ignored. A new run (finger re-placed, firmware reset) starts when n does not increase.

Both algorithms go through the same display rule as the tile (include/spo2_maxim.h): a value is accepted
if valid and 70-100, the last value is held through 2 rejected results and dashed on the 3rd, and there is
no value before the first accepted one in a run. RF is rounded half up to an integer first. Maxim's
recomputed display is cross-checked against the firmware's shown= field.

Reports per algorithm: computations, share raw-valid, share shown, median and range of shown values,
median absolute step between consecutive accepted values (consecutive n in one run; held repeats are not
steps, so an algorithm is not made to look steadier by being invalid more often), and the time to the
first shown value (n=1 is 4 s of contact, then 1 s per n). When both show a value at the same
computation: median and maximum absolute difference.

What this may conclude (D10): which algorithm gives a value more often and moves less at rest on this
sensor; not which is more accurate. There is no reference oximeter and oxygen is not varied.
--json/--md refuse to overwrite.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

D10 = "Which algorithm gives a value more often and moves less at rest on this sensor; not which is more accurate."
SHOW_MIN, SHOW_MAX, DASH_AFTER = 70, 100, 3   # include/spo2_maxim.h kShowMin, kShowMax, kDashAfter
FIRST_S, STRIDE_S = 4, 1                      # first computation after 4 s of contact, then every 1 s
LINE = re.compile(r"SPO2CMP: n=(\d+) maxim=(-?\d+) maxim_valid=(-?\d+) rf=(\S+) rf_valid=(-?\d+) "
                  r"rf_corr=(\S+) shown=(\S+)")


def number(text):
    """Float of a printed value; Arduino prints nan/inf/ovf, which become nan here."""
    try:
        return float(text)
    except ValueError:
        return math.nan


def round_half_up(x):
    return math.floor(x + 0.5)


def parse(path):
    """Runs of SPO2CMP rows, in capture order."""
    runs, last_n = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        n = int(m.group(1))
        if last_n is None or n <= last_n:
            runs.append([])
        last_n = n
        runs[-1].append({"n": n, "maxim": int(m.group(2)), "maxim_valid": int(m.group(3)) == 1,
                         "rf": number(m.group(4)), "rf_valid": int(m.group(5)) == 1, "rf_corr": number(m.group(6)),
                         "fw_shown": None if m.group(7) == "-" else int(m.group(7))})
    return runs


def display(values):
    """values: per computation, an int candidate or None (invalid). Returns (shown, accepted) per computation."""
    shown, streak, out = None, 0, []
    for v in values:
        accepted = v is not None and SHOW_MIN <= v <= SHOW_MAX
        if accepted:
            shown, streak = v, 0
        else:
            streak += 1
            if streak >= DASH_AFTER:
                shown = None
        out.append((shown, accepted))
    return out


def candidates(run, algo):
    if algo == "maxim":
        return [r["maxim"] if r["maxim_valid"] else None for r in run]
    return [round_half_up(r["rf"]) if r["rf_valid"] and math.isfinite(r["rf"]) else None for r in run]


def summarise(runs, algo):
    total = valid = 0
    shown_values, steps, first = [], [], []
    for i, run in enumerate(runs, 1):
        cands = candidates(run, algo)
        disp = display(cands)
        total += len(run)
        valid += sum(r["maxim_valid"] if algo == "maxim" else r["rf_valid"] for r in run)
        shown_values += [s for s, _ in disp if s is not None]
        for k in range(1, len(run)):
            if disp[k][1] and disp[k - 1][1] and run[k]["n"] == run[k - 1]["n"] + 1:
                steps.append(abs(cands[k] - cands[k - 1]))
        hit = next((r["n"] for r, (s, _) in zip(run, disp) if s is not None), None)
        first.append({"run": i, "n": hit, "contact_s": None if hit is None else FIRST_S + (hit - 1) * STRIDE_S})
    share = lambda k: round(k / total, 4) if total else None
    return {"computations": total, "raw_valid_share": share(valid), "shown_share": share(len(shown_values)),
            "shown_median": statistics.median(shown_values) if shown_values else None,
            "shown_min": min(shown_values) if shown_values else None,
            "shown_max": max(shown_values) if shown_values else None,
            "step_pairs": len(steps), "step_median_abs": statistics.median(steps) if steps else None,
            "first_shown": first}


def analyse(path):
    runs = parse(path)
    both, mismatches = [], 0
    for run in runs:
        dm, dr = display(candidates(run, "maxim")), display(candidates(run, "rf"))
        for row, (m, _), (r, _) in zip(run, dm, dr):
            mismatches += m != row["fw_shown"]
            if m is not None and r is not None:
                both.append(abs(m - r))
    return {"capture": str(path), "capture_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "d10": D10, "computations": sum(len(r) for r in runs),
            "runs": [{"run": i, "first_n": r[0]["n"], "last_n": r[-1]["n"], "lines": len(r),
                      "missing_n": r[-1]["n"] - r[0]["n"] + 1 - len(r)} for i, r in enumerate(runs, 1)],
            "algorithms": {"maxim": summarise(runs, "maxim"), "rf": summarise(runs, "rf")},
            "both_shown": {"n": len(both), "median_abs_diff": statistics.median(both) if both else None,
                           "max_abs_diff": max(both) if both else None},
            "firmware_shown_mismatches": mismatches}


def markdown(res):
    fmt = lambda v, pct=False: "-" if v is None else (f"{v:.1%}" if pct else str(v))
    a = res["algorithms"]
    lines = ["# SpO2 est algorithm comparison (ADR-0010)", "", f"**{D10}**", "",
             "No reference oximeter, oxygen not varied, one sensor. SpO2 est is uncalibrated and display only.", "",
             f"Capture `{res['capture']}` (SHA-256 `{res['capture_sha256']}`): {res['computations']} computations "
             f"in {len(res['runs'])} run(s).", "",
             "| run | n | lines | missing n |", "|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['first_n']}-{r['last_n']} | {r['lines']} | {r['missing_n']} |" for r in res["runs"]]
    lines += ["", "Same display rule for both (valid, 70-100, held through 2 rejected, dash on the 3rd; RF rounded half up).", "",
              "| | Maxim (tile) | Fraczkiewicz (serial only) |", "|---|---|---|"]
    rows = [("computations", "computations"), ("raw valid", "raw_valid_share"), ("shown", "shown_share"),
            ("shown median", "shown_median"), ("shown min", "shown_min"), ("shown max", "shown_max"),
            ("step pairs (consecutive accepted)", "step_pairs"), ("median abs step", "step_median_abs")]
    for label, key in rows:
        pct = key.endswith("_share")
        lines.append(f"| {label} | {fmt(a['maxim'][key], pct)} | {fmt(a['rf'][key], pct)} |")
    first = lambda f: ", ".join("-" if x["contact_s"] is None else f"{x['contact_s']} s" for x in f)
    lines.append(f"| first shown, per run (s of contact) | {first(a['maxim']['first_shown'])} | {first(a['rf']['first_shown'])} |")
    b = res["both_shown"]
    lines += ["", f"Both shown at {b['n']} computations: median |difference| {fmt(b['median_abs_diff'])}, "
              f"max {fmt(b['max_abs_diff'])}.", "",
              f"Maxim display recomputed here vs the firmware's shown= field: {res['firmware_shown_mismatches']} mismatch(es).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    for out in (args.json, args.md):
        if out and out.exists():
            parser.error(f"{out} exists; refusing to overwrite")
    res = analyse(args.capture)
    if args.json:
        with args.json.open("x", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    if args.md:
        with args.md.open("x", encoding="utf-8") as f:
            f.write(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

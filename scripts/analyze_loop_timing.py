"""Analyse a live-loop capture for sample loss around v2 decisions. Host only: reads a file.

Input: a capture written by build/v2/cap.py, one line per serial line:
    HH:MM:SS.mmm > <firmware line>
from either the shipped live image (`nano33ble_v2`, no LT lines) or the diagnostic
`nano33ble_v2_looptiming` image, which adds
    LT: k=v ...       one per decision, printed at the end of the pass after the decision pass
    LT_IDLE: k=v ...  every 5 s while no decision has arrived for 7 s
    LT_INFO: k=v ...  once at boot
(field meanings: outputs/firmware-v2-looptiming/README.md), or the second diagnostic
`nano33ble_v2_looptiming2`, which also prints, right after each LT: line (same n),
    LT2: k=v ...      decision sub-phases and the 4 kHz context sampler
    LT2_INFO: k=v ... once at boot
(field meanings: outputs/firmware-v2-looptiming2/README.md).

Two independent sources are reported and never merged:
  host  - decision intervals from host arrival times of `  Feat:` lines. cap.py reads with
          a 0.2 s timeout, so each timestamp can lag the board by up to ~0.2 s.
  board - LT/LT_IDLE fields, measured on the board in DWT cycles (64 MHz) and micros().

An interval is "contiguous" when no no-finger sample fell inside it (board: nf == 0) and the
window index advanced by exactly one; only those enter loss and interval statistics.
Loss = expected records (drain-to-drain elapsed x 50 /s) - records drained. A sensor that
runs slightly below 50 Hz (49.97 Hz measured) biases this by about -0.15 records per 5 s.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) > ?(.*)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?\d+)")
NAME_RE = re.compile(r"\b(oth\d+_name)=(\S+)")
HZ_RE = re.compile(r"\bHz: (\d+(?:\.\d+)?)")
WIN_RE = re.compile(r"\bWin: (\d+)")
SAMPLES_RE = re.compile(r"\bSamples: (\d+)")

RECORDS_PER_S = 50
STRIDE = 250
FIFO_MS = 640

# Decision-pass phases in LT lines; st excludes feat. The value is <name>_us / <name>_c.
PHASES = ["pass", "central", "drain", "dec", "push", "hrp", "hre", "st", "feat", "ble", "other",
          "post_gap", "post_pass", "ltp"]

# Decision sub-phases in LT2 lines (loop_timing_v2_phases.h, LT2_PHASE_KEYS).
LT2_PHASES = ["ingest", "align_ir", "filt_ir", "mean_ir", "align_red", "filt_red", "mean_red",
              "feat14", "feat2", "kan_base", "kan_pert", "sort", "tail"]

# Instructions each sub-phase executes in a subsequent (1,000-sample) decision: median over the
# timing-replay windows of the compiled lt2_evaluate() of nano33ble_v2_looptiming2, run in an
# emulator (scripts/emulate_decision_cost.py --mirror-elf; input dependence about 1% or less).
# TIMING_IMAGE_CPI is the board's cycles per emulated instruction on the timing image (median of
# 21 windows). insn x CPI / 64 MHz is what a phase takes with nothing else using the CPU.
EMULATED_INSN_SUBSEQUENT = {
    "align_ir": 10031, "filt_ir": 7730998, "mean_ir": 58740, "align_red": 10040, "filt_red": 7725846,
    "mean_red": 58740, "feat14": 1929036, "feat2": 198124, "kan_base": 66102, "kan_pert": 997571,
    "sort": 5627, "tail": 283,
}  # outputs/firmware-v2-looptiming2/mirror-validation.json, phase_insn_median_subsequent
TIMING_IMAGE_CPI = 1.473


def parse_kv(text):
    return {k: int(v) for k, v in KV_RE.findall(text)}


def parse_capture(lines):
    """Return {'lt': [...], 'idle': [...], 'info': [...], 'decisions': [...], 'hz': [...]}.

    Times are seconds since midnight of the host clock; a wrap past midnight adds 86400.
    """
    out = {"lt": [], "lt2": [], "lt2_info": [], "idle": [], "info": [], "decisions": [], "hz": [],
           "lines": 0}
    prev_t = None
    day = 0.0
    last_red = None  # (t, win, samples) of the latest `Red:` line
    for raw in lines:
        m = LINE_RE.match(raw.rstrip("\r\n"))
        if not m:
            continue
        hh, mm, ss, ms, body = m.groups()
        t = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
        if prev_t is not None and t + day < prev_t - 43200:
            day += 86400.0
        t += day
        prev_t = t
        out["lines"] += 1
        stripped = body.strip()
        if stripped.startswith("LT2_INFO:"):
            rec = parse_kv(stripped[len("LT2_INFO:"):])
            rec["t"] = t
            out["lt2_info"].append(rec)
        elif stripped.startswith("LT2:"):
            rec = parse_kv(stripped[len("LT2:"):])
            rec.update(dict(NAME_RE.findall(stripped)))
            rec["t"] = t
            out["lt2"].append(rec)
        elif stripped.startswith("LT_IDLE:"):
            rec = parse_kv(stripped[len("LT_IDLE:"):])
            rec["t"] = t
            out["idle"].append(rec)
        elif stripped.startswith("LT_INFO:"):
            rec = parse_kv(stripped[len("LT_INFO:"):])
            rec["t"] = t
            out["info"].append(rec)
        elif stripped.startswith("LT:"):
            rec = parse_kv(stripped[len("LT:"):])
            rec["t"] = t
            out["lt"].append(rec)
        elif stripped.startswith("Red:"):
            hz = HZ_RE.search(stripped)
            if hz:
                out["hz"].append({"t": t, "hz": float(hz.group(1))})
            w = WIN_RE.search(stripped)
            s = SAMPLES_RE.search(stripped)
            last_red = (t, int(w.group(1)) if w else None, int(s.group(1)) if s else None)
        elif stripped.startswith("Feat:"):
            # The decision line is split by the firmware's "\n  Feat:"; Win/Samples are on the
            # `Red:` half, which arrives just before.
            win = last_red[1] if last_red else None
            samples = last_red[2] if last_red else None
            out["decisions"].append({"t": t, "win": win, "samples": samples})
    return out


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": statistics.median(vals), "mean": statistics.fmean(vals),
            "min": min(vals), "max": max(vals)}


def host_intervals(decisions):
    """Consecutive decisions whose window index advanced by one (no pipeline reset between)."""
    rows = []
    for a, b in zip(decisions, decisions[1:]):
        contiguous = (a["win"] is not None and b["win"] is not None and b["win"] == a["win"] + 1
                      and (a["samples"] is None or b["samples"] is None
                           or b["samples"] - a["samples"] == STRIDE))
        dt = b["t"] - a["t"]
        rows.append({"win": b["win"], "t": b["t"], "dt_s": round(dt, 3), "contiguous": contiguous,
                     "effective_hz": round(STRIDE / dt, 3) if contiguous and dt > 0 else None,
                     "implied_loss_records": round(dt * RECORDS_PER_S - STRIDE, 2)
                     if contiguous else None})
    return rows


def analyse_lt(lt):
    rows = []
    prev_win = None
    for r in lt:
        contiguous = (r.get("nf", 0) == 0 and prev_win is not None and r.get("win") == prev_win + 1
                      and r.get("ddt_us", 0) > 0)
        prev_win = r.get("win")
        exp = r.get("exp_mrec", 0) / 1000.0
        loss = r.get("loss_mrec", 0) / 1000.0
        rows.append({
            "n": r.get("n"), "win": r.get("win"), "contiguous": contiguous,
            "conn": r.get("conn"),
            "ddt_ms": r.get("ddt_us", 0) / 1000.0, "el_ms": r.get("el_us", 0) / 1000.0,
            "drained": r.get("drained"), "expected": round(exp, 3), "loss": round(loss, 3),
            "pass_ms": r.get("pass_us", 0) / 1000.0, "dec_ms": r.get("dec_us", 0) / 1000.0,
            "feat_ms": r.get("feat_us", 0) / 1000.0, "st_ms": r.get("st_us", 0) / 1000.0,
            "st_n": r.get("st_n"), "ble_ms": r.get("ble_us", 0) / 1000.0,
            "post_gap_ms": r.get("post_gap_us", 0) / 1000.0, "post_pend": r.get("post_pend"),
            "post_got": r.get("post_got"), "post_wr": r.get("post_wr"), "post_rd": r.get("post_rd"),
            "frames": r.get("frames"), "dpos": r.get("dpos"),
            "p600": r.get("p600"), "p640": r.get("p640"), "g600": r.get("g600"),
            "g640": r.get("g640"), "e600": r.get("e600"), "short": r.get("short"),
            "maxpass_ms": r.get("maxpass_us", 0) / 1000.0,
            "maxgap_ms": r.get("maxgap_us", 0) / 1000.0, "ltp_ms": r.get("ltp_us", 0) / 1000.0,
        })
    good = [x for x, r in zip(rows, lt) if x["contiguous"]]
    good_raw = [r for x, r in zip(rows, lt) if x["contiguous"]]
    phases = {}
    for p in PHASES:
        us = [r[f"{p}_us"] for r in good_raw if f"{p}_us" in r]
        cyc = [r[f"{p}_c"] for r in good_raw if f"{p}_c" in r]
        phases[p] = {"us": summarize(us), "cycles": summarize(cyc)}
    counts = {k: sum(r.get(k, 0) for r in lt) for k in ("p600", "p640", "g600", "g640", "e600", "short")}
    summary = {
        "lt_lines": len(lt), "contiguous_intervals": len(good),
        "loss_records_per_decision": summarize([x["loss"] for x in good]),
        "drained_per_decision": summarize([x["drained"] for x in good]),
        "expected_per_decision": summarize([x["expected"] for x in good]),
        "decision_interval_ms_board": summarize([x["ddt_ms"] for x in good]),
        "effective_hz_board": summarize([STRIDE / (x["ddt_ms"] / 1000.0) for x in good if x["ddt_ms"] > 0]),
        "decision_pass_phases": phases,
        "max_pass_ms": max([x["maxpass_ms"] for x in rows], default=None),
        "max_gap_ms": max([x["maxgap_ms"] for x in rows], default=None),
        "counts_all_lt_intervals": counts,
        "post_pend_zero_after_decision": sum(1 for x in good if x["post_pend"] == 0),
        "intervals_with_loss_ge_1": sum(1 for x in good if x["loss"] >= 1.0),
    }
    return rows, summary


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _rank([a for a, _ in pairs]), _rank([b for _, b in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy), 3)


def analyse_lt2(lt2, lt, expected_insn=None, cpi=TIMING_IMAGE_CPI):
    """Join LT2 to LT by n. Fractions are of sampler ticks taken during the decision block."""
    expected_insn = expected_insn or EMULATED_INSN_SUBSEQUENT
    by_n = {r.get("n"): r for r in lt}
    rows = []
    threads, isrs = {}, {}
    for r in lt2:
        base = by_n.get(r.get("n"), {})
        n_s = r.get("smp_n", 0)

        def frac(k):
            return r.get(k, 0) / n_s if n_s else None

        phases_us = r.get("phases_us", 0)
        row = {"n": r.get("n"), "win": r.get("win"), "first": r.get("win") == 0,
               "dec_us": base.get("dec_us"), "phases_us": phases_us,
               "smp_n": n_s, "main_frac": frac("smp_main"), "oth_frac": frac("smp_oth"),
               "isr_frac": frac("smp_isr"), "smp_full": r.get("smp_full"),
               "sampler_cost_us": r.get("smp_cost_c", 0) / 64.0,
               "main_only_us": phases_us * frac("smp_main") if n_s else None,
               "stk_size": r.get("stk_size"), "stk_free": r.get("stk_free"),
               "phases": {p: {"us": r.get(f"{p}_us"), "c": r.get(f"{p}_c")} for p in LT2_PHASES}}
        if expected_insn and not row["first"]:
            row["cycles_per_insn"] = {p: r.get(f"{p}_c", 0) / expected_insn[p]
                                      for p in LT2_PHASES if expected_insn.get(p)}
        rows.append(row)
        for i in range(1, 5):
            if f"oth{i}_n" in r:
                name = r.get(f"oth{i}_name", str(r.get(f"oth{i}_id")))
                threads[name] = threads.get(name, 0) + r[f"oth{i}_n"]
        for i in range(1, 7):
            if f"isr{i}_n" in r:
                isrs[r[f"isr{i}_exc"]] = isrs.get(r[f"isr{i}_exc"], 0) + r[f"isr{i}_n"]
    sub = [x for x in rows if not x["first"] and x["smp_n"]]
    summary = {
        "lt2_lines": len(rows), "subsequent": len(sub),
        "phases_us": {p: summarize([x["phases"][p]["us"] for x in sub]) for p in LT2_PHASES},
        "phases_c": {p: summarize([x["phases"][p]["c"] for x in sub]) for p in LT2_PHASES},
        "phases_total_us": summarize([x["phases_us"] for x in sub]),
        "dec_us": summarize([x["dec_us"] for x in sub]),
        "main_frac": summarize([x["main_frac"] for x in sub]),
        "oth_frac": summarize([x["oth_frac"] for x in sub]),
        "isr_frac": summarize([x["isr_frac"] for x in sub]),
        "main_only_us": summarize([x["main_only_us"] for x in sub]),
        "sampler_cost_us": summarize([x["sampler_cost_us"] for x in sub]),
        "spearman_phases_vs_oth_frac": spearman([x["phases_us"] for x in sub], [x["oth_frac"] for x in sub]),
        "spearman_phases_vs_isr_frac": spearman([x["phases_us"] for x in sub], [x["isr_frac"] for x in sub]),
        "spearman_phases_vs_main_only": spearman([x["phases_us"] for x in sub],
                                                 [x["main_only_us"] for x in sub]),
        "ticks_by_other_thread": threads,
        "ticks_by_interrupted_exception": {str(k): v for k, v in sorted(isrs.items())},
        "ticks_total": sum(x["smp_n"] for x in rows),
        "tally_overflow_ticks": sum(x["smp_full"] or 0 for x in rows),
        "stk_free_min": min([x["stk_free"] for x in rows if x["stk_free"] is not None], default=None),
        "stk_size": rows[0]["stk_size"] if rows else None,
    }
    if sub and "cycles_per_insn" in sub[0]:
        summary["cycles_per_insn"] = {p: summarize([x["cycles_per_insn"][p] for x in sub])
                                      for p in sub[0]["cycles_per_insn"]}
        summary["expected_us_at_timing_cpi"] = {p: expected_insn[p] * cpi / 64.0 for p in expected_insn}
    return rows, summary


def analyse_idle(idle):
    if not idle:
        return None
    el = sum(r.get("el_us", 0) for r in idle)
    drained = sum(r.get("drained", 0) for r in idle)
    exp = sum(r.get("exp_mrec", 0) for r in idle) / 1000.0
    return {
        "lines": len(idle), "elapsed_s": el / 1e6, "drained": drained, "expected": round(exp, 3),
        "loss": round(exp - drained, 3),
        "max_pass_ms": max(r.get("maxpass_us", 0) for r in idle) / 1000.0,
        "max_gap_ms": max(r.get("maxgap_us", 0) for r in idle) / 1000.0,
        "passes": sum(r.get("passes", 0) for r in idle),
        "p600": sum(r.get("p600", 0) for r in idle), "p640": sum(r.get("p640", 0) for r in idle),
        "e600": sum(r.get("e600", 0) for r in idle),
        "nf": sum(r.get("nf", 0) for r in idle), "fin": sum(r.get("fin", 0) for r in idle),
        "ltp_ms": summarize([r.get("ltp_us", 0) / 1000.0 for r in idle]),
    }


def analyse(lines):
    cap = parse_capture(lines)
    host_rows = host_intervals(cap["decisions"])
    good_host = [r for r in host_rows if r["contiguous"]]
    dts = [r["dt_s"] for r in good_host]
    hist = {}
    for dt in dts:
        b = f"{round(dt * 10) / 10:.1f}"
        hist[b] = hist.get(b, 0) + 1
    report = {
        "capture_lines": cap["lines"],
        "host": {
            "note": "Host arrival times (cap.py 0.2 s read timeout): each dt can be off by ~0.2 s.",
            "decisions": len(cap["decisions"]),
            "contiguous_intervals": len(good_host),
            "interval_s": summarize(dts),
            "interval_s_histogram_0p1": dict(sorted(hist.items())),
            "effective_hz": summarize([r["effective_hz"] for r in good_host]),
            "implied_loss_records_per_decision": summarize([r["implied_loss_records"] for r in good_host]),
            "hz_field": summarize([h["hz"] for h in cap["hz"]]),
            "hz_field_last": cap["hz"][-1]["hz"] if cap["hz"] else None,
            "intervals": host_rows,
        },
        "lt_info": cap["info"],
        "board": None,
        "idle": analyse_idle(cap["idle"]),
    }
    if cap["lt"]:
        rows, summary = analyse_lt(cap["lt"])
        report["board"] = {"summary": summary, "decisions": rows}
    else:
        report["board"] = {"summary": None, "note": "No LT lines: capture is not from the "
                           "nano33ble_v2_looptiming image. Board-side timing unavailable."}
    report["lt2_info"] = cap["lt2_info"]
    report["phases"] = None
    if cap["lt2"]:
        rows2, summary2 = analyse_lt2(cap["lt2"], cap["lt"])
        report["phases"] = {"summary": summary2, "decisions": rows2}
    return report


def _fmt(s, key="median", nd=1):
    if not s:
        return "-"
    v = s[key]
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown(report, source):
    h = report["host"]
    out = [f"# Loop timing analysis: `{source}`", ""]
    out += ["## Host decision intervals (from `Feat:` arrival times; +-0.2 s each)", "",
            "| decisions | contiguous intervals | interval median s | min s | max s | effective Hz median | implied loss / decision (records, median) | `Hz:` field median | last `Hz:` |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {h['decisions']} | {h['contiguous_intervals']} | {_fmt(h['interval_s'], nd=3)} | "
            f"{_fmt(h['interval_s'], 'min', 3)} | {_fmt(h['interval_s'], 'max', 3)} | "
            f"{_fmt(h['effective_hz'], nd=2)} | {_fmt(h['implied_loss_records_per_decision'], nd=1)} | "
            f"{_fmt(h['hz_field'], nd=2)} | {h['hz_field_last'] if h['hz_field_last'] is not None else '-'} |", ""]
    if h["interval_s_histogram_0p1"]:
        out += ["Interval histogram (0.1 s bins): " + ", ".join(
            f"{k} s: {v}" for k, v in h["interval_s_histogram_0p1"].items()), ""]
    b = report["board"]
    if b["summary"] is None:
        out += ["## Board timing", "", b["note"], ""]
    else:
        s = b["summary"]
        out += ["## Board: per decision interval (contiguous only)", "",
                f"LT lines {s['lt_lines']}, contiguous {s['contiguous_intervals']}; "
                f"intervals with loss >= 1 record: {s['intervals_with_loss_ge_1']}; "
                f"post-decision drain found pending == 0: {s['post_pend_zero_after_decision']}.", "",
                "| metric | median | mean | min | max |", "|---|---|---|---|---|"]
        for name, key in (("loss records / decision", "loss_records_per_decision"),
                          ("drained / decision", "drained_per_decision"),
                          ("expected / decision", "expected_per_decision"),
                          ("decision interval ms", "decision_interval_ms_board"),
                          ("effective Hz", "effective_hz_board")):
            v = s[key]
            out.append(f"| {name} | {_fmt(v, nd=2)} | {_fmt(v, 'mean', 2)} | {_fmt(v, 'min', 2)} | {_fmt(v, 'max', 2)} |")
        out += ["", "## Board: decision-pass breakdown (us; contiguous decisions)", "",
                "| phase | median us | max us | median cycles | max cycles |", "|---|---|---|---|---|"]
        for p, v in s["decision_pass_phases"].items():
            out.append(f"| {p} | {_fmt(v['us'], nd=0)} | {_fmt(v['us'], 'max', 0)} | "
                       f"{_fmt(v['cycles'], nd=0)} | {_fmt(v['cycles'], 'max', 0)} |")
        c = s["counts_all_lt_intervals"]
        out += ["", f"Max pass {s['max_pass_ms']} ms, max drain gap {s['max_gap_ms']} ms. "
                f"Passes > 600 ms: {c['p600']}, > 640 ms: {c['p640']}; drain gaps > 600 ms: {c['g600']}, "
                f"> 640 ms: {c['g640']}; pending == 0 after a gap > 600 ms: {c['e600']}; short reads: {c['short']}.", ""]
    ph = report.get("phases")
    if ph:
        s2 = ph["summary"]
        out += ["## Board: decision sub-phases (LT2; subsequent decisions)", "",
                f"LT2 lines {s2['lt2_lines']}, subsequent {s2['subsequent']}. Sampler ticks during the "
                f"decision block (medians): loop thread {_fmt(s2['main_frac'], nd=3)}, other threads "
                f"{_fmt(s2['oth_frac'], nd=3)}, other ISRs {_fmt(s2['isr_frac'], nd=3)}. Loop-thread-only time "
                f"median {_fmt(s2['main_only_us'], nd=0)} us [{_fmt(s2['main_only_us'], 'min', 0)}.."
                f"{_fmt(s2['main_only_us'], 'max', 0)}] vs sum of phases {_fmt(s2['phases_total_us'], nd=0)} us "
                f"[{_fmt(s2['phases_total_us'], 'min', 0)}..{_fmt(s2['phases_total_us'], 'max', 0)}].", "",
                f"Spearman(sum of phases, other-thread fraction) = {s2['spearman_phases_vs_oth_frac']}; "
                f"(sum, ISR fraction) = {s2['spearman_phases_vs_isr_frac']}; "
                f"(sum, loop-thread-only time) = {s2['spearman_phases_vs_main_only']}.", "",
                "| sub-phase | median us | max us | median cycles | expected us at timing-image CPI "
                "| cycles/insn median | cycles/insn max |",
                "|---|---|---|---|---|---|---|"]
        exp = s2.get("expected_us_at_timing_cpi", {})
        cpi = s2.get("cycles_per_insn", {})
        for p in LT2_PHASES:
            e, c = exp.get(p), cpi.get(p)
            out.append(f"| {p} | {_fmt(s2['phases_us'][p], nd=0)} | {_fmt(s2['phases_us'][p], 'max', 0)} | "
                       f"{_fmt(s2['phases_c'][p], nd=0)} | {f'{e:.0f}' if e else '-'} | "
                       f"{_fmt(c, nd=3) if c else '-'} | {_fmt(c, 'max', 3) if c else '-'} |")
        out += ["", f"Ticks by other thread: {s2['ticks_by_other_thread']}; by interrupted exception number: "
                f"{s2['ticks_by_interrupted_exception']}; of {s2['ticks_total']} ticks "
                f"(tally overflow {s2['tally_overflow_ticks']}). Loop-thread stack: size {s2['stk_size']}, "
                f"lowest free {s2['stk_free_min']}.", ""]
    i = report["idle"]
    if i:
        out += ["## LT_IDLE", "",
                f"{i['lines']} lines over {i['elapsed_s']:.1f} s: drained {i['drained']} vs expected "
                f"{i['expected']} (loss {i['loss']}); max pass {i['max_pass_ms']} ms, max gap {i['max_gap_ms']} ms; "
                f"p600 {i['p600']}, p640 {i['p640']}, e600 {i['e600']}; no-finger samples {i['nf']}, finger samples {i['fin']}.", ""]
    else:
        out += ["## LT_IDLE", "", "No LT_IDLE lines.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", type=Path)
    ap.add_argument("--json", type=Path, help="write the JSON report here (refuses to overwrite)")
    ap.add_argument("--md", type=Path, help="write the markdown table here (refuses to overwrite)")
    args = ap.parse_args(argv)
    with open(args.capture, encoding="utf-8", errors="replace") as f:
        report = analyse(f)
    report["source"] = str(args.capture)
    md = markdown(report, args.capture.name)
    for path, text in ((args.json, json.dumps(report, indent=1)), (args.md, md)):
        if path is None:
            continue
        if path.exists():
            sys.exit(f"refusing to overwrite {path}")
        path.write_text(text + "\n", encoding="utf-8")
    print(md)
    return report


if __name__ == "__main__":
    main()

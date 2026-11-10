"""Analyse a peak-RAM probe capture. Host only: reads a file, opens no port.

Input: a capture written by build/v2/cap.py from a probe image (env nano33ble_peakram or
nano33ble_v2_f1a_peakram), one line per serial line:
    HH:MM:SS.mmm > <firmware line>
The probe (firmware/Arduino Nano 33/include/peak_ram_probe.h) prints
    PR_INFO: k=v ...  region table, paint ranges and threads; end of setup() and every 60 s
    PR: k=v ...       every 5 s: per-region reserved/peak/headroom and the whole-device
                      high-water (docs/peak-ram-design.md, sections 3 and 5)
next to the unchanged live lines.

The figure is the final PR: line's whole_hw. High-water marks never fall, so the final line
also holds the maximum over the capture; the analyser checks that. The validity criteria
are fixed in docs/peak-ram-runbook.md and in the constants below. A run that fails any of
them gives no paper figure.

With --image <archive>/firmware.bin --bossac-log <log> the report is also bound to the flash:
the log must show 'Verify successful' and a write of the image's size, and the image must
match its archive's SHA256SUMS (scripts/record_v2_upload_receipt.py knows only the r2
archives, so the probe flashes are bound here instead).

With --sampler the capture is from a sampler image (env nano33ble_peakram_sampler or
nano33ble_v2_f1a_peakram_sampler: live finger gate, TIMER4 instantaneous-total sampler; design
section 8) and the checks are the ones pre-registered on 30 Oct 2026 for the finger-on,
dashboard-connected run (docs/peak-ram-runbook.md, sampler section). Without it, nothing changes.

Writes <output>/report.json and <output>/report.md. Refuses an existing output directory.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) > ?(.*)$")
TOKEN_RE = re.compile(r"(\S+?)=(\S+)")
HZ_RE = re.compile(r"\bHz: (\d+(?:\.\d+)?)")

RAM_BYTES = 262144
IMAGES = ("nano33ble_peakram", "nano33ble_v2_f1a_peakram")

# Validity criteria, fixed 25 Oct 2026 before any probe flash (docs/peak-ram-runbook.md).
MIN_PR_LINES = 30                 # of 36 expected in a 180 s capture
MIN_FINAL_DECISIONS = 25          # forced decisions since boot, final PR: line
MIN_DECISIONS_IN_CAPTURE = 20     # final dec - first dec seen in the capture
MAX_GAP_S = 15.0                  # no PR: line missing for longer than this
MAX_LIVE_FINGER_FRACTION = 0.01   # the live gate's own no-finger verdict

PAPER_WORDING = ("whole-device RAM high-water on the probe image, no finger, forced decisions; "
                 "conservative upper bound")

# Sampler images (--sampler). Constants pre-registered 30 Oct 2026 before any sampler build or
# run (docs/peak-ram-runbook.md, "Sampler run ... validity checks").
SAMPLER_IMAGES = ("nano33ble_peakram_sampler", "nano33ble_v2_f1a_peakram_sampler")
SAMPLER_HZ = 4000
MIN_FINGER_FRACTION = 0.95        # live gate during the capture
MIN_DECISIONS_DURING = 25         # decisions during the capture
MIN_HZ = 49.5                     # every Hz: value in the capture
MIN_TICK_FRACTION = 0.90          # ticks serviced / (SAMPLER_HZ x capture time)
MAX_SKIP_FRACTION = 0.05          # skipped ticks / ticks serviced, during the capture
SAMPLER_WORDING = "peak RAM between {lo:,} B and {hi:,} B on the probe image, finger on, dashboard connected"
PR_SAMPLER_INT_KEYS = ["ble", "sub", "ble_conn", "ble_disc", "hl", "hl_st", "s_n", "s_ok", "s_skk", "s_skl",
                       "s_skh", "s_skt", "s_set", "samp_max", "s_win", "sm_heap", "sm_msp", "sm_stk",
                       "smx_heap", "smx_msp", "s_cyc_max", "s_cyc_sum"]
INFO_SAMPLER_INT_KEYS = ["smp", "smp_hz", "smp_prio", "prio_svc", "prio_pendsv", "prio_systick", "s_const", "s_nthr"]
INFO_SAMPLER_STR_KEYS = ["smp_timer", "vtor", "hlock"]
# Start diagnostics appended to PR_INFO: on 1 Nov 2026 (r2 sampler images, design section 8 note):
# prs_start()'s three preconditions and each step of the malloc-mutex lookup. Optional (the 31 Oct
# images do not print them), kept as strings, read by no check; they only add to the report.
INFO_SAMPLER_DIAG_INT_KEYS = ["st_hlock", "st_set", "st_vtor", "lk_step", "lk_ipsr", "lk_mask", "lk_own", "lk_cnt",
                              "lk_rel", "lk_stale"]
INFO_SAMPLER_DIAG_HEX_KEYS = ["lk_self", "lk_pre", "lk_head", "lk_id", "lk_post"]
SAMPLER_MONOTONIC = ["samp_max", "s_n", "s_ok", "ble_conn", "ble_disc"]
SKIP_KEYS = ["s_skk", "s_skl", "s_skh", "s_skt"]

# Keys every PR: line carries besides the region triples and T.<thread> entries.
PR_INT_KEYS = ["v", "t_ms", "rep", "dec", "lf", "lnf", "msp_loop", "main_pre", "main_setup",
               "main_loop", "nthr", "hs_cur", "hs_max", "hs_ovh", "hs_fail", "static", "fixed",
               "untouched", "ovl", "whole_hw", "ram", "scan_us", "rpt_alloc"]
PR_HEX_KEYS = ["brk", "heap_hw"]
PR_REGION_KEYS = ["heap", "msp"]
INFO_SPAN_KEYS = ["ram", "fixed", "data", "bss", "heap", "msp", "heap_paint", "msp_paint", "main_paint"]
INFO_INT_KEYS = ["v", "force", "gap", "wm", "period_ms", "heap_painted", "msp_painted",
                 "main_painted", "setup_painted", "nthr"]
INFO_STR_KEYS = ["img", "model", "hook", "pattern", "rtx_fill", "brk_ctor", "brk_setup"]
INFO_FLAGS = ["force", "wm", "heap_painted", "msp_painted", "main_painted", "setup_painted"]
# High-water fields that must never fall from one PR: line to the next.
MONOTONIC = ["whole_hw", "msp_loop", "main_loop", "hs_max"]
CONSTANT = ["main_pre", "main_setup", "static", "fixed", "ram"]


class CaptureError(ValueError):
    pass


def parse_tokens(text):
    return dict(TOKEN_RE.findall(text))


def parse_triple(value):
    """'R/P/H' or 'R/P/H/loc' -> dict; raises CaptureError on anything else."""
    parts = value.split("/")
    if len(parts) not in (3, 4):
        raise CaptureError(f"bad region value {value!r}")
    try:
        r, p, h = (int(x) for x in parts[:3])
    except ValueError as exc:
        raise CaptureError(f"bad region value {value!r}") from exc
    out = {"reserved": r, "peak": p, "headroom": h}
    if len(parts) == 4:
        out["loc"] = parts[3]
    return out


def parse_span(value):
    """'0xADDR+N' -> (addr, n)."""
    m = re.fullmatch(r"0x([0-9A-Fa-f]+)\+(\d+)", value)
    if not m:
        raise CaptureError(f"bad span {value!r}")
    return int(m.group(1), 16), int(m.group(2))


def parse_pr(text):
    """One PR: line body -> record. Missing or malformed fields raise CaptureError."""
    tok = parse_tokens(text)
    rec = {"threads": {}}
    for k in PR_INT_KEYS:
        if k not in tok:
            raise CaptureError(f"PR: line lacks {k}")
        try:
            rec[k] = int(tok[k])
        except ValueError as exc:
            raise CaptureError(f"PR: {k}={tok[k]!r} is not an integer") from exc
    for k in PR_HEX_KEYS:
        if k not in tok:
            raise CaptureError(f"PR: line lacks {k}")
        rec[k] = int(tok[k], 16)
    for k in PR_REGION_KEYS:
        if k not in tok:
            raise CaptureError(f"PR: line lacks {k}")
        rec[k] = parse_triple(tok[k])
    for k, v in tok.items():
        if k.startswith("T."):
            rec["threads"][k] = parse_triple(v)
    if "samp_max" in tok:
        rec["sampler"] = parse_sampler_pr(tok)
    return rec


def parse_sampler_pr(tok):
    """The sampler fields of a PR: line (present only on sampler images)."""
    out = {"threads": {}}
    for k in PR_SAMPLER_INT_KEYS:
        if k not in tok:
            raise CaptureError(f"PR: sampler line lacks {k}")
        try:
            out[k] = int(tok[k])
        except ValueError as exc:
            raise CaptureError(f"PR: {k}={tok[k]!r} is not an integer") from exc
    for k, v in tok.items():
        if k.startswith("S."):
            m = re.fullmatch(r"(\d+)/(\d+)", v)
            if not m:
                raise CaptureError(f"bad sampler thread {k}={v!r}")
            out["threads"][k] = {"at_max": int(m.group(1)), "max": int(m.group(2))}
    return out


def parse_info(text):
    tok = parse_tokens(text)
    rec = {"threads": {}}
    for k in INFO_STR_KEYS:
        if k not in tok:
            raise CaptureError(f"PR_INFO: line lacks {k}")
        rec[k] = tok[k]
    for k in INFO_INT_KEYS:
        if k not in tok:
            raise CaptureError(f"PR_INFO: line lacks {k}")
        rec[k] = int(tok[k])
    for k in INFO_SPAN_KEYS:
        if k not in tok:
            raise CaptureError(f"PR_INFO: line lacks {k}")
        rec[k] = parse_span(tok[k])
    for k, v in tok.items():
        if k.startswith("T."):
            m = re.fullmatch(r"0x([0-9A-Fa-f]+)\+(\d+)/(\w+)", v)
            if not m:
                raise CaptureError(f"bad PR_INFO thread {k}={v!r}")
            rec["threads"][k] = {"base": int(m.group(1), 16), "size": int(m.group(2)), "loc": m.group(3)}
    if "smp" in tok:
        smp = {}
        for k in INFO_SAMPLER_INT_KEYS:
            if k not in tok:
                raise CaptureError(f"PR_INFO: sampler line lacks {k}")
            smp[k] = int(tok[k])
        for k in INFO_SAMPLER_STR_KEYS:
            if k not in tok:
                raise CaptureError(f"PR_INFO: sampler line lacks {k}")
            smp[k] = tok[k]
        diag = {k: tok[k] for k in INFO_SAMPLER_DIAG_INT_KEYS + INFO_SAMPLER_DIAG_HEX_KEYS if k in tok}
        if diag:
            smp["diag"] = diag
        rec["sampler"] = smp
    return rec


def parse_capture(lines):
    """Return {'info', 'pr', 'errors', 'hz', 'ble_connected', 'first_t', 'last_t', 'lines'}.

    Host times are seconds since midnight; a wrap past midnight adds 86400. Bare lines
    (no cap.py prefix) are accepted with t = None.
    """
    out = {"info": [], "pr": [], "errors": [], "hz": [], "ble_connected": 0, "ble_disconnected": 0,
           "first_t": None, "last_t": None, "lines": 0}
    prev_t = None
    day = 0.0
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith("--- capture"):
            continue
        m = LINE_RE.match(line)
        t = None
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
            if prev_t is not None and t + day < prev_t - 43200:
                day += 86400.0
            t += day
            prev_t = t
            body = m.group(5)
            if out["first_t"] is None:
                out["first_t"] = t
            out["last_t"] = t
        else:
            body = line
        if not body.strip():
            continue
        out["lines"] += 1
        stripped = body.strip()
        try:
            if stripped.startswith("PR_INFO:"):
                rec = parse_info(stripped[len("PR_INFO:"):])
                rec["t"] = t
                out["info"].append(rec)
            elif stripped.startswith("PR:"):
                rec = parse_pr(stripped[len("PR:"):])
                rec["t"] = t
                out["pr"].append(rec)
        except CaptureError as exc:
            out["errors"].append({"t": t, "error": str(exc), "line": stripped[:200]})
        if "BLE connected to:" in stripped:
            out["ble_connected"] += 1
        if "BLE disconnected" in stripped:
            out["ble_disconnected"] += 1
        hz = HZ_RE.search(stripped)
        if hz:
            out["hz"].append(float(hz.group(1)))
    return out


def rederive(rec):
    """Independent arithmetic on one PR: record: (ok, problems)."""
    problems = []
    regions = {k: rec[k] for k in PR_REGION_KEYS}
    regions.update(rec["threads"])
    for name, r in regions.items():
        if r["peak"] + r["headroom"] != r["reserved"] or not 0 <= r["peak"] <= r["reserved"]:
            problems.append(f"{name}: reserved {r['reserved']} != peak {r['peak']} + headroom {r['headroom']}")
    if rec["ram"] != RAM_BYTES:
        problems.append(f"ram {rec['ram']} != {RAM_BYTES}")
    if rec["whole_hw"] + rec["untouched"] != rec["ram"]:
        problems.append("whole_hw + untouched != ram")
    if rec["ovl"] != 0:
        problems.append(f"ovl {rec['ovl']} != 0 (never-touched intervals overlap)")
    # With no overlap the never-touched union is exactly the sum of the headrooms.
    headrooms = sum(r["headroom"] for r in regions.values())
    if rec["ovl"] == 0 and headrooms != rec["untouched"]:
        problems.append(f"sum of headrooms {headrooms} != untouched {rec['untouched']}")
    if rec["nthr"] != len(rec["threads"]):
        problems.append(f"nthr {rec['nthr']} != {len(rec['threads'])} T. entries")
    return not problems, problems


def check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def validate(cap, expect_image=None):
    checks = []
    info, pr = cap["info"], cap["pr"]

    imgs = sorted({i["img"] for i in info})
    ok = bool(info) and len(imgs) == 1 and imgs[0] in IMAGES and (expect_image is None or imgs[0] == expect_image)
    checks.append(check("identity", ok, f"PR_INFO lines {len(info)}, img {imgs}, expected {expect_image or 'any probe image'}"))

    bad = [f"{k}={i[k]}" for i in info for k in INFO_FLAGS if i[k] != 1]
    bad += [f"gap={i['gap']}" for i in info if i["gap"] != 0]
    bad += [f"ram={i['ram']}" for i in info if i["ram"] != (0x20000000, RAM_BYTES)]
    checks.append(check("painted", bool(info) and not bad,
                        "wm, force and all four paint flags are 1, sections tile RAM" if not bad else ", ".join(sorted(set(bad)))))

    checks.append(check("parse", not cap["errors"], f"{len(cap['errors'])} malformed PR/PR_INFO lines"))
    checks.append(check("pr_lines", len(pr) >= MIN_PR_LINES, f"{len(pr)} PR: lines (need >= {MIN_PR_LINES})"))

    if not pr:
        checks.append(check("gaps", False, "no PR: lines"))
        return checks
    times = [r["t"] for r in pr]
    if None in times or cap["first_t"] is None:
        checks.append(check("gaps", False, "capture has no host timestamps (use build/v2/cap.py)"))
    else:
        gaps = [b - a for a, b in zip(times, times[1:])]
        edges = [times[0] - cap["first_t"], cap["last_t"] - times[-1]]
        worst = max(gaps + edges)
        fw = [(b["t_ms"] - a["t_ms"]) / 1000 for a, b in zip(pr, pr[1:])]
        worst_fw = max(fw) if fw else 0.0
        checks.append(check("gaps", worst <= MAX_GAP_S and worst_fw <= MAX_GAP_S,
                            f"longest stretch without a PR: line {worst:.1f} s host, {worst_fw:.1f} s board (max {MAX_GAP_S:.0f} s)"))

    reboot = [i for i, (a, b) in enumerate(zip(pr, pr[1:])) if b["t_ms"] <= a["t_ms"] or b["rep"] <= a["rep"]]
    checks.append(check("no_reboot", not reboot, "t_ms and rep rise on every line" if not reboot
                        else f"uptime or report counter fell after PR: line {reboot[0] + 1}"))

    first, last = pr[0], pr[-1]
    grew = last["dec"] - first["dec"]
    checks.append(check("decisions", last["dec"] >= MIN_FINAL_DECISIONS and grew >= MIN_DECISIONS_IN_CAPTURE,
                        f"final dec {last['dec']} (need >= {MIN_FINAL_DECISIONS}), {grew} during the capture (need >= {MIN_DECISIONS_IN_CAPTURE})"))

    region_sets = {tuple(sorted(r["threads"])) for r in pr}
    arith = [(i, p) for i, r in enumerate(pr) for p in rederive(r)[1]]
    checks.append(check("regions", len(region_sets) == 1 and not arith,
                        "same heap/msp/thread set on every line; arithmetic re-derives" if len(region_sets) == 1 and not arith
                        else f"{len(region_sets)} different thread sets; " + "; ".join(f"line {i + 1}: {p}" for i, p in arith[:3])))

    falls = []
    for a, b in zip(pr, pr[1:]):
        for k in MONOTONIC:
            if b[k] < a[k]:
                falls.append(f"{k} {a[k]}->{b[k]}")
        for k in PR_REGION_KEYS:
            if b[k]["peak"] < a[k]["peak"]:
                falls.append(f"{k} peak {a[k]['peak']}->{b[k]['peak']}")
        for k in set(a["threads"]) & set(b["threads"]):
            if b["threads"][k]["peak"] < a["threads"][k]["peak"]:
                falls.append(f"{k} peak")
        for k in CONSTANT:
            if b[k] != a[k]:
                falls.append(f"{k} changed {a[k]}->{b[k]}")
    checks.append(check("monotonic", not falls, "no high-water fell" if not falls else "; ".join(falls[:4])))

    samples = last["lf"] + last["lnf"]
    frac = last["lf"] / samples if samples else None
    checks.append(check("no_finger", samples > 0 and frac <= MAX_LIVE_FINGER_FRACTION,
                        f"live gate passed {last['lf']} of {samples} samples" +
                        (f" ({frac:.4%}, max {MAX_LIVE_FINGER_FRACTION:.0%})" if samples else "")))

    checks.append(check("no_central", cap["ble_connected"] == 0,
                        f"{cap['ble_connected']} 'BLE connected' lines (the workload is advertising only)"))
    return checks


def validate_sampler(cap, expect_image=None):
    """The checks pre-registered on 30 Oct 2026 for the finger-on, dashboard-connected run."""
    checks = []
    info, pr = cap["info"], cap["pr"]
    imgs = sorted({i["img"] for i in info})
    ok = bool(info) and len(imgs) == 1 and imgs[0] in SAMPLER_IMAGES and (expect_image is None or imgs[0] == expect_image)
    checks.append(check("identity", ok, f"PR_INFO lines {len(info)}, img {imgs}, expected {expect_image or 'any sampler image'}"))

    bad = [f"{k}={i[k]}" for i in info for k in ("wm", "heap_painted", "msp_painted", "main_painted", "setup_painted") if i[k] != 1]
    bad += [f"force={i['force']}" for i in info if i["force"] != 0]
    bad += [f"gap={i['gap']}" for i in info if i["gap"] != 0]
    bad += [f"ram={i['ram']}" for i in info if i["ram"] != (0x20000000, RAM_BYTES)]
    for i in info:
        smp = i.get("sampler")
        if smp is None:
            bad.append("no sampler fields")
            continue
        if smp["smp"] != 1:
            bad.append(f"smp={smp['smp']}")
        if smp["smp_hz"] != SAMPLER_HZ:
            bad.append(f"smp_hz={smp['smp_hz']}")
        if not (smp["smp_prio"] <= smp["prio_pendsv"] and smp["smp_prio"] <= smp["prio_systick"]):
            bad.append(f"smp_prio {smp['smp_prio']} above PendSV {smp['prio_pendsv']} / SysTick {smp['prio_systick']}")
        if int(smp["hlock"], 16) == 0:
            bad.append("hlock=0")
        if int(smp["vtor"], 16) != 0x20000000:
            bad.append(f"vtor={smp['vtor']}")
    detail = ", ".join(sorted(set(bad)))
    diags = [i["sampler"]["diag"] for i in info if i.get("sampler") and "diag" in i["sampler"]]
    if bad and diags:  # report text only: the criterion above is the pre-registered one
        detail += "; start diagnostics (last PR_INFO): " + " ".join(f"{k}={v}" for k, v in diags[-1].items())
    checks.append(check("config", bool(info) and not bad,
                        "force=0 smp=1, paints 1, sampler priority at or above PendSV/SysTick, malloc lock found, VTOR in RAM"
                        if not bad else detail))

    base = {c["name"]: c for c in validate(cap, None)}
    no_smp = sum(1 for r in pr if "sampler" not in r)
    parse = base["parse"]
    checks.append(check("parse", parse["ok"] and no_smp == 0, parse["detail"] + f"; {no_smp} PR: lines without sampler fields"))
    for name in ("pr_lines", "gaps", "no_reboot"):
        if name in base:
            checks.append(base[name])
    if not pr or no_smp:
        checks.append(check("sampler_lines", False, "no PR: lines, or PR: lines without sampler fields"))
        return checks
    smp = [r["sampler"] for r in pr]
    first, last = pr[0], pr[-1]

    unset = [r["rep"] for r in pr if r["sampler"]["s_set"] != 1]
    reg = base["regions"]
    checks.append(check("regions", reg["ok"] and not unset,
                        reg["detail"] + (f"; s_set=0 at reps {unset[:5]}" if unset else "; s_set=1 on every line")))

    falls = [] if base["monotonic"]["ok"] else [base["monotonic"]["detail"]]
    for a, b in zip(smp, smp[1:]):
        for k in SAMPLER_MONOTONIC:
            if b[k] < a[k]:
                falls.append(f"{k} {a[k]}->{b[k]}")
    checks.append(check("monotonic", not falls, "no high-water or sampler counter fell" if not falls else "; ".join(falls[:4])))

    dlf, dlnf = last["lf"] - first["lf"], last["lnf"] - first["lnf"]
    frac = dlf / (dlf + dlnf) if dlf + dlnf > 0 else None
    checks.append(check("finger", frac is not None and frac >= MIN_FINGER_FRACTION,
                        f"live gate passed {dlf} of {dlf + dlnf} samples during the capture"
                        + (f" ({frac:.2%}, need >= {MIN_FINGER_FRACTION:.0%})" if frac is not None else "")))

    not_conn = [r["rep"] for r in pr if r["sampler"]["ble"] != 1 or r["sampler"]["sub"] != 1]
    conn_moved = smp[-1]["ble_conn"] != smp[0]["ble_conn"] or smp[-1]["ble_disc"] != smp[0]["ble_disc"]
    ok = not not_conn and not conn_moved and cap["ble_connected"] == 0 and cap["ble_disconnected"] == 0
    checks.append(check("central", ok,
                        f"ble=1 sub=1 on {len(pr) - len(not_conn)}/{len(pr)} lines; ble_conn {smp[0]['ble_conn']}->{smp[-1]['ble_conn']}, "
                        f"ble_disc {smp[0]['ble_disc']}->{smp[-1]['ble_disc']}; 'BLE connected to:' lines {cap['ble_connected']}, "
                        f"'BLE disconnected' lines {cap['ble_disconnected']}"))

    hz = cap["hz"]
    checks.append(check("rate", bool(hz) and min(hz) >= MIN_HZ,
                        f"Hz {min(hz):.2f}-{max(hz):.2f} over {len(hz)} lines (need every value >= {MIN_HZ})" if hz else "no Hz: lines"))

    ddec = last["dec"] - first["dec"]
    checks.append(check("decisions", ddec >= MIN_DECISIONS_DURING,
                        f"{ddec} decisions during the capture (need >= {MIN_DECISIONS_DURING})"))

    dn = smp[-1]["s_n"] - smp[0]["s_n"]
    dt = (last["t_ms"] - first["t_ms"]) / 1000
    expected = SAMPLER_HZ * dt
    checks.append(check("samples", dt > 0 and dn >= MIN_TICK_FRACTION * expected,
                        f"{dn:,} ticks serviced in {dt:.1f} s ({dn / expected:.2%} of {expected:,.0f}; need >= {MIN_TICK_FRACTION:.0%})"
                        if dt > 0 else "capture shorter than one report"))

    dskip = sum(smp[-1][k] - smp[0][k] for k in SKIP_KEYS)
    share = dskip / dn if dn > 0 else None
    checks.append(check("skips", share is not None and share <= MAX_SKIP_FRACTION,
                        f"{dskip} skipped of {dn} ticks"
                        + (f" ({share:.3%}, max {MAX_SKIP_FRACTION:.0%})" if share is not None else "")
                        + "; by cause " + ", ".join(f"{k} {smp[-1][k] - smp[0][k]}" for k in SKIP_KEYS)))

    order = [r["rep"] for r in pr if not r["sampler"]["s_win"] <= r["sampler"]["samp_max"] <= r["whole_hw"]]
    checks.append(check("bound_order", not order, "s_win <= samp_max <= whole_hw on every line" if not order
                        else f"violated at reps {order[:5]}"))

    stable = [r for r in pr if r["sampler"]["hl_st"] == 1]
    mism = [r["rep"] for r in stable if r["sampler"]["hl"] != r["hs_cur"] + r["hs_ovh"]]
    checks.append(check("heap_source", not mism and 2 * len(stable) >= len(pr),
                        f"hl == hs_cur + hs_ovh on {len(stable) - len(mism)}/{len(stable)} stable lines; "
                        f"{len(stable)}/{len(pr)} lines stable" + (f"; mismatch at reps {mism[:5]}" if mism else "")))
    return checks


def summarise_sampler(cap):
    pr = cap["pr"]
    if not pr or any("sampler" not in r for r in pr):
        return None
    smp = [r["sampler"] for r in pr]
    last, s = pr[-1], smp[-1]
    wins = [x["s_win"] for x in smp[1:]]
    info = next((i["sampler"] for i in reversed(cap["info"]) if "sampler" in i), None)
    return {
        "lower_capture": max(wins) if wins else None,
        "samp_max": s["samp_max"], "whole_hw": last["whole_hw"],
        "samp_max_at_first_line": smp[0]["samp_max"],
        "samp_max_set_during_capture": s["samp_max"] > smp[0]["samp_max"],
        "breakdown_at_samp_max": {"s_const": None if info is None else info["s_const"], "heap": s["sm_heap"],
                                  "msp": s["sm_msp"], "stacks": s["sm_stk"],
                                  "threads": {k: v["at_max"] for k, v in s["threads"].items()}},
        "component_maxima": {"heap": s["smx_heap"], "msp": s["smx_msp"],
                             "threads": {k: v["max"] for k, v in s["threads"].items()}},
        "ticks": s["s_n"], "samples": s["s_ok"], "skips": {k: s[k] for k in SKIP_KEYS},
        "cycles": {"max": s["s_cyc_max"], "mean": s["s_cyc_sum"] / s["s_n"] if s["s_n"] else None},
        "ble": {"connected_lines": sum(x["ble"] for x in smp), "subscribed_lines": sum(x["sub"] for x in smp),
                "ble_conn": s["ble_conn"], "ble_disc": s["ble_disc"]},
        "s_win_by_line": [x["s_win"] for x in smp],
        "info": info,
    }


def summarise(cap):
    pr = cap["pr"]
    if not pr:
        return None
    last = pr[-1]
    regions = {}
    for k in PR_REGION_KEYS:
        regions[k] = dict(last[k], max_peak=max(r[k]["peak"] for r in pr))
    threads = {}
    for k, v in last["threads"].items():
        threads[k] = dict(v, max_peak=max(r["threads"][k]["peak"] for r in pr if k in r["threads"]))
    info = cap["info"][-1] if cap["info"] else None
    return {
        "final_line": {"t": last["t"], "t_ms": last["t_ms"], "rep": last["rep"], "dec": last["dec"]},
        "whole_hw": {"final": last["whole_hw"], "max": max(r["whole_hw"] for r in pr),
                     "ram": last["ram"], "share": last["whole_hw"] / last["ram"],
                     "untouched_final": last["untouched"]},
        "static_data_bss": last["static"],
        "fixed_nvic_crash": last["fixed"],
        "regions": regions,
        "threads": threads,
        "main_phases": {"pre_hook": last["main_pre"], "hook_to_setup_end": last["main_setup"],
                        "loop": last["main_loop"]},
        "msp_loop_peak": last["msp_loop"],
        "heap_stats": {"current": last["hs_cur"], "max": last["hs_max"], "overhead": last["hs_ovh"],
                       "failed_allocations": last["hs_fail"]},
        "heap_break": hex(last["brk"]), "heap_hw": hex(last["heap_hw"]),
        "scan_us_max": max(r["scan_us"] for r in pr),
        "rpt_alloc_max": max(r["rpt_alloc"] for r in pr),
        "live_gate": {"finger": last["lf"], "no_finger": last["lnf"]},
        "info": None if info is None else {k: info[k] for k in INFO_STR_KEYS + INFO_INT_KEYS},
        "info_spans": None if info is None else {k: [hex(info[k][0]), info[k][1]] for k in INFO_SPAN_KEYS},
        "info_threads": None if info is None else {k: dict(v, base=hex(v["base"])) for k, v in info["threads"].items()},
        "hz_range": [min(cap["hz"]), max(cap["hz"])] if cap["hz"] else None,
    }


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upload_binding(image, bossac_log):
    """Bind the capture to a verified bossac write of an archived image.

    Passes when the bossac log says 'Verify successful', the bytes written equal the image
    size, and the image matches the SHA256SUMS next to it (the probe archive's own list).
    """
    image, bossac_log = Path(image), Path(bossac_log)
    sha = sha256_file(image)
    log = bossac_log.read_text(errors="replace")
    written = re.search(r"Write (\d+) bytes to flash", log)
    sums = image.parent / "SHA256SUMS"
    listed = None
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip().lstrip("*") == image.name:
                listed = parts[0].lower()
    out = {"image": str(image), "image_sha256": sha, "image_bytes": image.stat().st_size,
           "archive_sha256sums": str(sums) if sums.exists() else None, "archive_listed_sha256": listed,
           "bossac_log": str(bossac_log), "bossac_log_sha256": sha256_file(bossac_log),
           "verify_successful": "Verify successful" in log,
           "bytes_written": int(written.group(1)) if written else None}
    out["ok"] = (out["verify_successful"] and out["bytes_written"] == out["image_bytes"]
                 and listed == sha)
    return out


def analyse(lines, expect_image=None, upload=None, sampler=False):
    cap = parse_capture(lines)
    checks = validate_sampler(cap, expect_image) if sampler else validate(cap, expect_image)
    if upload is not None:
        checks.append(check("upload", upload["ok"],
                            f"bossac 'Verify successful' {upload['verify_successful']}, wrote {upload['bytes_written']} "
                            f"of {upload['image_bytes']} B, image {upload['image_sha256'][:8]}... "
                            f"{'matches' if upload['archive_listed_sha256'] == upload['image_sha256'] else 'does NOT match'} "
                            "its archive SHA256SUMS"))
    summary = summarise(cap)
    valid = all(c["ok"] for c in checks)
    figure = None
    if sampler:
        ssum = summarise_sampler(cap)
        if valid:
            lo, hi = ssum["lower_capture"], ssum["whole_hw"]
            figure = {"lower_bytes": lo, "upper_bytes": hi, "ram_bytes": RAM_BYTES, "lower_share": lo / RAM_BYTES,
                      "upper_share": hi / RAM_BYTES, "samp_max_bytes": ssum["samp_max"],
                      "wording": SAMPLER_WORDING.format(lo=lo, hi=hi), "image": summary["info"]["img"]}
        return {"valid": valid, "mode": "sampler", "checks": checks, "summary": summary, "sampler": ssum,
                "paper_figure": figure, "pr_lines": len(cap["pr"]), "pr_info_lines": len(cap["info"]),
                "parse_errors": cap["errors"], "upload": upload}
    if valid:
        figure = {"whole_hw_bytes": summary["whole_hw"]["final"], "ram_bytes": summary["whole_hw"]["ram"],
                  "share": summary["whole_hw"]["share"], "wording": PAPER_WORDING,
                  "image": summary["info"]["img"]}
    return {"valid": valid, "checks": checks, "summary": summary, "paper_figure": figure,
            "pr_lines": len(cap["pr"]), "pr_info_lines": len(cap["info"]), "parse_errors": cap["errors"],
            "upload": upload}


def markdown(report, capture_name):
    out = [f"# Peak-RAM probe capture: `{capture_name}`", ""]
    out.append(f"**Valid: {'yes' if report['valid'] else 'NO'}.** "
               f"{report['pr_lines']} `PR:` lines, {report['pr_info_lines']} `PR_INFO:` lines.")
    out += ["", "| Check | Result | Detail |", "|---|---|---|"]
    for c in report["checks"]:
        out.append(f"| {c['name']} | {'pass' if c['ok'] else 'FAIL'} | {c['detail']} |")
    s = report["summary"]
    if s is None:
        out += ["", "No `PR:` lines; nothing to report."]
        return "\n".join(out) + "\n"
    w = s["whole_hw"]
    out += ["", "## Whole device", "",
            f"- `whole_hw` (final `PR:` line): **{w['final']:,} B** of {w['ram']:,} B ({w['share']:.1%}); "
            f"max over lines {w['max']:,} B.",
            f"- Static `.data` + `.bss` (this probe image): {s['static_data_bss']:,} B; NVIC + crash RAM {s['fixed_nvic_crash']:,} B.",
            f"- Never touched (union): {w['untouched_final']:,} B."]
    if report.get("mode") == "sampler":
        z = report["sampler"]
        if z is not None and z["cycles"]["mean"] is not None:
            out.append(f"- Sampler: largest `s_win` over the capture after the first line **{z['lower_capture']:,} B**; "
                       f"`samp_max` {z['samp_max']:,} B (set during the capture: {'yes' if z['samp_max_set_during_capture'] else 'no'}); "
                       f"ticks {z['ticks']:,}, samples {z['samples']:,}, skips {z['skips']}; body cycles max "
                       f"{z['cycles']['max']}, mean {z['cycles']['mean']:.0f}.")
        diag = ((z or {}).get("info") or {}).get("diag")
        if diag:
            out.append("- Sampler start diagnostics (last `PR_INFO:`): " + " ".join(f"`{k}={v}`" for k, v in diag.items()) + ".")
        if report["paper_figure"]:
            f = report["paper_figure"]
            out.append(f"- Paper: *\"{f['wording']}\"* ({f['lower_share']:.1%}-{f['upper_share']:.1%} of 256 KB).")
        else:
            out.append("- No paper figure: the run fails a validity check.")
    elif report["paper_figure"]:
        f = report["paper_figure"]
        out.append(f"- Paper: {f['whole_hw_bytes']:,} B ({f['share']:.1%} of 256 KB), *{f['wording']}*.")
    else:
        out.append("- No paper figure: the run fails a validity check.")
    out += ["", "## Regions (final line)", "", "| Region | Reserved | Peak | Headroom | Where |", "|---|---:|---:|---:|---|"]
    for k, r in s["regions"].items():
        out.append(f"| {k} | {r['reserved']:,} | {r['peak']:,} | {r['headroom']:,} | |")
    for k, r in s["threads"].items():
        out.append(f"| {k} | {r['reserved']:,} | {r['peak']:,} | {r['headroom']:,} | {r.get('loc', '')} |")
    m = s["main_phases"]
    out += ["", "Thread rows overlap the heap row (heap stacks) and the static RAM (`.bss` stacks), so the rows are "
            "not summed; `whole_hw` is built from never-touched address ranges instead.", "",
            f"Main-thread peak by phase: before the hook {m['pre_hook']:,} B, hook to end of `setup()` "
            f"{m['hook_to_setup_end']:,} B, loop {m['loop']:,} B. MSP loop-phase peak {s['msp_loop_peak']:,} B.",
            "",
            f"Mbed heap stats: current {s['heap_stats']['current']:,} B, max {s['heap_stats']['max']:,} B, "
            f"overhead {s['heap_stats']['overhead']:,} B, failed allocations {s['heap_stats']['failed_allocations']}. "
            f"Break {s['heap_break']}, heap high-water {s['heap_hw']}.",
            f"Scan cost max {s['scan_us_max']} us; bytes allocated during a report, max {s['rpt_alloc_max']}.",
            f"Live gate verdicts: finger {s['live_gate']['finger']}, no finger {s['live_gate']['no_finger']}.",
            f"`Hz:` range {s['hz_range']}." if s["hz_range"] else "No `Hz:` lines.",
            "", "Values on the `KAN_Prob`/`Candidate`/`Confirmed` lines of a probe capture are dark-floor "
            "forced decisions, not results."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", type=Path)
    ap.add_argument("--output", type=Path, required=True, help="fresh directory for report.json and report.md")
    ap.add_argument("--expect-image", choices=IMAGES + SAMPLER_IMAGES, help="env name the capture must identify as")
    ap.add_argument("--sampler", action="store_true", help="sampler image: apply the 30 Oct pre-registered checks")
    ap.add_argument("--image", type=Path, help="archived firmware.bin that was flashed (needs --bossac-log)")
    ap.add_argument("--bossac-log", type=Path, help="bossac log of that flash (needs --image)")
    args = ap.parse_args(argv)
    if (args.image is None) != (args.bossac_log is None):
        ap.error("--image and --bossac-log go together")
    if args.output.exists():
        print(f"refusing to overwrite existing {args.output}", file=sys.stderr)
        return 2
    data = args.capture.read_bytes()
    lines = data.decode("utf-8", "replace").splitlines()
    upload = upload_binding(args.image, args.bossac_log) if args.image else None
    report = analyse(lines, args.expect_image, upload, sampler=args.sampler)
    report["capture"] = {"path": str(args.capture), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if args.sampler:
        report["criteria"] = {"min_pr_lines": MIN_PR_LINES, "max_gap_s": MAX_GAP_S, "sampler_hz": SAMPLER_HZ,
                              "min_finger_fraction": MIN_FINGER_FRACTION, "min_decisions_during": MIN_DECISIONS_DURING,
                              "min_hz": MIN_HZ, "min_tick_fraction": MIN_TICK_FRACTION,
                              "max_skip_fraction": MAX_SKIP_FRACTION}
    else:
        report["criteria"] = {"min_pr_lines": MIN_PR_LINES, "min_final_decisions": MIN_FINAL_DECISIONS,
                              "min_decisions_in_capture": MIN_DECISIONS_IN_CAPTURE, "max_gap_s": MAX_GAP_S,
                              "max_live_finger_fraction": MAX_LIVE_FINGER_FRACTION}
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report, args.capture.name), encoding="utf-8")
    print(f"valid={report['valid']} pr_lines={report['pr_lines']}", end="")
    if report["summary"]:
        print(f" whole_hw={report['summary']['whole_hw']['final']}", end="")
    print(f" -> {args.output}")
    for c in report["checks"]:
        if not c["ok"]:
            print(f"  FAIL {c['name']}: {c['detail']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

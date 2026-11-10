"""Derived peak-RAM bracket for a peak-RAM probe capture. Host only: reads files, opens no port.

Whole-device peak RAM was not measured on 25 Oct 2026. The probe measured whole_hw, the
RAM high-water: RAM minus the address ranges never touched during the run. That is a union
over time, so it bounds the peak from above. This script derives a lower bound from the same
capture (docs/peak-ram-design.md, section "Peak versus high-water").

In use at an instant t (the quantity whose maximum is the peak):
    in_use(t) = fixed + static - sum_bss(SP_i - base_i)
                      + heap_live(t) - sum_heap(SP_i - base_i) + (StackTop - MSP(t))
  fixed      NVIC + crash RAM below .data
  static     .data + .bss; each .bss thread stack counts only from its SP up to its top
  heap_live  allocated chunks, header and padding included (= hs_cur + hs_ovh: every
             allocation goes through Mbed's stats wrapper, which counts chunk size - request
             as overhead); each heap thread stack counts only from its SP up to its top
  MSP        from the current MSP up to __StackTop
The whole_hw arithmetic is the same with SP_i replaced by base_i + never-touched bytes, heap_live
by heap_hw - __end__, and MSP by the lowest touched MSP word. Hence, for every t,
    whole_hw - in_use(t) = (heap_peak - heap_live(t)) + sum_i (P_i - depth_i(t)) + (msp_peak - msp_depth(t)).

Upper bound: in_use(t) <= whole_hw (conditions in the design doc). Lower bound: at an instant
t* when the main thread's deepest touched word was written, depth_main(t*) >= P_main; every
other stack and the MSP are >= 0; heap_live(t*) >= H_a, the tiled heap level at the last report
before the main peak first appears, under one assumption: no heap block was freed and
re-allocated between two consecutive reports. So
    lower = fixed + static - sum_bss size_i + H_a - sum_heap size_i + P_main.

Usage (defaults are the two 25 Oct captures):
    python scripts/peak_ram_bracket.py --output outputs/peak-ram-bracket-v1
Writes <output>/summary.json and <output>/README.md. Refuses an existing output directory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_peak_ram as apr  # noqa: E402

CAPTURES = {
    "v1": ("nano33ble_peakram", "outputs/peak-ram-board-20261025/nano33ble_peakram_nofinger_180s.txt"),
    "v2": ("nano33ble_v2_f1a_peakram", "outputs/peak-ram-board-20261025/nano33ble_v2_f1a_peakram_nofinger_180s.txt"),
}
# The analyser checks that apply to a capture on its own (the flash binding needs the flash
# log and is already on file in each committed report).
CAPTURE_CHECKS = ["identity", "painted", "parse", "pr_lines", "gaps", "no_reboot", "decisions",
                  "regions", "monotonic", "no_finger", "no_central"]
ASSUMPTION = ("No heap block was freed and re-allocated between two consecutive PR: reports "
              "(the reports show the arena below the break fully allocated at every report; "
              "they do not show what happened between them).")
WORDING = ("whole-device peak RAM is not measured; for the 25 Oct probe run (probe image, no finger, "
           "forced decisions, no central) a derived bracket puts it between {lo:,} B and {hi:,} B; "
           "the upper end is the RAM high-water")


class BracketError(ValueError):
    pass


def main_key(threads):
    keys = [k for k in threads if k.startswith("T.main@")]
    if len(keys) != 1:
        raise BracketError(f"expected one T.main thread, found {keys}")
    return keys[0]


def heap_live(rec):
    """Allocated chunk bytes (request + overhead) at the report."""
    return rec["hs_cur"] + rec["hs_ovh"]


def line_facts(rec, heap_base):
    span = rec["brk"] - heap_base
    return {"rep": rec["rep"], "t_ms": rec["t_ms"], "brk": rec["brk"], "heap_hw": rec["heap_hw"],
            "brk_span": span, "heap_live": heap_live(rec),
            "tiled": heap_live(rec) == span and rec["heap_hw"] == rec["brk"],
            "cur_eq_max": rec["hs_cur"] == rec["hs_max"]}


def bracket(cap):
    """Bracket for one parsed capture (analyze_peak_ram.parse_capture output)."""
    pr, info = cap["pr"], cap["info"]
    if not pr or not info:
        raise BracketError("no PR: or PR_INFO: lines")
    heap_base = info[-1]["heap"][0]
    if any(i["heap"][0] != heap_base for i in info):
        raise BracketError("PR_INFO heap base changes")
    last = pr[-1]
    mk = main_key(last["threads"])
    facts = [line_facts(r, heap_base) for r in pr]
    problems = []
    if not all(f["tiled"] for f in facts):
        bad = [f["rep"] for f in facts if not f["tiled"]]
        problems.append(f"arena below the break not fully allocated (or heap_hw != brk) at reps {bad}")
    if any(b["brk"] < a["brk"] for a, b in zip(facts, facts[1:])):
        problems.append("break fell between reports")
    if last["hs_fail"] != 0:
        problems.append(f"hs_fail {last['hs_fail']}")

    threads = last["threads"]
    bss = {k: v for k, v in threads.items() if v.get("loc") == "bss"}
    heap = {k: v for k, v in threads.items() if v.get("loc") == "heap"}
    if len(bss) + len(heap) != len(threads):
        problems.append("a thread stack is neither in .bss nor in the heap")
    p_main = threads[mk]["peak"]
    if p_main != last["main_loop"] or p_main <= max(last["main_pre"], last["main_setup"]):
        problems.append("main-thread peak is not from the loop phase")

    # First report where the loop-phase main peak reached its final value, and the one before it.
    b = next((i for i, r in enumerate(pr) if r["main_loop"] == p_main), None)
    a = -1 if b is None else b - 1
    if b is not None and a < 0:
        problems.append("main peak already reached at the first captured report: no report before it")

    upper = last["whole_hw"]
    heap_peak = last["heap"]["peak"]
    msp_peak = last["msp"]["peak"]
    sum_bss = sum(v["reserved"] for v in bss.values())
    sum_heap = sum(v["reserved"] for v in heap.values())
    others = {k: v["peak"] for k, v in threads.items() if k != mk}
    out = {
        "upper_bytes": upper,
        "upper_share": upper / last["ram"],
        "inputs": {
            "final_rep": last["rep"], "final_t_ms": last["t_ms"],
            "fixed": last["fixed"], "static": last["static"],
            "bss_stacks": {k: v["reserved"] for k, v in bss.items()},
            "heap_stacks": {k: v["reserved"] for k, v in heap.items()},
            "heap_base": hex(heap_base), "heap_peak": heap_peak, "msp_peak": msp_peak,
            "main_peak": p_main, "other_stack_peaks": others,
        },
        "heap_levels": [],
        "problems": problems,
    }
    # Distinct heap levels with the reports that show them.
    for f in facts:
        lv = out["heap_levels"]
        if lv and lv[-1]["heap_live"] == f["heap_live"] and lv[-1]["brk"] == hex(f["brk"]):
            lv[-1]["last_rep"], lv[-1]["last_t_ms"] = f["rep"], f["t_ms"]
        else:
            lv.append({"heap_live": f["heap_live"], "brk": hex(f["brk"]), "first_rep": f["rep"],
                       "first_t_ms": f["t_ms"], "last_rep": f["rep"], "last_t_ms": f["t_ms"]})
    out["tiled_lines"] = sum(f["tiled"] for f in facts)
    out["cur_eq_max_lines"] = sum(f["cur_eq_max"] for f in facts)
    out["lines"] = len(facts)

    # The arithmetic the brief proposed (subtract every non-main stack peak and the MSP peak
    # from whole_hw, heap unchanged), for comparison only.
    out["naive_lower_bytes"] = upper - sum(others.values()) - msp_peak

    if a >= 0:
        h_a = facts[a]["heap_live"]
        lower = last["fixed"] + last["static"] - sum_bss + h_a - sum_heap + p_main
        # Same number from the other side of the identity above.
        lower_check = upper - (heap_peak - h_a) - sum(others.values()) - msp_peak
        if lower != lower_check:
            problems.append(f"lower bound arithmetic disagrees: {lower} vs {lower_check}")
        out["main_peak_window"] = {
            "before": {"rep": pr[a]["rep"], "t_ms": pr[a]["t_ms"], "main_loop": pr[a]["main_loop"]},
            "first_at_peak": {"rep": pr[b]["rep"], "t_ms": pr[b]["t_ms"], "main_loop": pr[b]["main_loop"]},
            "heap_live_before": h_a, "heap_live_first_at_peak": facts[b]["heap_live"],
        }
        out["heap_term"] = h_a
        out["lower_bytes"] = lower
        out["lower_share"] = lower / last["ram"]
        out["width_bytes"] = upper - lower
        out["lower_terms"] = {"fixed": last["fixed"], "static_minus_bss_stacks": last["static"] - sum_bss,
                              "heap_live_minus_heap_stacks": h_a - sum_heap, "main_stack_depth": p_main,
                              "other_stacks": 0, "msp": 0}
    if problems:
        out["lower_bytes"] = None
        out["lower_share"] = None
        out["width_bytes"] = None
    return out


def run(root, captures=CAPTURES):
    models = {}
    for model, (image, rel) in captures.items():
        path = Path(root) / rel
        data = path.read_bytes()
        cap = apr.parse_capture(data.decode("utf-8", "replace").splitlines())
        checks = [c for c in apr.validate(cap, image) if c["name"] in CAPTURE_CHECKS]
        res = bracket(cap)
        if not all(c["ok"] for c in checks):
            res["problems"].append("capture fails an analyser check: "
                                   + ", ".join(c["name"] for c in checks if not c["ok"]))
            res["lower_bytes"] = res["lower_share"] = res["width_bytes"] = None
        res["image"] = image
        res["capture"] = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        res["analyser_checks"] = {c["name"]: c["ok"] for c in checks}
        if res["lower_bytes"] is not None:
            res["wording"] = WORDING.format(lo=res["lower_bytes"], hi=res["upper_bytes"])
        models[model] = res
    return {"what": "derived peak-RAM bracket from the 25 Oct 2026 probe captures (host only)",
            "measured": False,
            "assumption": ASSUMPTION,
            "method": "docs/peak-ram-design.md, section 'Peak versus high-water'",
            "models": models}


def readme(summary):
    m = summary["models"]
    out = ["# Derived peak-RAM bracket, 25 Oct probe run (host only)", "",
           "Written by `scripts/peak_ram_bracket.py` from the two committed captures in "
           "`outputs/peak-ram-board-20261025/` (read only, through `scripts/analyze_peak_ram.py`'s parser). "
           "No board, no serial port. Method and proofs: [docs/peak-ram-design.md](../../docs/peak-ram-design.md), "
           "section *Peak versus high-water*.", "",
           "**Whole-device peak RAM is not measured.** The probe measured the RAM high-water (`whole_hw`), "
           "a union of touched addresses over the run. That is the upper end below. The lower end is derived, "
           "not measured.", "",
           "| | v1 probe (`nano33ble_peakram`) | v2 probe (`nano33ble_v2_f1a_peakram`) |", "|---|---:|---:|"]

    def row(label, fn):
        out.append(f"| {label} | {fn(m['v1'])} | {fn(m['v2'])} |")

    row("Lower end (derived)", lambda r: f"**{r['lower_bytes']:,} B** ({r['lower_share']:.1%})" if r["lower_bytes"] else "not published")
    row("Upper end = RAM high-water `whole_hw`", lambda r: f"**{r['upper_bytes']:,} B** ({r['upper_share']:.1%})")
    row("Width", lambda r: f"{r['width_bytes']:,} B" if r["width_bytes"] else "—")
    row("Main-thread stack peak (loop phase)", lambda r: f"{r['inputs']['main_peak']:,} B")
    row("Heap in use at the main peak (term used)", lambda r: f"{r['heap_term']:,} B")
    row("Heap peak (break − `__end__`)", lambda r: f"{r['inputs']['heap_peak']:,} B")
    row("Other stack peaks dropped (sum)", lambda r: f"{sum(r['inputs']['other_stack_peaks'].values()):,} B")
    row("MSP peak dropped", lambda r: f"{r['inputs']['msp_peak']:,} B")
    row("Lines with the arena below the break fully allocated", lambda r: f"{r['tiled_lines']} / {r['lines']}")
    row("Lines with `hs_cur == hs_max`", lambda r: f"{r['cur_eq_max_lines']} / {r['lines']}")
    row("Main peak first seen (rep, board ms)", lambda r: f"rep {r['main_peak_window']['first_at_peak']['rep']}, {r['main_peak_window']['first_at_peak']['t_ms']:,}")
    out += ["",
            "Lower end = NVIC/crash RAM + static RAM outside the three `.bss` thread stacks + heap chunks in use "
            "outside the two heap thread stacks + the main-thread stack peak; every other stack and the MSP are "
            "taken as 0. Equivalently `whole_hw` − (heap peak − heap term) − (other stack peaks) − (MSP peak).", "",
            f"**Assumption (supported by the reports, not proved by them).** {summary['assumption']}", "",
            "**Heap term.** On every line of both captures `hs_cur + hs_ovh` (allocated chunk bytes) equals "
            "`brk − __end__` and `heap_hw == brk`, so the arena below the break was fully allocated at each report. "
            "v1's heap steps once by 88 B of chunk (56 B requested + 32 B overhead) between "
            f"rep {m['v1']['heap_levels'][0]['last_rep']} and rep {m['v1']['heap_levels'][1]['first_rep']}; "
            f"its main-thread peak was first seen at rep {m['v1']['main_peak_window']['first_at_peak']['rep']}, "
            "before that step, so the lower end uses the pre-step level "
            f"{m['v1']['heap_term']:,} B. Subtracting only the stack and MSP peaks from `whole_hw` would give "
            f"{m['v1']['naive_lower_bytes']:,} B for v1, {m['v1']['naive_lower_bytes'] - m['v1']['lower_bytes']} B "
            "higher, because it keeps the post-step heap. "
            f"v2's heap is {m['v2']['heap_term']:,} B on all lines.", "",
            "**Upper end.** `in_use(t) <= whole_hw` for every instant from the probe hook to the last report, "
            "provided no stack pointer ran below its stack's deepest written word (a frame allocated but its "
            "bottom never written) and no watermark edge collided with its fill pattern; MSP depth before the "
            "hook is outside the window. The 25 Oct data cannot test either condition.", "",
            "**Not used.** Every thread that is not running holds at least a 64 B saved context on its own stack "
            "(RTX `SVC_ContextSave`: r4–r11 below the 32 B hardware frame), and the thread-mode MSP sits 64 B below "
            "`__StackTop` (PR_INFO `msp_paint`). Counting those would raise the lower end by 320 B; it is left out "
            "to keep the bound on the fewest premises.", "",
            "**Wording for the paper and report** (per model):", ""]
    for k in ("v1", "v2"):
        if m[k].get("wording"):
            out.append(f"- {k}: *\"{m[k]['wording']}\"*.")
    out += ["",
            "Not covered: the unmodified live images, finger-on decisions, a connected Bluetooth central, other "
            "workloads, nested-interrupt worst cases. The values on `KAN_Prob`/`Candidate`/`Confirmed` lines of "
            "these captures are dark-floor forced decisions and are not results.", "",
            "| File | Content |", "|---|---|",
            "| `summary.json` | every input, heap level, the main-peak window and both bounds per model |", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True, help="fresh directory for summary.json and README.md")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite existing {args.output}", file=sys.stderr)
        return 2
    summary = run(args.root)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(readme(summary), encoding="utf-8")
    for k, r in summary["models"].items():
        print(f"{k}: lower={r['lower_bytes']} upper={r['upper_bytes']} problems={r['problems']}")
    return 0 if all(r["lower_bytes"] is not None for r in summary["models"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

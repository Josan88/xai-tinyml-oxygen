"""Host model of the peak-RAM probe's paint and scan (firmware/Arduino Nano 33/include/
peak_ram_probe.h). Used by tests/test_peak_ram.py on synthetic RAM; it never touches a board.

Each function mirrors one firmware function, word for word:
    heap_top          pr_heap_top
    msp_low           pr_msp_low
    rtx_stack_space   RTX svcRtxThreadGetStackSpace (rtx_thread.c)
    union_bytes       pr_union
    format_pr_line    pr_report's output format (the fields the analyser reads)
"""
from array import array

RAM_START = 0x20000000
RAM_SIZE = 0x40000
PATTERN = 0x9E3779B9      # PR_PATTERN
RTX_FILL = 0xCCCCCCCC     # osRtxStackFillPattern
RTX_MAGIC = 0xE25A2EA5    # osRtxStackMagicWord
SP_MARGIN = 256           # PR_SP_MARGIN


class Ram:
    """Word-addressed SRAM image, RAM_START .. RAM_START + RAM_SIZE."""

    def __init__(self, fill=0):
        self.words = array("I", [fill]) * (RAM_SIZE // 4)

    def _i(self, addr):
        if addr % 4 or not RAM_START <= addr < RAM_START + RAM_SIZE:
            raise IndexError(hex(addr))
        return (addr - RAM_START) // 4

    def get(self, addr):
        return self.words[self._i(addr)]

    def put(self, addr, value):
        self.words[self._i(addr)] = value & 0xFFFFFFFF

    def fill(self, lo, hi, value):
        for a in range(lo, hi, 4):
            self.put(a, value)

    def touch(self, lo, hi, seed=1):
        """Write data that is neither pattern into [lo, hi)."""
        for k, a in enumerate(range(lo, hi, 4)):
            self.put(a, (seed * 2654435761 + k * 40503) & 0x7FFFFFFF | 1)


def heap_top(ram, paint_lo, limit, brk_max, painted=True):
    top = limit
    if painted:
        top = paint_lo
        a = limit - 4
        while a >= paint_lo:
            if ram.get(a) != PATTERN:
                top = a + 4
                break
            a -= 4
    if brk_max > top:
        top = brk_max
    return top if top < limit else limit


def msp_low(ram, stack_limit, hi):
    for a in range(stack_limit, hi, 4):
        if ram.get(a) != PATTERN:
            return a
    return hi


def rtx_stack_space(ram, base, size):
    if ram.get(base) != RTX_MAGIC:
        return 0
    space = 4
    while space < size:
        if ram.get(base + space) != RTX_FILL:
            break
        space += 4
    return space


def rtx_create_stack(ram, base, size):
    """What RTX does at osThreadNew with watermarking on: magic word, then the fill."""
    ram.put(base, RTX_MAGIC)
    ram.fill(base + 4, base + size, RTX_FILL)


def union_bytes(spans):
    """(bytes in the union, plain sum of lengths), spans clamped to RAM as in pr_union."""
    clamped = []
    listed = 0
    for lo, hi in spans:
        lo = max(lo, RAM_START)
        hi = min(hi, RAM_START + RAM_SIZE)
        hi = max(hi, lo)
        listed += hi - lo
        clamped.append((lo, hi))
    clamped.sort()
    total, cur = 0, None
    for lo, hi in clamped:
        if hi == lo:
            continue
        if cur is None:
            cur = [lo, hi]
        elif lo <= cur[1]:
            cur[1] = max(cur[1], hi)
        else:
            total += cur[1] - cur[0]
            cur = [lo, hi]
    if cur is not None:
        total += cur[1] - cur[0]
    return total, listed


def format_pr_line(rec):
    """A PR: line from a dict holding the firmware's fields, in pr_report's key order."""
    def rph(r):
        return f"{r[0]}/{r[1]}/{r[0] - r[1]}"
    parts = ["PR: v=1"]
    for k in ("t_ms", "rep", "dec", "lf", "lnf"):
        parts.append(f"{k}={rec[k]}")
    parts.append(f"heap={rph(rec['heap'])}")
    parts.append(f"msp={rph(rec['msp'])}")
    for k in ("msp_loop", "main_pre", "main_setup", "main_loop"):
        parts.append(f"{k}={rec[k]}")
    parts.append(f"nthr={len(rec['threads'])}")
    for name, tid, size, peak, loc in rec["threads"]:
        parts.append(f"T.{name}@0x{tid:08X}={size}/{peak}/{size - peak}/{loc}")
    parts.append(f"brk=0x{rec['brk']:08X}")
    parts.append(f"heap_hw=0x{rec['heap_hw']:08X}")
    for k in ("hs_cur", "hs_max", "hs_ovh", "hs_fail", "static", "fixed", "untouched", "ovl",
              "whole_hw", "ram", "scan_us", "rpt_alloc"):
        parts.append(f"{k}={rec[k]}")
    return " ".join(parts)

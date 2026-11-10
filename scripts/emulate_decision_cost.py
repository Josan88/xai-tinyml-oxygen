"""Host probe: count instructions the COMPILED v2 decision block executes, per input. No board.

Runs the real `BasicOxygenRiskPipeline<250>::evaluate_window()` machine code from a firmware ELF in
the Unicorn emulator (Cortex-M4, Thumb-2). The image is built -mfloat-abi=soft, so the decision
block is integer code calling libgcc/newlib soft-float routines; no FPU emulation is involved.

What it shows: executed instruction counts per window and per callee (soft-float routine,
filter, features, KAN), for the same fixture windows the timing replay used and for
transformed copies of them at the live sensor's scale. Instruction counts are not cycles: flash
wait states, branch refills and multi-cycle instructions are not modelled, and nothing about
interrupts or other threads is modelled. A cycles-per-instruction ratio is reported by dividing the
board's measured timing-image cycles by the emulated count for the same fixture windows.

Requires: unicorn, capstone, pyelftools, numpy (a venv at build/looptiming/emu-venv was used).
Object layout (offsets below) was taken from arm-none-eabi-g++ offsetof on the same header.
"""
import argparse
import bisect
import json
import struct
import sys
from pathlib import Path

import numpy as np
from capstone import CS_ARCH_ARM, CS_MODE_MCLASS, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile
from unicorn import UC_ARCH_ARM, UC_HOOK_BLOCK, UC_MODE_MCLASS, UC_MODE_THUMB, Uc
from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_SP

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELF = ROOT / "outputs/firmware-v2-looptiming/nano33ble_v2_looptiming/firmware.elf"
FIXTURES = ROOT / "outputs/parity/v2-board-prepared/fixtures.npz"
BOARD_TIMING = ROOT / "outputs/timing-v2-board-20261015/report.json"

# BasicOxygenRiskPipeline<250> with OXY_MODEL_V2, arm-none-eabi-g++ 7.2.1 (offsetof).
OFF = dict(raw_ring=0, filtered=4000, aligned=6000, temp=10000, raw_features=14216, hann=14280,
           red_ring=15304, red_filtered=19304, ir_raw_mean=21304, red_raw_mean=21308,
           two_wl=21312, write_idx=21316, total=21320, since=21328, window_index=21332,
           consecutive=21336, primed=21337, threshold=21340, required=21344, size=21352)
DEC = dict(size=152, risk_probability=36, raw_features=80, two_wl=144)
BUFFER = 1000

THIS, SRET, STACK_TOP, STOP = 0x30000000, 0x30010000, 0x30080000, 0x3FFFF000


class Emulator:
    def __init__(self, elf_path):
        self.elf_path = Path(elf_path)
        with open(self.elf_path, "rb") as f:
            elf = ELFFile(f)
            self.segments = [(s["p_vaddr"], s.data(), s["p_memsz"]) for s in elf.iter_segments()
                             if s["p_type"] == "PT_LOAD"]
            symtab = elf.get_section_by_name(".symtab")
            funcs = [(s["st_value"] & ~1, s["st_size"], s.name) for s in symtab.iter_symbols()
                     if s["st_info"]["type"] == "STT_FUNC" and s["st_size"] > 0]
            self.objects = {s.name: s["st_value"] for s in symtab.iter_symbols()
                            if s["st_info"]["type"] == "STT_OBJECT"}
        funcs.sort()
        self.func_starts = [a for a, _, _ in funcs]
        self.funcs = funcs
        self.by_name = {}
        for a, size, name in funcs:
            self.by_name.setdefault(name, a)
        self.cs = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
        self.block_cache = {}
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        self.uc.mem_map(0x0, 0x100000)            # flash
        self.uc.mem_map(0x20000000, 0x40000)      # SRAM
        self.uc.mem_map(0x30000000, 0x80000)      # object, result, stack
        self.uc.mem_map(STOP, 0x1000)
        self.uc.mem_map(0xE0000000, 0x100000)    # PPB: DWT->CYCCNT reads as 0
        self.uc.mem_map(0x40000000, 0x100000)    # peripherals (not touched by the probed code)
        for vaddr, data, memsz in self.segments:
            self.uc.mem_write(vaddr, data)
            if memsz > len(data):
                self.uc.mem_write(vaddr + len(data), b"\0" * (memsz - len(data)))
        self.uc.hook_add(UC_HOOK_BLOCK, self._on_block)
        self.counts = {}
        self.total = 0
        self.markers = []
        self.marker_addr = None

    def mark_calls_to(self, name):
        """Replace `name` with `return 0` and record the instruction count at each call."""
        addr = self.sym(name)
        self.uc.mem_write(addr, bytes([0x00, 0x20, 0x70, 0x47]))  # movs r0, #0; bx lr
        self.marker_addr = addr

    def sym(self, needle):
        if needle in self.by_name:
            return self.by_name[needle]
        hits = [n for n in self.by_name if needle in n]
        if len(hits) != 1:
            raise KeyError(f"{needle}: {hits}")
        return self.by_name[hits[0]]

    def _owner(self, address):
        i = bisect.bisect_right(self.func_starts, address) - 1
        if i >= 0:
            a, size, name = self.funcs[i]
            if a <= address < a + size:
                return name
        return "?"

    def _on_block(self, uc, address, size, _):
        if address == self.marker_addr:
            self.markers.append(self.total)
        key = (address, size)
        hit = self.block_cache.get(key)
        if hit is None:
            code = bytes(uc.mem_read(address, size))
            n = sum(1 for _ in self.cs.disasm_lite(code, address))
            hit = (n, self._owner(address))
            self.block_cache[key] = hit
        n, owner = hit
        self.total += n
        self.counts[owner] = self.counts.get(owner, 0) + n

    def call(self, func_addr, r0=0, r1=0, max_insn=400_000_000):
        self.counts, self.total, self.markers = {}, 0, []
        uc = self.uc
        uc.reg_write(UC_ARM_REG_SP, STACK_TOP)
        uc.reg_write(UC_ARM_REG_LR, STOP | 1)
        uc.reg_write(UC_ARM_REG_R0, r0)
        uc.reg_write(UC_ARM_REG_R1, r1)
        uc.emu_start(func_addr | 1, STOP, count=max_insn)
        if uc.reg_read(UC_ARM_REG_PC) & ~1 != STOP:
            raise RuntimeError("did not return")
        return self.total, dict(self.counts)

    def cosf(self, x):
        self.call(self.sym("cosf"), r0=struct.unpack("<I", struct.pack("<f", x))[0])
        return struct.unpack("<f", struct.pack("<I", self.uc.reg_read(UC_ARM_REG_R0)))[0]


def f32(values):
    return np.asarray(values, dtype=np.float32)


def setup_object(emu, ir, red, n, hann):
    """Pipeline state right after push_sample #n (1-based count n) of streams ir/red."""
    raw = np.zeros(BUFFER, np.float32)
    rr = np.zeros(BUFFER, np.float32)
    for k in range(max(0, n - BUFFER), n):
        raw[k % BUFFER] = ir[k]
        rr[k % BUFFER] = red[k]
    blob = bytearray(OFF["size"])
    blob[OFF["raw_ring"]:OFF["raw_ring"] + 4000] = raw.tobytes()
    blob[OFF["red_ring"]:OFF["red_ring"] + 4000] = rr.tobytes()
    blob[OFF["hann"]:OFF["hann"] + 1024] = hann.tobytes()
    struct.pack_into("<i", blob, OFF["write_idx"], n % BUFFER)
    struct.pack_into("<Q", blob, OFF["total"], n)
    struct.pack_into("<I", blob, OFF["window_index"], 0)
    blob[OFF["primed"]] = 1
    struct.pack_into("<f", blob, OFF["threshold"], 0.5)
    blob[OFF["required"]] = 2
    emu.uc.mem_write(THIS, bytes(blob))
    emu.uc.mem_write(SRET, b"\0" * DEC["size"])


def run_window(emu, ir, red, n, hann):
    setup_object(emu, ir, red, n, hann)
    total, counts = emu.call(emu.sym("evaluate_window"), r0=SRET, r1=THIS)
    out = bytes(emu.uc.mem_read(SRET, DEC["size"]))
    prob = struct.unpack_from("<f", out, DEC["risk_probability"])[0]
    feats = list(struct.unpack_from("<16f", out, DEC["raw_features"]))
    return total, counts, prob, feats


def hann_window(emu):
    pi = np.float32(np.pi)
    out = []
    for i in range(256):
        arg = np.float32(np.float32(np.float32(2.0) * pi) * np.float32(i)) / np.float32(256.0)
        out.append(np.float32(0.5) * (np.float32(1.0) - np.float32(emu.cosf(float(arg)))))
    return f32(out)


GROUPS = [
    ("fmaf (soft double: f2d x3, dmul, dadd, d2f)", ("fmaf", "__aeabi_f2d", "__aeabi_dmul", "__muldf3",
                                                      "__aeabi_dadd", "__adddf3", "__aeabi_dsub",
                                                      "__aeabi_d2f", "__extendsfdf2", "__truncdfsf2")),
    ("soft float add/sub", ("__aeabi_fadd", "__addsf3", "__aeabi_fsub", "__subsf3", "__aeabi_frsub")),
    ("soft float mul", ("__aeabi_fmul", "__mulsf3")),
    ("soft float div", ("__aeabi_fdiv", "__divsf3")),
    ("soft float compare", ("__aeabi_fcmp", "__cmpsf2", "__gesf2", "__lesf2", "__eqsf2", "__aeabi_cfcmp",
                            "__aeabi_cfrcmple", "__unordsf2", "__aeabi_fcmpun")),
    ("int<->float", ("__aeabi_i2f", "__aeabi_ui2f", "__aeabi_f2iz", "__aeabi_f2uiz", "__floatsisf",
                     "__floatunsisf", "__fixsfsi")),
    ("sqrtf/expf/cosf/sinf", ("sqrtf", "__ieee754_sqrtf", "expf", "__ieee754_expf", "cosf", "sinf",
                              "__kernel_cosf", "__kernel_sinf", "__ieee754_rem_pio2f", "__kernel_rem_pio2f",
                              "scalbnf", "fabsf", "__errno", "__math")),
]


def group_counts(counts):
    out = {}
    assigned = set()
    for label, names in GROUPS:
        s = 0
        for k, v in counts.items():
            if k in assigned:
                continue
            if any(k == n or k.startswith(n) for n in names):
                s += v
                assigned.add(k)
        out[label] = s
    for k, v in counts.items():
        if k in assigned:
            continue
        label = ("ppg_bandpass_filter (own code)" if "ppg_bandpass_filter" in k else
                 "extract_features (own code incl. FFT)" if ("extract_features" in k or "cooley_tukey" in k) else
                 "KAN + XAI (own code)" if ("kan_" in k or "evaluate_kan" in k) else "other own code")
        out[label] = out.get(label, 0) + v
    return out


def variants(ir, red, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(len(ir)) / 50.0
    drift = 450.0 * np.sin(2 * np.pi * 0.03 * t + 0.7) + 150.0 * np.sin(2 * np.pi * 0.11 * t)
    return {
        "synthetic (timing replay input)": (f32(ir), f32(red)),
        "synthetic, integer": (f32(np.rint(ir)), f32(np.rint(red))),
        "live DC (+58000 IR, +61500 red)": (f32(ir.astype(np.float64) + 58000), f32(red.astype(np.float64) + 61500)),
        "live DC, integer": (f32(np.rint(ir.astype(np.float64) + 58000)), f32(np.rint(red.astype(np.float64) + 61500))),
        "live DC, integer, +drift +-600": (f32(np.rint(ir.astype(np.float64) + 58000 + drift)),
                                           f32(np.rint(red.astype(np.float64) + 61500 + 1.3 * drift))),
        "live DC, integer, +noise sd 3": (f32(np.rint(ir.astype(np.float64) + 58000 + rng.normal(0, 3, len(ir)))),
                                          f32(np.rint(red.astype(np.float64) + 61500 + rng.normal(0, 3, len(red))))),
    }


# lt2_evaluate() marker intervals (lt_now() calls micros() once): see loop_timing_v2_phases.h.
MIRROR_PHASES = {"align_ir": (0, 1), "filt_ir": (1, 2), "mean_ir": (2, 3), "align_red": (3, 4),
                 "filt_red": (4, 5), "mean_red": (5, 6), "feat14": (6, 7), "feat2": (7, 8),
                 "kan_base": (9, 10), "kan_pert": (10, 11), "sort": (11, 12), "tail": (13, 14)}


def decision_fields(blob):
    top = [(blob[44 + 12 * i], struct.unpack_from("<f", blob, 48 + 12 * i)[0]) for i in range(3)]
    return {"window_index": struct.unpack_from("<I", blob, 0)[0],
            "total": struct.unpack_from("<Q", blob, 8)[0], "start": struct.unpack_from("<Q", blob, 16)[0],
            "end": struct.unpack_from("<Q", blob, 24)[0], "delay": struct.unpack_from("<H", blob, 32)[0],
            "prob_bits": struct.unpack_from("<I", blob, 36)[0], "candidate": blob[40], "confirmed": blob[41],
            "consecutive": blob[42], "top": [(i, struct.unpack("<I", struct.pack("<f", d))[0]) for i, d in top],
            "features_bits": list(struct.unpack_from("<16I", blob, 80)), "two_wl": blob[144]}


def compare_mirror(orig_elf, mirror_elf, cases):
    """Original evaluate_window() (orig_elf) vs lt2_evaluate() (mirror_elf), same windows."""
    orig, mirr = Emulator(orig_elf), Emulator(mirror_elf)
    mirr.mark_calls_to("micros")
    hann = hann_window(orig)
    pipe = mirr.objects["pipeline"]
    mirror_fn = mirr.sym("lt2_evaluate")
    fx = np.load(FIXTURES)
    rows = []
    for case in cases:
        ir, red = fx[f"{case}_ir"], fx[f"{case}_red"]
        for consec in (0, 1):
            for n in range(750, len(ir) + 1, 250):
                setup_object(orig, ir, red, n, hann)
                orig.uc.mem_write(THIS + OFF["consecutive"], bytes([consec]))
                o_total, _ = orig.call(orig.sym("evaluate_window"), r0=SRET, r1=THIS)
                o = decision_fields(bytes(orig.uc.mem_read(SRET, DEC["size"])))
                setup_object(mirr, ir, red, n, hann)
                state = bytes(mirr.uc.mem_read(THIS, OFF["size"]))
                mirr.uc.mem_write(pipe, state)
                mirr.uc.mem_write(pipe + OFF["consecutive"], bytes([consec]))
                m_total, _ = mirr.call(mirror_fn, r0=SRET, r1=pipe)
                m = decision_fields(bytes(mirr.uc.mem_read(SRET, DEC["size"])))
                mk = mirr.markers
                phases = {k: mk[b] - mk[a] for k, (a, b) in MIRROR_PHASES.items()} if len(mk) >= 15 else {}
                rows.append({"case": case, "n": n, "consecutive_in": consec, "identical": o == m,
                             "orig_insn": o_total, "mirror_insn": m_total, "markers": len(mk), "phases": phases})
                print(f"mirror {case:22s} n={n:5d} c={consec} identical={o == m} orig={o_total:,} "
                      f"mirror={m_total:,} markers={len(mk)}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    ap.add_argument("--json", type=Path, help="write results here (refuses to overwrite)")
    ap.add_argument("--cases", default="normal,normal_risk_recovery,risk_after_reset")
    ap.add_argument("--mirror-elf", type=Path,
                    help="compare lt2_evaluate() in this ELF with evaluate_window() in --elf, then exit")
    args = ap.parse_args(argv)
    if args.json and args.json.exists():
        sys.exit(f"refusing to overwrite {args.json}")
    if args.mirror_elf:
        rows = compare_mirror(args.elf, args.mirror_elf, args.cases.split(","))
        sub = [r for r in rows if r["n"] >= 1000]
        summary = {"windows": len(rows), "identical": sum(r["identical"] for r in rows),
                   "phase_insn_median_subsequent": {k: float(np.median([r["phases"][k] for r in sub]))
                                                    for k in MIRROR_PHASES},
                   "phase_insn_median_first": {k: float(np.median([r["phases"][k] for r in rows if r["n"] == 750]))
                                               for k in MIRROR_PHASES}}
        print(json.dumps(summary, indent=1))
        if args.json:
            args.json.write_text(json.dumps({"orig_elf": str(args.elf), "mirror_elf": str(args.mirror_elf),
                                             "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
        return summary

    emu = Emulator(args.elf)
    hann = hann_window(emu)
    fx = np.load(FIXTURES)
    board = json.loads(BOARD_TIMING.read_text())
    board_blocks = [b for b in board["observations"]["blocks"] if b["repetition"] == 0 and b["decision"]]
    board_by_key = {(b["case"], b["total_samples"]): b for b in board_blocks}

    results = {"elf": str(args.elf), "variants": {}, "validation": []}
    for vname in variants(fx["normal_ir"], fx["normal_red"]):
        results["variants"][vname] = []
    for case in args.cases.split(","):
        ir0, red0 = fx[f"{case}_ir"], fx[f"{case}_red"]
        for vname, (ir, red) in variants(ir0, red0).items():
            for n in range(750, len(ir) + 1, 250):
                total, counts, prob, feats = run_window(emu, ir, red, n, hann)
                row = {"case": case, "n": n, "insn": total, "prob": prob, "groups": group_counts(counts),
                       "features": feats}
                results["variants"][vname].append(row)
                if vname.startswith("synthetic (timing"):
                    b = board_by_key.get((case, n))
                    if b:
                        bf = b["decision"]["features"]
                        results["validation"].append({
                            "case": case, "n": n, "board_cycles": b["cycles"], "emulated_insn": total,
                            "cycles_per_insn": b["cycles"] / total,
                            "prob_board": b["decision"]["probability"], "prob_emulated": prob,
                            "features_bit_identical": all(np.float32(x) == np.float32(y) for x, y in zip(feats, bf)),
                            "max_abs_feature_diff": max(abs(float(x) - float(y)) for x, y in zip(feats, bf)),
                        })
                print(f"{vname:34s} {case:22s} n={n:5d} insn={total:>10,d} prob={prob:.4f}", flush=True)

    summary = {}
    for vname, rows in results["variants"].items():
        subs = [r["insn"] for r in rows if r["n"] >= 1000]
        firsts = [r["insn"] for r in rows if r["n"] == 750]
        summary[vname] = {"subsequent_median": float(np.median(subs)), "subsequent_min": min(subs),
                          "subsequent_max": max(subs), "first_median": float(np.median(firsts)),
                          "groups_median_subsequent": {g: float(np.median([r["groups"].get(g, 0) for r in rows if r["n"] >= 1000]))
                                                       for g in rows[0]["groups"]}}
    base = summary["synthetic (timing replay input)"]["subsequent_median"]
    for v in summary.values():
        v["ratio_to_synthetic"] = v["subsequent_median"] / base
    results["summary"] = summary
    for vname, v in summary.items():
        print(f"{vname:34s} subsequent median {v['subsequent_median']:>12,.0f} "
              f"[{v['subsequent_min']:,}..{v['subsequent_max']:,}] ratio {v['ratio_to_synthetic']:.3f} "
              f"first {v['first_median']:,.0f}")
    if results["validation"]:
        cpi = [r["cycles_per_insn"] for r in results["validation"]]
        print(f"validation windows {len(cpi)}: features bit-identical "
              f"{sum(r['features_bit_identical'] for r in results['validation'])}/{len(cpi)}; "
              f"board cycles / emulated insn median {np.median(cpi):.3f} [{min(cpi):.3f}..{max(cpi):.3f}]")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()

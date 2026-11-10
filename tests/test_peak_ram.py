"""Host tests for the peak-RAM probe: the scan model on synthetic RAM, the capture parser on a
canned transcript, and the firmware header's keys/constants. No device, no port.

tests/fixtures/peak_ram_capture_canned.txt is SYNTHETIC (v1-probe layout, consistent
arithmetic, illustrative values); it is not board data and its numbers are not results."""
import io
import contextlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_peak_ram as apr
import peak_ram_scan_model as m

HEADER = ROOT / "firmware" / "Arduino Nano 33" / "include" / "peak_ram_probe.h"
CANNED = ROOT / "tests" / "fixtures" / "peak_ram_capture_canned.txt"

# v1 live layout (nano33ble ELF ba2673dd...), docs/peak-ram-design.md section 1.
HEAP_BASE, HEAP_LIMIT = 0x200159C8, 0x2003FC00
STACK_LIMIT, STACK_TOP = 0x2003FC00, 0x20040000
MAIN_BASE, MAIN_SIZE = 0x2000A0B0, 32768
IDLE_BASE, IDLE_SIZE = 0x20006C38, 512
TIMER_BASE, TIMER_SIZE = 0x20006E38, 768


def simulate():
    """Boot -> hook paint -> setup -> loop on a synthetic v1 RAM image, as the firmware sees it.

    Returns (ram, state); state holds the probe's own variables and the ground truth used to
    build the scenario.
    """
    ram = m.Ram(fill=0x00000000)
    ram.touch(HEAP_BASE, HEAP_LIMIT, seed=7)          # power-on garbage in the heap
    ram.touch(STACK_LIMIT, STACK_TOP, seed=8)         # and in the MSP region
    for base, size in ((MAIN_BASE, MAIN_SIZE), (IDLE_BASE, IDLE_SIZE), (TIMER_BASE, TIMER_SIZE)):
        m.rtx_create_stack(ram, base, size)
    st = {}
    # Boot: main thread runs constructors to depth 600 B; boot frames on MSP above 0x2003FF40.
    main_top = MAIN_BASE + MAIN_SIZE
    ram.touch(main_top - 600, main_top, seed=9)
    brk = HEAP_BASE + 1040                            # allocations before the hook
    # Hook (pr_paint_at_hook).
    st["paint_lo"] = (brk + 3) & ~3
    ram.fill(st["paint_lo"], HEAP_LIMIT, m.PATTERN)
    msp_at_hook = 0x2003FF40
    ram.fill(STACK_LIMIT, msp_at_hook, m.PATTERN)
    st["main_space_pre"] = m.rtx_stack_space(ram, MAIN_BASE, MAIN_SIZE)
    psp = main_top - 160
    ram.fill(MAIN_BASE + 4, psp - m.SP_MARGIN, m.RTX_FILL)
    # setup(): USBevt (256 B) and bleLoop (4096 B) stacks come from the heap, each behind an
    # 8 B allocator header and 8 B stats header; other allocations follow.
    usb = brk + 16
    m.rtx_create_stack(ram, usb, 256)
    ble = usb + 256 + 16
    m.rtx_create_stack(ram, ble, 4096)
    other_lo = ble + 4096
    ram.touch(other_lo, other_lo + 2000, seed=10)
    brk = other_lo + 2000
    ram.touch(STACK_LIMIT + 700, msp_at_hook, seed=11)  # ISRs in setup reach 0x2003FEBC
    ram.touch(main_top - 9000, main_top - 600, seed=12)  # setup() reaches 9,000 B
    # pr_setup_done.
    st["main_space_setup"] = m.rtx_stack_space(ram, MAIN_BASE, MAIN_SIZE)
    st["msp_low_setup"] = m.msp_low(ram, STACK_LIMIT, msp_at_hook)
    ram.fill(STACK_LIMIT, msp_at_hook, m.PATTERN)
    ram.fill(MAIN_BASE + 4, psp - m.SP_MARGIN, m.RTX_FILL)
    # loop(): decisions reach 6,000 B of main stack; ISRs reach 0x2003FD00; BLE thread uses
    # 1,100 B, USBevt 120 B, idle 96 B, timer 0 B; a 300 B block is allocated, written
    # only in its first 40 B (the break covers the rest).
    ram.touch(main_top - 6000, main_top - 160, seed=13)
    ram.touch(STACK_LIMIT + 256, msp_at_hook, seed=14)
    ram.touch(ble + 4096 - 1100, ble + 4096, seed=15)
    ram.touch(usb + 256 - 120, usb + 256, seed=16)
    ram.touch(IDLE_BASE + IDLE_SIZE - 96, IDLE_BASE + IDLE_SIZE, seed=17)
    ram.touch(brk, brk + 40, seed=18)
    brk += 300
    st.update(brk=brk, usb=usb, ble=ble, msp_at_hook=msp_at_hook)
    return ram, st


def scan(ram, st):
    """pr_report's arithmetic on the simulated RAM."""
    top = m.heap_top(ram, st["paint_lo"], HEAP_LIMIT, st["brk"])
    loop_low = m.msp_low(ram, STACK_LIMIT, st["msp_at_hook"])
    msp_low = min(loop_low, st["msp_low_setup"])
    threads = []
    main_now = m.rtx_stack_space(ram, MAIN_BASE, MAIN_SIZE)
    main_life = min(main_now, st["main_space_pre"], st["main_space_setup"])
    for name, tid, base, size, space, loc in (
            ("main", 0x200071A0, MAIN_BASE, MAIN_SIZE, main_life, "bss"),
            ("rtx_idle", 0x20006B70, IDLE_BASE, IDLE_SIZE, m.rtx_stack_space(ram, IDLE_BASE, IDLE_SIZE), "bss"),
            ("rtx_timer", 0x20006BB4, TIMER_BASE, TIMER_SIZE, m.rtx_stack_space(ram, TIMER_BASE, TIMER_SIZE), "bss"),
            ("USBevt", st["usb"] - 0x60, st["usb"], 256, m.rtx_stack_space(ram, st["usb"], 256), "heap"),
            ("anon", st["ble"] - 0x60, st["ble"], 4096, m.rtx_stack_space(ram, st["ble"], 4096), "heap")):
        threads.append((name, tid, base, size, space, loc))
    spans = [(top, HEAP_LIMIT), (STACK_LIMIT, msp_low)] + [(b, b + s) for _, _, b, _, s, _ in threads]
    untouched, listed = m.union_bytes(spans)
    return {"heap_top": top, "msp_low": msp_low, "msp_loop_low": loop_low, "main_now": main_now,
            "threads": threads, "untouched": untouched, "ovl": listed - untouched,
            "whole_hw": m.RAM_SIZE - untouched}


class ScanModelTests(unittest.TestCase):
    def setUp(self):
        self.ram, self.st = simulate()
        self.r = scan(self.ram, self.st)

    def test_regions_match_the_scenario(self):
        r, st = self.r, self.st
        self.assertEqual(r["heap_top"], st["brk"])            # 260 B allocated, unwritten: still used
        self.assertEqual(STACK_TOP - r["msp_low"], 1024 - 256)  # deepest ISR use, loop phase
        self.assertEqual(STACK_TOP - st["msp_low_setup"], 1024 - 700)
        self.assertEqual(MAIN_SIZE - st["main_space_pre"], 600)
        self.assertEqual(MAIN_SIZE - st["main_space_setup"], 9000)
        self.assertEqual(MAIN_SIZE - r["main_now"], 6000)       # loop phase alone
        peaks = {name: size - space for name, _, _, size, space, _ in r["threads"]}
        self.assertEqual(peaks, {"main": 9000, "rtx_idle": 96, "rtx_timer": 0, "USBevt": 120, "anon": 1100})

    def test_whole_device_is_ram_minus_disjoint_untouched(self):
        r, st = self.r, self.st
        self.assertEqual(r["ovl"], 0)
        expect_untouched = ((HEAP_LIMIT - st["brk"]) + 256 + (MAIN_SIZE - 9000) + (512 - 96)
                            + 768 + (256 - 120) + (4096 - 1100))
        self.assertEqual(r["untouched"], expect_untouched)
        self.assertEqual(r["whole_hw"], m.RAM_SIZE - expect_untouched)
        # Heap stacks are inside the heap high-water, so their untouched bottoms do not
        # overlap the heap's untouched top.
        self.assertLess(st["ble"] + 4096, r["heap_top"])

    def test_stray_write_above_the_break_is_caught(self):
        self.ram.put(HEAP_LIMIT - 8, 0x12345678)
        self.assertEqual(m.heap_top(self.ram, self.st["paint_lo"], HEAP_LIMIT, self.st["brk"]), HEAP_LIMIT - 4)

    def test_unpainted_regions_count_as_fully_used(self):
        self.assertEqual(m.heap_top(self.ram, self.st["paint_lo"], HEAP_LIMIT, 0, painted=False), HEAP_LIMIT)
        self.assertEqual(m.rtx_stack_space(self.ram, HEAP_BASE, 256), 0)  # no magic word: no watermark

    def test_pattern_collision_is_the_documented_limit(self):
        # A touched word equal to the pattern at the frontier reads as untouched (design, section 6).
        top = m.heap_top(self.ram, self.st["paint_lo"], HEAP_LIMIT, 0)
        self.ram.put(top - 4, m.PATTERN)
        self.assertEqual(m.heap_top(self.ram, self.st["paint_lo"], HEAP_LIMIT, 0), top - 4)

    def test_union_merges_overlap_and_clamps(self):
        total, listed = m.union_bytes([(0x20000010, 0x20000020), (0x20000018, 0x20000030),
                                       (0x20000040, 0x20000040), (0x2003FFF0, 0x20040010)])
        self.assertEqual(total, 0x20 + 0x10)
        self.assertEqual(listed - total, 8)

    def test_model_line_passes_the_analyser_arithmetic(self):
        r = self.r
        rec = {"t_ms": 30000, "rep": 6, "dec": 4, "lf": 0, "lnf": 1500,
               "heap": (HEAP_LIMIT - HEAP_BASE, r["heap_top"] - HEAP_BASE),
               "msp": (1024, STACK_TOP - r["msp_low"]), "msp_loop": STACK_TOP - r["msp_loop_low"],
               "main_pre": 600, "main_setup": 9000, "main_loop": MAIN_SIZE - r["main_now"],
               "threads": [(n, tid, size, size - space, loc) for n, tid, _, size, space, loc in r["threads"]],
               "brk": self.st["brk"], "heap_hw": r["heap_top"], "hs_cur": 7000, "hs_max": 7100,
               "hs_ovh": 96, "hs_fail": 0, "static": 88952, "fixed": 512, "untouched": r["untouched"],
               "ovl": r["ovl"], "whole_hw": r["whole_hw"], "ram": m.RAM_SIZE, "scan_us": 3000, "rpt_alloc": 0}
        parsed = apr.parse_pr(m.format_pr_line(rec)[len("PR:"):])
        ok, problems = apr.rederive(parsed)
        self.assertTrue(ok, problems)
        self.assertEqual(parsed["threads"]["T.main@0x200071A0"]["peak"], 9000)


class CannedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.lines = CANNED.read_text(encoding="utf-8").splitlines()

    def test_canned_capture_is_valid_and_reports_the_final_line(self):
        rep = apr.analyse(self.lines, "nano33ble_peakram")
        self.assertTrue(rep["valid"], [c for c in rep["checks"] if not c["ok"]])
        self.assertEqual(rep["pr_lines"], 36)
        self.assertEqual(rep["pr_info_lines"], 4)
        s = rep["summary"]
        self.assertEqual(s["whole_hw"]["final"], s["whole_hw"]["max"])
        self.assertEqual(s["whole_hw"]["final"], 70176)
        self.assertEqual(s["static_data_bss"], 88952)
        self.assertEqual(s["regions"]["heap"], {"reserved": 171656, "peak": 7860, "headroom": 163796, "max_peak": 7860})
        self.assertEqual(s["regions"]["msp"]["peak"], 768)
        self.assertEqual(s["threads"]["T.main@0x20007550"]["peak"], 9000)
        self.assertEqual(s["main_phases"], {"pre_hook": 600, "hook_to_setup_end": 9000, "loop": 6000})
        self.assertEqual(s["hz_range"], [49.96, 49.98])
        self.assertEqual(rep["paper_figure"]["whole_hw_bytes"], 70176)
        self.assertIn("conservative upper bound", rep["paper_figure"]["wording"])
        self.assertIn("70,176 B", apr.markdown(rep, CANNED.name))

    def test_wrong_image_is_rejected(self):
        rep = apr.analyse(self.lines, "nano33ble_v2_f1a_peakram")
        self.assertFalse(rep["valid"])
        self.assertIsNone(rep["paper_figure"])
        self.assertFalse(next(c for c in rep["checks"] if c["name"] == "identity")["ok"])

    def failing(self, lines):
        rep = apr.analyse(lines)
        return {c["name"] for c in rep["checks"] if not c["ok"]}

    def pr_indices(self):
        return [i for i, line in enumerate(self.lines) if "> PR: " in line]

    def test_gap_over_15_s_fails(self):
        idx = self.pr_indices()
        lines = [line for i, line in enumerate(self.lines) if i not in idx[10:13]]  # 20 s hole
        self.assertNotIn("gaps", self.failing([line for i, line in enumerate(self.lines)
                                               if i not in idx[10:12]]))  # exactly 15 s passes
        self.assertIn("gaps", self.failing(lines))

    def test_too_few_decisions_fails(self):
        lines = [re.sub(r" dec=\d+", " dec=3", line) for line in self.lines]
        self.assertIn("decisions", self.failing(lines))

    def test_reboot_fails(self):
        idx = self.pr_indices()
        lines = list(self.lines)
        lines[idx[20]] = re.sub(r" t_ms=\d+", " t_ms=4000", lines[idx[20]])
        self.assertIn("no_reboot", self.failing(lines))

    def test_falling_high_water_fails(self):
        idx = self.pr_indices()
        lines = list(self.lines)
        lines[idx[-1]] = lines[idx[-1]].replace("whole_hw=70176", "whole_hw=70172").replace(
            "untouched=191968", "untouched=191972").replace("heap=171656/7860/163796", "heap=171656/7856/163800")
        self.assertIn("monotonic", self.failing(lines))

    def test_arithmetic_mismatch_fails(self):
        idx = self.pr_indices()
        lines = list(self.lines)
        lines[idx[5]] = lines[idx[5]].replace("ovl=0", "ovl=4")
        self.assertIn("regions", self.failing(lines))

    def test_finger_and_central_fail(self):
        lines = [re.sub(r" lf=\d+", " lf=900", line) for line in self.lines]
        lines.append("10:03:01.000 > BLE connected to: 11:22:33:44:55:66")
        self.assertTrue({"no_finger", "no_central"} <= self.failing(lines))

    def test_missing_field_is_a_parse_failure(self):
        idx = self.pr_indices()
        lines = list(self.lines)
        lines[idx[3]] = lines[idx[3]].replace(" hs_fail=0", "")
        self.assertIn("parse", self.failing(lines))

    def test_main_refuses_an_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(apr.main([str(CANNED), "--output", str(out), "--expect-image", "nano33ble_peakram"]), 0)
            self.assertTrue((out / "report.json").exists() and (out / "report.md").exists())
            before = (out / "report.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(apr.main([str(CANNED), "--output", str(out)]), 2)
            self.assertEqual((out / "report.json").read_bytes(), before)


class UploadBindingTests(unittest.TestCase):
    def archive(self, d, log_text, listed=None):
        image = Path(d) / "firmware.bin"
        image.write_bytes(b"\x00" * 1000)
        sha = apr.sha256_file(image)
        (Path(d) / "SHA256SUMS").write_text(f"{listed or sha} *firmware.bin\n")
        log = Path(d) / "bossac.log"
        log.write_text(log_text)
        return image, log

    def test_verified_write_of_the_archived_image_passes(self):
        with tempfile.TemporaryDirectory() as d:
            image, log = self.archive(d, "Write 1000 bytes to flash (16 pages)\nVerify successful\n")
            up = apr.upload_binding(image, log)
            self.assertTrue(up["ok"])
            rep = apr.analyse(CANNED.read_text(encoding="utf-8").splitlines(), "nano33ble_peakram", up)
            self.assertTrue(rep["valid"])

    def test_short_write_unverified_or_wrong_image_fails(self):
        with tempfile.TemporaryDirectory() as d:
            image, log = self.archive(d, "Write 999 bytes to flash\nVerify successful\n")
            self.assertFalse(apr.upload_binding(image, log)["ok"])
        with tempfile.TemporaryDirectory() as d:
            image, log = self.archive(d, "Write 1000 bytes to flash\nVerify failed\n")
            self.assertFalse(apr.upload_binding(image, log)["ok"])
        with tempfile.TemporaryDirectory() as d:
            image, log = self.archive(d, "Write 1000 bytes to flash\nVerify successful\n", listed="0" * 64)
            up = apr.upload_binding(image, log)
            self.assertFalse(up["ok"])
            rep = apr.analyse(CANNED.read_text(encoding="utf-8").splitlines(), "nano33ble_peakram", up)
            self.assertFalse(rep["valid"])
            self.assertIsNone(rep["paper_figure"])


def function_body(text, signature):
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
        i += 1


KEY_CALL = re.compile(r'pr_(?:u|x|rph|span)\("(\w+)"')


class FirmwareHeaderTests(unittest.TestCase):
    """The analyser's keys and the model's constants must match the firmware header."""

    def setUp(self):
        self.text = HEADER.read_text(encoding="utf-8")

    def test_pr_keys_match(self):
        keys = KEY_CALL.findall(function_body(self.text, "static void pr_report()"))
        expected = [k for k in apr.PR_INT_KEYS if k != "v"] + apr.PR_HEX_KEYS + apr.PR_REGION_KEYS
        self.assertEqual(sorted(keys), sorted(expected))
        self.assertIn('pr_put("PR: v=1")', self.text)

    def test_pr_info_keys_match(self):
        body = function_body(self.text, "static void pr_info()")
        keys = KEY_CALL.findall(body)
        literal = re.search(r'pr_put\("PR_INFO: (.*?)"\s*PR_IMAGE\s*" model="\s*PR_MODEL\s*"(.*?)"\)', body)
        literal_keys = re.findall(r"(\w+)=", literal.group(1) + "img= model=" + literal.group(2))
        expected = apr.INFO_SPAN_KEYS + apr.INFO_INT_KEYS + apr.INFO_STR_KEYS
        self.assertEqual(sorted(set(keys) | set(literal_keys)), sorted(set(expected)))

    def test_constants_match_the_model(self):
        def const(name):
            return int(re.search(rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)u?;", self.text).group(1), 0)
        self.assertEqual(const("PR_PATTERN"), m.PATTERN)
        self.assertEqual(const("PR_SP_MARGIN"), m.SP_MARGIN)
        self.assertEqual(const("PR_RAM_START"), m.RAM_START)
        self.assertEqual(const("PR_RAM_SIZE"), m.RAM_SIZE)
        self.assertEqual(m.RAM_SIZE, apr.RAM_BYTES)
        self.assertEqual(const("PR_PERIOD_MS"), 5000)
        self.assertEqual(const("PR_INFO_EVERY"), 12)

    def test_probe_only_compiles_under_its_flag(self):
        self.assertIn("#ifdef PEAK_RAM_PROBE", self.text)
        main = (ROOT / "firmware" / "Arduino Nano 33" / "src" / "main.cpp").read_text(encoding="utf-8")
        for call in ("pr_poll();", "pr_setup_done();", "pr_on_decision();", "pr_count_live_gate(", "pr_force_payload"):
            pos = main.index(call)
            before = main[:pos]
            self.assertGreater(before.rfind("#ifdef PEAK_RAM_PROBE"), before.rfind("#endif"), call)


if __name__ == "__main__":
    unittest.main()

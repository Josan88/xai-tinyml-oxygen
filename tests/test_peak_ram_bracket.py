"""Host tests for scripts/peak_ram_bracket.py: the derived peak-RAM bracket. No device, no port.

The synthetic captures below are built with peak_ram_scan_model.format_pr_line; their numbers are
illustrative, not board data. The last class reads the two committed 25 Oct captures (read only)."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_peak_ram as apr
import peak_ram_bracket as pb
import peak_ram_scan_model as m

HEAP_BASE = 0x20015D78
INFO = ("PR_INFO: v=1 img=nano33ble_peakram model=v1 force=1 hook=ctor101 ram=0x20000000+262144 "
        "fixed=0x20000000+512 data=0x20000200+1952 bss=0x200009A0+87000 heap=0x20015D78+171656 "
        "msp=0x2003FC00+1024 gap=0 pattern=0x9E3779B9 rtx_fill=0xCCCCCCCC wm=1 period_ms=5000 "
        "heap_painted=1 msp_painted=1 main_painted=1 setup_painted=1 heap_paint=0x20015D78+171656 "
        "msp_paint=0x2003FC00+960 main_paint=0x2000A464+32452 brk_ctor=0x20015D78 brk_setup=0x2001B844 "
        "nthr=5 T.main@0x20007550=0x2000A460+32768/bss T.rtx_idle@0x20006FA0=0x20006FE8+512/bss "
        "T.anon@0x2001A308=0x2001A360+4096/heap T.rtx_timer@0x20006F5C=0x200071E8+768/bss "
        "T.USBevt@0x20016218=0x20016268+256/heap")
STATIC, FIXED = 88952, 512
SIZES = {"main": 32768, "rtx_idle": 512, "rtx_timer": 768, "anon": 4096, "USBevt": 256}
LOCS = {"main": "bss", "rtx_idle": "bss", "rtx_timer": "bss", "anon": "heap", "USBevt": "heap"}
TIDS = {"main": 0x20007550, "rtx_idle": 0x20006FA0, "rtx_timer": 0x20006F5C, "anon": 0x2001A308,
        "USBevt": 0x20016218}
OTHER = {"rtx_idle": 384, "rtx_timer": 104, "anon": 408, "USBevt": 136}
MSP = 384


def pr_line(rep, main_loop, cur, ovh, brk_span=None, hs_max=None):
    """One self-consistent PR: line; brk_span defaults to a fully allocated arena (cur + ovh)."""
    span = cur + ovh if brk_span is None else brk_span
    peaks = dict(OTHER, main=max(96, 792, main_loop))  # T.main: maximum over the three phases
    untouched = (171656 - span) + (1024 - MSP) + sum(SIZES[k] - peaks[k] for k in SIZES)
    rec = {"t_ms": 5000 * rep, "rep": rep, "dec": max(0, rep - 2), "lf": 0, "lnf": 250 * rep,
           "heap": (171656, span), "msp": (1024, MSP), "msp_loop": MSP, "main_pre": 96,
           "main_setup": 792, "main_loop": main_loop,
           "threads": [(k, TIDS[k], SIZES[k], peaks[k], LOCS[k]) for k in SIZES],
           "brk": HEAP_BASE + span, "heap_hw": HEAP_BASE + span, "hs_cur": cur,
           "hs_max": cur if hs_max is None else hs_max, "hs_ovh": ovh, "hs_fail": 0,
           "static": STATIC, "fixed": FIXED, "untouched": untouched, "ovl": 0,
           "whole_hw": m.RAM_SIZE - untouched, "ram": m.RAM_SIZE, "scan_us": 3000, "rpt_alloc": 0}
    return m.format_pr_line(rec)


def capture(lines):
    stamped = [f"10:00:{i:02d}.000 > {line}" for i, line in enumerate([INFO] + lines)]
    return apr.parse_capture(stamped)


def expected_lower(heap_term, main_peak):
    return FIXED + STATIC - (32768 + 512 + 768) + heap_term - (4096 + 256) + main_peak


class BracketArithmeticTests(unittest.TestCase):
    def test_flat_heap(self):
        lines = [pr_line(r, 1760 if r < 4 else 4600, 22580, 928) for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertEqual(b["problems"], [])
        self.assertEqual(b["lower_bytes"], expected_lower(23508, 4600))
        self.assertEqual(b["upper_bytes"], b["lower_bytes"] + sum(OTHER.values()) + MSP)
        self.assertEqual(b["main_peak_window"]["first_at_peak"]["rep"], 4)
        self.assertLessEqual(b["lower_bytes"], b["upper_bytes"])

    def test_heap_step_after_the_main_peak_uses_the_earlier_level(self):
        # v1's pattern: main peak first seen at rep 4, heap grows by 88 B at rep 6.
        lines = [pr_line(r, 1760 if r < 4 else 4584, 22580 if r < 6 else 22636, 928 if r < 6 else 960)
                 for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertEqual(b["heap_term"], 23508)
        self.assertEqual(b["lower_bytes"], expected_lower(23508, 4584))
        self.assertEqual(b["upper_bytes"] - b["lower_bytes"], 88 + sum(OTHER.values()) + MSP)
        self.assertEqual(b["naive_lower_bytes"] - b["lower_bytes"], 88)
        self.assertEqual([lv["heap_live"] for lv in b["heap_levels"]], [23508, 23596])

    def test_heap_step_before_the_main_peak_uses_the_report_before_the_peak(self):
        lines = [pr_line(r, 1760 if r < 8 else 4584, 22580 if r < 6 else 22636, 928 if r < 6 else 960)
                 for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertEqual(b["heap_term"], 23596)
        self.assertEqual(b["lower_bytes"], b["naive_lower_bytes"])

    def test_untiled_arena_withholds_the_lower_bound(self):
        # A free chunk below the break at one report: the reports no longer pin the heap level.
        lines = [pr_line(r, 1760 if r < 4 else 4600, 22580, 928, brk_span=23508 + (64 if r == 7 else 0))
                 for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertIsNone(b["lower_bytes"])
        self.assertTrue(any("not fully allocated" in p for p in b["problems"]))
        self.assertEqual(b["upper_bytes"], capture(lines)["pr"][-1]["whole_hw"])

    def test_main_peak_on_the_first_line_withholds_the_lower_bound(self):
        lines = [pr_line(r, 4600, 22580, 928) for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertIsNone(b["lower_bytes"])
        self.assertTrue(any("first captured report" in p for p in b["problems"]))

    def test_main_peak_from_setup_withholds_the_lower_bound(self):
        lines = [pr_line(r, 700, 22580, 928) for r in range(1, 12)]  # main_setup 792 is deeper
        b = pb.bracket(capture(lines))
        self.assertIsNone(b["lower_bytes"])

    def test_cur_below_max_is_reported_but_does_not_block(self):
        # hs_cur < hs_max says a transient was larger earlier; the tiling still pins each report.
        lines = [pr_line(r, 1760 if r < 4 else 4600, 22580, 928, hs_max=22700) for r in range(1, 12)]
        b = pb.bracket(capture(lines))
        self.assertEqual(b["cur_eq_max_lines"], 0)
        self.assertIsNotNone(b["lower_bytes"])

    def test_canned_fixture_is_not_tiled(self):
        cap = apr.parse_capture((ROOT / "tests" / "fixtures" / "peak_ram_capture_canned.txt")
                                .read_text(encoding="utf-8").splitlines())
        b = pb.bracket(cap)
        self.assertIsNone(b["lower_bytes"])


class CommittedCaptureTests(unittest.TestCase):
    """The two 25 Oct captures (read only)."""

    @classmethod
    def setUpClass(cls):
        cls.summary = pb.run(ROOT)

    def test_bracket_per_model(self):
        v1, v2 = self.summary["models"]["v1"], self.summary["models"]["v2"]
        self.assertEqual((v1["lower_bytes"], v1["upper_bytes"]), (79156, 80660))
        self.assertEqual((v2["lower_bytes"], v2["upper_bytes"]), (85220, 86644))
        for r in (v1, v2):
            self.assertEqual(r["problems"], [])
            self.assertEqual(r["tiled_lines"], 36)
            self.assertEqual(r["cur_eq_max_lines"], 36)
            self.assertTrue(all(r["analyser_checks"].values()))
            self.assertIn("is not measured", r["wording"])
            self.assertIn("the upper end is the RAM high-water", r["wording"])
        self.assertEqual(v1["heap_term"], 23508)                   # before the 88 B step
        self.assertEqual(v1["naive_lower_bytes"], 79244)           # the brief's rough figure
        self.assertEqual([lv["heap_live"] for lv in v1["heap_levels"]], [23508, 23596])
        self.assertEqual([lv["heap_live"] for lv in v2["heap_levels"]], [23508])
        self.assertFalse(self.summary["measured"])

    def test_main_refuses_an_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "b"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pb.main(["--output", str(out)]), 0)
            self.assertTrue((out / "summary.json").exists() and (out / "README.md").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(pb.main(["--output", str(out)]), 2)


if __name__ == "__main__":
    unittest.main()

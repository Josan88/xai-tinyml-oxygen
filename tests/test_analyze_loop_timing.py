"""Parser tests for scripts/analyze_loop_timing.py on synthetic lines; no device data."""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_loop_timing as alt

HEADER = ROOT / "firmware" / "Arduino Nano 33" / "include" / "loop_timing_v2.h"


def firmware_format(prefix):
    """The snprintf format string for `prefix` (LT: or LT_IDLE:) as written in the header."""
    text = HEADER.read_text(encoding="utf-8")
    start = text.index(f'"{prefix} ')
    end = text.index('\\r\\n"', start)
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', text[start:end + 5])
    return "".join(pieces).replace("\\r\\n", "")


def fill(fmt, value):
    return re.sub(r"%l?[udi]", str(value), fmt)


def red(win, samples, hz="47.50"):
    return (f"Red: 63038  IR: 60490  Hz: {hz}  SpO2_est: 95.4%  KAN_Prob: 0.89  Candidate: YES  "
            f"Confirmed: YES  Win: {win}  Samples: {samples}  TopFeature: ratio_of_ratios")


FEAT = "  Feat: mean=-0.6670 std=106.9323 two_wavelength_valid=1  Signal: OK  Status: Normal (Prob: 0.10 | Win: 9)"


def lt_line(**kw):
    base = {"n": 1, "win": 9, "tot": 3000, "conn": 0, "dpos": 0, "frames": 1, "wr": 5, "rd": 4,
            "pend": 1, "pass_us": 650000, "pass_c": 41600000, "dec_us": 433000, "dec_c": 27712000,
            "feat_us": 180000, "feat_c": 11520000, "post_gap_us": 655000, "post_pend": 0,
            "post_got": 0, "ddt_us": 5640000, "el_us": 5640000, "drained": 250,
            "exp_mrec": 282000, "loss_mrec": 32000, "nf": 0, "p600": 1, "p640": 1, "g600": 0,
            "g640": 0, "e600": 0, "short": 0, "maxpass_us": 650000, "maxgap_us": 40000}
    base.update(kw)
    return "LT: " + " ".join(f"{k}={v}" for k, v in base.items())


class ParserTests(unittest.TestCase):
    def test_capture_lines_are_classified(self):
        lines = [
            "--- capture COM8 115200 start 2026-10-15T17:10:39 duration 360s (build/v2/cap.py)\n",
            "17:10:40.023 > LT_INFO: build=LOOP_TIMING_V2 clock_hz=64000000 cyccnt=1\r\n",
            f"17:10:44.518 > {red(9, 3000)}\n",
            f"17:10:44.725 > {FEAT}\n",
            f"17:10:45.300 > {lt_line()}\n",
            "17:10:50.000 > LT_IDLE: n=2 el_us=5000000 drained=250 exp_mrec=250000 loss_mrec=0 maxpass_us=30000\n",
            "garbage without timestamp\n",
        ]
        cap = alt.parse_capture(lines)
        self.assertEqual(cap["lines"], 5)
        self.assertEqual(cap["info"][0]["clock_hz"], 64000000)
        self.assertEqual(cap["decisions"], [{"t": 17 * 3600 + 10 * 60 + 44.725, "win": 9, "samples": 3000}])
        self.assertEqual(cap["lt"][0]["loss_mrec"], 32000)
        self.assertEqual(cap["idle"][0]["drained"], 250)
        self.assertEqual(cap["hz"][0]["hz"], 47.5)

    def test_negative_values_and_midnight_wrap(self):
        cap = alt.parse_capture([f"23:59:59.900 > {lt_line(loss_mrec=-150, other_us=-3)}",
                                 f"00:00:00.100 > {lt_line(n=2)}"])
        self.assertEqual(cap["lt"][0]["loss_mrec"], -150)
        self.assertEqual(cap["lt"][0]["other_us"], -3)
        self.assertAlmostEqual(cap["lt"][1]["t"] - cap["lt"][0]["t"], 0.2, places=6)

    def test_host_intervals_respect_resets(self):
        decisions = [{"t": 100.0, "win": 5, "samples": 1500}, {"t": 105.64, "win": 6, "samples": 1750},
                     {"t": 110.64, "win": 7, "samples": 2000}, {"t": 130.0, "win": 0, "samples": 750}]
        rows = alt.host_intervals(decisions)
        self.assertTrue(rows[0]["contiguous"])
        self.assertAlmostEqual(rows[0]["implied_loss_records"], 32.0)
        self.assertAlmostEqual(rows[1]["implied_loss_records"], 0.0)
        self.assertFalse(rows[2]["contiguous"])
        self.assertIsNone(rows[2]["implied_loss_records"])

    def test_lt_intervals_exclude_resets_and_first_line(self):
        lines = [f"10:00:00.000 > {lt_line(n=1, win=3)}",
                 f"10:00:05.640 > {lt_line(n=2, win=4)}",
                 f"10:00:10.640 > {lt_line(n=3, win=5, loss_mrec=0, exp_mrec=250000, post_pend=30, post_got=30)}",
                 f"10:00:30.000 > {lt_line(n=4, win=6, nf=40)}"]
        report = alt.analyse(lines)
        s = report["board"]["summary"]
        self.assertEqual(s["lt_lines"], 4)
        self.assertEqual(s["contiguous_intervals"], 2)
        self.assertEqual(s["loss_records_per_decision"]["max"], 32.0)
        self.assertEqual(s["loss_records_per_decision"]["min"], 0.0)
        self.assertEqual(s["intervals_with_loss_ge_1"], 1)
        self.assertEqual(s["post_pend_zero_after_decision"], 1)
        self.assertEqual(s["decision_pass_phases"]["dec"]["us"]["median"], 433000)

    def test_capture_without_lt_lines_reports_host_only(self):
        lines = [f"10:00:00.000 > {red(1, 1000)}", f"10:00:00.200 > {FEAT}",
                 f"10:00:05.200 > {red(2, 1250)}", f"10:00:05.400 > {FEAT}"]
        report = alt.analyse(lines)
        self.assertIsNone(report["board"]["summary"])
        self.assertIn("No LT lines", report["board"]["note"])
        self.assertEqual(report["host"]["contiguous_intervals"], 1)
        self.assertAlmostEqual(report["host"]["interval_s"]["median"], 5.2)
        self.assertIsNone(report["idle"])
        self.assertIn("No LT lines", alt.markdown(report, "x.txt"))


class FirmwareFormatTests(unittest.TestCase):
    """The analyser's keys must match the firmware's snprintf formats."""

    def test_lt_format_keys_parse_and_fit_the_buffer(self):
        fmt = firmware_format("LT:")
        line = fill(fmt, 4294967295)
        self.assertLess(len(line) + 2, 1400)  # lt_line[1400], plus \r\n
        rec = alt.parse_kv(line[len("LT:"):])
        for p in alt.PHASES:
            self.assertIn(f"{p}_us", rec, p)
            self.assertIn(f"{p}_c", rec, p)
        for k in ("n", "win", "tot", "conn", "dpos", "frames", "wr", "rd", "pend", "st_n", "ble_n",
                  "post_wr", "post_rd", "post_pend", "post_got", "ddt_us", "el_us", "el_c",
                  "drained", "exp_mrec", "loss_mrec", "passes", "maxpass_us", "maxgap_us",
                  "p600", "p640", "g600", "g640", "e600", "short", "nf"):
            self.assertIn(k, rec, k)
        self.assertEqual(len(re.findall(r"%l?[udi]", fmt)), len(rec))

    def test_idle_format_keys_parse_and_fit_the_buffer(self):
        fmt = firmware_format("LT_IDLE:")
        line = fill(fmt, 4294967295)
        self.assertLess(len(line) + 2, 1400)
        rec = alt.parse_kv(line[len("LT_IDLE:"):])
        for k in ("n", "el_us", "el_c", "passes", "maxpass_us", "maxgap_us", "drained", "exp_mrec",
                  "loss_mrec", "p600", "p640", "g600", "g640", "e600", "short", "nf", "fin",
                  "conn", "ltp_us", "ltp_c"):
            self.assertIn(k, rec, k)
        self.assertEqual(len(re.findall(r"%l?[udi]", fmt)), len(rec))


PHASES_HEADER = ROOT / "firmware" / "Arduino Nano 33" / "include" / "loop_timing_v2_phases.h"
LITERALS = r'((?:"(?:[^"\\]|\\.)*"\s*)+)'


def lt2_formats():
    """(phase keys, [format of each lt2_append call in lt2_emit(), in order])."""
    text = PHASES_HEADER.read_text(encoding="utf-8")
    start = text.index("LT2_PHASE_KEYS[LT2_PHASES]")
    keys = re.findall(r'"(\w+)"', text[start:text.index("};", start)])
    body = text[text.index("static void lt2_emit()"):]
    fmts = ["".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m))
            for m in re.findall(r"lt2_append\((?:0|n),\s*" + LITERALS, body)]
    return keys, [f.replace("\\r\\n", "\r\n") for f in fmts]


def lt2_line(**kw):
    base = {"n": 46, "win": 9}
    for p in alt.LT2_PHASES:
        base[f"{p}_us"], base[f"{p}_c"] = 1000, 64000
    base.update({"filt_ir_us": 260000, "filt_ir_c": 16640000, "filt_red_us": 262000, "filt_red_c": 16768000,
                 "phases_us": 600000, "phases_c": 38400000, "smp_ok": 1, "smp_us": 250, "smp_n": 2400,
                 "smp_main": 1800, "smp_oth": 500, "smp_isr": 100, "smp_full": 0, "smp_cost_c": 300000,
                 "oth1_id": 536874000, "oth1_n": 500, "isr1_exc": 43, "isr1_n": 100,
                 "stk_size": 32768, "stk_free": 1200})
    base.update(kw)
    text = "LT2: " + " ".join(f"{k}={v}" for k, v in base.items())
    return text.replace("oth1_n=500", "oth1_n=500 oth1_name=application_unnamed_thread")


class Lt2Tests(unittest.TestCase):
    def test_lt2_line_parses_and_joins_to_lt(self):
        lines = [f"10:00:00.000 > {lt_line(n=45, win=8, dec_us=600000)}",
                 f"10:00:00.010 > {lt2_line(n=45, win=8)}",
                 "10:00:00.020 > LT2_INFO: sampler=1 timer=TIMER4 irq_priority=0 sample_us=250 main_thread=536871000",
                 f"10:00:05.000 > {lt_line(n=46, win=0, dec_us=433000)}",
                 f"10:00:05.010 > {lt2_line(n=46, win=0, smp_main=1700)}"]
        report = alt.analyse(lines)
        self.assertEqual(report["lt2_info"][0]["sample_us"], 250)
        s = report["phases"]["summary"]
        self.assertEqual(s["lt2_lines"], 2)
        self.assertEqual(s["subsequent"], 1)  # win 0 (750-sample first decision) is kept apart
        self.assertAlmostEqual(s["main_frac"]["median"], 0.75)
        self.assertAlmostEqual(s["oth_frac"]["median"], 500 / 2400)
        self.assertAlmostEqual(s["main_only_us"]["median"], 450000)
        self.assertEqual(s["ticks_by_other_thread"], {"application_unnamed_thread": 1000})
        self.assertEqual(s["ticks_by_interrupted_exception"], {"43": 200})
        self.assertEqual(s["stk_free_min"], 1200)
        self.assertEqual(report["phases"]["decisions"][0]["dec_us"], 600000)
        self.assertEqual(s["phases_us"]["filt_ir"]["median"], 260000)
        self.assertIn("decision sub-phases", alt.markdown(report, "x.txt"))

    def test_cycles_per_insn_uses_emulated_counts(self):
        rows, s = alt.analyse_lt2([alt.parse_kv(lt2_line(win=3))], [],
                                  expected_insn={p: 64000 / 1.5 for p in alt.LT2_PHASES})
        self.assertAlmostEqual(rows[0]["cycles_per_insn"]["ingest"], 1.5)
        self.assertAlmostEqual(s["expected_us_at_timing_cpi"]["ingest"], 64000 / 1.5 * alt.TIMING_IMAGE_CPI / 64)

    def test_firmware_lt2_keys_match_and_worst_case_fits(self):
        keys, fmts = lt2_formats()
        self.assertEqual(keys, alt.LT2_PHASES)
        self.assertEqual(len(fmts), 6)  # header, per phase, summary, per thread, per ISR, stack
        worst = {"%lu": "4294967295", "%ld": "-2147483648", "%d": "6"}

        def fill(fmt, s_values):
            out = fmt
            for v in s_values:
                out = out.replace("%s", v, 1)
            for k, v in worst.items():
                out = out.replace(k, v)
            return out

        line = fill(fmts[0], [])
        line += "".join(fill(fmts[1], [k, k]) for k in keys)
        line += fill(fmts[2], [])
        line += fill(fmts[3], ["x" * 32]) * 4
        line += fill(fmts[4], []) * 6
        line += fill(fmts[5], [])
        self.assertTrue(line.endswith("\r\n"))
        self.assertLess(len(line), 1400)  # lt_line[1400] incl. the terminator
        rec = alt.parse_kv(line[len("LT2:"):])
        for p in alt.LT2_PHASES:
            self.assertIn(f"{p}_us", rec)
            self.assertIn(f"{p}_c", rec)
        for k in ("phases_us", "smp_n", "smp_main", "smp_oth", "smp_isr", "smp_full", "smp_cost_c",
                  "stk_size", "stk_free"):
            self.assertIn(k, rec)


if __name__ == "__main__":
    unittest.main()

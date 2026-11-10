"""scripts/analyze_spo2_compare.py: SPO2CMP: lines from the -DOXY_SPO2_MAXIM image (ADR-0010). No serial port.

1. A synthetic capture with hand-computed expectations: two runs (n drops), a missing n, -999 values,
   nan correlation, an RF value that rounds above 100, and ordinary Red: status lines mixed in.
2. The existing capture parsers (build/v2/sit_summary.py, build/v2/session_summary.py) give the same
   result with and without SPO2CMP: lines, so the new line does not disturb them.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_spo2_compare as cmp  # noqa: E402

STATUS = ("Red: 50000  IR: 60000  Hz: 50.02  SpO2_est: {spo2}  KAN_Prob: 0.12  Candidate: NO  Confirmed: NO  "
          "Win: {win}  Samples: {samples}  TopFeature: ir_std  Signal: OK  Status: Normal (Prob: 0.12 | Win: {win})")


def cmp_line(n, maxim, mv, rf, rv, corr, shown):
    return f"SPO2CMP: n={n} maxim={maxim} maxim_valid={mv} rf={rf} rf_valid={rv} rf_corr={corr} shown={shown}"


# Run 1, n=1..7; then run 2, n=1,2,4 (n=3 lost). Expected values worked by hand in the comments.
#            n  maxim  mv  rf       rv  corr     fw shown | Maxim display   | RF (round half up) display
SYNTHETIC = [
    cmp_line(1, 97, 1, "96.4", 1, "0.950", "97"),     # 97 accepted       | 96 accepted
    cmp_line(2, 99, 1, "97.6", 1, "0.940", "99"),     # 99 accepted       | 98 accepted
    STATUS.format(spo2="99.0%", win=1, samples=500),
    cmp_line(3, 98, 1, "-999.0", 0, "0.420", "98"),   # 98 accepted       | invalid #1, hold 98
    cmp_line(4, 65, 1, "99.5", 1, "0.910", "98"),     # valid, <70: #1, 98 | 100 accepted
    cmp_line(5, -999, 0, "100.6", 1, "0.900", "98"),  # #2, hold 98       | 101 > 100: #1, hold 100
    cmp_line(6, -999, 0, "-999.0", 0, "nan", "-"),    # #3, dash          | #2, hold 100
    STATUS.format(spo2="---", win=2, samples=750),
    cmp_line(7, -999, 0, "93.2", 1, "0.880", "-"),    # dash              | 93 accepted
    cmp_line(1, -999, 0, "-999.0", 0, "nan", "-"),    # run 2: no value   | no value
    cmp_line(2, 95, 1, "94.5", 1, "0.990", "95"),     # 95 accepted       | 95 accepted
    cmp_line(4, 96, 1, "95.2", 1, "0.990", "96"),     # 96 accepted       | 95 accepted (n=3 lost)
]


def timestamped(lines, start="10:00:00.000", step_s=0.5):
    t = datetime.strptime(start, "%H:%M:%S.%f")
    out = ["--- capture COM8 115200 start 2026-11-05T10:00:00 duration 30s (build/v2/cap.py)"]
    for i, line in enumerate(lines):
        out.append(f"{(t + timedelta(seconds=i * step_s)).strftime('%H:%M:%S.%f')[:-3]} > {line}")
    return "\n".join(out) + "\n"


class AnalyserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cap = Path(self.tmp.name) / "cap.txt"
        self.cap.write_text(timestamped(SYNTHETIC), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_and_gaps(self):
        res = cmp.analyse(self.cap)
        self.assertEqual(res["computations"], 10)
        self.assertEqual([(r["first_n"], r["last_n"], r["lines"], r["missing_n"]) for r in res["runs"]],
                         [(1, 7, 7, 0), (1, 4, 3, 1)])

    def test_maxim(self):
        m = cmp.analyse(self.cap)["algorithms"]["maxim"]
        self.assertEqual(m["computations"], 10)
        self.assertAlmostEqual(m["raw_valid_share"], 6 / 10)
        self.assertAlmostEqual(m["shown_share"], 7 / 10)
        self.assertEqual((m["shown_median"], m["shown_min"], m["shown_max"]), (98, 95, 99))
        self.assertEqual(m["step_pairs"], 2)              # (97,99) and (99,98); 95@n2/96@n4 are not consecutive
        self.assertEqual(m["step_median_abs"], 1.5)
        self.assertEqual(m["first_shown"], [{"run": 1, "n": 1, "contact_s": 4}, {"run": 2, "n": 2, "contact_s": 5}])

    def test_rf_same_rule(self):
        r = cmp.analyse(self.cap)["algorithms"]["rf"]
        self.assertAlmostEqual(r["raw_valid_share"], 7 / 10)
        self.assertAlmostEqual(r["shown_share"], 9 / 10)
        self.assertEqual((r["shown_median"], r["shown_min"], r["shown_max"]), (98, 93, 100))
        self.assertEqual(r["step_pairs"], 1)              # (96,98) only
        self.assertEqual(r["step_median_abs"], 2)
        self.assertEqual(r["first_shown"], [{"run": 1, "n": 1, "contact_s": 4}, {"run": 2, "n": 2, "contact_s": 5}])

    def test_both_shown_and_firmware_cross_check(self):
        res = cmp.analyse(self.cap)
        self.assertEqual(res["both_shown"], {"n": 7, "median_abs_diff": 1, "max_abs_diff": 2})
        self.assertEqual(res["firmware_shown_mismatches"], 0)
        bad = Path(self.tmp.name) / "bad.txt"
        bad.write_text(timestamped([cmp_line(1, 97, 1, "96.0", 1, "0.9", "-")]), encoding="utf-8")
        self.assertEqual(cmp.analyse(bad)["firmware_shown_mismatches"], 1)

    def test_round_half_up(self):
        self.assertEqual([cmp.round_half_up(x) for x in (94.5, 95.5, 99.49, 100.5)], [95, 96, 99, 101])

    def test_outputs_refuse_to_overwrite_and_md_leads_with_d10(self):
        js, md = Path(self.tmp.name) / "o.json", Path(self.tmp.name) / "o.md"
        run = lambda: subprocess.run([sys.executable, str(ROOT / "scripts/analyze_spo2_compare.py"), str(self.cap),
                                      "--json", str(js), "--md", str(md)], capture_output=True, text=True)
        first = run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(js.read_text())["algorithms"]["maxim"]["computations"], 10)
        self.assertIn(cmp.D10, md.read_text(encoding="utf-8").splitlines()[2])
        before = (js.read_bytes(), md.read_bytes())
        second = run()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("refusing to overwrite", second.stderr)
        self.assertEqual((js.read_bytes(), md.read_bytes()), before)

    def test_no_spo2cmp_lines(self):
        empty = Path(self.tmp.name) / "none.txt"
        empty.write_text(timestamped([STATUS.format(spo2="97.0%", win=1, samples=500)]), encoding="utf-8")
        res = cmp.analyse(empty)
        self.assertEqual(res["computations"], 0)
        self.assertIsNone(res["algorithms"]["maxim"]["shown_median"])


SIT = ROOT / "build/v2/sit_summary.py"
SESSION = ROOT / "build/v2/session_summary.py"


@unittest.skipUnless(SIT.exists() and SESSION.exists(), "build/v2 parsers are local scratch (gitignored)")
class ExistingParsersIgnoreSpo2cmp(unittest.TestCase):
    """Same capture with and without SPO2CMP: lines -> the same summaries."""

    def capture_lines(self, with_cmp):
        lines = []
        for i in range(60):  # 30 s at 2 status lines/s, a decision every 5 s
            win = i // 10 + 1
            spo2 = "---" if i < 8 else f"{96 + i % 3}.0%"
            lines.append(STATUS.format(spo2=spo2, win=win, samples=250 * (win + 1)))
            if with_cmp and i % 2 == 1:
                lines.append(cmp_line(i // 2 + 1, 97 if i > 8 else -999, int(i > 8), "96.4" if i > 8 else "-999.0",
                                      int(i > 8), "0.950" if i > 8 else "nan", "97" if i > 8 else "-"))
        return lines

    def write(self, tmp, name, with_cmp):
        path = Path(tmp) / name
        lines = self.capture_lines(with_cmp)
        # Timestamps follow the status lines only, so both files carry identical times for them.
        t0, out, k = datetime(2026, 11, 5, 10, 0, 0), ["--- capture COM8 115200 start 2026-11-05T10:00:00 duration 30s"], 0
        for line in lines:
            if not line.startswith("SPO2CMP"):
                k += 1
            out.append(f"{(t0 + timedelta(seconds=0.5 * k)).strftime('%H:%M:%S.%f')[:-3]} > {line}")
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path

    def test_same_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, mixed = self.write(tmp, "plain.txt", False), self.write(tmp, "mixed.txt", True)
            self.assertGreater(mixed.read_text().count("SPO2CMP:"), 20)
            marks = Path(tmp) / "marks.txt"
            marks.write_text("10:00:00 SIT seated\n10:00:31 END\n", encoding="utf-8")
            sit = [subprocess.run([sys.executable, str(SIT), str(p)], capture_output=True, text=True) for p in (plain, mixed)]
            for done in sit:
                self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("SpO2_est median", sit[0].stdout)
            self.assertEqual(sit[0].stdout, sit[1].stdout)
            sess = [subprocess.run([sys.executable, str(SESSION), str(p), str(marks)], capture_output=True, text=True)
                    for p in (plain, mixed)]
            for done in sess:
                self.assertEqual(done.returncode, 0, done.stderr)
            a, b = (json.loads(done.stdout) for done in sess)
            a.pop("capture"), b.pop("capture")
            self.assertGreater(a["all_decisions_in_window"]["n"], 3)
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

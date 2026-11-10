"""Host tests for the peak-RAM sampler: the sampler model on synthetic states, the analyser's
--sampler mode on a synthetic capture, and the firmware header's keys and flags. No device, no port.

Every capture here is SYNTHETIC (built in this file); its numbers are illustrative, not board data."""
import contextlib
import io
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_peak_ram as apr
import peak_ram_sampler_model as sm
import peak_ram_scan_model as m

FW = ROOT / "firmware" / "Arduino Nano 33"
SAMPLER_H = FW / "include" / "peak_ram_sampler.h"
PROBE_H = FW / "include" / "peak_ram_probe.h"
MAIN = FW / "src" / "main.cpp"
INI = FW / "platformio.ini"

# v2 probe layout from the 25 Oct PR_INFO line (nano33ble_v2_f1a_peakram).
FIXED, STATIC = 512, 94984
HEAP_BASE, STACK_TOP = 0x20017508, 0x20040000
THREADS = [  # (name, tid, lo, size, loc)
    ("main", 0x20008CE0, 0x2000BBF0, 32768, "bss"),
    ("rtx_idle", 0x20008730, 0x20008778, 512, "bss"),
    ("application_unnamed_thre", 0x2001BA98, 0x2001BAF0, 4096, "heap"),
    ("rtx_timer", 0x200086EC, 0x20008978, 768, "bss"),
    ("USBevt", 0x200179A8, 0x200179F8, 256, "heap"),
]


def table(depths):
    """Threads whose saved SP sits `depth` below the top."""
    return [sm.Thread(tid, lo, size, lo + size - d) for (_, tid, lo, size, _), d in zip(THREADS, depths)]


def state(depths, curr_depth=4616, heap_live=23508, msp_depth=72, **kw):
    ths = table(depths)
    main = ths[0]
    return sm.State(threads=ths, curr=main.tid, psp=main.top - curr_depth, msp=STACK_TOP - msp_depth,
                    stack_top=STACK_TOP, heap_base=HEAP_BASE, brk=HEAP_BASE + heap_live, **kw)


CONST = sm.s_const(FIXED, STATIC, [t[3] for t in THREADS])


class SamplerModelTests(unittest.TestCase):
    def test_heap_in_use(self):
        self.assertEqual(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 1000, []), 1000)
        self.assertEqual(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 1000, [(HEAP_BASE + 200, 64), (HEAP_BASE + 600, 40)]), 896)
        self.assertIsNone(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 1000, [(HEAP_BASE + 980, 40)]))   # past the break
        self.assertIsNone(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 1000, [(HEAP_BASE - 8, 8)]))      # below __end__
        self.assertIsNone(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 1000, [(HEAP_BASE, 0)]))
        self.assertIsNone(sm.heap_in_use(HEAP_BASE, HEAP_BASE + 4000, [(HEAP_BASE + 8 * i, 8) for i in range(33)]))

    def test_a_sample_is_in_use_at_that_instant(self):
        depths = [0, 120, 400, 104, 136]
        st = state(depths, curr_depth=4616, heap_live=23508, msp_depth=72)
        s = sm.Sampler(table(depths), CONST)
        total = s.tick(st)
        sp_of = {t.tid: t.sp for t in st.threads}
        sp_of[st.curr] = st.psp
        expect = sm.in_use(FIXED, STATIC, [(t, loc) for t, (*_, loc) in zip(st.threads, THREADS)], sp_of,
                           23508, st.msp, STACK_TOP)
        self.assertEqual(total, expect)
        self.assertEqual(total, FIXED + STATIC - 34048 + 23508 - 4352 + 4616 + 120 + 400 + 104 + 136 + 72)
        self.assertEqual((s.max_heap, s.max_msp, s.max_stk), (23508, 72, 4616 + 120 + 400 + 104 + 136))

    def test_samples_never_exceed_whole_hw(self):
        # Each sampled SP points at a written word, so its depth <= the touched peak; the heap term
        # <= break - __end__ <= heap peak; the MSP read <= the MSP touched peak.
        rng = random.Random(42)
        peaks = [4616, 392, 408, 104, 136]
        heap_peak, msp_peak = 23596, 384
        space_of = {tid: size - p for (_, tid, _, size, _), p in zip(THREADS, peaks)}
        ths = table([0] * 5)
        hw = sm.whole_hw(FIXED, STATIC, [(t, loc) for t, (*_, loc) in zip(ths, THREADS)], space_of, heap_peak, msp_peak)
        s = sm.Sampler(ths, CONST)
        for _ in range(2000):
            depths = [rng.randint(64, p) for p in peaks]
            free = [(HEAP_BASE + 1000, 88)] if rng.random() < 0.3 else []
            st = state(depths, curr_depth=depths[0], heap_live=rng.randint(22000, heap_peak),
                       msp_depth=rng.randint(64, msp_peak), free_chunks=free)
            total = s.tick(st)
            self.assertIsNotNone(total)
            self.assertLessEqual(total, hw)
        self.assertLessEqual(s.max, hw)
        # With every component at its peak at once the sample reaches whole_hw exactly.
        st = state(peaks, curr_depth=peaks[0], heap_live=heap_peak, msp_depth=msp_peak)
        self.assertEqual(s.tick(st), hw)

    def test_skips_by_cause(self):
        depths = [0, 120, 400, 104, 136]
        s = sm.Sampler(table(depths), CONST)
        self.assertIsNone(s.tick(state(depths, shcsr=1 << 10)))            # PendSV active
        self.assertIsNone(s.tick(state(depths, exc_return=0xFFFFFFF1)))    # from handler mode
        self.assertIsNone(s.tick(state(depths, lock_owner=0x20008CE0, lock_count=1)))   # malloc mutex held
        self.assertIsNone(s.tick(state(depths, free_chunks=[(HEAP_BASE - 4, 8)])))
        bad = state(depths)
        bad.threads[1].obj_id = 0                                          # deleted thread
        self.assertIsNone(s.tick(bad))
        stray = state(depths)
        stray.curr = 0x20030000                                            # running thread not in the table
        self.assertIsNone(s.tick(stray))
        self.assertEqual((s.n, s.ok, s.skk, s.skl, s.skh, s.skt), (6, 0, 2, 1, 1, 2))

    def test_stale_owner_after_release_is_not_a_skip(self):
        # RTX leaves owner_thread set after the last release (design section 8, 1 Nov note): a mutex
        # with an owner and lock 0 is free. The 31 Oct test (owner or lock) skipped every such tick.
        depths = [0, 120, 400, 104, 136]
        s = sm.Sampler(table(depths), CONST)
        self.assertIsNotNone(s.tick(state(depths, lock_owner=0x20008CE0, lock_count=0)))
        self.assertIsNone(s.tick(state(depths, lock_owner=0x20008CE0, lock_count=2)))   # recursive hold
        self.assertEqual((s.n, s.ok, s.skl), (2, 1, 1))


MAIN_TID, BLE_TID = 0x20008E88, 0x2001BC40   # v2 thread ids from the 1 Nov PR_INFO line
MALLOC_MUTEX = 0x20008EA4                     # malloc_mutex_obj in the r2 v2 ELF (nm)


class MallocMutexLookupTests(unittest.TestCase):
    """prs_find_malloc_mutex against the RTX semantics read from the core's ELF."""

    def setUp(self):
        self.rtx = sm.Rtx()
        self.m = sm.Mutex(MALLOC_MUTEX)

    def test_rtx_release_keeps_the_owner(self):
        self.assertEqual(self.rtx.acquire(self.m, MAIN_TID), 0)
        self.assertIs(self.rtx.head(MAIN_TID), self.m)
        self.assertEqual(self.rtx.release(self.m, MAIN_TID), 0)
        self.assertEqual((self.m.owner, self.m.lock, self.rtx.head(MAIN_TID)), (MAIN_TID, 0, None))

    def test_31_oct_lookup_returns_null_on_this_rtx(self):
        # The 1 Nov board result: hlock=0x00000000 on both images.
        self.assertIsNone(sm.find_malloc_mutex_d7c9d54(self.rtx, self.m, MAIN_TID))
        self.assertEqual((self.m.owner, self.m.lock), (MAIN_TID, 0))

    def test_lookup_finds_the_mutex(self):
        m, k = sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID)
        self.assertIs(m, self.m)
        self.assertEqual(k, {"step": 0, "ipsr": 0, "mask": 0, "self": MAIN_TID, "pre": 0, "head": MALLOC_MUTEX,
                             "id": 0xF5, "own": 1, "cnt": 1, "post": 0, "rel": 0, "stale": 1})

    def test_lookup_with_another_mutex_already_owned(self):
        other = sm.Mutex(0x20013DBC)
        self.rtx.acquire(other, MAIN_TID)
        m, k = sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID)
        self.assertIs(m, self.m)
        self.assertEqual((k["pre"], k["head"], k["post"]), (other.addr, MALLOC_MUTEX, other.addr))

    def test_lookup_survives_a_hand_over_on_release(self):
        # A thread that blocked on the mutex while the lookup held it takes it over in the release:
        # lock stays 1, owner_thread is the waiter. The lookup still succeeds; lk_rel/lk_stale show it.
        m, k = sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID, while_held=lambda r: self.m.waiters.append(BLE_TID))
        self.assertIs(m, self.m)
        self.assertEqual((k["step"], k["rel"], k["stale"]), (0, 1, 0))
        self.assertEqual((self.m.owner, self.m.lock), (BLE_TID, 1))

    def test_each_step_can_fail(self):
        self.assertEqual(sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID, masked=True)[1]["step"], 1)
        self.assertEqual(sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID, ipsr=27)[1]["step"], 1)
        self.assertEqual((self.m.owner, self.m.lock), (0, 0))                 # nothing was locked
        self.assertEqual(sm.find_malloc_mutex(self.rtx, self.m, 0)[1]["step"], 2)
        self.rtx.acquire(self.m, MAIN_TID)                                     # already held: recursion
        m, k = sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID)
        self.assertIsNone(m)
        self.assertEqual((k["step"], self.m.lock), (3, 1))
        self.rtx.release(self.m, MAIN_TID)
        self.m.obj_id = 0
        self.assertEqual(sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID)[1]["step"], 4)
        self.m.obj_id = sm.RTX_ID_MUTEX

    def test_found_mutex_gives_samples_after_the_lookup(self):
        # End to end: after the lookup the owner is stale; the r2 test samples, the 31 Oct test skipped.
        m, _ = sm.find_malloc_mutex(self.rtx, self.m, MAIN_TID)
        depths = [0, 120, 400, 104, 136]
        s = sm.Sampler(table(depths), CONST)
        for _ in range(100):
            self.assertIsNotNone(s.tick(state(depths, lock_owner=m.owner, lock_count=m.lock)))
        old_skips = sum(1 for _ in range(100) if m.owner != 0 or m.lock != 0)
        self.assertEqual((s.ok, s.skl, old_skips), (100, 0, 100))

    def test_window_restarts_and_max_stays(self):
        depths = [0, 120, 400, 104, 136]
        s = sm.Sampler(table(depths), CONST)
        big = s.tick(state(depths, curr_depth=4616))
        r1 = s.report()
        small = s.tick(state(depths, curr_depth=1760))
        r2 = s.report()
        self.assertEqual((r1["s_win"], r1["samp_max"]), (big, big))
        self.assertEqual((r2["s_win"], r2["samp_max"]), (small, big))
        self.assertEqual(r2["max_depth"][0], 4616)
        self.assertEqual(r2["cmax_depth"][0], 4616)


# ---- synthetic sampler capture -------------------------------------------------------------
INFO = ("PR_INFO: v=1 img=nano33ble_v2_f1a_peakram_sampler model=v2 force=0 hook=ctor101 ram=0x20000000+262144 "
        "fixed=0x20000000+512 data=0x20000200+1952 bss=0x200009A0+93032 heap=0x20017508+165624 "
        "msp=0x2003FC00+1024 gap=0 pattern=0x9E3779B9 rtx_fill=0xCCCCCCCC wm=1 period_ms=5000 heap_painted=1 "
        "msp_painted=1 main_painted=1 setup_painted=1 heap_paint=0x20017508+165624 msp_paint=0x2003FC00+960 "
        "main_paint=0x2000BBF4+32452 brk_ctor=0x20017508 brk_setup=0x2001CFD4 nthr=5 "
        + " ".join(f"T.{n}@0x{tid:08X}=0x{lo:08X}+{size}/{loc}" for n, tid, lo, size, loc in THREADS)
        + f" smp=1 smp_timer=TIMER4 smp_hz=4000 smp_prio=7 prio_svc=6 prio_pendsv=7 prio_systick=7 "
          f"vtor=0x20000000 hlock=0x20008CBC s_const={CONST} s_nthr=5")
PEAKS = [4616, 392, 408, 104, 136]
MSP_PEAK = 384


def pr_line(i, **over):
    """PR: line i of a consistent synthetic sampler capture; keyword overrides patch fields."""
    heap = 23508
    untouched = (165624 - heap) + (1024 - MSP_PEAK) + sum(size - p for (_, _, _, size, _), p in zip(THREADS, PEAKS))
    hw = m.RAM_SIZE - untouched
    rec = {"t_ms": 30000 + 5000 * i, "rep": 6 + i, "dec": 3 + i, "lf": 1500 + 250 * i, "lnf": 2,
           "heap": (165624, heap), "msp": (1024, MSP_PEAK), "msp_loop": MSP_PEAK, "main_pre": 96, "main_setup": 792,
           "main_loop": PEAKS[0], "threads": [(n, tid, size, p, loc) for (n, tid, _, size, loc), p in zip(THREADS, PEAKS)],
           "brk": HEAP_BASE + heap, "heap_hw": HEAP_BASE + heap, "hs_cur": 22580, "hs_max": 22580, "hs_ovh": 928,
           "hs_fail": 0, "static": STATIC, "fixed": FIXED, "untouched": untouched, "ovl": 0, "whole_hw": hw,
           "ram": m.RAM_SIZE, "scan_us": 12000, "rpt_alloc": 0}
    samp = hw - 600
    snap = {"s_n": 100000 + 20000 * i, "s_ok": 100000 + 19990 * i, "s_skk": 0, "s_skl": 10 * i, "s_skh": 0, "s_skt": 0,
            "samp_max": samp, "s_win": samp - (0 if i % 7 == 3 else 2000), "sm_heap": heap, "sm_msp": 72,
            "sm_stk": 5000, "smx_heap": heap, "smx_msp": 72, "max_depth": [4500, 300, 300, 100, 100],
            "cmax_depth": [4560, 330, 300, 100, 120]}
    tail = dict(ble=1, sub=1, ble_conn=1, ble_disc=0, hl=heap, hl_st=1, s_set=1, cyc_max=420, cyc_sum=300 * snap["s_n"])
    for k, v in over.items():
        if k in rec:
            rec[k] = v
        elif k in snap:
            snap[k] = v
        else:
            tail[k] = v
    keys = [f"{n}@0x{tid:08X}" for n, tid, *_ in THREADS]
    return m.format_pr_line(rec) + " " + sm.format_fields(snap, thread_keys=keys, **tail)


def capture(n=36, info=INFO, extra=None, per_line=None):
    lines = ["--- capture COM8 115200 (SYNTHETIC sampler test capture, not board data)"]
    t = 0.0
    for i in range(n):
        if i % 12 == 0:
            lines.append(f"10:00:{t % 60:06.3f} > {info}".replace("10:00:", f"10:{int(t // 60):02d}:"))
        over = (per_line or {}).get(i, {})
        stamp = f"10:{int(t // 60):02d}:{t % 60:06.3f}"
        lines.append(f"{stamp} > Red: 90000  IR: 95000  Hz: {over.pop('hz', 50.05)}  SpO2_est: 96.0%  KAN_Prob: 0.10  "
                     "Candidate: NO  Confirmed: NO  Win: 3  Samples: 750  TopFeature: kurtosis  Signal: OK  "
                     "Status: Normal (Prob: 0.10 | Win: 3)")
        lines.append(f"{stamp} > {pr_line(i, **over)}")
        t += 5.0
    lines.append(f"10:{int(t // 60):02d}:{t % 60:06.3f} > Red: 90000  IR: 95000  Hz: 50.04  Status: Normal")
    return lines + (extra or [])


class SamplerAnalyserTests(unittest.TestCase):
    def failing(self, lines):
        rep = apr.analyse(lines, "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        return {c["name"] for c in rep["checks"] if not c["ok"]}

    def test_synthetic_capture_is_valid(self):
        rep = apr.analyse(capture(), "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        self.assertTrue(rep["valid"], [c for c in rep["checks"] if not c["ok"]])
        names = [c["name"] for c in rep["checks"]]
        for n in ("identity", "config", "parse", "pr_lines", "gaps", "no_reboot", "regions", "monotonic", "finger",
                  "central", "rate", "decisions", "samples", "skips", "bound_order", "heap_source"):
            self.assertIn(n, names)
        z, f = rep["sampler"], rep["paper_figure"]
        self.assertEqual(z["lower_capture"], z["samp_max"])           # a full window inside the capture
        self.assertEqual(f["upper_bytes"], rep["summary"]["whole_hw"]["final"])
        self.assertLess(f["lower_bytes"], f["upper_bytes"])
        self.assertIn("finger on, dashboard connected", f["wording"])
        self.assertIn("Sampler:", apr.markdown(rep, "synthetic"))

    def test_lower_end_ignores_the_first_window(self):
        # The first line's window began before the capture; it does not count for the lower end.
        per_line = {i: {"s_win": 80000} for i in range(1, 36)}
        per_line[0] = {"s_win": 86000}
        rep = apr.analyse(capture(per_line=per_line), "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        self.assertTrue(rep["valid"], [c for c in rep["checks"] if not c["ok"]])
        self.assertEqual(rep["sampler"]["lower_capture"], 80000)
        self.assertEqual(rep["paper_figure"]["lower_bytes"], 80000)
        self.assertEqual(rep["paper_figure"]["upper_bytes"], 86644)

    def test_each_criterion_can_fail(self):
        cases = {
            "finger": {i: {"lnf": 2 + 20 * i} for i in range(36)},                       # 250 vs 20 per line: < 95%
            "central": {20: {"ble_disc": 1}, **{i: {"ble_disc": 1} for i in range(21, 36)}},
            "rate": {10: {"hz": 49.4}},
            "samples": {i: {"s_n": 100000 + 17000 * i, "s_ok": 100000 + 16990 * i} for i in range(36)},
            "skips": {i: {"s_skl": 1100 * i} for i in range(36)},
            "bound_order": {35: {"samp_max": 999999, "s_win": 10}},
            "heap_source": {5: {"hl": 23500}},
            "regions": {7: {"s_set": 0}},
        }
        for name, per_line in cases.items():
            with self.subTest(name=name):
                self.assertIn(name, self.failing(capture(per_line=per_line)))
        self.assertIn("decisions", self.failing(capture(per_line={i: {"dec": 3 + i // 2} for i in range(36)})))
        self.assertIn("central", self.failing(capture(per_line={3: {"sub": 0}})))
        self.assertIn("central", self.failing(capture(extra=["10:03:01.000 > BLE connected to: 11:22:33:44:55:66"])))
        self.assertIn("config", self.failing(capture(info=INFO.replace("force=0", "force=1"))))
        self.assertIn("config", self.failing(capture(info=INFO.replace("prio_pendsv=7", "prio_pendsv=6"))))
        self.assertIn("config", self.failing(capture(info=INFO.replace("hlock=0x20008CBC", "hlock=0x00000000"))))
        self.assertIn("identity", self.failing(capture(info=INFO.replace("_peakram_sampler", "_peakram"))))

    def test_old_captures_fail_sampler_mode_and_old_mode_is_unchanged(self):
        cap = (ROOT / "outputs" / "peak-ram-board-20261025" / "nano33ble_v2_f1a_peakram_nofinger_180s.txt")
        lines = cap.read_text(encoding="utf-8", errors="replace").splitlines()
        rep = apr.analyse(lines, "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        self.assertFalse(rep["valid"])
        self.assertIsNone(rep["paper_figure"])
        old = apr.analyse(lines, "nano33ble_v2_f1a_peakram")
        self.assertTrue(old["valid"])
        self.assertNotIn("mode", old)
        self.assertNotIn("sampler", old)

    def test_start_diagnostics_are_parsed_and_change_no_check(self):
        diag = ("st_hlock=1 st_set=1 st_vtor=1 lk_step=0 lk_ipsr=0 lk_mask=0 lk_self=0x20008E88 lk_pre=0x00000000 "
                "lk_head=0x20008EA4 lk_id=0x000000F5 lk_own=1 lk_cnt=1 lk_post=0x00000000 lk_rel=0 lk_stale=1")
        info = INFO.replace("hlock=0x20008CBC", "hlock=0x20008EA4") + " " + diag
        rep = apr.analyse(capture(info=info), "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        self.assertTrue(rep["valid"], [c for c in rep["checks"] if not c["ok"]])
        d = rep["sampler"]["info"]["diag"]
        self.assertEqual(list(d), apr.INFO_SAMPLER_DIAG_INT_KEYS + apr.INFO_SAMPLER_DIAG_HEX_KEYS)
        self.assertEqual((d["lk_step"], d["lk_head"]), ("0", "0x20008EA4"))
        self.assertIn("start diagnostics", apr.markdown(rep, "synthetic"))
        # A failed start: config fails on smp/hlock as pre-registered; the detail carries the steps.
        failed = (INFO.replace("smp=1 ", "smp=0 ").replace("hlock=0x20008CBC", "hlock=0x00000000") + " "
                  + diag.replace("st_hlock=1", "st_hlock=0").replace("lk_step=0", "lk_step=7"))
        rep = apr.analyse(capture(info=failed), "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        cfg = next(c for c in rep["checks"] if c["name"] == "config")
        self.assertFalse(cfg["ok"])
        self.assertIn("smp=0", cfg["detail"])
        self.assertIn("hlock=0", cfg["detail"])
        self.assertIn("lk_step=7", cfg["detail"])
        # Without the diagnostics (31 Oct format) parsing and checks are as before.
        rep = apr.analyse(capture(), "nano33ble_v2_f1a_peakram_sampler", sampler=True)
        self.assertNotIn("diag", rep["sampler"]["info"])

    def test_1_nov_identify_captures_fail_config(self):
        # Board data (read only): the two 1 Nov identify captures of the 31 Oct images.
        run = ROOT / "outputs" / "peak-ram-sampler-board-20261101"
        for env in apr.SAMPLER_IMAGES:
            with self.subTest(env=env):
                lines = (run / f"{env}_identify.txt").read_text(encoding="utf-8", errors="replace").splitlines()
                rep = apr.analyse(lines, env, sampler=True)
                checks = {c["name"]: c for c in rep["checks"]}
                self.assertFalse(rep["valid"])
                self.assertIsNone(rep["paper_figure"])
                self.assertTrue(checks["identity"]["ok"])
                self.assertFalse(checks["config"]["ok"])
                self.assertEqual(checks["config"]["detail"], "hlock=0, smp=0")
                self.assertTrue(checks["parse"]["ok"], checks["parse"]["detail"])
                self.assertFalse(checks["samples"]["ok"])            # s_n = 0 on every line
                self.assertTrue(all(r["sampler"]["s_n"] == 0 for r in apr.parse_capture(lines)["pr"]))

    def test_main_writes_sampler_criteria(self):
        with tempfile.TemporaryDirectory() as d:
            capf = Path(d) / "cap.txt"
            capf.write_text("\n".join(capture()) + "\n", encoding="utf-8")
            out = Path(d) / "rep"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = apr.main([str(capf), "--output", str(out), "--expect-image", "nano33ble_v2_f1a_peakram_sampler",
                               "--sampler"])
            self.assertEqual(rc, 0)
            text = (out / "report.json").read_text(encoding="utf-8")
            self.assertIn('"max_skip_fraction": 0.05', text)
            self.assertIn('"min_tick_fraction": 0.9', text)


R2_ARCHIVE = ROOT / "outputs" / "firmware-peakram-sampler-prepared-r2"
OLD_ARCHIVE = ROOT / "outputs" / "firmware-peakram-sampler-prepared"
SESSION_1NOV = ROOT / "outputs" / "peak-ram-sampler-board-20261101"
IMAGES_R2 = {"nano33ble_peakram_sampler": ("908940113497dea93344c80c3ceba4efddf027e6f929e66e0a916ba63c1c2920", 349308),
             "nano33ble_v2_f1a_peakram_sampler": ("bd3a494d23108d1a065473da2ca6e7e4675e646400c2124b53bba66321abd465", 350660)}
IMAGES_OLD = {"nano33ble_peakram_sampler": ("acbbdc612fae571233aa3c6c5e4ea8f5dd7d52536716a79101e1b8c40f623d95", 348828),
              "nano33ble_v2_f1a_peakram_sampler": ("997957c60389e5a8c227ebad19dfac3b1acd1e44b30409f40bea14f864e33b20", 350172)}


class ArchiveBindingTests(unittest.TestCase):
    """The analyser's flash binding (upload_binding, behind --image and the flash-log option) on the r2
    and 31 Oct archives. Host only: the flash logs are the 1 Nov session's own files (read only) or
    copies of them with the byte count changed."""

    @staticmethod
    def flash_log(env):
        return next(SESSION_1NOV.glob(f"{env}-*-upload.log"))

    def test_archives_hold_the_listed_images(self):
        for root, images in ((R2_ARCHIVE, IMAGES_R2), (OLD_ARCHIVE, IMAGES_OLD)):
            for env, (sha, size) in images.items():
                with self.subTest(root=root.name, env=env):
                    image = root / env / "firmware.bin"
                    self.assertEqual(apr.sha256_file(image), sha)
                    self.assertEqual(image.stat().st_size, size)

    def test_1_nov_logs_bind_the_31_oct_images_only(self):
        for env, (sha, size) in IMAGES_OLD.items():
            with self.subTest(env=env):
                up = apr.upload_binding(OLD_ARCHIVE / env / "firmware.bin", self.flash_log(env))
                self.assertTrue(up["ok"], up)
                self.assertEqual((up["bytes_written"], up["archive_listed_sha256"]), (size, sha))
                up = apr.upload_binding(R2_ARCHIVE / env / "firmware.bin", self.flash_log(env))
                self.assertFalse(up["ok"])                          # another image's write
                self.assertEqual(up["archive_listed_sha256"], IMAGES_R2[env][0])

    def test_r2_archive_binds_a_matching_write(self):
        for env, (sha, size) in IMAGES_R2.items():
            with self.subTest(env=env), tempfile.TemporaryDirectory() as d:
                old_size = IMAGES_OLD[env][1]
                text = self.flash_log(env).read_text(errors="replace")
                log = Path(d) / "flash.log"
                log.write_text(text.replace(f"Write {old_size} bytes", f"Write {size} bytes"), encoding="utf-8")
                up = apr.upload_binding(R2_ARCHIVE / env / "firmware.bin", log)
                self.assertTrue(up["ok"], up)
                self.assertEqual((up["image_sha256"], up["archive_listed_sha256"]), (sha, sha))
                log.write_text(text.replace(f"Write {old_size} bytes", f"Write {size} bytes").replace("Verify successful", ""),
                               encoding="utf-8")
                self.assertFalse(apr.upload_binding(R2_ARCHIVE / env / "firmware.bin", log)["ok"])

    def test_r2_receipts(self):
        for env, (sha, size) in IMAGES_R2.items():
            with self.subTest(env=env):
                rec = json.loads((R2_ARCHIVE / env / "build-receipt.json").read_text(encoding="utf-8"))
                self.assertEqual((rec["env"], rec["firmware_bin_sha256"], rec["image_bytes"], rec["flashed"]),
                                 (env, sha, size, False))
                self.assertEqual(rec["base_commit"], "d7c9d54")


def function_body(text, signature):
    start = text.index(signature)
    depth, i = 0, text.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
        i += 1


KEY_CALL = re.compile(r'pr_(?:u|x)\("(\w+)"')


class SamplerFirmwareTests(unittest.TestCase):
    def setUp(self):
        self.h = SAMPLER_H.read_text(encoding="utf-8")
        self.probe = PROBE_H.read_text(encoding="utf-8")
        self.main = MAIN.read_text(encoding="utf-8")

    def test_report_keys_match_the_analyser(self):
        keys = KEY_CALL.findall(function_body(self.h, "static void prs_report("))
        self.assertEqual(keys, apr.PR_SAMPLER_INT_KEYS)
        self.assertIn('pr_put(" S.");', self.h)

    def test_info_keys_match_the_analyser(self):
        body = function_body(self.h, "static void prs_info()")
        keys = KEY_CALL.findall(body) + re.findall(r'pr_put\(" (\w+)=', body)
        diag = apr.INFO_SAMPLER_DIAG_INT_KEYS + apr.INFO_SAMPLER_DIAG_HEX_KEYS
        self.assertEqual(sorted(keys), sorted(apr.INFO_SAMPLER_INT_KEYS + apr.INFO_SAMPLER_STR_KEYS + diag))
        # The 31 Oct fields come first, unchanged and in order; the diagnostics are appended after them.
        keys = [a or b for a, b in re.findall(r'pr_(?:u|x)\("(\w+)"|pr_put\(" (\w+)=', body)]
        old = keys[:len(keys) - len(diag)]
        self.assertEqual(old, ["smp", "smp_timer", "smp_hz", "smp_prio", "prio_svc", "prio_pendsv", "prio_systick",
                               "vtor", "hlock", "s_const", "s_nthr"])
        self.assertEqual(sorted(keys[len(old):]), sorted(diag))
        hex_keys = re.findall(r'pr_x\("(\w+)"', body)
        self.assertEqual(sorted(hex_keys), sorted(["vtor", "hlock"] + apr.INFO_SAMPLER_DIAG_HEX_KEYS))

    def test_heap_lock_tests_follow_rtx(self):
        # RTX leaves owner_thread set after release: neither the ISR nor the lookup may need it cleared.
        body = function_body(self.h, 'extern "C" void prs_body(')
        self.assertIn("if (prs_hlock->lock != 0u) { ++prs.skl; break; }", body)
        self.assertNotIn("prs_hlock->owner_thread", body)
        self.assertNotIn("owner_thread == nullptr", self.h)
        find = function_body(self.h, "static osRtxMutex_t* prs_find_malloc_mutex()")
        for step in range(1, 8):
            self.assertIn(f"k.step = {step};", find)
        self.assertIn("else k.step = 0;", find)
        self.assertIn("return k.step == 0u ? m : nullptr;", find)
        start = function_body(self.h, "static void prs_start()")
        self.assertIn("if (prs_st_hlock == 0u || prs_st_set == 0u || prs_st_vtor == 0u) return;", start)
        self.assertLess(start.index("prs_st_vtor = SCB->VTOR == PR_RAM_START"), start.index("NRF_TIMER4->TASKS_STOP"))

    def test_constants_match(self):
        def const(name):
            return int(re.search(rf"\b{name}\s*=\s*(\d+)u;", self.h).group(1))
        self.assertEqual(const("PRS_HZ"), apr.SAMPLER_HZ)
        self.assertEqual(const("PRS_MAX_FREE"), sm.MAX_FREE)
        self.assertIn("NVIC_SetPriority(TIMER4_IRQn, prs_lvl);", self.h)
        self.assertIn("prs_lvl = prs_lvl_pendsv < prs_lvl_systick ? prs_lvl_pendsv : prs_lvl_systick;", self.h)

    def test_flags_layer_on_the_probe(self):
        self.assertTrue(self.h.lstrip().startswith("#pragma once"))
        self.assertIn("#ifdef PEAK_RAM_SAMPLER", self.h)
        self.assertIn('#ifdef PEAK_RAM_SAMPLER\n#include <reent.h>\n#include "peak_ram_sampler.h"', self.probe)
        for call in ("prs_info();", "prs_report(hs, n);", "prs_start();", "prs_poll_ble();"):
            pos = self.probe.index(call)
            before = self.probe[:pos]
            self.assertGreater(before.rfind("#ifdef PEAK_RAM_SAMPLER"), before.rfind("#endif"), call)
        self.assertIn('" force=" PR_FORCE " hook=ctor101"', self.probe)
        # main.cpp: the live gate is used only under PEAK_RAM_LIVE_GATE, nested in PEAK_RAM_PROBE.
        pos = self.main.index("return pr_live;")
        before = self.main[:pos]
        self.assertGreater(before.rfind("#ifdef PEAK_RAM_LIVE_GATE"), before.rfind("#endif"))
        self.assertIn("FINGER_PRESENT_THRESHOLD = PR_LIVE_FINGER_THRESHOLD;", self.main)

    def test_new_envs(self):
        ini = INI.read_text(encoding="utf-8")
        for env, base in (("nano33ble_peakram_sampler", "env:nano33ble_peakram"),
                          ("nano33ble_v2_f1a_peakram_sampler", "env:nano33ble_v2_f1a_peakram")):
            block = ini[ini.index(f"[env:{env}]"):].split("\n\n")[0]
            self.assertIn(f"extends = {base}", block)
            for flag in ("-DPEAK_RAM_PROBE", "-DPEAK_RAM_LIVE_GATE", "-DPEAK_RAM_SAMPLER", f".pio/build/{env}/firmware.map"):
                self.assertIn(flag, block)
        old = ini[ini.index("[env:nano33ble_peakram]"):].split("\n\n")[0]
        self.assertNotIn("LIVE_GATE", old)
        self.assertNotIn("SAMPLER", old)


if __name__ == "__main__":
    unittest.main()

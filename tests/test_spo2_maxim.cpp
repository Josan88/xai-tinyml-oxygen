// Host test of the -DOXY_SPO2_MAXIM SpO2 est module (include/spo2_maxim.h, ADR-0010) and the two
// vendored algorithms: Maxim MAXREFDES117 (src/spo2_rd117.cpp, rd117_*, one IR-index fix) and the
// serial-only comparison by Robert Fraczkiewicz (src/spo2_rfalg.cpp, rfalg_*).
// SpO2 est is display only: uncalibrated, not a TinyKAN input, not used by any decision. Nothing here
// says anything about accuracy on a person; the synthetic AC is chosen.
//
// Build (from the repo root, git-bash). Case 9 needs a second rd117 object with the upstream IR line,
// its public functions renamed rd117up_* so it links beside the fixed one:
//   Z='uv run --no-project --with ziglang==0.16.0 python -m ziglang c++'
//   $Z -std=c++17 -O2 -Wall -Wextra -DOXY_SPO2_MAXIM -DRD117_UPSTREAM_IR_INDEX \
//     -Drd117_heart_rate_and_oxygen_saturation=rd117up_heart_rate_and_oxygen_saturation \
//     -Drd117_find_peaks=rd117up_find_peaks -Drd117_peaks_above_min_height=rd117up_peaks_above_min_height \
//     -Drd117_remove_close_peaks=rd117up_remove_close_peaks -Drd117_sort_ascend=rd117up_sort_ascend \
//     -Drd117_sort_indices_descend=rd117up_sort_indices_descend \
//     -I "firmware/Arduino Nano 33/include" -c "firmware/Arduino Nano 33/src/spo2_rd117.cpp" \
//     -o build/spo2/rd117_upstream.o
//   $Z -std=c++17 -O2 -Wall -Wextra -DOXY_SPO2_MAXIM -I "firmware/Arduino Nano 33/include" \
//     tests/test_spo2_maxim.cpp "firmware/Arduino Nano 33/src/spo2_rd117.cpp" \
//     "firmware/Arduino Nano 33/src/spo2_rfalg.cpp" build/spo2/rd117_upstream.o -o build/spo2/test_spo2_maxim.exe
//   ./build/spo2/test_spo2_maxim.exe
//
// Synthetic PPG at 50 Hz: p(n) = 0.5(1 - cos(2 pi f n / 50)); IR and red dip together (systole).
// DC_ir 60000, DC_red 50000, AC_ir 600, AC_red = R * DC_red * AC_ir / DC_ir, so the target R is exact.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "spo2_maxim.h"

// Case 9 only: the rd117 object built with -DRD117_UPSTREAM_IR_INDEX (see the build line above).
void rd117up_heart_rate_and_oxygen_saturation(uint32_t *pun_ir_buffer, int32_t n_ir_buffer_length, uint32_t *pun_red_buffer,
                                              int32_t *pn_spo2, int8_t *pch_spo2_valid, int32_t *pn_heart_rate, int8_t *pch_hr_valid);

static const double kPi = 3.14159265358979323846;
static const double DC_IR = 60000.0, DC_RED = 50000.0, AC_IR = 600.0;

static int g_failures = 0;
static bool g_case_ok = true;
#define CHECK(cond, ...)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);     \
      std::printf(__VA_ARGS__);                              \
      std::printf("\n");                                     \
      g_case_ok = false;                                     \
      ++g_failures;                                          \
    }                                                        \
  } while (0)

static void begin_case(const char* name) { g_case_ok = true; std::printf("%s\n", name); }
static void end_case() { std::printf("  %s\n", g_case_ok ? "ok" : "FAILED"); }

static double curve(double R) { return -45.060 * R * R + 30.354 * R + 94.845; }

struct Gen {
  double fIr = 1.2, fRed = 1.2;  // Hz (1.2 Hz = 72 bpm)
  double R = 0.7;
  int redLag = 0;                // red lags IR by this many raw 50 Hz samples
  bool flat = false;             // no pulse: both channels at DC
  static double p(double n, double f) { return 0.5 * (1.0 - std::cos(2.0 * kPi * f * n / 50.0)); }
  uint32_t ir(long n) const {
    return flat ? (uint32_t)DC_IR : (uint32_t)std::llround(DC_IR - AC_IR * p((double)n, fIr));
  }
  uint32_t red(long n) const {
    const double acRed = R * DC_RED * AC_IR / DC_IR;
    return flat ? (uint32_t)DC_RED : (uint32_t)std::llround(DC_RED - acRed * p((double)(n - redLag), fRed));
  }
};

static Gen pulse(double R) { Gen g; g.R = R; return g; }
static Gen flat() { Gen g; g.flat = true; return g; }

// One computation (push() returned true), as the module left it.
struct Comp {
  long raw;          // raw samples pushed since the last reset, including this one
  int32_t maxim;     // lastRaw
  int8_t maximValid;
  bool shown;
  int16_t value;
  uint8_t streak;
  float rf;
  int8_t rfValid;
  float rfCorr;
};

struct Rig {
  spo2m::State s;
  long n = 0;           // global raw index (keeps the waveform phase across generator switches)
  long sinceReset = 0;
  Rig() { reset(); }
  void reset() { spo2m::reset(s); sinceReset = 0; }
  std::vector<Comp> feed(const Gen& g, long count) {
    std::vector<Comp> out;
    for (long i = 0; i < count; ++i, ++n) {
      ++sinceReset;
      if (spo2m::push(s, g.red(n), g.ir(n))) {
        out.push_back(Comp{sinceReset, s.lastRaw, s.lastValid, spo2m::has_value(s), s.shown, s.invalidStreak,
                           s.rfSpo2, s.rfValid, s.rfCorrel});
      }
    }
    return out;
  }
};

static bool display_ok(const Comp& c) { return c.maximValid && c.maxim >= spo2m::kShowMin && c.maxim <= spo2m::kShowMax; }

// ---- Maxim module cases -------------------------------------------------------------------

static void case_cadence() {
  begin_case("case 1 cadence: first computation on raw sample 200, then every 50");
  static Rig r;
  r.reset();
  std::vector<long> got;
  for (long i = 1; i <= 500; ++i, ++r.n) {
    const Gen g = pulse(0.7);
    const bool ran = spo2m::push(r.s, g.red(r.n), g.ir(r.n));
    if (ran) got.push_back(i);
    if (i < 200) CHECK(!ran && !spo2m::has_value(r.s), "computed or showed a value before 4 s (raw %ld)", i);
  }
  const std::vector<long> want = {200, 250, 300, 350, 400, 450, 500};
  CHECK(got == want, "push() returned true on %zu calls, expected raw 200,250,...,500", got.size());
  CHECK(r.s.computeN == 7, "computeN %u, expected 7", (unsigned)r.s.computeN);
  end_case();
}

static void case_pair_average() {
  begin_case("case 2 pair average: ring holds (a+b)/2 of each raw pair, oldest first");
  static spo2m::State s;
  spo2m::reset(s);
  auto rawIr = [](long n) { return (uint32_t)(20000 + 3 * n); };
  auto rawRed = [](long n) { return (uint32_t)(10000 + 7 * n); };
  long n = 0;
  int computations = 0;
  for (; n < 250; ++n) {
    if (spo2m::push(s, rawRed(n), rawIr(n))) {
      ++computations;
      const long firstPair = (n + 1) / 2 - spo2m::kWindow;  // oldest downsampled sample in the window
      for (int i = 0; i < spo2m::kWindow; ++i) {
        const long a = 2 * (firstPair + i), b = a + 1;
        CHECK(s.linIr[i] == (rawIr(a) + rawIr(b)) / 2, "linIr[%d] %u != (%u+%u)/2 at raw %ld", i,
              (unsigned)s.linIr[i], (unsigned)rawIr(a), (unsigned)rawIr(b), n + 1);
        CHECK(s.linRed[i] == (rawRed(a) + rawRed(b)) / 2, "linRed[%d] %u wrong at raw %ld", i, (unsigned)s.linRed[i], n + 1);
      }
    }
  }
  CHECK(computations == 2, "%d computations in 250 raw samples, expected 2 (raw 200 and 250)", computations);
  end_case();
}

static void case_plumbing() {
  begin_case("case 3 plumbing, fixed algorithm, red lag 0: shown within +-2 of round(curve(R)) at every computation");
  const double Rs[] = {0.5, 0.7, 0.9, 1.1};
  for (double R : Rs) {
    static Rig r;
    r.reset();
    const auto comps = r.feed(pulse(R), 600);
    const long want = std::lround(curve(R));
    CHECK(comps.size() == 9, "R=%.1f: %zu computations in 600 raw samples, expected 9", R, comps.size());
    int lo = 999, hi = -999;
    for (const auto& c : comps) {
      CHECK(c.maximValid == 1, "R=%.1f raw %ld: Maxim invalid (lastRaw %d)", R, c.raw, (int)c.maxim);
      CHECK(c.shown, "R=%.1f raw %ld: no value shown (lastRaw %d)", R, c.raw, (int)c.maxim);
      CHECK(std::labs((long)c.value - want) <= 2, "R=%.1f raw %ld: shown %d, curve %.2f (round %ld)", R, c.raw,
            (int)c.value, curve(R), want);
      if (c.maxim < lo) lo = c.maxim;
      if (c.maxim > hi) hi = c.maxim;
    }
    std::printf("  R=%.1f curve=%.2f maxim=%d..%d\n", R, curve(R), lo, hi);
  }
  end_case();
}

static void case_display_gate() {
  begin_case("case 4 display gate: R=1.3 (curve 58.1) is valid raw but never shown");
  static Rig r;
  r.reset();
  const auto comps = r.feed(pulse(1.3), 400);
  CHECK(!comps.empty(), "no computations");
  for (size_t i = 0; i < comps.size(); ++i) {
    const auto& c = comps[i];
    CHECK(c.maximValid == 1, "raw %ld: algorithm says invalid, expected valid raw", c.raw);
    CHECK(std::labs((long)c.maxim - std::lround(curve(1.3))) <= 2, "raw %ld: lastRaw %d, curve %.2f", c.raw, (int)c.maxim, curve(1.3));
    CHECK(!c.shown, "raw %ld: value %d shown outside 70-100", c.raw, (int)c.value);
    CHECK(c.streak == (uint8_t)(i + 1 < 255 ? i + 1 : 255), "raw %ld: invalid streak %u, expected %zu", c.raw, (unsigned)c.streak, i + 1);
  }
  std::printf("  lastRaw=%d (valid=%d)\n", (int)comps.back().maxim, (int)comps.back().maximValid);
  end_case();
}

static void case_out_of_range() {
  begin_case("case 5 out-of-range R=2.2 (crossed emitters): invalid, dash after 3 computations");
  static Rig r;
  r.reset();
  const auto comps = r.feed(pulse(2.2), 300);
  CHECK(comps.size() == 3, "%zu computations, expected 3", comps.size());
  for (size_t i = 0; i < comps.size(); ++i) {
    CHECK(comps[i].maximValid == 0 && comps[i].maxim == -999, "raw %ld: lastRaw %d valid %d, expected -999/0",
          comps[i].raw, (int)comps[i].maxim, (int)comps[i].maximValid);
    CHECK(!comps[i].shown, "raw %ld: a value is shown", comps[i].raw);
    CHECK(comps[i].streak == i + 1, "raw %ld: streak %u", comps[i].raw, (unsigned)comps[i].streak);
  }
  end_case();
}

static void case_flat() {
  begin_case("case 6 flat signal: invalid; no value before (none earlier) and a dash from the 3rd computation");
  static Rig r;
  r.reset();
  const auto comps = r.feed(flat(), 300);
  CHECK(comps.size() == 3, "%zu computations, expected 3", comps.size());
  for (size_t i = 0; i < comps.size(); ++i) {
    CHECK(comps[i].maximValid == 0, "raw %ld: flat window valid", comps[i].raw);
    CHECK(!comps[i].shown, "raw %ld: a value is shown with no earlier valid result", comps[i].raw);
    CHECK(comps[i].streak == i + 1, "raw %ld: streak %u", comps[i].raw, (unsigned)comps[i].streak);
  }
  end_case();
}

static void case_hold() {
  begin_case("case 7 hold rule: held through 2 display-invalid results, dashed on the 3rd, shown again when valid");
  static Rig r;
  r.reset();
  auto comps = r.feed(pulse(0.7), 400);
  CHECK(!comps.empty() && comps.back().shown, "valid stream never showed a value");
  int16_t held = comps.empty() ? -1 : comps.back().value;
  // Flat from here; the window is mixed for a while, so follow the module's own notion of invalid.
  comps = r.feed(flat(), 600);
  int invalidRun = 0;
  bool sawDash = false;
  for (const auto& c : comps) {
    if (display_ok(c)) {
      CHECK(invalidRun == 0, "raw %ld: valid result inside the invalid run", c.raw);
      held = c.value;
      continue;
    }
    ++invalidRun;
    if (invalidRun <= 2) {
      CHECK(c.shown && c.value == held, "raw %ld: invalid #%d did not hold %d (shown %d value %d)", c.raw, invalidRun,
            (int)held, (int)c.shown, (int)c.value);
    } else {
      CHECK(!c.shown, "raw %ld: invalid #%d still shows %d", c.raw, invalidRun, (int)c.value);
      sawDash = true;
    }
  }
  CHECK(sawDash, "never reached the 3rd consecutive invalid result");
  std::printf("  held value %d through 2 invalid results; dash from the 3rd\n", (int)held);
  // Valid again: shown again within the next 4 s of pulse.
  comps = r.feed(pulse(0.7), 400);
  bool back = false;
  for (const auto& c : comps) {
    if (c.shown) {
      back = true;
      CHECK(std::labs((long)c.value - std::lround(curve(0.7))) <= 2, "raw %ld: shown %d after recovery", c.raw, (int)c.value);
    }
  }
  CHECK(back, "no value shown again after 400 raw samples of valid pulse");
  CHECK(comps.empty() || comps.back().shown, "value not shown at the end of the recovered stream");
  end_case();
}

static void case_reset() {
  begin_case("case 8 reset: clears ring, value and streak; next result needs 200 raw samples");
  static Rig r;
  r.reset();
  r.feed(pulse(0.7), 300);
  CHECK(spo2m::has_value(r.s), "no value before the reset");
  r.reset();
  CHECK(!spo2m::has_value(r.s) && r.s.shown == -1, "value survives reset");
  CHECK(r.s.count == 0 && r.s.head == 0 && r.s.sinceCompute == 0 && r.s.pairN == 0, "ring not cleared");
  CHECK(r.s.invalidStreak == 0 && r.s.computeN == 0 && r.s.lastRaw == -999 && r.s.rfValid == 0, "counters not cleared");
  const auto comps = r.feed(pulse(0.7), 250);
  CHECK(comps.size() == 2 && comps[0].raw == 200 && comps[1].raw == 250, "after reset: %zu computations, first at raw %ld",
        comps.size(), comps.empty() ? -1L : comps[0].raw);
  end_case();
}

static void case_bugfix() {
  begin_case("case 9 bug-fix comparison: R=0.7, red lags IR by 2 raw samples; upstream vs fixed IR index");
  static Rig r;
  r.reset();
  Gen g = pulse(0.7);
  g.redLag = 2;
  const auto comps = r.feed(g, 200);
  CHECK(comps.size() == 1, "expected one computation");
  int32_t up = -999, upHr = -999, fx = -999, fxHr = -999;
  int8_t upV = 0, upHrV = 0, fxV = 0, fxHrV = 0;
  rd117up_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &up, &upV, &upHr, &upHrV);
  rd117_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &fx, &fxV, &fxHr, &fxHrV);
  std::printf("  upstream=%d (valid %d)  fixed=%d (valid %d)  curve(0.7)=%.2f\n", (int)up, (int)upV, (int)fx, (int)fxV, curve(0.7));
  CHECK(comps.empty() || comps[0].maxim == fx, "module result %d differs from a direct fixed call %d", comps.empty() ? 0 : (int)comps[0].maxim, (int)fx);
  // Report only: the next 10 windows (other beat phases), feeding one 1 s stride at a time.
  int upLo = 999, upHi = -999, fxLo = 999, fxHi = -999;
  for (int w = 0; w < 10; ++w) {
    r.feed(g, 50);
    int32_t u = -999, f = -999, h = -999; int8_t uv = 0, fv = 0, hv = 0;
    rd117up_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &u, &uv, &h, &hv);
    rd117_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &f, &fv, &h, &hv);
    if (u < upLo) upLo = u;
    if (u > upHi) upHi = u;
    if (f < fxLo) fxLo = f;
    if (f > fxHi) fxHi = f;
  }
  std::printf("  next 10 windows: upstream=%d..%d  fixed=%d..%d\n", upLo, upHi, fxLo, fxHi);
  CHECK(fxV == 1 && std::fabs((double)fx - curve(0.7)) <= 2.0, "fixed %d (valid %d) not within +-2 of %.2f", (int)fx, (int)fxV, curve(0.7));
  end_case();
}

// ---- Comparison algorithm (rfalg_*) cases --------------------------------------------------

static void case_rf_plumbing() {
  begin_case("case 10 rfalg plumbing: SpO2 within +-1 of curve(R), correl >= 0.8, at every computation");
  const double Rs[] = {0.5, 0.7, 0.9, 1.1};
  for (double R : Rs) {
    static Rig r;
    r.reset();
    const auto comps = r.feed(pulse(R), 600);
    float lo = 999.0f, hi = -999.0f, cmin = 9.0f;
    for (const auto& c : comps) {
      CHECK(c.rfValid == 1, "R=%.1f raw %ld: rfalg invalid (correl %.3f)", R, c.raw, (double)c.rfCorr);
      CHECK(std::fabs((double)c.rf - curve(R)) <= 1.0, "R=%.1f raw %ld: rf %.2f, curve %.2f", R, c.raw, (double)c.rf, curve(R));
      CHECK(c.rfCorr >= 0.8f, "R=%.1f raw %ld: correl %.3f < 0.8", R, c.raw, (double)c.rfCorr);
      if (c.rf < lo) lo = c.rf;
      if (c.rf > hi) hi = c.rf;
      if (c.rfCorr < cmin) cmin = c.rfCorr;
    }
    std::printf("  R=%.1f curve=%.2f rf=%.2f..%.2f min correl=%.4f\n", R, curve(R), (double)lo, (double)hi, (double)cmin);
  }
  end_case();
}

static void case_rf_rejects() {
  begin_case("case 11 rfalg rejects bad windows: flat, R=2.2, uncorrelated red/IR");
  {
    static Rig r;
    r.reset();
    for (const auto& c : r.feed(flat(), 300)) CHECK(c.rfValid == 0, "flat raw %ld: rfalg valid (%.2f)", c.raw, (double)c.rf);
  }
  {
    static Rig r;
    r.reset();
    for (const auto& c : r.feed(pulse(2.2), 300))
      CHECK(c.rfValid == 0, "R=2.2 raw %ld: rfalg valid (%.2f, correl %.3f)", c.raw, (double)c.rf, (double)c.rfCorr);
  }
  {
    static Rig r;
    r.reset();
    Gen g = pulse(0.7);
    g.fRed = 1.9;  // red is an independent pulse at 114 bpm against IR at 72 bpm
    const auto comps = r.feed(g, 600);
    std::printf("  uncorrelated (IR 1.2 Hz, red 1.9 Hz), per computation:\n");
    for (const auto& c : comps) {
      CHECK(c.rfCorr < 0.8f, "uncorrelated raw %ld: correl %.3f passes the 0.8 gate", c.raw, (double)c.rfCorr);
      CHECK(c.rfValid == 0, "uncorrelated raw %ld: rfalg valid (%.2f)", c.raw, (double)c.rf);
      std::printf("    raw %ld: rf_corr=%.3f rf_valid=%d | maxim=%d maxim_valid=%d shown=%s\n", c.raw, (double)c.rfCorr,
                  (int)c.rfValid, (int)c.maxim, (int)c.maximValid, c.shown ? std::to_string(c.value).c_str() : "-");
    }
  }
  end_case();
}

static void case_inputs_untouched() {
  begin_case("case 12 inputs untouched: both algorithms leave linIr/linRed byte-identical");
  static Rig r;
  r.reset();
  r.feed(pulse(0.9), 200);
  uint32_t ir0[spo2m::kWindow], red0[spo2m::kWindow];
  std::memcpy(ir0, r.s.linIr, sizeof(ir0));
  std::memcpy(red0, r.s.linRed, sizeof(red0));
  int32_t spo2 = 0, hr = 0; int8_t v = 0, hv = 0;
  rd117_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &spo2, &v, &hr, &hv);
  CHECK(std::memcmp(ir0, r.s.linIr, sizeof(ir0)) == 0 && std::memcmp(red0, r.s.linRed, sizeof(red0)) == 0, "rd117 modified its inputs");
  float f = 0, ratio = 0, corr = 0;
  rfalg_heart_rate_and_oxygen_saturation(r.s.linIr, spo2m::kWindow, r.s.linRed, &f, &v, &hr, &hv, &ratio, &corr);
  CHECK(std::memcmp(ir0, r.s.linIr, sizeof(ir0)) == 0 && std::memcmp(red0, r.s.linRed, sizeof(red0)) == 0, "rfalg modified its inputs");
  end_case();
}

struct RfOut { float spo2, ratio, correl; int8_t valid, hrValid; int32_t hr; };
static RfOut rf_call(uint32_t* ir, uint32_t* red) {
  RfOut o{-1.0f, -1.0f, -1.0f, -1, -1, -1};
  rfalg_heart_rate_and_oxygen_saturation(ir, spo2m::kWindow, red, &o.spo2, &o.valid, &o.hr, &o.hrValid, &o.ratio, &o.correl);
  return o;
}
static bool same(const RfOut& a, const RfOut& b) {
  return std::memcmp(&a.spo2, &b.spo2, sizeof(float)) == 0 && std::memcmp(&a.ratio, &b.ratio, sizeof(float)) == 0 &&
         std::memcmp(&a.correl, &b.correl, sizeof(float)) == 0 && a.valid == b.valid && a.hrValid == b.hrValid && a.hr == b.hr;
}

static void case_rf_reset() {
  begin_case("case 13 rfalg_reset: next call behaves like the first call of a fresh run");
  static Rig a, b;
  a.reset();
  a.feed(pulse(0.7), 200);  // window A: 72 bpm
  Gen gb = pulse(0.9);
  gb.fIr = gb.fRed = 2.0;   // window B: 120 bpm, so the carried peak interval differs
  b.reset();
  b.feed(gb, 200);
  rfalg_reset();
  const RfOut first = rf_call(a.s.linIr, a.s.linRed);
  rf_call(b.s.linIr, b.s.linRed);
  rf_call(b.s.linIr, b.s.linRed);
  const RfOut carried = rf_call(a.s.linIr, a.s.linRed);  // no reset: starts from B's interval
  rf_call(b.s.linIr, b.s.linRed);
  rfalg_reset();
  const RfOut again = rf_call(a.s.linIr, a.s.linRed);
  CHECK(same(first, again), "after reset: spo2 %.4f/%.4f hr %d/%d ratio %.4f/%.4f", (double)first.spo2, (double)again.spo2,
        (int)first.hr, (int)again.hr, (double)first.ratio, (double)again.ratio);
  std::printf("  fresh: hr=%d spo2=%.2f ratio=%.4f | after B without reset: hr=%d spo2=%.2f ratio=%.4f\n", (int)first.hr,
              (double)first.spo2, (double)first.ratio, (int)carried.hr, (double)carried.spo2, (double)carried.ratio);
  end_case();
}

int main() {
  case_cadence();
  case_pair_average();
  case_plumbing();
  case_display_gate();
  case_out_of_range();
  case_flat();
  case_hold();
  case_reset();
  case_bugfix();
  case_rf_plumbing();
  case_rf_rejects();
  case_inputs_untouched();
  case_rf_reset();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "ALL OK", g_failures);
  return g_failures ? 1 : 0;
}

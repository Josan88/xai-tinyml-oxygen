#pragma once
// SpO2 est for the -DOXY_SPO2_MAXIM live image (ADR-0010). Display only.
// Maxim MAXREFDES117 reference algorithm (vendored as rd117_*, one IR-index fix):
// R = (AC_red/DC_red)/(AC_IR/DC_IR) per beat, median of up to 5 beats in 4 s, then
// SpO2 ~= -45.060 R^2 + 30.354 R + 94.845 by table. Uncalibrated for this module and
// enclosure. Not a TinyKAN input. Not used by any decision, alert or explanation.
#include <stdint.h>
#include "spo2_rd117.h"   // RD117_FS 25, RD117_BUFFER_SIZE 100
#include "spo2_rfalg.h"   // comparison only (D8): result goes to the SPO2CMP: serial line, never BLE

namespace spo2m {

static const int     kRawPerSample = 2;                  // 50 Hz -> 25 Hz, pair average
static const int     kWindow       = RD117_BUFFER_SIZE;  // 100 samples = 4 s at 25 Hz
static const int     kStride       = RD117_FS;           // recompute every 1 s
static const int     kShowMin      = 70;                 // ISO 80601-2-61 accuracy range is 70-100 %
static const int     kShowMax      = 100;
static const uint8_t kDashAfter    = 3;                  // consecutive invalid results before a dash

struct State {
  uint32_t ringIr[kWindow], ringRed[kWindow];   // 25 Hz ring, oldest at head once full
  uint32_t linIr[kWindow],  linRed[kWindow];    // oldest-first copy handed to rd117
  uint16_t head, count;
  uint16_t sinceCompute;
  uint32_t pairIr, pairRed;
  uint8_t  pairN;
  int16_t  shown;          // last displayed value (%), or -1 for a dash
  uint8_t  invalidStreak;
  int32_t  lastRaw;        // last algorithm output, -999 when invalid (tests, logs)
  int8_t   lastValid;
  float    rfSpo2;         // comparison algorithm (D8), raw, -999 when invalid
  int8_t   rfValid;
  float    rfCorrel;       // its red/IR Pearson correlation
  uint32_t computeN;       // computations since the last reset (SPO2CMP: n=)
};

inline void reset(State& s) {
  s.head = s.count = s.sinceCompute = 0;
  s.pairIr = s.pairRed = 0; s.pairN = 0;
  s.shown = -1; s.invalidStreak = 0; s.lastRaw = -999; s.lastValid = 0;
  s.rfSpo2 = -999.0f; s.rfValid = 0; s.rfCorrel = 0.0f; s.computeN = 0;
  rfalg_reset();
}

// One 50 Hz record. Returns true when the algorithm ran on this call.
inline bool push(State& s, uint32_t red, uint32_t ir) {
  s.pairIr += ir; s.pairRed += red;
  if (++s.pairN < kRawPerSample) return false;
  const uint32_t dIr = s.pairIr / kRawPerSample, dRed = s.pairRed / kRawPerSample;
  s.pairIr = s.pairRed = 0; s.pairN = 0;
  s.ringIr[s.head] = dIr; s.ringRed[s.head] = dRed;
  s.head = (uint16_t)((s.head + 1) % kWindow);
  if (s.count < kWindow) s.count++;
  if (s.sinceCompute < 0xFFFF) s.sinceCompute++;
  if (s.count < kWindow || s.sinceCompute < kStride) return false;
  s.sinceCompute = 0;
  for (int i = 0; i < kWindow; ++i) {
    const int j = (s.head + i) % kWindow;
    s.linIr[i] = s.ringIr[j]; s.linRed[i] = s.ringRed[j];
  }
  int32_t spo2 = -999, hr = -999; int8_t spo2Valid = 0, hrValid = 0;
  rd117_heart_rate_and_oxygen_saturation(s.linIr, kWindow, s.linRed, &spo2, &spo2Valid, &hr, &hrValid);
  s.lastRaw = spo2; s.lastValid = spo2Valid;
  // Comparison only (D8): same buffers, same moment; never shown on BLE or the tile.
  int32_t rfHr = -999; int8_t rfHrValid = 0; float rfRatio = 0.0f;
  rfalg_heart_rate_and_oxygen_saturation(s.linIr, kWindow, s.linRed, &s.rfSpo2, &s.rfValid,
                                         &rfHr, &rfHrValid, &rfRatio, &s.rfCorrel);
  s.computeN++;
  if (spo2Valid && spo2 >= kShowMin && spo2 <= kShowMax) {
    s.shown = (int16_t)spo2; s.invalidStreak = 0;
  } else {
    if (s.invalidStreak < 0xFF) s.invalidStreak++;
    if (s.invalidStreak >= kDashAfter) s.shown = -1;
  }
  return true;
}

inline bool  has_value(const State& s) { return s.shown >= 0; }
inline float value(const State& s)     { return (float)s.shown; }

}  // namespace spo2m

// Host test of the P5 live image's BLE packet (include/ble_packet_p5.h), job F2.
// Build: zig c++ -std=c++17 -O2 -DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0 -I "firmware/Arduino Nano 33/include" tests/test_ble_packet_p5.cpp
// Checks, over every ordered top-3 of the 5 P5 input names, the widest values of every other field
// and field 10 from "-" to the widest s_hat inside its header bounds (KANSAT_S_HAT_BOUND_LO/HI):
//   1. the packet always builds (no -1) and fits the 220-byte buffer/characteristic;
//   2. it always has exactly 12 fields, field 10 = the text sat_est_text() gives, field 11 "label0";
//   3. fields 0-9 are the shipped packet's (no compaction is ever needed with the P5 names);
//   4. sat_est_text(): "-" with no finger, before the first window, or beat-invalid; else %.1f.
// Prints the worst-case packet length.
#include <cstdio>
#include <cstring>
#include <string>
#include "kan_inference.h"
#include "kan_sat_regressor_p5.h"
#include "ble_packet_p5.h"

static const int HR_MIN_BPM = 40, HR_MAX_BPM = 180;

// Fields 0-9 as main.cpp updateBLE() formats them without the flags (dataChar write omitted).
static std::string shipped_packet(bool noFinger, float spo2Estimate, unsigned long redVal, unsigned long irVal,
                                  bool saturated, bool isConfirmed, bool isCandidate, float risk_prob,
                                  const char* const names[3], const float deltas[3], int bpm) {
  char packet[220], explBuf[100], spo2Buf[8];
  const char* status;
  const char* signal;
  int n;
  if (noFinger) {
    snprintf(spo2Buf, sizeof(spo2Buf), "---");
    status = "No finger detected";
    snprintf(explBuf, sizeof(explBuf), "Place finger on sensor");
    signal = "NO FINGER";
    bpm = -1;
    n = snprintf(packet, sizeof(packet), "%s|%s|%s|%lu|%lu|%s|none:0.0000|none:0.0000|none:0.0000",
                 spo2Buf, status, explBuf, redVal, irVal, signal);
  } else {
    snprintf(spo2Buf, sizeof(spo2Buf), "%.1f", spo2Estimate);
    if (isConfirmed) {
      status = "CONFIRMED LOW OXYGEN RISK";
      snprintf(explBuf, sizeof(explBuf), "CONFIRMED RISK (Prob: %.2f) top driver: %s", risk_prob, names[0]);
    } else if (isCandidate) {
      status = "CANDIDATE RISK (Unconfirmed)";
      snprintf(explBuf, sizeof(explBuf), "Candidate risk detected (Prob: %.2f), awaiting confirmation", risk_prob);
    } else {
      status = "Normal";
      snprintf(explBuf, sizeof(explBuf), "Normal (Prob: %.2f)", risk_prob);
    }
    signal = saturated ? "CLIPPED" : "OK";
    n = snprintf(packet, sizeof(packet), "%s|%s|%s|%lu|%lu|%s|%s:%.4f|%s:%.4f|%s:%.4f",
                 spo2Buf, status, explBuf, redVal, irVal, signal,
                 names[0], deltas[0], names[1], deltas[1], names[2], deltas[2]);
  }
  char bpmBuf[5] = "-";
  if (bpm >= HR_MIN_BPM && bpm <= HR_MAX_BPM) snprintf(bpmBuf, sizeof(bpmBuf), "%d", bpm);
  return std::string(packet, (size_t)n) + "|" + bpmBuf;
}

int main() {
  long failures = 0, cases = 0;
  // 4. field 10 text rules.
  char t[BLE_SAT_EST_TEXT_BYTES];
  sat_est_text(t, sizeof(t), true, true, true, 95.0f);   failures += std::strcmp(t, "-") != 0;
  sat_est_text(t, sizeof(t), false, false, true, 95.0f); failures += std::strcmp(t, "-") != 0;
  sat_est_text(t, sizeof(t), false, true, false, 95.0f); failures += std::strcmp(t, "-") != 0;
  sat_est_text(t, sizeof(t), false, true, true, 94.66f); failures += std::strcmp(t, "94.7") != 0;
  const float sats[] = {KANSAT_S_HAT_BOUND_LO, KANSAT_S_HAT_BOUND_HI, 99.95f, 100.0f, 62.0f};
  const float spo2s[] = {100.0f, 97.3f, 0.0f};
  const unsigned long levels[] = {262143ul, 150ul, 99999ul};
  const float delta_sets[][3] = {{-0.9999f, -0.9999f, -0.9999f}, {0.4120f, -0.0310f, 0.0200f}, {0.0f, 0.0f, 0.0f}};
  const int bpms[] = {176, 180, 72, -1, 181};
  const float probs[] = {1.0f, 0.89f, 0.0f};
  size_t worst = 0;
  std::string worst_packet;
  for (int a = 0; a < KAN_INPUT_DIM; ++a)
  for (int b = 0; b < KAN_INPUT_DIM; ++b)
  for (int c = 0; c < KAN_INPUT_DIM; ++c) {
    if (a == b || a == c || b == c) continue;
    const char* const names[3] = {KAN_FEATURE_NAMES[a], KAN_FEATURE_NAMES[b], KAN_FEATURE_NAMES[c]};
    for (int state = 0; state < 4; ++state)               // 0 confirmed, 1 candidate, 2 normal, 3 no finger
    for (int window = 0; window < 3; ++window)            // 0 none yet, 1 beat-invalid, 2 beat-valid
    for (const float sat : sats)
    for (const float spo2 : spo2s)
    for (const unsigned long level : levels)
    for (int clip = 0; clip < 2; ++clip)
    for (const auto& deltas : delta_sets)
    for (const int bpm : bpms)
    for (const float prob : probs) {
      const bool noFinger = state == 3;
      const bool confirmed = state == 0, candidate = state <= 1;
      char satText[BLE_SAT_EST_TEXT_BYTES];
      sat_est_text(satText, sizeof(satText), noFinger, window > 0, window == 2, sat);
      char packet[BLE_PACKET_P5_BYTES];
      const int n = ble_packet_p5(packet, sizeof(packet), noFinger, spo2, level, level, clip != 0, confirmed, candidate,
                                  prob, names, deltas, bpm, HR_MIN_BPM, HR_MAX_BPM, satText, KAN_XAI_REF_TAG);
      ++cases;
      if (n < 0 || (size_t)n >= sizeof(packet) || (size_t)n != std::strlen(packet)) { ++failures; continue; }
      const std::string p(packet);
      if (p.size() > worst) { worst = p.size(); worst_packet = p; }
      int fields = 1;
      for (char ch : p) fields += ch == '|';
      const std::string expected = shipped_packet(noFinger, spo2, level, level, clip != 0, confirmed, candidate, prob,
                                                  names, deltas, bpm) + "|" + satText + "|label0";
      if (fields != 12 || p != expected) ++failures;
      const bool showSat = !noFinger && window == 2;
      if (showSat == (std::strcmp(satText, "-") == 0)) ++failures;
    }
  }
  std::printf("cases %ld failures %ld\n", cases, failures);
  std::printf("worst P5 packet %zu chars; buffer %zu bytes (%zu chars)\n", worst, BLE_PACKET_P5_BYTES, BLE_PACKET_P5_BYTES - 1);
  std::printf("worst P5 packet: %s\n", worst_packet.c_str());
  return failures ? 1 : 0;
}

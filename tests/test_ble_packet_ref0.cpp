// Host test of the -DOXY_XAI_REF_LABEL0 BLE packet (include/ble_packet_ref0.h).
// Build: zig c++ -std=c++17 -O2 -DOXY_MODEL_V2 -I "firmware/Arduino Nano 33/include" tests/test_ble_packet_ref0.cpp
// Checks, over every ordered top-3 of the 16 v2 feature names and the longest values of every
// other field:
//   1. the packet always builds (no -1) and fits the 220-byte buffer/characteristic;
//   2. it always has exactly 12 fields, field 10 "-" and field 11 "label0";
//   3. whenever the shipped packet (main.cpp updateBLE, copied below) plus "|-|label0" fits,
//      the ref0 packet is exactly that.
// Prints the worst-case length of the shipped packet and of the ref0 packet.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include "kan_inference.h"
#include "ble_packet_ref0.h"

static const int HR_MIN_BPM = 40, HR_MAX_BPM = 180;
static const unsigned long FINGER = 3000;

// Verbatim formatting of main.cpp updateBLE() without the flag (dataChar write omitted).
static std::string shipped_packet(bool fingerPresent, float spo2Estimate, unsigned long redVal, unsigned long irVal,
                                  bool saturated, bool isConfirmed, bool isCandidate, float risk_prob,
                                  const char* const names[3], const float deltas[3], int bpm) {
  char packet[220];
  const char* status;
  const char* signal;
  char explBuf[100];
  char spo2Buf[8];
  int n;
  if (!fingerPresent || irVal < FINGER) {
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
  if (n > 0 && (size_t)n < sizeof(packet)) {
    char bpmBuf[5] = "-";
    if (bpm >= HR_MIN_BPM && bpm <= HR_MAX_BPM) snprintf(bpmBuf, sizeof(bpmBuf), "%d", bpm);
    const size_t room = sizeof(packet) - (size_t)n;
    if (strlen(bpmBuf) + 2 > room) { bpmBuf[0] = '-'; bpmBuf[1] = '\0'; }
    if (room >= 3) snprintf(packet + n, room, "|%s", bpmBuf);
  }
  return packet;
}

int main() {
  const float spo2s[] = {100.0f, 97.3f, 0.0f};
  const unsigned long levels[] = {262143ul, 150ul, 99999ul};
  const float delta_sets[][3] = {{-0.9999f, -0.9999f, -0.9999f}, {0.4120f, -0.0310f, 0.0200f}, {0.0f, 0.0f, 0.0f}};
  const int bpms[] = {176, 180, 72, -1, 181};
  const float probs[] = {1.0f, 0.89f, 0.0f};
  long cases = 0, identical = 0, compacted = 0, failures = 0;
  size_t worst_shipped = 0, worst_ref0 = 0;
  std::string worst_ref0_packet;
  for (int a = 0; a < KAN_INPUT_DIM; ++a)
  for (int b = 0; b < KAN_INPUT_DIM; ++b)
  for (int c = 0; c < KAN_INPUT_DIM; ++c) {
    if (a == b || a == c || b == c) continue;
    const char* const names[3] = {KAN_FEATURE_NAMES[a], KAN_FEATURE_NAMES[b], KAN_FEATURE_NAMES[c]};
    for (int state = 0; state < 4; ++state)           // 0 confirmed, 1 candidate, 2 normal, 3 no finger
    for (const float spo2 : spo2s)
    for (const unsigned long level : levels)
    for (int sat = 0; sat < 2; ++sat)
    for (const auto& deltas : delta_sets)
    for (const int bpm : bpms)
    for (const float prob : probs) {
      const bool finger = state != 3;
      const unsigned long ir = finger ? (level < FINGER ? 262143ul : level) : level;
      const bool confirmed = state == 0, candidate = state <= 1;
      const std::string shipped = shipped_packet(finger, spo2, level, ir, sat, confirmed, candidate, prob, names, deltas, bpm);
      char packet[BLE_PACKET_BYTES];
      const int n = ble_packet_ref0(packet, sizeof(packet), !finger || ir < FINGER, spo2, level, ir, sat != 0,
                                    confirmed, candidate, prob, names, deltas, bpm, HR_MIN_BPM, HR_MAX_BPM, KAN_XAI_REF_TAG);
      ++cases;
      if (shipped.size() > worst_shipped) worst_shipped = shipped.size();
      if (n < 0 || (size_t)n >= sizeof(packet) || (size_t)n != strlen(packet)) { ++failures; continue; }
      const std::string ref0(packet);
      if (ref0.size() > worst_ref0) { worst_ref0 = ref0.size(); worst_ref0_packet = ref0; }
      int fields = 1;
      for (char ch : ref0) fields += ch == '|';
      const size_t tail = ref0.rfind("|-|label0");
      if (fields != 12 || tail == std::string::npos || tail + 9 != ref0.size()) { ++failures; continue; }
      const std::string expected = shipped + "|-|label0";
      const bool shipped_fits = shipped.size() + 9 < BLE_PACKET_BYTES && fields == 12
                                && std::count(shipped.begin(), shipped.end(), '|') == 9;
      if (shipped_fits) {
        if (ref0 == expected) ++identical; else ++failures;
      } else {
        ++compacted;
        if (ref0.find("Prob: ") == std::string::npos && finger) ++failures;
      }
    }
  }
  std::printf("cases %ld identical_to_shipped_plus_tail %ld compacted %ld failures %ld\n", cases, identical, compacted, failures);
  std::printf("worst shipped packet %zu chars; worst ref0 packet %zu chars; buffer %zu bytes (%zu chars)\n",
              worst_shipped, worst_ref0, BLE_PACKET_BYTES, BLE_PACKET_BYTES - 1);
  std::printf("worst ref0 packet: %s\n", worst_ref0_packet.c_str());
  return failures ? 1 : 0;
}

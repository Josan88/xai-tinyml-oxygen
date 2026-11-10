#pragma once
// BLE packet of the P5 live image (job F2, ADR-0008; -DOXY_MODEL_P5 -DOXY_XAI_REF_LABEL0). Other
// images do not include this file.
//
//   spo2|status|explanation|red|ir|signal|xai1|xai2|xai3|bpm|sat_est|xai_ref
//
// Fields 0-9 are formatted exactly as main.cpp formats them (field 0 is still the toy Red/IR
// estimate, field 9 the display HR). Field 10 is the model saturation estimate (s_hat, estimated
// ScalcO2) with one decimal, or "-": the caller passes it as text (sat_est_text()). Field 11 is the
// explainer reference tag (KAN_XAI_REF_TAG, "label0").
//
// Room. The packet buffer and the characteristic are 220 bytes (219 characters). The P5 input
// names are short (longest "sqi_ibi_ratio"), so the worst case with every field at its widest is
// well inside 219 (tests/test_ble_packet_p5.cpp asserts it). The fallback of ble_packet_ref0.h is
// kept so a field is never truncated: field 2 drops its trailing clause before anything is cut.
#include <stdio.h>
#include <string.h>

static const size_t BLE_PACKET_P5_BYTES = 220;  // char packet[220] and the characteristic size
static const size_t BLE_SAT_EST_TEXT_BYTES = 8;  // "-" or %.1f of s_hat within its header bounds

// Field 10: "-" with no finger, before the first completed window since a reset, or when that
// window was beat-invalid; otherwise s_hat with one decimal.
inline void sat_est_text(char* out, size_t size, bool noFinger, bool haveWindow, bool beatValid, float s_hat) {
  if (noFinger || !haveWindow || !beatValid) snprintf(out, size, "-");
  else snprintf(out, size, "%.1f", s_hat);
}

// Returns the packet length, or -1 if it cannot be built with every field whole.
inline int ble_packet_p5(char* packet, size_t size, bool noFinger, float spo2Estimate,
                         unsigned long redVal, unsigned long irVal, bool saturated,
                         bool isConfirmed, bool isCandidate, float risk_prob,
                         const char* const names[3], const float deltas[3],
                         int bpm, int bpmMin, int bpmMax, const char* satEst, const char* refTag) {
  char spo2Buf[8];
  char explBuf[100];
  char bpmBuf[5] = "-";
  const char* status;
  const char* signal;
  if (noFinger) bpm = -1;
  if (bpm >= bpmMin && bpm <= bpmMax) snprintf(bpmBuf, sizeof(bpmBuf), "%d", bpm);

  // Pass 0: shipped field 2. Pass 1: field 2 without its trailing clause. Pass 2: also bpm "-".
  for (int pass = 0; pass < 3; ++pass) {
    int n;
    if (noFinger) {
      snprintf(spo2Buf, sizeof(spo2Buf), "---");
      status = "No finger detected";
      snprintf(explBuf, sizeof(explBuf), "Place finger on sensor");
      signal = "NO FINGER";
      n = snprintf(packet, size, "%s|%s|%s|%lu|%lu|%s|none:0.0000|none:0.0000|none:0.0000",
                   spo2Buf, status, explBuf, redVal, irVal, signal);
    } else {
      snprintf(spo2Buf, sizeof(spo2Buf), "%.1f", spo2Estimate);
      if (isConfirmed) {
        status = "CONFIRMED LOW OXYGEN RISK";
        if (pass == 0) snprintf(explBuf, sizeof(explBuf), "CONFIRMED RISK (Prob: %.2f) top driver: %s", risk_prob, names[0]);
        else snprintf(explBuf, sizeof(explBuf), "CONFIRMED RISK (Prob: %.2f)", risk_prob);
      } else if (isCandidate) {
        status = "CANDIDATE RISK (Unconfirmed)";
        if (pass == 0) snprintf(explBuf, sizeof(explBuf), "Candidate risk detected (Prob: %.2f), awaiting confirmation", risk_prob);
        else snprintf(explBuf, sizeof(explBuf), "Candidate risk detected (Prob: %.2f)", risk_prob);
      } else {
        status = "Normal";
        snprintf(explBuf, sizeof(explBuf), "Normal (Prob: %.2f)", risk_prob);
      }
      signal = saturated ? "CLIPPED" : "OK";
      n = snprintf(packet, size, "%s|%s|%s|%lu|%lu|%s|%s:%.4f|%s:%.4f|%s:%.4f",
                   spo2Buf, status, explBuf, redVal, irVal, signal,
                   names[0], deltas[0], names[1], deltas[1], names[2], deltas[2]);
    }
    if (pass == 2) { bpmBuf[0] = '-'; bpmBuf[1] = '\0'; }
    if (n < 0 || (size_t)n >= size) continue;
    const size_t room = size - (size_t)n;  // includes the terminator
    const int tail = snprintf(packet + n, room, "|%s|%s|%s", bpmBuf, satEst, refTag);
    if (tail >= 0 && (size_t)tail < room) return n + tail;
  }
  packet[0] = '\0';
  return -1;
}

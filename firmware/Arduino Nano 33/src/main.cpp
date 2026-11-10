/*
  Semester 1 Milestone Demo: MAX30105 with demo-level SpO2 estimation and risk classification.
  Engineering prototype only - NOT a certified pulse oximeter.

  Sensor -> Arduino Nano 33 BLE Sense -> Serial Monitor + BLE Web Dashboard
  Demonstrates live sensor acquisition with low-oxygen-risk flagging.

  Hardware Connections (Breakout board to Arduino Nano 33 BLE Sense):
  - 3.3V = 3.3V
  - GND  = GND
  - SDA  = A4 (or SDA)
  - SCL  = A5 (or SCL)
  - INT  = Not connected

  Set DEMO_LOW_OXYGEN_MODE to true to force low-oxygen-risk output for presentation demos
  when a real finger reading stays in normal range.
*/

#include <Wire.h>
#include "MAX30105.h"
#include <ArduinoBLE.h>
#include "oxygen_risk_pipeline.h"
#ifdef OXY_SPO2_MAXIM
#if defined(OXY_MODEL_P5) || defined(OXY_XAI_REF_LABEL0)
#error "OXY_SPO2_MAXIM is for the v2 f1a image only (ADR-0010); the P5 and ref0 packets keep the demo formula"
#endif
#include "spo2_maxim.h"
static spo2m::State spo2State;   // file scope: ~1.6 KB off the thread stack
#endif

// Set to true for live demo: forces SpO2_est below 90% so the examiner
// can see "LOW OXYGEN RISK" and the explanation line on Serial Monitor.
static const bool DEMO_LOW_OXYGEN_MODE = false;

// Signal-quality threshold: if Red or IR is above this value, the reading is saturated.
static const uint32_t ADC_SATURATION_THRESHOLD = 260000;

// Finger present if BOTH channels exceed this. Ambient on this bench is ~150–200 ADC;
// 50000 was unreachable at ledBrightness 0x0F. 3000 is ~15× ambient and still far below clip.
#ifdef PEAK_RAM_PROBE
// Peak-RAM probe images only (include/peak_ram_probe.h): forced decisions. Every gate
// below compares IR against this constant, so 0 holds them open on the no-finger dark
// floor; isFingerPresent() counts the live gate's verdict at the live value.
static const uint32_t PR_LIVE_FINGER_THRESHOLD = 3000;
#ifdef PEAK_RAM_LIVE_GATE
// Sampler images: no forced decisions; the gate is the live one.
static const uint32_t FINGER_PRESENT_THRESHOLD = PR_LIVE_FINGER_THRESHOLD;
#else
static const uint32_t FINGER_PRESENT_THRESHOLD = 0;
#endif
#else
static const uint32_t FINGER_PRESENT_THRESHOLD = 3000;
#endif

MAX30105 particleSensor;

long startTime;
long samplesTaken = 0;

byte interruptPin = 3;

// Stateful OxygenRiskPipeline Instance & Cached State
OxygenRiskPipeline pipeline;
float currentRiskProb = 0.0f;
bool isCandidateRisk = false;
bool isConfirmedRisk = false;
uint64_t lastTotalSamples = 0;
uint32_t lastWindowIndex = 0;
FeatureAttribution currentTopAttributions[3] = {
  {0, 0.0f, "none"},
  {0, 0.0f, "none"},
  {0, 0.0f, "none"}
};
// Diagnostic only: the feature vector for the most recent completed window.
// Printed on window completion so ambient-light tests can compare feature values,
// which respond to noise in the 50 Hz stream that the printed DC levels cannot see.
#ifdef OXY_MODEL_V2
// v2 (ADR-0006): 14 IR features + ratio_of_ratios + ir_perfusion_index.
static const int FEATURE_COUNT = KAN_INPUT_DIM;
float currentFeatures[FEATURE_COUNT] = {0};
bool currentTwoWavelengthValid = false;
static const char* const* const FEATURE_NAMES = KAN_FEATURE_NAMES;
#elif defined(OXY_MODEL_P5)
// P5 (ADR-0008): the five beat inputs, then the beat state and the model saturation estimate
// (s_hat, estimated ScalcO2; display only) of the most recent completed window.
static const int FEATURE_COUNT = KAN_INPUT_DIM;
float currentFeatures[FEATURE_COUNT] = {0};
static const char* const* const FEATURE_NAMES = KAN_FEATURE_NAMES;
bool currentBeatValid = false;
uint8_t currentNBeats = 0;
float currentSatEst = 0.0f;
bool haveSatEst = false;  // false after a reset until the first completed window
#else
static const int FEATURE_COUNT = 14;
float currentFeatures[14] = {0};
static const char* const FEATURE_NAMES[14] = {
  "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
  "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
  "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5"
};
#endif

// Arduino's Print::printFloat emits "ovf" for any magnitude above ~4.29e9, which
// swallows the larger features (energy reaches ~7e13). Print those in scientific
// form so the diagnostic never silently loses a value.
static void printFloatSafe(float v) {
  if (isnan(v)) { Serial.print("nan"); return; }
  if (isinf(v)) { Serial.print(v > 0.0f ? "inf" : "-inf"); return; }
  if (v != 0.0f && (v > 4.0e9f || v < -4.0e9f)) {
    int exp10 = 0;
    float m = v;
    while (m >= 10.0f) { m /= 10.0f; exp10++; }
    while (m != 0.0f && m < 1.0f) { m *= 10.0f; exp10--; }
    Serial.print(m, 4);
    Serial.print('e');
    if (exp10 >= 0) Serial.print('+');
    Serial.print(exp10);
    return;
  }
  Serial.print(v, 4);
}

// BLE service with single combined characteristic
BLEService oxygenService("19B10000-E8F2-537E-4F6C-D104768A1214");

// Combined data: "spo2|status|explanation|red|ir|signal|xai1|xai2|xai3|bpm"
BLECharacteristic dataChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 220);

// BLE connection state
bool bleConnected = false;

#ifdef LOOP_TIMING_V2
#include "loop_timing_v2.h"
#endif
#ifdef PEAK_RAM_PROBE
#include "peak_ram_probe.h"
#endif

float estimateSpO2(uint32_t red, uint32_t ir) {
  // Demo-level SpO2 approximation.
  // NOT a calibrated pulse oximeter formula — for engineering demonstration only.
  if (DEMO_LOW_OXYGEN_MODE) {
    return 85.0f;
  }
  if (ir == 0 && red == 0) return 0.0f;
  float denom = (float)red + (float)ir;
  if (denom == 0.0f) return 0.0f;
  float irRatio = (float)ir / denom;
  // Maps IR/(Red+IR) to SpO2:
  //   irRatio ~0.50 -> ~97%
  //   irRatio ~0.45 -> ~95%
  //   irRatio ~0.60 -> ~99%
  float spo2 = 95.0f + 10.0f * (irRatio - 0.45f);
  if (spo2 > 100.0f) spo2 = 100.0f;
  if (spo2 < 0.0f)   spo2 = 0.0f;
  return spo2;
}

bool isSignalSaturated(uint32_t red, uint32_t ir) {
  return (red >= ADC_SATURATION_THRESHOLD || ir >= ADC_SATURATION_THRESHOLD);
}

bool isFingerPresent(uint32_t red, uint32_t ir) {
#ifdef PEAK_RAM_PROBE
#ifdef PEAK_RAM_LIVE_GATE
  // Sampler images: count the live gate's verdict and return it.
  const bool pr_live = red > PR_LIVE_FINGER_THRESHOLD && ir > PR_LIVE_FINGER_THRESHOLD;
  pr_count_live_gate(pr_live);
  return pr_live;
#else
  // Forced decisions: count what the live gate would say, then hold the gate open.
  pr_count_live_gate(red > PR_LIVE_FINGER_THRESHOLD && ir > PR_LIVE_FINGER_THRESHOLD);
  return true;
#endif
#else
  return (red > FINGER_PRESENT_THRESHOLD && ir > FINGER_PRESENT_THRESHOLD);
#endif
}

// MAX3010x FIFO (datasheet registers; the library keeps its copies file-static).
static const uint8_t MAX3010X_REG_FIFO_OVF  = 0x05;
static const uint8_t MAX3010X_REG_FIFO_DATA = 0x07;
static const uint8_t FIFO_DEPTH             = 32;
static const uint8_t BYTES_PER_RECORD       = 6;  // ledMode 2: Red + IR, 3 bytes each
static const uint8_t RECORDS_PER_READ       = 5;  // 30-byte Wire bursts

// One FIFO record: Red and IR sampled in the same slot pair.
struct PpgFrame {
  uint32_t red;
  uint32_t ir;
};

static uint32_t readFifoValue() {
  uint32_t value = (uint32_t)Wire.read() << 16;
  value |= (uint32_t)Wire.read() << 8;
  value |= (uint32_t)Wire.read();
  return value & 0x3FFFF;  // 18-bit ADC value
}

// Reads every unread record from the sensor's hardware FIFO into out[].
// Returns the number of records actually read.
//
// No overflow detection is used here. Register 0x05 (FIFO_OVF) was tried and is
// unusable on this part: it read as a constant 1, then oscillated 0/1 on every
// consecutive read, which is not the behaviour of a real loss counter. Acting on
// it reset the pipeline constantly and no window ever completed. Overflow is not
// reachable in this design instead: the hardware FIFO holds 32 records, which is
// 640 ms at 50 Hz, against a 250.8 ms worst-case decision block, so the FIFO is
// drained at least 2.5x before it could fill. That headroom is the safety
// argument, not the register.
uint8_t drainSensorFifo(PpgFrame out[FIFO_DEPTH]) {
  const uint8_t writePtr = particleSensor.getWritePointer();
  const uint8_t readPtr  = particleSensor.getReadPointer();

  // wr_ptr == rd_ptr means EMPTY here. A genuinely full 32-record FIFO would look
  // empty too, but that state is unreachable: see the note above.
  const uint8_t pending = (uint8_t)(writePtr - readPtr) & (FIFO_DEPTH - 1);
#ifdef LOOP_TIMING_V2
  lt_raw_wr = writePtr;
  lt_raw_rd = readPtr;
  lt_raw_pend = pending;
#endif
  if (pending == 0) return 0;

  Wire.beginTransmission(MAX30105_ADDRESS);
  Wire.write(MAX3010X_REG_FIFO_DATA);
  if (Wire.endTransmission() != 0) return 0;

  uint8_t got = 0;
  while (got < pending) {
    uint8_t chunk = pending - got;
    if (chunk > RECORDS_PER_READ) chunk = RECORDS_PER_READ;
    const int want = chunk * BYTES_PER_RECORD;

    int received = Wire.requestFrom(MAX30105_ADDRESS, want);
    if (received > Wire.available()) received = Wire.available();

    const int complete = received / BYTES_PER_RECORD;
    for (int i = 0; i < complete; ++i) {
      out[got].red = readFifoValue();
      out[got].ir  = readFifoValue();
      got++;
    }

    if (received < want) {
      while (Wire.available()) Wire.read();  // discard any partial record
      break;
    }
  }
  return got;
}

// Heart-rate estimate for display only; it is not a model input. Peak intervals on
// the last 8 s of the same 50 Hz infrared samples the pipeline ingests. A window
// gives a number only when its beats are evenly spaced; otherwise it gives -1.
static const int   HR_WINDOW        = 400;    // 8 s at 50 Hz
static const float HR_SAMPLE_HZ     = 50.0f;
static const int   HR_MIN_PEAK_GAP  = 17;     // samples; >= 333 ms between peaks
static const int   HR_SMOOTH_HALF   = 4;      // 9-sample (180 ms) smoothing
static const int   HR_BASE_HALF     = 37;     // 75-sample (1.5 s) baseline
static const float HR_PEAK_FRAC     = 0.6f;   // peaks must clear 60% of the window's range
static const int   HR_MIN_INTERVALS = 3;
static const float HR_MAX_GAP_RATIO = 1.4f;   // longest beat gap / shortest beat gap
static const int   HR_MIN_BPM       = 40;
static const int   HR_MAX_BPM       = 180;
static const int   HR_MAX_PEAKS     = HR_WINDOW / HR_MIN_PEAK_GAP + 1;

// File-scope: ~3.3 KB that must not land on the main-thread stack.
static int32_t  hrRing[HR_WINDOW];
static float    hrWork[HR_WINDOW];
static int16_t  hrPeaks[HR_MAX_PEAKS];
static uint16_t hrHead  = 0;
static uint16_t hrCount = 0;

// Display hold: the sent value is the median of the last 5 valid raw estimates,
// and it only moves when that median differs from the shown value by > 8 bpm.
static const uint8_t HR_HISTORY      = 5;
static const int     HR_HOLD_BPM     = 8;
static int     hrHistory[HR_HISTORY];
static int     hrSorted[HR_HISTORY];
static uint8_t hrHistHead  = 0;
static uint8_t hrHistCount = 0;
static int     hrShownBpm  = -1;

static void resetHeartRate() {
  hrHead = 0;
  hrCount = 0;
  hrHistHead = 0;
  hrHistCount = 0;
  hrShownBpm = -1;
}

static void pushHeartRateSample(uint32_t ir) {
  hrRing[hrHead] = (int32_t)ir;
  hrHead = (uint16_t)((hrHead + 1) % HR_WINDOW);
  if (hrCount < HR_WINDOW) hrCount++;
}

// i = 0 is the oldest sample once the ring is full.
static inline int32_t hrAt(int i) {
  return hrRing[(hrHead + i) % HR_WINDOW];
}

// Number of samples in the centred window [i - half, i + half], clipped to the ring.
static inline int hrSpan(int i, int half) {
  const int a = (i - half < 0) ? 0 : i - half;
  const int b = (i + half > HR_WINDOW - 1) ? HR_WINDOW - 1 : i + half;
  return b - a + 1;
}

// Returns bpm in [HR_MIN_BPM, HR_MAX_BPM], or -1 when unknown.
static int estimateHeartRateBpm() {
  if (hrCount < HR_WINDOW) return -1;

  // Baseline (1.5 s mean) minus smoothed signal (180 ms mean), from exact integer
  // running sums. This way round also inverts the trace, so systole (an IR dip)
  // becomes a peak. Subtracting ref keeps the sums far inside int32.
  const int32_t ref = hrAt(0);
  int32_t baseSum = 0, smoothSum = 0;
  for (int k = 0; k <= HR_BASE_HALF; ++k)   baseSum   += hrAt(k) - ref;
  for (int k = 0; k <= HR_SMOOTH_HALF; ++k) smoothSum += hrAt(k) - ref;
  float lo = 0.0f, hi = 0.0f;
  for (int i = 0; i < HR_WINDOW; ++i) {
    if (i > 0) {
      if (i + HR_BASE_HALF < HR_WINDOW)    baseSum   += hrAt(i + HR_BASE_HALF) - ref;
      if (i - HR_BASE_HALF - 1 >= 0)       baseSum   -= hrAt(i - HR_BASE_HALF - 1) - ref;
      if (i + HR_SMOOTH_HALF < HR_WINDOW)  smoothSum += hrAt(i + HR_SMOOTH_HALF) - ref;
      if (i - HR_SMOOTH_HALF - 1 >= 0)     smoothSum -= hrAt(i - HR_SMOOTH_HALF - 1) - ref;
    }
    const float d = (float)baseSum / hrSpan(i, HR_BASE_HALF)
                  - (float)smoothSum / hrSpan(i, HR_SMOOTH_HALF);
    hrWork[i] = d;
    if (i == 0 || d < lo) lo = d;
    if (i == 0 || d > hi) hi = d;
  }
  if (!(hi > lo)) return -1;  // flat trace

  // Local maxima above 60% of the range, at least HR_MIN_PEAK_GAP apart. A taller
  // maximum inside the gap replaces the previous peak.
  const float thr = lo + HR_PEAK_FRAC * (hi - lo);
  int peaks = 0;
  for (int i = 1; i < HR_WINDOW - 1; ++i) {
    const float v = hrWork[i];
    if (v < thr || v < hrWork[i - 1] || v <= hrWork[i + 1]) continue;
    if (peaks > 0 && i - hrPeaks[peaks - 1] < HR_MIN_PEAK_GAP) {
      if (v > hrWork[hrPeaks[peaks - 1]]) hrPeaks[peaks - 1] = (int16_t)i;
      continue;
    }
    if (peaks == HR_MAX_PEAKS) return -1;
    hrPeaks[peaks++] = (int16_t)i;
  }

  const int intervals = peaks - 1;
  if (intervals < HR_MIN_INTERVALS) return -1;

  // Evenly spaced beats only: the longest gap may be at most 1.4x the shortest.
  // Noise, missed beats, and double-counted beats all fail this.
  int minGap = HR_WINDOW, maxGap = 0;
  for (int k = 1; k < peaks; ++k) {
    const int g = hrPeaks[k] - hrPeaks[k - 1];
    if (g < minGap) minGap = g;
    if (g > maxGap) maxGap = g;
  }
  if ((float)maxGap > HR_MAX_GAP_RATIO * (float)minGap) return -1;

  const float meanGap = (float)(hrPeaks[peaks - 1] - hrPeaks[0]) / intervals;
  const int bpm = (int)(60.0f * HR_SAMPLE_HZ / meanGap + 0.5f);
  if (bpm < HR_MIN_BPM || bpm > HR_MAX_BPM) return -1;
  return bpm;
}

// The value sent over BLE. A rejected window sends -1 ("-") but keeps the history
// and hold, so the next accepted window resumes from the held value.
static int displayedHeartRateBpm() {
  const int raw = estimateHeartRateBpm();
  if (raw < 0) return -1;
  hrHistory[hrHistHead] = raw;
  hrHistHead = (uint8_t)((hrHistHead + 1) % HR_HISTORY);
  if (hrHistCount < HR_HISTORY) hrHistCount++;
  if (hrHistCount < HR_HISTORY) return -1;

  for (uint8_t i = 0; i < HR_HISTORY; ++i) {
    const int v = hrHistory[i];
    int j = i;
    while (j > 0 && hrSorted[j - 1] > v) { hrSorted[j] = hrSorted[j - 1]; --j; }
    hrSorted[j] = v;
  }
  const int median = hrSorted[HR_HISTORY / 2];

  if (hrShownBpm < 0 || abs(median - hrShownBpm) > HR_HOLD_BPM) hrShownBpm = median;
  return hrShownBpm;
}

// Clears the pipeline and cached decision so no window spans a break in the signal.
void resetRiskState() {
  pipeline.reset();
  currentRiskProb = 0.0f;
  isCandidateRisk = false;
  isConfirmedRisk = false;
  lastTotalSamples = 0;
  lastWindowIndex = 0;
  currentTopAttributions[0] = {0, 0.0f, "none"};
  currentTopAttributions[1] = {0, 0.0f, "none"};
  currentTopAttributions[2] = {0, 0.0f, "none"};
#ifdef OXY_MODEL_P5
  currentBeatValid = false;
  currentNBeats = 0;
  haveSatEst = false;
#endif
}

// Nano 33 BLE built-in RGB LED (active LOW: LOW = on, HIGH = off)
// Held permanently OFF: the LED is inside the enclosure and its light leaks into the photodiode.
void setRgbLed(bool r, bool g, bool b) {
  (void)r;
  (void)g;
  (void)b;
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
}

#ifndef OXY_SPO2_MAXIM
// Never called. Not built with -DOXY_SPO2_MAXIM, where SpO2 est can be below 90 (ADR-0010).
void printRiskStatus(bool fingerPresent, float spo2Estimate) {
  if (!fingerPresent) {
    Serial.println("  Status: NO FINGER");
  } else if (spo2Estimate < 90.0f) {
    Serial.print("  Status: LOW OXYGEN RISK");
    Serial.print("  | Explanation: Low oxygen detected because SpO2 (");
    Serial.print(spo2Estimate, 1);
    Serial.println("%) < 90% threshold.");
  } else {
    Serial.println("  Status: Normal");
  }
}
#endif

#if defined(OXY_MODEL_P5)
#ifndef OXY_XAI_REF_LABEL0
#error "The P5 live image uses the label-0-median explainer reference (-DOXY_XAI_REF_LABEL0)"
#endif
#include "ble_packet_p5.h"
// P5 image (ADR-0008, docs/xai-reference-label0-runbook.md). Both lines are printed at boot and
// again once per serial connection: the capture opens the port after the boot banner has gone.
static const char P5_MODEL_BANNER[] =
    "MODEL=P5: TinyKAN P5, 5 beat inputs r_beat pi_beat hr_bpm sqi_template sqi_ibi_ratio; "
    "s_hat = estimated ScalcO2 (model saturation estimate), display only, not an oxygen reading";
static const char XAI_REF_BANNER[] =
    "XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median of each standardized input, not the training mean";

// The shipped 10 fields plus field 10 (s_hat, one decimal, or "-") and field 11 (explainer
// reference tag). Every field is kept whole in 220 bytes; see ble_packet_p5.h.
void updateBLE(bool fingerPresent, float spo2Estimate, uint32_t redVal, uint32_t irVal,
               bool saturated, bool isConfirmed, bool isCandidate, float risk_prob,
               const FeatureAttribution* top_attributions, int bpm) {
  char packet[BLE_PACKET_P5_BYTES];
  static_assert(sizeof(packet) == 220, "packet buffer must equal the 220-byte characteristic");
  const char* const names[3] = {top_attributions[0].feature_name, top_attributions[1].feature_name,
                                top_attributions[2].feature_name};
  const float deltas[3] = {top_attributions[0].importance_delta, top_attributions[1].importance_delta,
                           top_attributions[2].importance_delta};
  const bool noFinger = !fingerPresent || irVal < FINGER_PRESENT_THRESHOLD;
  char satText[BLE_SAT_EST_TEXT_BYTES];
  sat_est_text(satText, sizeof(satText), noFinger, haveSatEst, currentBeatValid, currentSatEst);
  if (ble_packet_p5(packet, sizeof(packet), noFinger, spo2Estimate, (unsigned long)redVal,
                    (unsigned long)irVal, saturated, isConfirmed, isCandidate, risk_prob,
                    names, deltas, bpm, HR_MIN_BPM, HR_MAX_BPM, satText, KAN_XAI_REF_TAG) >= 0) {
    dataChar.writeValue(packet);
  }
}
#elif defined(OXY_XAI_REF_LABEL0)
#include "ble_packet_ref0.h"
// Label-0-reference image (docs/xai-reference-label0-runbook.md). Named at boot, and again
// once per serial connection: the capture opens the port after the boot banner has gone.
static const char XAI_REF_BANNER[] =
    "XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median of each standardized input, not the training mean";

// The shipped 10 fields plus field 10 (model saturation estimate, "-") and field 11 (explainer
// reference tag). Every field is kept whole in 220 bytes; see ble_packet_ref0.h.
void updateBLE(bool fingerPresent, float spo2Estimate, uint32_t redVal, uint32_t irVal,
               bool saturated, bool isConfirmed, bool isCandidate, float risk_prob,
               const FeatureAttribution* top_attributions, int bpm) {
  char packet[BLE_PACKET_BYTES];
  static_assert(sizeof(packet) == 220, "packet buffer must equal the 220-byte characteristic");
  const char* const names[3] = {top_attributions[0].feature_name, top_attributions[1].feature_name,
                                top_attributions[2].feature_name};
  const float deltas[3] = {top_attributions[0].importance_delta, top_attributions[1].importance_delta,
                           top_attributions[2].importance_delta};
  const bool noFinger = !fingerPresent || irVal < FINGER_PRESENT_THRESHOLD;
  if (ble_packet_ref0(packet, sizeof(packet), noFinger, spo2Estimate, (unsigned long)redVal,
                      (unsigned long)irVal, saturated, isConfirmed, isCandidate, risk_prob,
                      names, deltas, bpm, HR_MIN_BPM, HR_MAX_BPM, KAN_XAI_REF_TAG) >= 0) {
    dataChar.writeValue(packet);
  }
}
#else
void updateBLE(bool fingerPresent, float spo2Estimate, uint32_t redVal, uint32_t irVal,
               bool saturated, bool isConfirmed, bool isCandidate, float risk_prob,
               const FeatureAttribution* top_attributions, int bpm) {
  // Build one combined packet: "spo2|status|explanation|red|ir|signal|xai1|xai2|xai3|bpm"
  char packet[220];
  const char* status;
  const char* signal;
  char explBuf[100];
  char spo2Buf[8];
  int n;

  if (!fingerPresent || irVal < FINGER_PRESENT_THRESHOLD) {
    snprintf(spo2Buf, sizeof(spo2Buf), "---");
    status = "No finger detected";
    snprintf(explBuf, sizeof(explBuf), "Place finger on sensor");
    signal = "NO FINGER";
    bpm = -1;
    n = snprintf(packet, sizeof(packet), "%s|%s|%s|%lu|%lu|%s|none:0.0000|none:0.0000|none:0.0000",
             spo2Buf, status, explBuf, (unsigned long)redVal, (unsigned long)irVal, signal);
  } else {
#ifdef OXY_SPO2_MAXIM
    if (spo2Estimate < 0.0f) snprintf(spo2Buf, sizeof(spo2Buf), "---");  // no valid value in 70-100
    else snprintf(spo2Buf, sizeof(spo2Buf), "%.1f", spo2Estimate);
#else
    snprintf(spo2Buf, sizeof(spo2Buf), "%.1f", spo2Estimate);
#endif
    if (isConfirmed) {
      status = "CONFIRMED LOW OXYGEN RISK";
      snprintf(explBuf, sizeof(explBuf), "CONFIRMED RISK (Prob: %.2f) top driver: %s",
               risk_prob, top_attributions[0].feature_name);
    } else if (isCandidate) {
      status = "CANDIDATE RISK (Unconfirmed)";
      snprintf(explBuf, sizeof(explBuf), "Candidate risk detected (Prob: %.2f), awaiting confirmation",
               risk_prob);
    } else {
      status = "Normal";
      snprintf(explBuf, sizeof(explBuf), "Normal (Prob: %.2f)", risk_prob);
    }
    signal = saturated ? "CLIPPED" : "OK";

    n = snprintf(packet, sizeof(packet), "%s|%s|%s|%lu|%lu|%s|%s:%.4f|%s:%.4f|%s:%.4f",
                 spo2Buf, status, explBuf, (unsigned long)redVal, (unsigned long)irVal, signal,
                 top_attributions[0].feature_name, top_attributions[0].importance_delta,
                 top_attributions[1].feature_name, top_attributions[1].importance_delta,
                 top_attributions[2].feature_name, top_attributions[2].importance_delta);
  }

  // Field 9: bpm, or "-" when unknown. Falls back to "-" when the number would not
  // fit whole, so truncation can never turn 176 into 17.
  if (n > 0 && (size_t)n < sizeof(packet)) {
    char bpmBuf[5] = "-";
    if (bpm >= HR_MIN_BPM && bpm <= HR_MAX_BPM) snprintf(bpmBuf, sizeof(bpmBuf), "%d", bpm);
    const size_t room = sizeof(packet) - (size_t)n;  // includes the terminator
    if (strlen(bpmBuf) + 2 > room) { bpmBuf[0] = '-'; bpmBuf[1] = '\0'; }
    if (room >= 3) snprintf(packet + n, room, "|%s", bpmBuf);
  }

  dataChar.writeValue(packet);
}
#endif

#ifdef LED1_IDENTITY_CHECK
// LED1 identity diagnostic (env nano33ble_v2_led1check; outputs/firmware-v2-led1check/).
// Runs once in setup(), before startTime is set: LED1 (reg 0x0C) is switched off for
// 60 s and each labeled channel's per-second DC/min/max is printed, so the capture
// shows which label the LED1 slot lands in. Not the shipped image.
static void led1PrintMean(uint32_t sum, uint16_t n) {
  const uint32_t tenths = (sum * 10u + n / 2u) / n;  // 18-bit x ~50 records x 10 fits
  Serial.print(tenths / 10u);
  Serial.print('.');
  Serial.print(tenths % 10u);
}

static void led1IdentityCheck() {
  static PpgFrame frames[FIFO_DEPTH];  // static: main-thread stack is tight
  static const char *const PHASE_NAMES[3] = {"BOTH_ON", "LED1_OFF", "RESTORED"};
  static const uint16_t PHASE_SECONDS[3] = {15, 60, 10};

  Serial.println("LED1CHK begin: LED1_PA reg 0x0C via setPulseAmplitudeRed, LED2_PA reg 0x0D "
                 "via setPulseAmplitudeIR; labels: slot1 bytes -> red, slot2 bytes -> ir");
  const uint32_t t0 = millis();
  uint32_t secStart = t0;
  for (uint8_t phase = 0; phase < 3; ++phase) {
    if (phase == 1) {
      Serial.println("LED1_OFF");
      particleSensor.setPulseAmplitudeRed(0);
    } else if (phase == 2) {
      particleSensor.setPulseAmplitudeRed(0x3F);
      Serial.println("LED1_ON");
    }
    for (uint16_t s = 0; s < PHASE_SECONDS[phase]; ++s) {
      uint16_t n = 0;
      uint32_t redSum = 0, irSum = 0;
      uint32_t redMin = 0x3FFFF, redMax = 0, irMin = 0x3FFFF, irMax = 0;
      while ((uint32_t)(millis() - secStart) < 1000u) {
        const uint8_t count = drainSensorFifo(frames);
        for (uint8_t i = 0; i < count; ++i) {
          const uint32_t r = frames[i].red;
          const uint32_t ir = frames[i].ir;
          redSum += r;
          irSum += ir;
          if (r < redMin) redMin = r;
          if (r > redMax) redMax = r;
          if (ir < irMin) irMin = ir;
          if (ir > irMax) irMax = ir;
          ++n;
        }
        delay(20);
      }
      secStart += 1000u;

      Serial.print("LED1CHK phase=");
      Serial.print(PHASE_NAMES[phase]);
      Serial.print(" t=");
      Serial.print((millis() - t0) / 1000u);
      Serial.print(" n=");
      Serial.print(n);
      if (n == 0) {
        Serial.println(" red_dc=- ir_dc=- red_min=- red_max=- ir_min=- ir_max=-");
        continue;
      }
      Serial.print(" red_dc=");
      led1PrintMean(redSum, n);
      Serial.print(" ir_dc=");
      led1PrintMean(irSum, n);
      Serial.print(" red_min=");
      Serial.print(redMin);
      Serial.print(" red_max=");
      Serial.print(redMax);
      Serial.print(" ir_min=");
      Serial.print(irMin);
      Serial.print(" ir_max=");
      Serial.println(irMax);
    }
  }
  Serial.println("LED1CHK end");
  particleSensor.clearFIFO();
}
#endif

void setup()
{
  pinMode(interruptPin, INPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH);  // RGB LED off (active LOW); see setRgbLed()
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  setRgbLed(false, false, true); // Blue = startup / waiting for finger

  Serial.begin(115200);
  Serial.println();
  Serial.println("=== Semester 1 Milestone Demo ===");
  Serial.println("Sensor -> Arduino Nano 33 BLE Sense -> Serial Monitor + BLE Web Dashboard");
  Serial.println("Engineering risk-flagging prototype (NOT a certified oximeter)");
  if (DEMO_LOW_OXYGEN_MODE) {
    Serial.println("NOTE: DEMO_LOW_OXYGEN_MODE is ON (forces low-oxygen output for presentation)");
  }
#ifdef OXY_MODEL_V2
  Serial.println("Model: TinyKAN v2 (16 inputs: 14 IR + ratio_of_ratios + ir_perfusion_index; ADR-0006)");
#endif
#ifdef OXY_MODEL_P5
  Serial.println(P5_MODEL_BANNER);
#endif
#ifdef OXY_XAI_REF_LABEL0
  Serial.println(XAI_REF_BANNER);
#endif
  Serial.println();

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("ERROR: BLE initialization failed!");
    while (1);
  }

  BLE.setLocalName("OxygenRiskDemo");
  BLE.setAdvertisedService(oxygenService);

  oxygenService.addCharacteristic(dataChar);
  BLE.addService(oxygenService);

  dataChar.writeValue("---|Waiting...|Connect to Arduino to see live explanation|---|---|---");

  BLE.advertise();
  Serial.println("BLE advertising as 'OxygenRiskDemo'");
  Serial.println("Open Web Bluetooth dashboard in Chrome to view live data");
  Serial.println();

  // Initialize MAX30105 sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
  {
    Serial.println("ERROR: MAX30105 not found. Check wiring/power.");
    while (1);
  }

  byte ledBrightness = 0x3F;  // ~12.5 mA — 0x0F was too dim for finger IR to clear the gate
  byte sampleAverage = 4;     // Average 4 samples for smoother readings
  byte ledMode = 2;           // Red + IR only (Green is not needed for SpO2)
  int sampleRate = 200;       // 200 Hz / sampleAverage 4 = 50 FIFO records/s, the ML model's 50 Hz
  int pulseWidth = 118;       // Wider pulse for better SNR
  int adcRange = 16384;       // Full 18-bit ADC range to avoid clipping

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.enableAFULL();
  particleSensor.setFIFOAlmostFull(3);
#ifdef LED1_IDENTITY_CHECK
  led1IdentityCheck();
#endif

  startTime = millis();
#ifdef OXY_SPO2_MAXIM
  spo2m::reset(spo2State);
#endif

  Serial.println("Sensor initialized. Reading data...");
  Serial.println();
#ifdef LOOP_TIMING_V2
  lt_setup();
#endif
#ifdef PEAK_RAM_PROBE
  pr_setup_done();
#endif
}

void loop()
{
#ifdef PEAK_RAM_PROBE
  pr_poll();
#endif
#ifdef LOOP_TIMING_V2
  lt_pass_begin();
  LtStamp lt_a = lt_now();
#endif
#ifdef OXY_XAI_REF_LABEL0
  // USB output is dropped until the host raises DTR, so announce on each DTR rise.
  // Serial.dtr() reads a flag; `if (Serial)` would call delay(1) on every pass.
  static bool xaiRefAnnounced = false;
  const bool serialConnected = Serial.dtr();
#ifdef OXY_MODEL_P5
  if (serialConnected && !xaiRefAnnounced) Serial.println(P5_MODEL_BANNER);
#endif
  if (serialConnected && !xaiRefAnnounced) Serial.println(XAI_REF_BANNER);
  xaiRefAnnounced = serialConnected;
#endif
  // Check for BLE central connection
  BLEDevice central = BLE.central();
#ifdef LOOP_TIMING_V2
  lt_pass.central.add(lt_a, lt_now());
#endif

  if (central && !bleConnected) {
    bleConnected = true;
    Serial.print("BLE connected to: ");
    Serial.println(central.address());
    Serial.println();
  }

  if (!central && bleConnected) {
    bleConnected = false;
    Serial.println("BLE disconnected");
    Serial.println();
  }

  static PpgFrame frames[FIFO_DEPTH];  // static: keeps 256 B off the thread stack inference runs on
#ifdef LOOP_TIMING_V2
  lt_a = lt_now();
#endif
  const uint8_t frameCount = drainSensorFifo(frames);
#ifdef LOOP_TIMING_V2
  lt_on_drain(lt_a, lt_now(), frameCount);
#endif

  for (uint8_t f = 0; f < frameCount; ++f)
  {
    samplesTaken++;

    uint32_t redVal = frames[f].red;
    uint32_t irVal  = frames[f].ir;
    float readRate = (float)samplesTaken / ((millis() - startTime) / 1000.0);

    bool saturated = isSignalSaturated(redVal, irVal);
    bool fingerPresent = isFingerPresent(redVal, irVal);
#ifdef OXY_SPO2_MAXIM
    // Maxim reference algorithm (ADR-0010), display only; -1: no value (dash). The demo mode keeps 85.
    float spo2Estimate = -1.0f;
    if (fingerPresent && DEMO_LOW_OXYGEN_MODE) spo2Estimate = 85.0f;
    else if (fingerPresent && spo2m::has_value(spo2State)) spo2Estimate = spo2m::value(spo2State);
#else
    float spo2Estimate = fingerPresent ? estimateSpO2(redVal, irVal) : 0.0f;
#endif

    bool newWindow = false;
    if (!fingerPresent || irVal < FINGER_PRESENT_THRESHOLD) {
      // Reset pipeline state if no finger is detected
      resetRiskState();
      resetHeartRate();
#ifdef OXY_SPO2_MAXIM
      spo2m::reset(spo2State);
#endif
#ifdef LOOP_TIMING_V2
      lt_on_no_finger();
#endif
    } else {
#ifdef LOOP_TIMING_V2
      lt_a = lt_now();
#endif
#ifdef LOOP_PRIORITY_V2
      // F1a: the deciding push_sample below runs ~440 ms, and ArduinoBLE's Cordio
      // thread is the same priority as this loop, so round-robin stretches it past
      // the FIFO's 640 ms (FINDINGS addendum, 15 Oct 2026). Raise this thread for
      // that call only, and always restore it — loop() never blocks, so a permanent
      // raise would starve the BLE thread.
      const osPriority_t lt_saved_priority = osThreadGetPriority(osThreadGetId());
      osThreadSetPriority(osThreadGetId(), osPriorityAboveNormal);
#endif
      pushHeartRateSample(irVal);
#ifdef LOOP_TIMING_V2
      const LtStamp lt_b = lt_now();
      lt_pass.hrp.add(lt_a, lt_b);
#endif
      // Push sample into streaming OxygenRiskPipeline
#ifdef OXY_MODEL_V2
      // Red and IR from the same FIFO record (frames[f]).
#ifdef LOOP_TIMING_V2_PHASES
      // Diagnostic mirror of push_sample() with sub-phase timers (loop_timing_v2_phases.h).
      auto decision_opt = lt2_push_sample(pipeline, static_cast<float>(irVal), static_cast<float>(redVal));
#else
      auto decision_opt = pipeline.push_sample(static_cast<float>(irVal), static_cast<float>(redVal));
#endif
#elif defined(OXY_MODEL_P5)
      // Red and IR from the same FIFO record (frames[f]), as v2.
      auto decision_opt = pipeline.push_sample(static_cast<float>(irVal), static_cast<float>(redVal));
#else
      auto decision_opt = pipeline.push_sample(static_cast<float>(irVal));
#endif
#ifdef LOOP_TIMING_V2
      lt_on_push(lt_b, lt_now(), decision_opt.has_value(), f);
#endif
#ifdef LOOP_PRIORITY_V2
      osThreadSetPriority(osThreadGetId(), lt_saved_priority);
#endif
#ifdef OXY_SPO2_MAXIM
      if (spo2m::push(spo2State, redVal, irVal)) {
        // One line per computation (1 Hz), comparison only. Existing parsers ignore it.
        Serial.print("SPO2CMP: n=");          Serial.print(spo2State.computeN);
        Serial.print(" maxim=");              Serial.print(spo2State.lastRaw);
        Serial.print(" maxim_valid=");        Serial.print(spo2State.lastValid);
        Serial.print(" rf=");                 Serial.print(spo2State.rfSpo2, 1);
        Serial.print(" rf_valid=");           Serial.print(spo2State.rfValid);
        Serial.print(" rf_corr=");            Serial.print(spo2State.rfCorrel, 3);
        Serial.print(" shown=");
        if (spo2m::has_value(spo2State)) Serial.println(spo2State.shown); else Serial.println("-");
      }
#endif
      if (decision_opt.has_value()) {
#ifdef PEAK_RAM_PROBE
        pr_on_decision();
#endif
        newWindow = true;
        RiskDecision decision = decision_opt.value();
        currentRiskProb = decision.risk_probability;
        isCandidateRisk = decision.is_candidate_risk;
        isConfirmedRisk = decision.is_confirmed_risk;
        lastTotalSamples = decision.total_samples_ingested;
        lastWindowIndex = decision.window_index;
        for (int i = 0; i < 3; ++i) {
          currentTopAttributions[i] = decision.top_attributions[i];
        }
        for (int i = 0; i < FEATURE_COUNT; ++i) {
          currentFeatures[i] = decision.raw_features[i];
        }
#ifdef OXY_MODEL_V2
        currentTwoWavelengthValid = decision.two_wavelength_valid;
#endif
#ifdef OXY_MODEL_P5
        currentBeatValid = decision.beat_valid;
        currentNBeats = decision.n_beats;
        currentSatEst = decision.s_hat;
        haveSatEst = true;
#endif
      }
    }

    // Print every 25 samples (2 Hz at 50 Hz ingest) and always on a completed
    // TinyKAN window. The old ~3 Hz ingest was not caused by printing: blocking
    // getRed()/getIR()/getGreen() each waited for a new FIFO record, so one sample
    // cost 4 records at 12.5 records/s (sampleRate 50 / sampleAverage 4) = 3.125 Hz.
    const bool printNow = newWindow || (samplesTaken % 25 == 0);
    if (!printNow) {
      continue;
    }

#ifdef LOOP_TIMING_V2
    const LtStamp lt_st = lt_now();
#endif
    // Serial Monitor output
    Serial.print("Red: ");
    Serial.print(redVal);
    Serial.print("  IR: ");
    Serial.print(irVal);
    Serial.print("  Hz: ");
    Serial.print(readRate, 2);

    if (fingerPresent && irVal >= FINGER_PRESENT_THRESHOLD) {
      Serial.print("  SpO2_est: ");
#ifdef OXY_SPO2_MAXIM
      if (spo2Estimate >= 0.0f) {
        Serial.print(spo2Estimate, 1);
        Serial.print("%");
      } else {
        Serial.print("---");
      }
#else
      Serial.print(spo2Estimate, 1);
      Serial.print("%");
#endif
      Serial.print("  KAN_Prob: ");
      Serial.print(currentRiskProb, 2);
      Serial.print("  Candidate: ");
      Serial.print(isCandidateRisk ? "YES" : "NO");
      Serial.print("  Confirmed: ");
      Serial.print(isConfirmedRisk ? "YES" : "NO");
      Serial.print("  Win: ");
      Serial.print(lastWindowIndex);
      Serial.print("  Samples: ");
      Serial.print((unsigned long)lastTotalSamples);
      Serial.print("  TopFeature: ");
      Serial.print(currentTopAttributions[0].feature_name);

      // Second line, window completion only: the full feature vector. These
      // respond to broadband noise in the 50 Hz stream, which the printed DC
      // levels (one sample every 0.5 s) cannot resolve.
      if (newWindow) {
#ifdef LOOP_TIMING_V2
        const LtStamp lt_f = lt_now();
#endif
        Serial.print("\n  Feat:");
        for (int i = 0; i < FEATURE_COUNT; ++i) {
          Serial.print(" ");
          Serial.print(FEATURE_NAMES[i]);
          Serial.print("=");
          printFloatSafe(currentFeatures[i]);
        }
#ifdef OXY_MODEL_V2
        // 0: a raw mean or IR perfusion was not positive; the last two features are training means.
        Serial.print(" two_wavelength_valid=");
        Serial.print(currentTwoWavelengthValid ? 1 : 0);
#endif
#ifdef OXY_MODEL_P5
        // beat_valid=0: the five inputs above are their imputation means and s_hat is not shown.
        Serial.print(" n_beats=");
        Serial.print(currentNBeats);
        Serial.print(" beat_valid=");
        Serial.print(currentBeatValid ? 1 : 0);
        Serial.print(" s_hat=");
        if (currentBeatValid) printFloatSafe(currentSatEst);
        else Serial.print("-");
#endif
#ifdef LOOP_TIMING_V2
        lt_pass.feat.add(lt_f, lt_now());
#endif
      }

      if (saturated) {
        Serial.print("  Signal: CLIPPED");
      } else {
        Serial.print("  Signal: OK");
      }
    } else {
      Serial.print("  SpO2_est: ---");
      Serial.print("  Signal: NO FINGER");
    }

    // Print status & XAI telemetry
    if (!fingerPresent || irVal < FINGER_PRESENT_THRESHOLD) {
      Serial.println("  Status: NO FINGER");
    } else if (isConfirmedRisk) {
      Serial.print("  Status: CONFIRMED LOW OXYGEN RISK (Prob: ");
      Serial.print(currentRiskProb, 2);
      Serial.print(" | Win: ");
      Serial.print(lastWindowIndex);
      Serial.print(" | Samples: ");
      Serial.print((unsigned long)lastTotalSamples);
      Serial.print(" | Top3 Drivers: ");
      for (int i = 0; i < 3; ++i) {
        Serial.print("#");
        Serial.print(currentTopAttributions[i].feature_index);
        Serial.print(" ");
        Serial.print(currentTopAttributions[i].feature_name);
        Serial.print(":");
        Serial.print(currentTopAttributions[i].importance_delta, 4);
        if (i < 2) Serial.print(", ");
      }
      Serial.println(")");
    } else if (isCandidateRisk) {
      Serial.print("  Status: CANDIDATE RISK [Unconfirmed] (Prob: ");
      Serial.print(currentRiskProb, 2);
      Serial.print(" | Win: ");
      Serial.print(lastWindowIndex);
      Serial.println(")");
    } else {
      Serial.print("  Status: Normal (Prob: ");
      Serial.print(currentRiskProb, 2);
      Serial.print(" | Win: ");
      Serial.print(lastWindowIndex);
      Serial.println(")");
    }
#ifdef LOOP_TIMING_V2
    lt_pass.st.add(lt_st, lt_now());
    lt_pass.st_n++;
#endif

    // Update RGB LED to match current state (ADR-0005)
    if (!fingerPresent || irVal < FINGER_PRESENT_THRESHOLD) {
      setRgbLed(false, false, true); // Blue = no finger
    } else if (saturated) {
      setRgbLed(true, false, true);  // Purple = signal clipped
    } else if (isConfirmedRisk) {
      // Blink red for confirmed low-oxygen risk
      bool blink = (samplesTaken / 25) % 2 == 0; // Blink rate ~500ms at 50Hz
      setRgbLed(blink, false, false);
    } else {
      setRgbLed(false, true, false); // Green = normal / unconfirmed candidate
    }

    // Update BLE if connected
#ifdef PEAK_RAM_PROBE
    // Forced decisions: build and write the payload with or without a central; a
    // notification still goes out only to a subscribed central.
    if (bleConnected || pr_force_payload) {
#else
    if (bleConnected) {
#endif
#ifdef LOOP_TIMING_V2
      // Same calls in the same order; the bpm argument is evaluated first either way.
      const LtStamp lt_h = lt_now();
      const int lt_bpm = displayedHeartRateBpm();
      const LtStamp lt_w = lt_now();
      updateBLE(fingerPresent, spo2Estimate, redVal, irVal, saturated,
                isConfirmedRisk, isCandidateRisk, currentRiskProb, currentTopAttributions,
                lt_bpm);
      lt_pass.hre.add(lt_h, lt_w);
      lt_pass.ble.add(lt_w, lt_now());
      lt_pass.ble_n++;
#else
      updateBLE(fingerPresent, spo2Estimate, redVal, irVal, saturated,
                isConfirmedRisk, isCandidateRisk, currentRiskProb, currentTopAttributions,
                displayedHeartRateBpm());
#endif
    }
  }
#ifdef LOOP_TIMING_V2
  lt_pass_end();
#endif
}


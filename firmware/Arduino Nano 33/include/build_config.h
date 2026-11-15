#pragma once
// Build configuration: the one place that says which compile flags exist and which combinations are valid.
// Every image is (model) x (role); platformio.ini's header lists the env for each pair. Preprocessor checks
// only: this header defines no symbols and emits no code, so including it cannot change an image.
//
// Model (pick one; none = v1, 14 IR inputs on the as-coded labels):
//   OXY_MODEL_V2      TinyKAN v2, 16 paired IR/red inputs (ADR-0006); the "earlier spline-activation network"
//   OXY_MODEL_P5      P5, 5 beat inputs on the same paired context (ADR-0008)
//   OXY_MODEL_EKAN    edge KAN ekan16 as the scorer of the v2 pipeline (ADR-0011); needs OXY_MODEL_V2
// Explainer reference:
//   OXY_XAI_REF_LABEL0  label-0-median reference instead of the training mean (v2 and P5 only)
//   EKAN_XAI_REF_LABEL0 the same for the edge KAN (edge_kan_xai_reference.h); edge-KAN images only
// Display:
//   OXY_SPO2_MAXIM    SpO2 est from Maxim's reference algorithm (ADR-0010); v2-family live images only
// Role (at most one; none = the live sensor + BLE image):
//   PARITY_REPLAY_BUILD, CONTEXT_PARITY_BUILD, CONTEXT_TIMING_BUILD                       (v1)
//   CONTEXT_PARITY_V2_BUILD, CONTEXT_TIMING_V2_BUILD                                       (v2, also EKAN)
//   CONTEXT_PARITY_P5_BUILD, CONTEXT_TIMING_P5_BUILD                                       (P5)
//   CONTEXT_PARITY_EKAN_BUILD, CONTEXT_TIMING_EKAN_BUILD                                   (edge-KAN kernels)
// Live-image diagnostics (live role only):
//   LOOP_PRIORITY_V2, LOOP_TIMING_V2 [+ LOOP_TIMING_V2_PHASES], LED1_IDENTITY_CHECK,
//   PEAK_RAM_PROBE [+ PEAK_RAM_LIVE_GATE [+ PEAK_RAM_SAMPLER]]
// The local guards in the headers that use these flags stay where they are; this file repeats the
// cross-cutting rules so a bad combination fails here, first, with one message.

// ---- model
#if defined(OXY_MODEL_V2) && defined(OXY_MODEL_P5)
#error "build_config: OXY_MODEL_V2 and OXY_MODEL_P5 are separate models; define one"
#endif
#if defined(OXY_MODEL_EKAN) && !defined(OXY_MODEL_V2)
#error "build_config: OXY_MODEL_EKAN is the scorer of the v2 pipeline; add -DOXY_MODEL_V2"
#endif
#if defined(OXY_MODEL_EKAN) && defined(OXY_XAI_REF_LABEL0)
#error "build_config: the edge KAN uses the training-mean reference only"
#endif
#if defined(EKAN_XAI_REF_LABEL0) && !defined(OXY_MODEL_EKAN) && !defined(CONTEXT_PARITY_EKAN_BUILD)
#error "build_config: EKAN_XAI_REF_LABEL0 is the edge KAN's reference; it needs OXY_MODEL_EKAN or the edge context-parity image"
#endif
#if defined(EKAN_XAI_REF_LABEL0) && defined(OXY_XAI_REF_LABEL0)
#error "build_config: EKAN_XAI_REF_LABEL0 (edge KAN) and OXY_XAI_REF_LABEL0 (TinyKAN) are separate; define one"
#endif
#if defined(OXY_XAI_REF_LABEL0) && !defined(OXY_MODEL_V2) && !defined(OXY_MODEL_P5)
#error "build_config: OXY_XAI_REF_LABEL0 has a reference header for v2 and P5 only"
#endif
#if defined(OXY_SPO2_MAXIM) && (!defined(OXY_MODEL_V2) || defined(OXY_XAI_REF_LABEL0))
#error "build_config: OXY_SPO2_MAXIM is for v2-family live images with the training-mean reference (ADR-0010)"
#endif

// ---- role
#if (defined(PARITY_REPLAY_BUILD) + defined(CONTEXT_PARITY_BUILD) + defined(CONTEXT_TIMING_BUILD) + \
     defined(CONTEXT_PARITY_V2_BUILD) + defined(CONTEXT_TIMING_V2_BUILD) + \
     defined(CONTEXT_PARITY_P5_BUILD) + defined(CONTEXT_TIMING_P5_BUILD) + \
     defined(CONTEXT_PARITY_EKAN_BUILD) + defined(CONTEXT_TIMING_EKAN_BUILD)) > 1
#error "build_config: at most one replay or timing role per image"
#endif
#if (defined(CONTEXT_PARITY_V2_BUILD) || defined(CONTEXT_TIMING_V2_BUILD) || \
     defined(CONTEXT_PARITY_EKAN_BUILD) || defined(CONTEXT_TIMING_EKAN_BUILD)) && !defined(OXY_MODEL_V2)
#error "build_config: the v2 and edge-KAN replay roles need -DOXY_MODEL_V2"
#endif
#if (defined(CONTEXT_PARITY_P5_BUILD) || defined(CONTEXT_TIMING_P5_BUILD)) && !defined(OXY_MODEL_P5)
#error "build_config: the P5 replay roles need -DOXY_MODEL_P5"
#endif

// ---- live-image diagnostics
#if defined(LOOP_TIMING_V2_PHASES) && !defined(LOOP_TIMING_V2)
#error "build_config: LOOP_TIMING_V2_PHASES extends LOOP_TIMING_V2"
#endif
#if defined(PEAK_RAM_SAMPLER) && !defined(PEAK_RAM_LIVE_GATE)
#error "build_config: PEAK_RAM_SAMPLER runs the live finger gate (PEAK_RAM_LIVE_GATE)"
#endif
#if defined(PEAK_RAM_LIVE_GATE) && !defined(PEAK_RAM_PROBE)
#error "build_config: PEAK_RAM_LIVE_GATE extends PEAK_RAM_PROBE"
#endif
#if defined(PEAK_RAM_PROBE) && (defined(LOOP_TIMING_V2) || defined(LED1_IDENTITY_CHECK))
#error "build_config: PEAK_RAM_PROBE extends the live images only"
#endif

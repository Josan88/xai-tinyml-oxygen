# Architecture Decision Record (ADR): Mobile Gateway & Background Sync

**Status:** ON HOLD (Deferred / Future Work)  
**Date:** 2026-08-08  
**Context:** Capstone Project (EAT40005 / EAT40006) — *Lightweight Explainable TinyML Framework for Real-Time Oxygen Monitoring on Edge Devices*

---

## 📌 Context & Problem Statement

The TinyML edge device (Arduino Nano 33 BLE + MAX30105 PPG sensor) extracts 14 physiological features, executes a Kolmogorov-Arnold Network (TinyKAN) for oxygen risk classification, computes top-3 Mean-Perturbation XAI attributions on-device, and transmits telemetry via BLE GATT notifications.

An architectural evaluation was conducted to assess whether a **native Android background synchronization gateway app** (`android-gateway`) should be built alongside the existing **Web Bluetooth dashboard** (`github-pages-dashboard/`).

---

## ⚖️ Trade-off & Comparative Analysis

| Evaluation Criterion | Web Bluetooth Dashboard (`github-pages-dashboard`) | Native Android Background Gateway (`android-gateway`) |
| :--- | :--- | :--- |
| **Research Core Alignment** | **Directly satisfies Capstone Objective 4** (*"Transmit predictions and top-3 explanations via BLE to a Web Bluetooth dashboard"*). | Extends beyond core research into mobile software engineering & cloud plumbing. |
| **Deployment & Viva Friction** | **Zero installation overhead**: Runs instantly in Chrome/Edge on laptops and Android devices with one click. | Requires Android Studio, Gradle builds, APK sideloading, and Android runtime permission setup. |
| **Explainability Visualization** | Fully visualizes SpO2 gauges, low-oxygen risk status, and top-3 TinyKAN XAI attribution bar charts. | Similar Jetpack Compose visualization + Room SQLite database. |
| **Continuous Background Streaming** | ⚠️ Limited: Streaming pauses when the browser tab is backgrounded or mobile screen sleeps. | ✅ Full: Android Foreground Service (`FOREGROUND_SERVICE_CONNECTED_DEVICE`) streams 24/7 with screen off. |
| **Cloud Synchronization** | Optional manual session export / JavaScript Webhook. | Background `WorkManager` batch uploads (GZIP REST API) with exponential backoff. |

---

## 🎯 Decision: Put on Hold (Deferred)

The native Android app is **placed on hold** for the following reasons:

1. **Academic Scope Prioritization**: The primary research contribution is **on-device TinyML inference and Mean-Perturbation XAI efficiency on microcontrollers**, not mobile cloud integration.
2. **Existing Web Bluetooth Sufficiency**: The existing `github-pages-dashboard/` already provides real-time streaming, risk status alerts, and feature attribution visualization for presentation demos and viva examinations.
3. **Resource Preservation**: Avoiding the high maintenance overhead of Android build toolchains, Gradle dependencies, and OS battery optimization restrictions leaves full focus on Semester 2 embedded benchmarking (flash/RAM footprint, inference latency, explanation faithfulness).

---

## 🔮 Conditions for Re-activating the Android App

The Android Gateway architecture will be revisited if:
* [ ] A clinical/wearable deployment specifically requires **unattended 24/7 background telemetry logging** with the screen off.
* [ ] Audible / vibratory **emergency phone alarms** are required when low-oxygen risk occurs while the phone is in a pocket.
* [ ] Automated **centralized cloud telemetry archiving** is required across multiple patient nodes.

---

## 📐 Preserved Architecture Blueprint (For Future Implementation)

If re-activated, the recommended specification is:
* **Background Engine**: Android Foreground Service (`connectedDevice` type) with persistent status notification.
* **BLE Client**: Nordic Android-BLE-Library / Kotlin Coroutines Flow with auto-reconnection and MTU 512 negotiation.
* **Local Storage**: Room SQLite Database with in-memory packet buffering (flushed every 1–2s).
* **Optional Cloud Sync**: `WorkManager` with `PeriodicWorkRequest` (15m, `NetworkType.CONNECTED`) + threshold/urgent-risk trigger over GZIP REST API (`POST /api/v1/telemetry/batch`).
* **UI**: Jetpack Compose + Material 3 (SpO2 gauge, risk banner, top-3 TinyKAN XAI attribution charts).

---

## Correction (2026-10-09)

The context sentence above is left standing. It is not the shipped object. The firmware is a linear mix, then one cubic B-spline on each hidden node and one on the output. It is not an edge-spline KAN. The 14 inputs are infrared PPG features. The label is a median pulse-oximeter SpO2 below 90, not a blood-oxygen measurement. A top feature is not a physiological cause. Evidence: CONTEXT.md TinyKAN definition, and the Module 8 IEEE draft.

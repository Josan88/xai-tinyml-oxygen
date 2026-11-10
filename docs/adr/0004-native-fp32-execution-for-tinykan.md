# 0004 - Native FP32 Execution for TinyKAN on Cortex-M4F

We retain single-precision floating point (FP32) for TinyKAN weights, B-spline evaluation, and feature normalization instead of INT8 quantization. Because the nRF52840 features a hardware single-precision FPU and 1 MB Flash, storing 297 parameters (~1.2 KB) uses negligible resources while eliminating B-spline grid clipping and scale factor distortions.

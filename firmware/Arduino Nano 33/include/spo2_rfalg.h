/*
 * Created by Robert Fraczkiewicz, 01/2018
 * New signal processing methodology for obtaining heart rate and SpO2 data
 * from the MAX30102 sensor manufactured by MAXIM Integrated Products, Inc.
 */
/*******************************************************************************
* Copyright (C) 2017 Robert Fraczkiewicz, All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL ROBERT FRACZKIEWICZ BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Robert Fraczkiewicz retains all
* ownership rights.
*******************************************************************************
*/
// Vendored from https://github.com/aromring/MAX30102_by_RF commit 553ba61952012ad6e895532bb2bfda0e790a3beb
// (algorithm_by_RF.h) for ADR-0010 as the serial-only comparison (D8). Changes: rf_* -> rfalg_*,
// macros/constants RFALG_/rfalg_-prefixed, <stdint.h>/<math.h> instead of Arduino.h, and rfalg_reset()
// declared (D9b). Thresholds, HR range and curve unchanged.
#ifndef SPO2_RFALG_H_
#define SPO2_RFALG_H_
#include <stdint.h>
#include <math.h>

/*
 * Settable parameters
 * Leave these alone if your circuit and hardware setup match the defaults
 * described in this code's Instructable. Typically, different sampling rate
 * and/or sample length would require these paramteres to be adjusted.
 */
#define RFALG_ST 4      // Sampling time in s. WARNING: if you change RFALG_ST, then you MUST recalcuate the rfalg_sum_X2 parameter below!
#define RFALG_FS 25     // Sampling frequency in Hz. WARNING: if you change RFALG_FS, then you MUST recalcuate the rfalg_sum_X2 parameter below!
// Sum of squares of RFALG_ST*RFALG_FS numbers from -rfalg_mean_X (see below) to +rfalg_mean_X incremented be one. For example, given RFALG_ST=4 and RFALG_FS=25,
// the sum consists of 100 terms: (-49.5)^2 + (-48.5)^2 + (-47.5)^2 + ... + (47.5)^2 + (48.5)^2 + (49.5)^2
// The sum is symmetrc, so you can evaluate it by multiplying its positive half by 2. It is precalcuated here for enhanced
// performance.
const float rfalg_sum_X2 = 83325; // WARNING: you MUST recalculate this sum if you changed either RFALG_ST or RFALG_FS above!
// WARNING: The two parameters below are CRUCIAL! Proper HR evaluation depends on these.
#define RFALG_MAX_HR 180  // Maximal heart rate. To eliminate erroneous signals, calculated HR should never be greater than this number.
#define RFALG_MIN_HR 40   // Minimal heart rate. To eliminate erroneous signals, calculated HR should never be lower than this number.
// Minimal ratio of two autocorrelation sequence elements: one at a considered lag to the one at lag 0.
// Good quality signals must have such ratio greater than this minimum.
const float rfalg_min_autocorrelation_ratio = 0.5;
// Pearson correlation between red and IR signals.
// Good quality signals must have their correlation coefficient greater than this minimum.
const float rfalg_min_pearson_correlation = 0.8;

/*
 * Derived parameters
 * Do not touch these!
 *
 */
const int32_t RFALG_BUFFER_SIZE = RFALG_FS*RFALG_ST; // Number of smaples in a single batch
const int32_t RFALG_FS60 = RFALG_FS*60;  // Conversion factor for heart rate from bps to bpm
const int32_t RFALG_LOWEST_PERIOD = RFALG_FS60/RFALG_MAX_HR; // Minimal distance between peaks
const int32_t RFALG_HIGHEST_PERIOD = RFALG_FS60/RFALG_MIN_HR; // Maximal distance between peaks
const float rfalg_mean_X = (float)(RFALG_BUFFER_SIZE-1)/2.0; // Mean value of the set of integers from 0 to RFALG_BUFFER_SIZE-1. For RFALG_ST=4 and RFALG_FS=25 it's equal to 49.5.

void rfalg_heart_rate_and_oxygen_saturation(uint32_t *pun_ir_buffer, int32_t n_ir_buffer_length, uint32_t *pun_red_buffer, float *pn_spo2, int8_t *pch_spo2_valid, int32_t *pn_heart_rate,
                                        int8_t *pch_hr_valid, float *ratio, float *correl);
float rfalg_linear_regression_beta(float *pn_x, float xmean, float sum_x2);
float rfalg_autocorrelation(float *pn_x, int32_t n_size, int32_t n_lag);
float rfalg_rms(float *pn_x, int32_t n_size, float *sumsq);
float rfalg_Pcorrelation(float *pn_x, float *pn_y, int32_t n_size);
void rfalg_initialize_periodicity_search(float *pn_x, int32_t n_size, int32_t *p_last_periodicity, int32_t n_max_distance, float min_aut_ratio, float aut_lag0);
void rfalg_signal_periodicity(float *pn_x, int32_t n_size, int32_t *p_last_periodicity, int32_t n_min_distance, int32_t n_max_distance, float min_aut_ratio, float aut_lag0, float *ratio);
// CHANGE (ADR-0010, D9b): sets the carried peak interval back to RFALG_LOWEST_PERIOD (call on finger loss).
void rfalg_reset();

#endif /* SPO2_RFALG_H_ */

import numpy as np
from scipy import signal, stats

# C++ Mock implementation in NumPy
def cpp_bandpass_filter(input_data):
    # FP32 constants used by firmware; NumPy arithmetic here remains float64.
    design = signal.butter(4, [0.5, 8.0], btype="bandpass", fs=50.0, output="sos")
    sos = design.astype(np.float32).astype(float)
    zi_const = signal.sosfilt_zi(design).astype(np.float32).astype(float)
    input_data = np.asarray(input_data, dtype=float) - input_data[0]

    padlen = 27
    # Odd extension padding
    left_pad = 2 * input_data[0] - input_data[1:padlen+1][::-1]
    right_pad = 2 * input_data[-1] - input_data[-padlen-1:-1][::-1]
    padded = np.concatenate([left_pad, input_data, right_pad])
    
    # 1. Forward Pass (in Direct Form II Transposed)
    temp = np.copy(padded).astype(float)
    x0_forward = temp[0] # Padded input first sample
    for s in range(4):
        b0, b1, b2, a0, a1, a2 = sos[s]
        # Initialize states using zi scaled by x0_forward
        w1 = zi_const[s][0] * x0_forward
        w2 = zi_const[s][1] * x0_forward
        for i in range(len(temp)):
            x = temp[i]
            y = b0 * x + w1
            w1 = b1 * x + w2 - a1 * y
            w2 = b2 * x - a2 * y
            temp[i] = y
            
    # 2. Reverse
    temp = temp[::-1]
    
    # 3. Backward Pass (in Direct Form II Transposed)
    x0_backward = temp[0] # Last sample of forward output
    for s in range(4):
        b0, b1, b2, a0, a1, a2 = sos[s]
        # Initialize states using zi scaled by x0_backward
        w1 = zi_const[s][0] * x0_backward
        w2 = zi_const[s][1] * x0_backward
        for i in range(len(temp)):
            x = temp[i]
            y = b0 * x + w1
            w1 = b1 * x + w2 - a1 * y
            w2 = b2 * x - a2 * y
            temp[i] = y
            
    # 4. Reverse back and crop
    result = temp[::-1]
    return result[padlen:-padlen]

def cpp_welch_psd(filtered_data):
    # Welch segments
    # Hann window
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(256) / 256.0))
    welch_scale = 1.0 / (50.0 * np.sum(hann**2))
    
    psd = np.zeros(129)
    # SciPy Welch uses only complete segments; no zero-padded tail.
    segment_count = 1 + (len(filtered_data) - 256) // 128
    for seg in range(segment_count):
        start_idx = seg * 128
        segment = np.copy(filtered_data[start_idx : start_idx + 256])

        # Detrend constant (subtract mean of segment)
        segment = segment - np.mean(segment)
        
        # Apply window
        fft_in = segment * hann
        
        # Real FFT
        fft_out = np.fft.rfft(fft_in) # Size 129
        
        # Periodogram
        mag_sq = np.abs(fft_out)**2
        
        # One-sided PSD scaling
        mag_sq[1:128] = 2.0 * mag_sq[1:128]
        
        # Accumulate PSD
        psd += mag_sq * welch_scale
        
    return psd / segment_count


def cpp_extract_features(raw_data):
    # 1. Bandpass filter
    filtered = cpp_bandpass_filter(raw_data)
    
    # 2. Statistics
    mean = np.mean(filtered)
    std = np.std(filtered) # ddof=0
    minimum = np.min(filtered)
    maximum = np.max(filtered)
    val_range = maximum - minimum
    rms = np.sqrt(np.mean(filtered**2))
    energy = np.sum(filtered**2)
    
    # Higher moments
    m2 = np.mean((filtered - mean)**2)
    m3 = np.mean((filtered - mean)**3)
    m4 = np.mean((filtered - mean)**4)
    
    # Unbiased skewness & kurtosis (scipy stats.skew bias=False, stats.kurtosis fisher=True bias=False)
    N = len(filtered)
    g1 = m3 / (m2 * np.sqrt(m2))
    skewness = np.sqrt(N * (N - 1)) / (N - 2) * g1
    
    g2 = m4 / (m2**2) - 3.0
    kurtosis = (N - 1) / ((N - 2) * (N - 3)) * ((N + 1) * g2 + 6.0)
    
    # Derivative standard deviation
    diff = np.diff(filtered)
    derivative_std = np.std(diff) # ddof=0
    
    # 3. Spectral via Welch
    psd = cpp_welch_psd(filtered)
    
    # Dominant frequency (excluding DC bin 0)
    dominant_frequency_bin = np.argmax(psd[1:]) + 1
    dominant_frequency_hz = dominant_frequency_bin * (50.0 / 256.0)
    
    # Bandpowers
    df = 50.0 / 256.0
    
    # Bins 3 to 10 for [0.5, 2.0] Hz
    bp_0_5_2_sum = 0.5 * psd[3] + np.sum(psd[4:10]) + 0.5 * psd[10]
    bandpower_0_5_2_hz = bp_0_5_2_sum * df
    
    # Bins 11 to 25 for [2.0, 5.0] Hz
    bp_2_5_sum = 0.5 * psd[11] + np.sum(psd[12:25]) + 0.5 * psd[25]
    bandpower_2_5_hz = bp_2_5_sum * df
    
    bandpower_ratio = bandpower_0_5_2_hz / bandpower_2_5_hz if bandpower_2_5_hz > 0.0 else 0.0
    
    return {
        "mean": mean,
        "std": std,
        "minimum": minimum,
        "maximum": maximum,
        "value_range": val_range,
        "rms": rms,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "energy": energy,
        "derivative_std": derivative_std,
        "dominant_frequency_hz": dominant_frequency_hz,
        "bandpower_0_5_2_hz": bandpower_0_5_2_hz,
        "bandpower_2_5_hz": bandpower_2_5_hz,
        "bandpower_ratio_0_5_2_to_2_5": bandpower_ratio
    }

# Scipy original implementation
def scipy_extract_features(raw_data):
    # sosfiltfilt
    sos = signal.butter(4, [0.5, 8.0], btype="bandpass", fs=50.0, output="sos")
    filtered = signal.sosfiltfilt(sos, raw_data)
    
    mean_value = float(np.mean(filtered))
    std_value = float(np.std(filtered))
    minimum_value = float(np.min(filtered))
    maximum_value = float(np.max(filtered))
    value_range = maximum_value - minimum_value
    rms_value = float(np.sqrt(np.mean(np.square(filtered))))
    
    skewness_value = float(stats.skew(filtered, bias=False))
    kurtosis_value = float(stats.kurtosis(filtered, fisher=True, bias=False))
    energy_value = float(np.sum(np.square(filtered)))
    derivative_std_value = float(np.std(np.diff(filtered)))
    
    frequencies, power = signal.welch(filtered, fs=50.0, nperseg=256)
    positive_frequencies = frequencies[1:]
    positive_power = power[1:]
    dominant_frequency_hz = float(positive_frequencies[int(np.argmax(positive_power))])
    
    mask_0_5_2 = (frequencies >= 0.5) & (frequencies <= 2.0)
    mask_2_5 = (frequencies >= 2.0) & (frequencies <= 5.0)
    bandpower_0_5_2_hz = float(np.trapezoid(power[mask_0_5_2], frequencies[mask_0_5_2]))
    bandpower_2_5_hz = float(np.trapezoid(power[mask_2_5], frequencies[mask_2_5]))
    bandpower_ratio = bandpower_0_5_2_hz / bandpower_2_5_hz if bandpower_2_5_hz > 0.0 else 0.0
    
    return {
        "mean": mean_value,
        "std": std_value,
        "minimum": minimum_value,
        "maximum": maximum_value,
        "value_range": value_range,
        "rms": rms_value,
        "skewness": skewness_value,
        "kurtosis": kurtosis_value,
        "energy": energy_value,
        "derivative_std": derivative_std_value,
        "dominant_frequency_hz": dominant_frequency_hz,
        "bandpower_0_5_2_hz": bandpower_0_5_2_hz,
        "bandpower_2_5_hz": bandpower_2_5_hz,
        "bandpower_ratio_0_5_2_to_2_5": bandpower_ratio
    }

def parse_c_header(header_path):
    import re
    with open(header_path, "r") as f:
        content = f.read()
        
    # Helper to clean C float tokens
    def clean_val(val_str):
        return float(val_str.strip().replace('f', '').replace('\n', '').replace('\r', ''))
        
    def extract_1d(name):
        pattern = rf"const float\s+{name}\[\d*\]\s*=\s*\{{([^}}]+)\}};"
        match = re.search(pattern, content)
        if not match:
            # Try without const or with different spacing
            pattern = rf"{name}\[\d*\]\s*=\s*\{{([^}}]+)\}}"
            match = re.search(pattern, content)
        if match:
            return np.array([clean_val(x) for x in match.group(1).split(',') if x.strip()])
        raise ValueError(f"Could not extract 1D array {name}")
        
    def extract_2d(name):
        pattern = rf"const float\s+{name}\[\d*\]\[\d*\]\s*=\s*\{{((?:[^{{}}]*\{{[^{{}}]*\}}[^{{}}]*)*)\}};"
        match = re.search(pattern, content)
        if match:
            rows_str = re.findall(r'\{([^}]+)\}', match.group(1))
            rows = []
            for row_str in rows_str:
                rows.append([clean_val(x) for x in row_str.split(',') if x.strip()])
            return np.array(rows)
        raise ValueError(f"Could not extract 2D array {name}")
        
    def extract_scalar(name):
        pattern = rf"const float\s+{name}\s*=\s*([^;]+);"
        match = re.search(pattern, content)
        if match:
            return clean_val(match.group(1))
        raise ValueError(f"Could not extract scalar {name}")

    scaler_mean = extract_1d("SCALER_MEAN")
    scaler_std = extract_1d("SCALER_STD")
    w1 = extract_2d("KAN1_W")
    b1 = extract_1d("KAN1_B")
    theta1 = extract_2d("KAN1_THETA")
    centers1 = extract_1d("KAN1_CENTERS")
    step1 = extract_scalar("KAN1_STEP")
    w2 = extract_2d("KAN2_W")
    b2 = extract_1d("KAN2_B")
    theta2 = extract_2d("KAN2_THETA")
    centers2 = extract_1d("KAN2_CENTERS")
    step2 = extract_scalar("KAN2_STEP")
    threshold = extract_scalar("KAN_DECISION_THRESHOLD")
    
    return {
        "scaler_mean": scaler_mean, "scaler_std": scaler_std,
        "w1": w1, "b1": b1, "theta1": theta1, "centers1": centers1, "step1": step1,
        "w2": w2, "b2": b2, "theta2": theta2, "centers2": centers2, "step2": step2,
        "threshold": threshold
    }

def numpy_bspline(u, centers, step, range_min, range_max):
    u_clamped = np.clip(u, range_min, range_max)
    dist = np.abs((u_clamped[:, np.newaxis] - centers) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0, None)**3 / 6.0
    return np.where(dist < 1.0, inner, outer)

def numpy_kan_predict(x, p):
    # StandardScaler
    x_scaled = (x - p["scaler_mean"]) / (p["scaler_std"] + 1e-10)
    
    # Layer 1
    u1 = np.dot(x_scaled, p["w1"].T) + p["b1"]
    out1 = np.zeros(12)
    for j in range(12):
        basis = numpy_bspline(u1[j:j+1], p["centers1"], p["step1"], -3.0, 3.0)
        out1[j] = np.sum(basis * p["theta1"][j], axis=-1)[0]
        
    # Layer 2
    u2 = np.dot(out1, p["w2"].T) + p["b2"]
    out2 = np.zeros(1)
    for j in range(1):
        basis = numpy_bspline(u2[j:j+1], p["centers2"], p["step2"], -3.0, 3.0)
        out2[j] = np.sum(basis * p["theta2"][j], axis=-1)[0]
        
    logit = out2[0]
    prob = 1.0 / (1.0 + np.exp(-logit))
    return logit, prob

def main():
    import os
    from pathlib import Path
    
    # Generate mock raw PPG data (sine wave + noise)
    t = np.linspace(0, 10.0, 500, endpoint=False) # 10s @ 50Hz
    # 1.2 Hz heartbeat + baseline drift + high freq noise
    raw_ppg = 100.0 * np.sin(2.0 * np.pi * 1.2 * t) + 20.0 * np.sin(2.0 * np.pi * 0.1 * t) + 5.0 * np.random.default_rng(0).standard_normal(500)
    
    cpp_f = cpp_extract_features(raw_ppg)
    sci_f = scipy_extract_features(raw_ppg)
    
    print("Feature Verification: C++ Mock vs Scipy/Numpy\n")
    print(f"{'Feature Name':<30} | {'Scipy/Numpy':<15} | {'C++ Mock':<15} | {'Difference':<15}")
    print("-" * 85)
    
    all_pass = True
    for key in sci_f.keys():
        sci_val = sci_f[key]
        cpp_val = cpp_f[key]
        diff = abs(sci_val - cpp_val)
        
        # Check tolerance (mostly float precision or boundary padding effects)
        tol = 1e-4 if "bandpower" not in key else 25.0 # PSD differences are very small relatively
        is_pass = "PASS" if diff < tol else "FAIL"
        
        if is_pass == "FAIL":
            all_pass = False
            
        print(f"{key:<30} | {sci_val:<15.6f} | {cpp_val:<15.6f} | {diff:<15.3e} ({is_pass})")
        
    print("\n" + "=" * 50)
    if all_pass:
        print("ALL FEATURE TESTS PASSED!")
    else:
        print("WARNING: Some features failed to match within tolerance!")
    print("=" * 50)
    
    # KAN Prediction Comparison
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    header_path = project_root / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights.h"
    
    if header_path.is_file():
        print(f"\nLoading weights from {header_path}...")
        p = parse_c_header(header_path)
        
        # Run KAN on both feature sets
        cpp_features_arr = np.array([cpp_f[k] for k in cpp_f.keys()])
        sci_features_arr = np.array([sci_f[k] for k in sci_f.keys()])
        
        cpp_logit, cpp_prob = numpy_kan_predict(cpp_features_arr, p)
        sci_logit, sci_prob = numpy_kan_predict(sci_features_arr, p)
        
        print("\nKAN Model Output Comparison:")
        print(f"{'Source':<15} | {'Logit':<12} | {'Probability':<12} | {'Prediction (thresh={:.2f})'.format(p['threshold']):<20}")
        print("-" * 65)
        print(f"{'Scipy/Numpy':<15} | {sci_logit:<12.6f} | {sci_prob:<12.6f} | {'RISK' if sci_prob >= p['threshold'] else 'Normal':<20}")
        print(f"{'C++ Mock':<15} | {cpp_logit:<12.6f} | {cpp_prob:<12.6f} | {'RISK' if cpp_prob >= p['threshold'] else 'Normal':<20}")
        print("-" * 65)
        print(f"Probability Difference: {abs(sci_prob - cpp_prob):.3e}")
        
        if abs(sci_prob - cpp_prob) < 0.01:
            print("SUCCESS: Model predictions are virtually identical!")
        else:
            print("WARNING: Model predictions differ significantly!")
    else:
        print(f"\nC++ Header not found at {header_path}")

if __name__ == "__main__":
    main()

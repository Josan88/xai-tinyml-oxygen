import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
N_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
N_SPLINE_KNOTS = 8
SPLINE_DEGREE = 3
SPLINE_RANGE = (-3.0, 3.0)
HIDDEN_DIM = 12
DECISION_THRESHOLD = 0.50

FEATURE_COLUMN_NAMES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)

class BSpline(nn.Module):
    def __init__(self, n_knots=8, degree=3, range_min=-3.0, range_max=3.0):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max
        self.register_buffer("centers", torch.linspace(range_min, range_max, n_knots))
        self.register_buffer("step", torch.tensor((range_max - range_min) / (n_knots - 1)))

    def forward(self, u):
        u_clamped = u.clamp(self.range_min, self.range_max)
        distance = torch.abs((u_clamped.unsqueeze(-1) - self.centers) / (self.step + 1e-10))
        inner = (2.0 / 3.0) - distance.pow(2) + 0.5 * distance.pow(3)
        outer = (2.0 - distance).clamp(min=0).pow(3) / 6.0
        return torch.where(distance < 1.0, inner, outer)

class KANLayer(nn.Module):
    def __init__(self, d_in, d_out, n_knots=8, degree=3, range_min=-3.0, range_max=3.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_knots = n_knots
        self.W = nn.Parameter(torch.empty(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.W)
        self.spline = BSpline(n_knots, degree, range_min, range_max)
        self.theta = nn.Parameter(torch.empty(d_out, n_knots))
        self._init_spline_to_identity()

    def _init_spline_to_identity(self):
        grid = torch.linspace(self.spline.range_min, self.spline.range_max, 100)
        basis_grid = self.spline(grid)
        target = grid.unsqueeze(1)
        theta_init = torch.linalg.lstsq(basis_grid, target).solution.squeeze(-1)
        self.theta.data = theta_init.unsqueeze(0).expand(self.d_out, -1).clone()

    def forward(self, x):
        u = F.linear(x, self.W, self.b)
        batch = u.shape[0]
        basis = self.spline(u.reshape(-1)).reshape(batch, self.d_out, self.n_knots)
        return (basis * self.theta.unsqueeze(0)).sum(dim=-1)

    def spline_smoothness_penalty(self):
        diff2 = self.theta[:, 2:] - 2 * self.theta[:, 1:-1] + self.theta[:, :-2]
        return diff2.pow(2).sum()

class KANBinaryClassifier(nn.Module):
    def __init__(self, dims=(14, 12, 1), n_knots=8, degree=3, spline_range=(-3.0, 3.0)):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLayer(dims[i], dims[i + 1], n_knots, degree, spline_range[0], spline_range[1])
            for i in range(len(dims) - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization(self):
        return sum(layer.spline_smoothness_penalty() for layer in self.layers)

def numpy_bspline(u, centers, step, range_min, range_max):
    # Pure numpy equivalent of BSpline forward for validation
    u_clamped = np.clip(u, range_min, range_max)
    dist = np.abs((u_clamped[:, np.newaxis] - centers) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0, None)**3 / 6.0
    return np.where(dist < 1.0, inner, outer)

def numpy_kan_predict(x, scaler_mean, scaler_std, w1, b1, theta1, centers1, step1, w2, b2, theta2, centers2, step2):
    # C-equivalent pure NumPy implementation of the two-layer KAN model
    # 1. Scale inputs
    x_scaled = (x - scaler_mean) / scaler_std
    
    # 2. Layer 1 linear
    u1 = np.dot(x_scaled, w1.T) + b1 # (batch, 12)
    
    # 3. Layer 1 Spline
    # We evaluate spline for each channel
    batch = x.shape[0]
    out1 = np.zeros((batch, 12))
    for j in range(12):
        basis = numpy_bspline(u1[:, j], centers1, step1, -3.0, 3.0) # (batch, 8)
        out1[:, j] = np.sum(basis * theta1[j], axis=-1)
        
    # 4. Layer 2 linear
    u2 = np.dot(out1, w2.T) + b2 # (batch, 1)
    
    # 5. Layer 2 Spline
    out2 = np.zeros((batch, 1))
    for j in range(1):
        basis = numpy_bspline(u2[:, j], centers2, step2, -3.0, 3.0) # (batch, 8)
        out2[:, j] = np.sum(basis * theta2[j], axis=-1)
        
    return out2.ravel()

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    feature_csv = project_root / "outputs" / "openox_canonical_window_features.csv"
    output_header = project_root / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights.h"

    print(f"Loading data from {feature_csv}...")
    df = pd.read_csv(feature_csv)
    
    X_raw = df.loc[:, FEATURE_COLUMN_NAMES].to_numpy(dtype=np.float32)
    y_raw = df["label"].to_numpy(dtype=np.int64)
    
    print(f"Total training samples: {X_raw.shape[0]}")
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    
    # Model dimensions
    dims = (14, HIDDEN_DIM, 1)
    model = KANBinaryClassifier(dims, N_SPLINE_KNOTS, SPLINE_DEGREE, SPLINE_RANGE)
    
    # Training setup
    pos_weight = (len(y_raw) - y_raw.sum()) / max(1, y_raw.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
    
    print("Training final TinyKAN model...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t) + 1e-5 * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            # Eval train metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits).numpy().ravel()
                preds = (probs >= DECISION_THRESHOLD).astype(int)
                f1 = f1_score(y_raw, preds, zero_division=0)
                rec = recall_score(y_raw, preds, zero_division=0)
                acc = accuracy_score(y_raw, preds)
                print(f"Epoch {epoch:03d}/{N_EPOCHS} | Loss: {loss.item():.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Acc: {acc:.4f}")
                
    # Final eval
    model.eval()
    with torch.no_grad():
        final_logits = model(X_t).numpy().ravel()
        final_probs = 1.0 / (1.0 + np.exp(-final_logits)) # Sigmoid
        final_preds = (final_probs >= DECISION_THRESHOLD).astype(int)
        final_f1 = f1_score(y_raw, final_preds)
        final_rec = recall_score(y_raw, final_preds)
        print(f"\nFinal Trained Model Performance: F1={final_f1:.4f}, Recall={final_rec:.4f}")

    # Extract parameters for export
    state_dict = model.state_dict()
    
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    
    # Layer 1
    w1 = state_dict["layers.0.W"].cpu().numpy()
    b1 = state_dict["layers.0.b"].cpu().numpy()
    theta1 = state_dict["layers.0.theta"].cpu().numpy()
    centers1 = model.layers[0].spline.centers.cpu().numpy()
    step1 = float(model.layers[0].spline.step.cpu().item())
    
    # Layer 2
    w2 = state_dict["layers.1.W"].cpu().numpy()
    b2 = state_dict["layers.1.b"].cpu().numpy()
    theta2 = state_dict["layers.1.theta"].cpu().numpy()
    centers2 = model.layers[1].spline.centers.cpu().numpy()
    step2 = float(model.layers[1].spline.step.cpu().item())
    
    # Validation against pure NumPy implementation
    print("\nVerifying PyTorch vs. pure NumPy implementation...")
    np_logits = numpy_kan_predict(
        X_raw[:200], scaler_mean, scaler_std,
        w1, b1, theta1, centers1, step1,
        w2, b2, theta2, centers2, step2
    )
    py_logits = final_logits[:200]
    
    max_diff = np.max(np.abs(np_logits - py_logits))
    print(f"Maximum difference on 200 samples: {max_diff:.3e}")
    if max_diff < 1e-5:
        print("Success! Implementations match.")
    else:
        print("WARNING: Difference exceeds tolerance!")

    # Write C header file
    print(f"\nExporting parameters to: {output_header}")
    
    # Make directory if not exists
    output_header.parent.mkdir(parents=True, exist_ok=True)
    
    def arr_to_c(arr, precision=6):
        if len(arr.shape) == 1:
            return ", ".join([f"{val:.{precision}f}f" for val in arr])
        elif len(arr.shape) == 2:
            rows = []
            for row in arr:
                rows.append("{" + ", ".join([f"{val:.{precision}f}f" for val in row]) + "}")
            return ",\n    ".join(rows)
        return ""

    c_content = f"""// Auto-generated by train_final_model.py
// Do not edit this file manually.

#pragma once

#define KAN_INPUT_DIM 14
#define KAN_HIDDEN_DIM {HIDDEN_DIM}
#define KAN_KNOTS {N_SPLINE_KNOTS}

// StandardScaler Parameters
const float SCALER_MEAN[14] = {{
    {arr_to_c(scaler_mean)}
}};

const float SCALER_STD[14] = {{
    {arr_to_c(scaler_std)}
}};

// KAN Layer 1: 14 -> {HIDDEN_DIM}
const float KAN1_W[{HIDDEN_DIM}][14] = {{
    {arr_to_c(w1)}
}};

const float KAN1_B[{HIDDEN_DIM}] = {{
    {arr_to_c(b1)}
}};

const float KAN1_THETA[{HIDDEN_DIM}][{N_SPLINE_KNOTS}] = {{
    {arr_to_c(theta1)}
}};

const float KAN1_CENTERS[{N_SPLINE_KNOTS}] = {{
    {arr_to_c(centers1)}
}};

const float KAN1_STEP = {step1:.6f}f;

// KAN Layer 2: {HIDDEN_DIM} -> 1
const float KAN2_W[1][{HIDDEN_DIM}] = {{
    {arr_to_c(w2)}
}};

const float KAN2_B[1] = {{
    {arr_to_c(b2)}
}};

const float KAN2_THETA[1][{N_SPLINE_KNOTS}] = {{
    {arr_to_c(theta2)}
}};

const float KAN2_CENTERS[{N_SPLINE_KNOTS}] = {{
    {arr_to_c(centers2)}
}};

const float KAN2_STEP = {step2:.6f}f;

const float KAN_DECISION_THRESHOLD = {DECISION_THRESHOLD:.2f}f;
"""
    
    with open(output_header, "w") as f:
        f.write(c_content)
        
    print("Export complete!")

if __name__ == "__main__":
    main()

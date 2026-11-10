import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
N_EPOCHS = 80
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
N_SPLINE_KNOTS = 8
SPLINE_DEGREE = 3

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

FEATURE_COLUMN_NAMES = (
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5",
)

class BSpline(nn.Module):
    def __init__(self, n_knots=8, degree=3, range_min=-2.0, range_max=2.0):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max
        self.register_buffer("centers", torch.linspace(range_min, range_max, n_knots))
        self.register_buffer("step", torch.tensor((range_max - range_min) / (n_knots - 1)))

    def forward(self, u):
        if self.degree != 3:
            raise ValueError("This compact implementation supports cubic B-splines only.")
        u_clamped = u.clamp(self.range_min, self.range_max)
        distance = torch.abs((u_clamped.unsqueeze(-1) - self.centers) / (self.step + 1e-10))
        inner = (2.0 / 3.0) - distance.pow(2) + 0.5 * distance.pow(3)
        outer = (2.0 - distance).clamp(min=0).pow(3) / 6.0
        return torch.where(distance < 1.0, inner, outer)

class KANLayer(nn.Module):
    def __init__(self, d_in, d_out, n_knots=8, degree=3, range_min=-2.0, range_max=2.0):
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
    def __init__(self, dims=(14, 8, 1), n_knots=8, degree=3, spline_range=(-2.0, 2.0)):
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

def train_fold_model(x_train, y_train, hidden_dim, spline_range):
    torch.manual_seed(SEED)
    dims = (14, hidden_dim, 1)
    model = KANBinaryClassifier(dims, N_SPLINE_KNOTS, SPLINE_DEGREE, spline_range)
    pos_weight_value = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t) + 1e-5 * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    return model

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    feature_csv = project_root / "outputs" / "openox_canonical_window_features.csv"
    output_csv = project_root / "outputs" / "openox_tinykan_sweep_results.csv"

    print(f"Loading features from: {feature_csv}")
    feature_df = pd.read_csv(feature_csv)
    
    X = feature_df.loc[:, FEATURE_COLUMN_NAMES].to_numpy(dtype=np.float32)
    y = feature_df["label"].to_numpy(dtype=np.int64)
    groups = feature_df["recording_id"].to_numpy()

    # Define hyperparameter grid
    hidden_dims = [4, 8, 12, 16, 24, 32]
    spline_ranges = [(-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]
    thresholds = np.arange(0.1, 0.91, 0.05)

    sweep_records = []

    splitter = GroupKFold(n_splits=N_SPLITS)

    for hidden_dim in hidden_dims:
        for spline_range in spline_ranges:
            print(f"Evaluating hidden_dim={hidden_dim}, spline_range={spline_range}...")
            
            # Count parameters
            dummy_model = KANBinaryClassifier((14, hidden_dim, 1), N_SPLINE_KNOTS, SPLINE_DEGREE, spline_range)
            param_count = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)

            pooled_true = []
            pooled_prob = []

            for split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
                scaler = StandardScaler()
                x_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
                x_test = scaler.transform(X[test_idx]).astype(np.float32)
                y_train = y[train_idx]
                y_test = y[test_idx]

                model = train_fold_model(x_train, y_train, hidden_dim, spline_range)
                
                model.eval()
                with torch.no_grad():
                    x_test_t = torch.tensor(x_test, dtype=torch.float32)
                    logits = model(x_test_t)
                    probs = torch.sigmoid(logits).numpy().ravel()
                
                pooled_true.extend(y_test.tolist())
                pooled_prob.extend(probs.tolist())

            pooled_true = np.array(pooled_true)
            pooled_prob = np.array(pooled_prob)

            # Sweep decision threshold
            best_f1 = -1.0
            best_thresh = 0.5
            best_metrics = {}

            for thresh in thresholds:
                preds = (pooled_prob >= thresh).astype(int)
                f1 = f1_score(pooled_true, preds, zero_division=0)
                acc = accuracy_score(pooled_true, preds)
                prec = precision_score(pooled_true, preds, zero_division=0)
                rec = recall_score(pooled_true, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_metrics = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1
                    }

            record = {
                "hidden_dim": hidden_dim,
                "spline_range_min": spline_range[0],
                "spline_range_max": spline_range[1],
                "param_count": param_count,
                "best_threshold": round(best_thresh, 3),
                "accuracy": round(best_metrics["accuracy"], 4),
                "precision": round(best_metrics["precision"], 4),
                "recall": round(best_metrics["recall"], 4),
                "f1": round(best_metrics["f1"], 4),
            }
            sweep_records.append(record)
            print(f"  -> Best Threshold: {record['best_threshold']:.2f} | F1: {record['f1']:.4f} | Recall: {record['recall']:.4f} | Params: {param_count}")

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.sort_values(by="f1", ascending=False, inplace=True)
    sweep_df.to_csv(output_csv, index=False)

    print("\n" + "="*50)
    print("TOP 5 CONFIGURATIONS BY POOLED F1 SCORE:")
    print("="*50)
    print(sweep_df.head(5).to_string(index=False))
    print("="*50)
    print(f"Sweep results exported to {output_csv}")

if __name__ == "__main__":
    main()

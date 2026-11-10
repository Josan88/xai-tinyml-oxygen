"""Training jobs and the header-based host replica for the Lamaakal-style analysis.

Analysis models only. Nothing here writes a weights header, a firmware file, or any
existing outputs/ directory. Retrained TinyKANs exist to measure seed spread and
ablations; they are not the shipped model.

TinyKAN recipe (the classes are imported from scripts/train_tinykan_v2.py, which copies
them unchanged from scripts/train_final_model.py and notebook 12):
  B-spline node KAN, cubic, Adam lr 0.01, weight decay 1e-4, pos_weight BCE, spline
  smoothness 1e-5, grad-norm clip 10, full-batch, StandardScaler fitted on the training
  fold only, torch.manual_seed(seed) immediately before model construction.
  v1 CV: 80 epochs (notebook 12 / scripts/sweep_kan.py), the recipe of the frozen
  out-of-fold predictions. v2 CV: 100 epochs (scripts/train_tinykan_v2.py).
"""
from __future__ import annotations

import time

import numpy as np

KNOWN_LEARNERS = ("tinykan", "logreg", "knn", "svm_rbf", "random_forest", "hist_gb", "mlp_16", "mlp_64_32")

# Fixed, modest baseline settings (not tuned on the test folds). Recorded in summary.json.
BASELINE_SETTINGS = {
    "logreg": "LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42) "
              "[notebook 11 comparator configuration]",
    "knn": "KNeighborsClassifier(n_neighbors=25, weights='uniform'); unweighted, so 0.50 is not prevalence-balanced",
    "svm_rbf": "SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42) "
               "fitted on a fixed 8,000-window random subsample of each training fold (cost); Platt probabilities",
    "random_forest": "RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight='balanced', random_state=42)",
    "hist_gb": "HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, "
               "class_weight='balanced', early_stopping=False, random_state=42)",
    "mlp_16": "MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-4, max_iter=300, random_state=42), "
              "balanced sample weights; parameter-matched to TinyKAN",
    "mlp_64_32": "MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=300, random_state=42), "
                 "balanced sample weights",
}
SVM_SUBSAMPLE = 8000
BASELINE_SEED = 42


def tinykan_param_count(n_inputs, hidden, knots):
    return hidden * n_inputs + hidden + hidden * knots + hidden + 1 + knots


def tinykan_header_floats(n_inputs, hidden, knots):
    """Floats in the generated header: trainables + scaler mean/std + centers x2 + steps x2 + threshold."""
    return tinykan_param_count(n_inputs, hidden, knots) + 2 * n_inputs + 2 * knots + 2 + 1


def train_tinykan(x_scaled, y, hidden, knots, spline_range, epochs, seed):
    """scripts/train_tinykan_v2.train_kan with hidden/knots/range/epochs/seed exposed."""
    import torch
    import torch.nn as nn
    from train_tinykan_v2 import KANBinaryClassifier

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = KANBinaryClassifier((x_scaled.shape[1], hidden, 1), knots, 3, spline_range)
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t) + 1e-5 * model.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    model.eval()
    return model


def _tinykan_fold(x_train, y_train, x_test, spec):
    import torch

    model = train_tinykan(x_train, y_train, spec["hidden"], spec["knots"], tuple(spec["spline_range"]),
                          spec["epochs"], spec["seed"])
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32))).numpy().ravel()
    size = int(sum(t.numel() for t in model.parameters()))
    return p.astype(np.float64), size


def _baseline_fold(x_train, y_train, x_test, learner):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.utils.class_weight import compute_sample_weight

    if learner == "logreg":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=BASELINE_SEED)
        model.fit(x_train, y_train)
        size = int(model.coef_.size + model.intercept_.size)
    elif learner == "knn":
        model = KNeighborsClassifier(n_neighbors=25, weights="uniform")
        model.fit(x_train, y_train)
        size = int(x_train.size)
    elif learner == "svm_rbf":
        rng = np.random.default_rng(BASELINE_SEED)
        subset = np.sort(rng.choice(len(y_train), size=min(SVM_SUBSAMPLE, len(y_train)), replace=False))
        model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True,
                    random_state=BASELINE_SEED)
        model.fit(x_train[subset], y_train[subset])
        size = int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size + 2)
    elif learner == "random_forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced",
                                       n_jobs=1, random_state=BASELINE_SEED)
        model.fit(x_train, y_train)
        size = int(sum(tree.tree_.node_count for tree in model.estimators_))
    elif learner == "hist_gb":
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                               class_weight="balanced", early_stopping=False,
                                               random_state=BASELINE_SEED)
        model.fit(x_train, y_train)
        size = int(sum(len(predictor.nodes) for stage in model._predictors for predictor in stage))
    elif learner in ("mlp_16", "mlp_64_32"):
        layers = (16,) if learner == "mlp_16" else (64, 32)
        model = MLPClassifier(hidden_layer_sizes=layers, alpha=1e-4, max_iter=300, random_state=BASELINE_SEED)
        model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        size = int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    else:
        raise ValueError(f"Unknown learner {learner}")
    return model.predict_proba(x_test)[:, 1].astype(np.float64), size


def run_cv_job(spec, x, y, folds):
    """Out-of-fold probabilities for one job. x is raw features (float32), folds are 1..K."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # sklearn 1.9 deprecates SVC(probability=True) (removal in 1.11); it still runs Platt scaling here.
    warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*", category=FutureWarning)
    started = time.perf_counter()
    probabilities = np.full(len(y), np.nan)
    sizes = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        scaler = StandardScaler()  # fitted on the training fold only
        if spec["learner"] == "tinykan":
            x_train = scaler.fit_transform(x[train]).astype(np.float32)
            x_test = scaler.transform(x[test]).astype(np.float32)
            p, size = _tinykan_fold(x_train, y[train], x_test, spec)
        else:
            x_train = scaler.fit_transform(x[train].astype(np.float64))
            x_test = scaler.transform(x[test].astype(np.float64))
            p, size = _baseline_fold(x_train, y[train], x_test, spec["learner"])
        probabilities[test] = p
        sizes.append(size)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{spec['name']}: incomplete out-of-fold coverage")
    return {"name": spec["name"], "probabilities": probabilities, "fold_sizes": sizes,
            "seconds": time.perf_counter() - started}


def run_cv_job_from_arrays(payload):
    """Process-pool entry point: payload = (spec, x, y, folds)."""
    spec, x, y, folds = payload
    return run_cv_job(spec, x, y, folds)


class HeaderKAN:
    """Host replica of kan_inference.h for any input count, from a generated weights header.

    Constants are the header's FP32 values evaluated in float64, raw inputs are rounded to
    FP32 before standardisation, and u is clamped to [-3, 3] as in the firmware. This is
    the same arithmetic as scripts/evaluate_xai_faithfulness.FrozenKAN, generalised from 14
    inputs. It is not bitwise MCU arithmetic.
    """

    RANGE = (-3.0, 3.0)

    def __init__(self, header_path):
        from train_tinykan_v2 import parse_header

        raw = parse_header(header_path)
        self.p = {key: np.asarray(value, dtype=np.float32).astype(np.float64) for key, value in raw.items()}
        self.n_inputs = self.p["SCALER_MEAN"].shape[0]
        self.hidden = self.p["KAN1_W"].shape[0]
        self.knots = self.p["KAN1_CENTERS"].shape[0]

    def standardize(self, raw):
        raw = np.asarray(raw, dtype=np.float32).astype(np.float64)
        return (raw - self.p["SCALER_MEAN"]) / (self.p["SCALER_STD"] + 1e-10)

    def _spline(self, u, layer):
        lo, hi = self.RANGE
        distance = np.abs((np.clip(u, lo, hi)[..., None] - self.p[f"KAN{layer}_CENTERS"])
                          / (float(self.p[f"KAN{layer}_STEP"]) + 1e-10))
        inner = 2.0 / 3.0 - distance ** 2 + 0.5 * distance ** 3
        outer = np.maximum(2.0 - distance, 0.0) ** 3 / 6.0
        return np.where(distance < 1.0, inner, outer)

    def hidden_preactivation(self, scaled):
        return np.asarray(scaled, dtype=np.float64) @ self.p["KAN1_W"].T + self.p["KAN1_B"]

    def hidden_output(self, scaled):
        u1 = self.hidden_preactivation(scaled)
        return np.sum(self._spline(u1, 1) * self.p["KAN1_THETA"][None, :, :], axis=2)

    def output_preactivation(self, scaled):
        return self.hidden_output(scaled) @ self.p["KAN2_W"][0] + self.p["KAN2_B"][0]

    def predict(self, scaled):
        scaled = np.asarray(scaled, dtype=np.float64)
        if scaled.ndim != 2 or scaled.shape[1] != self.n_inputs or not np.isfinite(scaled).all():
            raise ValueError(f"Expected finite N x {self.n_inputs} standardized features")
        u2 = self.output_preactivation(scaled)
        logit = self._spline(u2, 2) @ self.p["KAN2_THETA"][0]
        return 1.0 / (1.0 + np.exp(-logit))

    def phi(self, u, layer, unit):
        """Learned univariate function phi(u) of one node, firmware basis, clamped u."""
        theta = self.p[f"KAN{layer}_THETA"][unit]
        return self._spline(np.asarray(u, dtype=np.float64), layer) @ theta

    def mean_perturbation(self, scaled):
        """N+1 passes: delta_i = P(x) - P(x with standardized feature i set to 0, the training mean)."""
        probability = self.predict(scaled)
        attribution = np.empty_like(scaled, dtype=np.float64)
        for feature in range(scaled.shape[1]):
            changed = np.array(scaled, dtype=np.float64, copy=True)
            changed[:, feature] = 0.0
            attribution[:, feature] = probability - self.predict(changed)
        return probability, attribution

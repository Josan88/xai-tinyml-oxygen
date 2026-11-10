"""Plot frozen TinyKAN univariate B-splines as an architecture figure.

Reads firmware `kan_model_weights.h`. Does not retrain, requantise, or emit
operator explanations. Live XAI remains the 15-pass mean-perturbation top-3.

Run from project root:
  python scripts/plot_tinykan_architecture_splines.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_H = PROJECT_ROOT / "firmware" / "Arduino Nano 33" / "include" / "kan_model_weights.h"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
CAPTION_PATH = OUTPUT_DIR / "tinykan_architecture_splines.md"
FIGURE_PATH = OUTPUT_DIR / "tinykan_architecture_splines.png"

RANGE_MIN = -3.0
RANGE_MAX = 3.0


def _floats(block: str) -> np.ndarray:
    return np.fromstring(block.replace("f", " "), sep=",", dtype=np.float64)


def parse_weights(text: str) -> dict:
    def matrix(name: str) -> np.ndarray:
        match = re.search(
            rf"const float {name}\[[^\]]+\](?:\[[^\]]+\])? = \{{(.*?)\}};",
            text,
            re.S,
        )
        if not match:
            raise RuntimeError(f"missing {name}")
        rows = re.findall(r"\{([^{}]+)\}", match.group(1))
        if rows:
            return np.vstack([_floats(row) for row in rows])
        return _floats(match.group(1))

    def scalar(name: str) -> float:
        match = re.search(rf"const float {name} = ([^;]+)f;", text)
        if not match:
            raise RuntimeError(f"missing {name}")
        return float(match.group(1))

    hidden = int(re.search(r"#define KAN_HIDDEN_DIM (\d+)", text).group(1))
    knots = int(re.search(r"#define KAN_KNOTS (\d+)", text).group(1))
    theta1 = matrix("KAN1_THETA")
    theta2 = matrix("KAN2_THETA")
    if theta1.shape != (hidden, knots) or theta2.shape != (1, knots):
        raise RuntimeError(f"unexpected theta shapes {theta1.shape} {theta2.shape}")
    return {
        "hidden": hidden,
        "knots": knots,
        "theta1": theta1,
        "theta2": theta2,
        "centers1": matrix("KAN1_CENTERS").reshape(-1),
        "centers2": matrix("KAN2_CENTERS").reshape(-1),
        "step1": scalar("KAN1_STEP"),
        "step2": scalar("KAN2_STEP"),
    }


def bspline_basis(u: np.ndarray, centers: np.ndarray, step: float) -> np.ndarray:
    """Firmware cubic basis in kan_inference.h (clamped u)."""
    u_clamped = np.clip(u, RANGE_MIN, RANGE_MAX)
    dist = np.abs((u_clamped[:, None] - centers[None, :]) / (step + 1e-10))
    inner = (2.0 / 3.0) - dist**2 + 0.5 * dist**3
    outer = np.clip(2.0 - dist, 0.0, None) ** 3 / 6.0
    return np.where(dist < 1.0, inner, np.where(dist < 2.0, outer, 0.0))


def phi(u: np.ndarray, theta_row: np.ndarray, centers: np.ndarray, step: float) -> np.ndarray:
    return bspline_basis(u, centers, step) @ theta_row


def main() -> None:
    weights = parse_weights(WEIGHTS_H.read_text(encoding="utf-8"))
    u = np.linspace(RANGE_MIN, RANGE_MAX, 401)
    hidden = weights["hidden"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 9.2), sharex=True)
    axes = axes.ravel()

    for j in range(hidden):
        ax = axes[j]
        y = phi(u, weights["theta1"][j], weights["centers1"], weights["step1"])
        ax.plot(u, y, color="#1565C0", lw=1.6)
        ax.axhline(0.0, color="#CCCCCC", lw=0.6)
        ax.axvline(0.0, color="#CCCCCC", lw=0.6)
        ax.set_title(f"h{j}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if j % 4 == 0:
            ax.set_ylabel(r"$\varphi(u)$", fontsize=8)

    ax = axes[hidden]
    y_out = phi(u, weights["theta2"][0], weights["centers2"], weights["step2"])
    ax.plot(u, y_out, color="#E65100", lw=1.8)
    ax.axhline(0.0, color="#CCCCCC", lw=0.6)
    ax.axvline(0.0, color="#CCCCCC", lw=0.6)
    ax.set_title("output", fontsize=9)
    ax.grid(True, alpha=0.25)

    for ax in axes[hidden + 1 :]:
        ax.set_visible(False)
    for ax in axes[: hidden + 1]:
        ax.set_xlim(RANGE_MIN, RANGE_MAX)
        ax.tick_params(labelsize=7)
    for ax in axes[8:12]:
        ax.set_xlabel("u (pre-activation, clamped)", fontsize=8)

    fig.suptitle(
        "Frozen TinyKAN (14, 12, 1) univariate B-splines — architecture",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = """# TinyKAN architecture splines

**File:** `outputs/figures/tinykan_architecture_splines.png`

**What this is.** Host plot of the frozen firmware B-splines $\\varphi(u)$ for TinyKAN `(14, 12, 1)`: twelve hidden units and the scalar output spline. Cubic 8-knot basis, $u$ clamped to $[-3, 3]$, coefficients from `firmware/Arduino Nano 33/include/kan_model_weights.h`, matching `kan_inference.h`.

Live explanations are the 15-pass mean-perturbation top-3 (ADR-0001).

**Paper use.** One architecture figure: this model is a KAN because each coordinate after linear mixing is a plotted 1-D spline. Caption: architecture.
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")
    print(f"saved {FIGURE_PATH}")
    print(f"saved {CAPTION_PATH}")


if __name__ == "__main__":
    main()

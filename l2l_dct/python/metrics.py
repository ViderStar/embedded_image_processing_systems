"""Image quality metrics."""

from __future__ import annotations

import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, peak: float = 255.0) -> float:
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / err)


def coding_gain(subband_vars: np.ndarray, input_var: float) -> float:
    """Subband coding gain CG in dB (eq. 4.3)."""
    subband_vars = np.maximum(subband_vars, 1e-12)
    geom = np.exp(np.mean(np.log(subband_vars)))
    return 10.0 * np.log10(input_var / geom)

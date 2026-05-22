"""
2-D non-separable wavelet transform based on Lab 1 CDF 9/7 lifting.

Separable 2-D transform:  Y = Θ · X · Θᵀ
Non-separable block form: y = Θ̈ · x,  where Θ̈ = D(Θ) · P · D(Θ) · P

Reference: Rybenkov & Petrovsky, "2-D non-separable integer implementation..."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LAB1_DIR = Path(__file__).resolve().parent.parent / "lab1"
sys.path.insert(0, str(LAB1_DIR))

from cdf97_lifting import (
    cdf97_forward_1d,
    cdf97_inverse_1d,
    cdf97_inverse_1d_fixed,
    from_fixed,
)


def vec_row_major(block: np.ndarray) -> np.ndarray:
    """Flatten n×n block to vector (row-major order)."""
    return block.reshape(-1, order="C")


def unvec_row_major(vec: np.ndarray, n: int) -> np.ndarray:
    return vec.reshape(n, n, order="C")


def permutation_matrix(n: int) -> np.ndarray:
    """
    Permutation matrix P for row↔column vectorization:
    tv(Xᵀ) = P · tv(X)
    """
    size = n * n
    p = np.zeros((size, size), dtype=np.float64)
    for r in range(n):
        for c in range(n):
            src = r * n + c
            dst = c * n + r
            p[dst, src] = 1.0
    return p


def build_1d_analysis_matrix(n: int) -> np.ndarray:
    """
    Build n×n analysis matrix Θ such that Θ·x = [low; high] for length-n signal.
    Uses floating-point reference for matrix construction; lifting uses fixed-point.
    """
    theta = np.zeros((n, n), dtype=np.float64)
    unit = np.eye(n)
    for col in range(n):
        low, high = cdf97_forward_1d(unit[:, col])
        low_f = low.astype(np.float64) / (1 << 16)
        high_f = high.astype(np.float64) / (1 << 16)
        half = n // 2
        theta[:half, col] = low_f[:half]
        theta[half:, col] = high_f[: (n - half)]
    return theta


def build_nonseparable_matrix(n: int) -> np.ndarray:
    """
    Build 2-D non-separable transform matrix Θ̈ = D(Θ) · P · D(Θ) · P.
    """
    theta = build_1d_analysis_matrix(n)
    p = permutation_matrix(n)
    d_theta = np.kron(np.eye(n), theta)
    return d_theta @ p @ d_theta @ p


def nonseparable_forward_2d_block(block: np.ndarray) -> np.ndarray:
    """Apply non-separable 2-D forward transform to n×n block via matrix multiply."""
    n = block.shape[0]
    theta2d = build_nonseparable_matrix(n)
    x_vec = vec_row_major(block.astype(np.float64))
    y_vec = theta2d @ x_vec
    return unvec_row_major(y_vec, n)


def nonseparable_inverse_2d_block(coeffs: np.ndarray) -> np.ndarray:
    """Apply non-separable 2-D inverse transform to n×n block."""
    n = coeffs.shape[0]
    theta2d = build_nonseparable_matrix(n)
    inv = np.linalg.inv(theta2d)
    y_vec = vec_row_major(coeffs.astype(np.float64))
    x_vec = inv @ y_vec
    return unvec_row_major(x_vec, n)


def separable_forward_2d(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-level 2-D separable DWT (row transform, then column transform).
    Returns LL, LH, HL, HH subbands (fixed-point int64).
    """
    if image.shape[0] % 2:
        image = image[:-1]
    if image.shape[1] % 2:
        image = image[:, :-1]
    h, w = image.shape

    row_low = np.zeros((h, w // 2), dtype=np.int64)
    row_high = np.zeros((h, w // 2), dtype=np.int64)
    for r in range(h):
        lo, hi = cdf97_forward_1d(image[r])
        row_low[r] = lo
        row_high[r] = hi

    def col_forward(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_lo = np.zeros((h // 2, w // 2), dtype=np.int64)
        out_hi = np.zeros((h // 2, w // 2), dtype=np.int64)
        for c in range(data.shape[1]):
            lo, hi = cdf97_forward_1d(data[:, c], already_fixed=True)
            out_lo[:, c] = lo
            out_hi[:, c] = hi
        return out_lo, out_hi

    ll, lh = col_forward(row_low)
    hl, hh = col_forward(row_high)
    return ll, lh, hl, hh


def separable_inverse_2d(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Inverse one-level 2-D separable DWT."""
    w2 = ll.shape[1]

    row_low = np.zeros((height, w2), dtype=np.int64)
    row_high = np.zeros((height, w2), dtype=np.int64)
    for c in range(w2):
        row_low[:, c] = cdf97_inverse_1d_fixed(ll[:, c], lh[:, c], height)
        row_high[:, c] = cdf97_inverse_1d_fixed(hl[:, c], hh[:, c], height)

    result = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        x = cdf97_inverse_1d_fixed(row_low[r], row_high[r], width)
        result[r] = from_fixed(x)
    return result


def verify_nonseparable_equals_separable(n: int = 8) -> float:
    """Verify that non-separable matrix matches separable Θ⊗Θ on random block."""
    theta = build_1d_analysis_matrix(n)
    theta2d_ns = build_nonseparable_matrix(n)
    theta2d_sep = np.kron(theta, theta)
    rng = np.random.default_rng(0)
    x = rng.random(n * n)
    err = np.linalg.norm(theta2d_ns @ x - theta2d_sep @ x) / np.linalg.norm(x)
    return float(err)

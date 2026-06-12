"""Reference DCT-II / DCT-III (ODCT) matrix transforms."""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 8

_MATRIX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def dct_scale(k: int, n: int = BLOCK_SIZE) -> float:
    return 1.0 / np.sqrt(n) if k == 0 else np.sqrt(2.0 / n)


def build_dct_matrix(n: int = BLOCK_SIZE) -> np.ndarray:
    """Build orthonormal DCT-II matrix C (n x n), cached."""
    if n not in _MATRIX_CACHE:
        c = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            for i in range(n):
                c[k, i] = dct_scale(k, n) * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        _MATRIX_CACHE[n] = (c, c.T)
    return _MATRIX_CACHE[n][0]


def build_idct_matrix(n: int = BLOCK_SIZE) -> np.ndarray:
    """Build orthonormal inverse DCT-III matrix D = C^T, cached."""
    if n not in _MATRIX_CACHE:
        build_dct_matrix(n)
    return _MATRIX_CACHE[n][1]


def dct_1d(signal: np.ndarray, n: int = BLOCK_SIZE) -> np.ndarray:
    return build_dct_matrix(n) @ np.asarray(signal, dtype=np.float64)


def idct_1d(coeffs: np.ndarray, n: int = BLOCK_SIZE) -> np.ndarray:
    return build_idct_matrix(n) @ np.asarray(coeffs, dtype=np.float64)


def dct_2d(block: np.ndarray, n: int = BLOCK_SIZE) -> np.ndarray:
    c = build_dct_matrix(n)
    b = np.asarray(block, dtype=np.float64)
    return c @ b @ c.T


def idct_2d(block: np.ndarray, n: int = BLOCK_SIZE) -> np.ndarray:
    d = build_idct_matrix(n)
    b = np.asarray(block, dtype=np.float64)
    return d @ b @ d.T


def c2d_batch(blocks: np.ndarray) -> np.ndarray:
    """Batched C @ block @ C.T for blocks shaped (N, 8, 8)."""
    c = build_dct_matrix(BLOCK_SIZE)
    return np.einsum("ij,njk,kl->nil", c, blocks, c.T, optimize=True)


def d2d_batch(blocks: np.ndarray) -> np.ndarray:
    """Batched D @ block @ D.T for blocks shaped (N, 8, 8)."""
    d = build_idct_matrix(BLOCK_SIZE)
    return np.einsum("ij,njk,kl->nil", d, blocks, d.T, optimize=True)

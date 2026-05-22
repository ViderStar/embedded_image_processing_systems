"""
CDF 9/7 wavelet transform — lifting scheme with fixed-point arithmetic.

Rational lifting coefficients (Wang et al., 2009):
    α = -3/2,  β = -1/16,  γ = 4/5,  δ = 15/32,  k = 4/5,  1/k = 5/4
"""

from __future__ import annotations

import numpy as np

FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# Rational lifting coefficients (numerator, denominator)
ALPHA = (-3, 2)
BETA = (-1, 16)
GAMMA = (4, 5)
DELTA = (15, 32)
K = (4, 5)
INV_K = (5, 4)


def mul_rational(value: np.int64 | int, num: int, den: int) -> np.int64:
    """Multiply fixed-point value by rational num/den with rounding."""
    return np.int64((np.int64(value) * num + (den // 2 * (1 if num >= 0 else -1))) // den)


def sym_index(i: int, n: int) -> int:
    """Half-sample symmetric extension index."""
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        else:
            i = 2 * n - i - 1
    return i


def to_fixed(image: np.ndarray) -> np.ndarray:
    return image.astype(np.int64) << FRAC_BITS


def from_fixed(fixed: np.ndarray) -> np.ndarray:
    rounded = (fixed + (SCALE // 2)) >> FRAC_BITS
    return np.clip(rounded, 0, 255).astype(np.uint8)


def cdf97_forward_1d(signal: np.ndarray, already_fixed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward 1-D CDF 9/7 lifting transform (analysis).

    Parameters
    ----------
    signal : 1-D array (even length recommended)
    already_fixed : if True, signal is already in Q16 fixed-point format

    Returns
    -------
    low, high : approximation and detail subbands
    """
    if already_fixed:
        x = np.asarray(signal, dtype=np.int64).copy()
    else:
        x = to_fixed(np.asarray(signal, dtype=np.float64))
    n = len(x)

    s = x[0::2].copy()
    d = x[1::2].copy()
    ns, nd = len(s), len(d)

    for i in range(nd):
        si = sym_index(i, ns)
        sn = sym_index(i + 1, ns)
        d[i] += mul_rational(s[sn] + s[si], *ALPHA)

    for i in range(ns):
        di = sym_index(i, nd)
        dm = sym_index(i - 1, nd)
        s[i] += mul_rational(d[di] + d[dm], *BETA)

    for i in range(nd):
        si = sym_index(i, ns)
        sn = sym_index(i + 1, ns)
        d[i] += mul_rational(s[sn] + s[si], *DELTA)

    for i in range(ns):
        di = sym_index(i, nd)
        dm = sym_index(i - 1, nd)
        s[i] += mul_rational(d[di] + d[dm], *GAMMA)

    for i in range(ns):
        s[i] = mul_rational(s[i], *K)
    for i in range(nd):
        d[i] = mul_rational(d[i], *INV_K)

    return s, d


def _inverse_lifting_steps(s: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core inverse lifting steps on subbands (fixed-point)."""
    s = np.asarray(s, dtype=np.int64).copy()
    d = np.asarray(d, dtype=np.int64).copy()
    ns, nd = len(s), len(d)

    for i in range(nd):
        d[i] = mul_rational(d[i], *K)
    for i in range(ns):
        s[i] = mul_rational(s[i], *INV_K)

    for i in range(ns):
        di = sym_index(i, nd)
        dm = sym_index(i - 1, nd)
        s[i] -= mul_rational(d[di] + d[dm], *GAMMA)

    for i in range(nd):
        si = sym_index(i, ns)
        sn = sym_index(i + 1, ns)
        d[i] -= mul_rational(s[sn] + s[si], *DELTA)

    for i in range(ns):
        di = sym_index(i, nd)
        dm = sym_index(i - 1, nd)
        s[i] -= mul_rational(d[di] + d[dm], *BETA)

    for i in range(nd):
        si = sym_index(i, ns)
        sn = sym_index(i + 1, ns)
        d[i] -= mul_rational(s[sn] + s[si], *ALPHA)

    return s, d


def cdf97_inverse_1d(low: np.ndarray, high: np.ndarray, length: int) -> np.ndarray:
    """
    Inverse 1-D CDF 9/7 lifting transform (synthesis).

    Parameters
    ----------
    low, high : subbands from forward transform
    length : original signal length
    """
    s, d = _inverse_lifting_steps(low, high)
    x = np.zeros(length, dtype=np.int64)
    x[0::2] = s
    x[1::2] = d[: length // 2]
    return from_fixed(x)


def cdf97_inverse_1d_subbands(low: np.ndarray, high: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse 1-D transform returning fixed-point polyphase subbands (for 2-D chaining)."""
    s, d = _inverse_lifting_steps(low, high)
    return s, d[: length // 2]


def cdf97_inverse_1d_fixed(low: np.ndarray, high: np.ndarray, length: int) -> np.ndarray:
    """Inverse 1-D transform returning fixed-point samples (without uint8 conversion)."""
    s, d = _inverse_lifting_steps(low, high)
    x = np.zeros(length, dtype=np.int64)
    x[0::2] = s
    x[1::2] = d[: length // 2]
    return x


def cdf97_forward_rows(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply 1-D forward transform to each row of a grayscale image."""
    h, w = image.shape
    low = np.zeros((h, (w + 1) // 2), dtype=np.int64)
    high = np.zeros((h, w // 2), dtype=np.int64)
    for row in range(h):
        lo, hi = cdf97_forward_1d(image[row])
        low[row, : len(lo)] = lo
        high[row, : len(hi)] = hi
    return low, high


def cdf97_inverse_rows(low: np.ndarray, high: np.ndarray, width: int) -> np.ndarray:
    """Apply 1-D inverse transform to each row."""
    h = low.shape[0]
    result = np.zeros((h, width), dtype=np.uint8)
    for row in range(h):
        result[row] = cdf97_inverse_1d(low[row], high[row], width)
    return result


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error (СКО)."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio (ПССШ / PSNR) in dB."""
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / err)

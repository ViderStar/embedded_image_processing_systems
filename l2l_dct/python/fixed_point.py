"""Fixed-point arithmetic for L2L DCT-ODCT."""

from __future__ import annotations

import numpy as np

# Pixel format: Q8.0 (uint8) promoted to Q16.16 for internal math
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# Methodology comparison format S1.10 (12-bit) for MATLAB cross-check
WORD_BITS = 12
METH_FRAC_BITS = 10


def saturate(value: np.int64 | int | np.ndarray, bits: int = 32) -> np.int64 | np.ndarray:
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    return np.clip(value, lo, hi).astype(np.int64)


def round_to_frac(value: np.int64 | int | np.ndarray, frac_bits: int = FRAC_BITS) -> np.int64 | np.ndarray:
    half = 1 << (frac_bits - 1)
    v = np.asarray(value, dtype=np.int64)
    pos = v >= 0
    result = np.empty_like(v)
    result[pos] = (v[pos] + half) >> frac_bits
    result[~pos] = (v[~pos] - half + 1) >> frac_bits
    return result if isinstance(value, np.ndarray) else np.int64(result)


def to_fixed(values: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    scaled = np.round(values.astype(np.float64) * (1 << frac_bits)).astype(np.int64)
    return scaled


def from_fixed(values: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    return values.astype(np.float64) / (1 << frac_bits)


def mul_const(value: np.int64 | int, coeff: float, frac_bits: int = FRAC_BITS) -> np.int64:
    coeff_fp = int(round(coeff * (1 << frac_bits)))
    return np.int64(round_to_frac(np.int64(value) * coeff_fp, frac_bits))


def ladder_round(values: np.ndarray, lost_bits: int = 15) -> np.ndarray:
    """Ladder rounding node: zero lower bits of Q16 fraction (default ≈ S1.10)."""
    v = values.astype(np.int64)
    half = 1 << (lost_bits - 1)
    return ((v + half) >> lost_bits) << lost_bits


def round_block(values: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    return ladder_round(values)


def to_uint8(values: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    rounded = from_fixed(values.astype(np.int64), frac_bits)
    return np.clip(np.round(rounded), 0, 255).astype(np.uint8)


def to_methodology_fixed(values: np.ndarray) -> np.ndarray:
    """Convert to S1.10 12-bit format for MATLAB comparison."""
    return saturate(round_to_frac(to_fixed(values), METH_FRAC_BITS), WORD_BITS)

"""
8-point DCT-II / IDCT-III.

Python: orthonormal matrix multiply (verified vs reference).
HLS (fpga/hls/): Loeffler flowgraph, 11 mul / 29 add.
"""

from __future__ import annotations

import numpy as np

from fixed_point import FRAC_BITS, from_fixed, to_fixed
from dct.reference import BLOCK_SIZE, build_dct_matrix, build_idct_matrix, c2d_batch, d2d_batch

_C = build_dct_matrix(BLOCK_SIZE)
_D = build_idct_matrix(BLOCK_SIZE)


def _as_float(x: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return from_fixed(x, frac_bits)
    return np.asarray(x, dtype=np.float64)


def loeffler_dct_8_float(f: np.ndarray) -> np.ndarray:
    return _C @ np.asarray(f, dtype=np.float64)


def loeffler_idct_8_float(f: np.ndarray) -> np.ndarray:
    return _D @ np.asarray(f, dtype=np.float64)


def loeffler_dct_8_fixed(x: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    return to_fixed(_C @ _as_float(x, frac_bits), frac_bits)


def loeffler_idct_8_fixed(x: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    return to_fixed(_D @ _as_float(x, frac_bits), frac_bits)


def dct_2d_loeffler(block: np.ndarray, fixed: bool = True) -> np.ndarray:
    b = _as_float(block) if fixed and np.issubdtype(block.dtype, np.integer) else np.asarray(block, dtype=np.float64)
    coeff = _C @ b @ _C.T
    return to_fixed(coeff) if fixed else coeff


def idct_2d_loeffler(block: np.ndarray, fixed: bool = True) -> np.ndarray:
    b = _as_float(block) if fixed else np.asarray(block, dtype=np.float64)
    restored = _D @ b @ _D.T
    return to_fixed(restored) if fixed else restored


def dct_2d_loeffler_batch(blocks: np.ndarray, fixed: bool = True) -> np.ndarray:
    if fixed and np.issubdtype(blocks.dtype, np.integer):
        b = from_fixed(blocks)
    else:
        b = np.asarray(blocks, dtype=np.float64)
    coeff = c2d_batch(b)
    return to_fixed(coeff) if fixed else coeff


def idct_2d_loeffler_batch(blocks: np.ndarray, fixed: bool = True) -> np.ndarray:
    if fixed and np.issubdtype(blocks.dtype, np.integer):
        b = from_fixed(blocks)
    else:
        b = np.asarray(blocks, dtype=np.float64)
    restored = d2d_batch(b)
    return to_fixed(restored) if fixed else restored

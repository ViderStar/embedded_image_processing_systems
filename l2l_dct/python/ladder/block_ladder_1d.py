"""1-D block ladder DCT-ODCT structural parameterization (eq. 5.24–5.27)."""

from __future__ import annotations

import numpy as np

from fixed_point import from_fixed, round_block, to_fixed, to_uint8
from dct.loeffler_8pt import loeffler_dct_8_fixed, loeffler_idct_8_fixed
from dct.reference import BLOCK_SIZE, build_dct_matrix, build_idct_matrix


def _matvec(matrix: np.ndarray, vec: np.ndarray, fixed: bool) -> np.ndarray:
    out = matrix @ np.asarray(vec, dtype=np.float64)
    if fixed:
        return round_block(to_fixed(out))
    return out


def ladder_forward_1d(
    x0: np.ndarray,
    x1: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
    fixed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x0)
    c = build_dct_matrix(n)
    d = build_idct_matrix(n)
    s = np.zeros(n, dtype=np.int64) if side is None else side.astype(np.int64)

    x0f = to_fixed(x0.astype(np.float64)) if fixed else x0.astype(np.float64)
    x1f = to_fixed(x1.astype(np.float64)) if fixed else x1.astype(np.float64)

    if variant == 1:
        y0 = _matvec(c, x0f, fixed)
        y1 = _matvec(d, x1f, fixed) + s
        if fixed:
            y1 = round_block(y1)
    else:
        y0 = _matvec(c, x0f, fixed) + s
        if fixed:
            y0 = round_block(y0)
        y1 = _matvec(d, x1f, fixed)

    return y0, y1


def ladder_inverse_1d(
    y0: np.ndarray,
    y1: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
    fixed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y0)
    c = build_dct_matrix(n)
    d = build_idct_matrix(n)
    s = np.zeros(n, dtype=np.int64) if side is None else side.astype(np.int64)

    if variant == 1:
        x0 = _matvec(d, y0, fixed)
        x1 = _matvec(c, from_fixed(y1 - s) if fixed else y1 - s, fixed)
    else:
        x0 = _matvec(d, from_fixed(y0 - s) if fixed else y0 - s, fixed)
        x1 = _matvec(c, y1, fixed)

    if fixed:
        return to_uint8(round_block(x0)), to_uint8(round_block(x1))
    return x0, x1


def ladder_forward_1d_loeffler(
    x0: np.ndarray,
    x1: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x0) != BLOCK_SIZE:
        return ladder_forward_1d(x0, x1, side, variant, fixed=True)

    s = np.zeros(BLOCK_SIZE, dtype=np.int64) if side is None else side.astype(np.int64)
    y0 = loeffler_dct_8_fixed(to_fixed(x0.astype(np.float64)))
    y1 = loeffler_dct_8_fixed(to_fixed(x1.astype(np.float64)))
    if variant == 1:
        y1 = round_block(y1 + s)
    else:
        y0 = round_block(y0 + s)
    return y0, y1


def ladder_inverse_1d_loeffler(
    y0: np.ndarray,
    y1: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y0) != BLOCK_SIZE:
        return ladder_inverse_1d(y0, y1, side, variant, fixed=True)

    s = np.zeros(BLOCK_SIZE, dtype=np.int64) if side is None else side.astype(np.int64)
    if variant == 1:
        x0 = loeffler_idct_8_fixed(y0)
        x1 = loeffler_idct_8_fixed(round_block(y1 - s))
    else:
        x0 = loeffler_idct_8_fixed(round_block(y0 - s))
        x1 = loeffler_idct_8_fixed(y1)
    return x0, x1

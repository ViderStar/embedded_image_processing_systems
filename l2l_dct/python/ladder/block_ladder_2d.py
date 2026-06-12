"""2-D block ladder DCT-ODCT (eq. 5.32–5.36)."""

from __future__ import annotations

import numpy as np

from fixed_point import round_block, to_fixed
from dct.loeffler_8pt import dct_2d_loeffler, idct_2d_loeffler
from dct.reference import dct_2d, idct_2d, BLOCK_SIZE
from ladder.block_ladder_1d import ladder_forward_1d, ladder_forward_1d_loeffler, ladder_inverse_1d, ladder_inverse_1d_loeffler


def split_block_n2n(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = block.shape[1] // 2
    return block[:, :n].copy(), block[:, n:].copy()


def merge_block_n2n(xi: np.ndarray, si: np.ndarray) -> np.ndarray:
    return np.hstack([xi, si])


def ladder_forward_2d(
    xi: np.ndarray,
    si: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
    use_loeffler: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = xi.shape[0]
    yi = np.zeros_like(xi, dtype=np.int64)
    ti = np.zeros_like(si, dtype=np.int64)
    fwd = ladder_forward_1d_loeffler if use_loeffler and n == BLOCK_SIZE else ladder_forward_1d

    for r in range(n):
        y0, y1 = fwd(xi[r], si[r], side[r] if side is not None else None, variant)
        yi[r] = y0
        ti[r] = y1

    yi_t = yi.T
    ti_t = ti.T
    yo = np.zeros_like(yi_t, dtype=np.int64)
    to = np.zeros_like(ti_t, dtype=np.int64)

    for c in range(n):
        y0, y1 = fwd(yi_t[c], ti_t[c], None, variant)
        yo[c] = y0
        to[c] = y1

    return yo.T, to.T


def ladder_inverse_2d(
    yi: np.ndarray,
    ti: np.ndarray,
    side: np.ndarray | None = None,
    variant: int = 1,
    use_loeffler: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = yi.shape[0]
    inv = ladder_inverse_1d_loeffler if use_loeffler and n == BLOCK_SIZE else ladder_inverse_1d

    xi_cols = np.zeros_like(yi, dtype=np.int64)
    si_cols = np.zeros_like(ti, dtype=np.int64)
    yi_t, ti_t = yi.T, ti.T

    for c in range(n):
        x0, x1 = inv(yi_t[c], ti_t[c], side[c] if side is not None else None, variant)
        xi_cols[c] = x0
        si_cols[c] = x1

    xi, si = xi_cols.T, si_cols.T
    xo = np.zeros_like(xi, dtype=np.int64)
    so = np.zeros_like(si, dtype=np.int64)

    for r in range(n):
        x0, x1 = inv(xi[r], si[r], side[r] if side is not None else None, variant)
        xo[r] = x0
        so[r] = x1

    return xo, so


def matrix_forward_2d(block: np.ndarray, fixed: bool = True) -> np.ndarray:
    xi, si = split_block_n2n(block.astype(np.float64))
    if fixed:
        yo = dct_2d_loeffler(to_fixed(xi), fixed=True)
        to = dct_2d_loeffler(to_fixed(si), fixed=True)
        return merge_block_n2n(yo, to)
    return merge_block_n2n(dct_2d(xi), dct_2d(si))


def matrix_inverse_2d(coeffs: np.ndarray, fixed: bool = True) -> np.ndarray:
    yi, ti = split_block_n2n(coeffs)
    if fixed:
        return merge_block_n2n(idct_2d_loeffler(yi, True), idct_2d_loeffler(ti, True))
    return merge_block_n2n(idct_2d(yi), idct_2d(ti))

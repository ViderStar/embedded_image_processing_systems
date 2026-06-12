"""Side Information Block (SIB) for perfect reconstruction (eq. 5.29–5.30)."""

from __future__ import annotations

import numpy as np

from fixed_point import round_block, to_fixed
from dct.reference import BLOCK_SIZE, build_idct_matrix


def sib_iterate(s_prev: np.ndarray, n: int = BLOCK_SIZE) -> np.ndarray:
    """One SIB iteration: s_i = round(D * s_{i-1})."""
    d = build_idct_matrix(n)
    return round_block(to_fixed(d @ s_prev.astype(np.float64)))


def build_sib_chain(num_steps: int, n: int = BLOCK_SIZE) -> list[np.ndarray]:
    chain = [np.zeros(n, dtype=np.int64)]
    for _ in range(num_steps):
        chain.append(sib_iterate(chain[-1], n))
    return chain

"""Unit tests for DCT/ODCT."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from dct.loeffler_8pt import loeffler_dct_8_float, loeffler_idct_8_float, dct_2d_loeffler, idct_2d_loeffler
from dct.reference import dct_1d, idct_1d, dct_2d, idct_2d
from fixed_point import to_fixed, from_fixed


def test_loeffler_matches_matrix():
    x = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float64)
    assert np.max(np.abs(dct_1d(x) - loeffler_dct_8_float(x))) < 1e-10


def test_idct_roundtrip():
    x = np.linspace(0, 255, 8)
    y = loeffler_dct_8_float(x)
    back = loeffler_idct_8_float(y)
    assert np.max(np.abs(x - back)) < 1e-10


def test_fixed_2d_roundtrip():
    block = np.random.randint(0, 256, (8, 8)).astype(np.float64)
    coeff = dct_2d_loeffler(to_fixed(block), fixed=True)
    restored = idct_2d_loeffler(coeff, fixed=True)
    err = np.max(np.abs(block - from_fixed(restored)))
    assert err < 1.0


if __name__ == "__main__":
    test_loeffler_matches_matrix()
    test_idct_roundtrip()
    test_fixed_2d_roundtrip()
    print("All tests passed.")

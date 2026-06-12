"""Integration tests for ladder blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from ladder.block_ladder_2d import ladder_forward_2d, ladder_inverse_2d


def test_ladder_block_loeffler():
    xi = np.random.randint(0, 256, (8, 8)).astype(np.float64)
    si = np.random.randint(0, 256, (8, 8)).astype(np.float64)
    yi, ti = ladder_forward_2d(xi, si, use_loeffler=True)
    xo, so = ladder_inverse_2d(yi, ti, use_loeffler=True)
    assert xo.shape == (8, 8) and so.shape == (8, 8)


if __name__ == "__main__":
    test_ladder_block_loeffler()
    print("Ladder test passed.")

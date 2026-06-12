"""Integration tests for SIB and chessboard demo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from l2l.synthesis import analysis_synthesis_roundtrip
from metrics import mse, psnr


def test_sib_perfect_reconstruction():
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    recon = analysis_synthesis_roundtrip(img, use_sib=True)
    assert mse(img, recon) == 0.0
    assert psnr(img, recon) == float("inf")


def test_no_sib_has_error():
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    recon = analysis_synthesis_roundtrip(img, use_sib=False)
    assert mse(img, recon) > 0.0


if __name__ == "__main__":
    test_sib_perfect_reconstruction()
    test_no_sib_has_error()
    print("SIB tests passed.")

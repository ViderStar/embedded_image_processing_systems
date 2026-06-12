"""
Pynq driver for L2L DCT HLS overlay.

Usage on Pynq board:
    from fpga.pynq.l2l_driver import L2LDctOverlay
    ov = L2LDctOverlay('l2l_dct.bit')
    result = ov.dct_block(pixel_block_8x8)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from pynq import Overlay, allocate
except ImportError:
    Overlay = None
    allocate = None


class L2LDctOverlay:
    """Driver for 8-point Loeffler DCT IP on Pynq."""

    def __init__(self, bitfile: str | Path) -> None:
        if Overlay is None:
            raise RuntimeError("pynq package not available (run on Pynq board)")
        self.overlay = Overlay(str(bitfile))
        self.ip = self.overlay.loeffler_dct_8pt_0

    def dct_1d(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) != 8:
            raise ValueError("Expected 8 samples")
        buf_in = allocate(shape=(8,), dtype=np.float32)
        buf_out = allocate(shape=(8,), dtype=np.float32)
        buf_in[:] = samples.astype(np.float32)
        self.ip.call(buf_in, buf_out)
        return np.array(buf_out)

    def dct_block_rows(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        out = np.zeros_like(block)
        for r in range(8):
            out[r] = self.dct_1d(block[r])
        return out

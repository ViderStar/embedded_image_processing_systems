"""Quantization for L2L lossy mode."""

from __future__ import annotations

import numpy as np

JPEG_LUMA_STEPS = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int64)


def uniform_quantize(values: np.ndarray, step: int) -> np.ndarray:
    step = max(step, 1)
    return (values // step) * step


def per_band_quantize(block: np.ndarray, scale: float = 1.0) -> np.ndarray:
    steps = np.maximum((JPEG_LUMA_STEPS * scale).astype(np.int64), 1)
    return (block // steps) * steps


def quantize_image_blocks(image: np.ndarray, block_size: int = 8, scale: float = 1.0) -> np.ndarray:
    """Apply per-band JPEG quantizer to all blocks in a coefficient image."""
    h, w = image.shape
    out = image.copy()
    steps = np.maximum((JPEG_LUMA_STEPS * scale).astype(np.int64), 1)
    for bi in range(h // block_size):
        for bj in range(w // block_size):
            r0, r1 = bi * block_size, (bi + 1) * block_size
            c0, c1 = bj * block_size, (bj + 1) * block_size
            out[r0:r1, c0:c1] = (out[r0:r1, c0:c1] // steps) * steps
    return out

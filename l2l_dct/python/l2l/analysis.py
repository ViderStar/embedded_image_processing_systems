"""L2L analysis — parallel C2D / D2D with SIB on both paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fixed_point import round_block, to_fixed
from dct.reference import BLOCK_SIZE, c2d_batch, d2d_batch


@dataclass
class AnalysisResult:
    yi: np.ndarray
    ti: np.ndarray
    side_y: np.ndarray | None
    side_t: np.ndarray | None
    xi_shape: tuple[int, int]
    block_size: int

    @property
    def side_info(self) -> np.ndarray | None:
        if self.side_y is None or self.side_t is None:
            return None
        return np.stack([self.side_y, self.side_t], axis=0)


def split_image_l2l(image: np.ndarray, block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, np.ndarray]:
    half = image.shape[0] // 2
    return image[:half].copy(), image[half:].copy()


def merge_image_l2l(xi: np.ndarray, si: np.ndarray) -> np.ndarray:
    return np.vstack([xi, si])


def _blocks_from_image(part: np.ndarray, block_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = part.shape
    nh, nw = h // block_size, w // block_size
    blocks = (
        part.reshape(nh, block_size, nw, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
        .astype(np.float64)
    )
    return blocks, (nh, nw, h, w)


def _blocks_to_image(blocks: np.ndarray, layout: tuple[int, int, int, int], block_size: int) -> np.ndarray:
    nh, nw, h, w = layout
    return (
        blocks.reshape(nh, nw, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
    )


def _forward_blocks(
    xi_blocks: np.ndarray,
    si_blocks: np.ndarray,
    use_sib: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    yi_exact = to_fixed(c2d_batch(xi_blocks))
    ti_exact = to_fixed(d2d_batch(si_blocks))
    yi = round_block(yi_exact)
    ti = round_block(ti_exact)

    if not use_sib:
        return yi, ti, None, None

    return yi, ti, (yi_exact - yi).astype(np.int64), (ti_exact - ti).astype(np.int64)


def analyze_image(
    image: np.ndarray,
    block_size: int = BLOCK_SIZE,
    use_sib: bool = False,
) -> AnalysisResult:
    xi_img, si_img = split_image_l2l(image, block_size)
    xi_blocks, layout = _blocks_from_image(xi_img, block_size)
    si_blocks, _ = _blocks_from_image(si_img, block_size)

    yi_b, ti_b, sy_b, st_b = _forward_blocks(xi_blocks, si_blocks, use_sib)
    h, w = xi_img.shape

    yi_full = _blocks_to_image(yi_b, layout, block_size)
    ti_full = _blocks_to_image(ti_b, layout, block_size)
    sy = _blocks_to_image(sy_b, layout, block_size) if sy_b is not None else None
    st = _blocks_to_image(st_b, layout, block_size) if st_b is not None else None

    return AnalysisResult(yi_full, ti_full, sy, st, (h, w), block_size)

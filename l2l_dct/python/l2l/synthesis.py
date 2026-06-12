"""L2L synthesis — inverse D2D / C2D paths."""

from __future__ import annotations

import numpy as np

from fixed_point import from_fixed, round_block, to_fixed
from dct.reference import BLOCK_SIZE, c2d_batch, d2d_batch
from l2l.analysis import AnalysisResult, _blocks_from_image, _blocks_to_image, merge_image_l2l


def _inverse_blocks(
    yi_blocks: np.ndarray,
    ti_blocks: np.ndarray,
    side_y: np.ndarray | None,
    side_t: np.ndarray | None,
    use_sib: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_sib and side_y is not None and side_t is not None:
        yi_fl = from_fixed(yi_blocks + side_y)
        ti_fl = from_fixed(ti_blocks + side_t)
        return d2d_batch(yi_fl), c2d_batch(ti_fl)

    yi_fl = from_fixed(yi_blocks)
    ti_fl = from_fixed(ti_blocks)
    xi = from_fixed(round_block(to_fixed(d2d_batch(yi_fl))))
    si = from_fixed(round_block(to_fixed(c2d_batch(ti_fl))))
    return xi, si


def synthesize_image(result: AnalysisResult, use_sib: bool = False) -> np.ndarray:
    bs = result.block_size
    yi_blocks, layout = _blocks_from_image(result.yi.astype(np.float64), bs)
    ti_blocks, _ = _blocks_from_image(result.ti.astype(np.float64), bs)

    sy_b = st_b = None
    if use_sib and result.side_y is not None and result.side_t is not None:
        sy_b, _ = _blocks_from_image(result.side_y.astype(np.float64), bs)
        st_b, _ = _blocks_from_image(result.side_t.astype(np.float64), bs)

    xi_out, si_out = _inverse_blocks(yi_blocks, ti_blocks, sy_b, st_b, use_sib)
    xi_img = _blocks_to_image(xi_out, layout, bs)
    si_img = _blocks_to_image(si_out, layout, bs)

    merged = merge_image_l2l(xi_img, si_img)
    return np.clip(np.round(merged), 0, 255).astype(np.uint8)


def analysis_synthesis_roundtrip(image: np.ndarray, use_sib: bool = False) -> np.ndarray:
    from l2l.analysis import analyze_image

    return synthesize_image(analyze_image(image, use_sib=use_sib), use_sib=use_sib)

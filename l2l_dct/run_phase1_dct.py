#!/usr/bin/env python3
"""Phase 1: DCT/ODCT fixed-point via Loeffler algorithm."""

from pathlib import Path

import numpy as np
from PIL import Image

from common import ASSETS_DIR, ensure_output, load_grayscale
from dct.loeffler_8pt import (
    dct_2d_loeffler_batch,
    idct_2d_loeffler_batch,
    loeffler_dct_8_fixed,
    loeffler_dct_8_float,
    loeffler_idct_8_fixed,
)
from dct.reference import c2d_batch, d2d_batch, dct_1d
from fixed_point import from_fixed, to_fixed
from l2l.analysis import _blocks_from_image, _blocks_to_image
from metrics import mse, psnr
from viz import save_dct_basis

BLOCK_SIZE = 8


def demo_1d() -> None:
    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, 8).astype(np.float64)
    print(f"1-D Loeffler vs matrix (float): max err = {np.max(np.abs(dct_1d(x) - loeffler_dct_8_float(x))):.2e}")

    xf = to_fixed(x)
    x_back = from_fixed(loeffler_idct_8_fixed(loeffler_dct_8_fixed(xf)))
    print(f"1-D fixed round-trip MSE: {mse(x, x_back):.6f}")


def demo_2d_image(image: np.ndarray, out_dir: Path) -> None:
    blocks, layout = _blocks_from_image(image.astype(np.float64), BLOCK_SIZE)
    coeff = dct_2d_loeffler_batch(to_fixed(blocks), fixed=True)
    restored = idct_2d_loeffler_batch(coeff, fixed=True)
    recon = np.clip(from_fixed(restored), 0, 255).astype(np.uint8)
    recon_img = _blocks_to_image(recon, layout, BLOCK_SIZE)

    ref_coeff = c2d_batch(blocks)
    ref_img = np.clip(d2d_batch(ref_coeff), 0, 255).astype(np.uint8)
    ref_recon = _blocks_to_image(ref_img, layout, BLOCK_SIZE)

    print(f"2-D Loeffler fixed PSNR: {psnr(image, recon_img):.2f} dB")
    print(f"2-D matrix float PSNR:   {psnr(image, ref_recon):.2f} dB")

    Image.fromarray(recon_img).save(out_dir / "phase1_loeffler_recon.png")
    save_dct_basis(out_dir / "phase1_dct_basis.png")


def main() -> None:
    out = ensure_output()
    img_path = ASSETS_DIR / "test_image.png"
    image = load_grayscale(img_path)
    print("=== Phase 1: DCT/ODCT (Loeffler, fixed-point) ===")
    print(f"Image: {img_path.name}, size {image.shape}")
    demo_1d()
    demo_2d_image(image, out)
    print(f"Results saved to {out}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3: L2L lossless codec."""

from common import ASSETS_DIR, ensure_output, load_grayscale
from l2l.codec import decode_lossless, encode_lossless, estimate_bpp
from metrics import mse, psnr
from viz import save_comparison


def main() -> None:
    out = ensure_output()
    image = load_grayscale(ASSETS_DIR / "test_image.png")
    print("=== Phase 3: L2L lossless ===")

    enc = encode_lossless(image)
    dec = decode_lossless(enc)
    print(f"MSE:  {mse(image, dec):.6e}")
    print(f"PSNR: {psnr(image, dec):.2f} dB")
    print(f"Estimated bpp: {estimate_bpp(enc):.4f}")

    save_comparison(image, dec, out / "phase3_lossless.png", "L2L lossless")
    print(f"Saved {out / 'phase3_lossless.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 4: L2L lossy codec and rate-distortion."""

from common import ASSETS_DIR, ensure_output, load_grayscale
from l2l.codec import decode_lossy, encode_lossy, rate_distortion_point
from metrics import psnr
from viz import save_comparison, save_rate_distortion


def main() -> None:
    out = ensure_output()
    image = load_grayscale(ASSETS_DIR / "test_image.png")
    print("=== Phase 4: L2L lossy ===")

    steps = [1, 2, 4, 8, 16, 32]
    rd = rate_distortion_point(image, steps)
    print("step | bpp   | PSNR")
    for step, bpp, p in rd:
        print(f"  {step:2d} | {bpp:.3f} | {p:.2f} dB")

    save_rate_distortion(rd, out / "phase4_rate_distortion.png")

    enc = encode_lossy(image, quant_step=8)
    dec = decode_lossy(enc)
    print(f"\nDemo quant_step=8: PSNR = {psnr(image, dec):.2f} dB")
    save_comparison(image, dec, out / "phase4_lossy_q8.png", "L2L lossy q=8")
    print(f"Figures saved to {out}/")


if __name__ == "__main__":
    main()

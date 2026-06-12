#!/usr/bin/env python3
"""Phase 2: Chessboard artifact demo (with/without SIB)."""

from common import ASSETS_DIR, ensure_output, load_grayscale
from l2l.analysis import analyze_image
from l2l.synthesis import analysis_synthesis_roundtrip, synthesize_image
from metrics import mse, psnr
from viz import save_chessboard_demo, save_comparison


def main() -> None:
    out = ensure_output()
    image = load_grayscale(ASSETS_DIR / "test_image.png")
    print("=== Phase 2: Chessboard effect (DCT-ODCT ladder) ===")
    print(f"Image size: {image.shape}")

    recon_no_sib = analysis_synthesis_roundtrip(image, use_sib=False)
    m_no = mse(image, recon_no_sib)
    p_no = psnr(image, recon_no_sib)
    print(f"Without SIB: MSE = {m_no:.6e}, PSNR = {p_no:.2f} dB")

    recon_with_sib = analysis_synthesis_roundtrip(image, use_sib=True)
    m_yes = mse(image, recon_with_sib)
    p_yes = psnr(image, recon_with_sib)
    print(f"With SIB:    MSE = {m_yes:.6e}, PSNR = {p_yes:.2f} dB")

    save_chessboard_demo(image, recon_no_sib, recon_with_sib, out / "phase2_chessboard.png")
    save_comparison(image, recon_no_sib, out / "phase2_no_sib_detail.png", "Без SIB")
    save_comparison(image, recon_with_sib, out / "phase2_with_sib_detail.png", "С SIB")
    print(f"Figures saved to {out}/")


if __name__ == "__main__":
    main()

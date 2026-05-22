#!/usr/bin/env python3
"""Lab 2: Non-separable 2-D wavelet transform based on Lab 1 CDF 9/7."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LAB_DIR = Path(__file__).resolve().parent
LAB1_DIR = LAB_DIR.parent / "lab1"
sys.path.insert(0, str(LAB1_DIR))

from cdf97_lifting import from_fixed, mse, psnr
from non_separable_2d import (
    nonseparable_forward_2d_block,
    nonseparable_inverse_2d_block,
    separable_forward_2d,
    separable_inverse_2d,
    verify_nonseparable_equals_separable,
)

IMAGE_PATH = LAB1_DIR / "test_image_gray.png"
OUTPUT_DIR = LAB_DIR / "output"


def load_grayscale(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape[0] % 2:
        arr = arr[:-1]
    if arr.shape[1] % 2:
        arr = arr[:, :-1]
    return arr


def visualize_2d_subbands(ll, lh, hl, hh, title_prefix: str, out_path: Path) -> None:
    ll_vis = from_fixed(ll)
    bands = [ll_vis]
    for band in (lh, hl, hh):
        b = from_fixed(band)
        b = b.astype(np.float64) - b.mean()
        bands.append(b)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    titles = ["LL (аппроксимация)", "LH (горизонтальные детали)",
              "HL (вертикальные детали)", "HH (диагональные детали)"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for band, title, (r, c) in zip(bands, titles, positions):
        if r == 0 and c == 0:
            axes[r, c].imshow(band, cmap="gray")
        else:
            vmax = max(abs(band.min()), abs(band.max()), 1)
            axes[r, c].imshow(band, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[r, c].set_title(f"{title_prefix}: {title}")
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def demo_block_nonseparable() -> None:
    """Demonstrate non-separable transform on 8×8 blocks."""
    image = load_grayscale(IMAGE_PATH)
    n = 8
    h_blocks = image.shape[0] // n
    w_blocks = image.shape[1] // n

    reconstructed = np.zeros_like(image, dtype=np.float64)
    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = image[bi * n : (bi + 1) * n, bj * n : (bj + 1) * n].astype(np.float64)
            coeffs = nonseparable_forward_2d_block(block)
            reconstructed[bi * n : (bi + 1) * n, bj * n : (bj + 1) * n] = (
                nonseparable_inverse_2d_block(coeffs)
            )

    recon_u8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
    sko = mse(image[: h_blocks * n, : w_blocks * n], recon_u8)
    pssr = psnr(image[: h_blocks * n, : w_blocks * n], recon_u8)
    print(f"\nБлочное неразделимое 2-D ({n}×{n}):")
    print(f"  СКО:  {sko:.6f}")
    print(f"  ПССШ: {pssr:.2f} dB")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    matrix_err = verify_nonseparable_equals_separable(8)
    print("=== Lab 2: Неразделимое 2-D преобразование ===")
    print(f"Проверка эквивалентности Θ̈ = D(Θ)·P·D(Θ)·P и Θ⊗Θ: относительная ошибка = {matrix_err:.2e}")

    image = load_grayscale(IMAGE_PATH)
    print(f"Изображение: {IMAGE_PATH.name}, размер {image.shape}")

    ll, lh, hl, hh = separable_forward_2d(image)
    reconstructed = separable_inverse_2d(ll, lh, hl, hh, image.shape[0], image.shape[1])

    sko = mse(image, reconstructed)
    pssr = psnr(image, reconstructed)
    print(f"\n2-D разложение (разделимая реализация, 1 уровень):")
    print(f"  СКО:  {sko:.6f}")
    print(f"  ПССШ: {pssr:.2f} dB")

    visualize_2d_subbands(ll, lh, hl, hh, "2-D DWT", OUTPUT_DIR / "lab2_subbands.png")
    Image.fromarray(reconstructed).save(OUTPUT_DIR / "lab2_reconstructed.png")

    demo_block_nonseparable()
    print(f"\nРезультаты сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

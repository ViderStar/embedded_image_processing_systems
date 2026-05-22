#!/usr/bin/env python3
"""Lab 1: CDF 9/7 fixed-point 1-D wavelet transform on a grayscale image."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cdf97_lifting import (
    ALPHA,
    BETA,
    DELTA,
    FRAC_BITS,
    GAMMA,
    INV_K,
    K,
    cdf97_forward_rows,
    cdf97_inverse_rows,
    from_fixed,
    mse,
    psnr,
)

LAB_DIR = Path(__file__).resolve().parent
IMAGE_PATH = LAB_DIR / "test_image_gray.png"
OUTPUT_DIR = LAB_DIR / "output"


def load_grayscale(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape[1] % 2 == 1:
        arr = arr[:, :-1]
    return arr


def visualize_subbands(original: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    """Visualize 1-D row-wise decomposition (approximation + horizontal detail)."""
    low_vis = from_fixed(low)
    high_vis = from_fixed(high)
    high_vis = high_vis - high_vis.mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")

    axes[1].imshow(low_vis, cmap="gray")
    axes[1].set_title("Низкочастотная подполоса (L)")
    axes[1].axis("off")

    vmax = max(abs(high_vis.min()), abs(high_vis.max()), 1)
    axes[2].imshow(high_vis, cmap="gray", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Высокочастотная подполоса (H)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lab1_subbands.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    image = load_grayscale(IMAGE_PATH)
    print(f"Изображение: {IMAGE_PATH.name}, размер {image.shape}")

    low, high = cdf97_forward_rows(image)
    reconstructed = cdf97_inverse_rows(low, high, image.shape[1])

    sko = mse(image, reconstructed)
    pssr = psnr(image, reconstructed)

    print("\n=== Результаты Lab 1 ===")
    print(f"Формат фиксированной запятой: Q{FRAC_BITS} ({FRAC_BITS} дробных бит)")
    print("Рациональные коэффициенты лестничной схемы CDF 9/7:")
    print(f"  α = {ALPHA[0]}/{ALPHA[1]} = {ALPHA[0]/ALPHA[1]:.4f}")
    print(f"  β = {BETA[0]}/{BETA[1]} = {BETA[0]/BETA[1]:.4f}")
    print(f"  γ = {GAMMA[0]}/{GAMMA[1]} = {GAMMA[0]/GAMMA[1]:.4f}")
    print(f"  δ = {DELTA[0]}/{DELTA[1]} = {DELTA[0]/DELTA[1]:.4f}")
    print(f"  k = {K[0]}/{K[1]},  1/k = {INV_K[0]}/{INV_K[1]}")
    print(f"\nСКО (MSE):  {sko:.6f}")
    print(f"ПССШ (PSNR): {pssr:.2f} dB")

    Image.fromarray(reconstructed).save(OUTPUT_DIR / "lab1_reconstructed.png")
    visualize_subbands(image, low, high)
    print(f"\nРезультаты сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

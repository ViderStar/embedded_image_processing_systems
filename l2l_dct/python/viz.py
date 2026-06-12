"""Visualization helpers for L2L demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    diff = np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))
    diff_vis = np.clip(diff * 8, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Исходное")
    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title("Восстановленное")
    axes[2].imshow(diff_vis, cmap="hot")
    axes[2].set_title("Разность ×8")
    if title:
        fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_chessboard_demo(
    original: np.ndarray,
    no_sib: np.ndarray,
    with_sib: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("а — исходное")
    axes[1].imshow(no_sib, cmap="gray")
    axes[1].set_title("б — без SIB (шахматная доска)")
    axes[2].imshow(with_sib, cmap="gray")
    axes[2].set_title("в — с SIB (без потерь)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_rate_distortion(
    points: list[tuple[int, float, float]],
    out_path: Path,
) -> None:
    bpps = [p[1] for p in points]
    psnrs = [p[2] for p in points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bpps, psnrs, "o-")
    ax.set_xlabel("bpp (оценка)")
    ax.set_ylabel("PSNR, dB")
    ax.set_title("Rate–distortion (L2L lossy)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_dct_basis(out_path: Path, n: int = 8) -> None:
    from dct.reference import build_dct_matrix

    c = build_dct_matrix(n)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for k, ax in enumerate(axes.ravel()):
        ax.plot(c[k], "o-")
        ax.set_title(f"Базис {k}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

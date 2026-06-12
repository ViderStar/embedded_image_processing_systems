#!/usr/bin/env python3
"""Shared utilities for L2L run scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = PROJECT_DIR / "python"
ASSETS_DIR = PROJECT_DIR / "assets"
OUTPUT_DIR = PROJECT_DIR / "output"

sys.path.insert(0, str(PYTHON_DIR))


def load_grayscale(path: Path, size: tuple[int, int] | None = (512, 512)) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    if h % 2:
        arr = arr[:-1]
    if w % 16:
        arr = arr[:, : (w // 16) * 16]
    if h % 16:
        arr = arr[: (h // 16) * 16]
    return arr


def ensure_output() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

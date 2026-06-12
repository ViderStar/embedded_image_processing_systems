"""L2L codec: lossless and lossy pipelines."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from l2l.analysis import AnalysisResult, analyze_image
from l2l.quantizer import quantize_image_blocks
from l2l.synthesis import synthesize_image


@dataclass
class EncodedImage:
    yi_bytes: bytes
    ti_bytes: bytes
    side_y_bytes: bytes | None
    side_t_bytes: bytes | None
    shape: tuple[int, int]
    block_size: int
    lossless: bool
    quant_step: int


def _pack_int64_array(arr: np.ndarray) -> bytes:
    flat = arr.astype(np.int64).ravel()
    header = struct.pack("III", arr.shape[0], arr.shape[1], len(flat))
    return header + flat.tobytes()


def _unpack_int64_array(data: bytes) -> np.ndarray:
    h, w, n = struct.unpack("III", data[:12])
    flat = np.frombuffer(data[12:], dtype=np.int64, count=n)
    return flat.reshape(h, w)


def encode_lossless(image: np.ndarray, block_size: int = 8) -> EncodedImage:
    result = analyze_image(image, block_size, use_sib=True)
    return EncodedImage(
        yi_bytes=zlib.compress(_pack_int64_array(result.yi)),
        ti_bytes=zlib.compress(_pack_int64_array(result.ti)),
        side_y_bytes=zlib.compress(_pack_int64_array(result.side_y)),
        side_t_bytes=zlib.compress(_pack_int64_array(result.side_t)),
        shape=image.shape,
        block_size=block_size,
        lossless=True,
        quant_step=0,
    )


def decode_lossless(encoded: EncodedImage) -> np.ndarray:
    yi = _unpack_int64_array(zlib.decompress(encoded.yi_bytes))
    result = AnalysisResult(
        yi=yi,
        ti=_unpack_int64_array(zlib.decompress(encoded.ti_bytes)),
        side_y=_unpack_int64_array(zlib.decompress(encoded.side_y_bytes)),
        side_t=_unpack_int64_array(zlib.decompress(encoded.side_t_bytes)),
        xi_shape=yi.shape,
        block_size=encoded.block_size,
    )
    return synthesize_image(result, use_sib=True)


def encode_lossy(image: np.ndarray, quant_step: int = 4, block_size: int = 8) -> EncodedImage:
    result = analyze_image(image, block_size, use_sib=False)
    yi_q = quantize_image_blocks(result.yi, block_size, quant_step)
    ti_q = quantize_image_blocks(result.ti, block_size, quant_step)
    return EncodedImage(
        yi_bytes=zlib.compress(_pack_int64_array(yi_q)),
        ti_bytes=zlib.compress(_pack_int64_array(ti_q)),
        side_y_bytes=None,
        side_t_bytes=None,
        shape=image.shape,
        block_size=block_size,
        lossless=False,
        quant_step=quant_step,
    )


def decode_lossy(encoded: EncodedImage) -> np.ndarray:
    yi = _unpack_int64_array(zlib.decompress(encoded.yi_bytes))
    result = AnalysisResult(
        yi=yi,
        ti=_unpack_int64_array(zlib.decompress(encoded.ti_bytes)),
        side_y=None,
        side_t=None,
        xi_shape=yi.shape,
        block_size=encoded.block_size,
    )
    return synthesize_image(result, use_sib=False)


def estimate_bpp(encoded: EncodedImage) -> float:
    total = len(encoded.yi_bytes) + len(encoded.ti_bytes)
    if encoded.side_y_bytes:
        total += len(encoded.side_y_bytes) + len(encoded.side_t_bytes)
    return total * 8 / (encoded.shape[0] * encoded.shape[1])


def rate_distortion_point(image: np.ndarray, steps: list[int]) -> list[tuple[int, float, float]]:
    from metrics import psnr

    return [
        (step, estimate_bpp(enc := encode_lossy(image, quant_step=step)), psnr(image, decode_lossy(enc)))
        for step in steps
    ]

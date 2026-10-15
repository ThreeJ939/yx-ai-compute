"""Sliding-window tiling helpers (aligned with yolo-world-demo)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def iter_tiles(
    width: int,
    height: int,
    tile: int = 640,
    overlap: float = 0.2,
) -> Iterator[tuple[int, int]]:
    """Yield top-left (x1, y1) for each tile covering the image."""
    if width <= 0 or height <= 0 or tile <= 0:
        return
    stride = max(1, int(tile * (1.0 - overlap)))
    max_x = max(width - tile, 0)
    max_y = max(height - tile, 0)
    xs = list(range(0, max_x + 1, stride))
    ys = list(range(0, max_y + 1, stride))
    if not xs or xs[-1] != max_x:
        xs.append(max_x)
    if not ys or ys[-1] != max_y:
        ys.append(max_y)
    for y1 in ys:
        for x1 in xs:
            yield x1, y1


def build_tile_crops(
    image_bgr: np.ndarray,
    tile: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, np.ndarray]]:
    """Build padded tile crops with origin offsets for full-image remapping.

    Returns list of ``(x1, y1, crop_bgr)`` where ``crop`` is always
    ``tile x tile`` (zero-padded at image edges).
    """
    h, w = image_bgr.shape[:2]
    out: list[tuple[int, int, np.ndarray]] = []
    for x1, y1 in iter_tiles(w, h, tile=tile, overlap=overlap):
        x2, y2 = x1 + tile, y1 + tile
        crop = np.zeros((tile, tile, 3), dtype=np.uint8)
        region = image_bgr[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)]
        crop[: region.shape[0], : region.shape[1]] = region
        out.append((x1, y1, crop))
    return out


def as_predict_result_list(results: Any, expected: int) -> list[Any]:
    """Normalize Ultralytics ``predict`` output to a list aligned with inputs.

    Raises ``ValueError`` if length does not match ``expected`` (batch alignment).
    """
    if results is None:
        raise ValueError("predict returned None")
    if not isinstance(results, (list, tuple)):
        results = [results]
    results = list(results)
    if len(results) != expected:
        raise ValueError(
            f"predict result count mismatch: got {len(results)}, expected {expected}"
        )
    return results

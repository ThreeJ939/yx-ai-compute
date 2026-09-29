"""Sliding-window tiling helpers (aligned with yolo-world-demo)."""

from __future__ import annotations

from collections.abc import Iterator


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

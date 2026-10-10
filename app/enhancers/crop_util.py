"""Crop helpers for LLM / VLM behavior enhancement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.detectors.base import RawDetection

FilterFn = Callable[[RawDetection], bool]


@dataclass
class CropItem:
    """One image crop linked back to source detection index(es)."""

    image_bgr: Any
    det_indices: list[int]
    bbox_xywh: list[float]


def filter_person(det: RawDetection) -> bool:
    text = f"{det.object_type} {det.label}".lower()
    return "person" in text or "people" in text or "human" in text


def filter_fire_smoke(det: RawDetection) -> bool:
    obj = (det.object_type or "").strip().lower()
    label = (det.label or "").strip().lower()
    return obj in {"fire", "smoke"} or label in {"fire", "smoke"}


# Backward-compatible alias
def _is_person(det: RawDetection) -> bool:
    return filter_person(det)


def filter_person_detections(
    detections: list[RawDetection],
) -> list[tuple[int, RawDetection]]:
    return [(i, d) for i, d in enumerate(detections) if filter_person(d)]


def filter_detections(
    detections: list[RawDetection],
    filter_fn: FilterFn,
) -> list[tuple[int, RawDetection]]:
    return [(i, d) for i, d in enumerate(detections) if filter_fn(d)]


def _clamp_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    x1 = max(0.0, min(float(img_w), x))
    y1 = max(0.0, min(float(img_h), y))
    x2 = max(0.0, min(float(img_w), x + w))
    y2 = max(0.0, min(float(img_h), y + h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _expand_xywh(bbox: list[float], padding: float, img_w: int, img_h: int) -> list[float]:
    x, y, w, h = bbox
    pad_x = w * padding
    pad_y = h * padding
    return _clamp_xywh(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, img_w, img_h)


def _crop_xywh(image_bgr: Any, bbox: list[float]) -> Any | None:
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return crop


def single_expand(
    image_bgr: Any,
    detections: list[RawDetection],
    filter_fn: FilterFn,
    padding: float = 0.3,
) -> list[CropItem]:
    """One crop per matching detection, expanded by padding ratio."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return []
    img_h, img_w = image_bgr.shape[:2]
    out: list[CropItem] = []
    for idx, det in filter_detections(detections, filter_fn):
        expanded = _expand_xywh(det.bbox, padding, img_w, img_h)
        crop = _crop_xywh(image_bgr, expanded)
        if crop is None:
            continue
        out.append(CropItem(image_bgr=crop, det_indices=[idx], bbox_xywh=expanded))
    return out


def multi_union(
    image_bgr: Any,
    detections: list[RawDetection],
    filter_fn: FilterFn,
    padding: float = 0.2,
) -> list[CropItem]:
    """Single crop covering the union of all matching boxes."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return []
    matched = filter_detections(detections, filter_fn)
    if not matched:
        return []

    img_h, img_w = image_bgr.shape[:2]
    xs1: list[float] = []
    ys1: list[float] = []
    xs2: list[float] = []
    ys2: list[float] = []
    indices: list[int] = []
    for idx, det in matched:
        x, y, w, h = det.bbox
        xs1.append(x)
        ys1.append(y)
        xs2.append(x + w)
        ys2.append(y + h)
        indices.append(idx)

    union = [min(xs1), min(ys1), max(xs2) - min(xs1), max(ys2) - min(ys1)]
    expanded = _expand_xywh(union, padding, img_w, img_h)
    crop = _crop_xywh(image_bgr, expanded)
    if crop is None:
        return []
    return [CropItem(image_bgr=crop, det_indices=indices, bbox_xywh=expanded)]


def single_person_expand(
    image_bgr: Any,
    detections: list[RawDetection],
    padding: float = 0.3,
) -> list[CropItem]:
    """Compatibility alias: single_expand with person filter."""
    return single_expand(image_bgr, detections, filter_person, padding=padding)


def multi_person_union(
    image_bgr: Any,
    detections: list[RawDetection],
    padding: float = 0.2,
) -> list[CropItem]:
    """Compatibility alias: multi_union with person filter."""
    return multi_union(image_bgr, detections, filter_person, padding=padding)


def crop_for_behavior(
    image_bgr: Any,
    detections: list[RawDetection],
    crop_mode: str,
    padding: float,
) -> list[CropItem]:
    mode = (crop_mode or "single_person_expand").strip().lower()
    if mode in {"multi_person_union", "multi_union_person"}:
        return multi_union(image_bgr, detections, filter_person, padding=padding)
    if mode in {"single_fire_smoke_expand", "fire_smoke_expand"}:
        return single_expand(image_bgr, detections, filter_fire_smoke, padding=padding)
    if mode in {"multi_fire_smoke_union", "fire_smoke_union"}:
        return multi_union(image_bgr, detections, filter_fire_smoke, padding=padding)
    # default: single person expand
    return single_expand(image_bgr, detections, filter_person, padding=padding)

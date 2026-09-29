"""Shared helpers for preview image / stream endpoints."""

from __future__ import annotations

import base64
import logging

from app.registry import ALGORITHM_SPECS, plan_request

logger = logging.getLogger(__name__)

_LABEL_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 200, 83),
    (255, 98, 41),
    (0, 0, 213),
    (0, 109, 255),
    (255, 0, 170),
    (212, 184, 0),
    (98, 17, 197),
    (23, 221, 100),
]


def _label_color(
    label: str, _cache: dict[str, tuple[int, int, int]] | None = None
) -> tuple[int, int, int]:
    if _cache is None:
        # module-level cache via function default mutation is avoided; use getattr
        cache = getattr(_label_color, "_cache", None)
        if cache is None:
            cache = {}
            setattr(_label_color, "_cache", cache)
    else:
        cache = _cache
    if label not in cache:
        cache[label] = _LABEL_COLORS_BGR[len(cache) % len(_LABEL_COLORS_BGR)]
    return cache[label]


def dedupe_prompts(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in lst:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def resolve_prompts(algorithms: str | None, prompts: str | None) -> list[str]:
    algo_codes = [a.strip() for a in (algorithms or "").split(",") if a.strip()]
    extra_prompts = [p.strip() for p in (prompts or "").split(",") if p.strip()]

    if algo_codes:
        plan = plan_request(algo_codes)
        if plan.unknown:
            logger.warning("Preview: unknown algorithm codes=%s", plan.unknown)
        return dedupe_prompts(plan.prompts + extra_prompts)
    if extra_prompts:
        return extra_prompts
    return dedupe_prompts(
        [p for spec in ALGORITHM_SPECS.values() for p in spec.prompts]
    )


def draw_bboxes(image_bgr, detections: list):
    """Draw bounding boxes + labels on a copy of the image."""
    import cv2  # noqa: PLC0415

    img = image_bgr.copy()
    for det in detections:
        x, y, w, h = det.bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = _label_color(det.label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{det.label}  {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.65, img.shape[1] / 1200))
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        bg_y1 = max(y1 - th - baseline - 6, 0)
        bg_y2 = y1
        cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 6, bg_y2), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 3, bg_y2 - baseline - 2),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
    return img


def decode_image_bytes(image_bytes: bytes):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_b64(image_bgr, quality: int = 85) -> str:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode()


def maybe_resize(image_bgr, max_width: int):
    """Downscale if wider than max_width; returns (image, scale_factor)."""
    import cv2  # noqa: PLC0415

    if max_width <= 0:
        return image_bgr, 1.0
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr, 1.0
    scale = max_width / float(w)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale


def detections_to_json(detections: list) -> list[dict]:
    return [
        {
            "label": d.label,
            "object_type": d.object_type,
            "confidence": round(d.confidence, 4),
            "bbox": [round(v, 1) for v in d.bbox],
        }
        for d in detections
    ]

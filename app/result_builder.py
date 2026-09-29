"""Build Kafka-compatible AiInferenceResult from detections."""

from __future__ import annotations

import time
import uuid

from app.detectors.base import RawDetection
from app.registry import get_spec, prompt_matches_detection
from app.schemas import AiInferenceRequest, AiInferenceResult, DetectionItem


def _new_det_id() -> str:
    return uuid.uuid4().hex[:12]


def filter_for_code(code: str, detections: list[RawDetection]) -> list[RawDetection]:
    spec = get_spec(code)
    if spec is None:
        return []
    if not spec.prompts:
        return []
    matched: list[RawDetection] = []
    for det in detections:
        for prompt in spec.prompts:
            if prompt_matches_detection(prompt, det.object_type, det.label):
                matched.append(det)
                break
    return matched


def build_result(
    request: AiInferenceRequest,
    all_detections: list[RawDetection],
    *,
    image_urls: dict[str, str] | None = None,
    stage: str | None = "yolo",
    partial: bool | None = None,
) -> AiInferenceResult:
    """One slot per requested algorithmType (including unknown → num=0)."""
    image_urls = image_urls or {}
    algorithm_type: list[str] = []
    algorithm_num: list[int] = []
    image_url: list[str] = []
    detection_items: list[DetectionItem] = []

    # Preserve request order; empty list → empty result slots (still valid for alignment check)
    codes = [c.strip() for c in (request.algorithm_types or []) if c and str(c).strip()]
    if not codes:
        # Consumer requires size > 0; emit a placeholder unknown slot
        codes = ["UNKNOWN"]

    for code in codes:
        matched = filter_for_code(code, all_detections)
        algorithm_type.append(code)
        algorithm_num.append(len(matched))
        image_url.append(image_urls.get(code, ""))

        for det in matched:
            detection_items.append(
                DetectionItem.model_validate(
                    {
                        "detId": _new_det_id(),
                        "objectType": det.object_type,
                        "confidence": det.confidence,
                        "bbox": list(det.bbox),
                        "label": det.label,
                        "algorithmType": code,
                    }
                )
            )

    ts = request.timestamp if request.timestamp else int(time.time() * 1000)
    return AiInferenceResult.model_validate(
        {
            "requestId": request.request_id,
            "taskId": request.task_id,
            "deviceId": request.device_id,
            "algorithmType": algorithm_type,
            "algorithmNum": algorithm_num,
            "imageUrl": image_url,
            "timestamp": ts,
            "detections": detection_items,
            "stage": stage,
            "partial": partial,
        }
    )


def build_zero_result(request: AiInferenceRequest, reason: str = "") -> AiInferenceResult:
    _ = reason
    codes = [c.strip() for c in (request.algorithm_types or []) if c and str(c).strip()]
    if not codes:
        codes = ["UNKNOWN"]
    n = len(codes)
    ts = request.timestamp if request.timestamp else int(time.time() * 1000)
    return AiInferenceResult.model_validate(
        {
            "requestId": request.request_id,
            "taskId": request.task_id,
            "deviceId": request.device_id,
            "algorithmType": codes,
            "algorithmNum": [0] * n,
            "imageUrl": [""] * n,
            "timestamp": ts,
            "detections": [],
            "stage": "yolo",
            "partial": False,
        }
    )

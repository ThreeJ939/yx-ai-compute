"""Build Kafka-compatible AiInferenceResult from detections."""

from __future__ import annotations

import time
import uuid

from app.detectors.base import RawDetection
from app.registry import get_spec, prompt_matches_detection
from app.schemas import AiInferenceRequest, AiInferenceResult, AlgorithmResult, DetectionItem


def _new_det_id() -> str:
    return uuid.uuid4().hex[:12]


def build_source_image_url(
    request: AiInferenceRequest,
    *,
    endpoint: str | None = None,
    secure: bool = False,
) -> str:
    """Original frame URL for results[].imageUrl (not annotated).

    Prefer path-style MinIO HTTP(S) URL when endpoint is provided; otherwise
    fall back to ``s3://{bucket}/{key}``.
    """
    bucket = (request.bucket or "").strip().strip("/")
    key = (request.image_object_key or "").strip().lstrip("/")
    if not bucket or not key:
        return ""
    host = (endpoint or "").strip().rstrip("/")
    if not host:
        return f"s3://{bucket}/{key}"
    if "://" in host:
        return f"{host}/{bucket}/{key}"
    scheme = "https" if secure else "http"
    return f"{scheme}://{host}/{bucket}/{key}"


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


def _to_detection_items(matched: list[RawDetection]) -> list[DetectionItem]:
    items: list[DetectionItem] = []
    for det in matched:
        item: dict = {
            "detId": _new_det_id(),
            "objectType": det.object_type,
            "confidence": det.confidence,
            "bbox": list(det.bbox),
            "label": det.label,
        }
        if det.behavior:
            item["behavior"] = det.behavior
        items.append(DetectionItem.model_validate(item))
    return items


def _request_codes(request: AiInferenceRequest) -> list[str]:
    codes = [c.strip() for c in (request.algorithm_types or []) if c and str(c).strip()]
    if not codes:
        # Consumer requires size > 0; emit a placeholder unknown slot
        return ["UNKNOWN"]
    return codes


def build_result(
    request: AiInferenceRequest,
    all_detections: list[RawDetection],
    *,
    image_urls: dict[str, str] | None = None,
    llm_enabled: bool = True,
    minio_endpoint: str | None = None,
    minio_secure: bool = False,
) -> AiInferenceResult:
    """One AlgorithmResult per requested algorithmType (including unknown → num=0).

    BEH_* (stage=behavior) slots are skipped entirely when llm_enabled is False.
    When LLM is on, only detections with a non-empty behavior are included.

    Each slot's imageUrl defaults to the request's original frame URL.
    """
    image_urls = image_urls or {}
    default_url = build_source_image_url(
        request, endpoint=minio_endpoint, secure=minio_secure
    )
    algo_results: list[AlgorithmResult] = []

    for code in _request_codes(request):
        spec = get_spec(code)
        if spec is not None and spec.stage == "behavior":
            if not llm_enabled:
                continue  # BEH_* and LLM off → skip slot
            matched = [d for d in filter_for_code(code, all_detections) if d.behavior]
        else:
            matched = filter_for_code(code, all_detections)

        detection_items = _to_detection_items(matched)
        algo_results.append(
            AlgorithmResult.model_validate(
                {
                    "algorithmType": code,
                    "algorithmNum": len(detection_items),
                    "imageUrl": image_urls.get(code, default_url),
                    "detections": detection_items,
                }
            )
        )

    ts = request.timestamp if request.timestamp else int(time.time() * 1000)
    return AiInferenceResult.model_validate(
        {
            "requestId": request.request_id,
            "taskId": request.task_id,
            "deviceId": request.device_id,
            "timestamp": ts,
            "results": algo_results,
        }
    )


def build_zero_result(
    request: AiInferenceRequest,
    reason: str = "",
    *,
    llm_enabled: bool = True,
    minio_endpoint: str | None = None,
    minio_secure: bool = False,
) -> AiInferenceResult:
    """Emit zero-count slots for failure paths; skip BEH_* when LLM is off."""
    _ = reason
    default_url = build_source_image_url(
        request, endpoint=minio_endpoint, secure=minio_secure
    )
    algo_results: list[AlgorithmResult] = []
    for code in _request_codes(request):
        spec = get_spec(code)
        if spec is not None and spec.stage == "behavior" and not llm_enabled:
            continue
        algo_results.append(
            AlgorithmResult.model_validate(
                {
                    "algorithmType": code,
                    "algorithmNum": 0,
                    "imageUrl": default_url,
                    "detections": [],
                }
            )
        )

    ts = request.timestamp if request.timestamp else int(time.time() * 1000)
    return AiInferenceResult.model_validate(
        {
            "requestId": request.request_id,
            "taskId": request.task_id,
            "deviceId": request.device_id,
            "timestamp": ts,
            "results": algo_results,
        }
    )

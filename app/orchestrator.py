"""Request orchestration: download → YOLO-World → optional enhancer → result."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.config import Settings
from app.detectors.yolo_world import YoloWorldDetector
from app.enhancers.base import EnhanceRequest
from app.enhancers.noop import NoopBehaviorEnhancer
from app.minio_client import MinioImageStore
from app.registry import get_spec, plan_request
from app.result_builder import build_result, build_zero_result
from app.schemas import AiInferenceRequest, AiInferenceResult

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        minio: MinioImageStore,
        detector: YoloWorldDetector | None = None,
        enhancer: NoopBehaviorEnhancer | None = None,
    ) -> None:
        self._settings = settings
        self._minio = minio
        self._detector = detector or YoloWorldDetector(
            weights=settings.yolo_world_weights,
            device=settings.device,
            clip_weights=settings.yolo_clip_weights,
            tiled=settings.yolo_tiled,
            tile_size=settings.yolo_tile_size,
            tile_overlap=settings.yolo_tile_overlap,
            imgsz=settings.yolo_imgsz,
        )
        self._enhancer = enhancer or NoopBehaviorEnhancer()
        # YOLO/ultralytics is sync; run in thread pool from async Kafka loop
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def process(self, request: AiInferenceRequest) -> AiInferenceResult:
        try:
            image_bytes = self._minio.download_bytes(
                request.bucket or "",
                request.image_object_key or "",
            )
        except Exception:
            logger.exception(
                "MinIO download failed requestId=%s bucket=%s key=%s",
                request.request_id,
                request.bucket,
                request.image_object_key,
            )
            raise

        image_bgr = _decode_image(image_bytes)
        if image_bgr is None:
            logger.error("Failed to decode image requestId=%s", request.request_id)
            return build_zero_result(request, reason="decode_failed")

        plan = plan_request(request.algorithm_types)
        if plan.unknown:
            logger.warning(
                "Unknown algorithmTypes=%s requestId=%s",
                plan.unknown,
                request.request_id,
            )

        try:
            detections = await _run_in_executor(
                self._pool,
                self._infer_sync,
                image_bgr,
                plan.prompts,
            )
        except Exception:
            logger.exception("Inference failed requestId=%s", request.request_id)
            return build_zero_result(request, reason="infer_failed")

        # Phase-1: call noop enhancer for behavior-stage codes (extension point)
        for code in plan.known:
            spec = get_spec(code)
            if spec is None or spec.stage != "behavior":
                continue
            enhance_result = self._enhancer.enhance(
                EnhanceRequest(
                    behavior=code,
                    image_bgr=image_bgr,
                    detections=list(detections),
                    algorithm_code=code,
                )
            )
            if enhance_result.applied:
                detections = enhance_result.detections

        return build_result(
            request,
            detections,
            stage="yolo",
            partial=False,
        )

    async def detect_direct(
        self,
        image_bytes: bytes,
        prompts: list[str],
        conf: float | None = None,
        iou: float | None = None,
    ):
        """Run YOLO-World directly on raw image bytes, bypassing Kafka / MinIO.

        Used by the preview HTTP endpoint. Returns ``DetectBatchResult``.
        """
        image_bgr = _decode_image(image_bytes)
        if image_bgr is None:
            raise ValueError("Cannot decode image bytes")

        effective_conf = conf if conf is not None else self._settings.conf_threshold
        effective_iou = iou if iou is not None else self._settings.iou_threshold

        if not prompts:
            from app.detectors.base import DetectBatchResult
            return DetectBatchResult()

        return await _run_in_executor(
            self._pool,
            self._detector.detect,
            image_bgr,
            prompts,
            effective_conf,
            effective_iou,
        )

    def _infer_sync(self, image_bgr: np.ndarray, prompts: list[str]):
        if not prompts:
            return []
        batch = self._detector.detect(
            image_bgr,
            prompts=prompts,
            conf=self._settings.conf_threshold,
            iou=self._settings.iou_threshold,
        )
        logger.info(
            "YOLO-World done prompts=%s detections=%s",
            prompts,
            len(batch.detections),
        )
        return batch.detections


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


async def _run_in_executor(pool: ThreadPoolExecutor, fn, *args):
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, lambda: fn(*args))

"""Request orchestration: download → multi-backend detect → optional enhancer → result."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.config import Settings
from app.detectors.base import DetectBatchResult, Detector, RawDetection
from app.detectors.yolo_world import YoloWorldDetector
from app.enhancers.base import BehaviorEnhancer, EnhanceRequest
from app.enhancers.noop import NoopBehaviorEnhancer
from app.minio_client import MinioImageStore
from app.registry import (
    SlotPlan,
    collect_prompts,
    get_spec,
    plan_request,
    prompt_matches_detection,
)
from app.result_builder import build_result, build_zero_result
from app.schemas import AiInferenceRequest, AiInferenceResult

logger = logging.getLogger(__name__)


def _apply_algo_conf_filter(
    dets: list[RawDetection],
    codes: list[str],
) -> list[RawDetection]:
    """Drop detections below per-algorithm conf_threshold when configured (> 0).

    - Specs with prompts: apply threshold only to matching labels.
    - Specs with empty prompts (e.g. fire_smoke): apply to the whole batch
      for this backend call.
    - Specs with conf_threshold <= 0: no post-filter.
    """
    prompt_rules: list[tuple[tuple[str, ...], float]] = []
    empty_prompt_threshold = 0.0
    for code in codes:
        spec = get_spec(code)
        if spec is None or spec.conf_threshold <= 0:
            continue
        if not spec.prompts:
            empty_prompt_threshold = max(empty_prompt_threshold, spec.conf_threshold)
        else:
            prompt_rules.append((spec.prompts, spec.conf_threshold))

    if empty_prompt_threshold <= 0 and not prompt_rules:
        return dets

    out: list[RawDetection] = []
    for det in dets:
        applicable: list[float] = []
        for prompts, thr in prompt_rules:
            if any(
                prompt_matches_detection(p, det.object_type, det.label)
                for p in prompts
            ):
                applicable.append(thr)
        if not applicable and empty_prompt_threshold > 0:
            applicable.append(empty_prompt_threshold)
        if not applicable or det.confidence >= max(applicable):
            out.append(det)
    return out


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        minio: MinioImageStore,
        detector: YoloWorldDetector | None = None,
        detector_registry: dict[str, Detector] | None = None,
        enhancer: BehaviorEnhancer | None = None,
    ) -> None:
        self._settings = settings
        self._minio = minio

        if detector_registry is not None:
            self._detectors: dict[str, Detector] = dict(detector_registry)
        else:
            world = detector or YoloWorldDetector(
                weights=settings.yolo_world_weights,
                device=settings.device,
                clip_weights=settings.yolo_clip_weights,
                tiled=settings.yolo_tiled,
                tile_size=settings.yolo_tile_size,
                tile_overlap=settings.yolo_tile_overlap,
                imgsz=settings.yolo_imgsz,
            )
            self._detectors = {"world": world}

        # Backward-compat alias used by older callers / tests
        self._detector = self._detectors.get("world")

        self._enhancer: BehaviorEnhancer = enhancer or NoopBehaviorEnhancer()
        # YOLO/LLM are sync; run in thread pool from async Kafka loop
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
                self._infer_plan_sync,
                image_bgr,
                plan,
                self._settings.yolo_conf,
                self._settings.yolo_iou,
            )
        except Exception:
            logger.exception("Inference failed requestId=%s", request.request_id)
            return build_zero_result(request, reason="infer_failed")

        llm_applied = False
        for code in plan.known:
            spec = get_spec(code)
            if spec is None or spec.stage != "behavior":
                continue
            try:
                enhance_result = await _run_in_executor(
                    self._pool,
                    self._enhancer.enhance,
                    EnhanceRequest(
                        behavior=code,
                        image_bgr=image_bgr,
                        detections=list(detections),
                        algorithm_code=code,
                    ),
                )
            except Exception:
                logger.exception(
                    "Enhancer failed code=%s requestId=%s",
                    code,
                    request.request_id,
                )
                continue
            if enhance_result.applied:
                detections = enhance_result.detections
                llm_applied = True

        return build_result(
            request,
            detections,
            stage="behavior" if llm_applied else "yolo",
            partial=False,
        )

    async def enhance_direct(
        self,
        image_bgr,
        detections: list,
        algorithm_codes: list[str],
    ) -> tuple[list, bool, list]:
        """Run behavior enhancers for given codes on an already-decoded image.

        Used by preview endpoints (image + stream) after detect_direct().
        Returns (updated_detections, llm_applied, vlm_crops).
        Does NOT raise; enhancer failures are logged and skipped.
        """
        llm_applied = False
        vlm_crops: list = []
        for code in algorithm_codes:
            spec = get_spec(code)
            if spec is None or spec.stage != "behavior":
                continue
            try:
                enhance_result = await _run_in_executor(
                    self._pool,
                    self._enhancer.enhance,
                    EnhanceRequest(
                        behavior=code,
                        image_bgr=image_bgr,
                        detections=list(detections),
                        algorithm_code=code,
                    ),
                )
            except Exception:
                logger.exception("Preview enhance failed code=%s", code)
                continue
            if getattr(enhance_result, "crops", None):
                vlm_crops.extend(enhance_result.crops)
            if enhance_result.applied:
                detections = enhance_result.detections
                llm_applied = True
            elif enhance_result.detections:
                # Keep YOLO boxes even when VLM did not confirm
                detections = enhance_result.detections
        return list(detections), llm_applied, vlm_crops

    async def detect_direct(
        self,
        image_bytes: bytes,
        prompts: list[str] | None = None,
        conf: float | None = None,
        iou: float | None = None,
        algorithm_codes: list[str] | None = None,
    ):
        """Run detectors on raw image bytes, bypassing Kafka / MinIO.

        Prefer ``algorithm_codes`` so non-world backends (fire_smoke) are included.
        ``prompts`` alone still works for world-only preview callers.
        """
        image_bgr = _decode_image(image_bytes)
        if image_bgr is None:
            raise ValueError("Cannot decode image bytes")

        effective_conf = conf if conf is not None else self._settings.yolo_conf
        effective_iou = iou if iou is not None else self._settings.yolo_iou

        if algorithm_codes:
            plan = plan_request(algorithm_codes)
            return await _run_in_executor(
                self._pool,
                self._infer_plan_to_batch,
                image_bgr,
                plan,
                effective_conf,
                effective_iou,
            )

        # Legacy world-only path
        if not prompts:
            return DetectBatchResult()

        world = self._detectors.get("world")
        if world is None:
            return DetectBatchResult()

        return await _run_in_executor(
            self._pool,
            world.detect,
            image_bgr,
            prompts,
            effective_conf,
            effective_iou,
        )

    def _infer_plan_to_batch(
        self,
        image_bgr: np.ndarray,
        plan: SlotPlan,
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        dets = self._infer_plan_sync(image_bgr, plan, conf, iou)
        return DetectBatchResult(detections=dets)

    def _infer_plan_sync(
        self,
        image_bgr: np.ndarray,
        plan: SlotPlan,
        conf: float,
        iou: float,
    ) -> list[RawDetection]:
        all_dets: list[RawDetection] = []
        backends = plan.by_backend or {}
        if not backends and plan.prompts:
            # Safety: prompts without by_backend → world
            backends = {"world": plan.known or ["__prompts__"]}

        for backend, codes in backends.items():
            detector = self._detectors.get(backend)
            if detector is None:
                logger.warning(
                    "No detector registered for backend=%s codes=%s; skipping",
                    backend,
                    codes,
                )
                continue
            if backend == "world":
                prompts = collect_prompts(codes, backend="world")
                if not prompts and plan.prompts:
                    prompts = list(plan.prompts)
                if not prompts:
                    continue
            else:
                prompts = []

            if backend == "fire_smoke":
                effective_conf = (
                    self._settings.fire_smoke_conf or conf
                )
                effective_iou = (
                    self._settings.fire_smoke_iou or iou
                )
            else:
                effective_conf = conf
                effective_iou = iou

            batch = detector.detect(
                image_bgr,
                prompts=prompts,
                conf=effective_conf,
                iou=effective_iou,
            )
            filtered = _apply_algo_conf_filter(batch.detections, codes)
            logger.info(
                "Detect done backend=%s codes=%s detections=%s filtered=%s conf=%.3f iou=%.3f",
                backend,
                codes,
                len(batch.detections),
                len(filtered),
                effective_conf,
                effective_iou,
            )
            all_dets.extend(filtered)
        return all_dets

    def _infer_sync(self, image_bgr: np.ndarray, prompts: list[str]):
        """Legacy helper: world-only detect by prompts."""
        if not prompts:
            return []
        world = self._detectors.get("world")
        if world is None:
            return []
        batch = world.detect(
            image_bgr,
            prompts=prompts,
            conf=self._settings.yolo_conf,
            iou=self._settings.yolo_iou,
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

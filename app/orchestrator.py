"""Request orchestration: download → multi-backend detect → optional enhancer → result."""

from __future__ import annotations

import asyncio
import copy
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.config import Settings
from app.detectors.base import DetectBatchResult, Detector, RawDetection
from app.detectors.yolo_world import YoloWorldDetector
from app.enhancers.base import BehaviorEnhancer, EnhanceRequest, EnhanceResult
from app.enhancers.noop import NoopBehaviorEnhancer
from app.minio_client import MinioImageStore
from app.registry import (
    SlotPlan,
    collect_prompts,
    get_spec,
    plan_request,
    prompt_matches_detection,
)
from app.result_builder import build_result, has_positive_detections
from app.schemas import AiInferenceRequest, AiInferenceResult

logger = logging.getLogger(__name__)


def _merge_behavior(
    base: list[RawDetection],
    enhance_results: list[EnhanceResult],
) -> list[RawDetection]:
    """Merge BEH_* enhance outputs onto a copy of base detections.

    Each enhancer may set ``behavior`` on matching boxes. Later results
    overwrite earlier non-empty behavior labels for the same index.
    """
    out = [copy.copy(d) for d in base]
    for er in enhance_results:
        if not er.detections:
            continue
        # Prefer length-aligned merge; fall back to shorter zip
        n = min(len(out), len(er.detections))
        for i in range(n):
            beh = getattr(er.detections[i], "behavior", None)
            if beh:
                out[i].behavior = beh
                out[i].confidence = er.detections[i].confidence
    return out


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
        # YOLO stays serial (GPU single-stream); LLM pool sized for crop concurrency
        self._yolo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        llm_workers = max(1, int(settings.llm_crop_concurrency))
        self._llm_pool = ThreadPoolExecutor(
            max_workers=llm_workers, thread_name_prefix="llm"
        )
        # Backward-compat alias (tests / older callers)
        self._pool = self._yolo_pool

    def close(self) -> None:
        self._yolo_pool.shutdown(wait=False, cancel_futures=True)
        self._llm_pool.shutdown(wait=False, cancel_futures=True)

    async def process(self, request: AiInferenceRequest) -> AiInferenceResult | None:
        """Run inference; return result only when there is at least one detection.

        Empty / failure outcomes return ``None`` so Kafka skips publish (alarm-only,
        lower result-topic load). Request offset is still committed by the consumer.
        """
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
            logger.error(
                "Failed to decode image requestId=%s; skip result publish",
                request.request_id,
            )
            return None

        plan = plan_request(request.algorithm_types)
        if plan.unknown:
            logger.warning(
                "Unknown algorithmTypes=%s requestId=%s",
                plan.unknown,
                request.request_id,
            )

        try:
            detections = await _run_in_executor(
                self._yolo_pool,
                self._infer_plan_sync,
                image_bgr,
                plan,
                self._settings.yolo_conf,
                self._settings.yolo_iou,
            )
        except Exception:
            logger.exception(
                "Inference failed requestId=%s; skip result publish",
                request.request_id,
            )
            return None

        beh_codes = [
            code
            for code in plan.known
            if (spec := get_spec(code)) is not None and spec.stage == "behavior"
        ]
        if beh_codes:
            tasks = [
                _run_in_executor(
                    self._llm_pool,
                    self._enhancer.enhance,
                    EnhanceRequest(
                        behavior=code,
                        image_bgr=image_bgr,
                        detections=list(detections),
                        algorithm_code=code,
                    ),
                )
                for code in beh_codes
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            applied_results: list[EnhanceResult] = []
            for code, item in zip(beh_codes, gathered):
                if isinstance(item, Exception):
                    logger.error(
                        "Enhancer failed code=%s requestId=%s",
                        code,
                        request.request_id,
                        exc_info=item,
                    )
                    continue
                if item.applied:
                    applied_results.append(item)
            if applied_results:
                detections = _merge_behavior(detections, applied_results)

        result = build_result(
            request,
            detections,
            llm_enabled=self._settings.llm_enabled,
            minio_endpoint=self._settings.minio_endpoint,
            minio_secure=self._settings.minio_secure,
        )
        if not has_positive_detections(result):
            logger.info(
                "No detections requestId=%s taskId=%s; skip result publish",
                request.request_id,
                request.task_id,
            )
            return None
        return result

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
        beh_codes = [
            code
            for code in algorithm_codes
            if (spec := get_spec(code)) is not None and spec.stage == "behavior"
        ]
        if not beh_codes:
            return list(detections), llm_applied, vlm_crops

        tasks = [
            _run_in_executor(
                self._llm_pool,
                self._enhancer.enhance,
                EnhanceRequest(
                    behavior=code,
                    image_bgr=image_bgr,
                    detections=list(detections),
                    algorithm_code=code,
                ),
            )
            for code in beh_codes
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        applied_results: list[EnhanceResult] = []
        for code, item in zip(beh_codes, gathered):
            if isinstance(item, Exception):
                logger.error("Preview enhance failed code=%s", code, exc_info=item)
                continue
            if getattr(item, "crops", None):
                vlm_crops.extend(item.crops)
            if item.applied:
                applied_results.append(item)
                llm_applied = True
            elif item.detections and not applied_results:
                # Keep YOLO boxes even when VLM did not confirm (no applied yet)
                detections = item.detections
        if applied_results:
            detections = _merge_behavior(detections, applied_results)
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
                self._yolo_pool,
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
            self._yolo_pool,
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, lambda: fn(*args))

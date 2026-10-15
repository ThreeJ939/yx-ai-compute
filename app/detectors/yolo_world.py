"""YOLO-World detector via Ultralytics, with optional sliding-window inference."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.detectors.base import DetectBatchResult, RawDetection
from app.detectors.clip_path import ensure_clip_caches
from app.detectors.tiling import as_predict_result_list, build_tile_crops

logger = logging.getLogger(__name__)


class YoloWorldDetector:
    """Thread-safe lazy-loaded YOLO-World wrapper with demo-style tiling."""

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        *,
        clip_weights: str = "",
        tiled: bool = True,
        tile_size: int = 640,
        tile_overlap: float = 0.2,
        tile_batch: int = 4,
        imgsz: int = 640,
    ) -> None:
        self._weights = weights
        self._device = device
        self._clip_weights = (clip_weights or "").strip()
        self._tiled = tiled
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_batch = max(1, int(tile_batch))
        self._imgsz = imgsz
        self._model: Any = None
        self._lock = threading.Lock()
        self._set_classes: tuple[str, ...] | None = None
        self._clip_ready = False

    def _ensure_clip(self) -> None:
        if self._clip_ready:
            return
        if self._clip_weights:
            path = ensure_clip_caches(self._clip_weights)
            logger.info("Using explicit CLIP weights: %s", path)
        self._clip_ready = True

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self._ensure_clip()
            model = self._load_model()
            self._model = model
            return self._model

    def _load_model(self) -> Any:
        logger.info(
            "Loading YOLO-World weights=%s device=%s clip=%s tiled=%s tile=%s overlap=%s batch=%s",
            self._weights,
            self._device,
            self._clip_weights or "(auto)",
            self._tiled,
            self._tile_size,
            self._tile_overlap,
            self._tile_batch,
        )
        try:
            from ultralytics import YOLOWorld

            return YOLOWorld(self._weights)
        except Exception:
            from ultralytics import YOLO

            logger.warning("YOLOWorld unavailable, falling back to YOLO(%s)", self._weights)
            return YOLO(self._weights)

    def _apply_classes(self, model: Any, classes: list[str]) -> None:
        """set_classes with optional local CLIP; fall back across Ultralytics versions."""
        self._ensure_clip()
        # Newer forks sometimes accept an explicit path kwarg
        if self._clip_weights:
            try:
                model.set_classes(classes, clip_model_path=self._clip_weights)
                return
            except TypeError:
                pass
        model.set_classes(classes)

    def detect(
        self,
        image_bgr: np.ndarray,
        prompts: list[str],
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        if not prompts:
            return DetectBatchResult()

        unique_prompts = list(dict.fromkeys(p.strip() for p in prompts if p and p.strip()))
        if not unique_prompts:
            return DetectBatchResult()

        if image_bgr is None or image_bgr.size == 0:
            return DetectBatchResult()

        model = self._ensure_model()
        prompt_key = tuple(unique_prompts)

        with self._lock:
            if self._set_classes != prompt_key:
                self._apply_classes(model, list(unique_prompts))
                self._set_classes = prompt_key

            h, w = image_bgr.shape[:2]
            use_tiles = self._tiled and (w > self._tile_size or h > self._tile_size)
            if use_tiles:
                return self._detect_tiled(model, image_bgr, unique_prompts, conf, iou)
            return self._detect_full(model, image_bgr, unique_prompts, conf, iou)

    def _detect_full(
        self,
        model: Any,
        image_bgr: np.ndarray,
        unique_prompts: list[str],
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        results = model.predict(
            source=image_bgr,
            imgsz=self._imgsz,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        return _boxes_to_detections(results, unique_prompts)

    def _detect_tiled(
        self,
        model: Any,
        image_bgr: np.ndarray,
        unique_prompts: list[str],
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        import torch
        from torchvision.ops import nms

        h, w = image_bgr.shape[:2]
        # Demo uses RGB array; Ultralytics accepts BGR ndarray as OpenCV default.
        # Keep BGR consistent with orchestrator decode path.
        all_boxes: list[Any] = []
        all_scores: list[Any] = []
        all_cls: list[Any] = []
        names: dict[int, str] = {}

        tiles = build_tile_crops(
            image_bgr, tile=self._tile_size, overlap=self._tile_overlap
        )
        batch_size = self._tile_batch
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start : start + batch_size]
            crops = [crop for _x1, _y1, crop in chunk]
            raw_results = model.predict(
                source=crops if len(crops) > 1 else crops[0],
                imgsz=self._imgsz,
                conf=conf,
                device=self._device,
                verbose=False,
            )
            try:
                results = as_predict_result_list(raw_results, len(crops))
            except ValueError:
                logger.exception(
                    "Tile batch align failed; falling back to serial for chunk start=%s size=%s",
                    start,
                    len(crops),
                )
                results = []
                for crop in crops:
                    one = model.predict(
                        source=crop,
                        imgsz=self._imgsz,
                        conf=conf,
                        device=self._device,
                        verbose=False,
                    )
                    results.extend(as_predict_result_list(one, 1))

            for (x1, y1, _crop), result in zip(chunk, results):
                names = result.names or names
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Clone: YOLO boxes may be inference tensors; inplace += is forbidden.
                xyxy = result.boxes.xyxy.detach().cpu().clone()
                xyxy[:, 0] = xyxy[:, 0] + x1
                xyxy[:, 2] = xyxy[:, 2] + x1
                xyxy[:, 1] = xyxy[:, 1] + y1
                xyxy[:, 3] = xyxy[:, 3] + y1
                xyxy[:, 0] = xyxy[:, 0].clamp(0, w)
                xyxy[:, 2] = xyxy[:, 2].clamp(0, w)
                xyxy[:, 1] = xyxy[:, 1].clamp(0, h)
                xyxy[:, 3] = xyxy[:, 3].clamp(0, h)
                all_boxes.append(xyxy)
                all_scores.append(result.boxes.conf.detach().cpu().clone())
                all_cls.append(result.boxes.cls.detach().cpu().long().clone())

        if not all_boxes:
            return DetectBatchResult()

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        clses = torch.cat(all_cls, dim=0)

        keep_idx: list[Any] = []
        for cid in clses.unique():
            mask = clses == cid
            idx = torch.where(mask)[0]
            kept = nms(boxes[mask], scores[mask], iou)
            keep_idx.append(idx[kept])
        keep = torch.cat(keep_idx)
        boxes, scores, clses = boxes[keep], scores[keep], clses[keep]

        out: list[RawDetection] = []
        for (xa, ya, xb, yb), score, cid in zip(
            boxes.tolist(), scores.tolist(), clses.tolist()
        ):
            cls_id = int(cid)
            name = str(
                names.get(
                    cls_id,
                    unique_prompts[cls_id] if cls_id < len(unique_prompts) else "object",
                )
            )
            out.append(
                RawDetection(
                    object_type=name,
                    label=name,
                    confidence=float(score),
                    bbox=[
                        float(xa),
                        float(ya),
                        float(max(0.0, xb - xa)),
                        float(max(0.0, yb - ya)),
                    ],
                    prompt=name,
                )
            )
        logger.debug(
            "Tiled detect done tiles_area=%sx%s tiles=%s batch=%s detections=%s",
            w,
            h,
            len(tiles),
            batch_size,
            len(out),
        )
        return DetectBatchResult(detections=out)


def _boxes_to_detections(results: Any, unique_prompts: list[str]) -> DetectBatchResult:
    out: list[RawDetection] = []
    if not results:
        return DetectBatchResult(detections=out)

    result = results[0]
    names = result.names or {}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return DetectBatchResult(detections=out)

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        cls_id = int(clss[i])
        name = str(
            names.get(
                cls_id,
                unique_prompts[cls_id] if cls_id < len(unique_prompts) else "object",
            )
        )
        out.append(
            RawDetection(
                object_type=name,
                label=name,
                confidence=float(confs[i]),
                bbox=[float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                prompt=name,
            )
        )
    return DetectBatchResult(detections=out)

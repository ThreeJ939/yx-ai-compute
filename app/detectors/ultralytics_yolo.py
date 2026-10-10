"""Generic fixed-class Ultralytics YOLO detector (no set_classes / CLIP)."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.detectors.base import DetectBatchResult, RawDetection
from app.detectors.tiling import iter_tiles

logger = logging.getLogger(__name__)


class UltralyticsYoloDetector:
    """Thread-safe lazy-loaded Ultralytics YOLO for dedicated weights.

    Unlike YOLO-World, classes are baked into the checkpoint; ``prompts`` are
    ignored (kept only for Detector Protocol compatibility).
    """

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        *,
        tiled: bool = True,
        tile_size: int = 640,
        tile_overlap: float = 0.2,
        imgsz: int = 640,
        name: str = "ultralytics",
    ) -> None:
        self._weights = weights
        self._device = device
        self._tiled = tiled
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._imgsz = imgsz
        self._name = name
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from ultralytics import YOLO  # noqa: PLC0415

            logger.info(
                "Loading Ultralytics YOLO name=%s weights=%s device=%s tiled=%s tile=%s",
                self._name,
                self._weights,
                self._device,
                self._tiled,
                self._tile_size,
            )
            self._model = YOLO(self._weights)
            return self._model

    def detect(
        self,
        image_bgr: np.ndarray,
        prompts: list[str],
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        # prompts unused — fixed-class model
        del prompts
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return DetectBatchResult()

        model = self._ensure_model()
        with self._lock:
            h, w = image_bgr.shape[:2]
            use_tiles = self._tiled and (w > self._tile_size or h > self._tile_size)
            if use_tiles:
                return self._detect_tiled(model, image_bgr, conf, iou)
            return self._detect_full(model, image_bgr, conf, iou)

    def _detect_full(
        self,
        model: Any,
        image_bgr: np.ndarray,
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
        return _boxes_to_detections(results)

    def _detect_tiled(
        self,
        model: Any,
        image_bgr: np.ndarray,
        conf: float,
        iou: float,
    ) -> DetectBatchResult:
        import torch  # noqa: PLC0415
        from torchvision.ops import nms  # noqa: PLC0415

        h, w = image_bgr.shape[:2]
        all_boxes: list[Any] = []
        all_scores: list[Any] = []
        all_cls: list[Any] = []
        names: dict[int, str] = {}

        tile = self._tile_size
        for x1, y1 in iter_tiles(w, h, tile=tile, overlap=self._tile_overlap):
            x2, y2 = x1 + tile, y1 + tile
            crop = np.zeros((tile, tile, 3), dtype=np.uint8)
            region = image_bgr[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)]
            crop[: region.shape[0], : region.shape[1]] = region

            result = model.predict(
                source=crop,
                imgsz=self._imgsz,
                conf=conf,
                device=self._device,
                verbose=False,
            )[0]
            names = result.names or names
            if result.boxes is None or len(result.boxes) == 0:
                continue

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

        keep_parts: list[Any] = []
        for cid in clses.unique():
            mask = clses == cid
            idx = torch.where(mask)[0]
            kept = nms(boxes[mask], scores[mask], iou)
            keep_parts.append(idx[kept])
        keep = torch.cat(keep_parts)
        boxes, scores, clses = boxes[keep], scores[keep], clses[keep]

        out: list[RawDetection] = []
        for (xa, ya, xb, yb), score, cid in zip(
            boxes.tolist(), scores.tolist(), clses.tolist()
        ):
            cls_id = int(cid)
            name = str(names.get(cls_id, f"class_{cls_id}"))
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
            "Tiled %s detect done area=%sx%s detections=%s",
            self._name,
            w,
            h,
            len(out),
        )
        return DetectBatchResult(detections=out)


def _boxes_to_detections(results: Any) -> DetectBatchResult:
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
        name = str(names.get(cls_id, f"class_{cls_id}"))
        out.append(
            RawDetection(
                object_type=name,
                label=name,
                confidence=float(confs[i]),
                bbox=[
                    float(x1),
                    float(y1),
                    float(max(0.0, x2 - x1)),
                    float(max(0.0, y2 - y1)),
                ],
                prompt=name,
            )
        )
    return DetectBatchResult(detections=out)

#!/usr/bin/env python3
"""Local smoke test: run YOLO-World without Kafka / MinIO.

Usage (from repo root, with .env / weights ready):

  python -m scripts.infer_local path/to/image.jpg
  python -m scripts.infer_local image.jpg -a ENGINEERING_VEHICLE_DETECTION,DET_PERSON
  python -m scripts.infer_local image.jpg -p "person,boat,excavator" -o out.jpg

Reads DEVICE / YOLO_* from .env via app.config.Settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/infer_local.py` as well as `python -m scripts.infer_local`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local YOLO-World infer (no Kafka/MinIO)")
    p.add_argument("image", type=str, help="Path to input image (jpg/png)")
    p.add_argument(
        "-a",
        "--algorithms",
        type=str,
        default="ENGINEERING_VEHICLE_DETECTION,DET_PERSON",
        help="Comma-separated algorithmTypes (registry codes)",
    )
    p.add_argument(
        "-p",
        "--prompts",
        type=str,
        default="",
        help="Optional raw prompts, comma-separated; overrides -a for YOLO classes "
        "(result slots still follow -a if provided)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Optional path to save annotated image",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override CONF_THRESHOLD from .env",
    )
    p.add_argument(
        "--no-tile",
        action="store_true",
        help="Disable sliding-window tiling for this run",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    return p.parse_args()


def _draw(image_bgr, detections) -> None:
    import cv2

    for det in detections:
        x, y, w, h = det.bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            image_bgr,
            label,
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def main() -> int:
    args = _parse_args()

    import cv2

    from app.config import Settings
    from app.detectors.yolo_world import YoloWorldDetector
    from app.registry import collect_prompts, plan_request
    from app.result_builder import build_result
    from app.schemas import AiInferenceRequest
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("infer_local")

    image_path = Path(args.image)
    if not image_path.is_file():
        log.error("Image not found: %s", image_path)
        return 1

    settings = Settings()  # loads .env from cwd
    conf = args.conf if args.conf is not None else settings.conf_threshold
    tiled = False if args.no_tile else settings.yolo_tiled

    algorithm_types = [c.strip() for c in args.algorithms.split(",") if c.strip()]
    if args.prompts.strip():
        prompts = [c.strip() for c in args.prompts.split(",") if c.strip()]
        plan = plan_request(algorithm_types)
        log.info("Using raw prompts=%s (algorithms for slots=%s)", prompts, algorithm_types)
    else:
        plan = plan_request(algorithm_types)
        prompts = plan.prompts or collect_prompts(algorithm_types)
        if plan.unknown:
            log.warning("Unknown algorithm codes (slots will be 0): %s", plan.unknown)
        log.info("algorithms=%s prompts=%s", algorithm_types, prompts)

    if not prompts:
        log.error("No prompts to detect; check -a / -p")
        return 1

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        log.error("Failed to read image: %s", image_path)
        return 1

    detector = YoloWorldDetector(
        weights=settings.yolo_world_weights,
        device=settings.device,
        clip_weights=settings.yolo_clip_weights,
        tiled=tiled,
        tile_size=settings.yolo_tile_size,
        tile_overlap=settings.yolo_tile_overlap,
        imgsz=settings.yolo_imgsz,
    )

    t0 = time.perf_counter()
    batch = detector.detect(
        image_bgr,
        prompts=prompts,
        conf=conf,
        iou=settings.iou_threshold,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "detect done: %d boxes in %.0f ms (device=%s tiled=%s conf=%.2f)",
        len(batch.detections),
        elapsed_ms,
        settings.device,
        tiled,
        conf,
    )

    request = AiInferenceRequest.model_validate(
        {
            "requestId": "local-test",
            "taskId": "local-task",
            "deviceId": "local-device",
            "algorithmTypes": algorithm_types or ["LOCAL"],
            "bucket": "local",
            "imageObjectKey": image_path.name,
            "timestamp": int(time.time() * 1000),
        }
    )
    result = build_result(request, batch.detections, stage="yolo", partial=False)
    print(json.dumps(result.to_kafka_dict(), ensure_ascii=False, indent=2))

    if args.output:
        vis = image_bgr.copy()
        _draw(vis, batch.detections)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        log.info("Wrote annotated image: %s", out.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

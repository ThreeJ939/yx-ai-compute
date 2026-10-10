#!/usr/bin/env python3
"""Local smoke test for dedicated fire/smoke YOLO weights.

Supports optional sliding-window tiling (same helper as YOLO-World) for
distant / small fire and smoke on large still images.

Does not use Kafka / MinIO / YOLO-World.

Usage (from repo root, with venv activated):

  python scripts/test_fire_smoke.py path/to/image.jpg
  python scripts/test_fire_smoke.py path/to/image.jpg --tiled --save
  python scripts/test_fire_smoke.py path/to/image.jpg --no-tiled -o out.jpg
  python scripts/test_fire_smoke.py path/to/video.mp4 --conf 0.2 --device cpu
  python scripts/test_fire_smoke.py path/to/dir --weights weights/fire-smoke-yolov8n.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_WEIGHTS = "weights/fire-smoke-yolov8n-2.pt"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test fire/smoke YOLOv8 weights (Ultralytics YOLO, not YOLO-World)",
    )
    p.add_argument(
        "source",
        type=str,
        help="Image, video path, or directory of images",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=_DEFAULT_WEIGHTS,
        help=f"Path to .pt weights (default: {_DEFAULT_WEIGHTS})",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda / 0")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sliding-window tiling for large still images (default: on; use --no-tiled to disable)",
    )
    p.add_argument("--tile-size", type=int, default=640, help="Tile edge length in pixels")
    p.add_argument(
        "--tile-overlap",
        type=float,
        default=0.2,
        help="Tile overlap ratio 0–1 (default: 0.2, same as YOLO_TILE_OVERLAP)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results under runs/detect/predict* (full-frame path only)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Optional annotated output path (jpg/png) for a single image",
    )
    return p.parse_args()


def _resolve_weights(path_str: str) -> Path | None:
    weights = Path(path_str)
    if weights.is_file():
        return weights
    alt = _ROOT / path_str
    if alt.is_file():
        return alt
    return None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES


def _detect_full(
    model: Any,
    image_bgr: np.ndarray,
    *,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
) -> list[dict[str, Any]]:
    result = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
        verbose=False,
    )[0]
    return _result_to_dets(result, model.names)


def _detect_tiled(
    model: Any,
    image_bgr: np.ndarray,
    *,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    tile_size: int,
    tile_overlap: float,
) -> list[dict[str, Any]]:
    """Sliding-window detect + per-class NMS (aligned with YoloWorldDetector)."""
    import torch  # noqa: PLC0415
    from torchvision.ops import nms  # noqa: PLC0415

    from app.detectors.tiling import iter_tiles  # noqa: PLC0415

    h, w = image_bgr.shape[:2]
    all_boxes: list[Any] = []
    all_scores: list[Any] = []
    all_cls: list[Any] = []
    names: dict[int, str] = dict(model.names or {})
    tile_count = 0

    for x1, y1 in iter_tiles(w, h, tile=tile_size, overlap=tile_overlap):
        tile_count += 1
        x2, y2 = x1 + tile_size, y1 + tile_size
        crop = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        region = image_bgr[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)]
        crop[: region.shape[0], : region.shape[1]] = region

        result = model.predict(
            source=crop,
            conf=conf,
            device=device,
            imgsz=imgsz,
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

    print(f"  tiled: size={w}x{h} tile={tile_size} overlap={tile_overlap} tiles={tile_count}")

    if not all_boxes:
        return []

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

    out: list[dict[str, Any]] = []
    for (xa, ya, xb, yb), score, cid in zip(
        boxes.tolist(), scores.tolist(), clses.tolist()
    ):
        cls_id = int(cid)
        label = str(names.get(cls_id, cls_id))
        out.append(
            {
                "label": label,
                "conf": float(score),
                "xyxy": [float(xa), float(ya), float(xb), float(yb)],
            }
        )
    return out


def _result_to_dets(result: Any, fallback_names: Any) -> list[dict[str, Any]]:
    names = result.names or fallback_names or {}
    if result.boxes is None or len(result.boxes) == 0:
        return []
    out: list[dict[str, Any]] = []
    for b in result.boxes:
        cls_id = int(b.cls.item())
        label = str(names.get(cls_id, cls_id))
        conf = float(b.conf.item())
        xyxy = [float(x) for x in b.xyxy[0].tolist()]
        out.append({"label": label, "conf": conf, "xyxy": xyxy})
    return out


def _print_dets(path: str, dets: list[dict[str, Any]]) -> None:
    print(f"{path} -> {len(dets)} boxes")
    for d in dets:
        xyxy = [round(v, 1) for v in d["xyxy"]]
        print(f"  {d['label']}  conf={d['conf']:.3f}  xyxy={xyxy}")


def _draw_dets(image_bgr: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    import cv2  # noqa: PLC0415

    img = image_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = (0, 0, 255) if "fire" in d["label"].lower() else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['conf']:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def _run_image(
    model: Any,
    image_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import cv2  # noqa: PLC0415

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"ERROR: cannot decode image: {image_path}", file=sys.stderr)
        return []

    h, w = image_bgr.shape[:2]
    use_tiles = (
        args.tiled
        and args.tile_size > 0
        and (w > args.tile_size or h > args.tile_size)
    )
    if use_tiles:
        print(f"mode=tiled image={w}x{h}")
        dets = _detect_tiled(
            model,
            image_bgr,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
    else:
        reason = "disabled" if not args.tiled else f"image {w}x{h} <= tile {args.tile_size}"
        print(f"mode=full ({reason})")
        dets = _detect_full(
            model,
            image_bgr,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
        )

    _print_dets(str(image_path), dets)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        annotated = _draw_dets(image_bgr, dets)
        if not cv2.imwrite(str(out), annotated):
            print(f"ERROR: failed to write {out}", file=sys.stderr)
        else:
            print(f"Wrote annotated image: {out.resolve()}")
    elif args.save and use_tiles:
        # Ultralytics --save path only for model.predict(source=file); write ourselves
        out_dir = _ROOT / "runs" / "detect" / "predict_tiled"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / image_path.name
        annotated = _draw_dets(image_bgr, dets)
        if cv2.imwrite(str(out), annotated):
            print(f"Wrote annotated image: {out.resolve()}")

    return dets


def _run_ultralytics_source(model: Any, source: Path, args: argparse.Namespace) -> int:
    """Video / directory / non-tiled batch via Ultralytics (no custom tiling)."""
    if args.tiled:
        print(
            "NOTE: sliding-window tiling applies to single images only; "
            "using full-frame predict for this source.",
            file=sys.stderr,
        )
    results = model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        save=args.save,
        verbose=False,
    )
    total = 0
    for i, r in enumerate(results):
        dets = _result_to_dets(r, model.names)
        total += len(dets)
        path = getattr(r, "path", None) or str(source)
        print(f"[{i}] ", end="")
        _print_dets(str(path), dets)
    if args.save:
        print("Annotated outputs saved under runs/detect/ (Ultralytics default)")
    return total


def main() -> int:
    args = _parse_args()
    weights = _resolve_weights(args.weights)
    if weights is None:
        print(
            f"ERROR: weights not found: {Path(args.weights).resolve()}\n"
            f"Place best.pt as {_ROOT / _DEFAULT_WEIGHTS}\n"
            "Download from: "
            "https://github.com/luminous0219/fire-and-smoke-detection-yolov8/tree/main/weights",
            file=sys.stderr,
        )
        return 1

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: source not found: {source.resolve()}", file=sys.stderr)
        return 1

    from ultralytics import YOLO  # noqa: PLC0415

    print(f"Loading weights={weights.resolve()} device={args.device}")
    model = YOLO(str(weights))
    print(f"classes: {model.names}")

    if _is_image_file(source):
        dets = _run_image(model, source, args)
        print(f"Done. total_boxes={len(dets)}")
        return 0

    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if _is_image_file(p)
        )
        if not images:
            print(f"ERROR: no images in {source}", file=sys.stderr)
            return 1
        total = 0
        for img in images:
            # -o only applies to single image; clear for batch
            batch_args = argparse.Namespace(**vars(args))
            batch_args.output = ""
            dets = _run_image(model, img, batch_args)
            total += len(dets)
        print(f"Done. images={len(images)} total_boxes={total}")
        return 0

    # video or other ultralytics-supported source
    total = _run_ultralytics_source(model, source, args)
    print(f"Done. total_boxes={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

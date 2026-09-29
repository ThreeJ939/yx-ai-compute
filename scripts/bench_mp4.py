#!/usr/bin/env python3
"""Benchmark YOLO-World on a local MP4 (no Kafka / MinIO).

Usage (from repo root, with .env / weights ready):

  python -m scripts.bench_mp4 path/to/video.mp4
  python -m scripts.bench_mp4 video.mp4 --sample-every 25 --no-tile
  python -m scripts.bench_mp4 video.mp4 --interval-ms 2000 -a DET_PERSON --max-infer 30
  python -m scripts.bench_mp4 video.mp4 --preview-dir /tmp/bench_preview --max-infer 5

Reads DEVICE / YOLO_* from .env. First infer includes model load; default
--warmup 1 excludes it from avg/p50/p95.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark YOLO-World on local MP4")
    p.add_argument("video", type=str, help="Path to input MP4 (or any OpenCV-readable video)")
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
        help="Optional raw prompts; overrides -a for YOLO classes",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="Infer every N decoded frames (0 = use --interval-ms instead)",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=2000.0,
        help="Infer about every N ms of video timeline when --sample-every is 0 "
        "(default 2000, similar to recognition grab interval)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after decoding this many frames (0 = entire video)",
    )
    p.add_argument(
        "--max-infer",
        type=int,
        default=50,
        help="Stop after this many inferences (default 50; 0 = no limit)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Exclude first N inferences from summary stats (model load)",
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
        help="Disable sliding-window tiling",
    )
    p.add_argument(
        "--preview-dir",
        type=str,
        default="",
        help="Optional directory to save annotated frames for each infer",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging + per-infer lines",
    )
    return p.parse_args()


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("bench_mp4")

    video_path = Path(args.video)
    if not video_path.is_file():
        log.error("Video not found: %s", video_path)
        return 1

    settings = Settings()
    conf = args.conf if args.conf is not None else settings.conf_threshold
    tiled = False if args.no_tile else settings.yolo_tiled

    algorithm_types = [c.strip() for c in args.algorithms.split(",") if c.strip()]
    if args.prompts.strip():
        prompts = [c.strip() for c in args.prompts.split(",") if c.strip()]
        log.info("Using raw prompts=%s", prompts)
    else:
        plan = plan_request(algorithm_types)
        prompts = plan.prompts or collect_prompts(algorithm_types)
        if plan.unknown:
            log.warning("Unknown algorithm codes: %s", plan.unknown)
        log.info("algorithms=%s prompts=%s", algorithm_types, prompts)

    if not prompts:
        log.error("No prompts to detect; check -a / -p")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Failed to open video: %s", video_path)
        return 1

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    log.info(
        "video=%s size=%sx%s fps=%.2f frames≈%s device=%s tiled=%s conf=%.2f",
        video_path.name,
        width,
        height,
        src_fps,
        frame_count_meta,
        settings.device,
        tiled,
        conf,
    )

    sample_every = args.sample_every
    if sample_every <= 0:
        if src_fps > 1e-3 and args.interval_ms > 0:
            sample_every = max(1, int(round(src_fps * (args.interval_ms / 1000.0))))
            log.info(
                "sample-every=%s (from interval-ms=%.0f @ %.2f fps)",
                sample_every,
                args.interval_ms,
                src_fps,
            )
        else:
            sample_every = 1
            log.info("sample-every=1 (fps unknown or interval disabled)")

    detector = YoloWorldDetector(
        weights=settings.yolo_world_weights,
        device=settings.device,
        clip_weights=settings.yolo_clip_weights,
        tiled=tiled,
        tile_size=settings.yolo_tile_size,
        tile_overlap=settings.yolo_tile_overlap,
        imgsz=settings.yolo_imgsz,
    )

    preview_dir: Path | None = None
    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    infer_times_ms: list[float] = []
    box_counts: list[int] = []
    decoded = 0
    inferred = 0
    wall_t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        if args.max_frames > 0 and decoded > args.max_frames:
            break

        if decoded % sample_every != 0:
            continue

        t0 = time.perf_counter()
        batch = detector.detect(
            frame,
            prompts=prompts,
            conf=conf,
            iou=settings.iou_threshold,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        inferred += 1
        infer_times_ms.append(elapsed_ms)
        box_counts.append(len(batch.detections))

        if args.verbose or inferred <= args.warmup + 1:
            log.info(
                "infer#%s frame=%s boxes=%s time=%.0f ms",
                inferred,
                decoded,
                len(batch.detections),
                elapsed_ms,
            )

        if preview_dir is not None:
            vis = frame.copy()
            _draw(vis, batch.detections)
            out = preview_dir / f"infer_{inferred:04d}_frame_{decoded:06d}.jpg"
            cv2.imwrite(str(out), vis)

        if args.max_infer > 0 and inferred >= args.max_infer:
            break

    cap.release()
    wall_s = time.perf_counter() - wall_t0

    if not infer_times_ms:
        log.error("No inferences ran; check video / sample settings")
        return 1

    warmup = max(0, min(args.warmup, len(infer_times_ms) - 1))
    stats_times = infer_times_ms[warmup:]
    if not stats_times:
        stats_times = infer_times_ms

    sorted_t = sorted(stats_times)
    avg = statistics.fmean(stats_times)
    p50 = _percentile(sorted_t, 50)
    p95 = _percentile(sorted_t, 95)
    eff_fps = 1000.0 / avg if avg > 0 else 0.0
    target_interval_ms = (
        (sample_every / src_fps) * 1000.0 if src_fps > 1e-3 else args.interval_ms
    )
    keep_up = avg < target_interval_ms if target_interval_ms > 0 else None

    print()
    print("=== bench_mp4 summary ===")
    print(f"video:           {video_path}")
    print(f"device/tiled:    {settings.device} / {tiled}")
    print(f"decoded_frames:  {decoded}")
    print(f"inferences:      {inferred} (warmup excluded from stats: {warmup})")
    print(f"sample_every:    {sample_every}")
    print(f"wall_time_s:     {wall_s:.2f}")
    print(f"infer_ms avg:    {avg:.1f}")
    print(f"infer_ms p50:    {p50:.1f}")
    print(f"infer_ms p95:    {p95:.1f}")
    print(f"infer_ms min/max:{min(stats_times):.1f} / {max(stats_times):.1f}")
    print(f"effective_fps:   {eff_fps:.2f}  (1000/avg_ms)")
    if src_fps > 1e-3:
        print(f"source_fps:      {src_fps:.2f}")
    print(f"target_interval: {target_interval_ms:.0f} ms (approx business gap)")
    if keep_up is not None:
        print(
            f"keep_up:         {'YES' if keep_up else 'NO'} "
            f"(avg infer {'<' if keep_up else '>='} target interval)"
        )
    if box_counts:
        print(
            f"boxes avg:       {statistics.fmean(box_counts[warmup:] or box_counts):.1f}"
        )
    print("=========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

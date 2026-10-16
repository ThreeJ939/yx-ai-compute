"""Entry point: FastAPI health server + optional Kafka consumer loop."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

from pathlib import Path

from app.api import create_app
from app.config import Settings, get_settings
from app.detectors.yolo_world import YoloWorldDetector
from app.enhancers.noop import NoopBehaviorEnhancer
from app.kafka_app import KafkaInferenceApp
from app.minio_client import MinioImageStore
from app.orchestrator import Orchestrator


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


def _resolve_weights_path(path_str: str) -> Path | None:
    """Resolve weights against CWD or repo root (app/../)."""
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    root_relative = Path(__file__).resolve().parents[1] / path_str
    if root_relative.is_file():
        return root_relative.resolve()
    return None


def _build_detector_registry(settings: Settings, logger: logging.Logger) -> dict:
    """Assemble backend → Detector map (world always; fire_smoke/dredge optional)."""
    from app.detectors.base import Detector  # noqa: PLC0415

    detectors: dict[str, Detector] = {
        "world": YoloWorldDetector(
            weights=settings.yolo_world_weights,
            device=settings.device,
            clip_weights=settings.yolo_clip_weights,
            tiled=settings.yolo_tiled,
            tile_size=settings.yolo_tile_size,
            tile_overlap=settings.yolo_tile_overlap,
            tile_batch=settings.yolo_tile_batch,
            imgsz=settings.yolo_imgsz,
        )
    }

    if settings.fire_smoke_enabled:
        weights_path = _resolve_weights_path(settings.fire_smoke_weights)
        if weights_path is None:
            logger.warning(
                "FIRE_SMOKE_ENABLED=true but weights missing: %s — skipping fire_smoke backend",
                settings.fire_smoke_weights,
            )
        else:
            from app.detectors.ultralytics_yolo import UltralyticsYoloDetector  # noqa: PLC0415

            detectors["fire_smoke"] = UltralyticsYoloDetector(
                weights=str(weights_path),
                device=settings.device,
                tiled=settings.fire_smoke_tiled,
                tile_size=settings.fire_smoke_tile_size,
                tile_overlap=settings.fire_smoke_tile_overlap,
                tile_batch=settings.fire_smoke_tile_batch,
                imgsz=settings.yolo_imgsz,
                name="fire_smoke",
            )
            logger.info("fire_smoke backend enabled weights=%s", weights_path)
    else:
        logger.info("fire_smoke backend disabled (FIRE_SMOKE_ENABLED=false)")

    if settings.dredge_enabled:
        dredge_path = _resolve_weights_path(settings.dredge_weights)
        if dredge_path is None:
            logger.warning(
                "DREDGE_ENABLED=true but weights missing: %s — skipping dredge backend",
                settings.dredge_weights,
            )
        else:
            from app.detectors.ultralytics_yolo import UltralyticsYoloDetector  # noqa: PLC0415

            detectors["dredge"] = UltralyticsYoloDetector(
                weights=str(dredge_path),
                device=settings.device,
                tiled=settings.dredge_tiled,
                tile_size=settings.dredge_tile_size,
                tile_overlap=settings.dredge_tile_overlap,
                tile_batch=settings.dredge_tile_batch,
                imgsz=settings.dredge_imgsz,
                name="dredge",
            )
            logger.info("dredge backend enabled weights=%s", dredge_path)
    else:
        logger.info("dredge backend disabled (DREDGE_ENABLED=false)")

    logger.info("Detector backends registered: %s", sorted(detectors.keys()))
    return detectors


async def _async_main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("app.main")

    minio = MinioImageStore(settings)
    if settings.llm_enabled:
        from app.enhancers.llm_vlm import LlmBehaviorEnhancer

        enhancer = LlmBehaviorEnhancer(settings)
        logger.info(
            "LLM enhancer enabled model=%s prompts=%s",
            settings.llm_model,
            settings.llm_prompts_path,
        )
    else:
        enhancer = NoopBehaviorEnhancer()
        logger.info("LLM enhancer disabled (NoopBehaviorEnhancer)")

    detector_registry = _build_detector_registry(settings, logger)
    orchestrator = Orchestrator(
        settings,
        minio,
        detector_registry=detector_registry,
        enhancer=enhancer,
    )

    async def process(request):
        return await orchestrator.process(request)

    kafka_app: KafkaInferenceApp | None = None
    if settings.kafka_enabled:
        kafka_app = KafkaInferenceApp(settings, process)
    else:
        logger.warning("KAFKA_ENABLED=false — HTTP/preview only, no Kafka consumer")

    api = create_app(settings, orchestrator=orchestrator)

    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=settings.health_port,
        log_level=settings.log_level.lower(),
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    stop_event = asyncio.Event()

    def _ask_stop(*_args) -> None:
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _ask_stop)
            except NotImplementedError:
                # Windows: signal handlers in asyncio are limited
                signal.signal(sig, lambda *_: _ask_stop())
    except Exception:
        logger.debug("Signal handler setup skipped", exc_info=True)

    wait_set: set[asyncio.Task] = set()

    if kafka_app is not None:
        await kafka_app.start()

    logger.info(
        "yx-ai-compute started serviceId=%s health_port=%s device=%s kafka=%s",
        settings.service_id,
        settings.health_port,
        settings.device,
        settings.kafka_enabled,
    )

    if kafka_app is not None:
        wait_set.add(asyncio.create_task(kafka_app.run_forever(), name="kafka-consumer"))

    wait_set.add(asyncio.create_task(server.serve(), name="uvicorn"))
    stopper = asyncio.create_task(stop_event.wait(), name="stop-wait")
    wait_set.add(stopper)

    done, pending = await asyncio.wait(
        wait_set,
        return_when=asyncio.FIRST_COMPLETED,
    )

    logger.info("Shutting down...")
    server.should_exit = True
    if kafka_app is not None:
        kafka_app.request_stop()

    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if kafka_app is not None:
        await kafka_app.stop()
    orchestrator.close()

    for task in done:
        if task is stopper:
            continue
        exc = task.exception() if not task.cancelled() else None
        if exc:
            logger.error("Task failed: %s", exc)


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

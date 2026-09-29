"""Entry point: FastAPI health server + optional Kafka consumer loop."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

from app.api import create_app
from app.config import get_settings
from app.kafka_app import KafkaInferenceApp
from app.minio_client import MinioImageStore
from app.orchestrator import Orchestrator


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


async def _async_main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("app.main")

    minio = MinioImageStore(settings)
    orchestrator = Orchestrator(settings, minio)

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

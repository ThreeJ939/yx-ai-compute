"""Kafka consumer/producer for inference request/result topics."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from app.config import Settings
from app.schemas import AiInferenceRequest, AiInferenceResult

logger = logging.getLogger(__name__)

ProcessFn = Callable[[AiInferenceRequest], Awaitable[AiInferenceResult | None]]


class KafkaInferenceApp:
    def __init__(self, settings: Settings, process_fn: ProcessFn) -> None:
        self._settings = settings
        self._process_fn = process_fn
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._running = False

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self._settings.kafka_bootstrap_servers,
        }
        if self._settings.kafka_has_sasl():
            kwargs["security_protocol"] = self._settings.kafka_security_protocol
            if self._settings.kafka_security_protocol == "PLAINTEXT":
                kwargs["security_protocol"] = "SASL_PLAINTEXT"
            kwargs["sasl_mechanism"] = self._settings.kafka_sasl_mechanism
            kwargs["sasl_plain_username"] = self._settings.kafka_username
            kwargs["sasl_plain_password"] = self._settings.kafka_password
        else:
            kwargs["security_protocol"] = self._settings.kafka_security_protocol
        return kwargs

    async def start(self) -> None:
        common = self._client_kwargs()
        self._producer = AIOKafkaProducer(
            **common,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        self._consumer = AIOKafkaConsumer(
            self._settings.kafka_request_topic,
            **common,
            group_id=self._settings.kafka_group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._producer.start()
        await self._consumer.start()
        self._running = True
        logger.info(
            "Kafka started: consume=%s produce=%s group=%s",
            self._settings.kafka_request_topic,
            self._settings.kafka_result_topic,
            self._settings.kafka_group_id,
        )

    async def stop(self) -> None:
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
        logger.info("Kafka stopped")

    def request_stop(self) -> None:
        self._running = False

    async def publish_result(self, result: AiInferenceResult) -> None:
        if self._producer is None:
            raise RuntimeError("Producer not started")
        payload = result.to_kafka_dict()
        await self._producer.send_and_wait(
            self._settings.kafka_result_topic,
            value=payload,
            key=result.task_id,
        )
        logger.info(
            "Published result requestId=%s taskId=%s slots=%s",
            result.request_id,
            result.task_id,
            len(result.algorithm_type),
        )

    async def run_forever(self) -> None:
        if self._consumer is None:
            raise RuntimeError("Consumer not started")
        assert self._consumer is not None
        while self._running:
            try:
                batch = await self._consumer.getmany(timeout_ms=1000, max_records=10)
            except asyncio.CancelledError:
                raise
            except KafkaError:
                logger.exception("Kafka poll error")
                await asyncio.sleep(1)
                continue

            for _tp, messages in batch.items():
                for msg in messages:
                    await self._handle_message(msg)

    async def _handle_message(self, msg: Any) -> None:
        assert self._consumer is not None
        raw = msg.value
        try:
            if isinstance(raw, (bytes, bytearray)):
                text = raw.decode("utf-8")
            else:
                text = str(raw)
            data = json.loads(text)
            request = AiInferenceRequest.model_validate(data)
        except Exception:
            logger.exception(
                "Poison message, skip commit offset=%s partition=%s",
                msg.offset,
                msg.partition,
            )
            await self._consumer.commit()
            return

        if not request.required_ok():
            logger.warning(
                "Invalid request missing fields, skip requestId=%s",
                getattr(request, "request_id", None),
            )
            await self._consumer.commit()
            return

        try:
            result = await self._process_fn(request)
            if result is not None:
                await self.publish_result(result)
            await self._consumer.commit()
        except KafkaError:
            logger.exception(
                "Transient Kafka error, will retry requestId=%s",
                request.request_id,
            )
            # do not commit — retry
        except Exception:
            # Transient failures (e.g. MinIO). Orchestrator returns zero-slot
            # AiInferenceResult for infer/decode errors instead of raising.
            logger.exception(
                "Transient processing error, will retry requestId=%s",
                request.request_id,
            )
            # do not commit — retry

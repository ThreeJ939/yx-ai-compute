"""Pydantic models aligned with yx-ai-recognition Kafka DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AiInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(alias="requestId")
    task_id: str = Field(alias="taskId")
    device_id: str = Field(alias="deviceId")
    algorithm_types: list[str] = Field(default_factory=list, alias="algorithmTypes")
    bucket: str | None = None
    image_object_key: str | None = Field(default=None, alias="imageObjectKey")
    timestamp: int | None = None
    stream_timestamp_us: int | None = Field(default=None, alias="streamTimestampUs")
    captured_at_ms: int | None = Field(default=None, alias="capturedAtMs")

    @field_validator("algorithm_types", mode="before")
    @classmethod
    def _coerce_algorithm_types(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def required_ok(self) -> bool:
        return bool(
            self.request_id
            and self.request_id.strip()
            and self.task_id
            and self.task_id.strip()
            and self.device_id
            and self.device_id.strip()
            and self.bucket
            and self.bucket.strip()
            and self.image_object_key
            and self.image_object_key.strip()
        )


class DetectionItem(BaseModel):
    """Aligned with AiDetectionItem; nested under AlgorithmResult."""

    model_config = ConfigDict(populate_by_name=True)

    det_id: str = Field(alias="detId")
    object_type: str = Field(alias="objectType")
    confidence: float
    bbox: list[float]
    label: str
    behavior: str | None = None


class AlgorithmResult(BaseModel):
    """One algorithm slot in the nested Kafka result."""

    model_config = ConfigDict(populate_by_name=True)

    algorithm_type: str = Field(alias="algorithmType")
    algorithm_num: int = Field(alias="algorithmNum")
    image_url: str = Field(default="", alias="imageUrl")
    detections: list[DetectionItem] = Field(default_factory=list)


class AiInferenceResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    task_id: str = Field(alias="taskId")
    device_id: str = Field(alias="deviceId")
    timestamp: int | None = None
    results: list[AlgorithmResult] = Field(default_factory=list)

    def to_kafka_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

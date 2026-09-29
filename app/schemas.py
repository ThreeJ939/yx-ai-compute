"""Pydantic models aligned with yx-ai-recognition Kafka DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    """Aligned with AiDetectionItem; extended with detId for future two-phase merge."""

    model_config = ConfigDict(populate_by_name=True)

    det_id: str = Field(alias="detId")
    object_type: str = Field(alias="objectType")
    confidence: float
    bbox: list[float]
    label: str
    algorithm_type: str | None = Field(default=None, alias="algorithmType")
    behavior: str | None = None


class AiInferenceResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    task_id: str = Field(alias="taskId")
    device_id: str = Field(alias="deviceId")
    algorithm_type: list[str] = Field(default_factory=list, alias="algorithmType")
    image_url: list[str] = Field(default_factory=list, alias="imageUrl")
    algorithm_num: list[int] = Field(default_factory=list, alias="algorithmNum")
    timestamp: int | None = None
    detections: list[DetectionItem] = Field(default_factory=list)
    stage: str | None = None
    partial: bool | None = None

    @model_validator(mode="after")
    def _slots_aligned(self) -> AiInferenceResult:
        n = len(self.algorithm_type)
        if len(self.image_url) != n or len(self.algorithm_num) != n:
            raise ValueError(
                "algorithmType, imageUrl, algorithmNum must have the same length"
            )
        return self

    def to_kafka_dict(self) -> dict[str, Any]:
        payload = self.model_dump(by_alias=True, exclude_none=True)
        # Always include detections list for forward compatibility
        payload["detections"] = [
            d.model_dump(by_alias=True, exclude_none=True) for d in self.detections
        ]
        return payload

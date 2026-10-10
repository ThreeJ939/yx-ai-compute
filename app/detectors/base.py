"""Detector abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RawDetection:
    object_type: str
    label: str
    confidence: float
    # xywh pixel coordinates
    bbox: list[float]
    prompt: str = ""
    # Optional LLM/VLM behavior label (e.g. fishing / fighting)
    behavior: str | None = None


@dataclass
class DetectBatchResult:
    detections: list[RawDetection] = field(default_factory=list)


class Detector(Protocol):
    def detect(
        self,
        image_bgr: Any,
        prompts: list[str],
        conf: float,
        iou: float,
    ) -> DetectBatchResult: ...

"""Behavior enhancer extension points (phase-1 noop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.detectors.base import RawDetection


@dataclass
class EnhanceRequest:
    behavior: str
    image_bgr: Any
    detections: list[RawDetection]
    algorithm_code: str


@dataclass
class EnhanceResult:
    detections: list[RawDetection]
    applied: bool = False
    # Preview debug: crops sent to VLM (image_b64 + metadata)
    crops: list[dict[str, Any]] = field(default_factory=list)


class BehaviorEnhancer(Protocol):
    def enhance(self, request: EnhanceRequest) -> EnhanceResult: ...

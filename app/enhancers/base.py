"""Behavior enhancer extension points (phase-1 noop)."""

from __future__ import annotations

from dataclasses import dataclass
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


class BehaviorEnhancer(Protocol):
    def enhance(self, request: EnhanceRequest) -> EnhanceResult: ...

"""No-op behavior enhancer for phase-1."""

from __future__ import annotations

from app.enhancers.base import EnhanceRequest, EnhanceResult


class NoopBehaviorEnhancer:
    def enhance(self, request: EnhanceRequest) -> EnhanceResult:
        return EnhanceResult(detections=list(request.detections), applied=False)

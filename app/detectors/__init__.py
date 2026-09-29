"""Detector package — import concrete classes from submodules to avoid heavy deps."""

from app.detectors.base import DetectBatchResult, RawDetection

__all__ = [
    "DetectBatchResult",
    "RawDetection",
]

"""LLM / VLM behavior enhancer (OpenAI-compatible vision)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import Settings
from app.detectors.base import RawDetection
from app.enhancers.base import EnhanceRequest, EnhanceResult
from app.enhancers.crop_util import CropItem, crop_for_behavior
from app.enhancers.llm_client import LlmClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str
    crop: str = "single_person_expand"
    crop_padding: float = 0.2


def load_llm_prompts(path: str | Path) -> dict[str, PromptTemplate]:
    """Load algorithm_code → PromptTemplate from YAML."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"LLM prompts file not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"LLM prompts root must be a mapping: {p}")

    out: dict[str, PromptTemplate] = {}
    for code, raw in data.items():
        if not isinstance(raw, dict):
            continue
        system = str(raw.get("system") or "").strip()
        user = str(raw.get("user") or "").strip()
        if not system or not user:
            logger.warning("Skipping LLM prompt %s: missing system/user", code)
            continue
        padding = raw.get("crop_padding", 0.2)
        try:
            padding_f = float(padding)
        except (TypeError, ValueError):
            padding_f = 0.2
        out[str(code).strip()] = PromptTemplate(
            system=system,
            user=user,
            crop=str(raw.get("crop") or "single_person_expand").strip(),
            crop_padding=padding_f,
        )
    return out


def apply_verdicts(
    detections: list[RawDetection],
    crops: list[CropItem],
    verdicts: list[dict[str, Any]],
    conf_threshold: float,
) -> list[RawDetection]:
    """Copy detections and set behavior when VLM confirms (conservative: never drop boxes)."""
    out = [copy.copy(d) for d in detections]
    for crop, verdict in zip(crops, verdicts):
        if not verdict.get("positive"):
            continue
        try:
            conf = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < conf_threshold:
            continue
        label = str(verdict.get("label") or "").strip()
        if not label:
            continue
        for idx in crop.det_indices:
            if 0 <= idx < len(out):
                out[idx].behavior = label
                # Prefer VLM confidence for confirmed behavior detections
                out[idx].confidence = conf
    return out


class LlmBehaviorEnhancer:
    """YOLO person boxes → crop → VLM → optional behavior labels."""

    def __init__(
        self,
        settings: Settings,
        client: LlmClient | None = None,
        prompts: dict[str, PromptTemplate] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or LlmClient(settings)
        if prompts is not None:
            self._prompts = prompts
        else:
            self._prompts = load_llm_prompts(settings.llm_prompts_path)
        logger.info(
            "LlmBehaviorEnhancer ready prompts=%s model=%s",
            list(self._prompts.keys()),
            settings.llm_model,
        )

    def enhance(self, request: EnhanceRequest) -> EnhanceResult:
        code = (request.algorithm_code or request.behavior or "").strip()
        template = self._prompts.get(code)
        if template is None:
            logger.warning("No LLM prompt template for algorithm_code=%s", code)
            return EnhanceResult(detections=list(request.detections), applied=False)

        crops = crop_for_behavior(
            request.image_bgr,
            request.detections,
            crop_mode=template.crop,
            padding=template.crop_padding,
        )
        if not crops:
            logger.info("LLM enhance skip code=%s: no matching crops", code)
            return EnhanceResult(detections=list(request.detections), applied=False)

        crops_b64: list[str] = []
        try:
            crops_b64 = [
                self._client.encode_crop_jpeg_b64(c.image_bgr) for c in crops
            ]
            verdicts = self._client.call(
                crops_b64,
                template.system,
                template.user,
                max_workers=self._settings.llm_crop_concurrency,
            )
        except Exception:
            logger.exception("LLM enhance failed code=%s; keeping YOLO detections", code)
            return EnhanceResult(
                detections=list(request.detections),
                applied=False,
                crops=_crop_preview_items(code, crops, crops_b64, verdicts=None),
            )

        if len(verdicts) != len(crops):
            logger.error(
                "LLM verdict count mismatch code=%s crops=%s verdicts=%s",
                code,
                len(crops),
                len(verdicts),
            )
            return EnhanceResult(
                detections=list(request.detections),
                applied=False,
                crops=_crop_preview_items(code, crops, crops_b64, verdicts=None),
            )

        out = apply_verdicts(
            request.detections,
            crops,
            verdicts,
            self._settings.llm_conf_threshold,
        )
        applied = any(d.behavior for d in out)
        logger.info(
            "LLM enhance done code=%s crops=%s applied=%s behaviors=%s",
            code,
            len(crops),
            applied,
            [d.behavior for d in out if d.behavior],
        )
        return EnhanceResult(
            detections=out,
            applied=applied,
            crops=_crop_preview_items(code, crops, crops_b64, verdicts=verdicts),
        )


def _crop_preview_items(
    algorithm_code: str,
    crops: list[CropItem],
    crops_b64: list[str],
    verdicts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build preview payloads for crops sent (or about to be sent) to VLM."""
    out: list[dict[str, Any]] = []
    for i, crop in enumerate(crops):
        item: dict[str, Any] = {
            "algorithm_code": algorithm_code,
            "det_indices": list(crop.det_indices),
            "bbox_xywh": [round(float(v), 1) for v in crop.bbox_xywh],
        }
        if i < len(crops_b64):
            item["image_b64"] = crops_b64[i]
        if verdicts is not None and i < len(verdicts):
            v = verdicts[i]
            item["positive"] = bool(v.get("positive"))
            try:
                item["confidence"] = round(float(v.get("confidence", 0.0)), 4)
            except (TypeError, ValueError):
                item["confidence"] = 0.0
            item["label"] = str(v.get("label") or "").strip()
        out.append(item)
    return out

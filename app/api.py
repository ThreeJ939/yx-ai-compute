"""HTTP endpoints: health, capabilities, and live preview."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app import __version__
from app.config import Settings
from app.preview_util import (
    decode_image_bytes,
    detections_to_json,
    draw_bboxes,
    encode_jpeg_b64,
    resolve_prompts,
)
from app.preview_ws import preview_stream_endpoint
from app.registry import get_spec, list_capabilities, plan_request

if TYPE_CHECKING:
    from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_PREVIEW_HTML_PATH = _STATIC_DIR / "preview.html"


def _load_preview_html() -> str:
    return _PREVIEW_HTML_PATH.read_text(encoding="utf-8")


def create_app(settings: Settings, orchestrator: "Orchestrator | None" = None) -> FastAPI:
    app = FastAPI(
        title="yx-ai-compute",
        version=__version__,
        description="AI compute service — YOLO-World inference over Kafka",
    )

    @app.get("/")
    def root() -> RedirectResponse:
        if settings.preview_enabled:
            return RedirectResponse(url="/preview")
        return RedirectResponse(url="/health")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "UP",
            "serviceId": settings.service_id,
            "version": __version__,
        }

    @app.get("/capabilities")
    def capabilities() -> dict:
        return {
            "serviceId": settings.service_id,
            "version": __version__,
            "algorithms": list_capabilities(),
        }

    if not settings.preview_enabled:
        return app

    if orchestrator is None:
        logger.warning(
            "preview_enabled=true but no Orchestrator passed to create_app(); "
            "preview endpoints will return 503"
        )

    @app.get("/preview", response_class=HTMLResponse, include_in_schema=False)
    def preview_page() -> str:
        try:
            return _load_preview_html()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="preview.html missing") from exc

    @app.post("/preview/image")
    async def preview_image(
        image: UploadFile = File(..., description="要识别的图片文件"),
        algorithms: Optional[str] = Form(
            None,
            description="算法代码，逗号分隔（如 DET_PERSON,DET_SHIP）",
        ),
        prompts: Optional[str] = Form(
            None,
            description="额外提示词，逗号分隔（可叠加在算法提示词上）",
        ),
        conf: Optional[float] = Form(None, description="置信度阈值，覆盖服务默认值"),
        iou: Optional[float] = Form(None, description="IOU 阈值，覆盖服务默认值"),
    ) -> JSONResponse:
        if orchestrator is None:
            raise HTTPException(
                status_code=503,
                detail="Preview unavailable: orchestrator not initialised",
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        # Parse algorithm codes (needed for multi-backend detect + LLM routing)
        algo_codes = [a.strip() for a in (algorithms or "").split(",") if a.strip()]
        merged_prompts = resolve_prompts(algorithms, prompts)
        # Dedicated backends (e.g. fire_smoke) may have empty world prompts.
        if not algo_codes and not merged_prompts:
            raise HTTPException(
                status_code=400,
                detail="No algorithms/prompts resolved; check algorithms / prompts params",
            )
        if algo_codes:
            from app.registry import plan_request  # noqa: PLC0415

            plan = plan_request(algo_codes)
            if not plan.known and not merged_prompts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown algorithms: {plan.unknown}",
                )

        # ── Multi-backend detection ───────────────────────────────────────────
        try:
            batch = await orchestrator.detect_direct(
                image_bytes=image_bytes,
                prompts=merged_prompts or None,
                conf=conf,
                iou=iou,
                algorithm_codes=algo_codes or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Preview inference error")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        # ── LLM behavior enhancement (if enabled and algo has stage=behavior) ─
        image_bgr = decode_image_bytes(image_bytes)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")

        detections = batch.detections
        llm_applied = False
        vlm_crops: list = []
        if settings.llm_enabled and algo_codes:
            behavior_codes = [
                c for c in algo_codes
                if (spec := get_spec(c)) is not None and spec.stage == "behavior"
            ]
            if behavior_codes:
                detections, llm_applied, vlm_crops = await orchestrator.enhance_direct(
                    image_bgr=image_bgr,
                    detections=detections,
                    algorithm_codes=behavior_codes,
                )

        # ── Draw and encode ───────────────────────────────────────────────────
        annotated = draw_bboxes(image_bgr, detections)
        try:
            annotated_b64 = encode_jpeg_b64(annotated, quality=90)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(
            {
                "total": len(detections),
                "detections": detections_to_json(detections),
                "prompts_used": merged_prompts,
                "annotated_image": annotated_b64,
                "llm_applied": llm_applied,
                "stage": "behavior" if llm_applied else "yolo",
                "vlm_crops": vlm_crops,
            }
        )

    @app.websocket("/preview/stream")
    async def preview_stream(websocket: WebSocket) -> None:
        await preview_stream_endpoint(websocket, orchestrator, settings)

    return app

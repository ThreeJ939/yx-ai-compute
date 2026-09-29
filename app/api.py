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
from app.registry import list_capabilities

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

        merged_prompts = resolve_prompts(algorithms, prompts)
        if not merged_prompts:
            raise HTTPException(
                status_code=400,
                detail="No prompts resolved; check algorithms / prompts params",
            )

        try:
            batch = await orchestrator.detect_direct(
                image_bytes=image_bytes,
                prompts=merged_prompts,
                conf=conf,
                iou=iou,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Preview inference error")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        image_bgr = decode_image_bytes(image_bytes)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")
        annotated = draw_bboxes(image_bgr, batch.detections)
        try:
            annotated_b64 = encode_jpeg_b64(annotated, quality=90)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(
            {
                "total": len(batch.detections),
                "detections": detections_to_json(batch.detections),
                "prompts_used": merged_prompts,
                "annotated_image": annotated_b64,
            }
        )

    @app.websocket("/preview/stream")
    async def preview_stream(websocket: WebSocket) -> None:
        await preview_stream_endpoint(websocket, orchestrator, settings)

    return app

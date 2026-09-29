"""WebSocket live preview: client-pushed frames or server-pulled RTSP/camera."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.preview_util import (
    decode_image_bytes,
    detections_to_json,
    draw_bboxes,
    encode_jpeg_b64,
    maybe_resize,
    resolve_prompts,
)

if TYPE_CHECKING:
    from app.config import Settings
    from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class PreviewStreamSession:
    def __init__(
        self,
        websocket: WebSocket,
        orchestrator: Orchestrator,
        settings: Settings,
    ) -> None:
        self.ws = websocket
        self.orch = orchestrator
        self.settings = settings
        self.algorithms = ""
        self.prompts = ""
        self.conf: float | None = None
        self.iou: float | None = None
        self.interval_ms = float(getattr(settings, "preview_interval_ms", 500) or 500)
        self.max_width = int(getattr(settings, "preview_max_width", 1280) or 1280)
        self.jpeg_quality = int(getattr(settings, "preview_jpeg_quality", 80) or 80)
        self._busy = False
        self._pull_task: asyncio.Task | None = None
        self._pull_stop = asyncio.Event()
        self._frame_idx = 0

    async def run(self) -> None:
        await self.ws.accept()
        await self._send({"op": "ready", "message": "preview stream connected"})
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send({"op": "error", "detail": "invalid JSON"})
                    continue
                await self._handle(msg)
        except WebSocketDisconnect:
            logger.info("Preview WS disconnected")
        except Exception:
            logger.exception("Preview WS error")
        finally:
            await self._stop_pull()

    async def _handle(self, msg: dict[str, Any]) -> None:
        op = (msg.get("op") or "").strip()
        if op == "config":
            self._apply_config(msg)
            await self._send(
                {
                    "op": "status",
                    "message": "config updated",
                    "interval_ms": self.interval_ms,
                    "prompts": resolve_prompts(self.algorithms, self.prompts),
                }
            )
            return
        if op == "frame":
            await self._handle_client_frame(msg)
            return
        if op == "start_source":
            await self._start_pull(msg)
            return
        if op == "stop":
            await self._stop_pull()
            await self._send({"op": "status", "message": "stopped"})
            return
        await self._send({"op": "error", "detail": f"unknown op: {op}"})

    def _apply_config(self, msg: dict[str, Any]) -> None:
        if "algorithms" in msg:
            self.algorithms = str(msg.get("algorithms") or "")
        if "prompts" in msg:
            self.prompts = str(msg.get("prompts") or "")
        if msg.get("conf") is not None:
            try:
                self.conf = float(msg["conf"])
            except (TypeError, ValueError):
                pass
        if msg.get("iou") is not None:
            try:
                self.iou = float(msg["iou"])
            except (TypeError, ValueError):
                pass
        if msg.get("interval_ms") is not None:
            try:
                self.interval_ms = max(50.0, float(msg["interval_ms"]))
            except (TypeError, ValueError):
                pass
        if msg.get("max_width") is not None:
            try:
                self.max_width = max(0, int(msg["max_width"]))
            except (TypeError, ValueError):
                pass

    async def _handle_client_frame(self, msg: dict[str, Any]) -> None:
        if self._busy:
            # Drop frame to keep real-time; client should keep sending.
            await self._send({"op": "dropped", "reason": "busy"})
            return
        b64 = msg.get("image") or ""
        if not b64:
            await self._send({"op": "error", "detail": "frame missing image"})
            return
        try:
            import base64

            # allow data URL prefix
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        except Exception:
            await self._send({"op": "error", "detail": "invalid base64 image"})
            return

        self._busy = True
        try:
            result = await self._infer_bytes(image_bytes)
            await self._send(result)
        except Exception as exc:
            logger.exception("Preview frame infer failed")
            await self._send({"op": "error", "detail": str(exc)})
        finally:
            self._busy = False

    async def _start_pull(self, msg: dict[str, Any]) -> None:
        await self._stop_pull()
        self._apply_config(msg)
        source = (msg.get("source") or "").strip()
        if not source:
            await self._send({"op": "error", "detail": "source required (rtsp://... or 0)"})
            return
        self._pull_stop = asyncio.Event()
        self._pull_task = asyncio.create_task(
            self._pull_loop(source), name="preview-pull"
        )
        await self._send({"op": "status", "message": f"pulling {source}"})

    async def _stop_pull(self) -> None:
        self._pull_stop.set()
        task = self._pull_task
        self._pull_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("pull task end", exc_info=True)

    async def _pull_loop(self, source: str) -> None:
        import cv2  # noqa: PLC0415

        # OpenCV accepts int camera index or URL string
        open_src: Any = source
        if source.isdigit():
            open_src = int(source)

        cap = await asyncio.to_thread(cv2.VideoCapture, open_src)
        if not cap.isOpened():
            await self._send({"op": "error", "detail": f"cannot open source: {source}"})
            return

        # Prefer lower latency for RTSP when possible
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        interval = max(0.05, self.interval_ms / 1000.0)
        try:
            while not self._pull_stop.is_set():
                t0 = time.perf_counter()
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await self._send({"op": "error", "detail": "stream ended or read failed"})
                    break
                if self._busy:
                    # skip infer but still pace
                    await asyncio.sleep(0.01)
                    continue
                self._busy = True
                try:
                    result = await self._infer_bgr(frame)
                    await self._send(result)
                except Exception as exc:
                    logger.exception("Pull infer failed")
                    await self._send({"op": "error", "detail": str(exc)})
                finally:
                    self._busy = False

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            raise
        finally:
            await asyncio.to_thread(cap.release)
            await self._send({"op": "status", "message": "pull stopped"})

    async def _infer_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        image_bgr = decode_image_bytes(image_bytes)
        if image_bgr is None:
            raise ValueError("Cannot decode image")
        return await self._infer_bgr(image_bgr)

    async def _infer_bgr(self, image_bgr) -> dict[str, Any]:
        import cv2  # noqa: PLC0415

        prompts = resolve_prompts(self.algorithms, self.prompts)
        if not prompts:
            raise ValueError("No prompts resolved")

        resized, _ = maybe_resize(image_bgr, self.max_width)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("encode for infer failed")
        jpeg_bytes = buf.tobytes()

        t0 = time.perf_counter()
        batch = await self.orch.detect_direct(
            image_bytes=jpeg_bytes,
            prompts=prompts,
            conf=self.conf,
            iou=self.iou,
        )
        ms = int((time.perf_counter() - t0) * 1000)

        # Re-decode resized jpeg to draw (same space as detections)
        drawn_src = decode_image_bytes(jpeg_bytes)
        annotated = draw_bboxes(drawn_src, batch.detections)
        annotated_b64 = encode_jpeg_b64(annotated, quality=self.jpeg_quality)

        self._frame_idx += 1
        return {
            "op": "result",
            "frame": self._frame_idx,
            "total": len(batch.detections),
            "detections": detections_to_json(batch.detections),
            "prompts_used": prompts,
            "annotated_image": annotated_b64,
            "ms": ms,
        }

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.debug("WS send failed", exc_info=True)


async def preview_stream_endpoint(
    websocket: WebSocket,
    orchestrator: Orchestrator | None,
    settings: Settings,
) -> None:
    if orchestrator is None:
        await websocket.accept()
        await websocket.send_text(
            json.dumps(
                {"op": "error", "detail": "Preview unavailable: orchestrator not initialised"},
                ensure_ascii=False,
            )
        )
        await websocket.close(code=1013)
        return
    session = PreviewStreamSession(websocket, orchestrator, settings)
    await session.run()

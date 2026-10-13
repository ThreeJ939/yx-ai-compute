"""OpenAI-compatible vision client for behavior enhancement."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
# Some Qwen / reasoning models wrap thinking before the final answer
_THINK_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>",
    re.IGNORECASE,
)


def parse_vlm_json(text: str) -> dict[str, Any]:
    """Extract JSON object from VLM text (tolerates markdown fences / think blocks)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty VLM response")

    raw = _THINK_BLOCK_RE.sub("", raw).strip()

    match = _JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(1).strip()

    # Try direct parse, then first {...} slice
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in VLM response: {text[:200]!r}") from None
        data = json.loads(raw[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError(f"VLM JSON is not an object: {type(data)}")
    return data


def normalize_verdict(data: dict[str, Any]) -> dict[str, Any]:
    positive = data.get("positive")
    if isinstance(positive, str):
        positive = positive.strip().lower() in {"true", "1", "yes", "y"}
    else:
        positive = bool(positive)

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    label = data.get("label")
    label_str = str(label).strip() if label is not None else ""
    return {
        "positive": positive,
        "confidence": confidence,
        "label": label_str,
    }


class LlmClient:
    """Thin wrapper around OpenAI-compatible chat.completions (vision)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(
            api_key=self._settings.llm_api_key or "EMPTY",
            base_url=self._settings.llm_base_url or None,
            timeout=max(1.0, self._settings.llm_timeout_ms / 1000.0),
        )
        return self._client

    @staticmethod
    def encode_crop_jpeg_b64(image_bgr: Any, quality: int = 85) -> str:
        import cv2  # noqa: PLC0415

        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("failed to encode crop JPEG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def call_one(
        self,
        image_b64: str,
        system: str,
        user: str,
    ) -> dict[str, Any]:
        """Call VLM once; returns normalized verdict dict."""
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        # Align with yx-agent-router ClassifierClient:
        # - response_format=json_object forces a JSON object reply
        # - disable thinking / reasoning so models don't emit analysis prose
        create_kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": self._settings.llm_temperature,
            "extra_body": {
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if getattr(self._settings, "llm_json_object", True):
            create_kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**create_kwargs)
        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected VLM response shape: {resp!r}") from exc

        parsed = parse_vlm_json(content)
        return normalize_verdict(parsed)

    def call(
        self,
        crops_b64: list[str],
        system: str,
        user: str,
        max_workers: int = 1,
    ) -> list[dict[str, Any]]:
        """Call VLM for each crop; failures propagate to caller.

        When ``max_workers`` > 1 and there are multiple crops, calls run
        concurrently via a short-lived ThreadPoolExecutor (sync OpenAI client).
        """
        n = len(crops_b64)
        if n == 0:
            return []
        workers = max(1, int(max_workers))
        if n == 1 or workers == 1:
            out: list[dict[str, Any]] = []
            for i, b64 in enumerate(crops_b64):
                logger.debug(
                    "VLM call crop=%s/%s model=%s",
                    i + 1,
                    n,
                    self._settings.llm_model,
                )
                out.append(self.call_one(b64, system, user))
            return out

        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        logger.debug(
            "VLM call parallel crops=%s workers=%s model=%s",
            n,
            workers,
            self._settings.llm_model,
        )
        with ThreadPoolExecutor(max_workers=min(workers, n), thread_name_prefix="vlm") as pool:
            futures = [
                pool.submit(self.call_one, b64, system, user) for b64 in crops_b64
            ]
            return [f.result() for f in futures]

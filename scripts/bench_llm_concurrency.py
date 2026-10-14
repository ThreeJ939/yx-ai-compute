#!/usr/bin/env python3
"""Benchmark VLM crop concurrency via OpenAI-compatible API.

Measures success rate, latency (p50/p95/p99), throughput, and p50 slowdown
vs concurrency=1 as concurrency increases.

Usage (from repo root, with .env ready):

  python -m scripts.bench_llm_concurrency path/to/crop.jpg
  python -m scripts.bench_llm_concurrency crop.jpg --levels 1,2,3,6,9 --requests 30
  python -m scripts.bench_llm_concurrency crop.jpg \\
      --base-url http://127.0.0.1:56986/v1 --model Qwen3.5-9B

Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_TIMEOUT_MS from .env when
flags are omitted. Prefer a person crop (not a full 4K frame).
"""

from __future__ import annotations

import argparse
import base64
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_settings():
    try:
        from app.config import Settings

        return Settings()
    except Exception:
        return None


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _one_call(
    client,
    model: str,
    image_b64: str,
    system: str,
    user: str,
    json_object: bool,
    max_tokens: int,
    temperature: float,
) -> float:
    """Return latency seconds; raises on failure."""
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
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    _ = resp.choices[0].message.content
    return time.perf_counter() - t0


def run_level(
    client,
    *,
    model: str,
    image_b64: str,
    system: str,
    user: str,
    json_object: bool,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    total_requests: int,
) -> dict:
    latencies: list[float] = []
    errors = 0
    wall0 = time.perf_counter()

    def task(_i: int) -> float:
        return _one_call(
            client,
            model,
            image_b64,
            system,
            user,
            json_object,
            max_tokens,
            temperature,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, i) for i in range(total_requests)]
        for fut in as_completed(futures):
            try:
                latencies.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  error: {exc}", file=sys.stderr)

    wall = time.perf_counter() - wall0
    ok = len(latencies)
    latencies.sort()
    return {
        "concurrency": concurrency,
        "total": total_requests,
        "ok": ok,
        "errors": errors,
        "success_rate": ok / total_requests if total_requests else 0.0,
        "wall_s": wall,
        "throughput_rps": ok / wall if wall > 0 else 0.0,
        "avg_s": statistics.mean(latencies) if latencies else float("nan"),
        "p50_s": _percentile(latencies, 50),
        "p95_s": _percentile(latencies, 95),
        "p99_s": _percentile(latencies, 99),
        "min_s": latencies[0] if latencies else float("nan"),
        "max_s": latencies[-1] if latencies else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench VLM image concurrency")
    parser.add_argument("image", type=Path, help="JPEG/PNG crop (person crop preferred)")
    parser.add_argument("--base-url", default=None, help="Override LLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="Override LLM_API_KEY")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL")
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument(
        "--levels",
        default="1,2,3,6,9",
        help="Concurrency levels, comma-separated (default: 1,2,3,6,9)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=30,
        help="Requests per concurrency level (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Serial warmup calls before bench (default: 2)",
    )
    parser.add_argument(
        "--no-json-object",
        action="store_true",
        help="Do not send response_format=json_object",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a vision classifier. Reply JSON only: "
            '{"positive":bool,"confidence":float,"label":string}'
        ),
    )
    parser.add_argument("--user", default="Is the person fishing? Reply JSON.")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1

    settings = _load_settings()
    base_url = args.base_url or (settings.llm_base_url if settings else None)
    api_key = args.api_key or (settings.llm_api_key if settings else "EMPTY") or "EMPTY"
    model = args.model or (settings.llm_model if settings else None)
    timeout_ms = args.timeout_ms or (settings.llm_timeout_ms if settings else 15000)
    max_tokens = settings.llm_max_tokens if settings else 256
    temperature = settings.llm_temperature if settings else 0.1
    json_object = not args.no_json_object
    if settings is not None and not args.no_json_object:
        json_object = bool(settings.llm_json_object)

    if not base_url or not model:
        print("Need --base-url/--model or LLM_* in .env", file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=max(1.0, timeout_ms / 1000.0),
    )
    image_b64 = _encode_image(args.image)
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    if not levels:
        print("empty --levels", file=sys.stderr)
        return 1

    print(f"endpoint={base_url}")
    print(
        f"model={model} timeout_ms={timeout_ms} "
        f"image={args.image} bytes={args.image.stat().st_size}"
    )
    print(f"levels={levels} requests_per_level={args.requests}")
    print()

    if args.warmup > 0:
        print(f"warmup x{args.warmup} ...")
        for _ in range(args.warmup):
            try:
                t = _one_call(
                    client,
                    model,
                    image_b64,
                    args.system,
                    args.user,
                    json_object,
                    max_tokens,
                    temperature,
                )
                print(f"  warmup ok {t:.3f}s")
            except Exception as exc:  # noqa: BLE001
                print(f"  warmup failed: {exc}", file=sys.stderr)
                return 1

    rows: list[dict] = []
    baseline_p50: float | None = None
    for c in levels:
        print(f"\n=== concurrency={c} ===")
        row = run_level(
            client,
            model=model,
            image_b64=image_b64,
            system=args.system,
            user=args.user,
            json_object=json_object,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=c,
            total_requests=args.requests,
        )
        if baseline_p50 is None and row["ok"] > 0:
            baseline_p50 = row["p50_s"]
        slowdown = (
            (row["p50_s"] / baseline_p50)
            if baseline_p50 and row["ok"] > 0
            else float("nan")
        )
        row["p50_slowdown_vs_c1"] = slowdown
        rows.append(row)
        print(
            f"  ok={row['ok']}/{row['total']} err={row['errors']} "
            f"rps={row['throughput_rps']:.2f} "
            f"p50={row['p50_s']:.3f}s p95={row['p95_s']:.3f}s "
            f"slowdown_p50={slowdown:.2f}x"
        )

    print("\n=== summary ===")
    print(
        f"{'C':>3} {'ok%':>6} {'rps':>7} {'p50':>8} {'p95':>8} "
        f"{'p99':>8} {'slowdown':>9}"
    )
    for r in rows:
        print(
            f"{r['concurrency']:3d} {r['success_rate'] * 100:5.1f}% "
            f"{r['throughput_rps']:7.2f} {r['p50_s']:8.3f} {r['p95_s']:8.3f} "
            f"{r['p99_s']:8.3f} {r['p50_slowdown_vs_c1']:8.2f}x"
        )

    print(
        "\nHow to read: if at C=3 success stays high and p50_slowdown < ~1.5x, "
        "LLM_CROP_CONCURRENCY=3 is reasonable; sharp rise in p95/errors means "
        "oversubscribe."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

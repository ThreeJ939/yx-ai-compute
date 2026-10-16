from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_id: str = "ai-compute-service"
    health_port: int = 18100
    log_level: str = "INFO"

    # Set false for local preview-only (HTTP) without Kafka
    kafka_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_username: str = ""
    kafka_password: str = ""
    kafka_security_protocol: str = "PLAINTEXT"
    kafka_sasl_mechanism: str = "PLAIN"
    kafka_request_topic: str = "ai.inference.request"
    kafka_result_topic: str = "ai.inference.result"
    kafka_group_id: str = "yx-ai-compute-group"
    # Max in-flight Kafka messages (YOLO serial, LLM stages overlap)
    kafka_max_concurrent: int = 5

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    yolo_world_weights: str = "weights/yolov8s-worldv2.pt"
    # Local OpenAI CLIP ViT-B/32 weights for offline set_classes (empty = auto download)
    yolo_clip_weights: str = ""
    device: str = "cpu"
    # YOLO-World (world backend) conf / iou — env YOLO_CONF / YOLO_IOU
    # Legacy aliases CONF_THRESHOLD / IOU_THRESHOLD still accepted.
    yolo_conf: float = Field(
        default=0.35,
        validation_alias=AliasChoices("YOLO_CONF", "CONF_THRESHOLD", "yolo_conf"),
    )
    yolo_iou: float = Field(
        default=0.45,
        validation_alias=AliasChoices("YOLO_IOU", "IOU_THRESHOLD", "yolo_iou"),
    )
    # Sliding-window inference for distant / small objects (yolo-world-demo style)
    yolo_tiled: bool = True
    yolo_tile_size: int = 640
    yolo_tile_overlap: float = 0.2
    yolo_imgsz: int = 640
    # How many tiles per YOLO predict() call (1 = serial; 4 typical for GPU)
    yolo_tile_batch: int = 4

    # Dedicated fire/smoke Ultralytics YOLO (not YOLO-World). Off by default.
    fire_smoke_weights: str = "weights/fire-smoke-yolov8n.pt"
    fire_smoke_enabled: bool = False
    # fire_smoke backend 独立推理参数（0.0 conf/iou = 沿用 YOLO_CONF / YOLO_IOU）
    fire_smoke_tiled: bool = True
    fire_smoke_tile_size: int = 640
    fire_smoke_tile_overlap: float = 0.2
    fire_smoke_tile_batch: int = 4
    fire_smoke_conf: float = 0.0
    fire_smoke_iou: float = 0.0

    # Dedicated dredge (illegal sand-mining) Ultralytics YOLO. Off by default.
    dredge_weights: str = "weights/dredge-yolov8n-640.pt"
    dredge_enabled: bool = False
    dredge_tiled: bool = True
    dredge_tile_size: int = 640
    dredge_tile_overlap: float = 0.2
    dredge_tile_batch: int = 4
    dredge_imgsz: int = 640
    # 0.0 conf/iou = fall back to YOLO_CONF / YOLO_IOU
    dredge_conf: float = 0.0
    dredge_iou: float = 0.0

    annotated_bucket: str = ""

    # Preview UI: GET /preview + POST /preview/image + WS /preview/stream
    preview_enabled: bool = True
    # Client/server stream pacing (ms between frames). CPU 建议 ≥800
    preview_interval_ms: int = 500
    # Downscale frames wider than this before infer (0 = no resize)
    preview_max_width: int = 1280
    preview_jpeg_quality: int = 80

    # LLM / VLM behavior enhancer (OpenAI-compatible vision API)
    llm_enabled: bool = False
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-vl-plus"
    llm_timeout_ms: int = 8000
    llm_max_tokens: int = 256
    llm_temperature: float = 0.1
    # VLM confidence below this is treated as unconfirmed
    llm_conf_threshold: float = 0.5
    llm_prompts_path: str = "config/llm_prompts.yaml"
    # Request OpenAI-style response_format=json_object (set false if local server rejects it)
    llm_json_object: bool = True
    # Per-request crop concurrency (align with LLM backend instance count)
    llm_crop_concurrency: int = 3

    def kafka_has_sasl(self) -> bool:
        return bool(self.kafka_username and self.kafka_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()

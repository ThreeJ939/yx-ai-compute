from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    yolo_world_weights: str = "weights/yolov8s-worldv2.pt"
    # Local OpenAI CLIP ViT-B/32 weights for offline set_classes (empty = auto download)
    yolo_clip_weights: str = ""
    device: str = "cpu"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    # Sliding-window inference for distant / small objects (yolo-world-demo style)
    yolo_tiled: bool = True
    yolo_tile_size: int = 640
    yolo_tile_overlap: float = 0.2
    yolo_imgsz: int = 640

    annotated_bucket: str = ""

    # Preview UI: GET /preview + POST /preview/image + WS /preview/stream
    preview_enabled: bool = True
    # Client/server stream pacing (ms between frames). CPU 建议 ≥800
    preview_interval_ms: int = 500
    # Downscale frames wider than this before infer (0 = no resize)
    preview_max_width: int = 1280
    preview_jpeg_quality: int = 80

    def kafka_has_sasl(self) -> bool:
        return bool(self.kafka_username and self.kafka_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()

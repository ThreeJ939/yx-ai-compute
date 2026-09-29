"""MinIO client for downloading ephemeral detection frames."""

from __future__ import annotations

import logging
from io import BytesIO

from minio import Minio

from app.config import Settings

logger = logging.getLogger(__name__)


class MinioImageStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def download_bytes(self, bucket: str, object_key: str) -> bytes:
        logger.debug("Downloading object s3://%s/%s", bucket, object_key)
        response = self._client.get_object(bucket, object_key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        if not data:
            raise ValueError(f"Empty object: {bucket}/{object_key}")
        return data

    def upload_bytes(
        self,
        bucket: str,
        object_key: str,
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload bytes; returns object key. Caller builds public URL if needed."""
        self._client.put_object(
            bucket,
            object_key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_key

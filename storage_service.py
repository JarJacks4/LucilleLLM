"""
LucilleLLM - Cloud Storage Service

Provides Google Cloud Storage integration for soundscape audio file hosting.
Supports signed URL generation, file upload/download, and existence checks.

Graceful degradation: all methods return safe defaults when GCS_AUDIO_BUCKET
is not configured, allowing the app to run without GCS in development mode.

Follows the singleton pattern from other services.
"""

import logging
from datetime import timedelta
from typing import List, Optional

from config import get_config

logger = logging.getLogger(__name__)


class StorageService:
    """
    Service for Google Cloud Storage audio file operations.

    Bucket layout:
        {GCS_AUDIO_PREFIX}{soundscape_id}.mp3
        e.g. soundscapes/nature_rain.mp3
    """

    def __init__(self):
        config = get_config()
        self._bucket_name = config.GCS_AUDIO_BUCKET
        self._prefix = config.GCS_AUDIO_PREFIX
        self._expiry_minutes = config.GCS_SIGNED_URL_EXPIRY_MINUTES
        self._client = None
        self._bucket = None

        if self._bucket_name:
            try:
                from google.cloud import storage as gcs
                self._client = gcs.Client()
                self._bucket = self._client.bucket(self._bucket_name)
                logger.info(
                    f"StorageService initialized — bucket={self._bucket_name}, "
                    f"prefix={self._prefix}"
                )
            except Exception as e:
                logger.warning(
                    f"StorageService: GCS client init failed ({e}). "
                    "Audio URLs will be empty."
                )
                self._client = None
                self._bucket = None
        else:
            logger.info(
                "StorageService: GCS_AUDIO_BUCKET not configured. "
                "Running without audio storage (development mode)."
            )

    @property
    def is_configured(self) -> bool:
        """Check if GCS is properly configured and connected."""
        return self._bucket is not None

    # ── Signed URL Generation ────────────────────────────────

    def get_signed_url(self, blob_path: str) -> str:
        """
        Generate a signed URL for a GCS blob.

        Args:
            blob_path: Full path within the bucket (e.g. "soundscapes/nature_rain.mp3")

        Returns:
            Signed URL string, or "" if GCS is not configured or blob doesn't exist.
        """
        if not self.is_configured:
            return ""
        try:
            blob = self._bucket.blob(blob_path)
            if not blob.exists():
                logger.debug(f"Blob not found: {blob_path}")
                return ""
            url = blob.generate_signed_url(
                expiration=timedelta(minutes=self._expiry_minutes),
                method="GET",
            )
            return url
        except Exception as e:
            logger.warning(f"Failed to generate signed URL for {blob_path}: {e}")
            return ""

    def get_audio_url(self, soundscape_id: str) -> str:
        """
        Convenience method: generate a signed URL for a soundscape audio file.

        Args:
            soundscape_id: e.g. "nature_rain"

        Returns:
            Signed URL string, or "" if unavailable.
        """
        blob_path = f"{self._prefix}{soundscape_id}.mp3"
        return self.get_signed_url(blob_path)

    # ── File Operations ──────────────────────────────────────

    def upload_file(self, local_path: str, blob_path: str) -> bool:
        """
        Upload a local file to GCS.

        Args:
            local_path: Path to local file
            blob_path: Destination path in bucket

        Returns:
            True if upload succeeded, False otherwise.
        """
        if not self.is_configured:
            logger.warning("Cannot upload: GCS not configured")
            return False
        try:
            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} -> gs://{self._bucket_name}/{blob_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed for {local_path}: {e}")
            return False

    def file_exists(self, blob_path: str) -> bool:
        """
        Check if a blob exists in the bucket.

        Args:
            blob_path: Full path within the bucket

        Returns:
            True if exists, False otherwise (including when GCS not configured).
        """
        if not self.is_configured:
            return False
        try:
            blob = self._bucket.blob(blob_path)
            return blob.exists()
        except Exception as e:
            logger.warning(f"Error checking blob existence for {blob_path}: {e}")
            return False

    def audio_file_exists(self, soundscape_id: str) -> bool:
        """Check if audio file exists for a soundscape."""
        blob_path = f"{self._prefix}{soundscape_id}.mp3"
        return self.file_exists(blob_path)

    def list_audio_files(self) -> List[str]:
        """
        List all audio blobs under the configured prefix.

        Returns:
            List of blob names (e.g. ["soundscapes/nature_rain.mp3", ...]),
            or [] if GCS not configured.
        """
        if not self.is_configured:
            return []
        try:
            blobs = self._client.list_blobs(
                self._bucket_name, prefix=self._prefix
            )
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.warning(f"Failed to list audio files: {e}")
            return []

    def delete_file(self, blob_path: str) -> bool:
        """
        Delete a blob from the bucket.

        Args:
            blob_path: Full path within the bucket

        Returns:
            True if deleted, False otherwise.
        """
        if not self.is_configured:
            return False
        try:
            blob = self._bucket.blob(blob_path)
            blob.delete()
            logger.info(f"Deleted gs://{self._bucket_name}/{blob_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {blob_path}: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────

_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create StorageService singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

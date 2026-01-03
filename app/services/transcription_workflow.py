"""High-level workflow orchestration for transcription jobs.

This module provides workflow orchestration that coordinates multiple services
to handle the complete transcription lifecycle, extracting business logic from
route handlers.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.validators.upload_validator import UploadValidator
from app.exceptions import InsufficientDiskSpaceError

logger = logging.getLogger(__name__)


class TranscriptionWorkflow:
    """Orchestrates the complete transcription workflow."""

    def __init__(self):
        """Initialize transcription workflow."""
        self.validator = UploadValidator()
        logger.info("TranscriptionWorkflow initialized")

    async def save_upload(
        self,
        file,
        filename: str
    ) -> Path:
        """
        Save uploaded file with validation and disk space checking.

        Args:
            file: UploadFile object from FastAPI
            filename: Original filename

        Returns:
            Path to saved file

        Raises:
            ValueError: If file validation fails
            InsufficientDiskSpaceError: If disk space is low
            Exception: If file saving fails
        """
        # Validate disk space before saving
        self.validator.check_disk_space()

        # Generate unique temp job ID for filename (to avoid collisions)
        temp_job_id = str(uuid.uuid4())
        upload_path = settings.temp_upload_dir / f"{temp_job_id}_{filename}"

        logger.info(f"Saving upload to {upload_path}")

        # Save file in chunks to handle large files efficiently
        chunk_size = 1024 * 1024  # 1MB chunks
        total_bytes = 0

        try:
            with upload_path.open("wb") as buffer:
                while chunk := await file.read(chunk_size):
                    buffer.write(chunk)
                    total_bytes += len(chunk)

            logger.info(f"Saved file to {upload_path} ({total_bytes / 1024 / 1024:.2f}MB)")

        except Exception as e:
            # Clean up partial file on error
            if upload_path.exists():
                try:
                    upload_path.unlink()
                    logger.info(f"Cleaned up partial file: {upload_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup partial file: {cleanup_error}")

            logger.error(f"Failed to save file: {e}")
            raise ValueError(f"Failed to save uploaded file: {e}")

        return upload_path

    def cleanup_upload(self, upload_path: Path) -> None:
        """
        Clean up uploaded file (called on job creation failure).

        Args:
            upload_path: Path to uploaded file
        """
        try:
            if upload_path and upload_path.exists():
                upload_path.unlink()
                logger.info(f"Cleaned up upload file: {upload_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup upload file {upload_path}: {e}")

    def validate_upload(
        self,
        file,
        filename: str,
        file_size: int
    ) -> str:
        """
        Validate uploaded file.

        Args:
            file: UploadFile object
            filename: Original filename
            file_size: File size in bytes

        Returns:
            str: File extension (without dot)

        Raises:
            ValueError: If validation fails
        """
        return self.validator.validate_file(file, filename, file_size)

    def validate_params(
        self,
        whisper_model,
        language: Optional[str],
        num_speakers: Optional[int] = None
    ) -> dict:
        """
        Validate transcription parameters.

        Args:
            whisper_model: Whisper model size
            language: Language code
            num_speakers: Expected number of speakers (hint for diarization)

        Returns:
            dict: Validated parameters

        Raises:
            ValueError: If validation fails
        """
        return self.validator.validate_transcription_params(
            whisper_model,
            language,
            num_speakers
        )

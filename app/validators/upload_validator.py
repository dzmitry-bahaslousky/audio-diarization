"""Validators for file uploads and processing parameters.

This module provides validation logic for audio file uploads and transcription
parameters, extracting business logic from route handlers.
"""

from pathlib import Path
from typing import Optional
import shutil
import logging

from app.config import settings
from app.exceptions import InvalidAudioError, InsufficientDiskSpaceError
from app.models.schemas import WhisperModel

logger = logging.getLogger(__name__)


class UploadValidator:
    """Validates file uploads and processing parameters."""

    @staticmethod
    def validate_file(
        file,
        filename: str,
        file_size: int
    ) -> str:
        """
        Validate uploaded audio file.

        Args:
            file: UploadFile object
            filename: Original filename
            file_size: File size in bytes

        Returns:
            str: Validated file extension (without dot)

        Raises:
            ValueError: If validation fails
        """
        # Check file exists
        if not file:
            raise ValueError("No file provided")

        # Check filename
        if not filename:
            raise ValueError("File must have a filename")

        # Check extension
        file_extension = Path(filename).suffix.lower().lstrip(".")
        if file_extension not in settings.allowed_extensions:
            allowed = ', '.join(settings.allowed_extensions)
            raise ValueError(
                f"File type '.{file_extension}' not allowed. "
                f"Allowed extensions: {allowed}"
            )

        # Check file size
        if file_size > settings.max_upload_size_bytes:
            size_mb = file_size / 1024 / 1024
            raise ValueError(
                f"File size {size_mb:.2f}MB exceeds maximum "
                f"{settings.max_upload_size_mb}MB"
            )

        if file_size == 0:
            raise ValueError("File is empty")

        logger.debug(f"File validation passed: {filename} ({file_size / 1024 / 1024:.2f}MB)")

        return file_extension

    @staticmethod
    def validate_transcription_params(
        whisper_model: WhisperModel,
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
            dict: Validated parameters ready for database

        Raises:
            ValueError: If validation fails
        """
        params = {}

        # Validate whisper_model (already validated by Pydantic enum)
        params['whisper_model'] = whisper_model.value

        # Validate language
        if language:
            # Normalize language code
            language = language.lower().strip()

            if len(language) > 5:
                raise ValueError("Language code too long (max 5 characters)")

            # Check for invalid characters (only alphanumeric and hyphen)
            if not all(c.isalnum() or c == '-' for c in language):
                raise ValueError("Language code contains invalid characters")

            params['language'] = language

        # Validate num_speakers
        if num_speakers is not None:
            if not isinstance(num_speakers, int):
                raise ValueError("num_speakers must be an integer")

            if num_speakers < 1 or num_speakers > 10:
                raise ValueError("num_speakers must be between 1 and 10")

            params['num_speakers'] = num_speakers

        logger.debug(f"Transcription params validated: {params}")

        return params

    @staticmethod
    def check_disk_space(min_free_gb: float = 5.0):
        """
        Check if sufficient disk space is available for processing.

        Args:
            min_free_gb: Minimum free space required in GB

        Raises:
            InsufficientDiskSpaceError: If disk space is insufficient
        """
        try:
            upload_stats = shutil.disk_usage(settings.temp_upload_dir)
            output_stats = shutil.disk_usage(settings.output_dir)

            upload_free_gb = upload_stats.free / (1024**3)
            output_free_gb = output_stats.free / (1024**3)

            if upload_free_gb < min_free_gb:
                raise InsufficientDiskSpaceError(
                    f"Upload directory has only {upload_free_gb:.2f}GB free, "
                    f"need at least {min_free_gb}GB"
                )

            if output_free_gb < min_free_gb:
                raise InsufficientDiskSpaceError(
                    f"Output directory has only {output_free_gb:.2f}GB free, "
                    f"need at least {min_free_gb}GB"
                )

            logger.debug(
                f"Disk space check passed: "
                f"upload={upload_free_gb:.2f}GB, output={output_free_gb:.2f}GB"
            )

        except InsufficientDiskSpaceError:
            raise
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            # Don't fail on disk space check errors, just warn
            logger.warning("Could not verify disk space, proceeding anyway")

    @staticmethod
    def validate_audio_path(audio_path: Path) -> None:
        """
        Validate audio file path exists and is readable.

        Args:
            audio_path: Path to audio file

        Raises:
            InvalidAudioError: If file doesn't exist or isn't readable
        """
        if not audio_path.exists():
            raise InvalidAudioError(f"Audio file not found: {audio_path}")

        if not audio_path.is_file():
            raise InvalidAudioError(f"Audio path is not a file: {audio_path}")

        # Check if file is readable
        try:
            with audio_path.open('rb') as f:
                # Try to read first byte
                f.read(1)
        except PermissionError:
            raise InvalidAudioError(f"No permission to read file: {audio_path}")
        except Exception as e:
            raise InvalidAudioError(f"Cannot read audio file: {e}")

        logger.debug(f"Audio path validation passed: {audio_path}")

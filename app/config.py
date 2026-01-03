"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Whisper Configuration
    whisper_model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"] = "medium"
    whisper_device: Literal["cpu", "cuda", "mps"] = "cpu"  # NOTE: MPS not supported by faster-whisper

    # WhisperX Performance Settings
    whisperx_compute_type: Literal["float16", "int8", "float32"] = "int8"  # int8 recommended for CPU
    whisperx_batch_size: int = Field(16, ge=1, le=32, description="Batch size for inference (higher=faster but more memory)")
    whisperx_enable_alignment: bool = Field(True, description="Enable word-level timestamps via wav2vec2")
    whisperx_enable_diarization: bool = Field(True, description="Enable speaker identification")
    whisperx_min_speakers: Optional[int] = Field(None, ge=1, le=10, description="Minimum expected speakers (optional)")
    whisperx_max_speakers: Optional[int] = Field(None, ge=1, le=10, description="Maximum expected speakers (optional)")

    # Hugging Face Token (required for WhisperX diarization)
    hf_token: str = Field(..., description="Hugging Face API token for diarization models")

    # Application Settings
    max_upload_size_mb: int = Field(500, ge=1, le=1000)
    allowed_extensions: str = "mp3,wav,m4a,flac,ogg,wma"
    temp_upload_dir: Path = Path("./uploads")
    output_dir: Path = Path("./outputs")

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1000, le=65535)
    api_reload: bool = True

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Database (Phase 5)
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/audio_diarization",
        description="PostgreSQL connection string (using psycopg3 driver)"
    )

    # Celery (Phase 5)
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis)"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )

    @field_validator("allowed_extensions")
    @classmethod
    def parse_extensions(cls, v: str) -> set[str]:
        """Convert comma-separated extensions to set."""
        return {ext.strip().lower() for ext in v.split(",")}

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()

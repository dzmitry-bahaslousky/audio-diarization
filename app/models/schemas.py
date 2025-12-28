"""Pydantic models for API request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# Enums
class TranscriptionStatus(str, Enum):
    """Status of transcription job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


# Request Models
class TranscriptionRequest(BaseModel):
    """Request parameters for transcription (excluding file)."""
    num_speakers: Optional[int] = Field(None, ge=1, le=10, description="Expected number of speakers")
    whisper_model: WhisperModel = Field(WhisperModel.MEDIUM, description="Whisper model to use")
    language: Optional[str] = Field(None, max_length=5, description="Audio language code (e.g., 'en', 'es')")

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v: Optional[str]) -> Optional[str]:
        """Ensure language code is lowercase."""
        return v.lower() if v else None


# Response Models
class TranscriptionJobResponse(BaseModel):
    """Response when transcription job is created."""
    job_id: str = Field(..., description="Unique job identifier")
    status: TranscriptionStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TranscriptionStatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    status: TranscriptionStatus
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class SpeakerSegment(BaseModel):
    """Individual speaker segment in transcription."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")


class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    job_id: str
    status: TranscriptionStatus
    audio_duration: float = Field(..., description="Duration in seconds")
    num_speakers: int = Field(..., description="Number of speakers detected")
    language: Optional[str] = Field(None, description="Detected language")
    segments: list[SpeakerSegment] = Field(..., description="Transcribed segments with speakers")
    created_at: datetime
    completed_at: datetime
    processing_time: float = Field(..., description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    job_id: Optional[str] = Field(None, description="Job ID if applicable")

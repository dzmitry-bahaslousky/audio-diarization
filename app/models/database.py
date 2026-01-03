"""Database models for job storage."""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, Float, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.models.schemas import TranscriptionStatus

Base = declarative_base()


class TranscriptionJob(Base):
    """Database model for transcription jobs."""

    __tablename__ = "transcription_jobs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Job metadata
    filename = Column(String(255), nullable=False)
    status = Column(SQLEnum(TranscriptionStatus), nullable=False, default=TranscriptionStatus.PENDING)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Processing parameters
    whisper_model = Column(String(50), nullable=False)
    language = Column(String(10), nullable=True)
    num_speakers = Column(Integer, nullable=True)  # Hint for expected speakers

    # Results (stored as JSON)
    audio_duration = Column(Float, nullable=True)
    detected_language = Column(String(10), nullable=True)
    detected_speakers = Column(Integer, nullable=True)  # Actual detected speaker count
    segments = Column(JSON, nullable=True)  # List of segment dicts with speaker labels
    full_text = Column(Text, nullable=True)
    speaker_timeline = Column(Text, nullable=True)  # Human-readable speaker timeline
    speaker_groups = Column(JSON, nullable=True)  # Segments grouped by speaker

    # Error handling
    error_message = Column(Text, nullable=True)

    # File paths (for cleanup)
    upload_path = Column(String(500), nullable=True)
    wav_path = Column(String(500), nullable=True)

    def to_dict(self):
        """Convert to dict format compatible with current API."""
        result_dict = {
            'audio_duration': self.audio_duration,
            'language': self.detected_language,
            'num_speakers': self.detected_speakers,
            'segments': self.segments or [],
            'text': self.full_text or '',
            'speaker_timeline': self.speaker_timeline or '',
            'speaker_groups': self.speaker_groups or {},
            'num_segments': len(self.segments) if self.segments else 0
        }

        return {
            'status': self.status,
            'filename': self.filename,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'whisper_model': self.whisper_model,
            'language': self.language,
            'num_speakers': self.num_speakers,
            'error': self.error_message,
            'result': result_dict if self.status == TranscriptionStatus.COMPLETED else None
        }

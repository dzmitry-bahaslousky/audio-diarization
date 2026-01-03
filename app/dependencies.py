"""Dependency injection for services and repositories.

This module provides factory functions for creating service and repository instances
using FastAPI's dependency injection system. Services are cached as singletons per
worker process for efficiency.
"""

from functools import lru_cache
from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.audio_processor import AudioProcessor
from app.services.transcription import TranscriptionService
from app.services.export import ExportService
from app.services.transcription_workflow import TranscriptionWorkflow
from app.repositories.transcription_repository import TranscriptionRepository


# Service factories (singleton pattern for stateful services)
# These are cached per worker process and reused across requests

@lru_cache()
def get_audio_processor() -> AudioProcessor:
    """
    Get or create audio processor singleton.

    Returns:
        AudioProcessor instance
    """
    return AudioProcessor()


@lru_cache()
def get_transcription_service() -> TranscriptionService:
    """
    Get or create transcription service singleton.

    Models are loaded lazily on first use and cached.

    Returns:
        TranscriptionService instance
    """
    return TranscriptionService()


@lru_cache()
def get_export_service() -> ExportService:
    """
    Get or create export service singleton.

    Stateless service for formatting output.

    Returns:
        ExportService instance
    """
    return ExportService()


@lru_cache()
def get_transcription_workflow() -> TranscriptionWorkflow:
    """
    Get or create transcription workflow singleton.

    Orchestrates file upload and validation workflow.

    Returns:
        TranscriptionWorkflow instance
    """
    return TranscriptionWorkflow()


# Repository factory (new instance per request for thread safety)

def get_transcription_repository(
    db: Session = Depends(get_db)
) -> TranscriptionRepository:
    """
    Create transcription repository for current request.

    Note: Repository instances are NOT cached because they hold database
    sessions that should be scoped to individual requests.

    Args:
        db: Database session from FastAPI dependency

    Returns:
        TranscriptionRepository instance
    """
    return TranscriptionRepository(db)


# Type aliases for cleaner route signatures
# Use these with FastAPI's Depends() for dependency injection

AudioProcessorDep = Annotated[AudioProcessor, Depends(get_audio_processor)]
TranscriptionServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
TranscriptionWorkflowDep = Annotated[TranscriptionWorkflow, Depends(get_transcription_workflow)]
TranscriptionRepositoryDep = Annotated[TranscriptionRepository, Depends(get_transcription_repository)]

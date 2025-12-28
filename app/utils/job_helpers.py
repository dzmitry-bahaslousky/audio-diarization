"""Helper utilities for job status checking and retrieval.

This module provides reusable utility functions for common job-related
operations, eliminating code duplication across route handlers.
"""

from typing import Optional
from fastapi import HTTPException, status
import logging

from app.models.database import TranscriptionJob
from app.models.schemas import TranscriptionStatus
from app.repositories.transcription_repository import TranscriptionRepository

logger = logging.getLogger(__name__)


def get_job_or_404(
    job_id: str,
    repo: TranscriptionRepository
) -> TranscriptionJob:
    """
    Get job by ID or raise 404 error.

    Args:
        job_id: Job identifier
        repo: Repository instance

    Returns:
        TranscriptionJob

    Raises:
        HTTPException: 404 if job not found
    """
    job = repo.get_job(job_id)
    if not job:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    return job


def check_job_completed(job: TranscriptionJob) -> None:
    """
    Check if job is completed, raise appropriate error if not.

    Args:
        job: Job to check

    Raises:
        HTTPException: 425 if not completed, 500 if failed
    """
    if job.status in [TranscriptionStatus.PROCESSING, TranscriptionStatus.PENDING]:
        logger.info(f"Job {job.id} not yet completed, status: {job.status}")
        raise HTTPException(
            status_code=425,  # Too Early
            detail="Job not completed yet. Check status endpoint for progress."
        )

    if job.status == TranscriptionStatus.FAILED:
        logger.error(f"Job {job.id} failed: {job.error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {job.error_message or 'Unknown error'}"
        )


def get_job_metadata(job: TranscriptionJob) -> dict:
    """
    Extract common metadata from job.

    Args:
        job: Transcription job

    Returns:
        dict: Metadata dictionary with common fields
    """
    return {
        'filename': job.filename,
        'created_at': job.created_at,
        'completed_at': job.completed_at,
        'whisper_model': job.whisper_model,
        'language': job.language,
        'num_speakers': job.num_speakers
    }


def format_job_status_response(job: TranscriptionJob, job_id: str) -> dict:
    """
    Format job for status response.

    Args:
        job: Transcription job
        job_id: Job identifier

    Returns:
        dict: Formatted status response
    """
    return {
        'job_id': job_id,
        'status': job.status.value if isinstance(job.status, TranscriptionStatus) else job.status,
        'filename': job.filename,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'error': job.error_message
    }


def format_job_result_response(job: TranscriptionJob, job_id: str) -> dict:
    """
    Format job for result response.

    Args:
        job: Transcription job
        job_id: Job identifier

    Returns:
        dict: Formatted result response
    """
    job_dict = job.to_dict()

    return {
        'job_id': job_id,
        'status': job.status.value,
        'filename': job.filename,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat(),
        **job_dict.get('result', {})
    }


def get_job_result_data(job: TranscriptionJob) -> tuple[dict, dict]:
    """
    Extract result data and metadata from completed job.

    Args:
        job: Transcription job (must be completed)

    Returns:
        tuple: (result_dict, metadata_dict)
    """
    job_dict = job.to_dict()
    result = job_dict.get('result', {})
    metadata = get_job_metadata(job)

    return result, metadata

"""Transcription API routes."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    Form,
    Query,
    Depends,
    status
)
from fastapi.responses import PlainTextResponse, JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import (
    TranscriptionRepositoryDep,
    TranscriptionWorkflowDep,
    ExportServiceDep
)
from app.models.schemas import (
    TranscriptionJobResponse,
    TranscriptionStatus,
    WhisperModel,
    ErrorResponse
)
from app.celery_app import process_transcription_task
from app.exceptions import JobCreationError, InsufficientDiskSpaceError
from app.utils.job_helpers import (
    get_job_or_404,
    check_job_completed,
    get_job_metadata,
    format_job_status_response,
    format_job_result_response,
    get_job_result_data
)

router = APIRouter()
logger = logging.getLogger(__name__)


def optional_int_form(
    num_speakers: str = Form(
        "",
        description="Expected number of speakers (1-10)",
        example="2"
    )
) -> Optional[int]:
    """
    Convert form field to optional int, treating empty strings as None.

    This handles HTML forms where empty fields send "" instead of null.
    Validates that num_speakers is between 1 and 10 if provided.

    Args:
        num_speakers: Form field value

    Returns:
        Parsed integer or None if empty

    Raises:
        HTTPException: If value is not empty and not a valid integer or out of range
    """
    if num_speakers == "" or num_speakers is None:
        return None

    try:
        num = int(num_speakers)
        if not (1 <= num <= 10):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="num_speakers must be between 1 and 10"
            )
        return num
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"num_speakers must be a valid integer, got: {num_speakers}"
        )


def optional_language_form(
    language: str = Form(
        "",
        description="Audio language code (e.g., 'en', 'es')",
        max_length=5,
        example="en"
    )
) -> Optional[str]:
    """
    Convert form field to optional language code, treating empty strings as None.

    Validates that language code is max 5 characters if provided.

    Args:
        language: Form field value

    Returns:
        String value or None if empty

    Raises:
        HTTPException: If language code is too long
    """
    if language == "" or language is None:
        return None

    language = language.strip()

    if len(language) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Language code must be max 5 characters"
        )

    return language


@router.post(
    "/transcribe",
    response_model=TranscriptionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or parameters"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    repo: TranscriptionRepositoryDep,
    workflow: TranscriptionWorkflowDep,
    num_speakers: Annotated[Optional[int], Depends(optional_int_form)] = None,
    language: Annotated[Optional[str], Depends(optional_language_form)] = None,
    whisper_model: Annotated[WhisperModel, Form(description="Whisper model size")] = WhisperModel.MEDIUM
) -> TranscriptionJobResponse:
    """
    Upload audio file and start transcription with speaker diarization.

    Args:
        file: Audio file (MP3, WAV, M4A, etc.)
        num_speakers: Optional hint for number of speakers
        whisper_model: Whisper model to use (larger = more accurate but slower)
        language: Optional language code for better accuracy

    Returns:
        TranscriptionJobResponse with job_id for status tracking

    Raises:
        HTTPException: If file validation fails or upload error occurs
    """
    logger.info(f"Received transcription request: {file.filename}")

    # Get file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    # Validate file using workflow
    try:
        workflow.validate_upload(file, file.filename, file_size)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Validate parameters using workflow
    try:
        params = workflow.validate_params(num_speakers, whisper_model, language)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Save uploaded file using workflow
    try:
        upload_path = await workflow.save_upload(file, file.filename)
    except InsufficientDiskSpaceError as e:
        raise HTTPException(
            status_code=507,  # 507 Insufficient Storage
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    # Create database record using repository (with automatic transaction management)
    try:
        job = repo.create_job(
            filename=file.filename,
            upload_path=str(upload_path),
            **params  # Use validated params from workflow
        )
        job_id = str(job.id)
        logger.info(f"Created job {job_id} for file {file.filename}")
    except JobCreationError as e:
        # Clean up uploaded file on database error using workflow
        workflow.cleanup_upload(upload_path)
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create transcription job"
        )

    # Queue Celery task
    process_transcription_task.delay(job_id)

    return TranscriptionJobResponse(
        job_id=job_id,
        status=TranscriptionStatus.PENDING,
        message=f"Transcription job queued for {file.filename}"
    )


@router.get(
    "/status/{job_id}",
    response_model=dict,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"}
    }
)
async def get_transcription_status(
    job_id: str,
    repo: TranscriptionRepositoryDep
) -> dict:
    """
    Get the status of a transcription job.

    Args:
        job_id: Unique job identifier

    Returns:
        Job status information

    Raises:
        HTTPException: If job not found
    """
    logger.info(f"Status check for job {job_id}")

    job = get_job_or_404(job_id, repo)
    return format_job_status_response(job, job_id)


@router.get(
    "/result/{job_id}",
    response_model=dict,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        425: {"model": ErrorResponse, "description": "Job not completed yet"},
        500: {"model": ErrorResponse, "description": "Job failed"}
    }
)
async def get_transcription_result(
    job_id: str,
    repo: TranscriptionRepositoryDep
) -> dict:
    """
    Get the transcription result for a completed job.

    Args:
        job_id: Unique job identifier

    Returns:
        Complete transcription with timestamps

    Raises:
        HTTPException: If job not found, not completed, or failed
    """
    logger.info(f"Result request for job {job_id}")

    job = get_job_or_404(job_id, repo)
    check_job_completed(job)
    return format_job_result_response(job, job_id)


@router.get(
    "/export/{job_id}/txt",
    response_class=PlainTextResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        425: {"model": ErrorResponse, "description": "Job not completed yet"},
        500: {"model": ErrorResponse, "description": "Job failed"}
    }
)
async def export_txt(
    job_id: str,
    repo: TranscriptionRepositoryDep,
    export_service: ExportServiceDep,
    format: str = Query("timeline", description="Format: 'simple', 'timeline', or 'detailed'")
) -> str:
    """
    Export transcription as plain text.

    Args:
        job_id: Unique job identifier
        format: Export format variant
            - 'simple': Just the transcription text
            - 'timeline': Text with speaker labels
            - 'detailed': Text with timestamps and speakers

    Returns:
        Formatted plain text transcription

    Raises:
        HTTPException: If job not found, not completed, or failed
    """
    logger.info(f"TXT export request for job {job_id}, format={format}")

    job = get_job_or_404(job_id, repo)
    check_job_completed(job)
    result, metadata = get_job_result_data(job)

    # Format based on requested style
    if format == "simple":
        content = export_service.format_txt_simple(result, metadata)
    elif format == "detailed":
        content = export_service.format_txt_detailed(result, metadata)
    else:  # default to timeline
        content = export_service.format_txt_timeline(result, metadata)

    return content


@router.get(
    "/export/{job_id}/json",
    response_class=JSONResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        425: {"model": ErrorResponse, "description": "Job not completed yet"},
        500: {"model": ErrorResponse, "description": "Job failed"}
    }
)
async def export_json(
    job_id: str,
    repo: TranscriptionRepositoryDep,
    export_service: ExportServiceDep
) -> JSONResponse:
    """
    Export transcription as structured JSON.

    Args:
        job_id: Unique job identifier

    Returns:
        Complete transcription data in JSON format

    Raises:
        HTTPException: If job not found, not completed, or failed
    """
    logger.info(f"JSON export request for job {job_id}")

    job = get_job_or_404(job_id, repo)
    check_job_completed(job)
    result, metadata = get_job_result_data(job)

    json_content = export_service.format_json(result, metadata, job_id)

    return JSONResponse(
        content=json.loads(json_content),
        media_type="application/json"
    )

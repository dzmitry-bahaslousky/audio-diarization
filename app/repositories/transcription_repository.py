"""Repository for transcription job data access with automatic transaction management.

This module implements the Repository pattern to encapsulate all database operations
for transcription jobs, providing:
- Clean separation between data access and business logic
- Automatic transaction management with rollback on errors
- Type-safe database operations
- Centralized error handling for database operations

Design Patterns:
- Repository Pattern: Encapsulates data access logic
- Unit of Work Pattern: Transaction management via context managers
- Dependency Injection: Repository receives session from caller
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Generator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import logging

from app.models.database import TranscriptionJob
from app.models.schemas import TranscriptionStatus
from app.exceptions import (
    JobCreationError,
    JobNotFoundError,
    DatabaseConnectionError,
    ValidationError
)

logger = logging.getLogger(__name__)


class TranscriptionRepository:
    """Handles all database operations for transcription jobs with transaction safety."""

    def __init__(self, db: Session):
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy session (injected dependency)

        Note:
            The repository does NOT own the session lifecycle.
            The session is managed by the caller (via get_repository context manager).
        """
        self.db = db
        self._validate_session()

    def _validate_session(self) -> None:
        """Validate that the database session is properly configured.

        Raises:
            ValueError: If session is invalid
        """
        if self.db is None:
            raise ValueError("Database session cannot be None")

        if not isinstance(self.db, Session):
            raise ValueError(
                f"Expected SQLAlchemy Session, got {type(self.db).__name__}"
            )

    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """Context manager for database transactions with automatic rollback.

        Provides automatic transaction management:
        - Commits on successful completion
        - Rolls back on any exception
        - Re-raises exceptions after rollback

        Usage:
            ```python
            with repo.transaction():
                # Database operations
                job = TranscriptionJob(...)
                self.db.add(job)
                # Automatic commit on success, rollback on error
            ```

        Yields:
            Database session for transaction operations

        Raises:
            DatabaseConnectionError: If database error occurs
            Exception: Other exceptions are re-raised after rollback

        Example:
            ```python
            # Multiple operations in a single transaction
            with repo.transaction():
                job = repo.create_job(...)
                repo.update_status(job.id, TranscriptionStatus.PROCESSING)
                # Both operations committed together
            ```
        """
        try:
            yield self.db
            self.db.commit()
            logger.debug("Transaction committed successfully")

        except IntegrityError as e:
            logger.error(f"Database integrity error, rolling back: {e}")
            self.db.rollback()
            raise JobCreationError(
                "Database constraint violation",
                details={"error": str(e)},
                original_error=e
            )

        except SQLAlchemyError as e:
            logger.error(f"Database error, rolling back transaction: {e}")
            self.db.rollback()
            raise DatabaseConnectionError(
                "Database operation failed",
                details={"error": str(e)},
                original_error=e
            )

        except Exception as e:
            logger.error(f"Unexpected error, rolling back transaction: {e}")
            self.db.rollback()
            raise

    def create_job(
        self,
        filename: str,
        upload_path: str,
        whisper_model: str,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> TranscriptionJob:
        """
        Create a new transcription job with transaction safety.

        Args:
            filename: Original filename
            upload_path: Path to uploaded file
            whisper_model: Whisper model size
            language: Optional language code
            num_speakers: Expected number of speakers (hint for diarization)

        Returns:
            TranscriptionJob: Created job object

        Raises:
            JobCreationError: If job creation fails
        """
        try:
            with self.transaction():
                job = TranscriptionJob(
                    filename=filename,
                    status=TranscriptionStatus.PENDING,
                    whisper_model=whisper_model,
                    language=language,
                    num_speakers=num_speakers,
                    upload_path=upload_path,
                    created_at=datetime.utcnow()
                )
                self.db.add(job)
                self.db.flush()  # Get ID before commit
                self.db.refresh(job)
                logger.info(f"Created job {job.id} for file {filename}")
                return job
        except DatabaseConnectionError as e:
            logger.error(f"Failed to create job for {filename}: {e}")
            raise JobCreationError(f"Could not create transcription job: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating job: {e}")
            raise JobCreationError(f"Failed to create job: {e}")

    def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        """
        Retrieve job by ID.

        Args:
            job_id: Job identifier

        Returns:
            TranscriptionJob or None if not found

        Raises:
            DatabaseConnectionError: If database error occurs
        """
        try:
            job = self.db.query(TranscriptionJob).filter(
                TranscriptionJob.id == job_id
            ).first()

            if job:
                logger.debug(f"Retrieved job {job_id}, status: {job.status}")
            else:
                logger.warning(f"Job {job_id} not found")

            return job
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve job {job_id}: {e}")
            raise DatabaseConnectionError(f"Failed to retrieve job: {e}")

    def update_status(
        self,
        job_id: str,
        status: TranscriptionStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update job status with transaction safety.

        Args:
            job_id: Job identifier
            status: New status
            error_message: Optional error message for failures

        Returns:
            bool: True if updated, False if job not found

        Raises:
            DatabaseConnectionError: If database error occurs
        """
        try:
            with self.transaction():
                job = self.get_job(job_id)
                if not job:
                    logger.error(f"Cannot update status: job {job_id} not found")
                    return False

                old_status = job.status
                job.status = status

                if error_message:
                    job.error_message = error_message

                # Set completed_at for terminal states
                if status in [TranscriptionStatus.COMPLETED, TranscriptionStatus.FAILED]:
                    job.completed_at = datetime.utcnow()

                logger.info(
                    f"Updated job {job_id} status: {old_status} -> {status}"
                    + (f" (error: {error_message})" if error_message else "")
                )

                return True
        except DatabaseConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to update status for job {job_id}: {e}")
            raise DatabaseConnectionError(f"Failed to update job status: {e}")

    def update_results(
        self,
        job_id: str,
        audio_duration: float,
        detected_language: str,
        segments: List[dict],
        full_text: str,
        wav_path: Optional[str] = None,
        detected_speakers: Optional[int] = None,
        speaker_timeline: Optional[str] = None,
        speaker_groups: Optional[dict] = None
    ) -> bool:
        """
        Update job with processing results and mark as completed.

        Args:
            job_id: Job identifier
            audio_duration: Duration of audio in seconds
            detected_language: Detected language code
            segments: List of transcription segments (with speaker labels from WhisperX)
            full_text: Complete transcription text
            wav_path: Optional path to converted WAV file
            detected_speakers: Number of speakers detected by WhisperX diarization
            speaker_timeline: Human-readable speaker timeline
            speaker_groups: Segments grouped by speaker

        Returns:
            bool: True if updated, False if job not found

        Raises:
            DatabaseConnectionError: If database error occurs
        """
        try:
            with self.transaction():
                job = self.get_job(job_id)
                if not job:
                    logger.error(f"Cannot update results: job {job_id} not found")
                    return False

                job.status = TranscriptionStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.audio_duration = audio_duration
                job.detected_language = detected_language
                job.segments = segments
                job.full_text = full_text

                if wav_path:
                    job.wav_path = wav_path

                # Update speaker-related fields from WhisperX diarization
                if detected_speakers is not None:
                    job.detected_speakers = detected_speakers

                if speaker_timeline:
                    job.speaker_timeline = speaker_timeline

                if speaker_groups:
                    job.speaker_groups = speaker_groups

                logger.info(
                    f"Updated job {job_id} with results: "
                    f"{len(segments)} segments, "
                    f"{audio_duration:.2f}s duration"
                    + (f", {detected_speakers} speakers" if detected_speakers else "")
                )

                return True
        except DatabaseConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to update results for job {job_id}: {e}")
            raise DatabaseConnectionError(f"Failed to update job results: {e}")

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the database.

        Args:
            job_id: Job identifier

        Returns:
            bool: True if deleted, False if job not found

        Raises:
            DatabaseConnectionError: If database error occurs
        """
        try:
            with self.transaction():
                job = self.get_job(job_id)
                if not job:
                    logger.warning(f"Cannot delete: job {job_id} not found")
                    return False

                self.db.delete(job)
                logger.info(f"Deleted job {job_id}")

                return True
        except DatabaseConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise DatabaseConnectionError(f"Failed to delete job: {e}")

    def list_jobs(
        self,
        status: Optional[TranscriptionStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TranscriptionJob]:
        """
        List jobs with optional filtering.

        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of TranscriptionJob objects

        Raises:
            DatabaseConnectionError: If database error occurs
        """
        try:
            query = self.db.query(TranscriptionJob)

            if status:
                query = query.filter(TranscriptionJob.status == status)

            query = query.order_by(TranscriptionJob.created_at.desc())
            query = query.limit(limit).offset(offset)

            jobs = query.all()
            logger.debug(f"Listed {len(jobs)} jobs (status={status}, limit={limit})")

            return jobs
        except SQLAlchemyError as e:
            logger.error(f"Failed to list jobs: {e}")
            raise DatabaseConnectionError(f"Failed to list jobs: {e}")

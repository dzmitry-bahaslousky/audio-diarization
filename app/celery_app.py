"""Celery application and background tasks with comprehensive error handling."""

from celery import Celery
from pathlib import Path
import logging
import shutil

from app.config import settings
from app.database import get_repository
from app.repositories.transcription_repository import TranscriptionRepository
from app.models.schemas import TranscriptionStatus
from app.services.audio_processor import AudioProcessor
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService
from app.exceptions import (
    TransientError,
    PermanentError,
    ModelLoadError,
    AudioProcessingError,
    InvalidAudioError,
    InsufficientDiskSpaceError,
    TranscriptionError,
    DiarizationError,
    AlignmentError
)
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "audio_diarization",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
)

# Initialize services (singleton pattern for Celery workers)
# These are initialized once per worker process and reused across tasks
audio_processor = AudioProcessor()
transcription_service = TranscriptionService()
diarization_service = DiarizationService()
alignment_service = AlignmentService()


def check_disk_space(min_free_gb: float = 5.0):
    """
    Check if sufficient disk space is available.

    Args:
        min_free_gb: Minimum free space required in GB

    Raises:
        InsufficientDiskSpaceError: If disk space is insufficient
    """
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

    logger.debug(f"Disk space OK: upload={upload_free_gb:.2f}GB, output={output_free_gb:.2f}GB")


@celery_app.task(
    bind=True,
    name="process_transcription",
    autoretry_for=(TransientError, ModelLoadError, AudioProcessingError),
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
    max_retries=3,
    default_retry_delay=60,  # Start with 60 seconds
    acks_late=True,  # Only ack after task completes
    reject_on_worker_lost=True,  # Re-queue if worker crashes
)
def process_transcription_task(self, job_id: str):
    """
    Celery task for processing transcription with diarization.

    Implements automatic retry for transient errors with exponential backoff.
    Permanent errors (invalid audio, disk space) fail immediately.

    Args:
        job_id: UUID of the transcription job
    """
    logger.info(f"Starting Celery task for job {job_id} (attempt {self.request.retries + 1}/{self.max_retries + 1})")

    # Track file paths for cleanup
    file_path = None
    wav_path = None
    should_cleanup = False

    # Check disk space before processing
    try:
        check_disk_space()
    except InsufficientDiskSpaceError as e:
        logger.error(f"Insufficient disk space for job {job_id}: {e}")
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(job_id, TranscriptionStatus.FAILED, str(e))
        raise  # Don't retry disk space errors

    # Get job details and update status to PROCESSING
    with get_repository(TranscriptionRepository) as repo:
        job = repo.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        repo.update_status(job_id, TranscriptionStatus.PROCESSING)

        file_path = Path(job.upload_path)
        language = job.language
        whisper_model = job.whisper_model
        num_speakers = job.num_speakers

    # Generate wav_path before processing so we can always clean it up if created
    wav_path = settings.output_dir / f"{job_id}_converted.wav"

    try:
        # Run async processing in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _process_audio_async(job_id, file_path, wav_path, language, whisper_model, num_speakers)
        )
        loop.close()

        logger.info(f"Job {job_id} completed successfully")
        should_cleanup = True  # Success - cleanup files

    except (DiarizationError, TranscriptionError, AlignmentError) as e:
        # ML processing errors are permanent - don't retry
        # (timeout, model failures, alignment issues won't fix themselves)
        logger.error(f"ML processing error for job {job_id}: {e}")
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(job_id, TranscriptionStatus.FAILED, f"Processing error: {e}")
        should_cleanup = True  # Cleanup on permanent error
        raise PermanentError(f"ML processing failed: {e}")

    except PermanentError as e:
        # Don't retry permanent errors (invalid audio, disk space, etc.)
        logger.error(f"Permanent error for job {job_id}: {e}")
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(job_id, TranscriptionStatus.FAILED, f"Permanent error: {e}")
        should_cleanup = True  # Cleanup on permanent error
        raise  # Don't retry

    except TransientError as e:
        # Log retry attempt for transient errors
        logger.warning(
            f"Transient error for job {job_id}, will retry "
            f"(attempt {self.request.retries + 1}/{self.max_retries + 1}): {e}"
        )
        # Update status to show retry
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(
                job_id,
                TranscriptionStatus.PROCESSING,
                error_message=f"Retrying after error: {e}"
            )
        # DON'T cleanup - we'll retry
        raise  # Automatic retry via autoretry_for

    except Exception as e:
        # Unknown errors - retry cautiously
        logger.error(f"Unknown error for job {job_id}: {e}", exc_info=True)

        # Don't retry too many times for unknown errors
        if self.request.retries >= 1:
            logger.error(f"Job {job_id} failed after {self.request.retries + 1} attempts")
            with get_repository(TranscriptionRepository) as repo:
                repo.update_status(
                    job_id,
                    TranscriptionStatus.FAILED,
                    f"Failed after {self.request.retries + 1} attempts: {e}"
                )
            should_cleanup = True  # Final failure - cleanup
            raise PermanentError(f"Job failed after retries: {e}")

        # Retry once for unknown errors - DON'T cleanup yet
        logger.warning(f"Retrying job {job_id} after unknown error (attempt 1/2)")
        raise self.retry(exc=e, countdown=30)

    finally:
        # Only cleanup if we're not going to retry
        if should_cleanup and file_path is not None:
            _cleanup_files(job_id, file_path, wav_path)


async def _process_audio_async(
    job_id: str,
    file_path: Path,
    wav_path: Path,
    language: str,
    whisper_model: str,
    num_speakers: int
) -> None:
    """
    Async helper for audio processing with comprehensive error handling.

    Args:
        job_id: Job identifier
        file_path: Path to uploaded audio file
        wav_path: Path where converted WAV file should be saved
        language: Language code for transcription
        whisper_model: Whisper model size to use
        num_speakers: Number of speakers hint

    Raises:
        Various exceptions from services (InvalidAudioError, TranscriptionError, etc.)
    """
    # Convert to WAV
    await audio_processor.convert_to_wav(file_path, wav_path)

    # Get audio info
    audio_info = await audio_processor.get_audio_info(wav_path)

    # Run transcription and diarization in parallel for performance
    transcription_result, diarization_result = await asyncio.gather(
        transcription_service.transcribe(wav_path, language, whisper_model),
        diarization_service.diarize(wav_path, num_speakers)
    )

    # Format and align segments
    trans_segments = transcription_service.format_segments(transcription_result)
    aligned_segments = alignment_service.align(trans_segments, diarization_result['segments'])
    speaker_timeline = alignment_service.get_speaker_timeline(aligned_segments)
    speaker_groups = alignment_service.group_by_speaker(aligned_segments)

    # Update database with results using repository pattern
    with get_repository(TranscriptionRepository) as repo:
        success = repo.update_results(
            job_id=job_id,
            audio_duration=audio_info['duration'],
            detected_language=transcription_result.get('language', 'unknown'),
            detected_speakers=diarization_result['num_speakers'],
            segments=aligned_segments,
            full_text=transcription_result.get('text', ''),
            speaker_timeline=speaker_timeline,
            speaker_groups=speaker_groups,
            wav_path=str(wav_path)
        )

        if not success:
            logger.error(f"Failed to update results for job {job_id}")
            raise Exception("Failed to update job results in database")

    logger.info(
        f"Processing complete for job {job_id}: "
        f"{len(aligned_segments)} segments, "
        f"{diarization_result['num_speakers']} speakers, "
        f"{audio_info['duration']:.2f}s duration"
    )


def _cleanup_files(job_id: str, *file_paths: Path):
    """
    Clean up temporary files with comprehensive error handling.

    Args:
        job_id: Job identifier (for logging)
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        if file_path is None:
            continue

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file for job {job_id}: {file_path}")
        except PermissionError as e:
            logger.error(f"Permission denied deleting {file_path} for job {job_id}: {e}")
        except OSError as e:
            logger.error(f"OS error deleting {file_path} for job {job_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting {file_path} for job {job_id}: {e}")

"""Celery application and background tasks with comprehensive error handling."""

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from pathlib import Path
import logging
import shutil

# Fix PyTorch 2.6 weights_only=True default for model loading
# Monkey-patch torch.load to use weights_only=False for trusted model sources
# MUST be done before importing any model-loading code
import torch

_original_torch_load = torch.load

def _trusted_load(*args, **kwargs):
    """Wrapper for torch.load that defaults to weights_only=False.

    PyTorch 2.6+ defaults to weights_only=True for security, but WhisperX
    models use pickle features that require weights_only=False.
    This is safe for models from trusted sources (OpenAI, HuggingFace).
    """
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _trusted_load
print("[CELERY INIT] Patched torch.load to use weights_only=False for model loading")

from app.config import settings
from app.database import get_repository
from app.repositories.transcription_repository import TranscriptionRepository
from app.models.schemas import TranscriptionStatus
from app.services.audio_processor import AudioProcessor
from app.services.transcription import TranscriptionService
from app.exceptions import (
    TransientError,
    PermanentError,
    ModelLoadError,
    AudioProcessingError,
    InvalidAudioError,
    InsufficientDiskSpaceError,
    TranscriptionError,
    AlignmentError,
    DiarizationError
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
    task_time_limit=settings.celery_task_time_limit,  # Configurable via env
    task_soft_time_limit=settings.celery_task_soft_time_limit,  # Configurable via env
)

# Initialize services (singleton pattern for Celery workers)
# These are initialized once per worker process and reused across tasks
audio_processor = AudioProcessor()
transcription_service = TranscriptionService()


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
    autoretry_for=(TransientError, ModelLoadError, AudioProcessingError, AlignmentError, DiarizationError),
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

    except TranscriptionError as e:
        # ML processing errors are permanent - don't retry
        # (timeout, model failures won't fix themselves)
        logger.error(f"Transcription error for job {job_id}: {e}")
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(job_id, TranscriptionStatus.FAILED, f"Transcription error: {e}")
        should_cleanup = True  # Cleanup on permanent error
        raise PermanentError(f"Transcription failed: {e}")

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

    except SoftTimeLimitExceeded:
        # Task exceeded soft time limit - file is too long for current timeout settings
        timeout_mins = settings.celery_task_soft_time_limit // 60
        logger.error(
            f"Job {job_id} exceeded soft time limit ({timeout_mins} minutes). "
            "Audio file may be too long for current timeout settings."
        )
        error_msg = (
            f"Processing exceeded time limit ({timeout_mins} minutes). "
            f"For long audio files, increase CELERY_TASK_SOFT_TIME_LIMIT and "
            f"CELERY_TASK_TIME_LIMIT in your .env file."
        )
        with get_repository(TranscriptionRepository) as repo:
            repo.update_status(job_id, TranscriptionStatus.FAILED, error_msg)
        should_cleanup = True  # Cleanup on timeout
        raise PermanentError(error_msg)

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
    num_speakers: int = None
) -> None:
    """
    Async helper for audio processing with WhisperX transcription and diarization.

    Args:
        job_id: Job identifier
        file_path: Path to uploaded audio file
        wav_path: Path where converted WAV file should be saved
        language: Language code for transcription
        whisper_model: Whisper model size to use
        num_speakers: Expected number of speakers (hint for WhisperX diarization)

    Raises:
        Various exceptions from services (InvalidAudioError, TranscriptionError, etc.)
    """
    # Convert to WAV
    await audio_processor.convert_to_wav(file_path, wav_path)

    # Get audio info
    audio_info = await audio_processor.get_audio_info(wav_path)

    logger.info(
        f"Processing audio file with duration {audio_info['duration']:.2f}s"
        + (f", {num_speakers} expected speakers" if num_speakers else "")
    )

    # Run WhisperX transcription with alignment and diarization
    transcription_result = await transcription_service.transcribe(
        wav_path,
        language,
        whisper_model,
        num_speakers
    )

    # Format segments (includes speaker labels from WhisperX diarization)
    segments = transcription_service.format_segments(transcription_result)

    # Extract metadata from results
    full_text = transcription_result.get('text', '')
    detected_language = transcription_result.get('language', 'unknown')

    # Extract speaker data from WhisperX results
    detected_speakers = len(set(
        seg.get('speaker') for seg in segments if seg.get('speaker')
    )) if any(seg.get('speaker') for seg in segments) else None

    # Build speaker timeline (human-readable format)
    speaker_timeline = None
    if detected_speakers:
        timeline_lines = []
        for seg in segments:
            if seg.get('speaker'):
                timestamp = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
                timeline_lines.append(f"[{timestamp}] {seg['speaker']}: {seg['text'][:50]}...")
        speaker_timeline = '\n'.join(timeline_lines) if timeline_lines else None

    # Build speaker groups (segments grouped by speaker)
    speaker_groups = {}
    if detected_speakers:
        for seg in segments:
            speaker = seg.get('speaker')
            if speaker:
                if speaker not in speaker_groups:
                    speaker_groups[speaker] = []
                speaker_groups[speaker].append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })

    # Update database with results using repository pattern
    with get_repository(TranscriptionRepository) as repo:
        success = repo.update_results(
            job_id=job_id,
            audio_duration=audio_info['duration'],
            detected_language=detected_language,
            segments=segments,
            full_text=full_text,
            wav_path=str(wav_path),
            detected_speakers=detected_speakers,
            speaker_timeline=speaker_timeline,
            speaker_groups=speaker_groups if speaker_groups else None
        )

        if not success:
            logger.error(f"Failed to update results for job {job_id}")
            raise Exception("Failed to update job results in database")

    logger.info(
        f"Processing complete for job {job_id}: "
        f"{len(segments)} segments, "
        f"{audio_info['duration']:.2f}s duration"
        + (f", {detected_speakers} speakers detected" if detected_speakers else "")
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

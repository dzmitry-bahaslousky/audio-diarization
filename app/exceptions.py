"""Custom application exceptions for audio diarization service.

This module defines a hierarchical exception structure that enables:
- Clear distinction between transient and permanent failures
- Automatic retry logic based on exception type
- Detailed error context for debugging
- Type-safe error handling with proper inheritance
"""

from typing import Optional, Dict, Any


class AudioDiarizationException(Exception):
    """Base exception for all application errors.

    All custom exceptions inherit from this base class to enable
    type-safe exception handling and centralized error tracking.

    Attributes:
        message: Human-readable error message
        details: Additional context for debugging
        original_error: Wrapped exception if applicable
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize exception with message and optional context.

        Args:
            message: Human-readable error message
            details: Additional context dictionary
            original_error: Original exception if wrapping
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Format exception as string with context."""
        base_msg = self.message
        if self.details:
            details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} ({details_str})"
        if self.original_error:
            base_msg = f"{base_msg} | Caused by: {self.original_error}"
        return base_msg


# Transient errors (should retry)
class TransientError(AudioDiarizationException):
    """Errors that may succeed on retry.

    These errors indicate temporary conditions (network issues, resource
    contention, rate limits) that are likely to resolve with retry.

    Celery tasks configured with autoretry_for will automatically retry
    when these exceptions are raised.
    """
    pass


class ModelLoadError(TransientError):
    """Failed to load ML model (may be temporary resource issue).

    Common causes:
    - GPU out of memory (temporary)
    - Model download network error
    - Concurrent model loading contention

    This is a transient error as these conditions often resolve on retry.
    """
    pass


class AudioProcessingError(TransientError):
    """Temporary audio processing failure.

    Common causes:
    - FFmpeg temporary failures
    - Filesystem temporary issues
    - Resource contention

    Permanent audio format issues should raise InvalidAudioError instead.
    """
    pass


class DatabaseConnectionError(TransientError):
    """Database temporarily unavailable.

    Common causes:
    - Connection pool exhausted
    - Network hiccup
    - Database restart

    Should retry as connection is likely to recover.
    """
    pass


class NetworkError(TransientError):
    """Network-related error during model download or API calls.

    Common causes:
    - Model download timeout
    - Hugging Face API rate limiting
    - Network connectivity issues
    """
    pass


# Permanent errors (should not retry)
class PermanentError(AudioDiarizationException):
    """Errors that will not succeed on retry.

    These errors indicate fundamental problems that cannot be resolved
    by retrying the same operation (invalid input, missing resources,
    configuration errors).

    Tasks should fail immediately when these are raised.
    """
    pass


class InvalidAudioError(PermanentError):
    """Audio file is corrupt or invalid.

    Common causes:
    - File corrupted during upload
    - Unsupported codec
    - File is not actually audio
    - Zero-length or malformed file

    This is permanent - file will not magically become valid on retry.
    """
    pass


class InsufficientDiskSpaceError(PermanentError):
    """Not enough disk space for processing.

    This is treated as permanent because:
    - Unlikely to resolve quickly
    - Indicates system-level issue
    - Should not consume retry attempts

    Requires manual intervention to free up space.
    """
    pass


class InvalidConfigurationError(PermanentError):
    """Application configuration is invalid.

    Common causes:
    - Missing required environment variables
    - Invalid configuration values
    - Incompatible configuration combinations

    Requires code/config changes, not retryable.
    """
    pass


class ValidationError(PermanentError):
    """Input validation failed.

    Raised when user input fails validation rules.
    Not retryable as input will not change on retry.
    """
    pass


# Service-specific errors
class TranscriptionError(AudioDiarizationException):
    """WhisperX transcription failed.

    Base class for transcription-related errors.
    Subclasses indicate specific failure modes.

    Note: This is intentionally NOT a subclass of TransientError or
    PermanentError to allow case-by-case classification based on
    the underlying cause.
    """
    pass


class AlignmentError(TransientError):
    """WhisperX alignment failed (wav2vec2 word-level timestamps).

    Alignment adds word-level timestamps to transcription segments.

    Common causes:
    - Model download issues (transient)
    - Unsupported language (permanent, but gracefully degraded)
    - Resource exhaustion (transient)

    Marked as transient because most failures are recoverable.
    """
    pass


class DiarizationError(TransientError):
    """WhisperX diarization failed (speaker identification).

    Diarization identifies different speakers in audio.

    Common causes:
    - Hugging Face token issues (configuration)
    - Model download issues (transient)
    - Resource exhaustion (transient)

    Marked as transient because most failures are recoverable.
    """
    pass


# Repository errors
class JobCreationError(AudioDiarizationException):
    """Raised when job creation fails.

    Common causes:
    - Database constraint violation
    - Connection issues
    - Invalid job parameters
    """
    pass


class JobNotFoundError(AudioDiarizationException):
    """Raised when job is not found.

    This is typically not an error condition but rather an expected
    case when querying for non-existent jobs.
    """
    pass


# Utility functions
def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.

    Args:
        error: Exception to check

    Returns:
        True if error is transient and retryable

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     if is_transient_error(e):
        ...         retry_operation()
    """
    return isinstance(error, TransientError)


def is_permanent_error(error: Exception) -> bool:
    """Check if an error is permanent and should not be retried.

    Args:
        error: Exception to check

    Returns:
        True if error is permanent and should fail immediately
    """
    return isinstance(error, PermanentError)

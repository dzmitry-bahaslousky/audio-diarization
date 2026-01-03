"""Custom application exceptions for audio diarization service."""


class AudioDiarizationException(Exception):
    """Base exception for all application errors."""
    pass


# Transient errors (should retry)
class TransientError(AudioDiarizationException):
    """Errors that may succeed on retry."""
    pass


class ModelLoadError(TransientError):
    """Failed to load ML model (may be temporary resource issue)."""
    pass


class AudioProcessingError(TransientError):
    """Temporary audio processing failure."""
    pass


class DatabaseConnectionError(TransientError):
    """Database temporarily unavailable."""
    pass


# Permanent errors (should not retry)
class PermanentError(AudioDiarizationException):
    """Errors that will not succeed on retry."""
    pass


class InvalidAudioError(PermanentError):
    """Audio file is corrupt or invalid."""
    pass


class InsufficientDiskSpaceError(PermanentError):
    """Not enough disk space for processing."""
    pass


class InvalidConfigurationError(PermanentError):
    """Application configuration is invalid."""
    pass


# Service-specific errors
class TranscriptionError(AudioDiarizationException):
    """WhisperX transcription failed."""
    pass


class AlignmentError(TransientError):
    """WhisperX alignment failed (wav2vec2 word-level timestamps). Retryable."""
    pass


class DiarizationError(TransientError):
    """WhisperX diarization failed (speaker identification). Retryable."""
    pass


# Repository errors
class JobCreationError(AudioDiarizationException):
    """Raised when job creation fails."""
    pass


class JobNotFoundError(AudioDiarizationException):
    """Raised when job is not found."""
    pass

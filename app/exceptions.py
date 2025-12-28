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
    """Whisper transcription failed."""
    pass


class DiarizationError(AudioDiarizationException):
    """Pyannote diarization failed."""
    pass


class AlignmentError(AudioDiarizationException):
    """Alignment between transcription and diarization failed."""
    pass


# Repository errors
class JobCreationError(AudioDiarizationException):
    """Raised when job creation fails."""
    pass


class JobNotFoundError(AudioDiarizationException):
    """Raised when job is not found."""
    pass

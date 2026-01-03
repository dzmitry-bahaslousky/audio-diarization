"""Type definitions and protocols for the audio transcription application.

This module provides:
- Protocol definitions for duck-typed interfaces
- Type aliases for complex types
- Generic types for reusable patterns
- NewType definitions for semantic typing
"""

from pathlib import Path
from typing import Protocol, TypeVar, Dict, List, Any, Optional, TypeAlias
from sqlalchemy.orm import Session

# Type variables for generics
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)

# Type aliases for common complex types
AudioSegment: TypeAlias = Dict[str, Any]
"""Represents a transcription segment with timing and text."""

TranscriptionResult: TypeAlias = Dict[str, Any]
"""Complete transcription result from ML pipeline."""

JobMetadata: TypeAlias = Dict[str, Any]
"""Metadata associated with a transcription job."""

SpeakerGroups: TypeAlias = Dict[str, List[AudioSegment]]
"""Segments grouped by speaker identifier."""


# Protocols for duck-typed interfaces

class SupportsTranscription(Protocol):
    """Protocol for services that can transcribe audio files.

    This protocol defines the interface for transcription services,
    allowing for different implementations (WhisperX, OpenAI Whisper, etc.)
    without tight coupling.
    """

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            model_size: Optional model size override
            num_speakers: Optional hint for speaker count

        Returns:
            Transcription result with segments and metadata
        """
        ...

    def format_segments(self, result: TranscriptionResult) -> List[AudioSegment]:
        """Format transcription result into standardized segments.

        Args:
            result: Raw transcription output

        Returns:
            List of formatted audio segments
        """
        ...


class SupportsAudioProcessing(Protocol):
    """Protocol for audio processing services.

    Defines interface for audio format conversion and validation.
    """

    async def convert_to_wav(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """Convert audio file to WAV format.

        Args:
            input_path: Source audio file
            output_path: Destination WAV file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels

        Returns:
            Path to converted file
        """
        ...

    async def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get audio file metadata.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with duration, sample_rate, channels, etc.
        """
        ...


class SupportsRepository(Protocol[T_co]):
    """Protocol for repository pattern implementations.

    Generic protocol that can be specialized for different entity types.

    Type parameter:
        T_co: Covariant type of entities managed by repository
    """

    def create(self, **kwargs: Any) -> T_co:
        """Create a new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity
        """
        ...

    def get(self, id: str) -> Optional[T_co]:
        """Retrieve entity by ID.

        Args:
            id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        ...

    def update(self, id: str, **kwargs: Any) -> bool:
        """Update entity.

        Args:
            id: Entity identifier
            **kwargs: Attributes to update

        Returns:
            True if updated, False if not found
        """
        ...

    def delete(self, id: str) -> bool:
        """Delete entity.

        Args:
            id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        ...


class SupportsValidation(Protocol):
    """Protocol for validator services.

    Defines interface for input validation logic.
    """

    def validate_file(
        self,
        file: Any,
        filename: str,
        file_size: int
    ) -> str:
        """Validate uploaded file.

        Args:
            file: Upload file object
            filename: Original filename
            file_size: Size in bytes

        Returns:
            Validated file extension

        Raises:
            ValueError: If validation fails
        """
        ...


class SupportsExport(Protocol):
    """Protocol for export formatting services.

    Defines interface for formatting transcription results
    into various output formats.
    """

    def format_txt_simple(
        self,
        result: TranscriptionResult,
        metadata: JobMetadata
    ) -> str:
        """Format as simple text.

        Args:
            result: Transcription result
            metadata: Job metadata

        Returns:
            Plain text transcription
        """
        ...

    def format_json(
        self,
        result: TranscriptionResult,
        metadata: JobMetadata,
        job_id: str
    ) -> str:
        """Format as JSON.

        Args:
            result: Transcription result
            metadata: Job metadata
            job_id: Job identifier

        Returns:
            JSON string
        """
        ...


class SupportsCircuitBreaker(Protocol):
    """Protocol for circuit breaker pattern implementations.

    Provides fault tolerance by failing fast when a service
    is experiencing repeated failures.
    """

    async def call_async(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            State dictionary with metrics
        """
        ...


class HasDatabaseSession(Protocol):
    """Protocol for objects that provide database sessions.

    This enables dependency injection of session providers
    without coupling to specific implementations.
    """

    @property
    def db(self) -> Session:
        """Get database session.

        Returns:
            SQLAlchemy session
        """
        ...


# Utility type guards

def is_valid_audio_segment(obj: Any) -> bool:
    """Type guard to check if object is a valid audio segment.

    Args:
        obj: Object to check

    Returns:
        True if object has required segment fields

    Example:
        >>> segment = {"start": 0.0, "end": 1.5, "text": "Hello"}
        >>> if is_valid_audio_segment(segment):
        ...     process_segment(segment)
    """
    if not isinstance(obj, dict):
        return False

    required_fields = {'start', 'end', 'text'}
    return all(
        field in obj and obj[field] is not None
        for field in required_fields
    )


def is_valid_transcription_result(obj: Any) -> bool:
    """Type guard to check if object is a valid transcription result.

    Args:
        obj: Object to check

    Returns:
        True if object has required result fields
    """
    if not isinstance(obj, dict):
        return False

    required_fields = {'segments', 'language'}
    return all(field in obj for field in required_fields)

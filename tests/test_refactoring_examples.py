"""Example tests demonstrating testing patterns for refactored code.

This module provides examples of:
- Testing with Protocol-based mocks
- Exception context testing
- Type guard testing
- Decorator testing
- Repository pattern testing
"""

import pytest
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from unittest.mock import MagicMock

# Import types and protocols
from app.types import (
    is_valid_audio_segment,
    is_valid_transcription_result,
    AudioSegment,
    TranscriptionResult
)

# Import exceptions
from app.exceptions import (
    AudioDiarizationException,
    TransientError,
    ModelLoadError,
    InvalidAudioError,
    is_transient_error,
    is_permanent_error
)

# Import decorators
from app.decorators import (
    retry,
    timeit,
    handle_errors,
    validate_args
)


# ============================================================================
# Exception Testing
# ============================================================================

class TestExceptionHierarchy:
    """Test enhanced exception hierarchy."""

    def test_base_exception_with_context(self):
        """Test that base exception stores context properly."""
        details = {"job_id": "123", "file": "test.mp3"}
        original = ValueError("Original error")

        exc = AudioDiarizationException(
            "Processing failed",
            details=details,
            original_error=original
        )

        assert exc.message == "Processing failed"
        assert exc.details == details
        assert exc.original_error is original
        assert "job_id=123" in str(exc)
        assert "Caused by: Original error" in str(exc)

    def test_transient_error_detection(self):
        """Test transient error type guards."""
        transient = ModelLoadError("Model load failed")
        permanent = InvalidAudioError("Corrupt file")

        assert is_transient_error(transient)
        assert not is_transient_error(permanent)
        assert not is_permanent_error(transient)
        assert is_permanent_error(permanent)

    def test_exception_chaining(self):
        """Test that exception chaining works correctly."""
        original = ConnectionError("Network failed")

        try:
            raise TransientError(
                "Transient failure",
                details={"attempt": 1},
                original_error=original
            )
        except TransientError as e:
            assert e.original_error is original
            assert isinstance(e, AudioDiarizationException)


# ============================================================================
# Type Guard Testing
# ============================================================================

class TestTypeGuards:
    """Test type guard functions."""

    def test_valid_audio_segment(self):
        """Test audio segment validation."""
        valid_segment = {
            "start": 0.0,
            "end": 1.5,
            "text": "Hello world",
            "speaker": "SPEAKER_00"
        }

        assert is_valid_audio_segment(valid_segment)

    def test_invalid_audio_segment_missing_fields(self):
        """Test that segments missing required fields are rejected."""
        invalid = {"start": 0.0, "text": "Missing end"}
        assert not is_valid_audio_segment(invalid)

    def test_invalid_audio_segment_null_values(self):
        """Test that segments with null required fields are rejected."""
        invalid = {"start": None, "end": 1.0, "text": "Null start"}
        assert not is_valid_audio_segment(invalid)

    def test_valid_transcription_result(self):
        """Test transcription result validation."""
        valid_result = {
            "segments": [],
            "language": "en",
            "text": "Full text"
        }

        assert is_valid_transcription_result(valid_result)

    def test_invalid_transcription_result(self):
        """Test that invalid results are rejected."""
        invalid = {"segments": []}  # Missing language
        assert not is_valid_transcription_result(invalid)


# ============================================================================
# Protocol-Based Mock Testing
# ============================================================================

class MockTranscriptionService:
    """Mock transcription service implementing SupportsTranscription protocol."""

    async def transcribe(
        self,
        _audio_path: Path,
        language: Optional[str] = None,
        _model_size: Optional[str] = None,
        _num_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """Mock transcription."""
        return {
            "text": "Mock transcription",
            "language": language or "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Mock", "speaker": "SPEAKER_00"}
            ]
        }

    def format_segments(self, result: TranscriptionResult) -> List[AudioSegment]:
        """Mock segment formatting."""
        return result.get("segments", [])


class MockAudioProcessor:
    """Mock audio processor implementing SupportsAudioProcessing protocol."""

    async def convert_to_wav(
        self,
        _input_path: Path,
        output_path: Path,
        _sample_rate: int = 16000,
        _channels: int = 1
    ) -> Path:
        """Mock conversion."""
        return output_path

    async def get_audio_info(self, _file_path: Path) -> Dict[str, Any]:
        """Mock audio info."""
        return {
            "duration": 60.0,
            "sample_rate": 16000,
            "channels": 1,
            "codec": "pcm_s16le"
        }


class TestProtocolMocks:
    """Test using protocol-based mocks."""

    @pytest.mark.asyncio
    async def test_mock_transcription_service(self):
        """Test that mock implements protocol correctly."""
        mock_service = MockTranscriptionService()

        # Protocol compliance check (type checker validates this)
        result = await mock_service.transcribe(Path("test.wav"))

        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    @pytest.mark.asyncio
    async def test_mock_audio_processor(self):
        """Test that audio processor mock works."""
        mock_processor = MockAudioProcessor()

        # Test conversion
        output = await mock_processor.convert_to_wav(
            Path("input.mp3"),
            Path("output.wav")
        )
        assert output == Path("output.wav")

        # Test info extraction
        info = await mock_processor.get_audio_info(Path("test.wav"))
        assert info["duration"] == 60.0


# ============================================================================
# Decorator Testing
# ============================================================================

class TestDecorators:
    """Test decorator functionality."""

    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test that retry decorator succeeds on first attempt."""
        call_count = 0

        @retry(max_attempts=3)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_function()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_with_failures(self):
        """Test that retry decorator retries on transient errors."""
        call_count = 0

        @retry(max_attempts=3, backoff_factor=0.01)  # Fast backoff for testing
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        result = await failing_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_decorator_exhausts_attempts(self):
        """Test that retry decorator gives up after max attempts."""
        call_count = 0

        @retry(max_attempts=3, backoff_factor=0.01)
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise TransientError("Persistent failure")

        with pytest.raises(TransientError):
            await always_failing()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeit_decorator(self):
        """Test that timeit decorator logs execution time."""
        @timeit()
        async def timed_function():
            await asyncio.sleep(0.01)
            return "done"

        # Should complete without error and log timing
        result = await timed_function()
        assert result == "done"

    def test_validate_args_decorator_valid(self):
        """Test argument validation with valid inputs."""
        @validate_args(
            age=lambda x: isinstance(x, int) and x >= 0,
            name=lambda x: isinstance(x, str) and len(x) > 0
        )
        def create_user(name: str, age: int):
            return {"name": name, "age": age}

        result = create_user("Alice", 25)
        assert result == {"name": "Alice", "age": 25}

    def test_validate_args_decorator_invalid(self):
        """Test argument validation with invalid inputs."""
        @validate_args(
            age=lambda x: isinstance(x, int) and x >= 0
        )
        def create_user(name: str, age: int):
            return {"name": name, "age": age}

        with pytest.raises(ValueError, match="Validation failed"):
            create_user("Bob", -1)

    @pytest.mark.asyncio
    async def test_handle_errors_decorator(self):
        """Test error transformation decorator."""
        @handle_errors({ValueError: InvalidAudioError})
        async def risky_operation():
            raise ValueError("Something went wrong")

        with pytest.raises(InvalidAudioError) as exc_info:
            await risky_operation()

        assert "Something went wrong" in str(exc_info.value)


# ============================================================================
# Repository Pattern Testing
# ============================================================================

class TestRepositoryPattern:
    """Test repository pattern with mocks."""

    def test_repository_with_mock_session(self):
        """Test repository with mocked database session."""
        from app.repositories.transcription_repository import TranscriptionRepository

        # Create mock session
        mock_session = MagicMock()

        # Repository should accept any session-like object
        repo = TranscriptionRepository(mock_session)

        assert repo.db is mock_session

    def test_repository_transaction_commit(self):
        """Test that transaction commits on success."""
        from app.repositories.transcription_repository import TranscriptionRepository

        mock_session = MagicMock()
        repo = TranscriptionRepository(mock_session)

        with repo.transaction():
            pass  # Successful transaction

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()

    def test_repository_transaction_rollback(self):
        """Test that transaction rolls back on error."""
        from app.repositories.transcription_repository import TranscriptionRepository

        mock_session = MagicMock()
        repo = TranscriptionRepository(mock_session)

        with pytest.raises(ValueError):
            with repo.transaction():
                raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()


# ============================================================================
# Integration Testing Example
# ============================================================================

class TestIntegrationPatterns:
    """Example integration tests with refactored code."""

    @pytest.mark.asyncio
    async def test_full_processing_pipeline_with_mocks(self):
        """Test complete processing pipeline with protocol mocks."""
        # Setup mocks
        audio_processor = MockAudioProcessor()
        transcription_service = MockTranscriptionService()

        # Simulate processing pipeline
        audio_path = Path("test.mp3")
        wav_path = Path("test.wav")

        # Convert audio
        converted = await audio_processor.convert_to_wav(audio_path, wav_path)
        assert converted == wav_path

        # Get audio info
        info = await audio_processor.get_audio_info(converted)
        assert info["duration"] > 0

        # Transcribe
        result = await transcription_service.transcribe(converted)
        assert is_valid_transcription_result(result)

        # Format segments
        segments = transcription_service.format_segments(result)
        assert len(segments) > 0
        assert all(is_valid_audio_segment(seg) for seg in segments)


# ============================================================================
# Property-Based Testing Example
# ============================================================================

class TestPropertyBased:
    """Example property-based tests using hypothesis."""

    def test_exception_string_representation_is_consistent(self):
        """Property: Exception string should always contain message."""
        from hypothesis import given
        from hypothesis.strategies import text, dictionaries

        @given(
            message=text(min_size=1),
            details=dictionaries(text(), text())
        )
        def property_test(message: str, details: dict):
            exc = AudioDiarizationException(message, details=details)
            exc_str = str(exc)
            assert message in exc_str

        property_test()

    def test_audio_segment_validation_properties(self):
        """Property: Valid segments must have start < end."""
        valid_segment = {
            "start": 0.0,
            "end": 1.0,
            "text": "test"
        }

        # Property: start must be less than end
        assert valid_segment["start"] < valid_segment["end"]

        # Property: valid segment passes validation
        assert is_valid_audio_segment(valid_segment)


# ============================================================================
# Fixture Examples
# ============================================================================

@pytest.fixture
def mock_transcription_service():
    """Fixture providing mock transcription service."""
    return MockTranscriptionService()


@pytest.fixture
def mock_audio_processor():
    """Fixture providing mock audio processor."""
    return MockAudioProcessor()


@pytest.fixture
def sample_audio_segment() -> AudioSegment:
    """Fixture providing a valid audio segment."""
    return {
        "start": 0.0,
        "end": 1.5,
        "text": "Sample text",
        "speaker": "SPEAKER_00",
        "confidence": 0.95
    }


@pytest.fixture
def sample_transcription_result() -> TranscriptionResult:
    """Fixture providing a valid transcription result."""
    return {
        "text": "Full transcription text",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "First", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "Second", "speaker": "SPEAKER_01"}
        ]
    }


class TestWithFixtures:
    """Example tests using fixtures."""

    def test_with_mock_service_fixture(self, mock_transcription_service):
        """Test using injected mock service."""
        assert mock_transcription_service is not None

    def test_with_sample_segment_fixture(self, sample_audio_segment):
        """Test using sample audio segment."""
        assert is_valid_audio_segment(sample_audio_segment)
        assert sample_audio_segment["start"] < sample_audio_segment["end"]

    def test_with_sample_result_fixture(self, sample_transcription_result):
        """Test using sample transcription result."""
        assert is_valid_transcription_result(sample_transcription_result)
        assert len(sample_transcription_result["segments"]) == 2

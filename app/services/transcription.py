"""Whisper transcription service with comprehensive error handling."""

import asyncio
import logging
import whisper
from pathlib import Path
from typing import Dict, Optional

from app.config import settings
from app.exceptions import (
    ModelLoadError,
    TranscriptionError,
    InvalidAudioError,
    TransientError
)
from app.utils.circuit_breaker import (
    get_whisper_circuit_breaker,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Handles audio transcription using Whisper."""

    def __init__(self):
        """Initialize transcription service."""
        self.model = None
        self.current_model_size = None
        self._load_attempts = 0
        self.circuit_breaker = get_whisper_circuit_breaker()
        logger.info("TranscriptionService initialized")

    async def load_model(self, model_size: str = None) -> None:
        """
        Load Whisper model with error handling and circuit breaker protection.

        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)

        Raises:
            ModelLoadError: If model loading fails
            CircuitBreakerOpenError: If too many recent failures
        """
        model_size = model_size or settings.whisper_model

        if self.model and self.current_model_size == model_size:
            logger.info(f"Model {model_size} already loaded")
            return

        logger.info(f"Loading Whisper model: {model_size} (attempt {self._load_attempts + 1})")

        try:
            # Use circuit breaker to prevent repeated failures
            self.model = await self.circuit_breaker.call_async(
                self._load_model_internal,
                model_size
            )
            self.current_model_size = model_size
            self._load_attempts = 0
            logger.info(f"Whisper model {model_size} loaded successfully")

        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open, cannot load model: {e}")
            raise ModelLoadError(f"Model loading circuit breaker open: {e}")

        except Exception as e:
            self._load_attempts += 1
            logger.error(f"Failed to load Whisper model {model_size}: {e}", exc_info=True)

            # Check if it's a resource error (transient)
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                raise ModelLoadError(f"Resource error loading model (may be transient): {e}")

            # Check if it's a download/network error (transient)
            if "download" in str(e).lower() or "connection" in str(e).lower():
                raise ModelLoadError(f"Network error loading model (will retry): {e}")

            # Unknown error - treat as transient for first few attempts
            if self._load_attempts < 3:
                raise ModelLoadError(f"Error loading model (attempt {self._load_attempts}): {e}")

            # Give up after multiple attempts
            raise ModelLoadError(f"Failed to load model after {self._load_attempts} attempts: {e}")

    async def _load_model_internal(self, model_size: str):
        """
        Internal method to load model (wrapped by circuit breaker).

        Args:
            model_size: Model size to load

        Returns:
            Loaded Whisper model
        """
        return await asyncio.to_thread(
            whisper.load_model,
            model_size,
            device=settings.whisper_device
        )

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        model_size: str = None
    ) -> Dict:
        """
        Transcribe audio file using Whisper with comprehensive error handling.

        Args:
            audio_path: Path to audio file (preferably WAV)
            language: Optional language code (e.g., 'en', 'es')
            model_size: Optional model size override

        Returns:
            Transcription result with segments and timestamps

        Raises:
            TranscriptionError: If transcription fails
            InvalidAudioError: If audio file is invalid
            ModelLoadError: If model cannot be loaded
        """
        # Validate audio file exists
        if not audio_path.exists():
            raise InvalidAudioError(f"Audio file not found: {audio_path}")

        # Check file size is reasonable
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # 500MB limit for converted WAV
            raise InvalidAudioError(f"Audio file too large: {file_size_mb:.2f}MB")

        if file_size_mb == 0:
            raise InvalidAudioError(f"Audio file is empty: {audio_path}")

        try:
            # Ensure model is loaded
            await self.load_model(model_size)
        except ModelLoadError as e:
            # Re-raise with context
            raise TranscriptionError(f"Cannot transcribe, model load failed: {e}")

        logger.info(f"Transcribing {audio_path} (size: {file_size_mb:.2f}MB)")

        try:
            # Transcribe with timeout protection (30 minute max)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.transcribe,
                    str(audio_path),
                    language=language,
                    word_timestamps=True,  # Enable word-level timestamps
                    verbose=False
                ),
                timeout=3600  # 60 minutes
            )

            # Validate result
            if not result or 'segments' not in result:
                raise TranscriptionError("Whisper returned invalid result (missing segments)")

            logger.info(
                f"Transcription complete: {len(result['segments'])} segments, "
                f"detected language: {result.get('language', 'unknown')}"
            )

            return result

        except asyncio.TimeoutError:
            raise TranscriptionError(f"Transcription timed out after 30 minutes")

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}", exc_info=True)

            # Check for common error patterns
            if "corrupt" in str(e).lower() or "invalid" in str(e).lower():
                raise InvalidAudioError(f"Audio file appears corrupt: {e}")

            if "memory" in str(e).lower():
                raise TransientError(f"Out of memory during transcription: {e}")

            raise TranscriptionError(f"Transcription failed: {e}")

    def format_segments(self, transcription_result: Dict) -> list[Dict]:
        """
        Format Whisper output into standardized segments with input validation.

        Args:
            transcription_result: Raw Whisper output

        Returns:
            List of formatted segments

        Raises:
            ValueError: If input is invalid
        """
        if not transcription_result or 'segments' not in transcription_result:
            raise ValueError("Invalid transcription result format (missing 'segments' key)")

        segments = []

        for i, seg in enumerate(transcription_result['segments']):
            # Validate required fields
            if 'start' not in seg or 'end' not in seg or 'text' not in seg:
                logger.warning(f"Skipping invalid segment {i}: missing required fields")
                continue

            # Validate timing
            if seg['start'] < 0 or seg['end'] <= seg['start']:
                logger.warning(f"Skipping segment {i} with invalid timing: {seg['start']}-{seg['end']}")
                continue

            segments.append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
                'confidence': seg.get('no_speech_prob', 0.0),
                'words': seg.get('words', [])
            })

        if not segments:
            logger.warning("No valid segments found in transcription result")

        return segments

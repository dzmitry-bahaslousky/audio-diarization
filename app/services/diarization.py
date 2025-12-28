"""Pyannote speaker diarization service with comprehensive error handling."""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# Suppress torchcodec warning (we use ffmpeg-python for audio decoding)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

import torch
import torchaudio
from pyannote.audio import Pipeline

from app.config import settings
from app.exceptions import (
    ModelLoadError,
    DiarizationError,
    InvalidAudioError,
    TransientError
)
from app.utils.circuit_breaker import (
    get_pyannote_circuit_breaker,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)


class DiarizationService:
    """Handles speaker diarization using Pyannote."""

    def __init__(self):
        """Initialize diarization service."""
        self.pipeline = None
        self._load_attempts = 0
        self.circuit_breaker = get_pyannote_circuit_breaker()
        logger.info("DiarizationService initialized")

    async def load_pipeline(self) -> None:
        """
        Load Pyannote diarization pipeline with error handling and circuit breaker protection.

        Requires HF_TOKEN to be set in environment.

        Raises:
            ModelLoadError: If pipeline loading fails
            CircuitBreakerOpenError: If too many recent failures
        """
        if self.pipeline:
            logger.info("Diarization pipeline already loaded")
            return

        logger.info(f"Loading Pyannote diarization pipeline (attempt {self._load_attempts + 1})...")

        try:
            # Use circuit breaker to prevent repeated failures
            self.pipeline = await self.circuit_breaker.call_async(
                self._load_pipeline_internal
            )

            # Set device (cpu, cuda, or mps)
            if settings.diarization_device != "cpu":
                self.pipeline.to(settings.diarization_device)

            self._load_attempts = 0
            logger.info(f"Diarization pipeline loaded on {settings.diarization_device}")

        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open for diarization pipeline: {e}")
            raise ModelLoadError(f"Diarization pipeline circuit breaker open: {e}")

        except Exception as e:
            self._load_attempts += 1
            logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)

            # Check for authentication errors (permanent)
            if "401" in str(e) or "authentication" in str(e).lower() or "token" in str(e).lower():
                raise ModelLoadError(
                    f"Invalid HuggingFace token. Please check HF_TOKEN environment variable. "
                    f"Ensure you've accepted the model license at: "
                    f"https://huggingface.co/{settings.diarization_model}"
                )

            # Network errors (transient)
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelLoadError(f"Network error loading pipeline (will retry): {e}")

            # Resource errors (transient)
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                raise ModelLoadError(f"Resource error loading pipeline: {e}")

            # Generic error
            raise ModelLoadError(f"Failed to load diarization pipeline: {e}")

    async def _load_pipeline_internal(self):
        """
        Internal method to load pipeline (wrapped by circuit breaker).

        Returns:
            Loaded Pyannote pipeline
        """
        return await asyncio.to_thread(
            Pipeline.from_pretrained,
            settings.diarization_model,
            token=settings.hf_token
        )

    async def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        Perform speaker diarization on audio file with comprehensive error handling.

        Args:
            audio_path: Path to audio file (WAV format recommended)
            num_speakers: Optional hint for number of speakers

        Returns:
            Diarization result with speaker segments

        Raises:
            DiarizationError: If diarization fails
            InvalidAudioError: If audio is invalid
            ModelLoadError: If pipeline cannot be loaded
        """
        # Validate audio file
        if not audio_path.exists():
            raise InvalidAudioError(f"Audio file not found: {audio_path}")

        try:
            # Ensure pipeline is loaded
            await self.load_pipeline()
        except ModelLoadError as e:
            raise DiarizationError(f"Cannot diarize, pipeline load failed: {e}")

        logger.info(f"Performing diarization on {audio_path}")
        if num_speakers:
            logger.info(f"Using num_speakers hint: {num_speakers}")

        try:
            # Load audio into memory (required since torchcodec is not available)
            waveform, sample_rate = await asyncio.to_thread(
                torchaudio.load,
                str(audio_path)
            )

            # Validate audio loaded correctly
            if waveform is None or waveform.numel() == 0:
                raise InvalidAudioError("Audio file is empty or corrupt")

            logger.debug(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")

            # Prepare audio dictionary for Pyannote
            audio_dict = {
                'waveform': waveform,
                'sample_rate': sample_rate
            }

            # Run diarization in background thread (CPU/GPU intensive) with timeout
            diarization_params = {}
            if num_speakers:
                diarization_params['num_speakers'] = num_speakers

            diarization = await asyncio.wait_for(
                asyncio.to_thread(
                    self.pipeline,
                    audio_dict,
                    **diarization_params
                ),
                timeout=1800  # 30 minute timeout
            )

            # Extract speaker segments
            segments = self.extract_segments(diarization)

            if not segments:
                raise DiarizationError("No speaker segments detected")

            logger.info(
                f"Diarization complete: {len(segments)} speaker segments, "
                f"{len(set(s['speaker'] for s in segments))} unique speakers"
            )

            return {
                'segments': segments,
                'num_speakers': len(set(s['speaker'] for s in segments))
            }

        except asyncio.TimeoutError:
            raise DiarizationError("Diarization timed out after 30 minutes")

        except InvalidAudioError:
            raise  # Re-raise as-is

        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)

            # Check for common error patterns
            if "memory" in str(e).lower():
                raise TransientError(f"Out of memory during diarization: {e}")

            if "corrupt" in str(e).lower() or "invalid" in str(e).lower():
                raise InvalidAudioError(f"Audio file appears corrupt: {e}")

            raise DiarizationError(f"Diarization failed: {e}")

    def extract_segments(self, diarization) -> List[Dict]:
        """
        Extract speaker segments from Pyannote diarization output with validation.

        Args:
            diarization: Pyannote DiarizeOutput object (v4.x)

        Returns:
            List of speaker segments with timestamps

        Raises:
            DiarizationError: If extraction fails
        """
        try:
            segments = []

            # In pyannote.audio 4.x, DiarizeOutput is a dataclass with speaker_diarization field
            # speaker_diarization is an Annotation object with itertracks() method
            annotation = diarization.speaker_diarization

            # Each iteration yields (Segment, track_id, speaker_label)
            for segment, track, speaker in annotation.itertracks(yield_label=True):
                # Validate segment
                if segment.start < 0 or segment.end <= segment.start:
                    logger.warning(f"Skipping invalid segment timing: {segment}")
                    continue

                segments.append({
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'speaker': str(speaker),
                    'duration': float(segment.duration)
                })

            return segments

        except Exception as e:
            logger.error(f"Failed to extract segments: {e}")
            raise DiarizationError(f"Failed to extract diarization segments: {e}")

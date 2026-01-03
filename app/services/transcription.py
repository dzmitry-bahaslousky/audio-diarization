"""WhisperX transcription service with alignment and diarization."""

import asyncio
import gc
import logging
import torch
import whisperx
from pathlib import Path
from typing import Dict, Optional, Tuple

# Fix PyTorch 2.6+ weights_only=True issue for OmegaConf in worker processes
# This ensures the fix is applied even if the worker process forked before the main registration
try:
    import omegaconf
    import typing
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import Container, ContainerMetadata, Node

    safe_globals = [ListConfig, DictConfig, Container, ContainerMetadata, Node, typing.Any, list]
    for attr_name in dir(omegaconf):
        attr = getattr(omegaconf, attr_name)
        if isinstance(attr, type) and attr not in safe_globals:
            safe_globals.append(attr)

    torch.serialization.add_safe_globals(safe_globals)
    print(f"[TranscriptionService] Registered {len(safe_globals)} OmegaConf types, typing.Any, and list as safe globals")
except Exception:
    pass  # Already registered or not needed

from app.config import settings
from app.exceptions import (
    ModelLoadError,
    TranscriptionError,
    InvalidAudioError,
    TransientError,
    AlignmentError,
    DiarizationError
)
from app.utils.circuit_breaker import (
    get_whisper_circuit_breaker,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Handles audio transcription using WhisperX with alignment and diarization."""

    def __init__(self):
        """Initialize WhisperX transcription service."""
        # Transcription model
        self.model = None
        self.current_model_size = None
        self.current_compute_type = None

        # Alignment models (cached per language)
        self.alignment_models = {}  # Dict[language_code, (model, metadata)]

        # Error tracking
        self._load_attempts = 0
        self.circuit_breaker = get_whisper_circuit_breaker()

        logger.info("WhisperX TranscriptionService initialized")

    async def load_model(
        self,
        model_size: str = None,
        compute_type: str = None
    ) -> None:
        """
        Load WhisperX model with error handling and circuit breaker protection.

        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3, turbo)
            compute_type: Compute precision (float16, int8, float32)

        Raises:
            ModelLoadError: If model loading fails
            CircuitBreakerOpenError: If too many recent failures
        """
        model_size = model_size or settings.whisper_model
        compute_type = compute_type or settings.whisperx_compute_type

        # Check for MPS device (not supported by faster-whisper/ctranslate2)
        device = settings.whisper_device
        if device == "mps":
            logger.warning(
                "MPS device not supported by faster-whisper/ctranslate2. "
                "Falling back to CPU. Please update WHISPER_DEVICE=cpu in .env"
            )
            # Temporarily override for this session
            import app.config
            app.config.settings.whisper_device = "cpu"
            device = "cpu"

        # Auto-adjust compute type based on device
        if device == "cpu" and compute_type == "float16":
            logger.warning("float16 not supported on CPU, switching to int8")
            compute_type = "int8"

        # Check if model already loaded with same parameters
        if (self.model and
            self.current_model_size == model_size and
            self.current_compute_type == compute_type):
            logger.info(f"Model {model_size} ({compute_type}) already loaded")
            return

        logger.info(
            f"Loading WhisperX model: {model_size} "
            f"(compute_type={compute_type}, device={settings.whisper_device}, "
            f"attempt {self._load_attempts + 1})"
        )

        try:
            # Use circuit breaker to prevent repeated failures
            self.model = await self.circuit_breaker.call_async(
                self._load_model_internal,
                model_size,
                compute_type
            )
            self.current_model_size = model_size
            self.current_compute_type = compute_type
            self._load_attempts = 0
            logger.info(f"WhisperX model {model_size} loaded successfully")

        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open, cannot load model: {e}")
            raise ModelLoadError(f"Model loading circuit breaker open: {e}")

        except Exception as e:
            self._load_attempts += 1
            logger.error(f"Failed to load WhisperX model {model_size}: {e}", exc_info=True)

            # Check if it's a resource error (transient)
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "mps" in str(e).lower():
                raise ModelLoadError(f"Resource error loading model (may be transient): {e}")

            # Check if it's a download/network error (transient)
            if "download" in str(e).lower() or "connection" in str(e).lower():
                raise ModelLoadError(f"Network error loading model (will retry): {e}")

            # Unknown error - treat as transient for first few attempts
            if self._load_attempts < 3:
                raise ModelLoadError(f"Error loading model (attempt {self._load_attempts}): {e}")

            # Give up after multiple attempts
            raise ModelLoadError(f"Failed to load model after {self._load_attempts} attempts: {e}")

    async def _load_model_internal(self, model_size: str, compute_type: str):
        """
        Internal method to load WhisperX model (wrapped by circuit breaker).

        Args:
            model_size: Model size to load
            compute_type: Precision type for inference

        Returns:
            Loaded WhisperX model
        """
        # Free previous model memory if exists
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Clear MPS cache if available
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        return await asyncio.to_thread(
            whisperx.load_model,
            model_size,
            device=settings.whisper_device,
            compute_type=compute_type
        )

    async def load_alignment_model(self, language_code: str) -> Tuple[any, any]:
        """
        Load WhisperX alignment model for a specific language (with caching).

        Alignment models provide accurate word-level timestamps using wav2vec2.

        Args:
            language_code: Language code (e.g., 'en', 'es', 'de', 'fr')

        Returns:
            Tuple of (alignment_model, metadata)

        Raises:
            AlignmentError: If alignment model cannot be loaded
        """
        # Return cached model if available
        if language_code in self.alignment_models:
            logger.info(f"Using cached alignment model for language: {language_code}")
            return self.alignment_models[language_code]

        logger.info(f"Loading alignment model for language: {language_code}")

        try:
            model_a, metadata = await asyncio.to_thread(
                whisperx.load_align_model,
                language_code=language_code,
                device=settings.whisper_device
            )

            # Cache for future use
            self.alignment_models[language_code] = (model_a, metadata)
            logger.info(f"Alignment model loaded and cached for: {language_code}")

            return model_a, metadata

        except Exception as e:
            logger.error(f"Failed to load alignment model for {language_code}: {e}", exc_info=True)

            # Check for unsupported language
            if "not supported" in str(e).lower() or "not found" in str(e).lower():
                logger.warning(f"Alignment not supported for language: {language_code}")
                raise AlignmentError(
                    f"Alignment model not available for language '{language_code}'"
                )

            # Check for network errors
            if "download" in str(e).lower() or "connection" in str(e).lower():
                raise AlignmentError(
                    f"Network error downloading alignment model for '{language_code}': {e}"
                )

            raise AlignmentError(f"Failed to load alignment model: {e}")

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        model_size: str = None,
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        Transcribe audio file using WhisperX with alignment and diarization.

        This is a 3-step process:
        1. Transcribe using WhisperX (batched, fast)
        2. Align for word-level timestamps (optional, language-specific)
        3. Diarize for speaker identification (optional)

        Args:
            audio_path: Path to audio file (preferably WAV)
            language: Optional language code (e.g., 'en', 'es')
            model_size: Optional model size override
            num_speakers: Optional hint for expected number of speakers

        Returns:
            Transcription result with segments, timestamps, and speaker labels

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

        logger.info(
            f"Transcribing {audio_path} (size: {file_size_mb:.2f}MB, "
            f"batch_size={settings.whisperx_batch_size})"
        )

        audio = None
        try:
            # Step 1: Load audio using WhisperX
            audio = await asyncio.to_thread(
                whisperx.load_audio,
                str(audio_path)
            )

            # Step 2: Transcribe with batched inference (timeout: 60 minutes)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.transcribe,
                    audio,
                    batch_size=settings.whisperx_batch_size,
                    language=language
                ),
                timeout=3600  # 60 minutes
            )

            # Validate result
            if not result or 'segments' not in result:
                raise TranscriptionError("WhisperX returned invalid result (missing segments)")

            detected_language = result.get('language', language or 'unknown')

            logger.info(
                f"Initial transcription complete: {len(result['segments'])} segments, "
                f"detected language: {detected_language}"
            )

            # Step 3: Alignment for word-level timestamps (optional)
            if settings.whisperx_enable_alignment and result['segments']:
                try:
                    result = await self._align_segments(result, audio, detected_language)
                    logger.info("Alignment successful, word-level timestamps added")
                except AlignmentError as e:
                    logger.warning(
                        f"Alignment failed, proceeding with segment-level timestamps only: {e}"
                    )
                    # Continue with non-aligned result (graceful degradation)

            # Step 4: Diarization for speaker identification (optional)
            if settings.whisperx_enable_diarization and result['segments']:
                try:
                    result = await self._diarize_segments(result, audio, num_speakers)
                    logger.info("Diarization successful, speaker labels added")
                except DiarizationError as e:
                    logger.warning(
                        f"Diarization failed, proceeding without speaker labels: {e}"
                    )
                    # Continue without speaker labels (graceful degradation)

            # Step 5: Format to Whisper-compatible output
            formatted_result = self._format_whisperx_output(result, detected_language)

            return formatted_result

        except asyncio.TimeoutError:
            raise TranscriptionError(f"Transcription timed out after 60 minutes")

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}", exc_info=True)

            # Check for common error patterns
            if "corrupt" in str(e).lower() or "invalid" in str(e).lower():
                raise InvalidAudioError(f"Audio file appears corrupt: {e}")

            if "memory" in str(e).lower():
                raise TransientError(f"Out of memory during transcription: {e}")

            raise TranscriptionError(f"Transcription failed: {e}")

        finally:
            # Clean up audio array from memory
            if audio is not None:
                del audio
                gc.collect()

    async def _align_segments(
        self,
        transcription_result: Dict,
        audio: any,
        language_code: str
    ) -> Dict:
        """
        Align transcription segments with audio for word-level timestamps.

        Uses wav2vec2-based alignment models for precise word timing.

        Args:
            transcription_result: Output from WhisperX transcribe
            audio: Audio array from whisperx.load_audio
            language_code: Detected language code

        Returns:
            Transcription result with aligned word timestamps

        Raises:
            AlignmentError: If alignment fails
        """
        try:
            # Load alignment model for this language
            model_a, metadata = await self.load_alignment_model(language_code)

            # Perform alignment with timeout (10 minutes)
            aligned_result = await asyncio.wait_for(
                asyncio.to_thread(
                    whisperx.align,
                    transcription_result["segments"],
                    model_a,
                    metadata,
                    audio,
                    settings.whisper_device,
                    return_char_alignments=False  # Word-level only
                ),
                timeout=600  # 10 minutes
            )

            # Replace segments with aligned version
            transcription_result["segments"] = aligned_result["segments"]
            return transcription_result

        except AlignmentError:
            # Re-raise our custom exceptions
            raise

        except asyncio.TimeoutError:
            raise AlignmentError("Alignment timed out after 10 minutes")

        except Exception as e:
            logger.error(f"Alignment failed: {e}", exc_info=True)
            raise AlignmentError(f"Alignment failed: {e}")

    async def _diarize_segments(
        self,
        transcription_result: Dict,
        audio: any,
        num_speakers: Optional[int] = None
    ) -> Dict:
        """
        Add speaker diarization to transcription segments.

        Uses WhisperX's integrated pyannote diarization pipeline.

        Args:
            transcription_result: Output from WhisperX transcribe (optionally aligned)
            audio: Audio array from whisperx.load_audio
            num_speakers: Optional hint for expected number of speakers

        Returns:
            Transcription result with speaker labels assigned to words/segments

        Raises:
            DiarizationError: If diarization fails
        """
        try:
            # Import diarization pipeline from whisperx
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers

            # Initialize diarization pipeline
            logger.info("Initializing diarization pipeline")
            diarize_model = await asyncio.to_thread(
                DiarizationPipeline,
                use_auth_token=settings.hf_token,
                device=settings.whisper_device
            )

            # Determine min/max speakers
            min_speakers = settings.whisperx_min_speakers or (num_speakers if num_speakers else None)
            max_speakers = settings.whisperx_max_speakers or (num_speakers if num_speakers else None)

            logger.info(f"Running diarization (min_speakers={min_speakers}, max_speakers={max_speakers})")

            # Perform diarization with timeout (15 minutes)
            diarize_segments = await asyncio.wait_for(
                asyncio.to_thread(
                    diarize_model,
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                ),
                timeout=900  # 15 minutes
            )

            # Assign speakers to words/segments
            logger.info("Assigning speakers to words")
            result_with_speakers = await asyncio.to_thread(
                assign_word_speakers,
                diarize_segments,
                transcription_result
            )

            # Update transcription result with speaker assignments
            transcription_result["segments"] = result_with_speakers["segments"]

            return transcription_result

        except asyncio.TimeoutError:
            raise DiarizationError("Diarization timed out after 15 minutes")

        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)

            # Check for HF token errors
            if "token" in str(e).lower() or "auth" in str(e).lower():
                raise DiarizationError(f"Hugging Face authentication failed. Check HF_TOKEN: {e}")

            # Check for resource errors
            if "memory" in str(e).lower():
                raise DiarizationError(f"Out of memory during diarization: {e}")

            raise DiarizationError(f"Diarization failed: {e}")

    def _format_whisperx_output(self, whisperx_result: Dict, language: str) -> Dict:
        """
        Format WhisperX output to match OpenAI Whisper structure.

        Ensures backward compatibility with existing database schema and API.

        Args:
            whisperx_result: Raw WhisperX output
            language: Detected language code

        Returns:
            Whisper-compatible result dictionary
        """
        # Extract full text from segments
        full_text = " ".join([seg.get('text', '').strip() for seg in whisperx_result.get('segments', [])])

        return {
            'text': full_text,
            'language': language,
            'segments': whisperx_result.get('segments', [])
        }

    def format_segments(self, transcription_result: Dict) -> list[Dict]:
        """
        Format WhisperX output into standardized segments with input validation.

        Compatible with both aligned (word-level) and non-aligned (segment-level) output.
        Includes speaker labels from diarization if available.

        Args:
            transcription_result: Raw WhisperX output

        Returns:
            List of formatted segments with speaker labels

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

            formatted_segment = {
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
                'speaker': seg.get('speaker'),  # May be None if diarization disabled/failed
                'confidence': 1.0 - seg.get('no_speech_prob', 0.0),  # Convert to confidence
                'words': seg.get('words', [])  # Include word-level timestamps if aligned
            }

            segments.append(formatted_segment)

        if not segments:
            logger.warning("No valid segments found in transcription result")

        return segments

"""Audio preprocessing and conversion utilities."""

import asyncio
import ffmpeg
import logging
from pathlib import Path
from typing import Dict, Optional

from app.exceptions import AudioProcessingError, InvalidAudioError

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio format conversion and preprocessing."""

    def __init__(self):
        """Initialize audio processor."""
        logger.info("AudioProcessor initialized")

    async def convert_to_wav(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        Convert audio file to WAV format (16kHz, mono).

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            sample_rate: Target sample rate (default: 16000 Hz)
            channels: Number of audio channels (default: 1 = mono)

        Returns:
            Path to converted WAV file

        Raises:
            InvalidAudioError: If input file is not valid audio
            AudioProcessingError: If conversion fails
        """
        try:
            logger.info(f"Converting {input_path} to WAV format")

            # Validate input file exists
            if not input_path.exists():
                raise InvalidAudioError(f"Input file not found: {input_path}")

            # Use ffmpeg-python to convert
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='pcm_s16le',  # 16-bit PCM
                ac=channels,         # mono
                ar=sample_rate       # 16kHz
            )

            # Run conversion in background thread (blocking operation)
            await asyncio.to_thread(
                ffmpeg.run,
                stream,
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True
            )

            logger.info(f"Successfully converted to {output_path}")
            return output_path

        except ffmpeg.Error as e:
            # Parse ffmpeg error message for actionable errors
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")

            # Provide specific error messages based on ffmpeg output
            if "Invalid data found" in error_msg or "could not find codec" in error_msg:
                raise InvalidAudioError(
                    f"File is not a valid audio file or is corrupted: {input_path.name}"
                )
            elif "No such file" in error_msg or "does not exist" in error_msg:
                raise InvalidAudioError(f"Input file not found: {input_path}")
            elif "Permission denied" in error_msg:
                raise AudioProcessingError(
                    f"Permission denied accessing file: {input_path}"
                )
            elif "Disk quota exceeded" in error_msg or "No space left" in error_msg:
                raise AudioProcessingError("Insufficient disk space for conversion")
            else:
                # Generic conversion error
                raise AudioProcessingError(
                    f"Audio conversion failed: {error_msg[:200]}"
                )

        except OSError as e:
            # File system errors
            logger.error(f"File system error during conversion: {e}")
            raise AudioProcessingError(f"File system error: {e}")

        except InvalidAudioError:
            # Re-raise our custom exceptions
            raise

        except AudioProcessingError:
            # Re-raise our custom exceptions
            raise

    async def get_audio_info(self, file_path: Path) -> Dict[str, any]:
        """
        Get audio file metadata.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio metadata

        Raises:
            InvalidAudioError: If file is not valid audio or cannot be probed
            AudioProcessingError: If metadata extraction fails
        """
        try:
            # Validate file exists
            if not file_path.exists():
                raise InvalidAudioError(f"Audio file not found: {file_path}")

            # Run ffprobe in background thread
            probe = await asyncio.to_thread(
                ffmpeg.probe, str(file_path)
            )

            audio_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'audio'),
                None
            )

            if not audio_stream:
                raise InvalidAudioError(
                    f"No audio stream found in file: {file_path.name}"
                )

            duration = float(probe['format'].get('duration', 0))

            return {
                'duration': duration,
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(audio_stream.get('bit_rate', 0)),
            }

        except ffmpeg.Error as e:
            # FFmpeg probe errors
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFprobe error: {error_msg}")

            if "Invalid data found" in error_msg or "could not find codec" in error_msg:
                raise InvalidAudioError(
                    f"File is not a valid audio file: {file_path.name}"
                )
            elif "No such file" in error_msg:
                raise InvalidAudioError(f"Audio file not found: {file_path}")
            else:
                raise AudioProcessingError(
                    f"Failed to probe audio file: {error_msg[:200]}"
                )

        except (ValueError, KeyError) as e:
            # Parsing errors in probe data
            logger.error(f"Failed to parse audio metadata: {e}")
            raise InvalidAudioError(
                f"Invalid or corrupted audio file: {file_path.name}"
            )

        except OSError as e:
            # File system errors
            logger.error(f"File system error reading audio info: {e}")
            raise AudioProcessingError(f"Cannot access file: {e}")

        except InvalidAudioError:
            # Re-raise our custom exceptions
            raise

        except AudioProcessingError:
            # Re-raise our custom exceptions
            raise

    async def validate_audio(self, file_path: Path) -> bool:
        """
        Validate audio file can be read and has valid properties.

        Args:
            file_path: Path to audio file

        Returns:
            True if valid

        Raises:
            InvalidAudioError: If file is not valid audio or has invalid properties
            AudioProcessingError: If validation fails
        """
        info = await self.get_audio_info(file_path)

        # Validate duration
        if info['duration'] <= 0:
            raise InvalidAudioError(
                f"Audio file has zero or negative duration: {file_path.name}"
            )

        # Validate sample rate
        if info['sample_rate'] <= 0:
            raise InvalidAudioError(
                f"Invalid sample rate in audio file: {file_path.name}"
            )

        logger.info(
            f"Audio validated: {info['duration']:.2f}s, "
            f"{info['sample_rate']}Hz, {info['codec']}"
        )
        return True

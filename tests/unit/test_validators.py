"""Unit tests for UploadValidator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import shutil

from app.validators.upload_validator import UploadValidator
from app.models.schemas import WhisperModel
from app.exceptions import InsufficientDiskSpaceError


@pytest.fixture
def validator():
    """Create an UploadValidator instance."""
    return UploadValidator()


class TestValidateFile:
    """Test cases for validate_file method."""

    def test_validate_file_success(self, validator):
        """Test successful file validation."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.mp3"
        file_size = 10 * 1024 * 1024  # 10MB

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024  # 500MB

            # Execute
            result = validator.validate_file(mock_file, filename, file_size)

        # Verify
        assert result == "mp3"

    def test_validate_file_no_file_provided(self, validator):
        """Test validation when no file is provided."""
        # Execute & Verify
        with pytest.raises(ValueError, match="No file provided"):
            validator.validate_file(None, "test.mp3", 1024)

    def test_validate_file_no_filename(self, validator):
        """Test validation when filename is empty."""
        # Setup
        mock_file = Mock()

        # Execute & Verify
        with pytest.raises(ValueError, match="File must have a filename"):
            validator.validate_file(mock_file, "", 1024)

    def test_validate_file_invalid_extension(self, validator):
        """Test validation with invalid file extension."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.xyz"
        file_size = 1024

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}

            # Execute & Verify
            with pytest.raises(ValueError, match="File type '.xyz' not allowed"):
                validator.validate_file(mock_file, filename, file_size)

    def test_validate_file_too_large(self, validator):
        """Test validation with file size exceeding limit."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.mp3"
        file_size = 600 * 1024 * 1024  # 600MB

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024  # 500MB
            mock_settings.max_upload_size_mb = 500

            # Execute & Verify
            with pytest.raises(ValueError, match="exceeds maximum"):
                validator.validate_file(mock_file, filename, file_size)

    def test_validate_file_empty(self, validator):
        """Test validation with zero-size file."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.mp3"
        file_size = 0

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024

            # Execute & Verify
            with pytest.raises(ValueError, match="File is empty"):
                validator.validate_file(mock_file, filename, file_size)

    def test_validate_file_case_insensitive_extension(self, validator):
        """Test validation handles case-insensitive extensions."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.MP3"
        file_size = 1024

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024

            # Execute
            result = validator.validate_file(mock_file, filename, file_size)

        # Verify
        assert result == "mp3"


class TestValidateTranscriptionParams:
    """Test cases for validate_transcription_params method."""

    def test_validate_params_all_valid(self, validator):
        """Test validation with all valid parameters."""
        # Execute
        result = validator.validate_transcription_params(
            num_speakers=2,
            whisper_model=WhisperModel.MEDIUM,
            language="en"
        )

        # Verify
        assert result == {
            'num_speakers': 2,
            'whisper_model': 'medium',
            'language': 'en'
        }

    def test_validate_params_optional_num_speakers(self, validator):
        """Test validation with no num_speakers."""
        # Execute
        result = validator.validate_transcription_params(
            num_speakers=None,
            whisper_model=WhisperModel.MEDIUM,
            language=None
        )

        # Verify
        assert result == {'whisper_model': 'medium'}
        assert 'num_speakers' not in result
        assert 'language' not in result

    def test_validate_params_invalid_num_speakers_too_low(self, validator):
        """Test validation with num_speakers below minimum."""
        # Execute & Verify
        with pytest.raises(ValueError, match="num_speakers must be between 1 and 10"):
            validator.validate_transcription_params(
                num_speakers=0,
                whisper_model=WhisperModel.MEDIUM,
                language=None
            )

    def test_validate_params_invalid_num_speakers_too_high(self, validator):
        """Test validation with num_speakers above maximum."""
        # Execute & Verify
        with pytest.raises(ValueError, match="num_speakers must be between 1 and 10"):
            validator.validate_transcription_params(
                num_speakers=11,
                whisper_model=WhisperModel.MEDIUM,
                language=None
            )

    def test_validate_params_language_normalization(self, validator):
        """Test language code normalization."""
        # Execute
        result = validator.validate_transcription_params(
            num_speakers=None,
            whisper_model=WhisperModel.MEDIUM,
            language="  EN-US  "
        )

        # Verify
        assert result['language'] == 'en-us'

    def test_validate_params_language_too_long(self, validator):
        """Test validation with language code too long."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Language code too long"):
            validator.validate_transcription_params(
                num_speakers=None,
                whisper_model=WhisperModel.MEDIUM,
                language="toolong123"
            )

    def test_validate_params_language_invalid_characters(self, validator):
        """Test validation with invalid characters in language code."""
        # Execute & Verify
        with pytest.raises(ValueError, match="Language code contains invalid characters"):
            validator.validate_transcription_params(
                num_speakers=None,
                whisper_model=WhisperModel.MEDIUM,
                language="en@us"
            )

    def test_validate_params_whisper_model_enum(self, validator):
        """Test validation extracts value from WhisperModel enum."""
        # Execute
        result = validator.validate_transcription_params(
            num_speakers=None,
            whisper_model=WhisperModel.LARGE,
            language=None
        )

        # Verify
        assert result['whisper_model'] == 'large'


class TestCheckDiskSpace:
    """Test cases for check_disk_space method."""

    def test_check_disk_space_sufficient(self, validator):
        """Test disk space check when sufficient space is available."""
        # Setup
        mock_usage = MagicMock()
        mock_usage.free = 10 * (1024**3)  # 10GB free

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.temp_upload_dir = Path("/uploads")
            mock_settings.output_dir = Path("/outputs")

            with patch('shutil.disk_usage', return_value=mock_usage):
                # Execute - should not raise
                validator.check_disk_space(min_free_gb=5.0)

    def test_check_disk_space_insufficient_upload(self, validator):
        """Test disk space check when upload directory has insufficient space."""
        # Setup
        def mock_disk_usage(path):
            mock_usage = MagicMock()
            if 'upload' in str(path):
                mock_usage.free = 2 * (1024**3)  # 2GB free (insufficient)
            else:
                mock_usage.free = 10 * (1024**3)  # 10GB free
            return mock_usage

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.temp_upload_dir = Path("/uploads")
            mock_settings.output_dir = Path("/outputs")

            with patch('shutil.disk_usage', side_effect=mock_disk_usage):
                # Execute & Verify
                with pytest.raises(InsufficientDiskSpaceError, match="Upload directory"):
                    validator.check_disk_space(min_free_gb=5.0)

    def test_check_disk_space_insufficient_output(self, validator):
        """Test disk space check when output directory has insufficient space."""
        # Setup
        def mock_disk_usage(path):
            mock_usage = MagicMock()
            if 'output' in str(path):
                mock_usage.free = 2 * (1024**3)  # 2GB free (insufficient)
            else:
                mock_usage.free = 10 * (1024**3)  # 10GB free
            return mock_usage

        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.temp_upload_dir = Path("/uploads")
            mock_settings.output_dir = Path("/outputs")

            with patch('shutil.disk_usage', side_effect=mock_disk_usage):
                # Execute & Verify
                with pytest.raises(InsufficientDiskSpaceError, match="Output directory"):
                    validator.check_disk_space(min_free_gb=5.0)

    def test_check_disk_space_error_handling(self, validator):
        """Test disk space check handles errors gracefully."""
        # Setup - disk_usage raises an error
        with patch('app.validators.upload_validator.settings') as mock_settings:
            mock_settings.temp_upload_dir = Path("/uploads")
            mock_settings.output_dir = Path("/outputs")

            with patch('shutil.disk_usage', side_effect=OSError("Disk error")):
                # Execute - should not raise, just log warning
                validator.check_disk_space(min_free_gb=5.0)


class TestValidateAudioPath:
    """Test cases for validate_audio_path method."""

    def test_validate_audio_path_success(self, validator, tmp_path):
        """Test successful audio path validation."""
        # Setup - create a temporary file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Execute - should not raise
        validator.validate_audio_path(audio_file)

    def test_validate_audio_path_not_exists(self, validator):
        """Test validation when file doesn't exist."""
        # Setup
        audio_file = Path("/nonexistent/test.mp3")

        # Execute & Verify
        with pytest.raises(ValueError, match="Audio file not found"):
            validator.validate_audio_path(audio_file)

    def test_validate_audio_path_not_file(self, validator, tmp_path):
        """Test validation when path is a directory."""
        # Setup - create a directory instead of file
        audio_dir = tmp_path / "audio_dir"
        audio_dir.mkdir()

        # Execute & Verify
        with pytest.raises(ValueError, match="Audio path is not a file"):
            validator.validate_audio_path(audio_dir)

    def test_validate_audio_path_not_readable(self, validator, tmp_path):
        """Test validation when file is not readable."""
        # Setup
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock open to raise PermissionError
        with patch('pathlib.Path.open', side_effect=PermissionError("No permission")):
            # Execute & Verify
            with pytest.raises(ValueError, match="No permission to read file"):
                validator.validate_audio_path(audio_file)

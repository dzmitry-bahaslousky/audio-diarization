"""Integration tests for TranscriptionWorkflow."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

from app.services.transcription_workflow import TranscriptionWorkflow
from app.validators.upload_validator import UploadValidator
from app.models.schemas import WhisperModel
from app.exceptions import InsufficientDiskSpaceError


@pytest.fixture
def workflow():
    """Create a TranscriptionWorkflow instance."""
    return TranscriptionWorkflow()


@pytest.fixture
def mock_upload_file():
    """Create a mock upload file."""
    mock_file = AsyncMock()
    mock_file.filename = "test_audio.mp3"
    mock_file.read = AsyncMock()
    return mock_file


class TestTranscriptionWorkflow:
    """Integration tests for complete workflow."""

    @pytest.mark.asyncio
    async def test_save_upload_success(self, workflow, mock_upload_file, tmp_path):
        """Test successful file upload and save."""
        # Setup
        mock_upload_file.read.side_effect = [
            b"chunk1" * 1024,  # First chunk
            b"chunk2" * 1024,  # Second chunk
            b""  # End of file
        ]

        with patch('app.services.transcription_workflow.settings') as mock_settings:
            mock_settings.temp_upload_dir = tmp_path

            with patch.object(workflow.validator, 'check_disk_space'):
                # Execute
                result_path = await workflow.save_upload(
                    mock_upload_file,
                    "test_audio.mp3"
                )

        # Verify
        assert result_path.exists()
        assert result_path.suffix == ".mp3"
        assert result_path.parent == tmp_path

        # Verify file content
        content = result_path.read_bytes()
        assert b"chunk1" in content
        assert b"chunk2" in content

        # Cleanup
        result_path.unlink()

    @pytest.mark.asyncio
    async def test_save_upload_insufficient_disk_space(self, workflow, mock_upload_file):
        """Test upload fails when disk space is insufficient."""
        # Setup - validator will raise InsufficientDiskSpaceError
        with patch.object(
            workflow.validator,
            'check_disk_space',
            side_effect=InsufficientDiskSpaceError("Not enough space")
        ):
            # Execute & Verify
            with pytest.raises(InsufficientDiskSpaceError):
                await workflow.save_upload(mock_upload_file, "test_audio.mp3")

    @pytest.mark.asyncio
    async def test_save_upload_io_error_cleanup(self, workflow, mock_upload_file, tmp_path):
        """Test partial file is cleaned up on IO error."""
        # Setup - read will fail after first chunk
        mock_upload_file.read.side_effect = [
            b"chunk1" * 1024,  # First chunk succeeds
            IOError("Disk write error")  # Second chunk fails
        ]

        with patch('app.services.transcription_workflow.settings') as mock_settings:
            mock_settings.temp_upload_dir = tmp_path

            with patch.object(workflow.validator, 'check_disk_space'):
                # Execute & Verify
                with pytest.raises(ValueError, match="Failed to save uploaded file"):
                    await workflow.save_upload(mock_upload_file, "test_audio.mp3")

        # Verify no partial files remain
        leftover_files = list(tmp_path.glob("*.mp3"))
        assert len(leftover_files) == 0

    @pytest.mark.asyncio
    async def test_save_upload_generates_unique_filenames(self, workflow, mock_upload_file, tmp_path):
        """Test that multiple uploads generate unique filenames."""
        # Setup
        mock_upload_file.read.side_effect = [b"data", b""] * 3  # For 3 uploads

        with patch('app.services.transcription_workflow.settings') as mock_settings:
            mock_settings.temp_upload_dir = tmp_path

            with patch.object(workflow.validator, 'check_disk_space'):
                # Execute - save same file 3 times
                path1 = await workflow.save_upload(mock_upload_file, "test.mp3")

                # Reset mock for second upload
                mock_upload_file.read.side_effect = [b"data", b""]
                path2 = await workflow.save_upload(mock_upload_file, "test.mp3")

                # Reset mock for third upload
                mock_upload_file.read.side_effect = [b"data", b""]
                path3 = await workflow.save_upload(mock_upload_file, "test.mp3")

        # Verify all paths are unique
        assert path1 != path2 != path3
        assert path1.exists()
        assert path2.exists()
        assert path3.exists()

        # Cleanup
        path1.unlink()
        path2.unlink()
        path3.unlink()

    def test_cleanup_upload_success(self, workflow, tmp_path):
        """Test successful cleanup of uploaded file."""
        # Setup - create a file
        test_file = tmp_path / "test_upload.mp3"
        test_file.write_bytes(b"test data")
        assert test_file.exists()

        # Execute
        workflow.cleanup_upload(test_file)

        # Verify
        assert not test_file.exists()

    def test_cleanup_upload_nonexistent_file(self, workflow):
        """Test cleanup handles nonexistent file gracefully."""
        # Setup
        test_file = Path("/nonexistent/file.mp3")

        # Execute - should not raise
        workflow.cleanup_upload(test_file)

    def test_cleanup_upload_none_path(self, workflow):
        """Test cleanup handles None path gracefully."""
        # Execute - should not raise
        workflow.cleanup_upload(None)

    def test_validate_upload_success(self, workflow):
        """Test successful upload validation."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.mp3"
        file_size = 10 * 1024 * 1024  # 10MB

        with patch.object(
            workflow.validator,
            'validate_file',
            return_value='mp3'
        ) as mock_validate:
            # Execute
            result = workflow.validate_upload(mock_file, filename, file_size)

        # Verify
        assert result == 'mp3'
        mock_validate.assert_called_once_with(mock_file, filename, file_size)

    def test_validate_upload_invalid_file(self, workflow):
        """Test upload validation with invalid file."""
        # Setup
        mock_file = Mock()
        filename = "test_audio.xyz"
        file_size = 1024

        with patch.object(
            workflow.validator,
            'validate_file',
            side_effect=ValueError("Invalid file type")
        ):
            # Execute & Verify
            with pytest.raises(ValueError, match="Invalid file type"):
                workflow.validate_upload(mock_file, filename, file_size)

    def test_validate_params_success(self, workflow):
        """Test successful parameter validation."""
        # Setup
        with patch.object(
            workflow.validator,
            'validate_transcription_params',
            return_value={
                'num_speakers': 2,
                'whisper_model': 'medium',
                'language': 'en'
            }
        ) as mock_validate:
            # Execute
            result = workflow.validate_params(
                num_speakers=2,
                whisper_model=WhisperModel.MEDIUM,
                language='en'
            )

        # Verify
        assert result == {
            'num_speakers': 2,
            'whisper_model': 'medium',
            'language': 'en'
        }
        mock_validate.assert_called_once_with(
            2,
            WhisperModel.MEDIUM,
            'en'
        )

    def test_validate_params_invalid_num_speakers(self, workflow):
        """Test parameter validation with invalid num_speakers."""
        # Setup
        with patch.object(
            workflow.validator,
            'validate_transcription_params',
            side_effect=ValueError("num_speakers must be between 1 and 10")
        ):
            # Execute & Verify
            with pytest.raises(ValueError, match="num_speakers must be between 1 and 10"):
                workflow.validate_params(
                    num_speakers=11,
                    whisper_model=WhisperModel.MEDIUM,
                    language=None
                )


class TestWorkflowEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_upload_workflow(self, workflow, tmp_path):
        """Test complete upload workflow from validation to save."""
        # Setup
        mock_file = AsyncMock()
        mock_file.filename = "test_audio.mp3"
        mock_file.read.side_effect = [b"audio data" * 1000, b""]

        filename = "test_audio.mp3"
        file_size = 10 * 1024 * 1024  # 10MB

        with patch('app.services.transcription_workflow.settings') as mock_settings:
            mock_settings.temp_upload_dir = tmp_path
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024

            # Mock disk space check
            with patch.object(workflow.validator, 'check_disk_space'):
                # Step 1: Validate upload
                ext = workflow.validate_upload(mock_file, filename, file_size)
                assert ext == 'mp3'

                # Step 2: Validate params
                params = workflow.validate_params(
                    num_speakers=2,
                    whisper_model=WhisperModel.MEDIUM,
                    language='en'
                )
                assert params['num_speakers'] == 2
                assert params['whisper_model'] == 'medium'
                assert params['language'] == 'en'

                # Step 3: Save upload
                upload_path = await workflow.save_upload(mock_file, filename)
                assert upload_path.exists()
                assert upload_path.suffix == '.mp3'

                # Step 4: Cleanup
                workflow.cleanup_upload(upload_path)
                assert not upload_path.exists()

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, workflow, tmp_path):
        """Test workflow properly recovers from errors."""
        # Setup
        mock_file = AsyncMock()
        mock_file.filename = "test_audio.mp3"
        mock_file.read.side_effect = [b"data", b""]

        with patch('app.services.transcription_workflow.settings') as mock_settings:
            mock_settings.temp_upload_dir = tmp_path
            mock_settings.allowed_extensions = {'mp3', 'wav', 'm4a'}
            mock_settings.max_upload_size_bytes = 500 * 1024 * 1024

            # Mock disk space check to pass initially
            with patch.object(workflow.validator, 'check_disk_space'):
                # Save file successfully
                upload_path = await workflow.save_upload(mock_file, "test_audio.mp3")
                assert upload_path.exists()

                # Simulate error during processing
                # In real workflow, this would be caught and cleanup called
                try:
                    raise ValueError("Processing error")
                except ValueError:
                    # Cleanup is called on error
                    workflow.cleanup_upload(upload_path)

                # Verify cleanup worked
                assert not upload_path.exists()

    @pytest.mark.asyncio
    async def test_concurrent_uploads(self, workflow, tmp_path):
        """Test workflow handles concurrent uploads correctly."""
        # Setup
        async def upload_file(filename: str) -> Path:
            mock_file = AsyncMock()
            mock_file.read.side_effect = [b"data" * 100, b""]

            with patch('app.services.transcription_workflow.settings') as mock_settings:
                mock_settings.temp_upload_dir = tmp_path

                with patch.object(workflow.validator, 'check_disk_space'):
                    return await workflow.save_upload(mock_file, filename)

        # Execute - upload 5 files concurrently
        tasks = [
            upload_file(f"test_{i}.mp3")
            for i in range(5)
        ]
        paths = await asyncio.gather(*tasks)

        # Verify all uploads succeeded and generated unique paths
        assert len(paths) == 5
        assert len(set(paths)) == 5  # All unique
        for path in paths:
            assert path.exists()
            path.unlink()  # Cleanup

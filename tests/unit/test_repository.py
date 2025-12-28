"""Unit tests for TranscriptionRepository."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.transcription_repository import TranscriptionRepository
from app.models.database import TranscriptionJob
from app.models.schemas import TranscriptionStatus
from app.exceptions import JobCreationError, JobNotFoundError, DatabaseConnectionError


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = Mock(spec=Session)
    session.commit = Mock()
    session.rollback = Mock()
    session.add = Mock()
    session.query = Mock()
    return session


@pytest.fixture
def repository(mock_db_session):
    """Create a TranscriptionRepository with mocked database."""
    return TranscriptionRepository(mock_db_session)


class TestTranscriptionRepository:
    """Test cases for TranscriptionRepository."""

    def test_create_job_success(self, repository, mock_db_session):
        """Test successful job creation."""
        # Setup
        job_data = {
            'filename': 'test.mp3',
            'upload_path': '/uploads/test.mp3',
            'whisper_model': 'medium',
            'language': 'en',
            'num_speakers': 2
        }

        # Create a mock job that will be returned
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = '12345'

        # Mock the session.add to not raise errors
        mock_db_session.add.return_value = None
        mock_db_session.commit.return_value = None

        # Mock the refresh to set the id
        def mock_refresh(job):
            job.id = '12345'
        mock_db_session.refresh = Mock(side_effect=mock_refresh)

        # Execute
        with patch('app.repositories.transcription_repository.TranscriptionJob') as MockJob:
            MockJob.return_value = mock_job
            result = repository.create_job(**job_data)

        # Verify
        assert result == mock_job
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(mock_job)

    def test_create_job_database_error(self, repository, mock_db_session):
        """Test job creation with database error triggers rollback."""
        # Setup
        job_data = {
            'filename': 'test.mp3',
            'upload_path': '/uploads/test.mp3',
            'whisper_model': 'medium'
        }

        # Mock commit to raise SQLAlchemyError
        mock_db_session.commit.side_effect = SQLAlchemyError("Database error")

        # Execute & Verify
        with pytest.raises(JobCreationError):
            with patch('app.repositories.transcription_repository.TranscriptionJob'):
                repository.create_job(**job_data)

        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()

    def test_get_job_success(self, repository, mock_db_session):
        """Test successful job retrieval."""
        # Setup
        job_id = '12345'
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.get_job(job_id)

        # Verify
        assert result == mock_job
        mock_db_session.query.assert_called_once_with(TranscriptionJob)

    def test_get_job_not_found(self, repository, mock_db_session):
        """Test job retrieval when job doesn't exist."""
        # Setup
        job_id = 'nonexistent'

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.get_job(job_id)

        # Verify
        assert result is None

    def test_update_status_success(self, repository, mock_db_session):
        """Test successful status update."""
        # Setup
        job_id = '12345'
        new_status = TranscriptionStatus.COMPLETED
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = job_id
        mock_job.status = TranscriptionStatus.PENDING

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.update_status(job_id, new_status)

        # Verify
        assert result == mock_job
        assert mock_job.status == new_status
        mock_db_session.commit.assert_called_once()

    def test_update_status_job_not_found(self, repository, mock_db_session):
        """Test status update for non-existent job."""
        # Setup
        job_id = 'nonexistent'
        new_status = TranscriptionStatus.COMPLETED

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # Execute & Verify
        with pytest.raises(JobNotFoundError):
            repository.update_status(job_id, new_status)

    def test_update_results_success(self, repository, mock_db_session):
        """Test successful results update."""
        # Setup
        job_id = '12345'
        result_data = {
            'segments': [{'text': 'Hello', 'speaker': 'SPEAKER_00'}],
            'full_text': 'Hello',
            'speakers': ['SPEAKER_00']
        }
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.update_results(job_id, result_data)

        # Verify
        assert result == mock_job
        assert mock_job.status == TranscriptionStatus.COMPLETED
        assert mock_job.result_json == result_data
        assert mock_job.completed_at is not None
        mock_db_session.commit.assert_called_once()

    def test_update_results_database_error(self, repository, mock_db_session):
        """Test results update with database error triggers rollback."""
        # Setup
        job_id = '12345'
        result_data = {'segments': []}
        mock_job = Mock(spec=TranscriptionJob)

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query
        mock_db_session.commit.side_effect = SQLAlchemyError("Database error")

        # Execute & Verify
        with pytest.raises(DatabaseConnectionError):
            repository.update_results(job_id, result_data)

        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()

    def test_mark_failed_success(self, repository, mock_db_session):
        """Test successful failure marking."""
        # Setup
        job_id = '12345'
        error_message = 'Processing failed'
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.mark_failed(job_id, error_message)

        # Verify
        assert result == mock_job
        assert mock_job.status == TranscriptionStatus.FAILED
        assert mock_job.error_message == error_message
        assert mock_job.completed_at is not None
        mock_db_session.commit.assert_called_once()

    def test_transaction_context_manager_success(self, repository, mock_db_session):
        """Test transaction context manager commits on success."""
        # Execute
        with repository.transaction() as session:
            assert session == mock_db_session
            # Simulate some work
            pass

        # Verify
        mock_db_session.commit.assert_called_once()
        mock_db_session.rollback.assert_not_called()

    def test_transaction_context_manager_rollback_on_error(self, repository, mock_db_session):
        """Test transaction context manager rolls back on error."""
        # Execute & Verify
        with pytest.raises(DatabaseConnectionError):
            with repository.transaction():
                # Simulate error during transaction
                raise SQLAlchemyError("Database error")

        # Verify
        mock_db_session.rollback.assert_called_once()
        mock_db_session.commit.assert_not_called()

    def test_get_all_jobs(self, repository, mock_db_session):
        """Test retrieving all jobs."""
        # Setup
        mock_jobs = [
            Mock(spec=TranscriptionJob, id='1'),
            Mock(spec=TranscriptionJob, id='2')
        ]

        mock_query = Mock()
        mock_query.order_by.return_value.all.return_value = mock_jobs
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.get_all_jobs()

        # Verify
        assert result == mock_jobs
        assert len(result) == 2
        mock_db_session.query.assert_called_once_with(TranscriptionJob)

    def test_delete_job_success(self, repository, mock_db_session):
        """Test successful job deletion."""
        # Setup
        job_id = '12345'
        mock_job = Mock(spec=TranscriptionJob)
        mock_job.id = job_id

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_job
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.delete_job(job_id)

        # Verify
        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_job)
        mock_db_session.commit.assert_called_once()

    def test_delete_job_not_found(self, repository, mock_db_session):
        """Test deletion of non-existent job."""
        # Setup
        job_id = 'nonexistent'

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # Execute
        result = repository.delete_job(job_id)

        # Verify
        assert result is False
        mock_db_session.delete.assert_not_called()
        mock_db_session.commit.assert_not_called()

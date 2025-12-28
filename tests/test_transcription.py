"""Tests for transcription API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint():
    """Test detailed health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "upload_dir_exists" in data
    assert "output_dir_exists" in data


def test_transcribe_no_file():
    """Test transcribe endpoint with no file."""
    response = client.post("/api/transcribe")
    assert response.status_code == 422  # Unprocessable Entity


def test_transcribe_empty_file():
    """Test transcribe endpoint with empty file."""
    files = {"file": ("empty.mp3", b"", "audio/mpeg")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_invalid_extension():
    """Test transcribe endpoint with invalid file extension."""
    files = {"file": ("test.txt", b"fake audio data", "text/plain")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_transcribe_valid_file():
    """Test transcribe endpoint with valid MP3 file."""
    # Create a minimal MP3 file (fake data for testing)
    fake_mp3_data = b"fake mp3 data" * 100  # Make it non-empty

    files = {"file": ("test_audio.mp3", fake_mp3_data, "audio/mpeg")}
    data = {
        "whisper_model": "medium",
        "num_speakers": 2,
        "language": "en"
    }

    response = client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 202  # Accepted

    result = response.json()
    assert "job_id" in result
    assert result["status"] == "pending"


# TODO: Add more tests in future phases
# - Test status endpoint
# - Test result endpoint
# - Test file size limits
# - Integration tests with actual audio files

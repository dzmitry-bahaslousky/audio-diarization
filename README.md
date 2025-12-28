# Audio Transcription API with Speaker Diarization

A production-ready web application that transcribes audio files and identifies speakers using OpenAI Whisper and Pyannote, with asynchronous background processing powered by Celery.

## Features

### Core Functionality
- **Audio Transcription** - OpenAI Whisper models (tiny to large-v3)
- **Speaker Diarization** - Pyannote speaker identification and segmentation
- **Intelligent Alignment** - Matches speakers to transcribed text using temporal overlap
- **Multiple Export Formats** - TXT (simple/timeline/detailed) and structured JSON

### Technical Features
- **Async Background Processing** - Celery task queue with Redis broker
- **Parallel ML Processing** - Transcription and diarization run simultaneously
- **Automatic Retry Logic** - Exponential backoff for transient errors
- **Repository Pattern** - Clean database abstraction with automatic transaction management
- **RESTful API** - Built with FastAPI and automatic OpenAPI/Swagger docs
- **Job Status Tracking** - Real-time progress monitoring via PostgreSQL
- **Comprehensive Error Handling** - Distinguishes transient vs permanent failures
- **File Validation** - Upload size limits, format verification, content-type checking
- **Multiple Audio Formats** - MP3, WAV, M4A, FLAC, OGG, WMA

## Prerequisites

- **Python 3.10 or higher**
- **FFmpeg** - Audio format conversion
- **PostgreSQL** - Job persistence and result storage
- **Redis** - Message broker for Celery tasks
- **Hugging Face account** - API token for Pyannote models
- **Docker** (optional) - For running PostgreSQL and Redis via docker-compose
- **NVIDIA GPU with CUDA** (optional) - For faster processing

## Installation

### 1. Clone the repository

```bash
cd /Users/jcs/VSCode/audio-diarization
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token:

```env
HF_TOKEN=your_huggingface_token_here
```

**Get your Hugging Face token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Accept the conditions for: https://huggingface.co/pyannote/speaker-diarization-community-1

### 5. Install FFmpeg (if not already installed)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 6. Start PostgreSQL and Redis

Using Docker (recommended):

```bash
docker-compose up -d
```

Or install and run locally:
- PostgreSQL 15+
- Redis 7+

### 7. Verify database connection

The database will be automatically initialized on first startup. You can check the connection:

```bash
docker-compose logs postgres
docker-compose logs redis
```

## Running the Application

The application requires three components running simultaneously:

### 1. Start PostgreSQL and Redis (if not already running)

```bash
docker-compose up -d
```

### 2. Start FastAPI Server

**Development mode with hot reload:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode with multiple workers:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Start Celery Worker (in separate terminal)

**Development mode:**
```bash
celery -A app.celery_app worker --loglevel=info
```

**Production mode with multiple workers:**
```bash
celery -A app.celery_app worker --loglevel=info --concurrency=4
```

**With debug logging:**
```bash
celery -A app.celery_app worker --loglevel=debug
```

### Access the Application

- **API Base URL**: http://localhost:8000
- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Quick Start Example

```bash
# 1. Upload audio file for transcription
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@audio.mp3" \
  -F "num_speakers=2" \
  -F "whisper_model=medium"

# Response: {"job_id": "abc123...", "status": "pending", ...}

# 2. Check job status
curl "http://localhost:8000/api/status/abc123"

# 3. Get results when completed
curl "http://localhost:8000/api/result/abc123"

# 4. Export as text
curl "http://localhost:8000/api/export/abc123/txt?format=timeline"
```

## API Endpoints

### POST /api/transcribe

Upload an audio file for transcription with speaker diarization.

**Parameters:**
- `file` (required): Audio file (MP3, WAV, M4A, etc.)
- `num_speakers` (optional): Expected number of speakers (1-10)
- `whisper_model` (optional): Model size (tiny, base, small, medium, large)
- `language` (optional): Language code (e.g., 'en', 'es')

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@audio.mp3" \
  -F "num_speakers=2" \
  -F "whisper_model=medium" \
  -F "language=en"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Transcription job created for audio.mp3. Processing will begin shortly.",
  "created_at": "2025-12-27T10:30:00Z"
}
```

### GET /api/status/{job_id}

Check the current status of a transcription job.

**Example:**
```bash
curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
```

**Response (processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "filename": "audio.mp3",
  "created_at": "2025-12-27T10:30:00Z",
  "whisper_model": "medium"
}
```

**Response (completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "audio.mp3",
  "created_at": "2025-12-27T10:30:00Z",
  "completed_at": "2025-12-27T10:35:00Z",
  "audio_duration": 180.5,
  "detected_language": "en",
  "detected_speakers": 2,
  "whisper_model": "medium"
}
```

### GET /api/result/{job_id}

Get the complete transcription result for a completed job.

**Example:**
```bash
curl "http://localhost:8000/api/result/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "filename": "audio.mp3",
  "metadata": {
    "audio_duration": 180.5,
    "detected_language": "en",
    "detected_speakers": 2,
    "whisper_model": "medium",
    "created_at": "2025-12-27T10:30:00Z",
    "completed_at": "2025-12-27T10:35:00Z"
  },
  "result": {
    "full_text": "Complete transcription text...",
    "segments": [
      {
        "start": 0.0,
        "end": 3.5,
        "speaker": "SPEAKER_00",
        "text": "Hello, how are you?"
      }
    ],
    "speaker_timeline": "...",
    "speaker_groups": {
      "SPEAKER_00": [...],
      "SPEAKER_01": [...]
    }
  }
}
```

### GET /api/export/{job_id}/txt

Export transcription as plain text in various formats.

**Query Parameters:**
- `format` - Export format variant:
  - `simple` - Just the transcription text
  - `timeline` - Text with speaker labels (default)
  - `detailed` - Full timestamps with speakers

**Example:**
```bash
curl "http://localhost:8000/api/export/550e8400-e29b-41d4-a716-446655440000/txt?format=timeline"
```

### GET /api/export/{job_id}/json

Export complete transcription data as structured JSON.

**Example:**
```bash
curl "http://localhost:8000/api/export/550e8400-e29b-41d4-a716-446655440000/json"
```

### GET /

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Audio Transcription API",
  "version": "1.0.0"
}
```

### GET /health

Detailed health check with system status.

**Response:**
```json
{
  "status": "healthy",
  "upload_dir_exists": true,
  "output_dir_exists": true,
  "max_upload_mb": 500
}
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration from environment variables
│   ├── database.py                  # Database session management
│   ├── celery_app.py                # Celery app and background tasks
│   ├── dependencies.py              # Dependency injection factories
│   ├── exceptions.py                # Custom exception hierarchy
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py              # SQLAlchemy ORM models
│   │   └── schemas.py               # Pydantic models for API validation
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   └── transcription.py         # API endpoints
│   │
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── transcription_repository.py  # Database access layer
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_processor.py       # FFmpeg audio conversion
│   │   ├── transcription.py         # Whisper transcription
│   │   ├── diarization.py           # Pyannote speaker diarization
│   │   ├── alignment.py             # Speaker-text alignment
│   │   ├── export.py                # Output formatting
│   │   └── transcription_workflow.py # Workflow orchestration
│   │
│   ├── validators/
│   │   ├── __init__.py
│   │   └── upload_validator.py      # File upload validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py               # General utilities
│       ├── job_helpers.py           # Job-specific utilities
│       └── circuit_breaker.py       # Circuit breaker pattern
│
├── tests/
│   ├── __init__.py
│   └── test_transcription.py        # Test suite
│
├── uploads/                         # Temporary upload storage (gitignored)
├── outputs/                         # Processed results (gitignored)
│
├── docker-compose.yml               # PostgreSQL and Redis services
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .env                             # Local environment config (gitignored)
├── .gitignore
├── .dockerignore
├── CLAUDE.md                        # Development guide for Claude Code
└── README.md                        # This file
```

## Architecture

### Layered Design

The application follows a clean layered architecture:

- **API Layer** (`app/routers/`) - HTTP request handling and routing
- **Service Layer** (`app/services/`) - Business logic and ML model management
- **Repository Layer** (`app/repositories/`) - Database access abstraction
- **Validation Layer** (`app/validators/`) - Input validation and sanitization

### Async Processing Pipeline

1. **FastAPI receives request** → Validates file, saves to disk, creates DB record (PENDING)
2. **Job queued to Celery** → Task sent to Redis broker
3. **Celery worker processes** → ML pipeline runs in background
4. **Results stored** → Database updated (COMPLETED/FAILED), temp files cleaned

### Parallel ML Processing

Transcription (Whisper) and diarization (Pyannote) run **simultaneously** using `asyncio.gather()` for significant performance gains:

- **Serial Processing**: ~40 minutes for 1 hour of audio
- **Parallel Processing**: ~20-30 minutes for 1 hour of audio

### Key Design Patterns

1. **Repository Pattern** - Clean database abstraction with automatic transaction management
2. **Dependency Injection** - FastAPI `Depends()` for service lifecycle management
3. **Error Handling** - Custom exception hierarchy distinguishing transient vs permanent errors
4. **Retry Logic** - Automatic retry with exponential backoff for transient failures
5. **Service Singletons** - ML models loaded once per worker and cached globally

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

## Configuration

All configuration is managed through environment variables in `.env` file:

### Required Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token (required for Pyannote models) |
| `DATABASE_URL` | PostgreSQL connection string |
| `CELERY_BROKER_URL` | Redis URL for Celery task queue |
| `CELERY_RESULT_BACKEND` | Redis URL for Celery results |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | medium | Whisper model size (tiny, base, small, medium, large, large-v2, large-v3) |
| `WHISPER_DEVICE` | cpu | Device for Whisper (cpu, cuda, mps) |
| `DIARIZATION_DEVICE` | cpu | Device for Pyannote (cpu, cuda, mps) |
| `DIARIZATION_MODEL` | pyannote/speaker-diarization-community-1 | Pyannote model name |
| `MAX_UPLOAD_SIZE_MB` | 500 | Maximum file upload size in MB |
| `ALLOWED_EXTENSIONS` | mp3,wav,m4a,flac,ogg,wma | Allowed audio file extensions |
| `TEMP_UPLOAD_DIR` | ./uploads | Directory for temporary uploads |
| `OUTPUT_DIR` | ./outputs | Directory for processed outputs |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Example .env Configuration

```env
# Required
HF_TOKEN=your_huggingface_token_here
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/audio_diarization
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Optional
WHISPER_MODEL=medium
WHISPER_DEVICE=cpu
DIARIZATION_DEVICE=cpu
MAX_UPLOAD_SIZE_MB=500
LOG_LEVEL=INFO
```

## GPU Support

For CUDA GPU support:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For Mac M1/M2 (MPS):

```bash
# Already included in requirements.txt
# Set WHISPER_DEVICE=mps and DIARIZATION_DEVICE=mps in .env
```

## Troubleshooting

### Common Issues

#### "Failed to load model" or Model Loading Errors
- Ensure you have accepted the model conditions on Hugging Face: https://huggingface.co/pyannote/speaker-diarization-community-1
- Verify your `HF_TOKEN` is correct in `.env`
- Check Celery worker logs for detailed error: `docker-compose logs -f` or `celery -A app.celery_app worker --loglevel=debug`
- First request per worker is slow (loading models) - this is normal

#### "FFmpeg not found" or Audio Processing Errors
- Install FFmpeg using the instructions above
- Verify installation: `ffmpeg -version`
- Ensure FFmpeg is in your system PATH
- On macOS with Homebrew: `brew install ffmpeg`

#### Database Connection Errors
- Ensure PostgreSQL is running: `docker-compose ps`
- Check database logs: `docker-compose logs postgres`
- Verify `DATABASE_URL` in `.env` matches docker-compose settings
- Try restarting: `docker-compose restart postgres`

#### Celery Worker Not Processing Jobs
- Ensure Redis is running: `docker-compose ps`
- Check Redis logs: `docker-compose logs redis`
- Verify Celery worker is running: Check terminal where you started the worker
- Check for task errors in Celery logs
- Ensure `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` are correct

#### Out of Memory / OOM Errors
- Use a smaller Whisper model (tiny, base, small instead of medium/large)
- Process shorter audio files
- Enable GPU acceleration if available
- Reduce Celery worker concurrency: `celery -A app.celery_app worker --concurrency=1`
- On macOS with MPS: Set `WHISPER_DEVICE=mps` and `DIARIZATION_DEVICE=mps`

#### Jobs Stuck in "PROCESSING" Status
- Check Celery worker is running and not crashed
- Review worker logs for errors
- Verify sufficient disk space (5GB+ free recommended)
- Check if worker hit time limit (1 hour max per job)
- For very long audio files, consider splitting into smaller segments

#### "Insufficient disk space" Errors
- Free up disk space (need at least 5GB free)
- Clean up old uploads and outputs: `rm -rf uploads/* outputs/*`
- Increase disk space on system

#### Slow Processing Times
- Enable GPU acceleration (`WHISPER_DEVICE=cuda` or `WHISPER_DEVICE=mps`)
- Use smaller Whisper model for faster processing
- Ensure transcription and diarization are running in parallel (they should by default)
- Check CPU/GPU utilization during processing

### Debugging Tips

**View all logs:**
```bash
# Docker services
docker-compose logs -f

# Celery worker (if running locally)
celery -A app.celery_app worker --loglevel=debug
```

**Check job status in database:**
```bash
docker-compose exec postgres psql -U postgres -d audio_diarization -c "SELECT id, filename, status, created_at, error_message FROM transcription_jobs ORDER BY created_at DESC LIMIT 5;"
```

**Restart all services:**
```bash
docker-compose down
docker-compose up -d
# Then restart FastAPI and Celery worker
```

## Performance Considerations

### Processing Time Estimates

**With GPU (CUDA or Apple Silicon MPS):**

| Audio Length | Whisper (medium) | Diarization | Total (parallel) |
|--------------|------------------|-------------|------------------|
| 1 minute     | ~10-30s          | ~5-15s      | ~15-30s          |
| 10 minutes   | ~2-5 min         | ~1-2 min    | ~3-5 min         |
| 1 hour       | ~15-30 min       | ~5-10 min   | ~20-30 min       |

**CPU-only processing:** Expect 2-4x slower processing times.

### Model Size Trade-offs

- **tiny/base** - Fast but lower accuracy, good for testing and development
- **small** - Balanced for development, reasonable accuracy
- **medium** - **Recommended for production** - good balance of accuracy and speed
- **large/large-v2/large-v3** - Best accuracy but slow, requires more VRAM (8GB+)

### Optimization Tips

1. **Use GPU acceleration** when available (set `WHISPER_DEVICE=cuda` or `WHISPER_DEVICE=mps`)
2. **Parallel processing** is enabled by default (transcription + diarization run simultaneously)
3. **Model caching** - First request per worker is slow, subsequent requests are fast
4. **Adjust concurrency** - Reduce worker concurrency if running out of memory
5. **Choose appropriate model size** - Balance accuracy needs with performance requirements

## Production Deployment

### Recommended Setup

```bash
# 1. Use production-grade environment variables
DATABASE_URL=postgresql://user:pass@prod-db:5432/audio_diarization
CELERY_BROKER_URL=redis://prod-redis:6379/0

# 2. Run multiple Uvicorn workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 3. Run multiple Celery workers
celery -A app.celery_app worker --loglevel=info --concurrency=4

# 4. Use process manager (systemd, supervisor, or Docker)
# 5. Set up reverse proxy (nginx, traefik)
# 6. Configure SSL/TLS certificates
# 7. Set up monitoring and logging
```

### Security Considerations

- Set `ALLOWED_EXTENSIONS` to restrict file types
- Configure `MAX_UPLOAD_SIZE_MB` appropriately
- Use environment variables for sensitive data (never commit `.env`)
- Set up firewall rules to restrict database and Redis access
- Use HTTPS in production
- Implement rate limiting if publicly accessible
- Regularly update dependencies for security patches

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For bugs and feature requests, please open an issue.

## Support

- **Documentation**: See `CLAUDE.md` for detailed development guide
- **Issues**: Report bugs at GitHub Issues
- **API Documentation**: Available at `/docs` endpoint when running

## Acknowledgments

- **OpenAI Whisper** - Speech recognition: https://github.com/openai/whisper
- **Pyannote Audio** - Speaker diarization: https://github.com/pyannote/pyannote-audio
- **FastAPI** - Web framework: https://fastapi.tiangolo.com/
- **Celery** - Distributed task queue: https://docs.celeryproject.org/
- **PostgreSQL** - Database: https://www.postgresql.org/
- **Redis** - Message broker: https://redis.io/

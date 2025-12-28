"""Main FastAPI application entry point."""

import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, check_db_connection
from app.routers import transcription

# Suppress torchcodec warning (we use ffmpeg-python for audio processing)
warnings.filterwarnings("ignore", message=".*torchcodec.*")

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Audio Transcription API...")
    logger.info(f"Upload directory: {settings.temp_upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Max upload size: {settings.max_upload_size_mb}MB")
    logger.info(f"Whisper model: {settings.whisper_model}")

    # Initialize database
    logger.info("Checking database connection...")
    if not check_db_connection():
        logger.error("Database connection failed!")
        raise RuntimeError("Cannot connect to database")

    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully")

    # Create required directories
    logger.info("Creating required directories...")
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created upload directory: {settings.temp_upload_dir}")
    logger.info(f"Created output directory: {settings.output_dir}")

    yield

    # Shutdown
    logger.info("Shutting down Audio Transcription API...")


# Initialize FastAPI app
app = FastAPI(
    title="Audio Transcription API",
    description="Audio transcription with speaker diarization using Whisper and Pyannote",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    transcription.router,
    prefix="/api",
    tags=["transcription"]
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Audio Transcription API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "upload_dir_exists": settings.temp_upload_dir.exists(),
        "output_dir_exists": settings.output_dir.exists(),
        "max_upload_mb": settings.max_upload_size_mb
    }

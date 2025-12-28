"""Utility helper functions."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse HH:MM:SS.mmm timestamp to seconds.

    Args:
        timestamp: Formatted timestamp string

    Returns:
        Time in seconds
    """
    parts = timestamp.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def clean_temp_files(job_id: str, upload_dir: Path, output_dir: Path) -> None:
    """
    Clean up temporary files for a job.

    Args:
        job_id: Job identifier
        upload_dir: Upload directory path
        output_dir: Output directory path
    """
    # Find and remove files matching job_id
    for file_path in upload_dir.glob(f"{job_id}_*"):
        try:
            file_path.unlink()
            logger.info(f"Deleted temp file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")

    for file_path in output_dir.glob(f"{job_id}_*"):
        try:
            file_path.unlink()
            logger.info(f"Deleted output file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")

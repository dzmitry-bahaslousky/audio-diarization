"""Export service for formatting transcription results."""

import json
from datetime import datetime
from typing import Dict, List, Optional


class ExportService:
    """Handles export formatting for transcription results."""

    @staticmethod
    def format_txt_simple(result: Dict, metadata: Dict) -> str:
        """
        Generate simple plain text export (just the transcription).

        Args:
            result: Transcription result dict
            metadata: Job metadata (filename, timestamps, etc.)

        Returns:
            Plain text string
        """
        # Just return the full transcription text
        return result.get('text', '')

    @staticmethod
    def format_txt_timeline(result: Dict, metadata: Dict) -> str:
        """
        Generate timeline format with speaker labels.

        Args:
            result: Transcription result dict
            metadata: Job metadata

        Returns:
            Formatted text with speaker labels
        """
        lines = []

        # Add header with metadata
        lines.append(f"Transcription: {metadata.get('filename', 'Unknown')}")
        lines.append(f"Duration: {result.get('audio_duration', 0):.2f}s")
        lines.append(f"Language: {result.get('language', 'unknown').upper()}")
        lines.append(f"Speakers: {result.get('num_speakers', 0)}")
        lines.append(f"Date: {metadata.get('completed_at', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

        # Use pre-formatted speaker timeline
        lines.append(result.get('speaker_timeline', '').strip())

        return '\n'.join(lines)

    @staticmethod
    def format_txt_detailed(result: Dict, metadata: Dict) -> str:
        """
        Generate detailed format with timestamps and speaker labels.

        Args:
            result: Transcription result dict
            metadata: Job metadata

        Returns:
            Detailed formatted text with timestamps
        """
        lines = []

        # Add header
        lines.append(f"Transcription: {metadata.get('filename', 'Unknown')}")
        lines.append(f"Duration: {result.get('audio_duration', 0):.2f}s")
        lines.append(f"Language: {result.get('language', 'unknown').upper()}")
        lines.append(f"Speakers: {result.get('num_speakers', 0)}")
        lines.append(f"Model: {metadata.get('whisper_model', 'unknown')}")
        lines.append(f"Created: {metadata.get('created_at', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Completed: {metadata.get('completed_at', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

        # Add segments with timestamps
        segments = result.get('segments', [])
        for segment in segments:
            start = segment['start']
            end = segment['end']
            speaker = segment['speaker']
            text = segment['text'].strip()

            # Format: [00:00:12 - 00:00:15] SPEAKER_00: Hello, how are you?
            timestamp = f"[{ExportService._format_timestamp(start)} - {ExportService._format_timestamp(end)}]"
            lines.append(f"{timestamp} {speaker}: {text}")

        return '\n'.join(lines)

    @staticmethod
    def format_json(result: Dict, metadata: Dict, job_id: str) -> str:
        """
        Generate structured JSON export.

        Args:
            result: Transcription result dict
            metadata: Job metadata
            job_id: Job identifier

        Returns:
            Formatted JSON string
        """
        export_data = {
            'job_id': job_id,
            'metadata': {
                'filename': metadata.get('filename'),
                'created_at': metadata.get('created_at').isoformat() if metadata.get('created_at') else None,
                'completed_at': metadata.get('completed_at').isoformat() if metadata.get('completed_at') else None,
                'whisper_model': metadata.get('whisper_model'),
                'audio_duration': result.get('audio_duration'),
                'language': result.get('language'),
                'num_speakers': result.get('num_speakers'),
                'num_segments': result.get('num_segments')
            },
            'transcription': {
                'full_text': result.get('text'),
                'segments': result.get('segments', []),
                'speaker_timeline': result.get('speaker_timeline'),
                'speaker_groups': result.get('speaker_groups', {})
            }
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Format seconds as HH:MM:SS.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

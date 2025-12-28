"""Alignment service for merging transcription and diarization results."""

import logging
from typing import Dict, List, Optional

from app.exceptions import AlignmentError

logger = logging.getLogger(__name__)


class AlignmentService:
    """Handles alignment of transcription segments with speaker labels."""

    @staticmethod
    def align(
        transcription_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Align transcription segments with speaker labels.

        Args:
            transcription_segments: Whisper transcription segments
            diarization_segments: Pyannote diarization segments

        Returns:
            List of aligned segments with text and speaker labels

        Raises:
            AlignmentError: If input validation fails or alignment cannot be performed
        """
        # Validate inputs
        if transcription_segments is None:
            raise AlignmentError("Transcription segments cannot be None")

        if diarization_segments is None:
            raise AlignmentError("Diarization segments cannot be None")

        if not isinstance(transcription_segments, list):
            raise AlignmentError(
                f"Transcription segments must be a list, got {type(transcription_segments)}"
            )

        if not isinstance(diarization_segments, list):
            raise AlignmentError(
                f"Diarization segments must be a list, got {type(diarization_segments)}"
            )

        if len(transcription_segments) == 0:
            logger.warning("No transcription segments to align")
            return []

        logger.info(
            f"Aligning {len(transcription_segments)} transcription segments "
            f"with {len(diarization_segments)} speaker segments"
        )

        # Handle case of no diarization segments
        if len(diarization_segments) == 0:
            logger.warning("No diarization segments, marking all speakers as UNKNOWN")

        aligned_results = []

        try:
            for idx, trans_seg in enumerate(transcription_segments):
                # Validate required fields in transcription segment
                if not isinstance(trans_seg, dict):
                    raise AlignmentError(
                        f"Transcription segment {idx} must be a dict, got {type(trans_seg)}"
                    )

                if 'start' not in trans_seg or 'end' not in trans_seg:
                    raise AlignmentError(
                        f"Transcription segment {idx} missing required 'start' or 'end' field"
                    )

                if 'text' not in trans_seg:
                    raise AlignmentError(
                        f"Transcription segment {idx} missing required 'text' field"
                    )

                trans_start = float(trans_seg['start'])
                trans_end = float(trans_seg['end'])

                # Validate temporal consistency
                if trans_start < 0 or trans_end < 0:
                    raise AlignmentError(
                        f"Transcription segment {idx} has negative timestamp: "
                        f"start={trans_start}, end={trans_end}"
                    )

                if trans_end < trans_start:
                    raise AlignmentError(
                        f"Transcription segment {idx} has end before start: "
                        f"start={trans_start}, end={trans_end}"
                    )

                # Find speaker with maximum overlap
                best_speaker = "UNKNOWN"
                max_overlap = 0.0

                for diar_idx, diar_seg in enumerate(diarization_segments):
                    # Validate required fields in diarization segment
                    if not isinstance(diar_seg, dict):
                        raise AlignmentError(
                            f"Diarization segment {diar_idx} must be a dict, got {type(diar_seg)}"
                        )

                    if 'start' not in diar_seg or 'end' not in diar_seg:
                        raise AlignmentError(
                            f"Diarization segment {diar_idx} missing 'start' or 'end' field"
                        )

                    if 'speaker' not in diar_seg:
                        raise AlignmentError(
                            f"Diarization segment {diar_idx} missing 'speaker' field"
                        )

                    diar_start = float(diar_seg['start'])
                    diar_end = float(diar_seg['end'])

                    # Calculate temporal overlap
                    overlap_start = max(trans_start, diar_start)
                    overlap_end = min(trans_end, diar_end)
                    overlap = max(0.0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = diar_seg['speaker']

                # Create aligned segment
                aligned_results.append({
                    'start': trans_start,
                    'end': trans_end,
                    'text': trans_seg['text'],
                    'speaker': best_speaker,
                    'words': trans_seg.get('words', []),
                    'confidence': trans_seg.get('confidence', 0.0)
                })

        except (KeyError, ValueError, TypeError) as e:
            # Catch data structure or type conversion errors
            logger.error(f"Data validation error during alignment: {e}")
            raise AlignmentError(f"Invalid segment data structure: {e}")

        logger.info(f"Alignment complete: {len(aligned_results)} aligned segments")
        return aligned_results

    @staticmethod
    def group_by_speaker(aligned_segments: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group aligned segments by speaker.

        Args:
            aligned_segments: List of aligned segments

        Returns:
            Dictionary mapping speaker labels to their segments

        Raises:
            AlignmentError: If input validation fails
        """
        # Validate input
        if aligned_segments is None:
            raise AlignmentError("Aligned segments cannot be None")

        if not isinstance(aligned_segments, list):
            raise AlignmentError(
                f"Aligned segments must be a list, got {type(aligned_segments)}"
            )

        if len(aligned_segments) == 0:
            logger.warning("No aligned segments to group")
            return {}

        speaker_groups = {}

        try:
            for idx, segment in enumerate(aligned_segments):
                if not isinstance(segment, dict):
                    raise AlignmentError(
                        f"Segment {idx} must be a dict, got {type(segment)}"
                    )

                if 'speaker' not in segment:
                    raise AlignmentError(
                        f"Segment {idx} missing required 'speaker' field"
                    )

                speaker = segment['speaker']
                if speaker not in speaker_groups:
                    speaker_groups[speaker] = []
                speaker_groups[speaker].append(segment)

        except (KeyError, TypeError) as e:
            logger.error(f"Error grouping segments by speaker: {e}")
            raise AlignmentError(f"Invalid segment structure: {e}")

        return speaker_groups

    @staticmethod
    def get_speaker_timeline(aligned_segments: List[Dict]) -> str:
        """
        Generate a formatted speaker timeline.

        Args:
            aligned_segments: List of aligned segments

        Returns:
            Formatted timeline string

        Raises:
            AlignmentError: If input validation fails
        """
        # Validate input
        if aligned_segments is None:
            raise AlignmentError("Aligned segments cannot be None")

        if not isinstance(aligned_segments, list):
            raise AlignmentError(
                f"Aligned segments must be a list, got {type(aligned_segments)}"
            )

        if len(aligned_segments) == 0:
            logger.warning("No aligned segments for timeline")
            return ""

        timeline = []
        current_speaker = None

        try:
            for idx, segment in enumerate(aligned_segments):
                if not isinstance(segment, dict):
                    raise AlignmentError(
                        f"Segment {idx} must be a dict, got {type(segment)}"
                    )

                if 'speaker' not in segment:
                    raise AlignmentError(
                        f"Segment {idx} missing required 'speaker' field"
                    )

                if 'text' not in segment:
                    raise AlignmentError(
                        f"Segment {idx} missing required 'text' field"
                    )

                speaker = segment['speaker']
                text = str(segment['text']).strip()

                if speaker != current_speaker:
                    timeline.append(f"\n{speaker}:")
                    current_speaker = speaker

                timeline.append(f"  {text}")

        except (KeyError, TypeError) as e:
            logger.error(f"Error generating timeline: {e}")
            raise AlignmentError(f"Invalid segment structure: {e}")

        return '\n'.join(timeline)

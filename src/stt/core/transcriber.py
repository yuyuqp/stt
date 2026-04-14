"""Transcription engine using faster-whisper."""

from pathlib import Path
from typing import NamedTuple

from faster_whisper import WhisperModel


class TranscriptionResult(NamedTuple):
    """Result of a transcription operation."""
    text: str
    language: str
    language_probability: float


class TranscriptionConfig(NamedTuple):
    """Configuration for transcription."""
    model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True


class Transcriber:
    """Manages audio transcription using faster-whisper model."""

    def __init__(self):
        """Initialize transcriber with model cache."""
        self.model_cache: dict[tuple[str, str, str], WhisperModel] = {}

    def transcribe(
        self,
        audio_path: str,
        config: TranscriptionConfig | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            config: Transcription configuration (uses defaults if None)

        Returns:
            TranscriptionResult with text and language info

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if config is None:
            config = TranscriptionConfig()

        # Get or create model
        model_key = (config.model_name, config.device, config.compute_type)
        if model_key not in self.model_cache:
            self.model_cache[model_key] = WhisperModel(
                config.model_name,
                device=config.device,
                compute_type=config.compute_type,
            )

        model = self.model_cache[model_key]

        # Perform transcription
        segments, info = model.transcribe(
            audio_path,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
        )

        # Format output
        lines: list[str] = []
        for segment in segments:
            lines.append(f"[{segment.start:8.2f} -> {segment.end:8.2f}] {segment.text}")

        text = "\n".join(lines) + ("\n" if lines else "")

        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
        )

    def transcribe_with_segments(
        self,
        audio_path: str,
        config: TranscriptionConfig | None = None,
    ) -> tuple[list[dict], str, float]:
        """Transcribe audio and return raw segments.

        Args:
            audio_path: Path to audio file
            config: Transcription configuration

        Returns:
            Tuple of (segments list, language, language_probability)
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if config is None:
            config = TranscriptionConfig()

        model_key = (config.model_name, config.device, config.compute_type)
        if model_key not in self.model_cache:
            self.model_cache[model_key] = WhisperModel(
                config.model_name,
                device=config.device,
                compute_type=config.compute_type,
            )

        model = self.model_cache[model_key]

        segments, info = model.transcribe(
            audio_path,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
        )

        segments_list = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in segments
        ]

        return segments_list, info.language, info.language_probability

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.model_cache.clear()

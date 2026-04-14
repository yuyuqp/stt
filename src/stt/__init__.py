"""STT - Speech-to-Text application using faster-whisper."""

__version__ = "0.1.0"
__author__ = "Your Name"

from stt.core.transcriber import Transcriber, TranscriptionConfig, TranscriptionResult
from stt.core.subtitle import convert_file as convert_subtitle
from stt.ui.gui import launch_gui
from stt.ui.tui import launch_tui

__all__ = [
    "Transcriber",
    "TranscriptionConfig",
    "TranscriptionResult",
    "convert_subtitle",
    "launch_gui",
    "launch_tui",
]

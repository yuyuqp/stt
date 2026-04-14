"""
Textual TUI for faster-whisper STT application.
"""

import threading
from pathlib import Path
from typing import Any

from textual.app import ComposeResult, App
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)
from textual.screen import Screen
from textual.message import Message
from textual.binding import Binding

from faster_whisper import WhisperModel


class TranscriptionComplete(Message):
    """Message sent when transcription completes."""
    def __init__(self, text: str, info: str) -> None:
        self.text = text
        self.info = info
        super().__init__()


class TranscriptionError(Message):
    """Message sent when transcription fails."""
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class StatusBox(Static):
    """Displays transcription status."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update("Ready")


class TranscriptionScreen(Screen):
    """Main transcription interface."""
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
        color: $text;
    }

    #title {
        width: 1fr;
        height: auto;
        border: solid $accent;
        text-align: center;
        padding: 1;
        background: $boost;
    }

    #file_row {
        width: 1fr;
        height: auto;
        padding: 1;
    }

    #file_row Input {
        width: 1fr;
        margin: 0 1;
    }

    #file_row Label {
        width: auto;
        height: 1;
    }

    #file_row Button {
        width: auto;
        height: 1;
    }

    #config_row {
        width: 1fr;
        height: auto;
        padding: 1;
        layout: horizontal;
    }

    #config_row Vertical {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    #config_row Label {
        width: 1fr;
        height: auto;
    }

    #config_row Select {
        width: 1fr;
        height: auto;
    }

    #button_row {
        width: 1fr;
        height: auto;
        padding: 1;
    }

    #button_row Button {
        margin-right: 1;
    }

    #status {
        width: 1fr;
        height: auto;
        border: solid $success;
        padding: 1;
        text-align: center;
        background: $panel;
    }

    #transcript_view {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
    }

    Label {
        text-align: left;
    }

    .label {
        width: auto;
    }

    .input_field {
        width: 1fr;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_cache: dict[tuple[str, str, str], WhisperModel] = {}
        self.current_text = ""
        self.current_info = ""
        self.transcribing = False

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Vertical():
            yield Label("STT (faster-whisper) - Terminal UI", id="title")
            
            # File input section
            with Horizontal(id="file_row"):
                yield Label("Audio file:", classes="label")
                yield Input(id="audio_input", placeholder="Path to audio file", classes="input_field")
                yield Button("Browse", id="browse_btn")
            
            # Configuration section
            with Horizontal(id="config_row"):
                with Vertical():
                    yield Label("Model:")
                    yield Select(
                        [
                            ("tiny", "tiny"),
                            ("base", "base"),
                            ("small", "small"),
                            ("medium", "medium"),
                            ("large-v3", "large-v3"),
                        ],
                        value="large-v3",
                        id="model_select"
                    )
                
                with Vertical():
                    yield Label("Device:")
                    yield Select(
                        [
                            ("auto", "auto"),
                            ("cuda", "cuda"),
                            ("cpu", "cpu"),
                        ],
                        value="cuda",
                        id="device_select"
                    )
                
                with Vertical():
                    yield Label("Compute:")
                    yield Select(
                        [
                            ("float16", "float16"),
                            ("float32", "float32"),
                            ("int8", "int8"),
                        ],
                        value="float16",
                        id="compute_select"
                    )
                
                with Vertical():
                    yield Label("Beam Size:")
                    yield Select(
                        [(str(i), str(i)) for i in range(1, 21)],
                        value="5",
                        id="beam_select"
                    )
                
                with Vertical():
                    yield Label("VAD Filter:")
                    yield Checkbox(value=True, id="vad_checkbox")

            # Action buttons
            with Horizontal(id="button_row"):
                yield Button("Transcribe", id="transcribe_btn", variant="primary")
                yield Button("Transcribe & Save", id="transcribe_save_btn", variant="primary")
                yield Button("Save", id="save_btn", variant="warning")
                yield Button("Clear", id="clear_btn")

            # Status
            yield StatusBox("Ready", id="status")

            # Output section
            yield Label("Transcript:")
            yield TextArea(id="transcript_view")

    def on_mount(self) -> None:
        """Initialize screen."""
        self.title = "STT - faster-whisper"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "browse_btn":
            self.action_browse()
        elif event.button.id == "transcribe_btn":
            self.action_transcribe()
        elif event.button.id == "transcribe_save_btn":
            self.action_transcribe_and_save()
        elif event.button.id == "save_btn":
            self.action_save()
        elif event.button.id == "clear_btn":
            self.action_clear()

    def action_browse(self) -> None:
        """Open file browser."""
        def browse_worker():
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)

                path = filedialog.askopenfilename(
                    title="Select audio file",
                    filetypes=[
                        ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma"),
                        ("All files", "*.*"),
                    ],
                )

                root.destroy()

                if path:
                    def set_path():
                        self.query_one("#audio_input", Input).value = path
                    self.call_from_thread(set_path)
            except Exception as e:
                def show_error():
                    self.update_status(f"Error: {e}")
                self.call_from_thread(show_error)

        threading.Thread(target=browse_worker, daemon=True).start()

    def action_transcribe(self) -> None:
        """Perform transcription."""
        self._do_transcribe(save_after=False)

    def action_transcribe_and_save(self) -> None:
        """Perform transcription and save."""
        self._do_transcribe(save_after=True)

    def _do_transcribe(self, save_after: bool = False) -> None:
        """Internal transcription handler."""
        audio_input = self.query_one("#audio_input", Input)
        audio_path = audio_input.value.strip()

        if not audio_path or not Path(audio_path).exists():
            self.update_status("Error: Select a valid audio file")
            return

        if self.transcribing:
            self.update_status("Already transcribing...")
            return

        # Get configuration
        model = self.query_one("#model_select", Select).value
        device = self.query_one("#device_select", Select).value
        compute = self.query_one("#compute_select", Select).value
        beam = int(self.query_one("#beam_select", Select).value)
        vad = self.query_one("#vad_checkbox", Checkbox).value

        def transcribe_worker():
            try:
                self.transcribing = True
                self.call_from_thread(lambda: self.update_status("Preparing model…"))

                # Get or create model
                key = (model, device, compute)
                if key not in self.model_cache:
                    self.call_from_thread(lambda: self.update_status("Loading model…"))
                    self.model_cache[key] = WhisperModel(
                        model, device=device, compute_type=compute
                    )

                model_obj = self.model_cache[key]

                self.call_from_thread(lambda: self.update_status("Transcribing…"))

                segments, info = model_obj.transcribe(
                    audio_path,
                    beam_size=beam,
                    vad_filter=vad,
                )

                lines: list[str] = []
                for s in segments:
                    lines.append(f"[{s.start:8.2f} -> {s.end:8.2f}] {s.text}")
                text = "\n".join(lines) + ("\n" if lines else "")
                info_str = f"Language: {info.language} | Probability: {info.language_probability:.2f}"

                self.post_message(TranscriptionComplete(text, info_str))

                if save_after:
                    self.call_from_thread(self.action_save)

            except Exception as e:
                self.post_message(TranscriptionError(str(e)))

            finally:
                self.transcribing = False

        threading.Thread(target=transcribe_worker, daemon=True).start()

    def action_save(self) -> None:
        """Save transcript to file."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            text_area = self.query_one("#transcript_view", TextArea)
            content = text_area.text.rstrip("\n")

            if not content.strip():
                self.update_status("Nothing to save")
                return

            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)

            audio_input = self.query_one("#audio_input", Input)
            suggested = (
                Path(audio_input.value).with_suffix(".txt").name
                if audio_input.value.strip()
                else "transcript.txt"
            )

            out_path = filedialog.asksaveasfilename(
                title="Save transcript",
                defaultextension=".txt",
                initialfile=suggested,
                filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            )

            root.destroy()

            if out_path:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
                self.update_status(f"Saved: {out_path}")

        except Exception as e:
            self.update_status(f"Save error: {e}")

    def action_clear(self) -> None:
        """Clear the transcript."""
        text_area = self.query_one("#transcript_view", TextArea)
        text_area.text = ""
        self.current_text = ""
        self.current_info = ""
        self.query_one("#audio_input", Input).value = ""
        self.update_status("Cleared")

    def on_transcription_complete(self, message: TranscriptionComplete) -> None:
        """Handle transcription completion."""
        self.current_text = message.text
        self.current_info = message.info
        text_area = self.query_one("#transcript_view", TextArea)
        text_area.text = message.text
        self.update_status(f"Done. {message.info}")

    def on_transcription_error(self, message: TranscriptionError) -> None:
        """Handle transcription error."""
        self.update_status(f"Error: {message.error}")

    def update_status(self, text: str) -> None:
        """Update status display."""
        status = self.query_one("#status", StatusBox)
        status.update(text)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class STTApp(App):
    """Main STT application."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def on_mount(self) -> None:
        """Initialize the app."""
        self.title = "STT (faster-whisper)"
        self.push_screen(TranscriptionScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def launch_tui() -> int:
    """Launch the Textual TUI."""
    app = STTApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_tui())

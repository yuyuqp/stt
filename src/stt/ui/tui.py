"""Textual TUI for STT application."""

import threading
from pathlib import Path
from typing import Any

from textual.app import ComposeResult, App
from textual.containers import Horizontal, Vertical
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

from stt.core.transcriber import Transcriber, TranscriptionConfig
from stt.core.utils.windows_cuda import setup_windows_cuda_dlls


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


# ---------------------------------------------------------------------------
# Home screen
# ---------------------------------------------------------------------------


class HomeScreen(Screen):
    """Main menu – lets the user choose between features."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        align: center middle;
        background: $surface;
        color: $text;
    }

    #home_title {
        width: auto;
        height: auto;
        border: solid $accent;
        text-align: center;
        padding: 1 4;
        background: $boost;
        margin-bottom: 2;
    }

    #home_buttons {
        width: auto;
        height: auto;
        layout: vertical;
        align: center middle;
    }

    #home_buttons Button {
        width: 36;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="home_buttons"):
            yield Label("STT – faster-whisper", id="home_title")
            yield Button("🎙  Transcribe Audio", id="btn_transcribe", variant="primary")
            yield Button("📄  Convert Subtitles", id="btn_convert", variant="success")
            yield Button("Quit", id="btn_quit", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_transcribe":
            self.app.push_screen(TranscriptionScreen(defaults=self.app.defaults))
        elif event.button.id == "btn_convert":
            self.app.push_screen(SubtitleConversionScreen())
        elif event.button.id == "btn_quit":
            self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Subtitle conversion screen
# ---------------------------------------------------------------------------


class ConversionComplete(Message):
    """Message sent when subtitle conversion completes."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        super().__init__()


class ConversionError(Message):
    """Message sent when subtitle conversion fails."""

    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class SubtitleConversionScreen(Screen):
    """Screen for converting subtitle files between formats."""

    BINDINGS = [
        Binding("ctrl+c", "go_home", "Home", show=True),
        Binding("escape", "go_home", "Home", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
        color: $text;
    }

    #conv_title {
        width: 1fr;
        height: auto;
        border: solid $accent;
        text-align: center;
        padding: 1;
        background: $boost;
    }

    #conv_file_row {
        width: 1fr;
        height: auto;
        padding: 1;
    }

    #conv_file_row Input {
        width: 1fr;
        margin: 0 1;
    }

    #conv_file_row Label {
        width: auto;
        height: 1;
    }

    #conv_file_row Button {
        width: auto;
        height: 1;
    }

    #conv_format_row {
        width: 1fr;
        height: auto;
        padding: 1;
        layout: horizontal;
    }

    #conv_format_row Label {
        width: auto;
        height: 1;
        margin-right: 1;
    }

    #conv_format_row Select {
        width: 20;
    }

    #conv_output_row {
        width: 1fr;
        height: auto;
        padding: 1;
    }

    #conv_output_row Input {
        width: 1fr;
        margin: 0 1;
    }

    #conv_output_row Label {
        width: auto;
        height: 1;
    }

    #conv_button_row {
        width: 1fr;
        height: auto;
        padding: 1;
    }

    #conv_button_row Button {
        margin-right: 1;
    }

    #conv_status {
        width: 1fr;
        height: auto;
        border: solid $success;
        padding: 1;
        text-align: center;
        background: $panel;
    }

    #conv_detected {
        width: 1fr;
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }

    #conv_preview {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        from stt.core.subtitle import SUPPORTED_FORMATS

        with Vertical():
            yield Label("Subtitle Converter", id="conv_title")

            with Horizontal(id="conv_file_row"):
                yield Label("Input file:", classes="label")
                yield Input(
                    id="conv_input",
                    placeholder="Path to .srt or .vtt file",
                    classes="input_field",
                )
                yield Button("Browse", id="conv_browse_btn")

            yield Label("", id="conv_detected")

            with Horizontal(id="conv_format_row"):
                yield Label("Convert to:")
                yield Select(
                    [(f, f) for f in SUPPORTED_FORMATS],
                    value=SUPPORTED_FORMATS[0],
                    id="conv_format_select",
                )

            with Horizontal(id="conv_output_row"):
                yield Label("Output file:", classes="label")
                yield Input(
                    id="conv_output",
                    placeholder="Leave blank to auto-name",
                    classes="input_field",
                )

            with Horizontal(id="conv_button_row"):
                yield Button("Convert", id="conv_convert_btn", variant="primary")
                yield Button("Clear", id="conv_clear_btn")
                yield Button("← Home", id="conv_home_btn")

            yield StatusBox("Ready", id="conv_status")

            yield Label("Preview:")
            yield TextArea(id="conv_preview", read_only=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "conv_browse_btn":
            self._browse_input()
        elif event.button.id == "conv_convert_btn":
            self._do_convert()
        elif event.button.id == "conv_clear_btn":
            self._clear()
        elif event.button.id == "conv_home_btn":
            self.action_go_home()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "conv_input":
            self._update_detected_label(event.value.strip())

    def _update_detected_label(self, path: str) -> None:
        from stt.core.subtitle import detect_format

        detected = self.query_one("#conv_detected", Label)
        if path and Path(path).exists():
            fmt = detect_format(path)
            detected.update(f"Detected format: {fmt}")
        else:
            detected.update("")

    def _browse_input(self) -> None:
        def browse_worker():
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)

                path = filedialog.askopenfilename(
                    title="Select subtitle file",
                    filetypes=[
                        ("Subtitle files", "*.srt *.vtt"),
                        ("SRT", "*.srt"),
                        ("VTT", "*.vtt"),
                        ("All files", "*.*"),
                    ],
                )
                root.destroy()

                if path:
                    def set_path():
                        self.query_one("#conv_input", Input).value = path
                        self._update_detected_label(path)

                    self.app.call_from_thread(set_path)
            except Exception as e:
                def show_err():
                    self._set_status(f"Browse error: {e}")

                self.app.call_from_thread(show_err)

        threading.Thread(target=browse_worker, daemon=True).start()

    def _do_convert(self) -> None:
        from stt.core.subtitle import convert_file

        input_path = self.query_one("#conv_input", Input).value.strip()
        output_path = self.query_one("#conv_output", Input).value.strip() or None
        fmt = self.query_one("#conv_format_select", Select).value

        if not input_path:
            self._set_status("Error: Select an input file")
            return

        if not Path(input_path).exists():
            self._set_status("Error: Input file not found")
            return

        def worker():
            try:
                self.app.call_from_thread(lambda: self._set_status("Converting…"))
                result_path = convert_file(input_path, fmt, output_path)
                preview_text = result_path.read_text(encoding="utf-8")
                self.post_message(ConversionComplete(str(result_path)))
                self.app.call_from_thread(
                    lambda: self.query_one("#conv_preview", TextArea).__setattr__(
                        "text", preview_text
                    )
                )
            except Exception as e:
                self.post_message(ConversionError(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _clear(self) -> None:
        self.query_one("#conv_input", Input).value = ""
        self.query_one("#conv_output", Input).value = ""
        self.query_one("#conv_preview", TextArea).text = ""
        self.query_one("#conv_detected", Label).update("")
        self._set_status("Cleared")

    def on_conversion_complete(self, message: ConversionComplete) -> None:
        self._set_status(f"Done → {message.output_path}")

    def on_conversion_error(self, message: ConversionError) -> None:
        self._set_status(f"Error: {message.error}")

    def _set_status(self, text: str) -> None:
        self.query_one("#conv_status", StatusBox).update(text)

    def action_go_home(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Transcription screen (unchanged behaviour, back-button added)
# ---------------------------------------------------------------------------


class TranscriptionScreen(Screen):
    """Main transcription interface."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("escape", "go_home", "Home", show=True),
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

    def __init__(self, *args, defaults: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transcriber = Transcriber()
        self.defaults = defaults or {}
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
                yield Input(
                    id="audio_input",
                    placeholder="Path to audio file",
                    classes="input_field",
                )
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
                        value=self.defaults.get("model", "large-v3"),
                        id="model_select",
                    )

                with Vertical():
                    yield Label("Device:")
                    yield Select(
                        [
                            ("auto", "auto"),
                            ("cuda", "cuda"),
                            ("cpu", "cpu"),
                        ],
                        value=self.defaults.get("device", "cuda"),
                        id="device_select",
                    )

                with Vertical():
                    yield Label("Compute:")
                    yield Select(
                        [
                            ("float16", "float16"),
                            ("float32", "float32"),
                            ("int8", "int8"),
                        ],
                        value=self.defaults.get("compute_type", "float16"),
                        id="compute_select",
                    )

                with Vertical():
                    yield Label("Beam Size:")
                    yield Select(
                        [(str(i), str(i)) for i in range(1, 21)],
                        value=str(self.defaults.get("beam_size", 5)),
                        id="beam_select",
                    )

                with Vertical():
                    yield Label("VAD Filter:")
                    yield Checkbox(
                        value=self.defaults.get("vad_filter", True),
                        id="vad_checkbox",
                    )

                with Vertical():
                    yield Label("Condition Prev Text:")
                    yield Checkbox(
                        value=self.defaults.get("condition_on_previous_text", True),
                        id="condition_previous_text_checkbox",
                    )

            # Action buttons
            with Horizontal(id="button_row"):
                yield Button("Transcribe", id="transcribe_btn", variant="primary")
                yield Button(
                    "Transcribe & Save", id="transcribe_save_btn", variant="primary"
                )
                yield Button("Save", id="save_btn", variant="warning")
                yield Button("Clear", id="clear_btn")
                yield Button("← Home", id="home_btn")

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
        elif event.button.id == "home_btn":
            self.action_go_home()

    def action_browse(self) -> None:
        """Open file browser."""

        def browse_worker():
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)

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

                    self.app.call_from_thread(set_path)
            except Exception as e:

                def show_error():
                    self.update_status(f"Error: {e}")

                self.app.call_from_thread(show_error)

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
        condition_on_previous_text = self.query_one(
            "#condition_previous_text_checkbox", Checkbox
        ).value

        def transcribe_worker():
            try:
                self.transcribing = True
                self.app.call_from_thread(
                    lambda: self.update_status("Preparing model…")
                )

                setup_windows_cuda_dlls(
                    cublas_bin=self.defaults.get("cublas_bin"),
                    cudnn_bin=self.defaults.get("cudnn_bin"),
                    check=False,
                    force_load=False,
                )

                config = TranscriptionConfig(
                    model_name=model,
                    device=device,
                    compute_type=compute,
                    beam_size=beam,
                    vad_filter=vad,
                    condition_on_previous_text=condition_on_previous_text,
                )

                result = self.transcriber.transcribe(audio_path, config)

                info_str = f"Language: {result.language} | Probability: {result.language_probability:.2f}"

                self.post_message(TranscriptionComplete(result.text, info_str))

                if save_after:
                    self.app.call_from_thread(self.action_save)

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
            root.wm_attributes("-topmost", 1)

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

    def action_go_home(self) -> None:
        """Return to the home screen."""
        self.app.pop_screen()


class STTApp(App):
    """Main STT application."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self, defaults: dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.defaults = defaults or {}

    def on_mount(self) -> None:
        """Initialize the app."""
        self.title = "STT (faster-whisper)"
        self.push_screen(HomeScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def launch_tui(defaults: dict[str, Any] | None = None) -> int:
    """Launch the Textual TUI.

    Args:
        defaults: Default configuration values

    Returns:
        Exit code
    """
    app = STTApp(defaults=defaults)
    app.run()
    return 0

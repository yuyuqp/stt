# STT - Refactored Project Structure

## Overview

The STT project has been refactored for improved maintainability, testability, and scalability. The new structure follows Python best practices with clear separation of concerns.

## Directory Structure

```
stt/
├── src/
│   └── stt/                          # Main package
│       ├── __init__.py               # Package initialization & exports
│       ├── cli.py                    # Command-line argument parsing
│       ├── main.py                   # Main entry point
│       │
│       ├── core/                     # Core transcription functionality
│       │   ├── __init__.py
│       │   ├── transcriber.py        # Transcription engine
│       │   └── utils/
│       │       ├── __init__.py
│       │       └── windows_cuda.py   # Windows CUDA DLL setup
│       │
│       └── ui/                       # User interface modules
│           ├── __init__.py
│           ├── gui.py                # Tkinter-based GUI
│           └── tui.py                # Textual-based TUI
│
├── tests/                            # Unit and integration tests (optional)
├── fw_stt.py                         # Legacy entry point (backward compatibility)
├── fw_stt_tui.py                     # Legacy TUI entry point (backward compatibility)
├── pyproject.toml                    # Modern Python packaging configuration
├── README.md                         # Main documentation
└── TUI_IMPLEMENTATION.md             # TUI implementation details
```

## Module Organization

### `stt.core` - Transcription Engine

**Files:**
- `transcriber.py` - Main transcription logic

**Key Classes:**

```python
class TranscriptionConfig(NamedTuple):
    """Configuration for transcription."""
    model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True

class TranscriptionResult(NamedTuple):
    """Result of a transcription operation."""
    text: str
    language: str
    language_probability: float

class Transcriber:
    """Manages audio transcription using faster-whisper model."""
    def transcribe(audio_path: str, config: TranscriptionConfig) -> TranscriptionResult
    def transcribe_with_segments(...) -> tuple[list[dict], str, float]
    def clear_cache() -> None
```

**Features:**
- Model caching for performance
- Clean configuration via NamedTuple
- Structured return values
- Segment-level access option

### `stt.core.utils` - Windows CUDA Support

**Files:**
- `windows_cuda.py` - Windows CUDA DLL setup

**Key Functions:**
- `setup_windows_cuda_dlls()` - Configure CUDA paths on Windows
- `add_dll_dir()` - Add DLL directory to search path
- `default_site_packages_bin()` - Find default NVIDIA packages

### `stt.ui` - User Interfaces

#### GUI Module (`gui.py`)

**Function:**
```python
def launch_gui(defaults: dict[str, Any] | None = None) -> int
```

Features:
- Tkinter-based interface
- Optional drag-and-drop support
- Real-time transcription
- Save transcripts to file

#### TUI Module (`tui.py`)

**Function:**
```python
def launch_tui(defaults: dict[str, Any] | None = None) -> int
```

Features:
- Textual framework for modern terminal UI
- Interactive file browser
- Configuration controls
- Real-time status updates
- Keyboard navigation

### CLI Module (`cli.py`)

**Function:**
```python
def build_parser() -> argparse.ArgumentParser
```

Provides argument parsing for:
- Audio file input
- Model selection
- Device configuration
- Compute type selection
- Beam search size
- VAD filter toggle
- Windows CUDA configuration
- GUI/TUI/CLI mode selection

### Main Entry Point (`main.py`)

**Function:**
```python
def main(argv: list[str] | None = None) -> int
```

Handles:
- Argument parsing
- Mode routing (CLI, GUI, TUI)
- Configuration setup
- Transcription orchestration

## Usage

### As a Package

```python
from stt import Transcriber, TranscriptionConfig, launch_gui, launch_tui

# Direct transcription
transcriber = Transcriber()
config = TranscriptionConfig(model_name="base", device="cpu")
result = transcriber.transcribe("audio.wav", config)
print(result.text)

# Launch GUI
launch_gui()

# Launch TUI
launch_tui()
```

### As a CLI Tool

```bash
# CLI mode
python fw_stt.py audio.wav -o transcript.txt

# GUI mode
python fw_stt.py --gui

# TUI mode
python fw_stt.py --tui

# With options
python fw_stt.py audio.wav --model small --device cpu --beam-size 3
```

### Via pip (after installation)

```bash
# Install in editable mode
pip install -e .

# Use command
stt audio.wav --tui
```

## Design Principles

### Separation of Concerns

- **Core Logic**: `stt.core` handles all transcription
- **User Interfaces**: `stt.ui` provides different interfaces
- **CLI**: `stt.cli` handles argument parsing
- **Utilities**: `stt.core.utils` handles platform-specific code

### Dependency Injection

- Configuration passed as parameters
- No global state
- Easy to test and extend

### Type Hints

- Full Python 3.10+ type hints
- Clear function signatures
- NamedTuples for data structures

### Backward Compatibility

- Legacy `fw_stt.py` and `fw_stt_tui.py` still work
- Act as wrappers around refactored modules
- Gradual migration path

## Testing

Tests should be organized in `tests/` directory:

```
tests/
├── __init__.py
├── test_transcriber.py
├── test_cli.py
├── test_ui_gui.py
└── test_ui_tui.py
```

Run tests:
```bash
pytest
pytest --cov=src/stt  # With coverage
```

## Dependencies

**Core:**
- `faster-whisper>=1.0.0`

**UI:**
- `textual>=0.20.0` (for TUI)
- `tkinter` (usually included, for GUI)

**Optional:**
- `tkinterdnd2>=0.3.0` (for drag-and-drop in GUI)

**Development:**
- `pytest>=7.0`
- `pytest-cov>=4.0`
- `black>=23.0`
- `isort>=5.0`
- `flake8>=6.0`
- `mypy>=1.0`

## Package Installation

### Editable Installation

```bash
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### With Optional Drag-and-Drop

```bash
pip install -e ".[dnd]"
```

### With All Extras

```bash
pip install -e ".[dev,dnd]"
```

## Future Enhancements

1. **Diarization Support** - Speaker identification via `pyannote`
2. **Output Formats** - SRT, VTT subtitle formats; JSON, DOCX export
3. **Batch Processing** - Process multiple files
4. **Settings Persistence** - Save user preferences
5. **Plugins** - Extensible architecture for custom processors
6. **API Service** - FastAPI server for remote transcription
7. **Web UI** - Streamlit or FastAPI frontend

## Migration Guide

### For Users

Old way:
```bash
python fw_stt.py --tui
```

New way (same):
```bash
python fw_stt.py --tui
```

OR (if installed):
```bash
stt --tui
```

### For Developers

Old (monolithic):
```python
# Everything was in fw_stt.py
from fw_stt import _transcribe_text
```

New (modular):
```python
# Import from appropriate module
from stt.core.transcriber import Transcriber, TranscriptionConfig
from stt.ui.gui import launch_gui
from stt.ui.tui import launch_tui
```

## Code Style

The project follows:
- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Type checking

Configure your editor to use these tools automatically.

## Contributing

1. Create a feature branch
2. Add tests for new functionality
3. Run formatters and linters
4. Submit a pull request

## License

See LICENSE file for details.

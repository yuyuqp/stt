"""Command-line interface argument parsing."""

import argparse

from stt.core.subtitle import SUPPORTED_FORMATS


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the transcription command.

    Returns:
        ArgumentParser configured for STT application
    """
    p = argparse.ArgumentParser(
        prog="stt",
        description=(
            "Speech-to-text transcription and subtitle conversion.\n\n"
            "Subcommands:\n"
            "  convert    Convert subtitle files between formats (srt, vtt, txt)\n\n"
            "Run without a subcommand to transcribe an audio file or launch the UI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio",
        nargs="?",
        help="Path to the audio file (omit to launch GUI)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="conversation.txt",
        help="Output transcript path (default: conversation.txt)",
    )
    p.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Compute device (default: cuda)",
    )
    p.add_argument(
        "--compute-type",
        default="float16",
        help='Compute type (e.g. "float16", "int8") (default: float16)',
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size (default: 5)",
    )

    vad = p.add_mutually_exclusive_group()
    vad.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        default=True,
    )
    vad.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
    )

    condition = p.add_mutually_exclusive_group()
    condition.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_true",
        default=True,
        help="Condition each segment on previously generated text (default: enabled)",
    )
    condition.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable conditioning on previous text between segments",
    )

    p.add_argument(
        "--cublas-bin",
        default=None,
        help="Override cublas bin directory for Windows DLL loading",
    )
    p.add_argument(
        "--cudnn-bin",
        default=None,
        help="Override cudnn bin directory for Windows DLL loading",
    )
    p.add_argument(
        "--skip-dll-check",
        action="store_true",
        help="Skip existence checks for CUDA DLLs",
    )
    p.add_argument(
        "--skip-dll-load",
        action="store_true",
        help="Skip force-loading CUDA DLLs (still adds DLL directories)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI (ignores the audio argument)",
    )
    p.add_argument(
        "--tui",
        action="store_true",
        help="Launch the Terminal User Interface (TUI)",
    )

    return p


def build_convert_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``convert`` subcommand.

    Returns:
        ArgumentParser for subtitle conversion
    """
    p = argparse.ArgumentParser(
        prog="stt convert",
        description=(
            "Convert a subtitle file between formats.\n\n"
            "Supported input formats : .srt, .vtt\n"
            "Supported output formats: srt, vtt, txt (plain text without timestamps)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="Path to the source subtitle file (.srt or .vtt)",
    )
    p.add_argument(
        "--to",
        dest="output_format",
        required=True,
        choices=SUPPORTED_FORMATS,
        metavar="FORMAT",
        help=f"Target format. Choices: {', '.join(SUPPORTED_FORMATS)}",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file path.  When omitted the output is placed next to the "
            "input file with the appropriate extension."
        ),
    )
    return p

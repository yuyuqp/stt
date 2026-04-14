"""Command-line interface argument parsing."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        ArgumentParser configured for STT application
    """
    p = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper (optionally priming CUDA DLL paths on Windows)."
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

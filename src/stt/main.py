"""Main entry point for STT application."""

from pathlib import Path

from stt.cli import build_parser
from stt.core.transcriber import Transcriber, TranscriptionConfig
from stt.core.utils.windows_cuda import setup_windows_cuda_dlls
from stt.ui.gui import launch_gui
from stt.ui.tui import launch_tui


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code
    """
    args = build_parser().parse_args(argv)

    # Prepare defaults dict for UI modes
    defaults = {
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "vad_filter": args.vad_filter,
        "cublas_bin": args.cublas_bin,
        "cudnn_bin": args.cudnn_bin,
    }

    # Handle TUI mode
    if args.tui:
        return launch_tui(defaults=defaults)

    # Handle GUI mode
    if args.gui or not args.audio:
        return launch_gui(defaults=defaults)

    # Handle CLI mode (transcribe file)
    setup_windows_cuda_dlls(
        cublas_bin=args.cublas_bin,
        cudnn_bin=args.cudnn_bin,
        check=not args.skip_dll_check,
        force_load=not args.skip_dll_load,
    )

    transcriber = Transcriber()
    config = TranscriptionConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )

    try:
        result = transcriber.transcribe(args.audio, config)
        print(f"Language: {result.language}, Probability: {result.language_probability:.2f}")

        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.text)

        print(f"Transcript saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

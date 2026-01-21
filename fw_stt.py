import argparse
import ctypes
import importlib.util
import os
from pathlib import Path

from faster_whisper import WhisperModel


_DLL_HANDLES: list[object] = []


def _add_dll_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    h = os.add_dll_directory(path)
    _DLL_HANDLES.append(h)  # keep handle alive
    os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")  # belt + suspenders
    return True


def _default_nvidia_bin(pkg: str) -> str | None:
    spec = importlib.util.find_spec(pkg)
    if spec is None or spec.submodule_search_locations is None:
        return None
    base = next(iter(spec.submodule_search_locations), None)
    if not base:
        return None
    return str(Path(base) / "bin")


def _setup_windows_cuda_dlls(
    *,
    cublas_bin: str | None,
    cudnn_bin: str | None,
    check: bool,
    force_load: bool,
) -> None:
    if os.name != "nt":
        return

    # Fallbacks from your working environment (kept for reliability).
    fallback_cublas = (
        r"C:\Users\yuyue\AppData\Local\Programs\Python\Python312\Lib\site-packages\nvidia\cublas\bin"
    )
    fallback_cudnn = (
        r"C:\Users\yuyue\AppData\Local\Programs\Python\Python312\Lib\site-packages\nvidia\cudnn\bin"
    )

    cublas_bin = cublas_bin or _default_nvidia_bin("nvidia.cublas") or fallback_cublas
    cudnn_bin = cudnn_bin or _default_nvidia_bin("nvidia.cudnn") or fallback_cudnn

    added = [
        ("cublas", _add_dll_dir(cublas_bin), cublas_bin),
        ("cudnn", _add_dll_dir(cudnn_bin), cudnn_bin),
    ]

    print("Added DLL dirs:")
    for name, ok, path in added:
        print(f"  {name}: {ok}  ({path})")

    if not check and not force_load:
        return

    cublas_dll = os.path.join(cublas_bin, "cublas64_12.dll")
    cudnn_dll = os.path.join(cudnn_bin, "cudnn_ops64_9.dll")

    if check:
        print("Exists cublas64_12.dll:", os.path.exists(cublas_dll))
        print("Exists cudnn_ops64_9.dll:", os.path.exists(cudnn_dll))

    if force_load:
        if not os.path.exists(cublas_dll):
            raise FileNotFoundError(f"Missing DLL: {cublas_dll}")
        if not os.path.exists(cudnn_dll):
            raise FileNotFoundError(f"Missing DLL: {cudnn_dll}")

        try:
            ctypes.WinDLL(cublas_dll)
            print("WinDLL load OK: cublas64_12.dll")
        except OSError:
            print("WinDLL load FAILED: cublas64_12.dll")
            raise

        try:
            ctypes.WinDLL(cudnn_dll)
            print("WinDLL load OK: cudnn_ops64_9.dll")
        except OSError:
            print("WinDLL load FAILED: cudnn_ops64_9.dll")
            raise


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper (optionally priming CUDA DLL paths on Windows)."
    )
    p.add_argument("audio", help="Path to the audio file")
    p.add_argument(
        "-o",
        "--output",
        default="conversation.txt",
        help="Output transcript path (default: conversation.txt)",
    )
    p.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
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
    p.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")

    vad = p.add_mutually_exclusive_group()
    vad.add_argument("--vad-filter", dest="vad_filter", action="store_true", default=True)
    vad.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")

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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    _setup_windows_cuda_dlls(
        cublas_bin=args.cublas_bin,
        cudnn_bin=args.cudnn_bin,
        check=not args.skip_dll_check,
        force_load=not args.skip_dll_load,
    )

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, info = model.transcribe(
        args.audio,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )

    print("language:", info.language, "prob:", info.language_probability)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{s.start:8.2f} -> {s.end:8.2f}] {s.text}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


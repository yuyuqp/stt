import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import sysconfig
import threading
from typing import Any

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


def _default_site_packages_bin(*parts: str) -> str | None:
    # Avoid hard-coded absolute paths; work for venv + system installs.
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        base = paths.get(key)
        if not base:
            continue
        candidate = Path(base).joinpath(*parts)
        if candidate.is_dir():
            return str(candidate)
    return None


def _setup_windows_cuda_dlls(
    *,
    cublas_bin: str | None,
    cudnn_bin: str | None,
    check: bool,
    force_load: bool,
) -> None:
    if os.name != "nt":
        return

    # Fallbacks from the current interpreter's site-packages (venv/system).
    fallback_cublas = _default_site_packages_bin("nvidia", "cublas", "bin")
    fallback_cudnn = _default_site_packages_bin("nvidia", "cudnn", "bin")

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
    p.add_argument("audio", nargs="?", help="Path to the audio file (omit to launch GUI)")
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


def _transcribe_text(
    *,
    audio_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
    model_cache: dict[tuple[str, str, str], WhisperModel] | None = None,
) -> tuple[str, str]:
    if model_cache is None:
        model_cache = {}

    key = (model_name, device, compute_type)
    model = model_cache.get(key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        model_cache[key] = model

    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    lines: list[str] = []
    for s in segments:
        lines.append(f"[{s.start:8.2f} -> {s.end:8.2f}] {s.text}")
    return "\n".join(lines) + ("\n" if lines else ""), f"language: {info.language} prob: {info.language_probability}"


def _launch_gui(defaults: argparse.Namespace) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    # Optional drag-and-drop support via tkinterdnd2.
    TkBase: Any
    DND_FILES: Any
    try:
        from tkinterdnd2 import DND_FILES as _DND_FILES  # type: ignore
        from tkinterdnd2 import TkinterDnD as _TkinterDnD  # type: ignore

        TkBase = _TkinterDnD.Tk
        DND_FILES = _DND_FILES
        dnd_available = True
    except Exception:
        TkBase = tk.Tk
        DND_FILES = None
        dnd_available = False

    root = TkBase()
    root.title("STT (faster-whisper)")
    root.minsize(720, 520)

    model_cache: dict[tuple[str, str, str], WhisperModel] = {}

    audio_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Drop an audio file here or click Browse…")
    running_var = tk.BooleanVar(value=False)

    model_var = tk.StringVar(value=str(getattr(defaults, "model", "large-v3")))
    device_var = tk.StringVar(value=str(getattr(defaults, "device", "cuda")))
    compute_var = tk.StringVar(value=str(getattr(defaults, "compute_type", "float16")))
    beam_var = tk.IntVar(value=int(getattr(defaults, "beam_size", 5)))
    vad_var = tk.BooleanVar(value=bool(getattr(defaults, "vad_filter", True)))

    def set_running(is_running: bool) -> None:
        running_var.set(is_running)
        state = "disabled" if is_running else "normal"
        browse_btn.configure(state=state)
        transcribe_btn.configure(state=state)
        save_btn.configure(state=state if transcript_text.get("1.0", "end").strip() else "disabled")

    def set_status(text: str) -> None:
        status_var.set(text)

    def _safe_split_drop(data: str) -> list[str]:
        # Tcl list parsing handles braces around paths with spaces.
        try:
            return list(root.tk.splitlist(data))
        except Exception:
            return [data]

    def set_audio_path(path: str) -> None:
        p = path.strip().strip('"')
        if not p:
            return
        if not os.path.exists(p):
            messagebox.showerror("File not found", f"Audio file not found:\n{p}")
            return
        audio_var.set(p)
        set_status(f"Selected: {p}")

    def on_browse() -> None:
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma"),
                ("All files", "*.*"),
            ],
        )
        if path:
            set_audio_path(path)

    def on_drop(event: Any) -> None:
        paths = _safe_split_drop(getattr(event, "data", "") or "")
        if not paths:
            return
        set_audio_path(paths[0])

    def do_transcribe(save_after: bool) -> None:
        audio_path = audio_var.get().strip()
        if not audio_path:
            messagebox.showwarning("No input", "Choose an audio file first.")
            return

        def worker() -> None:
            try:
                root.after(0, lambda: (set_running(True), set_status("Preparing model…")))

                _setup_windows_cuda_dlls(
                    cublas_bin=getattr(defaults, "cublas_bin", None),
                    cudnn_bin=getattr(defaults, "cudnn_bin", None),
                    check=False,
                    force_load=False,
                )

                text, info = _transcribe_text(
                    audio_path=audio_path,
                    model_name=model_var.get().strip(),
                    device=device_var.get().strip(),
                    compute_type=compute_var.get().strip(),
                    beam_size=int(beam_var.get()),
                    vad_filter=bool(vad_var.get()),
                    model_cache=model_cache,
                )

                def finish_ui() -> None:
                    transcript_text.delete("1.0", "end")
                    transcript_text.insert("1.0", text)
                    set_status(f"Done. {info}")
                    set_running(False)
                    if save_after:
                        on_save()

                root.after(0, finish_ui)
            except Exception as e:
                err_msg = str(e)
                root.after(
                    0,
                    lambda err_msg=err_msg: (
                        set_running(False),
                        set_status("Failed."),
                        messagebox.showerror("Transcription failed", err_msg),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_transcribe() -> None:
        do_transcribe(save_after=False)

    def on_transcribe_and_save() -> None:
        do_transcribe(save_after=True)

    def on_save() -> None:
        content = transcript_text.get("1.0", "end").rstrip("\n")
        if not content.strip():
            messagebox.showwarning("Nothing to save", "Transcribe something first.")
            return

        suggested = Path(audio_var.get()).with_suffix(".txt").name if audio_var.get().strip() else "conversation.txt"
        out_path = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            initialfile=suggested,
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not out_path:
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        set_status(f"Saved: {out_path}")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(3, weight=1)

    top = ttk.Frame(root, padding=12)
    top.grid(row=0, column=0, sticky="nsew")
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text="Audio file:").grid(row=0, column=0, sticky="w")
    audio_entry = ttk.Entry(top, textvariable=audio_var)
    audio_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
    browse_btn = ttk.Button(top, text="Browse…", command=on_browse)
    browse_btn.grid(row=0, column=2, sticky="e")

    drop_hint = "(Drag & drop enabled)" if dnd_available else "(Install tkinterdnd2 for drag & drop)"
    drop_frame = ttk.LabelFrame(top, text=f"Drop zone {drop_hint}", padding=12)
    drop_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    drop_label = ttk.Label(drop_frame, textvariable=status_var, anchor="center")
    drop_label.grid(row=0, column=0, sticky="ew")
    drop_frame.columnconfigure(0, weight=1)

    if dnd_available:
        try:
            drop_frame.drop_target_register(DND_FILES)
            drop_frame.dnd_bind("<<Drop>>", on_drop)
            drop_label.drop_target_register(DND_FILES)
            drop_label.dnd_bind("<<Drop>>", on_drop)
        except Exception:
            # If DnD registration fails for any reason, just keep Browse working.
            pass

    opts = ttk.Frame(root, padding=(12, 0, 12, 12))
    opts.grid(row=1, column=0, sticky="ew")
    for i in range(8):
        opts.columnconfigure(i, weight=0)
    opts.columnconfigure(7, weight=1)

    ttk.Label(opts, text="Model").grid(row=0, column=0, sticky="w")
    model_box = ttk.Combobox(opts, textvariable=model_var, width=18, values=[
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
    ])
    model_box.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Device").grid(row=0, column=2, sticky="w")
    device_box = ttk.Combobox(opts, textvariable=device_var, width=10, values=["auto", "cuda", "cpu"])
    device_box.grid(row=0, column=3, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Compute").grid(row=0, column=4, sticky="w")
    compute_box = ttk.Combobox(opts, textvariable=compute_var, width=12, values=["float16", "float32", "int8"])
    compute_box.grid(row=0, column=5, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Beam").grid(row=0, column=6, sticky="w")
    beam_spin = ttk.Spinbox(opts, from_=1, to=20, textvariable=beam_var, width=5)
    beam_spin.grid(row=0, column=7, sticky="w")

    vad_check = ttk.Checkbutton(opts, text="VAD filter", variable=vad_var)
    vad_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    actions = ttk.Frame(root, padding=(12, 0, 12, 12))
    actions.grid(row=2, column=0, sticky="ew")
    transcribe_btn = ttk.Button(actions, text="Transcribe", command=on_transcribe)
    transcribe_btn.grid(row=0, column=0, sticky="w")
    ttk.Button(actions, text="Transcribe & Save…", command=on_transcribe_and_save).grid(row=0, column=1, sticky="w", padx=(8, 0))
    save_btn = ttk.Button(actions, text="Save…", command=on_save, state="disabled")
    save_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))

    # Transcript preview
    preview = ttk.Frame(root, padding=(12, 0, 12, 12))
    preview.grid(row=3, column=0, sticky="nsew")
    preview.columnconfigure(0, weight=1)
    preview.rowconfigure(0, weight=1)

    transcript_text = tk.Text(preview, wrap="word")
    transcript_text.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(preview, orient="vertical", command=transcript_text.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    transcript_text.configure(yscrollcommand=scroll.set)

    def poll_save_state() -> None:
        if not running_var.get():
            save_btn.configure(state="normal" if transcript_text.get("1.0", "end").strip() else "disabled")
        root.after(300, poll_save_state)

    poll_save_state()

    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.tui:
        from fw_stt_tui import launch_tui
        return launch_tui()

    if args.gui or not args.audio:
        return _launch_gui(args)

    _setup_windows_cuda_dlls(
        cublas_bin=args.cublas_bin,
        cudnn_bin=args.cudnn_bin,
        check=not args.skip_dll_check,
        force_load=not args.skip_dll_load,
    )

    text, info = _transcribe_text(
        audio_path=args.audio,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        model_cache={},
    )

    print(info)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Tkinter GUI for STT application."""

import os
import threading
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from stt.core.transcriber import Transcriber, TranscriptionConfig
from stt.core.utils.windows_cuda import setup_windows_cuda_dlls


# Optional drag-and-drop support via tkinterdnd2.
TkBase: Any
DND_FILES: Any
try:
    from tkinterdnd2 import DND_FILES as _DND_FILES  # type: ignore
    from tkinterdnd2 import TkinterDnD as _TkinterDnD  # type: ignore

    TkBase = _TkinterDnD.Tk
    DND_FILES = _DND_FILES
    DND_AVAILABLE = True
except Exception:
    TkBase = tk.Tk
    DND_FILES = None
    DND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Subtitle Conversion window
# ---------------------------------------------------------------------------


def _open_subtitle_converter(parent: tk.Misc) -> None:
    """Open the subtitle converter as a separate Toplevel window."""
    from stt.core.subtitle import SUPPORTED_FORMATS, convert_file, detect_format

    win = tk.Toplevel(parent)
    win.title("Subtitle Converter")
    win.minsize(640, 480)
    win.columnconfigure(0, weight=1)

    # --- Input file row ---
    file_frame = ttk.LabelFrame(win, text="Input file", padding=8)
    file_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
    file_frame.columnconfigure(1, weight=1)

    input_var = tk.StringVar()
    detected_var = tk.StringVar(value="Detected format: —")

    ttk.Label(file_frame, text="Path:").grid(row=0, column=0, sticky="w")
    input_entry = ttk.Entry(file_frame, textvariable=input_var)
    input_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))

    def browse_input() -> None:
        path = filedialog.askopenfilename(
            parent=win,
            title="Select subtitle file",
            filetypes=[
                ("Subtitle files", "*.srt *.vtt"),
                ("SRT", "*.srt"),
                ("VTT", "*.vtt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            input_var.set(path)
            _refresh_detected()

    ttk.Button(file_frame, text="Browse…", command=browse_input).grid(
        row=0, column=2, sticky="e"
    )
    ttk.Label(file_frame, textvariable=detected_var, foreground="gray").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(4, 0)
    )

    def _refresh_detected(*_: Any) -> None:
        path = input_var.get().strip()
        if path and Path(path).exists():
            fmt = detect_format(path)
            detected_var.set(f"Detected format: {fmt}")
        else:
            detected_var.set("Detected format: —")

    input_var.trace_add("write", _refresh_detected)

    # --- Conversion options ---
    opts_frame = ttk.LabelFrame(win, text="Conversion options", padding=8)
    opts_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

    format_var = tk.StringVar(value=SUPPORTED_FORMATS[0])
    ttk.Label(opts_frame, text="Convert to:").grid(row=0, column=0, sticky="w")
    fmt_box = ttk.Combobox(
        opts_frame,
        textvariable=format_var,
        values=list(SUPPORTED_FORMATS),
        state="readonly",
        width=8,
    )
    fmt_box.grid(row=0, column=1, sticky="w", padx=(6, 0))

    # --- Output path ---
    out_frame = ttk.LabelFrame(win, text="Output file (optional)", padding=8)
    out_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=4)
    out_frame.columnconfigure(1, weight=1)

    output_var = tk.StringVar()
    ttk.Label(out_frame, text="Path:").grid(row=0, column=0, sticky="w")
    ttk.Entry(out_frame, textvariable=output_var).grid(
        row=0, column=1, sticky="ew", padx=(6, 6)
    )

    def browse_output() -> None:
        fmt = format_var.get()
        path = filedialog.asksaveasfilename(
            parent=win,
            title="Save converted subtitle",
            defaultextension=f".{fmt}",
            filetypes=[
                (fmt.upper(), f"*.{fmt}"),
                ("All files", "*.*"),
            ],
        )
        if path:
            output_var.set(path)

    ttk.Button(out_frame, text="Browse…", command=browse_output).grid(
        row=0, column=2, sticky="e"
    )
    ttk.Label(out_frame, text="Leave blank to auto-name beside the input file", foreground="gray").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(4, 0)
    )

    # --- Status / action ---
    status_var = tk.StringVar(value="Ready")
    ttk.Label(win, textvariable=status_var).grid(
        row=3, column=0, sticky="w", padx=12, pady=(4, 0)
    )

    btn_frame = ttk.Frame(win)
    btn_frame.grid(row=4, column=0, sticky="ew", padx=12, pady=4)

    def do_convert() -> None:
        src = input_var.get().strip()
        dst = output_var.get().strip() or None
        fmt = format_var.get()

        if not src:
            messagebox.showwarning("No input", "Choose a subtitle file first.", parent=win)
            return
        if not Path(src).exists():
            messagebox.showerror("File not found", f"Input file not found:\n{src}", parent=win)
            return

        status_var.set("Converting…")
        win.update_idletasks()

        def worker() -> None:
            try:
                out_path = convert_file(src, fmt, dst)
                preview = out_path.read_text(encoding="utf-8")

                def finish() -> None:
                    status_var.set(f"Done → {out_path}")
                    preview_text.delete("1.0", "end")
                    preview_text.insert("1.0", preview)

                win.after(0, finish)
            except Exception as exc:
                err = str(exc)

                def show_err() -> None:
                    status_var.set("Error")
                    messagebox.showerror("Conversion failed", err, parent=win)

                win.after(0, show_err)

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(btn_frame, text="Convert", command=do_convert).grid(row=0, column=0)
    ttk.Button(btn_frame, text="Close", command=win.destroy).grid(
        row=0, column=1, padx=(8, 0)
    )

    # --- Preview ---
    preview_frame = ttk.LabelFrame(win, text="Preview", padding=8)
    preview_frame.grid(row=5, column=0, sticky="nsew", padx=12, pady=(4, 12))
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)
    win.rowconfigure(5, weight=1)

    preview_text = tk.Text(preview_frame, wrap="word", state="normal")
    preview_text.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    preview_text.configure(yscrollcommand=scroll.set)


# ---------------------------------------------------------------------------
# Home window
# ---------------------------------------------------------------------------


def _build_home(root: tk.Misc, defaults: dict[str, Any]) -> None:
    """Build the home screen widgets inside *root*."""
    root.columnconfigure(0, weight=1)

    ttk.Label(
        root,
        text="STT – faster-whisper",
        font=("TkDefaultFont", 16, "bold"),
        anchor="center",
    ).grid(row=0, column=0, pady=(24, 4), sticky="ew")

    ttk.Label(
        root,
        text="Select a feature to get started",
        anchor="center",
        foreground="gray",
    ).grid(row=1, column=0, sticky="ew", pady=(0, 24))

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=2, column=0)

    def open_transcribe() -> None:
        for widget in root.winfo_children():
            widget.destroy()
        _build_transcriber(root, defaults)

    def open_convert() -> None:
        _open_subtitle_converter(root)

    ttk.Button(
        btn_frame,
        text="🎙  Transcribe Audio",
        command=open_transcribe,
        width=26,
    ).grid(row=0, column=0, pady=6)

    ttk.Button(
        btn_frame,
        text="📄  Convert Subtitles",
        command=open_convert,
        width=26,
    ).grid(row=1, column=0, pady=6)


# ---------------------------------------------------------------------------
# Transcription UI (extracted from the original launch_gui)
# ---------------------------------------------------------------------------


def _build_transcriber(root: tk.Misc, defaults: dict[str, Any]) -> None:
    """Build the transcription UI inside *root*."""
    transcriber = Transcriber()

    audio_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Drop an audio file here or click Browse…")
    running_var = tk.BooleanVar(value=False)

    model_var = tk.StringVar(value=str(defaults.get("model", "large-v3")))
    device_var = tk.StringVar(value=str(defaults.get("device", "cuda")))
    compute_var = tk.StringVar(value=str(defaults.get("compute_type", "float16")))
    beam_var = tk.IntVar(value=int(defaults.get("beam_size", 5)))
    vad_var = tk.BooleanVar(value=bool(defaults.get("vad_filter", True)))
    condition_var = tk.BooleanVar(
        value=bool(defaults.get("condition_on_previous_text", True))
    )

    # We need a reference to the actual Tk root to schedule after() calls
    toplevel_root: tk.Tk = root.winfo_toplevel()  # type: ignore[assignment]

    def set_running(is_running: bool) -> None:
        running_var.set(is_running)
        state = "disabled" if is_running else "normal"
        browse_btn.configure(state=state)
        transcribe_btn.configure(state=state)
        save_btn.configure(
            state=state if transcript_text.get("1.0", "end").strip() else "disabled"
        )

    def set_status(text: str) -> None:
        status_var.set(text)

    def _safe_split_drop(data: str) -> list[str]:
        try:
            return list(toplevel_root.tk.splitlist(data))
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
                toplevel_root.after(0, lambda: (set_running(True), set_status("Preparing model…")))

                setup_windows_cuda_dlls(
                    cublas_bin=defaults.get("cublas_bin"),
                    cudnn_bin=defaults.get("cudnn_bin"),
                    check=False,
                    force_load=False,
                )

                config = TranscriptionConfig(
                    model_name=model_var.get().strip(),
                    device=device_var.get().strip(),
                    compute_type=compute_var.get().strip(),
                    beam_size=int(beam_var.get()),
                    vad_filter=bool(vad_var.get()),
                    condition_on_previous_text=bool(condition_var.get()),
                )

                result = transcriber.transcribe(audio_path, config)

                def finish_ui() -> None:
                    transcript_text.delete("1.0", "end")
                    transcript_text.insert("1.0", result.text)
                    set_status(
                        f"Done. language: {result.language} prob: {result.language_probability:.2f}"
                    )
                    set_running(False)
                    if save_after:
                        on_save()

                toplevel_root.after(0, finish_ui)
            except Exception as e:
                err_msg = str(e)

                def show_error() -> None:
                    set_running(False)
                    set_status("Failed.")
                    messagebox.showerror("Transcription failed", err_msg)

                toplevel_root.after(0, show_error)

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

        suggested = (
            Path(audio_var.get()).with_suffix(".txt").name
            if audio_var.get().strip()
            else "conversation.txt"
        )
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

    drop_hint = (
        "(Drag & drop enabled)" if DND_AVAILABLE else "(Install tkinterdnd2 for drag & drop)"
    )
    drop_frame = ttk.LabelFrame(top, text=f"Drop zone {drop_hint}", padding=12)
    drop_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    drop_label = ttk.Label(drop_frame, textvariable=status_var, anchor="center")
    drop_label.grid(row=0, column=0, sticky="ew")
    drop_frame.columnconfigure(0, weight=1)

    if DND_AVAILABLE:
        try:
            drop_frame.drop_target_register(DND_FILES)
            drop_frame.dnd_bind("<<Drop>>", on_drop)
            drop_label.drop_target_register(DND_FILES)
            drop_label.dnd_bind("<<Drop>>", on_drop)
        except Exception:
            pass

    opts = ttk.Frame(root, padding=(12, 0, 12, 12))
    opts.grid(row=1, column=0, sticky="ew")
    for i in range(8):
        opts.columnconfigure(i, weight=0)
    opts.columnconfigure(7, weight=1)

    ttk.Label(opts, text="Model").grid(row=0, column=0, sticky="w")
    model_box = ttk.Combobox(
        opts,
        textvariable=model_var,
        width=18,
        values=["tiny", "base", "small", "medium", "large-v3"],
    )
    model_box.grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Device").grid(row=0, column=2, sticky="w")
    device_box = ttk.Combobox(
        opts, textvariable=device_var, width=10, values=["auto", "cuda", "cpu"]
    )
    device_box.grid(row=0, column=3, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Compute").grid(row=0, column=4, sticky="w")
    compute_box = ttk.Combobox(
        opts,
        textvariable=compute_var,
        width=12,
        values=["float16", "float32", "int8"],
    )
    compute_box.grid(row=0, column=5, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Beam").grid(row=0, column=6, sticky="w")
    beam_spin = ttk.Spinbox(opts, from_=1, to=20, textvariable=beam_var, width=5)
    beam_spin.grid(row=0, column=7, sticky="w")

    vad_check = ttk.Checkbutton(opts, text="VAD filter", variable=vad_var)
    vad_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    condition_check = ttk.Checkbutton(
        opts,
        text="Condition on previous text",
        variable=condition_var,
    )
    condition_check.grid(row=1, column=2, columnspan=3, sticky="w", pady=(8, 0))

    actions = ttk.Frame(root, padding=(12, 0, 12, 12))
    actions.grid(row=2, column=0, sticky="ew")
    transcribe_btn = ttk.Button(actions, text="Transcribe", command=on_transcribe)
    transcribe_btn.grid(row=0, column=0, sticky="w")
    ttk.Button(actions, text="Transcribe & Save…", command=on_transcribe_and_save).grid(
        row=0, column=1, sticky="w", padx=(8, 0)
    )
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
            save_btn.configure(
                state="normal" if transcript_text.get("1.0", "end").strip() else "disabled"
            )
        toplevel_root.after(300, poll_save_state)

    poll_save_state()


# ---------------------------------------------------------------------------
# Public launch function
# ---------------------------------------------------------------------------


def launch_gui(
    defaults: dict[str, Any] | None = None,
) -> int:
    """Launch the Tkinter GUI.

    Args:
        defaults: Default configuration values

    Returns:
        Exit code
    """
    if defaults is None:
        defaults = {}

    root = TkBase()
    root.title("STT (faster-whisper)")
    root.minsize(720, 520)

    _build_home(root, defaults)

    root.mainloop()
    return 0

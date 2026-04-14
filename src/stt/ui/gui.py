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

    transcriber = Transcriber()

    audio_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Drop an audio file here or click Browse…")
    running_var = tk.BooleanVar(value=False)

    model_var = tk.StringVar(value=str(defaults.get("model", "large-v3")))
    device_var = tk.StringVar(value=str(defaults.get("device", "cuda")))
    compute_var = tk.StringVar(value=str(defaults.get("compute_type", "float16")))
    beam_var = tk.IntVar(value=int(defaults.get("beam_size", 5)))
    vad_var = tk.BooleanVar(value=bool(defaults.get("vad_filter", True)))

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

                root.after(0, finish_ui)
            except Exception as e:
                err_msg = str(e)

                def show_error() -> None:
                    set_running(False)
                    set_status("Failed.")
                    messagebox.showerror("Transcription failed", err_msg)

                root.after(0, show_error)

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
        root.after(300, poll_save_state)

    poll_save_state()

    root.mainloop()
    return 0

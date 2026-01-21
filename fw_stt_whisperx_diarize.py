import argparse
import os
import threading
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe + diarize with WhisperX (speaker-labeled, Narrator-annotated)."
    )
    p.add_argument("audio", nargs="?", help="Path to the audio file (omit to launch GUI)")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output transcript path (default: <audio>.txt)",
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
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")

    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token for pyannote diarization (or set HF_TOKEN env var)",
    )
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)

    p.add_argument(
        "--no-align",
        action="store_true",
        help="Skip word-level alignment step",
    )
    p.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip diarization (speaker labeling)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI (ignores the audio argument)",
    )

    p.add_argument(
        "--narrator-label",
        default="Narrator",
        help='Speaker label prefix (default: "Narrator")',
    )
    return p


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _torch_allowlist_omegaconf_for_weights_only() -> None:
    """Make PyTorch 'weights_only' checkpoint loading accept OmegaConf containers.

    PyTorch 2.6+ defaults torch.load(..., weights_only=True). Some third-party
    checkpoints include OmegaConf objects (e.g., ListConfig/DictConfig) which are
    blocked by default, causing a WeightsUnpickler error.

    We keep this narrowly scoped to OmegaConf container classes (common + low risk).
    """

    try:
        import torch
        from omegaconf import DictConfig, ListConfig

        try:
            from torch.serialization import add_safe_globals
        except Exception:
            add_safe_globals = None  # type: ignore

        if add_safe_globals is not None:
            add_safe_globals([DictConfig, ListConfig])
    except Exception:
        # Best-effort only; if unavailable, downstream may still work.
        return


def _speaker_to_narrator(speaker: str, mapping: dict[str, str], prefix: str) -> str:
    if speaker in mapping:
        return mapping[speaker]
    idx = len(mapping) + 1
    mapping[speaker] = f"{prefix} {idx}"
    return mapping[speaker]


def _format_segments_as_text(segments: list[dict[str, Any]], narrator_prefix: str) -> str:
    speaker_map: dict[str, str] = {}
    lines: list[str] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        speaker = (seg.get("speaker") or "").strip()

        if speaker:
            narrator = _speaker_to_narrator(speaker, speaker_map, narrator_prefix)
            who = narrator
        else:
            who = narrator_prefix

        if not text:
            continue
        lines.append(f"[{start:8.2f} -> {end:8.2f}] {who}: {text}")

    return "\n".join(lines) + ("\n" if lines else "")


def _transcribe_diarize_text(
    *,
    audio_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int,
    hf_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    do_align: bool,
    do_diarize: bool,
    narrator_label: str,
) -> tuple[str, str]:
    try:
        import whisperx
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: whisperx. Install with: pip install -U whisperx"
        ) from e

    _torch_allowlist_omegaconf_for_weights_only()
    device = _resolve_device(device)

    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)

    info = f"language: {result.get('language', '?')}"

    if do_align and result.get("language"):
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception:
            # Alignment is nice-to-have; diarization/segment output still works without it.
            pass

    if do_diarize:
        if not hf_token:
            raise RuntimeError(
                "Diarization requires a Hugging Face token. Provide --hf-token or set HF_TOKEN env var."
            )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = result.get("segments") or []
    return _format_segments_as_text(segments, narrator_label), info


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
    root.title("STT (WhisperX + Diarization)")
    root.minsize(760, 560)

    audio_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Drop an audio file here or click Browse…")
    running_var = tk.BooleanVar(value=False)

    model_var = tk.StringVar(value=str(getattr(defaults, "model", "large-v3")))
    device_var = tk.StringVar(value=str(getattr(defaults, "device", "cuda")))
    compute_var = tk.StringVar(value=str(getattr(defaults, "compute_type", "float16")))
    batch_var = tk.IntVar(value=int(getattr(defaults, "batch_size", 16)))
    narrator_var = tk.StringVar(value=str(getattr(defaults, "narrator_label", "Narrator")))

    diarize_var = tk.BooleanVar(value=not bool(getattr(defaults, "no_diarize", False)))
    align_var = tk.BooleanVar(value=not bool(getattr(defaults, "no_align", False)))

    hf_token_var = tk.StringVar(value=str(getattr(defaults, "hf_token", "") or ""))
    min_spk_var = tk.StringVar(value="" if getattr(defaults, "min_speakers", None) is None else str(defaults.min_speakers))
    max_spk_var = tk.StringVar(value="" if getattr(defaults, "max_speakers", None) is None else str(defaults.max_speakers))

    def set_running(is_running: bool) -> None:
        running_var.set(is_running)
        state = "disabled" if is_running else "normal"
        browse_btn.configure(state=state)
        transcribe_btn.configure(state=state)
        transcribe_save_btn.configure(state=state)
        save_btn.configure(state=state if transcript_text.get("1.0", "end").strip() else "disabled")

    def set_status(text: str) -> None:
        status_var.set(text)

    def _safe_split_drop(data: str) -> list[str]:
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

    def _parse_optional_int(s: str) -> int | None:
        s = (s or "").strip()
        if not s:
            return None
        return int(s)

    def do_transcribe(save_after: bool) -> None:
        audio_path = audio_var.get().strip()
        if not audio_path:
            messagebox.showwarning("No input", "Choose an audio file first.")
            return

        def worker() -> None:
            try:
                root.after(0, lambda: (set_running(True), set_status("Transcribing (WhisperX)…")))
                text, info = _transcribe_diarize_text(
                    audio_path=audio_path,
                    model_name=model_var.get().strip(),
                    device=device_var.get().strip(),
                    compute_type=compute_var.get().strip(),
                    batch_size=int(batch_var.get()),
                    hf_token=hf_token_var.get().strip() or None,
                    min_speakers=_parse_optional_int(min_spk_var.get()),
                    max_speakers=_parse_optional_int(max_spk_var.get()),
                    do_align=bool(align_var.get()),
                    do_diarize=bool(diarize_var.get()),
                    narrator_label=narrator_var.get().strip() or "Narrator",
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
                        messagebox.showerror("Failed", err_msg),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_transcribe() -> None:
        do_transcribe(save_after=False)

    def on_transcribe_and_save() -> None:
        do_transcribe(save_after=True)

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
            pass

    opts = ttk.LabelFrame(root, text="Options", padding=12)
    opts.grid(row=1, column=0, sticky="ew", padx=12)
    for i in range(8):
        opts.columnconfigure(i, weight=0)
    opts.columnconfigure(7, weight=1)

    ttk.Label(opts, text="Model").grid(row=0, column=0, sticky="w")
    ttk.Entry(opts, textvariable=model_var, width=18).grid(row=0, column=1, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Device").grid(row=0, column=2, sticky="w")
    ttk.Combobox(opts, textvariable=device_var, width=10, values=["auto", "cuda", "cpu"]).grid(
        row=0, column=3, sticky="w", padx=(6, 18)
    )

    ttk.Label(opts, text="Compute").grid(row=0, column=4, sticky="w")
    ttk.Entry(opts, textvariable=compute_var, width=12).grid(row=0, column=5, sticky="w", padx=(6, 18))

    ttk.Label(opts, text="Batch").grid(row=0, column=6, sticky="w")
    ttk.Spinbox(opts, from_=1, to=64, textvariable=batch_var, width=5).grid(row=0, column=7, sticky="w")

    ttk.Checkbutton(opts, text="Align", variable=align_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Checkbutton(opts, text="Diarize", variable=diarize_var).grid(row=1, column=1, sticky="w", pady=(8, 0))

    ttk.Label(opts, text="Narrator label").grid(row=1, column=2, sticky="w", pady=(8, 0))
    ttk.Entry(opts, textvariable=narrator_var, width=14).grid(row=1, column=3, sticky="w", padx=(6, 18), pady=(8, 0))

    ttk.Label(opts, text="HF token").grid(row=2, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(opts, textvariable=hf_token_var, width=54, show="•").grid(
        row=2, column=1, columnspan=4, sticky="ew", padx=(6, 18), pady=(8, 0)
    )

    ttk.Label(opts, text="Min spk").grid(row=2, column=5, sticky="w", pady=(8, 0))
    ttk.Entry(opts, textvariable=min_spk_var, width=6).grid(row=2, column=6, sticky="w", padx=(6, 12), pady=(8, 0))

    ttk.Label(opts, text="Max spk").grid(row=2, column=7, sticky="w", pady=(8, 0))
    ttk.Entry(opts, textvariable=max_spk_var, width=6).grid(row=2, column=7, sticky="e", pady=(8, 0))

    actions = ttk.Frame(root, padding=(12, 12, 12, 12))
    actions.grid(row=2, column=0, sticky="ew")
    transcribe_btn = ttk.Button(actions, text="Transcribe", command=on_transcribe)
    transcribe_btn.grid(row=0, column=0, sticky="w")
    transcribe_save_btn = ttk.Button(actions, text="Transcribe & Save…", command=on_transcribe_and_save)
    transcribe_save_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
    save_btn = ttk.Button(actions, text="Save…", command=on_save, state="disabled")
    save_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))

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

    if args.gui or not args.audio:
        return _launch_gui(args)

    audio_path = args.audio
    output_path = args.output or str(Path(audio_path).with_suffix(".txt"))

    text, info = _transcribe_diarize_text(
        audio_path=audio_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        do_align=not args.no_align,
        do_diarize=not args.no_diarize,
        narrator_label=args.narrator_label,
    )

    print(info)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

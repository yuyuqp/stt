"""Subtitle format detection, parsing, and conversion utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import srt as srt_lib


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SubtitleEntry:
    """A single subtitle cue with timing and text."""

    index: int
    start: timedelta
    end: timedelta
    text: str


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_VTT_SIGNATURE = "WEBVTT"
_SRT_BLOCK_RE = re.compile(
    r"^\d+\s*\n\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}",
    re.MULTILINE,
)


def detect_format(path: str | Path) -> str:
    """Detect the subtitle format of a file.

    Checks the file extension first, then inspects content for confirmation.

    Args:
        path: Path to the subtitle file.

    Returns:
        ``"vtt"``, ``"srt"``, or ``"unknown"``.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ext if ext in ("vtt", "srt") else "unknown"

    stripped = content.lstrip()
    if stripped.startswith(_VTT_SIGNATURE):
        return "vtt"
    if _SRT_BLOCK_RE.search(content):
        return "srt"
    if ext in ("vtt", "srt"):
        return ext
    return "unknown"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_srt(content: str) -> list[SubtitleEntry]:
    """Parse SRT content into a list of :class:`SubtitleEntry` objects.

    Args:
        content: Raw SRT text.

    Returns:
        List of :class:`SubtitleEntry` objects in order.
    """
    subs = list(srt_lib.parse(content))
    return [
        SubtitleEntry(
            index=s.index,
            start=s.start,
            end=s.end,
            text=s.content.strip(),
        )
        for s in subs
    ]


# VTT timestamp pattern: HH:MM:SS.mmm  or  MM:SS.mmm
_VTT_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})"
)
_VTT_CUE_ARROW = re.compile(r"\s+-->\s+")


def _parse_vtt_timestamp(ts: str) -> timedelta:
    """Convert a VTT timestamp string to a :class:`timedelta`."""
    m = _VTT_TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid VTT timestamp: {ts!r}")
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    millis = int(m.group(4))
    return timedelta(
        hours=hours, minutes=minutes, seconds=seconds, milliseconds=millis
    )


def parse_vtt(content: str) -> list[SubtitleEntry]:
    """Parse WebVTT content into a list of :class:`SubtitleEntry` objects.

    Args:
        content: Raw WebVTT text.

    Returns:
        List of :class:`SubtitleEntry` objects in order.
    """
    entries: list[SubtitleEntry] = []
    index = 1

    # Split on blank lines
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Skip the WEBVTT header block and NOTE/STYLE/REGION blocks
        first = lines[0].strip()
        if first.startswith("WEBVTT"):
            continue
        if first.startswith(("NOTE", "STYLE", "REGION")):
            continue

        # Detect the timing line (may be preceded by an optional cue id)
        timing_line_idx = None
        for i, line in enumerate(lines):
            if _VTT_CUE_ARROW.search(line):
                timing_line_idx = i
                break

        if timing_line_idx is None:
            continue

        timing_line = lines[timing_line_idx]
        # Strip position/align/size/line settings after the timestamps
        timing_parts = _VTT_CUE_ARROW.split(timing_line, maxsplit=1)
        if len(timing_parts) != 2:
            continue

        start_str = timing_parts[0].strip()
        # The end timestamp may be followed by cue settings (space-separated)
        end_str = timing_parts[1].strip().split()[0]

        try:
            start = _parse_vtt_timestamp(start_str)
            end = _parse_vtt_timestamp(end_str)
        except ValueError:
            continue

        # Text is everything after the timing line
        text_lines = lines[timing_line_idx + 1 :]
        # Remove VTT tags like <c>, <ruby>, <rt>, positional tags, etc.
        text = "\n".join(text_lines).strip()
        text = re.sub(r"<[^>]+>", "", text)

        if not text:
            continue

        entries.append(SubtitleEntry(index=index, start=start, end=end, text=text))
        index += 1

    return entries


def parse_file(path: str | Path) -> list[SubtitleEntry]:
    """Detect and parse a subtitle file.

    Args:
        path: Path to a ``.srt`` or ``.vtt`` file.

    Returns:
        List of :class:`SubtitleEntry` objects.

    Raises:
        ValueError: If the format cannot be detected.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Subtitle file not found: {p}")

    content = p.read_text(encoding="utf-8", errors="replace")
    fmt = detect_format(p)

    if fmt == "srt":
        return parse_srt(content)
    if fmt == "vtt":
        return parse_vtt(content)
    raise ValueError(
        f"Unsupported or unrecognised subtitle format for file: {p}\n"
        "Supported formats: .srt, .vtt"
    )


# ---------------------------------------------------------------------------
# Formatters / serialisers
# ---------------------------------------------------------------------------


def _td_to_srt(td: timedelta) -> str:
    """Format a timedelta as an SRT timestamp ``HH:MM:SS,mmm``."""
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _td_to_vtt(td: timedelta) -> str:
    """Format a timedelta as a VTT timestamp ``HH:MM:SS.mmm``."""
    return _td_to_srt(td).replace(",", ".")


def to_srt(entries: Sequence[SubtitleEntry]) -> str:
    """Serialise a sequence of subtitle entries to SRT format.

    Args:
        entries: Subtitle entries to serialise.

    Returns:
        SRT-formatted string.
    """
    blocks: list[str] = []
    for i, entry in enumerate(entries, start=1):
        ts = f"{_td_to_srt(entry.start)} --> {_td_to_srt(entry.end)}"
        blocks.append(f"{i}\n{ts}\n{entry.text}\n")
    return "\n".join(blocks)


def to_vtt(entries: Sequence[SubtitleEntry]) -> str:
    """Serialise a sequence of subtitle entries to WebVTT format.

    Args:
        entries: Subtitle entries to serialise.

    Returns:
        WebVTT-formatted string.
    """
    lines = ["WEBVTT", ""]
    for entry in entries:
        ts = f"{_td_to_vtt(entry.start)} --> {_td_to_vtt(entry.end)}"
        lines.append(ts)
        lines.append(entry.text)
        lines.append("")
    return "\n".join(lines)


def to_plain_text(entries: Sequence[SubtitleEntry]) -> str:
    """Extract plain text from subtitle entries (no timestamps).

    Args:
        entries: Subtitle entries to serialise.

    Returns:
        Plain text with one paragraph per cue.
    """
    return "\n".join(entry.text for entry in entries)


# ---------------------------------------------------------------------------
# High-level conversion helper
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = ("srt", "vtt", "txt")


def convert_file(
    input_path: str | Path,
    output_format: str,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a subtitle file to the requested format and write the result.

    Args:
        input_path: Source subtitle file (``.srt`` or ``.vtt``).
        output_format: Target format – one of ``"srt"``, ``"vtt"``, or
            ``"txt"`` (plain text without timestamps).
        output_path: Destination path.  When ``None``, the output is placed
            next to the input file with the appropriate extension.

    Returns:
        The path of the written output file.

    Raises:
        ValueError: If ``output_format`` is not supported.
        FileNotFoundError: If the input file does not exist.
    """
    fmt = output_format.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format!r}. "
            f"Choose from: {', '.join(SUPPORTED_FORMATS)}"
        )

    input_path = Path(input_path)
    entries = parse_file(input_path)

    if fmt == "srt":
        text = to_srt(entries)
    elif fmt == "vtt":
        text = to_vtt(entries)
    else:
        text = to_plain_text(entries)

    if output_path is None:
        output_path = input_path.with_suffix(f".{fmt}")
    output_path = Path(output_path)

    output_path.write_text(text, encoding="utf-8")
    return output_path

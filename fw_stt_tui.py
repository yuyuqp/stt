"""Legacy TUI entry point - wrapper around refactored STT application.

This file maintains backward compatibility with the original fw_stt_tui.py interface.
New development should use stt.ui.tui module.
"""

import sys
from pathlib import Path

# Add src to path so we can import the refactored modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stt.ui.tui import launch_tui


if __name__ == "__main__":
    raise SystemExit(launch_tui())

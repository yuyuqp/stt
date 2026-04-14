"""Legacy entry point - wrapper around refactored STT application.

This file maintains backward compatibility with the original fw_stt.py interface.
New development should use the refactored modules in src/stt/.
"""

import sys
from pathlib import Path

# Add src to path so we can import the refactored modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stt.main import main

if __name__ == "__main__":
    raise SystemExit(main())



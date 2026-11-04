"""Workspace-safe entry point for local image and webcam inference."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    """Import the runtime after making the src layout discoverable."""
    runtime_cli = importlib.import_module("parcel_sorter.runtime.cli")
    runtime_cli.main()


if __name__ == "__main__":
    main()

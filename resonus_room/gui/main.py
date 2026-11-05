"""Standalone GUI launcher of Resonus Room."""

from __future__ import annotations

import sys

from resonus_core.gui.host import run_standalone

from resonus_room import __version__
from resonus_room.gui.plugin import Plugin

APP_ID = "cz.resonus.Room"  # docs/DECISIONS.md D-26


def main() -> int:
    """Start the standalone application."""
    return run_standalone(Plugin, APP_ID, __version__, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

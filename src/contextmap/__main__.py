"""Run the command-line interface with ``python -m contextmap``."""

from __future__ import annotations

import sys

from contextmap.runtime.cli import main

if __name__ == "__main__":
    sys.exit(main())

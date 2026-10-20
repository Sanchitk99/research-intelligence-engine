"""Convenience script to run corpus embedding and Qdrant indexing."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.indexing.cli import main

if __name__ == "__main__":
    sys.exit(main())


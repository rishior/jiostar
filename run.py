"""
Single entry point for all operations.

Usage:
    python run.py download        Download the dataset
    python run.py validate        Validate character offsets
    python run.py query "..."     Query the contracts
    python run.py evaluate        Run evaluation
    python run.py improve         Run self-improvement loop
    python run.py build-index     Build the search index
    python run.py webapp          Start the web application
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import cli


if __name__ == "__main__":
    cli()


"""Pytest configuration and shared fixtures.

All test fixtures in this project are synthetic and clearly labelled as such.
Synthetic values are used only to verify software behaviour, never as ocean
science results.
"""
import sys
from pathlib import Path

# Ensure the project root is on the Python path so imports work without install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

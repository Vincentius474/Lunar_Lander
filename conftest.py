"""
    To Ensure src/ folder is on sys.path so tests can import `lunar_lander` without install.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
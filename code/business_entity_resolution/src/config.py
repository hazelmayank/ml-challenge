"""Paths shared by all pipeline steps. Override the data folder with BER_DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("BER_DATA", ROOT / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUTPUT = Path(os.environ.get("BER_OUTPUT", ROOT / "output"))
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

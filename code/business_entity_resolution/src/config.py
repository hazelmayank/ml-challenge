"""Paths shared by all pipeline steps. Override the data folder with BER_DATA."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("BER_DATA", ROOT / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUTPUT = Path(os.environ.get("BER_OUTPUT", ROOT / "output"))
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def parts_dir(folder: Path, fingerprint: str) -> Path:
    """Resumable parts folder that is wiped when its input changed (stale parts are never reused)."""
    stamp = folder / "input_fingerprint.txt"
    if folder.exists() and (not stamp.exists() or stamp.read_text() != fingerprint):
        print(f"input changed: clearing {folder.name}", flush=True)
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fingerprint)
    return folder

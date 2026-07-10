from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
DEFAULT_GEE_KEY_PATH = ROOT / "gee-key.json"


def display_path(path: Path) -> str:
    """Return a readable path relative to the repository when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


"""Loads the frozen World_Monitor signal snapshot once at import."""
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "data" / "signals_20260530.json"

_raw = json.loads(_PATH.read_text(encoding="utf-8"))

SNAPSHOT: dict[str, dict] = _raw["signals"]
SNAPSHOT_AS_OF: str = _raw["as_of"]

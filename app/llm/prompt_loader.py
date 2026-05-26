from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the system-prompt text for an agent (e.g. load_prompt('macro')).
    Raises FileNotFoundError for an unknown name."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No prompt named {name!r} at {path}")
    return path.read_text(encoding="utf-8").strip()

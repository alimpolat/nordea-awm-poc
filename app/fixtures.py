"""Shared fixture loader for the POC agents.

All file I/O is cached at process level with lru_cache so repeated calls
within a single run are free. BeautifulSoup strips HTML to collapsed plain text.

Cache-safety note
-----------------
load_portfolio and load_meeting_notes return MUTABLE objects (dict / list).
To avoid cache-poisoning across agents that run concurrently under asyncio.gather,
the lru_cache is applied to an inner function that caches the raw text / tuple;
the public functions return a FRESH object on every call.
"""
import json
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

_DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _portfolio_raw(client_id: str) -> str:
    """Cache the raw JSON text (immutable str) so disk I/O happens only once."""
    return (_DATA / f"{client_id}_portfolio.json").read_text(encoding="utf-8")


def load_portfolio(client_id: str = "bergstrom") -> dict:
    """Return the portfolio JSON for client_id as a FRESH plain dict each call."""
    return json.loads(_portfolio_raw(client_id))


# ---------------------------------------------------------------------------
# IPS text — str is immutable, lru_cache is safe
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_ips_text(client_id: str = "bergstrom") -> str:
    """Return the IPS HTML converted to collapsed plain text."""
    html = (_DATA / "corpus" / f"{client_id}_ips.html").read_text(encoding="utf-8")
    return _html_text(html)


# ---------------------------------------------------------------------------
# Meeting notes
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _meeting_notes_tuple(client_id: str) -> tuple[str, ...]:
    """Cache the parsed notes as an immutable tuple so disk I/O happens only once."""
    pattern = f"{client_id}_meeting_notes_*.html"
    paths = sorted((_DATA / "corpus").glob(pattern))
    return tuple(_html_text(p.read_text(encoding="utf-8")) for p in paths)


def load_meeting_notes(client_id: str = "bergstrom") -> list[str]:
    """Return all meeting notes for client_id, sorted by filename (chronological), as plain text.

    Each element of the returned list corresponds to one note file.
    Returns a FRESH list each call to prevent cache-poisoning.
    """
    return list(_meeting_notes_tuple(client_id))


def _html_text(html: str) -> str:
    """Strip HTML tags and collapse all whitespace to single spaces."""
    return " ".join(BeautifulSoup(html, "html.parser").get_text(separator=" ").split())

from __future__ import annotations

import re
from typing import Iterable, Set

_CAMPAIGN_SUFFIX_PATTERN = re.compile(r"(_S\d+|\s*S\d+|\s*Show\s*\d+)$", re.IGNORECASE)
_CAMPAIGN_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9]+")
_CITY_TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}")


def normalize_show_id(raw_name: str | None) -> str:
    """Return a canonical identifier extracted from the campaign or show name.

    The function keeps alphanumeric characters, removes common suffixes such as
    ``_S2`` or ``Show 3`` and uppercases the result. If *raw_name* is empty the
    function returns an empty string.
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""

    cleaned = raw_name.strip()
    cleaned = _CAMPAIGN_SUFFIX_PATTERN.sub("", cleaned)
    cleaned = _CAMPAIGN_SANITIZE_PATTERN.sub("", cleaned)
    return cleaned.upper()


def extract_city_tokens(text: str | None) -> Set[str]:
    """Extract potential city/location tokens used to match campaigns.

    Tokens are returned in lowercase to simplify comparisons.
    """
    if not text or not isinstance(text, str):
        return set()

    return {token.lower() for token in _CITY_TOKEN_PATTERN.findall(text)}


def contains_any(text: str | None, tokens: Iterable[str]) -> bool:
    """Check whether *text* contains any of the provided *tokens* (case-insensitive)."""
    if not text or not isinstance(text, str):
        return False

    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens if token)

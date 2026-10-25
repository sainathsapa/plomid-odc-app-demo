"""Secret hygiene (Phase 11) — credentials never reach logs, errors, or dumps."""

from __future__ import annotations

import re
from typing import Any

_PASSWORD_RE = re.compile(r"(://[^:/@\s]+:)([^@\s]+)(@)")


def scrub_dsn(text: str) -> str:
    """Replace DSN passwords with ***. Idempotent, safe on plain text."""
    return _PASSWORD_RE.sub(r"\1***\3", text)


def scrub_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy a mapping with secret-ish keys redacted (dsn, password, token)."""
    redacted = dict(payload)
    for key in list(redacted):
        lowered = str(key).lower()
        if any(token in lowered for token in ("dsn", "password", "passwd", "token", "secret")):
            redacted[key] = "***"
    return redacted

"""Compiled regex patterns and scrubbing utilities for review text.

The patterns target the highest-signal PII surfaces present in mobile review
bodies (emails, phones, @-handles, long opaque identifiers). They run before
any text leaves the service, so the redaction must be applied uniformly to
both ``title`` and ``body``.

Regex is intentionally conservative — false positives (a redacted version
number) are cheaper than leaking a real phone number to a downstream LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from re import Pattern

# Replacement tokens. Kept stable so downstream prompts can reference them
# (e.g. "ignore [EMAIL] tokens").
EMAIL_TOKEN = "[EMAIL]"
PHONE_TOKEN = "[PHONE]"
USER_TOKEN = "[USER]"
ID_TOKEN = "[ID]"


# Email: standard local@domain.tld shape. Allows +tags and sub-domains.
EMAIL_RE: Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)

# Phone: optional country code, optional area-code grouping, 7+ digits total.
# Anchored with non-word lookarounds to avoid swallowing app-version numbers
# like "1.2.3456789" — but we still strip standalone long digit runs via the
# numeric-ID rule below.
PHONE_RE: Pattern[str] = re.compile(
    r"""
    (?<!\w)                         # left boundary
    (?:\+\d{1,3}[\s.\-]?)?          # optional country code
    (?:\(\d{2,4}\)|\d{2,4})         # area code (parenthesised or bare)
    (?:[\s.\-]?\d{2,4}){2,4}        # 2-4 more digit groups
    (?!\w)                          # right boundary
    """,
    re.VERBOSE,
)

# @handles: twitter/instagram style. 2-30 word chars, must be preceded by a
# non-word char so we do not eat email locals.
USERNAME_RE: Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{2,30}\b",
)

# UUIDs (canonical 8-4-4-4-12).
UUID_RE: Pattern[str] = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
)

# Long hex / opaque id tokens (>= 24 chars of hex, e.g. mongo ObjectId-likes,
# session tokens). Stricter than a generic alphanumeric rule to keep recall
# of normal English text.
HEX_ID_RE: Pattern[str] = re.compile(r"\b[0-9a-fA-F]{24,}\b")

# Standalone long digit runs (order numbers, ticket numbers, account ids).
# Requires >= 7 digits which keeps prices / versions intact.
NUMERIC_ID_RE: Pattern[str] = re.compile(r"(?<!\w)\d{7,}(?!\w)")

# Multi-whitespace collapse (spaces, tabs, NBSPs but NOT newlines so we keep
# paragraph structure intact for downstream summarisation).
WS_RE: Pattern[str] = re.compile(r"[^\S\n]+")

# Repeated blank lines collapse to a single newline.
BLANK_LINES_RE: Pattern[str] = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class ScrubResult:
    """Result of scrubbing one piece of text.

    ``redactions`` is a per-category count and is useful both for tests and
    for emitting structured log lines without leaking the original content.
    """

    text: str
    redactions: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.redactions.values())


def _replace_count(pattern: Pattern[str], replacement: str, text: str) -> tuple[str, int]:
    """``re.subn`` wrapper that returns ``(new_text, count)``."""
    new_text, n = pattern.subn(replacement, text)
    return new_text, n


def scrub_pii(text: str) -> ScrubResult:
    """Remove emails, phones, @-handles and opaque IDs from ``text``.

    Order matters: emails are stripped first so the local-part is not later
    misread as a phone number; usernames are stripped before any remaining
    @-prefixed fragments can confuse downstream parsing. UUIDs / hex IDs run
    before phone so the phone regex cannot eat trailing digit groups of a
    UUID; numeric-ID runs last because phone is the stricter shape.
    """
    counts: dict[str, int] = {"email": 0, "phone": 0, "username": 0, "id": 0}

    text, counts["email"] = _replace_count(EMAIL_RE, EMAIL_TOKEN, text)
    text, counts["username"] = _replace_count(USERNAME_RE, USER_TOKEN, text)

    # Run UUID + hex ID scrubs before phone so the phone regex does not
    # cannibalise digit groups inside a UUID. Numeric IDs come last because
    # phone is the stricter pattern (with separators / parens / country code).
    text, n_uuid = _replace_count(UUID_RE, ID_TOKEN, text)
    text, n_hex = _replace_count(HEX_ID_RE, ID_TOKEN, text)

    text, counts["phone"] = _replace_count(PHONE_RE, PHONE_TOKEN, text)

    text, n_num = _replace_count(NUMERIC_ID_RE, ID_TOKEN, text)
    counts["id"] = n_uuid + n_hex + n_num

    return ScrubResult(text=text, redactions=counts)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of inline whitespace, trim, and squash blank lines.

    Unicode whitespace (NBSP, zero-width space, etc.) is first folded into
    ASCII space via NFKC so the regex catches it.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("​", "").replace("﻿", "")
    folded = WS_RE.sub(" ", folded)
    # Strip the space we may have left adjacent to a newline.
    folded = re.sub(r" *\n *", "\n", folded)
    folded = BLANK_LINES_RE.sub("\n", folded)
    return folded.strip()

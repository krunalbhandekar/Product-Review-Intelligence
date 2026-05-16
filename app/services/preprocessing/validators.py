"""Validation helpers for the preprocessing pipeline.

Kept separate from ``service.py`` so callers (filters, dashboards, ad-hoc
notebooks) can reuse the predicates without pulling in the orchestration
layer.
"""

from __future__ import annotations

import hashlib

from app.domain.models import Review

# A review whose stripped body is shorter than this is treated as empty —
# "ok", "👍", "...". They carry no signal for downstream analysis but show
# up frequently in app-store feeds.
MIN_BODY_CHARS = 3


def is_empty_text(text: str | None) -> bool:
    """Return True if ``text`` is missing or contains no meaningful content."""
    if text is None:
        return True
    return len(text.strip()) < MIN_BODY_CHARS


def is_empty_review(review: Review) -> bool:
    """A review is empty when its body has no meaningful content.

    Title is ignored on purpose: a title-only review still lacks the actual
    user feedback we want to analyse.
    """
    return is_empty_text(review.body)


def review_identity(review: Review) -> tuple[str, str]:
    """Stable identity tuple for source-level deduplication.

    Two payloads sharing ``(source, review_id)`` are the same upstream entry,
    even if their text differs after re-fetching.
    """
    return (review.source.value, review.review_id)


def content_fingerprint(review: Review) -> str:
    """Hash of the normalised body + rating, for near-duplicate detection.

    Used when the same review is re-posted across countries with a different
    ``review_id`` (common for bot/spam reviews). Title is folded in so two
    bodies with the same text but different titles do not collide.
    """
    payload = "\n".join(
        [
            (review.title or "").strip().lower(),
            review.body.strip().lower(),
            str(review.rating),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Preprocessing service: PII redaction, normalisation, deduplication.

The service is the single entry point the rest of the pipeline uses to turn
raw ``Review`` objects from the importers into "clean" reviews ready to be
sent to Groq (or any other downstream LLM). It guarantees:

  * No email, phone, @-handle or opaque ID survives in ``title`` / ``body``.
  * Whitespace is normalised and the text is trimmed.
  * Empty / placeholder reviews are dropped.
  * Duplicate reviews (by upstream id and by content) are dropped on a
    best-effort basis within a single stream.

Both sync (``preprocess``) and async (``preprocess_stream``) entry points
are provided so the service composes with the existing async importers
without forcing every caller to be async.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from app.core.logging import get_logger
from app.domain.models import Review
from app.services.preprocessing.regex_utils import ScrubResult, normalize_whitespace, scrub_pii
from app.services.preprocessing.validators import (
    content_fingerprint,
    is_empty_review,
    review_identity,
)

log = get_logger("preprocessing")

DedupeMode = Literal["off", "id", "content", "both"]


@dataclass
class PreprocessStats:
    """Per-run counters. Useful for monitoring and tests."""

    seen: int = 0
    kept: int = 0
    dropped_empty: int = 0
    dropped_duplicate_id: int = 0
    dropped_duplicate_content: int = 0
    redactions: dict[str, int] = field(
        default_factory=lambda: {"email": 0, "phone": 0, "username": 0, "id": 0}
    )

    def _add_redactions(self, scrub: ScrubResult) -> None:
        for key, value in scrub.redactions.items():
            self.redactions[key] = self.redactions.get(key, 0) + value


class PreprocessingService:
    """Stateless-by-default review cleaner.

    Deduplication state lives on the instance, so a single service is meant
    to be used for one logical batch / stream (e.g. one app's weekly pull).
    Call ``reset()`` to reuse the instance across batches.
    """

    def __init__(self, *, dedupe: DedupeMode = "both") -> None:
        self._dedupe = dedupe
        self._seen_ids: set[tuple[str, str]] = set()
        self._seen_fingerprints: set[str] = set()
        self.stats = PreprocessStats()

    # ---- public API -------------------------------------------------------

    def reset(self) -> None:
        """Drop deduplication state and stats. Configuration is preserved."""
        self._seen_ids.clear()
        self._seen_fingerprints.clear()
        self.stats = PreprocessStats()

    def clean_text(self, text: str | None) -> tuple[str, ScrubResult] | None:
        """Normalise + scrub a single string. Returns ``None`` if empty."""
        if text is None:
            return None
        scrub = scrub_pii(text)
        cleaned = normalize_whitespace(scrub.text)
        if not cleaned:
            return None
        return cleaned, scrub

    def preprocess_review(self, review: Review) -> Review | None:
        """Clean and validate one review. Returns ``None`` if it is dropped.

        Drops apply in this order: duplicate-by-id, empty, duplicate-by-content.
        Empty checks run *after* PII scrubbing because a body that is only
        an email address must be dropped, not redacted to "[EMAIL]" alone.
        """
        self.stats.seen += 1

        if self._dedupe in ("id", "both"):
            identity = review_identity(review)
            if identity in self._seen_ids:
                self.stats.dropped_duplicate_id += 1
                return None
            self._seen_ids.add(identity)

        body_result = self.clean_text(review.body)
        title_result = self.clean_text(review.title)

        if body_result is None:
            self.stats.dropped_empty += 1
            return None

        body_clean, body_scrub = body_result
        self.stats._add_redactions(body_scrub)

        title_clean: str | None = None
        if title_result is not None:
            title_clean, title_scrub = title_result
            self.stats._add_redactions(title_scrub)

        cleaned = review.model_copy(update={"body": body_clean, "title": title_clean})

        if is_empty_review(cleaned):
            self.stats.dropped_empty += 1
            return None

        if self._dedupe in ("content", "both"):
            fp = content_fingerprint(cleaned)
            if fp in self._seen_fingerprints:
                self.stats.dropped_duplicate_content += 1
                return None
            self._seen_fingerprints.add(fp)

        self.stats.kept += 1
        return cleaned

    def preprocess(self, reviews: Iterable[Review]) -> Iterator[Review]:
        """Sync generator over an in-memory iterable of reviews."""
        for review in reviews:
            cleaned = self.preprocess_review(review)
            if cleaned is not None:
                yield cleaned

    async def preprocess_stream(
        self, reviews: AsyncIterator[Review]
    ) -> AsyncIterator[Review]:
        """Async generator wrapping an importer stream.

        Designed to be dropped between an importer and any downstream sink::

            async for review in service.preprocess_stream(source.stream(...)):
                ...
        """
        async for review in reviews:
            cleaned = self.preprocess_review(review)
            if cleaned is not None:
                yield cleaned

        log.info(
            "preprocessing.stream.done",
            seen=self.stats.seen,
            kept=self.stats.kept,
            dropped_empty=self.stats.dropped_empty,
            dropped_duplicate_id=self.stats.dropped_duplicate_id,
            dropped_duplicate_content=self.stats.dropped_duplicate_content,
            redactions=self.stats.redactions,
        )

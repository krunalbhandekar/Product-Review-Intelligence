"""Example: show what the preprocessing service does to a handful of reviews.

Run with::

    python -m examples.preprocessing_demo

Prints a side-by-side of raw -> cleaned for a small in-memory sample plus
the final ``PreprocessStats`` so the redaction + drop behaviour is visible
without needing network access.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.enums import Platform
from app.domain.models import Review
from app.services.preprocessing import PreprocessingService


def _sample() -> list[Review]:
    base = {
        "source": Platform.IOS,
        "app_id": "310633997",
        "rating": 4,
        "posted_at": datetime(2026, 5, 10, tzinfo=UTC),
        "lang": "en",
        "country": "us",
    }
    return [
        Review(
            review_id="r1",
            title="Great   app!!",
            body=(
                "Loved it.   Contact me at jane.doe+demo@example.com or "
                "+1 (415) 555-1234.\n\n\nFollow @jane_doe — order #1234567890 "
                "ref 550e8400-e29b-41d4-a716-446655440000."
            ),
            author="jane",
            **base,
        ),
        Review(  # duplicate of r1 by upstream id — dropped
            review_id="r1",
            title="dup",
            body="dup",
            author="jane",
            **base,
        ),
        Review(  # body collapses to empty after scrub — dropped
            review_id="r2",
            title=None,
            body="ok",
            author=None,
            **base,
        ),
        Review(
            review_id="r3",
            title=None,
            body="Battery life is amazing. 10/10.",
            author=None,
            **{**base, "rating": 5},
        ),
        Review(  # same content as r3 from a different country — dropped
            review_id="r3-uk",
            title=None,
            body="Battery life is amazing. 10/10.",
            author=None,
            **{**base, "country": "gb", "rating": 5},
        ),
    ]


def main() -> None:
    svc = PreprocessingService(dedupe="both")
    raw = _sample()
    kept = list(svc.preprocess(raw))

    print(f"Input:  {len(raw)} reviews")
    print(f"Kept:   {len(kept)} reviews\n")

    for review in raw:
        print(f"--- raw {review.review_id} ---")
        print(f"  title: {review.title!r}")
        print(f"  body : {review.body!r}")
    print()
    for review in kept:
        print(f"--- clean {review.review_id} ---")
        print(f"  title: {review.title!r}")
        print(f"  body : {review.body!r}")

    print("\nStats:")
    print(f"  seen                  : {svc.stats.seen}")
    print(f"  kept                  : {svc.stats.kept}")
    print(f"  dropped (empty)       : {svc.stats.dropped_empty}")
    print(f"  dropped (dup id)      : {svc.stats.dropped_duplicate_id}")
    print(f"  dropped (dup content) : {svc.stats.dropped_duplicate_content}")
    print(f"  redactions            : {svc.stats.redactions}")


if __name__ == "__main__":
    main()

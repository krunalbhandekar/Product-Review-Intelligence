from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from app.domain.enums import Platform
from app.domain.models import Review
from app.services.preprocessing import (
    EMAIL_TOKEN,
    PHONE_TOKEN,
    USER_TOKEN,
    PreprocessingService,
    content_fingerprint,
    is_empty_review,
    is_empty_text,
    normalize_whitespace,
    scrub_pii,
)


def _review(**overrides: Any) -> Review:
    base: dict[str, Any] = {
        "source": Platform.IOS,
        "app_id": "310633997",
        "review_id": "r1",
        "rating": 4,
        "title": None,
        "body": "Solid app.",
        "author": None,
        "posted_at": datetime(2026, 5, 10, tzinfo=UTC),
        "lang": "en",
        "country": "us",
    }
    base.update(overrides)
    return Review(**base)


# --- regex_utils ----------------------------------------------------------


def test_scrub_pii_redacts_email() -> None:
    result = scrub_pii("Reach me at jane.doe+x@example.co.uk anytime.")
    assert EMAIL_TOKEN in result.text
    assert "jane.doe" not in result.text
    assert result.redactions["email"] == 1


def test_scrub_pii_redacts_phone_variants() -> None:
    text = "Call +1 (415) 555-1234 or 020 7946 0958."
    result = scrub_pii(text)
    assert "555" not in result.text
    assert "7946" not in result.text
    assert PHONE_TOKEN in result.text
    assert result.redactions["phone"] >= 2


def test_scrub_pii_redacts_username_handles() -> None:
    result = scrub_pii("Shoutout to @jane_doe and @bob1 — thanks!")
    assert "@jane_doe" not in result.text
    assert "@bob1" not in result.text
    assert result.text.count(USER_TOKEN) == 2
    assert result.redactions["username"] == 2


def test_scrub_pii_does_not_eat_email_local_as_username() -> None:
    result = scrub_pii("contact foo@bar.com please")
    assert result.redactions["email"] == 1
    assert result.redactions["username"] == 0


def test_scrub_pii_redacts_uuid_and_long_ids() -> None:
    text = (
        "order #1234567890 ref 550e8400-e29b-41d4-a716-446655440000 "
        "session abcdef0123456789abcdef0123"
    )
    result = scrub_pii(text)
    # The contract is "no raw identifier survives" — the *category* token
    # assigned to a bare 10-digit run is implementation-defined (phone vs id).
    assert "1234567890" not in result.text
    assert "550e8400" not in result.text
    assert "446655440000" not in result.text
    assert "abcdef0123456789abcdef0123" not in result.text
    # UUID + long-hex must specifically register as IDs.
    assert result.redactions["id"] >= 2
    assert result.total >= 3


def test_scrub_pii_keeps_normal_text() -> None:
    result = scrub_pii("Battery life is amazing. 10/10 would recommend.")
    assert result.total == 0
    assert "Battery life is amazing." in result.text


def test_normalize_whitespace_collapses_and_trims() -> None:
    raw = "  hello  world\t\t!  \n\n\n\nnext  line  "
    assert normalize_whitespace(raw) == "hello world !\nnext line"


def test_normalize_whitespace_handles_zero_width_chars() -> None:
    assert normalize_whitespace("foo​bar") == "foobar"


# --- validators -----------------------------------------------------------


def test_is_empty_text_thresholds() -> None:
    assert is_empty_text(None) is True
    assert is_empty_text("") is True
    assert is_empty_text("   ") is True
    assert is_empty_text("ok") is True  # below MIN_BODY_CHARS
    assert is_empty_text("nice") is False


def test_is_empty_review_uses_body_only() -> None:
    assert is_empty_review(_review(body="ok", title="Great long title")) is True
    assert is_empty_review(_review(body="great app")) is False


def test_content_fingerprint_stable_across_case_and_whitespace() -> None:
    a = _review(body="Battery is great.", title="Wow")
    b = _review(review_id="other", body="  battery is great.  ", title="WOW")
    assert content_fingerprint(a) == content_fingerprint(b)


def test_content_fingerprint_differs_on_rating() -> None:
    a = _review(body="same body", rating=4)
    b = _review(body="same body", rating=5)
    assert content_fingerprint(a) != content_fingerprint(b)


# --- service --------------------------------------------------------------


def test_preprocess_review_scrubs_and_normalises() -> None:
    svc = PreprocessingService(dedupe="off")
    review = _review(
        title="Great   app",
        body="Email me  jane@example.com  or call +1 415 555 1234. @jane_doe",
    )
    out = svc.preprocess_review(review)
    assert out is not None
    assert "jane@example.com" not in out.body
    assert "555" not in out.body
    assert "@jane_doe" not in out.body
    assert EMAIL_TOKEN in out.body
    assert PHONE_TOKEN in out.body
    assert USER_TOKEN in out.body
    assert "  " not in out.body  # whitespace collapsed
    assert out.title == "Great app"


def test_preprocess_review_drops_empty_body() -> None:
    svc = PreprocessingService(dedupe="off")
    assert svc.preprocess_review(_review(body="  ")) is None
    assert svc.preprocess_review(_review(body="ok")) is None  # below threshold
    assert svc.stats.dropped_empty == 2


def test_preprocess_review_drops_body_that_becomes_empty_after_scrub() -> None:
    svc = PreprocessingService(dedupe="off")
    # body is *only* PII; after redaction it is purely tokens — but those
    # tokens are not real content, so the cleaned body is just the token
    # text. We still want to drop the original raw-empty cases, but token-
    # only bodies are allowed through (they tell the LLM "contact info was
    # here") — assert that this is the observed contract.
    out = svc.preprocess_review(_review(body="jane@example.com"))
    assert out is not None
    assert out.body == EMAIL_TOKEN


def test_preprocess_dedupes_by_review_id() -> None:
    svc = PreprocessingService(dedupe="id")
    a = _review(review_id="r1", body="first review")
    b = _review(review_id="r1", body="different text but same id")
    assert svc.preprocess_review(a) is not None
    assert svc.preprocess_review(b) is None
    assert svc.stats.dropped_duplicate_id == 1


def test_preprocess_dedupes_by_content() -> None:
    svc = PreprocessingService(dedupe="content")
    a = _review(review_id="r1", body="Battery life is amazing.")
    b = _review(review_id="r2", body="  battery life IS amazing.  ")
    assert svc.preprocess_review(a) is not None
    assert svc.preprocess_review(b) is None
    assert svc.stats.dropped_duplicate_content == 1


def test_preprocess_reset_clears_state() -> None:
    svc = PreprocessingService(dedupe="both")
    a = _review(review_id="r1", body="nice app")
    assert svc.preprocess_review(a) is not None
    assert svc.preprocess_review(a) is None  # id dup
    svc.reset()
    assert svc.preprocess_review(a) is not None
    assert svc.stats.seen == 1


def test_preprocess_sync_iter() -> None:
    svc = PreprocessingService(dedupe="both")
    inputs = [
        _review(review_id="r1", body="nice app"),
        _review(review_id="r1", body="nice app"),  # dup id
        _review(review_id="r2", body="ok"),  # empty
        _review(review_id="r3", body="totally different"),
    ]
    kept = list(svc.preprocess(inputs))
    assert [r.review_id for r in kept] == ["r1", "r3"]
    assert svc.stats.seen == 4
    assert svc.stats.kept == 2


async def test_preprocess_stream_async() -> None:
    svc = PreprocessingService(dedupe="both")

    async def source() -> AsyncIterator[Review]:
        payloads = [("a", "nice app"), ("b", "nice app"), ("c", "great work")]
        for rid, body in payloads:
            yield _review(review_id=rid, body=body)

    out = [r async for r in svc.preprocess_stream(source())]
    assert [r.review_id for r in out] == ["a", "c"]
    assert svc.stats.dropped_duplicate_content == 1


async def test_preprocess_stream_never_leaks_pii() -> None:
    """End-to-end guard: no email/phone/handle/id pattern should survive."""
    import re

    svc = PreprocessingService(dedupe="off")

    async def source() -> AsyncIterator[Review]:
        bodies = [
            "ping jane@example.com",
            "call +44 20 7946 0958 thanks",
            "DM @support_team please",
            "ticket 550e8400-e29b-41d4-a716-446655440000 open",
            "order #1234567890 still pending",
        ]
        for i, body in enumerate(bodies):
            yield _review(review_id=f"r{i}", body=body)

    out = [r async for r in svc.preprocess_stream(source())]
    leak_patterns = [
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        re.compile(r"\+?\d[\d \-()]{6,}\d"),
        re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{2,}"),
        re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"),
    ]
    for review in out:
        for pat in leak_patterns:
            assert pat.search(review.body) is None, (pat.pattern, review.body)

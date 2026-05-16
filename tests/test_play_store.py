from datetime import UTC, datetime

from app.domain.enums import Platform
from app.services.ingest.play_store import _to_review


def test_to_review_maps_required_fields() -> None:
    raw = {
        "reviewId": "abc-123",
        "userName": "alice",
        "content": "Great app",
        "score": 5,
        "at": datetime(2026, 5, 1, tzinfo=UTC),
    }
    r = _to_review(raw, app_id="com.example", country="us", lang="en")
    assert r.source == Platform.ANDROID
    assert r.review_id == "abc-123"
    assert r.rating == 5
    assert r.body == "Great app"
    assert r.author == "alice"
    assert r.country == "us"
    assert r.lang == "en"


def test_to_review_assumes_utc_when_naive() -> None:
    raw = {
        "reviewId": "xyz",
        "userName": None,
        "content": "",
        "score": 3,
        "at": datetime(2026, 4, 1),
    }
    r = _to_review(raw, app_id="com.example", country="us", lang="en")
    assert r.posted_at.tzinfo is not None
    assert r.body == ""
    assert r.author is None

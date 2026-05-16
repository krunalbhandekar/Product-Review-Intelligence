from datetime import UTC, datetime, timedelta

import pytest

from app.services.ingest.play_store import PlayStoreSource


@pytest.mark.live
async def test_play_store_streams_real_reviews() -> None:
    src = PlayStoreSource(page_size=20, max_reviews=5)
    until = datetime.now(UTC)
    since = until - timedelta(days=60)

    collected = []
    async for review in src.stream(app_id="com.whatsapp", since=since, until=until):
        collected.append(review)

    assert collected, "expected at least one review from public Play Store"
    for r in collected:
        assert 1 <= r.rating <= 5
        assert since <= r.posted_at <= until
        assert r.app_id == "com.whatsapp"

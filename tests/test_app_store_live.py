from __future__ import annotations

import pytest

from app.services.ingest.app_store import AppStoreSource
from app.services.ingest.time_window import lookback_window


@pytest.mark.live
async def test_app_store_streams_real_reviews() -> None:
    # WhatsApp Messenger on the US App Store.
    app_id = "310633997"
    since, until = lookback_window(8)

    collected = []
    async with AppStoreSource(country="us", max_reviews=5) as src:
        async for review in src.stream(app_id=app_id, since=since, until=until):
            collected.append(review)

    assert collected, "expected at least one review from public App Store"
    for r in collected:
        assert 1 <= r.rating <= 5
        assert since <= r.posted_at <= until
        assert r.app_id == app_id
        assert r.country == "us"

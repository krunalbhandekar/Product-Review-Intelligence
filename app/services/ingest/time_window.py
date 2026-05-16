from datetime import UTC, datetime, timedelta


def lookback_window(
    weeks: int,
    *,
    until: datetime | None = None,
    min_weeks: int = 8,
    max_weeks: int = 12,
) -> tuple[datetime, datetime]:
    """Return a ``(since, until)`` tuple covering the most recent ``weeks``.

    Clamped to ``[min_weeks, max_weeks]`` to keep ingestion windows within the
    8-12 week band the pipeline operates on. ``until`` defaults to now (UTC).
    """
    if min_weeks > max_weeks:
        raise ValueError("min_weeks must be <= max_weeks")
    clamped = max(min_weeks, min(max_weeks, weeks))
    end = until or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return end - timedelta(weeks=clamped), end

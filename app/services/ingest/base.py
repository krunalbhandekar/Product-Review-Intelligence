from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.domain.models import Review


@runtime_checkable
class ReviewSource(Protocol):
    def stream(
        self,
        *,
        app_id: str,
        since: datetime,
        until: datetime,
    ) -> AsyncIterator[Review]: ...

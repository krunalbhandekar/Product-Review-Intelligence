from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import Platform


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Platform
    app_id: str
    review_id: str
    rating: int = Field(ge=1, le=5)
    title: str | None = None
    body: str
    author: str | None = None
    posted_at: datetime
    lang: str
    country: str
    app_version: str | None = None

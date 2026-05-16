"""Pydantic schemas for the weekly pulse output.

The :class:`WeeklyPulse` is the structured-JSON contract emitted by the
pulse generator. The companion markdown rendering is produced by
:class:`PulseFormatter` and is derivable from these fields — keep the JSON
authoritative and the markdown a view onto it, never the other way around.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.summarization.schemas import Prevalence, Sentiment

PulseTone = Literal["positive", "negative", "neutral", "mixed"]


class PulseTheme(BaseModel):
    """A single ranked theme on the weekly pulse."""

    model_config = ConfigDict(extra="ignore")

    rank: int = Field(ge=1, le=3)
    label: str = Field(min_length=1, max_length=80)
    headline: str = Field(min_length=1, max_length=240)
    sentiment: Sentiment = "neutral"
    prevalence: Prevalence = "medium"
    evidence_count: int = Field(default=0, ge=0)


class PulseQuote(BaseModel):
    """A representative user quote tied to one of the top themes."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=300)
    theme_label: str | None = Field(default=None, max_length=80)


class PulseAction(BaseModel):
    """A short, leadership-readable action idea."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=240)
    theme_label: str | None = Field(default=None, max_length=80)


class WeeklyPulseMeta(BaseModel):
    """Run metadata surfaced alongside the pulse body."""

    model_config = ConfigDict(extra="ignore")

    generated_at: datetime
    window_start: datetime | None = None
    window_end: datetime | None = None
    reports_count: int = Field(default=0, ge=0)
    total_reviews: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)


class WeeklyPulse(BaseModel):
    """Structured weekly pulse. ``WeeklyPulseGenerator.render_markdown``
    converts this into the leadership-facing markdown view.

    The ``headline`` + ``executive_summary`` together MUST fit within the
    250-word target. The generator clips ``executive_summary`` to enforce
    this; callers should treat that field as already shaped for delivery.
    """

    model_config = ConfigDict(extra="ignore")

    headline: str = Field(min_length=1, max_length=160)
    executive_summary: str = Field(default="", max_length=2000)
    overall_tone: PulseTone = "neutral"
    themes: list[PulseTheme] = Field(default_factory=list, max_length=3)
    quotes: list[PulseQuote] = Field(default_factory=list, max_length=3)
    actions: list[PulseAction] = Field(default_factory=list, max_length=3)
    meta: WeeklyPulseMeta

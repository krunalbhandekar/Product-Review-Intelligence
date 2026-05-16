"""Pydantic schemas for the chunk-based summarisation pipeline.

These models are the structured-output contract with Groq. Every LLM call in
this pipeline asks for ``response_format={"type": "json_object"}`` and the raw
response is validated against one of these models — any drift (missing fields,
wrong types) is caught at the boundary, not deep inside the aggregator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["positive", "negative", "neutral", "mixed"]
Prevalence = Literal["low", "medium", "high"]


class ChunkTheme(BaseModel):
    """A single theme extracted from one chunk of reviews."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)
    sentiment: Sentiment = "neutral"
    is_pain_point: bool = False
    evidence_count: int = Field(default=1, ge=1)
    sample_quotes: list[str] = Field(default_factory=list, max_length=3)


class ChunkSummary(BaseModel):
    """Map-step output: one chunk -> one summary + themes."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: int = Field(ge=0)
    review_count: int = Field(ge=0)
    summary: str = Field(default="", max_length=1200)
    themes: list[ChunkTheme] = Field(default_factory=list)


class FinalTheme(BaseModel):
    """Reduce-step output: one consolidated, deduplicated theme."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    sentiment: Sentiment = "neutral"
    prevalence: Prevalence = "medium"
    supporting_quotes: list[str] = Field(default_factory=list, max_length=3)
    action_hint: str | None = Field(default=None, max_length=300)


class SummaryStats(BaseModel):
    """Pipeline counters surfaced alongside the report."""

    model_config = ConfigDict(extra="ignore")

    total_reviews: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    themes_premerged: int = 0
    themes_final: int = 0


class SummaryReport(BaseModel):
    """Final pipeline output."""

    model_config = ConfigDict(extra="ignore")

    executive_summary: str = Field(default="", max_length=2000)
    themes: list[FinalTheme] = Field(default_factory=list)
    action_ideas: list[str] = Field(default_factory=list, max_length=3)
    stats: SummaryStats = Field(default_factory=SummaryStats)

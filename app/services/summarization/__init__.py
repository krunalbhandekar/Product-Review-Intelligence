"""Chunk-based summarisation pipeline for product reviews.

Public surface kept intentionally small: callers should generally only need
:class:`SummarizationService` and the response schemas. Lower-level pieces
(chunker, aggregator, Groq client) are exported for testing and advanced
composition.
"""

from app.services.summarization.aggregator import ThemeAggregator, premerge_themes
from app.services.summarization.chunker import ReviewChunk, ReviewChunker, estimate_tokens
from app.services.summarization.groq_client import GroqClient, GroqConfig
from app.services.summarization.schemas import (
    ChunkSummary,
    ChunkTheme,
    FinalTheme,
    Prevalence,
    Sentiment,
    SummaryReport,
    SummaryStats,
)
from app.services.summarization.service import SummarizationService

__all__ = [
    "ChunkSummary",
    "ChunkTheme",
    "FinalTheme",
    "GroqClient",
    "GroqConfig",
    "Prevalence",
    "ReviewChunk",
    "ReviewChunker",
    "Sentiment",
    "SummarizationService",
    "SummaryReport",
    "SummaryStats",
    "ThemeAggregator",
    "estimate_tokens",
    "premerge_themes",
]

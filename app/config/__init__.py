"""Configuration package.

Re-exports the public surface from :mod:`app.config.settings` so the
established ``from app.config import get_settings`` import path keeps
working alongside the new ``from app.config.settings import settings``.
"""

from app.config.settings import (
    AppConfig,
    GroqConfig,
    PipelineConfig,
    Settings,
    SummaryConfig,
    get_settings,
    settings,
)

__all__ = [
    "AppConfig",
    "GroqConfig",
    "PipelineConfig",
    "Settings",
    "SummaryConfig",
    "get_settings",
    "settings",
]

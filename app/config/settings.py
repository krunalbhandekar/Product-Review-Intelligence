"""Application settings: static configs in code, secrets in env.

Layout
------
Static operational knobs (model names, chunk sizes, retry counts, etc.)
live as frozen dataclasses on this module. They are version-controlled,
diffable, and code-reviewed like any other behaviour.

Secrets and deployment-specific values (API keys, recipient email,
target Google Doc id) load from ``.env`` via a lightweight
``pydantic-settings`` shim. There are only a handful of these — enough
to fit in one screen of ``.env.example`` so onboarding stays fast.

The merged :class:`Settings` instance is exposed two ways:

* ``from app.config.settings import settings`` — module-level singleton
  for normal application code,
* ``from app.config.settings import get_settings`` — cached factory for
  injection sites that want to call it explicitly.

Tests should construct ``Settings(...)`` directly with overrides rather
than mutate the singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Static configs (no env binding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Core app identity and runtime knobs."""

    APP_NAME: str = "Product-Review-Intelligence"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


@dataclass(frozen=True)
class GroqConfig:
    """Groq client tuning. Used to build the runtime GroqConfig in
    ``app.services.summarization.groq_client`` — kept distinct from
    that dataclass so this layer stays declarative and stable."""

    MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.2
    TIMEOUT: float = 30.0
    MAX_RETRIES: int = 3
    # Tried in order when the primary model returns 429. Tuple, not
    # list, so the static config stays immutable.
    FALLBACK_MODELS: tuple[str, ...] = ("llama3-8b-8192", "gemma2-9b-it")
    TOKEN_THRESHOLD: int = 3500
    MAX_429_STREAK: int = 3
    REQUEST_CONCURRENCY: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Ingest + summarisation pipeline shape + MCP HTTP-layer knobs."""

    MAX_REVIEWS_PER_RUN: int = 50
    # Window size in whole weeks. Was previously a request-body field
    # and an env var; now a code constant — operators change it via PR.
    LOOKBACK_WEEKS: int = 12
    CHUNK_TARGET_TOKENS: int = 3500
    CHUNK_SIZE: int = 50
    SUMMARIZATION_CONCURRENCY: int = 1
    MAX_REVIEW_BODY_CHARS: int = 1500
    # MCP client tuning — lives here rather than its own block because
    # MCP is internal infrastructure plumbing, not user-facing surface.
    MCP_TIMEOUT: float = 30.0
    MCP_MAX_RETRIES: int = 3
    MCP_RETRY_BACKOFF_SECONDS: float = 0.5
    EMAIL_SUBJECT_PREFIX: str = "[Weekly Pulse]"


@dataclass(frozen=True)
class SummaryConfig:
    """Output-shaping caps for the leadership-facing pulse."""

    MAX_THEMES: int = 5
    MAX_QUOTES: int = 3
    MAX_ACTION_ITEMS: int = 3
    SUMMARY_MAX_WORDS: int = 250


# ---------------------------------------------------------------------------
# Secrets + deployment-specific values (env-loaded)
# ---------------------------------------------------------------------------


class _EnvLoaded(BaseSettings):
    """Internal loader for ``.env``. Kept private — callers see the
    fields surfaced as plain attributes on :class:`Settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets
    api_key: str = "dev-key"
    groq_api_key: str = ""
    # Deployment-specific identifiers
    google_doc_id: str = ""
    email_to: str = ""
    mcp_base_url: str = ""
    playstore_app_id: str = ""
    appstore_app_id: str = ""


# ---------------------------------------------------------------------------
# Merged settings object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Single object passed through the app. Composes the four static
    config groups with the env-loaded deployment values.

    Access pattern::

        settings.pipeline.LOOKBACK_WEEKS
        settings.groq.MODEL
        settings.summary.MAX_QUOTES
        settings.api_key          # secret
        settings.google_doc_id    # deployment-specific
    """

    app: AppConfig = field(default_factory=AppConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    # Secrets / deployment-specific
    api_key: str = "dev-key"
    groq_api_key: str = ""
    google_doc_id: str = ""
    email_to: str = ""
    mcp_base_url: str = ""
    playstore_app_id: str = ""
    appstore_app_id: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        """Build a Settings instance with env-loaded secrets and the
        in-code static configs. The entry point used by
        :func:`get_settings`; tests should prefer constructing
        ``Settings(...)`` directly with the overrides they need."""
        env = _EnvLoaded()
        return cls(
            api_key=env.api_key,
            groq_api_key=env.groq_api_key,
            google_doc_id=env.google_doc_id,
            email_to=env.email_to,
            mcp_base_url=env.mcp_base_url,
            playstore_app_id=env.playstore_app_id,
            appstore_app_id=env.appstore_app_id,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings.from_env()


# Module-level singleton. Importing this is the normal path —
# ``get_settings()`` exists for callers that want lazy evaluation.
settings: Settings = get_settings()


__all__ = [
    "AppConfig",
    "GroqConfig",
    "PipelineConfig",
    "Settings",
    "SummaryConfig",
    "get_settings",
    "settings",
]

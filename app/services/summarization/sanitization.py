"""LLM-output sanitisation: pre-Pydantic defensive shaping.

Both summarisation stages (chunk map and reduce) take a raw JSON object
from the LLM and validate it against a Pydantic schema. A single
malformed entry — wrong type, oversized list, null where a string is
required, six quotes when the cap is three — used to blow up the whole
stage. That made the pipeline brittle in exactly the way LLM output
demands you not be.

Everything in this module is *pre*-validation: we clip, coerce, and
drop entries to the schema's known constraints so ``model_validate``
sees a clean dict. Pydantic remains the safety net — if we miss a
case, the per-entry try/except at the call site logs and skips
without dragging down the rest.

Public API:
    * ``sanitize_llm_output(payload, kind)`` — top-level entrypoint.
      ``kind="chunk"`` shapes for :class:`ChunkTheme`; ``kind="final"``
      shapes for :class:`FinalTheme`.
    * ``sanitize_action_ideas`` / ``sanitize_executive_summary`` —
      list / string helpers used by the reduce step alongside themes.
"""

from __future__ import annotations

from typing import Any, Literal

from app.core.logging import get_logger

log = get_logger("summarization.sanitization")

# Schema caps (mirrors ``schemas.py``). Duplicated here so the
# sanitisation layer can act before Pydantic does. If a cap moves in
# the schema, update both — the schema remains authoritative.
_CHUNK_LABEL_MAX = 80
_CHUNK_DESC_MAX = 400
_CHUNK_QUOTE_MAX_ITEMS = 3
_FINAL_LABEL_MAX = 80
_FINAL_DESC_MAX = 500
_FINAL_QUOTE_MAX_ITEMS = 3
_FINAL_ACTION_HINT_MAX = 300
_QUOTE_TEXT_MAX = 500  # not a schema cap; keeps oversized prose bounded
_ACTION_IDEA_MAX = 240
_ACTION_IDEAS_MAX_ITEMS = 3
_EXEC_SUMMARY_MAX = 2000

_VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
_VALID_PREVALENCES = {"low", "medium", "high"}

Kind = Literal["chunk", "final"]


def _clip(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len]


def _coerce_str(value: Any) -> str:
    """``None``/``int``/``bool`` etc. → empty string or string repr."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return str(value).strip()
    except Exception:
        return ""


def _coerce_quotes(raw: Any, max_items: int) -> list[str]:
    """Coerce a quotes field into a list of clean, capped strings.

    Tolerates: non-list inputs (treat as empty), nested arrays, null
    items, non-string items, and oversized arrays.
    """
    if raw is None:
        return []
    # Some LLMs return a single string instead of a list.
    if isinstance(raw, str):
        s = raw.strip()
        return [_clip(s, _QUOTE_TEXT_MAX)] if s else []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if len(out) >= max_items:
            break
        s = _coerce_str(item)
        if s:
            out.append(_clip(s, _QUOTE_TEXT_MAX))
    return out


def _coerce_sentiment(raw: Any) -> str | None:
    s = _coerce_str(raw).lower()
    return s if s in _VALID_SENTIMENTS else None


def _coerce_prevalence(raw: Any) -> str | None:
    s = _coerce_str(raw).lower()
    return s if s in _VALID_PREVALENCES else None


def _coerce_int(raw: Any, *, default: int = 1, min_value: int = 1) -> int:
    if isinstance(raw, bool):  # bool is an int subclass — exclude first
        return default
    if isinstance(raw, int):
        return raw if raw >= min_value else default
    if isinstance(raw, float):
        n = int(raw)
        return n if n >= min_value else default
    if isinstance(raw, str):
        try:
            n = int(raw.strip())
            return n if n >= min_value else default
        except ValueError:
            return default
    return default


def _sanitize_chunk_theme(entry: Any) -> dict[str, Any] | None:
    """Shape one entry into a ChunkTheme-ready dict, or ``None`` to drop."""
    if not isinstance(entry, dict):
        return None
    label = _coerce_str(entry.get("label"))
    if not label:
        return None
    description = _coerce_str(entry.get("description")) or label
    out: dict[str, Any] = {
        "label": _clip(label, _CHUNK_LABEL_MAX),
        "description": _clip(description, _CHUNK_DESC_MAX),
        "sample_quotes": _coerce_quotes(
            entry.get("sample_quotes"), _CHUNK_QUOTE_MAX_ITEMS
        ),
    }
    sentiment = _coerce_sentiment(entry.get("sentiment"))
    if sentiment is not None:
        out["sentiment"] = sentiment
    if "is_pain_point" in entry:
        out["is_pain_point"] = bool(entry.get("is_pain_point"))
    if "evidence_count" in entry:
        out["evidence_count"] = _coerce_int(
            entry.get("evidence_count"), default=1, min_value=1
        )
    return out


def _sanitize_final_theme(entry: Any) -> dict[str, Any] | None:
    """Shape one entry into a FinalTheme-ready dict, or ``None`` to drop."""
    if not isinstance(entry, dict):
        return None
    label = _coerce_str(entry.get("label"))
    if not label:
        return None
    description = _coerce_str(entry.get("description")) or label
    # Accept either ``supporting_quotes`` (final-stage canonical) or
    # ``sample_quotes`` (chunk-stage carry-over) so a slightly drifted
    # LLM response doesn't lose its quotes entirely.
    quotes_raw = entry.get("supporting_quotes")
    if quotes_raw is None:
        quotes_raw = entry.get("sample_quotes")
    out: dict[str, Any] = {
        "label": _clip(label, _FINAL_LABEL_MAX),
        "description": _clip(description, _FINAL_DESC_MAX),
        "supporting_quotes": _coerce_quotes(quotes_raw, _FINAL_QUOTE_MAX_ITEMS),
    }
    sentiment = _coerce_sentiment(entry.get("sentiment"))
    if sentiment is not None:
        out["sentiment"] = sentiment
    prevalence = _coerce_prevalence(entry.get("prevalence"))
    if prevalence is not None:
        out["prevalence"] = prevalence
    action_hint = _coerce_str(entry.get("action_hint"))
    if action_hint:
        out["action_hint"] = _clip(action_hint, _FINAL_ACTION_HINT_MAX)
    return out


def sanitize_llm_output(payload: Any, *, kind: Kind) -> list[dict[str, Any]]:
    """Top-level entry: return a list of sanitised theme dicts.

    ``payload`` may be:
      * a list of theme dicts (the common case),
      * a dict with a ``themes`` key (some models wrap once more),
      * ``None`` or any other shape (returns an empty list).
    """
    raw_list: Any = payload
    if isinstance(payload, dict):
        raw_list = payload.get("themes")
    if not isinstance(raw_list, list):
        return []

    shaper = _sanitize_chunk_theme if kind == "chunk" else _sanitize_final_theme
    out: list[dict[str, Any]] = []
    for entry in raw_list:
        shaped = shaper(entry)
        if shaped is not None:
            out.append(shaped)
    return out


def sanitize_action_ideas(raw: Any) -> list[str]:
    """List-of-strings normaliser used by the reduce step."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [_clip(s, _ACTION_IDEA_MAX)] if s else []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if len(out) >= _ACTION_IDEAS_MAX_ITEMS:
            break
        s = _coerce_str(item)
        if s:
            out.append(_clip(s, _ACTION_IDEA_MAX))
    return out


def sanitize_executive_summary(raw: Any) -> str:
    s = _coerce_str(raw)
    return _clip(s, _EXEC_SUMMARY_MAX)


__all__ = [
    "Kind",
    "sanitize_action_ideas",
    "sanitize_executive_summary",
    "sanitize_llm_output",
]
